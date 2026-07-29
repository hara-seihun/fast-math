#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PROJECT_ROOT.parent
ROUTE = (
    RESEARCH_ROOT
    / "problems/cayley-ci/scratch/"
    "structural-mechanisms--q60-centrally-unbalanced-2wl-strata"
)
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from fast_math.ci import (  # noqa: E402
    cayley_graphs,
    graph_coherent_configuration,
    graph_wl2_refinement,
)


def load_route():
    name = "_fast_math_ci_q60_wl2_route"
    spec = importlib.util.spec_from_file_location(
        name,
        ROUTE / "scan_one_unbalanced_2wl.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the Q60 2-WL route")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(
    route,
    rows,
    compute_tensor: bool,
    threads: int,
) -> dict[str, object]:
    connections = np.asarray(
        [[row["connection_mask"]] for row in rows],
        dtype=np.uint64,
    )
    graphs = cayley_graphs(
        route.TABLE,
        connections,
        inverse_indices=route.INVERSES,
        threads=threads,
        backend="native",
    )
    started = perf_counter()
    results = [
        (
            graph_coherent_configuration(
                adjacency,
                max_tensor_entries=300_000,
                backend="native",
            )
            if compute_tensor
            else graph_wl2_refinement(
                adjacency,
                backend="native",
            )
        )
        for adjacency in graphs.adjacency_words
    ]
    wall = perf_counter() - started
    return {
        "wall_seconds": wall,
        "native_seconds": sum(
            result.elapsed_seconds for result in results
        ),
        "signatures": [
            (
                result.relations.tobytes(),
                result.relation_count,
                result.iterations,
            )
            for result in results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    route = load_route()
    rows = [
        json.loads(line)
        for line in (
            ROUTE / "one-unbalanced-collisions.jsonl"
        ).read_text(encoding="utf-8").splitlines()[: args.limit]
    ]
    full_runs = []
    refine_runs = []
    for repeat in range(args.repeats):
        if repeat % 2 == 0:
            full_runs.append(run(route, rows, True, args.threads))
            refine_runs.append(run(route, rows, False, args.threads))
        else:
            refine_runs.append(run(route, rows, False, args.threads))
            full_runs.append(run(route, rows, True, args.threads))
    expected = full_runs[0]["signatures"]
    if any(run["signatures"] != expected for run in full_runs[1:]):
        raise AssertionError("full coherent configurations changed")
    if any(run["signatures"] != expected for run in refine_runs):
        raise AssertionError("refinement-only output changed")
    full_wall = median(run["wall_seconds"] for run in full_runs)
    refine_wall = median(run["wall_seconds"] for run in refine_runs)
    payload = {
        "schema": 1,
        "date": "2026-07-29",
        "graphs": len(rows),
        "repeats": args.repeats,
        "threads": args.threads,
        "full_wall_seconds": [
            run["wall_seconds"] for run in full_runs
        ],
        "refinement_only_wall_seconds": [
            run["wall_seconds"] for run in refine_runs
        ],
        "full_native_seconds": [
            run["native_seconds"] for run in full_runs
        ],
        "refinement_only_native_seconds": [
            run["native_seconds"] for run in refine_runs
        ],
        "speedup": full_wall / refine_wall,
        "exact_stable_relation_parity": True,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="ascii")


if __name__ == "__main__":
    main()
