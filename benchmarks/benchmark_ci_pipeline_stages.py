#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import inspect
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PROJECT_ROOT.parent
ATLAS_ROUTE = (
    RESEARCH_ROOT
    / "problems/cayley-ci/scratch/"
    "atlas--dihedral-generalized-dihedral-first-exact-defect-atlas"
)
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from fast_math.canonical import canonicalize_colored_digraphs  # noqa: E402
from fast_math.ci import (  # noqa: E402
    atom_subsets_to_element_words,
    cayley_graphs,
    enumerate_subset_orbits,
)


def load_atlas_builder():
    name = "_fast_math_ci_stage_atlas_builder"
    spec = importlib.util.spec_from_file_location(
        name,
        ATLAS_ROUTE / "build_atlas.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load retained atlas builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonicalize(
    adjacency: np.ndarray,
    colors: np.ndarray,
    *,
    threads: int,
    collect_generators: bool,
):
    parameters = inspect.signature(
        canonicalize_colored_digraphs
    ).parameters
    kwargs = {"backend": "native"}
    if "threads" in parameters:
        kwargs["threads"] = threads
    if "collect_automorphism_generators" in parameters:
        kwargs["collect_automorphism_generators"] = collect_generators
    return canonicalize_colored_digraphs(adjacency, colors, **kwargs)


def one_run(models, threads: int, collect_generators: bool):
    totals = defaultdict(float)
    group_rows = []
    atlas_fibers = []
    for model in models:
        group_started = perf_counter()

        started = perf_counter()
        partition = enumerate_subset_orbits(
            np.asarray(model.atom_permutations, dtype=np.uint32),
            backend="native",
        )
        totals["subset_wall_seconds"] += perf_counter() - started
        totals["subset_kernel_seconds"] += partition.elapsed_seconds

        started = perf_counter()
        connections = atom_subsets_to_element_words(
            partition.representative_words,
            model.atoms,
            group_order=len(model.elements),
        )
        totals["connection_pack_wall_seconds"] += perf_counter() - started

        started = perf_counter()
        graphs = cayley_graphs(
            np.asarray(model.multiply_table, dtype=np.uint32),
            connections,
            inverse_indices=np.asarray(
                model.inverse_indices,
                dtype=np.uint32,
            ),
            threads=threads,
            backend="native",
        )
        totals["cayley_wall_seconds"] += perf_counter() - started
        totals["cayley_kernel_seconds"] += graphs.elapsed_seconds

        colors = np.zeros(
            (len(connections), len(model.elements)),
            dtype=np.uint32,
        )
        started = perf_counter()
        canonical = canonicalize(
            graphs.adjacency_words,
            colors,
            threads=threads,
            collect_generators=collect_generators,
        )
        totals["canonical_wall_seconds"] += perf_counter() - started
        totals["canonical_kernel_seconds"] += canonical.elapsed_seconds

        fibers: dict[int, list[int]] = defaultdict(list)
        for mask, class_id in zip(
            map(int, partition.representatives),
            map(int, canonical.class_ids),
            strict=True,
        ):
            fibers[class_id].append(mask)
        atlas_fibers.append(
            sorted(tuple(sorted(fiber)) for fiber in fibers.values())
        )
        group_rows.append(
            {
                "group": model.spec.slug,
                "connection_orbits": len(partition.representatives),
                "graph_fibers": len(fibers),
                "wall_seconds": perf_counter() - group_started,
                "canonical_search_nodes": canonical.search_nodes,
                "automorphism_generators": len(
                    canonical.automorphism_generators
                ),
            }
        )
    totals["total_wall_seconds"] = sum(
        row["wall_seconds"] for row in group_rows
    )
    return {
        "totals": dict(totals),
        "groups": group_rows,
        "fiber_partition": atlas_fibers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--no-generators",
        action="store_true",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    builder = load_atlas_builder()
    models = [builder.build_group(spec) for spec in builder.atlas_specs()]
    runs = [
        one_run(
            models,
            args.threads,
            not args.no_generators,
        )
        for _ in range(args.repeats)
    ]
    expected = runs[0]["fiber_partition"]
    if any(run["fiber_partition"] != expected for run in runs[1:]):
        raise AssertionError("atlas fiber partition changed between runs")
    metric_names = sorted(runs[0]["totals"])
    payload = {
        "schema": 1,
        "date": "2026-07-29",
        "threads": args.threads,
        "repeats": args.repeats,
        "collect_automorphism_generators": not args.no_generators,
        "automorphism_orbits": sum(
            row["connection_orbits"] for row in runs[0]["groups"]
        ),
        "graph_fibers": sum(
            row["graph_fibers"] for row in runs[0]["groups"]
        ),
        "metrics": {
            name: {
                "samples": [run["totals"][name] for run in runs],
                "median": median(
                    run["totals"][name] for run in runs
                ),
            }
            for name in metric_names
        },
        "groups": runs[0]["groups"],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="ascii")


if __name__ == "__main__":
    main()
