#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
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
    atom_subsets_to_element_words,
    canonicalize_cayley_graphs,
    cayley_graphs,
    deduplicate_subset_orbits,
)
from fast_math.graph64 import encode_graph6  # noqa: E402


def load_route():
    name = "_fast_math_ci_q60_live_route"
    spec = importlib.util.spec_from_file_location(
        name,
        ROUTE / "search_one_unbalanced_defects.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the Q60 route")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def partition(labels) -> tuple[tuple[int, ...], ...]:
    classes: dict[object, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        classes[label].append(index)
    return tuple(sorted(tuple(indices) for indices in classes.values()))


def one_run(route, rotation_class, threads: int) -> dict[str, object]:
    timings: dict[str, float] = {}
    started = perf_counter()
    rotations = np.asarray(rotation_class, dtype=np.uint64)
    subsets = (
        rotations[:, np.newaxis]
        | route.reflection_masks[np.newaxis, :]
    ).reshape(-1)
    timings["subset_generation_seconds"] = perf_counter() - started

    started = perf_counter()
    quotient = deduplicate_subset_orbits(
        subsets,
        route.ACTION_GENERATORS,
        atom_count=30,
        backend="native",
    )
    timings["subset_orbits_seconds"] = perf_counter() - started

    started = perf_counter()
    connections = atom_subsets_to_element_words(
        quotient.representative_words,
        route.ATOMS,
        group_order=60,
        threads=threads,
        backend="native",
    )
    timings["atom_expansion_seconds"] = perf_counter() - started

    started = perf_counter()
    graphs = cayley_graphs(
        route.TABLE,
        connections,
        inverse_indices=route.INVERSES,
        threads=threads,
        backend="native",
    )
    encoded = encode_graph6(
        graphs.graph64_masks,
        threads=threads,
        backend="native",
    )
    baseline_labels = route.labelg(encoded.records)
    timings["graph6_labelg_seconds"] = perf_counter() - started

    started = perf_counter()
    native = canonicalize_cayley_graphs(
        route.TABLE,
        connections,
        inverse_indices=route.INVERSES,
        threads=threads,
        collect_automorphism_generators=False,
        construction_backend="native",
        canonical_backend="native",
    )
    timings["in_process_nauty_seconds"] = perf_counter() - started

    baseline_partition = partition(baseline_labels)
    native_partition = partition(map(int, native.class_ids))
    if native_partition != baseline_partition:
        raise AssertionError(
            "in-process nauty and labelg disagree on Q60 fibers"
        )
    timings["baseline_total_seconds"] = sum(
        timings[name]
        for name in (
            "subset_generation_seconds",
            "subset_orbits_seconds",
            "atom_expansion_seconds",
            "graph6_labelg_seconds",
        )
    )
    timings["optimized_total_seconds"] = sum(
        timings[name]
        for name in (
            "subset_generation_seconds",
            "subset_orbits_seconds",
            "atom_expansion_seconds",
            "in_process_nauty_seconds",
        )
    )
    return {
        "timings": timings,
        "raw_subsets": len(subsets),
        "automorphism_orbits": len(connections),
        "graph_fibers": len(native_partition),
        "subset_kernel_seconds": quotient.elapsed_seconds,
        "canonical_kernel_seconds": native.canonical.elapsed_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    route = load_route()
    route.reflection_masks = (
        np.arange(1 << 15, dtype=np.uint64) << np.uint64(15)
    )
    rotation_classes = route.rotation_orbits()
    rotation_class = max(rotation_classes, key=len)
    runs = [
        one_run(route, rotation_class, args.threads)
        for _ in range(args.repeats)
    ]
    metric_names = sorted(runs[0]["timings"])
    baseline = median(
        run["timings"]["baseline_total_seconds"] for run in runs
    )
    optimized = median(
        run["timings"]["optimized_total_seconds"] for run in runs
    )
    payload = {
        "schema": 1,
        "date": "2026-07-29",
        "threads": args.threads,
        "repeats": args.repeats,
        "rotation_orbit_size": len(rotation_class),
        "raw_subsets": runs[0]["raw_subsets"],
        "automorphism_orbits": runs[0]["automorphism_orbits"],
        "graph_fibers": runs[0]["graph_fibers"],
        "metrics": {
            name: {
                "samples": [
                    run["timings"][name] for run in runs
                ],
                "median": median(
                    run["timings"][name] for run in runs
                ),
            }
            for name in metric_names
        },
        "speedup": baseline / optimized,
        "subset_kernel_seconds": [
            run["subset_kernel_seconds"] for run in runs
        ],
        "canonical_kernel_seconds": [
            run["canonical_kernel_seconds"] for run in runs
        ],
        "fiber_parity": True,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="ascii")


if __name__ == "__main__":
    main()
