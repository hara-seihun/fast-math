#!/usr/bin/env python3
"""Benchmark colex rank/visit backends against the hand-written loop.

The replaced pattern, seen in the fleet's orbit-enumeration scouts, is a
per-subset Python loop: walk the mask bits, sum math.comb lookups, then
test-and-set one bit in a visited bitmap by hand. That is exactly the
reference backend here; the benchmark measures it against the native
backend on one representative marking workload.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from statistics import median
from time import perf_counter, process_time

import numpy as np

from fast_math import colex_rank, colex_visit


def measure(
    subset_masks: np.ndarray,
    element_count: int,
    weight: int,
    backend: str,
    repeats: int,
) -> tuple[list[float], list[float], np.ndarray]:
    visited = np.zeros(
        (comb(element_count, weight) + 63) // 64, dtype=np.uint64
    )
    expected = colex_visit(
        subset_masks, element_count, weight, visited.copy(), backend=backend
    )
    wall_seconds = []
    cpu_seconds = []
    for _ in range(repeats):
        wall_started = perf_counter()
        cpu_started = process_time()
        actual = colex_visit(
            subset_masks,
            element_count,
            weight,
            visited.copy(),
            backend=backend,
        )
        cpu_seconds.append(process_time() - cpu_started)
        wall_seconds.append(perf_counter() - wall_started)
        np.testing.assert_array_equal(actual, expected)
    return wall_seconds, cpu_seconds, expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch", type=int, default=131_072)
    parser.add_argument("--element-count", type=int, default=48)
    parser.add_argument("--weight", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    n, k = args.element_count, args.weight
    rng = np.random.default_rng(20260823)
    subset_masks = np.zeros(args.batch, dtype=np.uint64)
    for index in range(args.batch):
        elements = rng.choice(n, size=k, replace=False)
        mask = np.uint64(0)
        for element in elements:
            mask |= np.uint64(1) << np.uint64(element)
        subset_masks[index] = mask

    report = {
        "kernel": "colex subset ranking with orbit marking",
        "shape": {
            "element_count": n,
            "weight": k,
            "subsets_per_batch": int(args.batch),
            "visited_bits": comb(n, k),
        },
        "repeats": args.repeats,
        "backends": {},
    }

    rank_wall = {}
    for backend in ("reference", "native"):
        ranks = colex_rank(subset_masks, n, backend=backend)
        times = []
        for _ in range(args.repeats):
            started = perf_counter()
            actual = colex_rank(subset_masks, n, backend=backend)
            times.append(perf_counter() - started)
            np.testing.assert_array_equal(actual, ranks)
        rank_wall[backend] = times

    for backend in ("reference", "native"):
        wall, cpu, expected = measure(
            subset_masks, n, k, backend, args.repeats
        )
        report["backends"][backend] = {
            "visit_wall_seconds": wall,
            "visit_cpu_seconds": cpu,
            "rank_wall_seconds": rank_wall[backend],
            "newly_visited_total": int(expected.sum()),
        }

    reference_wall = median(report["backends"]["reference"]["visit_wall_seconds"])
    native_wall = median(report["backends"]["native"]["visit_wall_seconds"])
    report["visit_wall_speedup"] = reference_wall / native_wall
    report["rank_wall_speedup"] = (
        median(rank_wall["reference"]) / median(rank_wall["native"])
    )

    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
