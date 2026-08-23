#!/usr/bin/env python3
"""Benchmark encoded point spans against the repeated scratch-tree loop.

The rank workload has the shape used by ``f5_components.cpp::rankbasis`` and
related component scouts: many small encoded point sets, each decoded and row
reduced separately. The query workload covers the companion membership and
coordinate extraction loops used by the GL matchers.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter, process_time

import numpy as np

from fast_math import fp_point_span, fp_span_ranks


def measure(call, repeats: int):
    wall_seconds = []
    cpu_seconds = []
    result = None
    for _ in range(repeats):
        wall_started = perf_counter()
        cpu_started = process_time()
        result = call()
        cpu_seconds.append(process_time() - cpu_started)
        wall_seconds.append(perf_counter() - wall_started)
    return result, wall_seconds, cpu_seconds


def compare_spans(reference, native) -> None:
    assert reference.rank == native.rank
    for name in (
        "pivot_indices",
        "pivot_columns",
        "reduced_basis_codes",
        "independent_points",
        "query_members",
        "query_coordinates",
        "query_quotient_codes",
    ):
        np.testing.assert_array_equal(
            getattr(reference, name), getattr(native, name)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rank-spans", type=int, default=20_000)
    parser.add_argument("--points-per-span", type=int, default=8)
    parser.add_argument("--points", type=int, default=5_000)
    parser.add_argument("--queries", type=int, default=50_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(20260823)
    rank_prime, rank_width = 5, 6
    rank_points = rng.integers(
        0,
        rank_prime**rank_width,
        size=args.rank_spans * args.points_per_span,
        dtype=np.uint64,
    )
    rank_offsets = np.arange(
        0,
        len(rank_points) + 1,
        args.points_per_span,
        dtype=np.uint64,
    )

    query_prime, query_width = 251, 6
    query_points = rng.integers(
        0,
        query_prime**query_width,
        size=args.points,
        dtype=np.uint64,
    )
    queries = rng.integers(
        0,
        query_prime**query_width,
        size=args.queries,
        dtype=np.uint64,
    )

    report = {
        "kernel": "spans of encoded F_p points",
        "machine": {
            "node": platform.node(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "repeats": args.repeats,
        "workloads": {},
    }

    rank_results = {}
    for backend in ("reference", "native"):
        result, wall, cpu = measure(
            lambda backend=backend: fp_span_ranks(
                rank_points,
                rank_offsets,
                rank_prime,
                rank_width,
                backend=backend,
            ),
            args.repeats,
        )
        rank_results[backend] = result
        report["workloads"].setdefault(
            "ragged_rank_batches",
            {
                "source_pattern": "f5_components.cpp::rankbasis",
                "shape": {
                    "prime": rank_prime,
                    "width": rank_width,
                    "span_count": args.rank_spans,
                    "points_per_span": args.points_per_span,
                },
                "backends": {},
            },
        )["backends"][backend] = {
            "wall_seconds": wall,
            "cpu_seconds": cpu,
            "rank_sum": int(result.sum()),
        }
    np.testing.assert_array_equal(rank_results["reference"], rank_results["native"])

    span_results = {}
    for backend in ("reference", "native"):
        result, wall, cpu = measure(
            lambda backend=backend: fp_point_span(
                query_points,
                queries,
                query_prime,
                query_width,
                backend=backend,
            ),
            args.repeats,
        )
        span_results[backend] = result
        report["workloads"].setdefault(
            "span_membership_and_coordinates",
            {
                "source_pattern": "ci2/verify8.py::classes_from_topparts",
                "shape": {
                    "prime": query_prime,
                    "width": query_width,
                    "point_count": args.points,
                    "query_count": args.queries,
                },
                "backends": {},
            },
        )["backends"][backend] = {
            "wall_seconds": wall,
            "cpu_seconds": cpu,
            "rank": int(result.rank),
        }
    compare_spans(span_results["reference"], span_results["native"])

    for workload in report["workloads"].values():
        before = median(workload["backends"]["reference"]["wall_seconds"])
        after = median(workload["backends"]["native"]["wall_seconds"])
        workload["median_reference_wall_seconds"] = before
        workload["median_native_wall_seconds"] = after
        workload["wall_speedup"] = before / after

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
