#!/usr/bin/env python3
"""Benchmark packed union-closure reference and native backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter, process_time

import numpy as np

from fast_math import union_closed_family_masks


def measure(
    family_masks: np.ndarray,
    ground_size: int,
    backend: str,
    repeats: int,
    iterations: int,
) -> tuple[list[float], list[float], np.ndarray]:
    expected = union_closed_family_masks(
        family_masks,
        ground_size,
        backend=backend,
    )
    wall_seconds = []
    cpu_seconds = []
    for _ in range(repeats):
        wall_started = perf_counter()
        cpu_started = process_time()
        for _ in range(iterations):
            actual = union_closed_family_masks(
                family_masks,
                ground_size,
                backend=backend,
            )
        cpu_seconds.append((process_time() - cpu_started) / iterations)
        wall_seconds.append((perf_counter() - wall_started) / iterations)
        np.testing.assert_array_equal(actual, expected)
    return wall_seconds, cpu_seconds, expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(20260728)
    fixtures = {
        "ground4_all_nonempty": (
            np.arange(1, 1 << 16, dtype=np.uint64),
            4,
        ),
        "ground5_random_65536": (
            rng.integers(
                0,
                1 << 32,
                size=65_536,
                dtype=np.uint64,
            ),
            5,
        ),
        "ground6_random_65536": (
            rng.integers(
                0,
                np.iinfo(np.uint64).max,
                size=65_536,
                dtype=np.uint64,
            ),
            6,
        ),
    }

    results = {
        "repeats": args.repeats,
        "iterations_per_repeat": args.iterations,
        "fixtures": {},
    }
    for name, (family_masks, ground_size) in fixtures.items():
        reference_wall, reference_cpu, reference = measure(
            family_masks,
            ground_size,
            "reference",
            args.repeats,
            args.iterations,
        )
        native_wall, native_cpu, native = measure(
            family_masks,
            ground_size,
            "native",
            args.repeats,
            args.iterations,
        )
        np.testing.assert_array_equal(native, reference)
        reference_wall_median = median(reference_wall)
        reference_cpu_median = median(reference_cpu)
        native_wall_median = median(native_wall)
        native_cpu_median = median(native_cpu)
        results["fixtures"][name] = {
            "family_count": len(family_masks),
            "ground_size": ground_size,
            "closed_count": int(reference.sum()),
            "reference_wall_seconds": reference_wall,
            "reference_cpu_seconds": reference_cpu,
            "native_wall_seconds": native_wall,
            "native_cpu_seconds": native_cpu,
            "reference_wall_median": reference_wall_median,
            "reference_cpu_median": reference_cpu_median,
            "native_wall_median": native_wall_median,
            "native_cpu_median": native_cpu_median,
            "wall_speedup": reference_wall_median / native_wall_median,
            "cpu_speedup": reference_cpu_median / native_cpu_median,
            "output_identical": True,
        }

    text = json.dumps(results, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
