#!/usr/bin/env python3
"""Benchmark stable 3-WL and 4-WL against the executable reference backend."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter, process_time

import numpy as np

from fast_math import graph_wlk_refinement


def random_graph(vertex_count: int, seed: int) -> np.ndarray:
    random = np.random.default_rng(seed)
    adjacency = np.zeros((vertex_count, vertex_count), dtype=np.uint8)
    for left in range(vertex_count):
        for right in range(left + 1, vertex_count):
            present = random.random() < 0.25
            adjacency[left, right] = present
            adjacency[right, left] = present
    return adjacency


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    workloads = ((3, 20), (4, 10))
    report = {
        "kernel": "stable exact higher-order WL",
        "machine": {
            "node": platform.node(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "repeats": args.repeats,
        "workloads": {},
    }
    for dimension, vertex_count in workloads:
        adjacency = random_graph(vertex_count, 20260823 + dimension)
        results = {}
        timings = {}
        for backend in ("reference", "native"):
            result, wall, cpu = measure(
                lambda backend=backend: graph_wlk_refinement(
                    adjacency,
                    dimension,
                    backend=backend,
                ),
                args.repeats,
            )
            results[backend] = result
            timings[backend] = {
                "wall_seconds": wall,
                "cpu_seconds": cpu,
                "color_count": result.color_count,
                "iterations": result.iterations,
            }
        np.testing.assert_array_equal(
            results["native"].colors,
            results["reference"].colors,
        )
        np.testing.assert_array_equal(
            results["native"].color_sizes,
            results["reference"].color_sizes,
        )
        reference_median = median(timings["reference"]["wall_seconds"])
        native_median = median(timings["native"]["wall_seconds"])
        report["workloads"][f"{dimension}-WL"] = {
            "shape": {
                "vertex_count": vertex_count,
                "tuple_count": vertex_count**dimension,
            },
            "backends": timings,
            "median_reference_wall_seconds": reference_median,
            "median_native_wall_seconds": native_median,
            "wall_speedup": reference_median / native_median,
        }

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
