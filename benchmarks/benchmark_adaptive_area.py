#!/usr/bin/env python3
"""Benchmark the complete adaptive-oracle area route on the restriction lattice.

The measured route is the one research searches actually run: expand a batch of
degree-bounded targets, solve every optimal adaptive policy, and read back the
root area and first query. The reference backend is the executable model that
the native kernel must reproduce.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter, process_time

import numpy as np

from fast_math import adaptive_areas, exact_adaptive_areas


def build_targets(
    coordinate_count: int,
    target_count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(
        (-1.0, 1.0), size=(target_count, 1 << coordinate_count)
    )


def measure(
    tables: np.ndarray,
    backend: str,
    threads: int,
    repeats: int,
) -> tuple[list[float], list[float], np.ndarray]:
    expected = adaptive_areas(tables, threads=threads, backend=backend).areas
    wall_seconds: list[float] = []
    cpu_seconds: list[float] = []
    for _ in range(repeats):
        wall_started = perf_counter()
        cpu_started = process_time()
        batch = adaptive_areas(tables, threads=threads, backend=backend)
        cpu_seconds.append(process_time() - cpu_started)
        wall_seconds.append(perf_counter() - wall_started)
        np.testing.assert_allclose(batch.areas, expected, rtol=0, atol=1e-12)
    return wall_seconds, cpu_seconds, expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates", type=int, nargs="+", default=[6, 8, 10])
    parser.add_argument("--targets", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    records = []
    for coordinate_count in arguments.coordinates:
        budget = arguments.targets * 3**6
        target_count = max(8, budget // 3**coordinate_count)
        tables = build_targets(coordinate_count, target_count, arguments.seed)
        record: dict[str, object] = {
            "coordinate_count": coordinate_count,
            "restriction_count": 3**coordinate_count,
            "target_count": target_count,
        }
        for backend, threads in (
            ("reference", 1),
            ("native", 1),
            ("native", arguments.threads),
        ):
            reference_budget = coordinate_count <= 8 or backend == "native"
            if not reference_budget:
                continue
            wall, cpu, areas = measure(tables, backend, threads, arguments.repeats)
            key = f"{backend}_{threads}"
            record[key] = {
                "wall_seconds": median(wall),
                "cpu_seconds": median(cpu),
                "targets_per_second": target_count / median(wall),
                "area_checksum": float(np.sum(areas)),
            }
        native = record.get(f"native_{arguments.threads}")
        reference = record.get("reference_1")
        if native and reference:
            record["speedup"] = (
                reference["wall_seconds"] / native["wall_seconds"]
            )
        records.append(record)
        print(json.dumps(record), flush=True)

    exact_tables = np.random.default_rng(arguments.seed).integers(
        -1, 2, size=(256, 1 << 8)
    )
    started = perf_counter()
    exact = exact_adaptive_areas(exact_tables, threads=arguments.threads)
    exact_record = {
        "exact_coordinate_count": 8,
        "exact_target_count": 256,
        "wall_seconds": perf_counter() - started,
        "area_numerator_checksum": int(exact.area_numerators.sum()),
    }
    records.append(exact_record)
    print(json.dumps(exact_record), flush=True)

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
