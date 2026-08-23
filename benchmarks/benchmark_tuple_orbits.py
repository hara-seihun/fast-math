#!/usr/bin/env python3
"""Benchmark tuple-orbit canonicalization against the hand-written loop.

The replaced fleet pattern is a per-code Python orbit walk: BFS over
generator images of the digit tuple, deduplicating through a Python set.
The reference backend here is that loop; the benchmark measures it against
the native backend on one representative batch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter, process_time

import numpy as np

from fast_math import tuple_orbit_canonicalize, tuple_orbit_space


def dihedral(width: int) -> np.ndarray:
    return np.array(
        [np.roll(np.arange(width), -1), np.arange(width)[::-1]],
        dtype=np.uint32,
    )


def measure_canonicalize(
    generators, base, width, codes, backend, repeats
):
    expected, _ = tuple_orbit_canonicalize(
        generators, base, width, codes, backend=backend
    )
    wall, cpu = [], []
    for _ in range(repeats):
        wall_started = perf_counter()
        cpu_started = process_time()
        actual, flags = tuple_orbit_canonicalize(
            generators, base, width, codes, backend=backend
        )
        cpu.append(process_time() - cpu_started)
        wall.append(perf_counter() - wall_started)
        np.testing.assert_array_equal(actual, expected)
        assert flags.dtype == np.bool_
    return wall, cpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=200_000)
    parser.add_argument("--base", type=int, default=3)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    base, width = args.base, args.width
    generators = dihedral(width)
    rng = np.random.default_rng(20260823)
    codes = rng.integers(0, base**width, size=args.batch, dtype=np.uint64)

    report = {
        "kernel": "digit-tuple orbits under a permutation group",
        "shape": {
            "base": base,
            "width": width,
            "group": f"dihedral D_{width} (2*{width} elements)",
            "batch_codes": int(args.batch),
        },
        "repeats": args.repeats,
        "backends": {},
    }
    for backend in ("reference", "native"):
        wall, cpu = measure_canonicalize(
            generators, base, width, codes, backend, args.repeats
        )
        report["backends"][backend] = {
            "canonicalize_wall_seconds": wall,
            "canonicalize_cpu_seconds": cpu,
        }

    # Full-space pass with Burnside validation, both backends.
    space_reports = {}
    for backend in ("reference", "native"):
        started = perf_counter()
        space = tuple_orbit_space(dihedral(10), 2, 10, backend=backend)
        space_reports[backend] = {
            "space_wall_seconds": perf_counter() - started,
            "orbit_count": space.orbit_count,
            "burnside_orbit_count": space.burnside_orbit_count,
            "burnside_valid": space.burnside_valid,
        }
    assert (
        space_reports["reference"]["orbit_count"]
        == space_reports["native"]["orbit_count"]
    )
    report["space_shape"] = {"base": 2, "width": 10}
    report["space"] = space_reports

    reference_wall = median(
        report["backends"]["reference"]["canonicalize_wall_seconds"]
    )
    native_wall = median(
        report["backends"]["native"]["canonicalize_wall_seconds"]
    )
    report["canonicalize_wall_speedup"] = reference_wall / native_wall

    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
