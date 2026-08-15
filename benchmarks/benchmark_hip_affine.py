#!/usr/bin/env python3
"""Benchmark the local ROCm affine contour-ranking path against NumPy."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from fast_math import AffineHipPlan


def timed(repeats: int, function):
    durations = []
    output = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = function()
        durations.append(time.perf_counter() - started)
    return output, durations


def numpy_metrics(base, basis, steps, edge_slice):
    values = base[None, :] + steps @ basis
    products = values[:, 1:] * np.conj(values[:, :-1])
    phases = np.arctan2(products.imag, products.real)
    return (
        np.rint(np.sum(phases, axis=1) / (2 * np.pi)).astype(np.int64),
        np.max(np.abs(phases), axis=1),
        np.min(np.abs(values[:, edge_slice]), axis=1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=13661)
    parser.add_argument("--directions", type=int, default=20)
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    base = np.exp(0.003j * np.arange(args.points)).astype(np.complex64)
    basis = (0.0005 * (
        rng.normal(size=(args.directions, args.points))
        + 1j * rng.normal(size=(args.directions, args.points))
    )).astype(np.complex64)
    steps = rng.normal(size=(args.population, args.directions)).astype(np.float32)
    edge_slice = slice(100, args.points - 100)
    plan = AffineHipPlan(base, basis)
    expected, numpy_times = timed(
        args.repeats,
        lambda: numpy_metrics(base, basis, steps, edge_slice),
    )
    actual, hip_times = timed(
        args.repeats,
        lambda: plan.contour_metrics(
            steps, edge_slice=edge_slice, batch_size=None
        ),
    )
    record = {
        "benchmark": "hip_affine_contour_metrics",
        "device_architecture": "gfx1151",
        "point_count": args.points,
        "direction_count": args.directions,
        "population": args.population,
        "dtype": "complex64",
        "plan_setup_seconds": plan.setup_seconds,
        "numpy_best_seconds": min(numpy_times),
        "hip_best_seconds": min(hip_times),
        "hip_speedup_best": min(numpy_times) / min(hip_times),
        "winding_disagreements": int(
            np.count_nonzero(expected[0] != actual.windings)
        ),
        "maximum_phase_max_abs_error": float(
            np.max(np.abs(expected[1] - actual.maximum_phases))
        ),
        "edge_floor_max_abs_error": float(
            np.max(np.abs(expected[2] - actual.edge_floors))
        ),
        "numpy_median_seconds": statistics.median(numpy_times),
        "hip_median_seconds": statistics.median(hip_times),
    }
    encoded = json.dumps(record, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    plan.close()


if __name__ == "__main__":
    main()
