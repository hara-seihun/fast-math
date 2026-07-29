#!/usr/bin/env python3
"""Benchmark a real affine contour-ranking workload on NumPy and Metal."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from fast_math import AffineMetalPlan  # noqa: E402


DEFAULT_PACKET = (
    Path(__file__).resolve().parents[2]
    / "problems"
    / "riemann-hypothesis"
    / "scratch"
    / "proof-system--adaptive-multiprecision-gpu-contour-optimizer"
    / "L40-degree40-q21-full-strip-boundary.npz"
)


def parse_populations(text: str) -> tuple[int, ...]:
    populations = tuple(int(item) for item in text.split(","))
    if not populations or any(population <= 0 for population in populations):
        raise argparse.ArgumentTypeError("populations must be positive")
    return populations


def timed(repeats: int, function):
    durations = []
    output = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = function()
        durations.append(time.perf_counter() - started)
    return output, durations


def numpy_metrics(
    base: np.ndarray,
    basis: np.ndarray,
    steps: np.ndarray,
    edge_slice: slice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = base[None, :] + steps @ basis
    products = values[:, 1:] * np.conj(values[:, :-1])
    phases = np.arctan2(products.imag, products.real)
    windings = np.rint(np.sum(phases, axis=1) / (2 * math.pi)).astype(
        np.int64
    )
    maximum_phases = np.max(np.abs(phases), axis=1)
    edge_floors = np.min(np.abs(values[:, edge_slice]), axis=1)
    return windings, maximum_phases, edge_floors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument(
        "--populations",
        type=parse_populations,
        default=(85, 256, 1024, 4096),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    with np.load(args.packet) as packet:
        base = np.ascontiguousarray(packet["base"], dtype=np.complex64)
        basis = np.ascontiguousarray(packet["basis"], dtype=np.complex64)
        horizontal_count = int(packet["horizontal_count"])

    point_count = base.size
    direction_count = basis.shape[0]
    edge_slice = slice(horizontal_count + 1, 2 * horizontal_count + 1)
    rng = np.random.default_rng(args.seed)
    plan = AffineMetalPlan(base, basis)

    records = []
    for population in args.populations:
        steps = np.ascontiguousarray(
            rng.normal(size=(population, direction_count)),
            dtype=np.float32,
        )

        numpy_output = numpy_metrics(base, basis, steps, edge_slice)
        metal_result = plan.contour_metrics(
            steps, edge_slice=edge_slice
        )
        metal_output = (
            metal_result.windings,
            metal_result.maximum_phases,
            metal_result.edge_floors,
        )

        numpy_output, numpy_durations = timed(
            args.repeats,
            lambda: numpy_metrics(base, basis, steps, edge_slice),
        )
        metal_result, metal_durations = timed(
            args.repeats,
            lambda: plan.contour_metrics(
                steps, edge_slice=edge_slice
            ),
        )
        metal_output = (
            metal_result.windings,
            metal_result.maximum_phases,
            metal_result.edge_floors,
        )
        numpy_best = min(numpy_durations)
        metal_best = min(metal_durations)
        records.append(
            {
                "population": population,
                "numpy_seconds": {
                    "best": numpy_best,
                    "median": statistics.median(numpy_durations),
                },
                "metal_seconds": {
                    "best": metal_best,
                    "median": statistics.median(metal_durations),
                },
                "metal_speedup_best": numpy_best / metal_best,
                "winding_disagreements": int(
                    np.count_nonzero(numpy_output[0] != metal_output[0])
                ),
                "maximum_phase_max_abs_error": float(
                    np.max(np.abs(numpy_output[1] - metal_output[1]))
                ),
                "edge_floor_max_abs_error": float(
                    np.max(np.abs(numpy_output[2] - metal_output[2]))
                ),
            }
        )
        plan.clear_cache()

    result = {
        "benchmark": "affine_contour_metrics",
        "packet": str(args.packet),
        "point_count": point_count,
        "direction_count": direction_count,
        "dtype": "complex64",
        "plan_setup_seconds": plan.setup_seconds,
        "repeats": args.repeats,
        "records": records,
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
