#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from fast_math import (  # noqa: E402
    evaluate_taylor_basis,
    segmented_complex_stats,
    taylor_coefficients,
)


def best_call(function, repeats: int = 3):
    timings = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - started)
    assert result is not None
    return result, min(timings), timings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = []

    sample_count = 2_000_000
    x = np.arange(sample_count, dtype=np.float64)
    values = np.exp(0.0001j * x) * (0.5 + 0.1 * np.cos(0.00007 * x))
    offsets = np.arange(0, sample_count + 1, 100, dtype=np.uint64)
    segment_results = {}
    for backend in ("reference", "native"):
        result, best, timings = best_call(
            lambda backend=backend: segmented_complex_stats(
                values,
                offsets,
                threads=args.threads,
                backend=backend,
            ),
            repeats=args.repeats,
        )
        segment_results[backend] = result
        records.append(
            {
                "benchmark": "segmented_complex_stats",
                "backend": backend,
                "sample_count": sample_count,
                "segment_count": len(offsets) - 1,
                "threads": args.threads,
                "wall_seconds_best": best,
                "wall_seconds_all": timings,
                "kernel_seconds_last": result.elapsed_seconds,
            }
        )
    np.testing.assert_allclose(
        segment_results["native"].sums,
        segment_results["reference"].sums,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        segment_results["native"].variation,
        segment_results["reference"].variation,
        rtol=2e-14,
    )

    coefficient_count = 690_988
    logarithms = np.log(np.arange(1, coefficient_count + 1, dtype=np.float64))
    base = np.exp(-0.03 * logarithms**2 + 0.2j * logarithms)
    coefficient_results = {}
    for backend in ("reference", "native"):
        result, best, timings = best_call(
            lambda backend=backend: taylor_coefficients(
                base,
                logarithms,
                maximum_order=4,
                threads=args.threads,
                backend=backend,
            ),
            repeats=args.repeats,
        )
        coefficient_results[backend] = result
        records.append(
            {
                "benchmark": "taylor_coefficients",
                "backend": backend,
                "sample_count": coefficient_count,
                "order_count": 5,
                "threads": args.threads,
                "wall_seconds_best": best,
                "wall_seconds_all": timings,
                "kernel_seconds_last": result.elapsed_seconds,
            }
        )
    np.testing.assert_allclose(
        coefficient_results["native"].coefficients,
        coefficient_results["reference"].coefficients,
        rtol=3e-15,
        atol=3e-15,
    )

    evaluation_count = 500_000
    delta = 0.01 * np.sin(np.arange(evaluation_count) * 0.0003)
    basis = coefficient_results["native"].coefficients[
        :, :evaluation_count
    ]
    evaluation_results = {}
    for backend in ("reference", "native"):
        result, best, timings = best_call(
            lambda backend=backend: evaluate_taylor_basis(
                basis,
                delta,
                threads=args.threads,
                backend=backend,
            ),
            repeats=args.repeats,
        )
        evaluation_results[backend] = result
        records.append(
            {
                "benchmark": "taylor_evaluate",
                "backend": backend,
                "sample_count": evaluation_count,
                "order_count": 5,
                "threads": args.threads,
                "wall_seconds_best": best,
                "wall_seconds_all": timings,
                "kernel_seconds_last": result.elapsed_seconds,
            }
        )
    np.testing.assert_allclose(
        evaluation_results["native"].values,
        evaluation_results["reference"].values,
        rtol=3e-15,
        atol=3e-15,
    )
    np.testing.assert_allclose(
        evaluation_results["native"].log_moments,
        evaluation_results["reference"].log_moments,
        rtol=3e-15,
        atol=3e-15,
    )

    for record in records:
        print(json.dumps(record, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
