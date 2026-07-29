#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import resource
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from lambda_fast import (  # noqa: E402
    accumulate_coefficients,
    available_backends,
    native_version,
)


CASES = {
    "tiny": {
        "inverse_limit": 64,
        "primary_size": 4_000,
        "low_size": 2_000,
        "transformed_first": 71,
        "transformed_size": 32,
        "output_limit": 40_000,
    },
    "smoke": {
        "inverse_limit": 256,
        "primary_size": 25_000,
        "low_size": 12_500,
        "transformed_first": 317,
        "transformed_size": 64,
        "output_limit": 250_000,
    },
    "medium": {
        "inverse_limit": 1_000,
        "primary_size": 100_000,
        "low_size": 50_000,
        "transformed_first": 1_201,
        "transformed_size": 128,
        "output_limit": 2_000_000,
    },
    "checkpoint-shape": {
        "inverse_limit": 1_000,
        "primary_size": 690_988,
        "low_size": 400_000,
        "transformed_first": 2_189,
        "transformed_size": 2,
        "output_limit": 100_000_000,
    },
}


def build_inputs(case: dict[str, int]):
    divisor = np.arange(case["inverse_limit"] + 1, dtype=np.float64)
    inverse = np.zeros_like(divisor)
    inverse[1:] = np.where(
        (divisor[1:].astype(np.int64) & 1) == 0,
        -1.0,
        1.0,
    ) / np.sqrt(divisor[1:])

    primary_index = np.arange(1, case["primary_size"] + 1, dtype=np.float64)
    primary = np.exp(-primary_index / max(1, case["primary_size"]))
    low = primary[: case["low_size"]] * 0.73

    transformed_index = np.arange(
        case["transformed_first"],
        case["transformed_first"] + case["transformed_size"],
        dtype=np.float64,
    )
    transformed = np.exp(0.001j * transformed_index) / np.sqrt(
        transformed_index
    )
    return inverse, primary, transformed, low


def sampled_digest(common: np.ndarray, low: np.ndarray) -> str:
    sample_count = min(4096, len(common))
    indices = np.linspace(
        0, len(common) - 1, num=sample_count, dtype=np.int64
    )
    digest = hashlib.sha256()
    digest.update(common[indices].tobytes())
    digest.update(low[indices].tobytes())
    return digest.hexdigest()


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value)
    return int(value) * 1024


def run_backend(
    backend: str,
    case_name: str,
    case: dict[str, int],
    inputs,
    *,
    tile_size: int,
    threads: int,
    repeats: int,
):
    timings: list[float] = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = accumulate_coefficients(
            *inputs,
            transformed_first=case["transformed_first"],
            output_limit=case["output_limit"],
            tile_size=tile_size,
            threads=threads,
            backend=backend,
        )
        timings.append(time.perf_counter() - started)
    assert result is not None
    best = min(timings)
    return result, {
        "benchmark": "accumulate_coefficients",
        "case": case_name,
        "backend": backend,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "lambda_fast_native": native_version(),
        **case,
        "tile_size": tile_size,
        "threads": threads,
        "repeats": repeats,
        "wall_seconds_best": best,
        "wall_seconds_all": timings,
        "kernel_seconds_last": result.elapsed_seconds,
        "primary_pairs": result.pairs.primary,
        "transformed_pairs": result.pairs.transformed,
        "low_pairs": result.pairs.low,
        "total_pairs": result.pairs.total,
        "pair_updates_per_second": result.pairs.total / best,
        "peak_rss_bytes": peak_rss_bytes(),
        "sampled_output_sha256": sampled_digest(result.common, result.low),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, default="smoke")
    parser.add_argument(
        "--backend",
        choices=("reference", "native", "both"),
        default="both",
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=1 << 14)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive")
    case = CASES[args.case]
    if case["output_limit"] > 20_000_000 and not args.allow_large:
        parser.error("large cases require --allow-large")

    requested = (
        ("reference", "native")
        if args.backend == "both"
        else (args.backend,)
    )
    installed = available_backends()
    missing = [backend for backend in requested if backend not in installed]
    if missing:
        parser.error(f"unavailable backends: {', '.join(missing)}")

    inputs = build_inputs(case)
    results = {}
    records = []
    for backend in requested:
        result, record = run_backend(
            backend,
            args.case,
            case,
            inputs,
            tile_size=args.tile_size,
            threads=args.threads,
            repeats=args.repeats,
        )
        results[backend] = result
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    if len(results) == 2:
        np.testing.assert_allclose(
            results["native"].common,
            results["reference"].common,
            rtol=0,
            atol=2e-15,
        )
        np.testing.assert_allclose(
            results["native"].low,
            results["reference"].low,
            rtol=0,
            atol=2e-15,
        )
        if results["native"].pairs != results["reference"].pairs:
            raise AssertionError("native/reference pair counts differ")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "benchmark": "lambda-fast-accumulate",
            "records": records,
        }
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
