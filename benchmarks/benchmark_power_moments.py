#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from lambda_fast import power_moments  # noqa: E402


CASES = {
    "smoke": 100_000,
    "medium": 2_000_000,
    "large": 10_000_000,
}


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, default="medium")
    parser.add_argument(
        "--backend",
        choices=("reference", "native", "both"),
        default="both",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=1 << 16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    count = CASES[args.case]
    x = np.arange(count, dtype=np.float64)
    values = (
        0.35
        + 0.2 * np.cos(0.0001 * x)
        + 0.15j * np.sin(0.00013 * x)
    )
    derivatives = (
        -0.00002 * np.sin(0.0001 * x)
        + 0.0000195j * np.cos(0.00013 * x)
    )
    backends = (
        ("reference", "native")
        if args.backend == "both"
        else (args.backend,)
    )
    outputs = {}
    records = []
    for backend in backends:
        timings = []
        result = None
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = power_moments(
                values,
                derivatives,
                mesh_step=4.0,
                minimum_power=3,
                maximum_power=12,
                chunk_size=args.chunk_size,
                threads=args.threads,
                backend=backend,
            )
            timings.append(time.perf_counter() - started)
        assert result is not None
        outputs[backend] = result
        best = min(timings)
        record = {
            "benchmark": "power_moments",
            "case": args.case,
            "backend": backend,
            "sample_count": count,
            "minimum_power": 3,
            "maximum_power": 12,
            "threads": args.threads,
            "chunk_size": args.chunk_size,
            "wall_seconds_best": best,
            "wall_seconds_all": timings,
            "kernel_seconds_last": result.elapsed_seconds,
            "samples_per_second": count / best,
            "peak_rss_bytes": peak_rss_bytes(),
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    if len(outputs) == 2:
        for actual, expected in zip(
            outputs["native"].moments,
            outputs["reference"].moments,
            strict=True,
        ):
            np.testing.assert_allclose(
                [
                    actual.value,
                    actual.ordinary,
                    actual.phase_current,
                    actual.radial,
                ],
                [
                    expected.value,
                    expected.ordinary,
                    expected.phase_current,
                    expected.radial,
                ],
                rtol=5e-12,
                atol=1e-12,
            )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
