#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from lambda_fast import truncated_inverse  # noqa: E402


CASES = {
    "small": 32_768,
    "medium": 131_072,
    "adaptive": 690_988,
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
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    limit = CASES[args.case]
    backends = (
        ("reference", "native")
        if args.backend == "both"
        else (args.backend,)
    )
    results = {}
    records = []
    for backend in backends:
        timings = []
        result = None
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = truncated_inverse(
                limit, 0.142_908_675_2, backend=backend
            )
            timings.append(time.perf_counter() - started)
        assert result is not None
        results[backend] = result
        best = min(timings)
        record = {
            "benchmark": "truncated_inverse",
            "case": args.case,
            "backend": backend,
            "limit": limit,
            "update_count": result.update_count,
            "wall_seconds_best": best,
            "wall_seconds_all": timings,
            "kernel_seconds_last": result.elapsed_seconds,
            "updates_per_second": result.update_count / best,
            "peak_rss_bytes": peak_rss_bytes(),
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    if len(results) == 2:
        import numpy as np

        np.testing.assert_allclose(
            results["native"].coefficients,
            results["reference"].coefficients,
            rtol=2e-13,
            atol=2e-13,
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
