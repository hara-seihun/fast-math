#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = PROJECT_ROOT / "benchmarks" / "benchmark_real_checkpoint.py"


def positive_ints(text: str) -> list[int]:
    values = [int(value) for value in text.split(",")]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=positive_ints, default="1,2,4,6,8")
    parser.add_argument(
        "--tile-sizes",
        type=positive_ints,
        default="4096,8192,16384,32768,65536",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    records = []
    for thread_count in args.threads:
        for tile_size in args.tile_sizes:
            runs = []
            for _ in range(args.repeats):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(BENCHMARK),
                        "--threads",
                        str(thread_count),
                        "--tile-size",
                        str(tile_size),
                    ],
                    cwd=PROJECT_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                runs.append(json.loads(completed.stdout.strip().splitlines()[-1]))
            best = min(runs, key=lambda record: record["wall_seconds"])
            record = {
                "threads": thread_count,
                "tile_size": tile_size,
                "wall_seconds_best": best["wall_seconds"],
                "native_seconds_best": best["native_seconds"],
                "pair_updates_per_second": best["pair_updates_per_second"],
                "peak_rss_bytes": best["peak_rss_bytes"],
                "two_level_delta": best["two_level_delta"],
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    best = min(records, key=lambda record: record["wall_seconds_best"])
    result = {
        "benchmark": "real_l01529_fused_two_level_tuning",
        "checkpoint": "N1448739-Q40",
        "repeats": args.repeats,
        "best": best,
        "records": records,
    }
    print(json.dumps({"best": best}, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
