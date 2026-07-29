#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from benchmark_sparse_rank import run_once


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--prime", type=int, default=1_000_000_007)
    parser.add_argument("--target-rank", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    left_samples = []
    right_samples = []
    left_record = None
    right_record = None
    for repeat in range(args.repeats):
        order = (
            (("left", args.left), ("right", args.right))
            if repeat % 2 == 0
            else (("right", args.right), ("left", args.left))
        )
        for name, executable in order:
            result = run_once(
                executable,
                args.matrix,
                args.prime,
                args.target_rank,
            )
            if name == "left":
                left_record = result
                left_samples.append(float(result["wall_seconds"]))
            else:
                right_record = result
                right_samples.append(float(result["wall_seconds"]))
    assert left_record is not None and right_record is not None
    if (
        left_record["rank"] != right_record["rank"]
        or left_record["target_reached"] != right_record["target_reached"]
        or left_record["pivot_digest"] != right_record["pivot_digest"]
    ):
        raise AssertionError("A/B rank or pivot witnesses differ")

    left_median = statistics.median(left_samples)
    right_median = statistics.median(right_samples)
    record = {
        "benchmark": "sparse_modular_rank_ab",
        "matrix": str(args.matrix.resolve()),
        "prime": args.prime,
        "target_rank": args.target_rank,
        "rank": left_record["rank"],
        "pivot_digest": left_record["pivot_digest"],
        "repeats": args.repeats,
        "left_name": args.left_name,
        "left_executable": str(args.left.resolve()),
        "left_seconds_all": left_samples,
        "left_seconds_best": min(left_samples),
        "left_seconds_median": left_median,
        "right_name": args.right_name,
        "right_executable": str(args.right.resolve()),
        "right_seconds_all": right_samples,
        "right_seconds_best": min(right_samples),
        "right_seconds_median": right_median,
        "right_over_left_median": right_median / left_median,
    }
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
