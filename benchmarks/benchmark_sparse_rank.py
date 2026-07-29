#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_once(
    executable: Path,
    matrix: Path,
    prime: int,
    target_rank: int,
) -> dict[str, object]:
    completed = subprocess.run(
        [
            str(executable),
            str(matrix),
            str(prime),
            str(target_rank),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--prime", type=int, default=1_000_000_007)
    parser.add_argument("--target-rank", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline-seconds", type=float)
    parser.add_argument(
        "--executable",
        type=Path,
        default=PROJECT_ROOT / "build" / "fast_math_sparse_rank_benchmark",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    samples = [
        run_once(
            args.executable,
            args.matrix,
            args.prime,
            args.target_rank,
        )
        for _ in range(args.repeats)
    ]
    wall_samples = [float(sample["wall_seconds"]) for sample in samples]
    parse_samples = [float(sample["parse_seconds"]) for sample in samples]
    record = dict(samples[0])
    record.update(
        {
            "matrix": str(args.matrix.resolve()),
            "repeats": args.repeats,
            "wall_seconds_best": min(wall_samples),
            "wall_seconds_median": statistics.median(wall_samples),
            "wall_seconds_all": wall_samples,
            "parse_seconds_all": parse_samples,
        }
    )
    if args.baseline_seconds is not None:
        record["baseline_seconds"] = args.baseline_seconds
        record["speedup_best"] = (
            args.baseline_seconds / min(wall_samples)
        )
        record["speedup_median"] = (
            args.baseline_seconds / statistics.median(wall_samples)
        )

    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
