#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = PROJECT_ROOT / "benchmarks" / "benchmark_accumulate.py"


def run(arguments: list[str]) -> list[dict[str, object]]:
    completed = subprocess.run(
        [sys.executable, str(BENCHMARK), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    records.extend(
        run(
            [
                "--case",
                "smoke",
                "--backend",
                "both",
                "--threads",
                "1",
                "--tile-size",
                str(1 << 16),
                "--repeats",
                "2",
            ]
        )
    )
    records.extend(
        run(
            [
                "--case",
                "medium",
                "--backend",
                "reference",
                "--threads",
                "1",
                "--tile-size",
                str(1 << 16),
                "--repeats",
                "2",
            ]
        )
    )
    for threads in (1, 2, 4, 8, 10):
        records.extend(
            run(
                [
                    "--case",
                    "medium",
                    "--backend",
                    "native",
                    "--threads",
                    str(threads),
                    "--tile-size",
                    str(1 << 16),
                    "--repeats",
                    "2",
                ]
            )
        )
    for tile_size in (1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18):
        records.extend(
            run(
                [
                    "--case",
                    "medium",
                    "--backend",
                    "native",
                    "--threads",
                    "10",
                    "--tile-size",
                    str(tile_size),
                    "--repeats",
                    "2",
                ]
            )
        )

    document = {
        "suite": "lambda-fast-initial",
        "records": records,
    }
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="ascii")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
