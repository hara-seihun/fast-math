#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_once(
    binary: Path,
    backend: str,
    count: int,
    precision: int,
    threads: int,
    chunk_size: int,
) -> dict:
    completed = subprocess.run(
        [
            str(binary),
            backend,
            str(count),
            str(precision),
            str(threads),
            str(chunk_size),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200_000)
    parser.add_argument("--precision", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = PROJECT_ROOT / "build" / "fast_math_arb_cache_benchmark"
    if not binary.exists():
        raise SystemExit("Arb benchmark was not built; FLINT may be unavailable")

    records = []
    by_backend: dict[str, list[dict]] = {
        "baseline": [],
        "optimized": [],
        "ordered": [],
    }
    for _ in range(args.repeats):
        for backend in ("baseline", "optimized", "ordered"):
            record = run_once(
                binary,
                backend,
                args.count,
                args.precision,
                args.threads,
                args.chunk_size,
            )
            records.append(record)
            by_backend[backend].append(record)

    hashes = {
        record["center_hash"]
        for record in records
    }
    if len(hashes) != 1:
        raise AssertionError("optimized Arb cache centers are not byte-identical")
    baseline = min(
        record["wall_seconds"] for record in by_backend["baseline"]
    )
    optimized = min(
        record["wall_seconds"] for record in by_backend["optimized"]
    )
    ordered = min(
        record["wall_seconds"] for record in by_backend["ordered"]
    )
    baseline_error = max(
        record["weighted_error_upper"]
        for record in by_backend["baseline"]
    )
    optimized_error = max(
        record["weighted_error_upper"]
        for record in by_backend["optimized"]
    )
    ordered_error = max(
        record["weighted_error_upper"]
        for record in by_backend["ordered"]
    )
    if optimized_error > baseline_error:
        raise AssertionError(
            "optimized Arb weighted error bound is weaker than baseline"
        )
    serial_certificates = {
        record["weighted_error"] for record in by_backend["optimized"]
    }
    ordered_certificates = {
        record["weighted_error"] for record in by_backend["ordered"]
    }
    if len(serial_certificates | ordered_certificates) != 1:
        raise AssertionError(
            "ordered Arb reduction changed the serial certificate"
        )
    summary = {
        "benchmark": "arb_source_cache_hot_loop_summary",
        "count": args.count,
        "cache_values": 2 * args.count,
        "precision_bits": args.precision,
        "repeats": args.repeats,
        "center_hash": hashes.pop(),
        "baseline_seconds_best": baseline,
        "optimized_seconds_best": optimized,
        "ordered_seconds_best": ordered,
        "baseline_weighted_error_upper": baseline_error,
        "optimized_weighted_error_upper": optimized_error,
        "ordered_weighted_error_upper": ordered_error,
        "threads": args.threads,
        "chunk_size": args.chunk_size,
        "baseline_to_serial_speedup": baseline / optimized,
        "serial_to_ordered_speedup": optimized / ordered,
        "baseline_to_ordered_speedup": baseline / ordered,
    }
    records.append(summary)
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
