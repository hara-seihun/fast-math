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
RESEARCH_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from lambda_fast import fused_two_level  # noqa: E402
from lambda_fast.two_level import WEIGHT_DTYPE  # noqa: E402


ROUTE = (
    RESEARCH_ROOT
    / "problems"
    / "riemann-hypothesis"
    / "scratch"
    / "adjacent-unconditional--debruijn-newman-l01529-phase-faithful-finite-bridge"
)


def load_events(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROUTE / "build" / "checkpoints" / "N1448739-Q40",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=1 << 13)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    source_certificate = json.loads(
        (checkpoint / "source" / "certificate.json").read_text(
            encoding="ascii"
        )
    )
    events = load_events(checkpoint / "replay" / "replay.jsonl")
    setup = next(event for event in events if event["event"] == "setup")
    complete = next(event for event in events if event["event"] == "complete")
    expected = complete["output_phase_two_level_scout"]

    cutoff = int(source_certificate["N"])
    split = int(source_certificate["A"])
    inverse_limit = int(source_certificate["P"])
    output_limit = int(source_certificate["R"])
    primary = np.fromfile(
        checkpoint / "source" / "primary-f64le.bin",
        dtype="<f8",
        count=cutoff,
    )
    low = np.fromfile(
        checkpoint / "source" / "low-dual-f64le.bin",
        dtype="<f8",
        count=split,
    )
    transformed = np.fromfile(
        checkpoint / "source" / "transformed-complex-f64le.bin",
        dtype="<c16",
        count=int(source_certificate["transformed_count"]),
    )
    coefficients = np.fromfile(
        ROUTE / "build" / "mollifier-L01529-P2000-full-f64le.bin",
        dtype="<f8",
        count=inverse_limit,
    )
    inverse = np.zeros(inverse_limit + 1, dtype=np.float64)
    inverse[1:] = coefficients
    weights = np.fromfile(
        checkpoint
        / "weights"
        / "weights-u64-u64-f64le-lower-upper.bin",
        dtype=WEIGHT_DTYPE,
    )

    started = time.perf_counter()
    result = fused_two_level(
        inverse,
        primary,
        transformed,
        low,
        weights,
        transformed_first=int(source_certificate["stationary_lo"]),
        output_limit=output_limit,
        gamma_abs=float(source_certificate["gamma_bound_binary64"]),
        sigma=float(source_certificate["sigma_binary64"]),
        q_primary=complex(*setup["q_primary"]),
        q_dual=complex(*setup["q_dual"]),
        outer_ratio=float(expected["outer_ratio"]),
        tile_size=args.tile_size,
        threads=args.threads,
        backend="native",
    )
    wall_seconds = time.perf_counter() - started

    expected_records = expected["records"]
    if len(result.records) != len(expected_records):
        raise AssertionError("outer record count changed")
    for actual, retained in zip(
        result.records, expected_records, strict=True
    ):
        np.testing.assert_allclose(
            [actual.first_center.real, actual.first_center.imag],
            retained["first_center"],
            rtol=0,
            atol=5e-12,
        )
        np.testing.assert_allclose(
            [actual.second_center.real, actual.second_center.imag],
            retained["second_center"],
            rtol=0,
            atol=5e-12,
        )
        np.testing.assert_allclose(
            actual.two_level_upper,
            retained["two_level_upper"],
            rtol=0,
            atol=5e-12,
        )
    np.testing.assert_allclose(
        result.two_level_upper,
        expected["two_level_upper"],
        rtol=0,
        atol=5e-12,
    )
    np.testing.assert_allclose(
        result.weighted_l1_upper,
        expected["weighted_l1_upper"],
        rtol=0,
        atol=5e-12,
    )

    record = {
        "benchmark": "real_l01529_fused_two_level",
        "checkpoint": checkpoint.name,
        "N": cutoff,
        "P": inverse_limit,
        "R": output_limit,
        "threads": args.threads,
        "tile_size": args.tile_size,
        "primary_pairs": result.pairs.primary,
        "transformed_pairs": result.pairs.transformed,
        "low_pairs": result.pairs.low,
        "total_pairs": result.pairs.total,
        "fine_weight_block_count": result.fine_weight_block_count,
        "fine_piece_count": result.fine_piece_count,
        "outer_block_count": len(result.records),
        "two_level_upper": result.two_level_upper,
        "retained_two_level_upper": expected["two_level_upper"],
        "two_level_delta": (
            result.two_level_upper - expected["two_level_upper"]
        ),
        "weighted_l1_upper": result.weighted_l1_upper,
        "native_seconds": result.elapsed_seconds,
        "wall_seconds": wall_seconds,
        "pair_updates_per_second": result.pairs.total / wall_seconds,
        "peak_rss_bytes": peak_rss_bytes(),
    }
    print(json.dumps(record, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
