#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from scipy.sparse import load_npz

from fast_math import sparse_rank_mod_u32


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--prime", type=int, default=1_000_003)
    parser.add_argument("--target-rank", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--baseline-seconds", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    load_started = time.perf_counter()
    matrix = load_npz(args.matrix).tocsr()
    load_seconds = time.perf_counter() - load_started
    results = [
        sparse_rank_mod_u32(
            matrix.indptr,
            matrix.indices,
            matrix.data,
            column_count=matrix.shape[1],
            prime=args.prime,
            target_rank=args.target_rank,
            backend="native",
        )
        for _ in range(args.repeats)
    ]
    first = results[0]
    for result in results[1:]:
        if result.rank != first.rank:
            raise AssertionError("rank changed across repeats")
        if result.elimination_steps != first.elimination_steps:
            raise AssertionError("elimination count changed across repeats")
        if not (
            (result.pivot_rows == first.pivot_rows).all()
            and (result.pivot_columns == first.pivot_columns).all()
        ):
            raise AssertionError("pivot witness changed across repeats")

    samples = [result.elapsed_seconds for result in results]
    record = {
        "benchmark": "sparse_modular_rank_npz",
        "matrix": str(args.matrix.resolve()),
        "rows": first.row_count,
        "columns": first.column_count,
        "input_nonzeros": first.input_nonzeros,
        "prime": first.prime,
        "target_rank": args.target_rank,
        "rank": first.rank,
        "target_reached": first.target_reached,
        "peeled_pivots": first.peeled_pivots,
        "residual_rows": first.residual_rows,
        "residual_columns": first.residual_columns,
        "residual_nonzeros": first.residual_nonzeros,
        "processed_rows": first.processed_rows,
        "dependent_rows": first.dependent_rows,
        "elimination_steps": first.elimination_steps,
        "basis_nonzeros": first.basis_nonzeros,
        "maximum_basis_size": first.maximum_basis_size,
        "maximum_working_size": first.maximum_working_size,
        "preprocessing_seconds": first.preprocessing_seconds,
        "repeats": args.repeats,
        "kernel_seconds_all": samples,
        "kernel_seconds_best": min(samples),
        "kernel_seconds_median": statistics.median(samples),
        "load_seconds": load_seconds,
    }
    if args.baseline_seconds is not None:
        record["baseline_seconds"] = args.baseline_seconds
        record["speedup_best"] = (
            args.baseline_seconds / min(samples)
        )
        record["speedup_median"] = (
            args.baseline_seconds / statistics.median(samples)
        )

    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="ascii")
    return 0 if first.target_reached else 1


if __name__ == "__main__":
    raise SystemExit(main())
