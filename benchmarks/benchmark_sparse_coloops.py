#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
from scipy.sparse import load_npz, save_npz

from fast_math import sparse_block_coloops_mod_u32


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--prime", type=int, default=1_000_003)
    parser.add_argument("--row-block-size", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--residual-output", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    load_started = time.perf_counter()
    matrix = load_npz(args.matrix).tocsr()
    load_seconds = time.perf_counter() - load_started
    results = [
        sparse_block_coloops_mod_u32(
            matrix.indptr,
            matrix.indices,
            matrix.data,
            column_count=matrix.shape[1],
            prime=args.prime,
            row_block_size=args.row_block_size,
            backend="native",
        )
        for _ in range(args.repeats)
    ]
    first = results[0]
    for result in results[1:]:
        np.testing.assert_array_equal(
            result.residual_mask,
            first.residual_mask,
        )
        np.testing.assert_array_equal(
            result.removed_columns,
            first.removed_columns,
        )
        np.testing.assert_array_equal(
            result.certificate_row_starts,
            first.certificate_row_starts,
        )
        np.testing.assert_array_equal(
            result.certificate_coefficients,
            first.certificate_coefficients,
        )

    residual_write_seconds = 0.0
    residual_nonzeros = int(matrix[:, first.residual_mask].nnz)
    if args.residual_output is not None:
        residual_started = time.perf_counter()
        residual = matrix[:, first.residual_mask].tocsr()
        args.residual_output.parent.mkdir(parents=True, exist_ok=True)
        save_npz(args.residual_output, residual, compressed=True)
        residual_write_seconds = time.perf_counter() - residual_started

    samples = [result.elapsed_seconds for result in results]
    record = {
        "benchmark": "sparse_block_coloops",
        "matrix": str(args.matrix.resolve()),
        "rows": first.row_count,
        "columns": first.column_count,
        "input_nonzeros": first.input_nonzeros,
        "prime": first.prime,
        "row_block_size": first.row_block_size,
        "block_count": first.block_count,
        "block_incidences": first.block_incidences,
        "maximum_block_columns": first.maximum_block_columns,
        "active_columns": first.active_column_count,
        "removed_columns": len(first.removed_columns),
        "residual_columns": first.residual_column_count,
        "residual_nonzeros": residual_nonzeros,
        "blocks_processed": first.blocks_processed,
        "repeats": args.repeats,
        "kernel_seconds_all": samples,
        "kernel_seconds_best": min(samples),
        "kernel_seconds_median": statistics.median(samples),
        "load_seconds": load_seconds,
        "residual_write_seconds": residual_write_seconds,
        "residual_output": (
            str(args.residual_output.resolve())
            if args.residual_output is not None
            else None
        ),
    }
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
