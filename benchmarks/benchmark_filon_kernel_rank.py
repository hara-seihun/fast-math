#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


RANKS = (2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)


def residuals(matrix: np.ndarray) -> dict[str, float]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    squared = singular_values * singular_values
    tail_energy = np.cumsum(squared[::-1])[::-1]
    total = float(tail_energy[0])
    return {
        str(rank): (
            float(np.sqrt(tail_energy[rank] / total))
            if rank < len(singular_values) and total != 0.0
            else 0.0
        )
        for rank in RANKS
        if rank <= len(singular_values)
    }


def analyze_window(
    kernel: np.memmap,
    *,
    start: int,
    count: int,
) -> dict[str, object]:
    started = time.perf_counter()
    matrix = np.asarray(kernel[:, start : start + count]).copy()
    load_seconds = time.perf_counter() - started
    norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix[:, norms != 0.0] / norms[norms != 0.0]
    return {
        "start": start,
        "stop": start + count,
        "load_seconds": load_seconds,
        "maximum_column_norm": float(norms.max(initial=0.0)),
        "minimum_nonzero_column_norm": float(
            norms[norms != 0.0].min(initial=np.inf)
        ),
        "raw_relative_frobenius_residual": residuals(matrix),
        "normalized_relative_frobenius_residual": residuals(normalized),
    }


def analyze_indices(
    kernel: np.memmap,
    *,
    indices: np.ndarray,
) -> dict[str, object]:
    started = time.perf_counter()
    matrix = np.asarray(kernel[:, indices]).copy()
    load_seconds = time.perf_counter() - started
    norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix[:, norms != 0.0] / norms[norms != 0.0]
    return {
        "sample_count": len(indices),
        "first_column": int(indices[0]),
        "last_column": int(indices[-1]),
        "load_seconds": load_seconds,
        "maximum_column_norm": float(norms.max(initial=0.0)),
        "minimum_nonzero_column_norm": float(
            norms[norms != 0.0].min(initial=np.inf)
        ),
        "raw_relative_frobenius_residual": residuals(matrix),
        "normalized_relative_frobenius_residual": residuals(normalized),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--window-columns", type=int, default=1024)
    parser.add_argument("--scattered-columns", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows <= 0:
        raise ValueError("rows must be positive")
    if args.window_columns <= 0:
        raise ValueError("window-columns must be positive")
    if args.scattered_columns <= 0:
        raise ValueError("scattered-columns must be positive")

    item_size = np.dtype(np.complex128).itemsize
    byte_count = args.kernel.stat().st_size
    row_bytes = args.rows * item_size
    if byte_count % row_bytes != 0:
        raise ValueError("kernel size is not divisible by rows")
    column_count = byte_count // row_bytes
    count = min(args.window_columns, column_count)
    starts = {
        "head": 0,
        "middle": max(0, (column_count - count) // 2),
        "tail": max(0, column_count - count),
    }
    kernel = np.memmap(
        args.kernel,
        dtype=np.complex128,
        mode="r",
        shape=(args.rows, column_count),
    )
    started = time.perf_counter()
    windows = {
        name: analyze_window(kernel, start=start, count=count)
        for name, start in starts.items()
    }
    tail_start = min(column_count - 1, max(count, 4096))
    scattered_count = min(
        args.scattered_columns,
        column_count - tail_start,
    )
    scattered_indices = np.unique(
        np.linspace(
            tail_start,
            column_count - 1,
            scattered_count,
            dtype=np.int64,
        ),
    )
    record = {
        "benchmark": "filon_kernel_numerical_rank",
        "kernel": str(args.kernel.resolve()),
        "rows": args.rows,
        "columns": column_count,
        "bytes": byte_count,
        "window_columns": count,
        "windows": windows,
        "scattered_tail": analyze_indices(
            kernel,
            indices=scattered_indices,
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
