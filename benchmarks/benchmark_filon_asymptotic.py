#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def chebyshev_lobatto_differentiation(degree: int) -> np.ndarray:
    indices = np.arange(degree + 1, dtype=np.int64)
    nodes = np.cos(np.pi * indices / degree)
    scales = np.ones(degree + 1)
    scales[[0, -1]] = 2.0
    scales *= np.where(indices % 2 == 0, 1.0, -1.0)
    differences = nodes[:, None] - nodes[None, :]
    matrix = np.divide(
        scales[:, None],
        scales[None, :] * differences,
        out=np.zeros_like(differences),
        where=differences != 0.0,
    )
    matrix[np.diag_indices_from(matrix)] = -matrix.sum(axis=1)
    return matrix


def endpoint_derivatives(
    degree: int,
    term_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    differentiation = chebyshev_lobatto_differentiation(degree)
    positive = np.empty((term_count, degree + 1), dtype=np.float64)
    negative = np.empty_like(positive)
    power = np.eye(degree + 1)
    for order in range(term_count):
        positive[order] = power[0]
        negative[order] = power[-1]
        power = differentiation @ power
    return positive, negative


def approximate_weights(
    h: np.ndarray,
    *,
    degree: int,
    eta: float,
    length: float,
    term_count: int,
) -> np.ndarray:
    positive, negative = endpoint_derivatives(degree, term_count)
    t = h.astype(np.float64) * eta / 2.0
    exp_positive = np.exp(1j * t)
    exp_negative = np.exp(-1j * t)
    inverse_it = 1.0 / (1j * t)
    factor = inverse_it.copy()
    weights = np.zeros((degree + 1, len(h)), dtype=np.complex128)
    sign = 1.0
    for order in range(term_count):
        boundary = (
            positive[order, :, None] * exp_positive
            - negative[order, :, None] * exp_negative
        )
        weights += sign * boundary * factor
        sign = -sign
        factor *= inverse_it
    return length / 2.0 * weights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--length", type=float, required=True)
    parser.add_argument(
        "--columns",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--terms",
        type=int,
        nargs="+",
        default=[2, 3, 4, 6, 8, 12],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.degree <= 0:
        raise ValueError("degree must be positive")
    if any(column <= 0 for column in args.columns):
        raise ValueError("columns must be positive")
    if any(term <= 0 for term in args.terms):
        raise ValueError("terms must be positive")

    row_count = args.degree // 2 + 1
    item_size = np.dtype(np.complex128).itemsize
    column_count = args.kernel.stat().st_size // (row_count * item_size)
    columns = np.asarray(args.columns, dtype=np.int64)
    if np.any(columns >= column_count):
        raise ValueError("column is outside kernel")
    kernel = np.memmap(
        args.kernel,
        dtype=np.complex128,
        mode="r",
        shape=(row_count, column_count),
    )
    expected = np.asarray(kernel[:, columns]).copy()
    expected_norm = np.linalg.norm(expected, axis=0)
    records = {}
    started = time.perf_counter()
    for term_count in args.terms:
        actual = approximate_weights(
            columns,
            degree=args.degree,
            eta=args.eta,
            length=args.length,
            term_count=term_count,
        )[:row_count]
        error = np.abs(actual - expected)
        records[str(term_count)] = {
            "maximum_absolute_error": float(error.max(initial=0.0)),
            "maximum_relative_column_error": float(
                np.max(
                    np.linalg.norm(actual - expected, axis=0)
                    / expected_norm,
                    initial=0.0,
                )
            ),
            "relative_column_errors": [
                float(value)
                for value in (
                    np.linalg.norm(actual - expected, axis=0)
                    / expected_norm
                )
            ],
        }

    record = {
        "benchmark": "filon_asymptotic_weights",
        "kernel": str(args.kernel.resolve()),
        "degree": args.degree,
        "eta": args.eta,
        "length": args.length,
        "columns": args.columns,
        "terms": records,
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
