#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from fast_math import filon_chebyshev_inner_product


def build_correlation(
    output_count: int,
    *,
    block_size: int,
) -> np.ndarray:
    correlation = np.empty(
        2 * output_count - 1,
        dtype=np.complex128,
    )
    decay = max(1.0, output_count / 7.0)
    for begin in range(0, output_count, block_size):
        end = min(output_count, begin + block_size)
        lags = np.arange(begin, end, dtype=np.float64)
        correlation[begin:end] = (
            np.exp(-lags / decay)
            * np.exp(0.000173j * lags)
            * (1.0 + 0.125 * np.cos(0.000311 * lags))
        )
    correlation[output_count:] = np.conjugate(
        correlation[1:output_count][::-1]
    )
    return correlation


def exact_inner_product(
    correlation: np.ndarray,
    weights: np.ndarray,
    output_count: int,
) -> complex:
    result = correlation[0] * weights[0]
    if output_count > 1:
        result += np.dot(
            correlation[1:output_count],
            weights[1:output_count],
        )
        result += np.dot(
            correlation[-1 : -output_count : -1],
            np.conjugate(weights[1:output_count]),
        )
    return complex(result)


def timed(work, repeat: int) -> tuple[object, list[float]]:
    durations = []
    result = None
    for _ in range(repeat):
        started = time.perf_counter()
        result = work()
        durations.append(time.perf_counter() - started)
    return result, durations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--node-index", type=int, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--length", type=float, required=True)
    parser.add_argument("--output-count", type=int, required=True)
    parser.add_argument(
        "--cutoff-terms",
        type=int,
        nargs=2,
        action="append",
        metavar=("CUTOFF", "TERMS"),
        required=True,
    )
    parser.add_argument("--threads", type=int, nargs="+", default=[1])
    parser.add_argument("--chunk-size", type=int, default=1 << 16)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.degree <= 0:
        raise ValueError("degree must be positive")
    if not 0 <= args.node_index <= args.degree // 2:
        raise ValueError("node-index is outside the stored kernel rows")
    if args.output_count <= 0 or args.repeat <= 0:
        raise ValueError("output-count and repeat must be positive")

    row_count = args.degree // 2 + 1
    item_size = np.dtype(np.complex128).itemsize
    kernel_column_count = (
        args.kernel.stat().st_size // (row_count * item_size)
    )
    if args.output_count > kernel_column_count:
        raise ValueError("output-count exceeds the kernel")
    kernel = np.memmap(
        args.kernel,
        dtype=np.complex128,
        mode="r",
        shape=(row_count, kernel_column_count),
    )
    weights = np.asarray(
        kernel[args.node_index, : args.output_count]
    )
    correlation = build_correlation(
        args.output_count,
        block_size=args.chunk_size,
    )

    exact_value, numpy_seconds = timed(
        lambda: exact_inner_product(
            correlation,
            weights,
            args.output_count,
        ),
        args.repeat,
    )
    exact_native, native_exact_seconds = timed(
        lambda: filon_chebyshev_inner_product(
            correlation,
            weights,
            degree=args.degree,
            node_index=args.node_index,
            output_count=args.output_count,
            eta=args.eta,
            length=args.length,
            term_count=1,
            chunk_size=args.chunk_size,
            threads=1,
            backend="native",
        ),
        args.repeat,
    )

    hybrids = []
    for cutoff, term_count in args.cutoff_terms:
        if not 1 <= cutoff <= args.output_count:
            raise ValueError("cutoff is outside output range")
        exact_prefix = weights[:cutoff]
        for threads in args.threads:
            hybrid, durations = timed(
                lambda: filon_chebyshev_inner_product(
                    correlation,
                    exact_prefix,
                    degree=args.degree,
                    node_index=args.node_index,
                    output_count=args.output_count,
                    eta=args.eta,
                    length=args.length,
                    term_count=term_count,
                    chunk_size=args.chunk_size,
                    threads=threads,
                    backend="native",
                ),
                args.repeat,
            )
            hybrids.append(
                {
                    "cutoff": cutoff,
                    "term_count": term_count,
                    "threads": threads,
                    "seconds": durations,
                    "best_seconds": min(durations),
                    "median_seconds": float(np.median(durations)),
                    "absolute_error": abs(hybrid.value - exact_value),
                    "relative_error": (
                        abs(hybrid.value - exact_value)
                        / max(abs(exact_value), np.finfo(float).tiny)
                    ),
                    "value": [hybrid.value.real, hybrid.value.imag],
                }
            )

    record = {
        "benchmark": "filon_chebyshev_inner_product",
        "kernel": str(args.kernel.resolve()),
        "degree": args.degree,
        "node_index": args.node_index,
        "eta": args.eta,
        "length": args.length,
        "output_count": args.output_count,
        "correlation_count": len(correlation),
        "chunk_size": args.chunk_size,
        "repeat": args.repeat,
        "exact_value": [exact_value.real, exact_value.imag],
        "numpy_seconds": numpy_seconds,
        "numpy_best_seconds": min(numpy_seconds),
        "native_exact_value": [
            exact_native.value.real,
            exact_native.value.imag,
        ],
        "native_exact_seconds": native_exact_seconds,
        "native_exact_best_seconds": min(native_exact_seconds),
        "native_exact_absolute_error": abs(
            exact_native.value - exact_value
        ),
        "hybrids": hybrids,
    }
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
