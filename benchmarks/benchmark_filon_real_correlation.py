#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from fast_math import filon_chebyshev_inner_product


def timed(work, repeat: int) -> tuple[list[object], list[float]]:
    values = []
    seconds = []
    for _ in range(repeat):
        started = time.perf_counter()
        values.append(work())
        seconds.append(time.perf_counter() - started)
    return values, seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("correlation", type=Path)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--node-index", type=int, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--length", type=float, required=True)
    parser.add_argument("--output-count", type=int, required=True)
    parser.add_argument("--cutoff", type=int, required=True)
    parser.add_argument("--term-count", type=int, required=True)
    parser.add_argument("--threads", type=int, nargs="+", required=True)
    parser.add_argument("--chunk-size", type=int, default=1 << 16)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    item_size = np.dtype(np.complex128).itemsize
    correlation_count = args.correlation.stat().st_size // item_size
    row_count = args.degree // 2 + 1
    kernel_column_count = (
        args.kernel.stat().st_size // (row_count * item_size)
    )
    if args.output_count > kernel_column_count:
        raise ValueError("output-count exceeds the kernel")
    if correlation_count < 2 * args.output_count - 1:
        raise ValueError("correlation does not contain both lag directions")
    if not 1 <= args.cutoff <= args.output_count:
        raise ValueError("cutoff is outside output range")

    correlation = np.memmap(
        args.correlation,
        dtype=np.complex128,
        mode="r",
        shape=(correlation_count,),
    )
    kernel = np.memmap(
        args.kernel,
        dtype=np.complex128,
        mode="r",
        shape=(row_count, kernel_column_count),
    )
    weights = np.asarray(
        kernel[args.node_index, : args.output_count]
    )
    exact_prefix = weights[: args.cutoff]

    def exact_inner_product() -> complex:
        result = correlation[0] * weights[0]
        if args.output_count > 1:
            result += np.dot(
                correlation[1 : args.output_count],
                weights[1 : args.output_count],
            )
            result += np.dot(
                correlation[-1 : -args.output_count : -1],
                np.conjugate(weights[1 : args.output_count]),
            )
        return complex(result)

    exact_values, exact_seconds = timed(
        exact_inner_product, args.repeat
    )
    exact_value = exact_values[-1]
    native_rows = []
    for threads in args.threads:
        values, seconds = timed(
            lambda: filon_chebyshev_inner_product(
                correlation,
                exact_prefix,
                degree=args.degree,
                node_index=args.node_index,
                output_count=args.output_count,
                eta=args.eta,
                length=args.length,
                term_count=args.term_count,
                chunk_size=args.chunk_size,
                threads=threads,
                backend="native",
            ),
            args.repeat,
        )
        value = values[-1].value
        native_rows.append(
            {
                "threads": threads,
                "seconds": seconds,
                "best_seconds": min(seconds),
                "median_seconds": float(np.median(seconds)),
                "value": [value.real, value.imag],
                "relative_error": (
                    abs(value - exact_value)
                    / max(abs(exact_value), np.finfo(float).tiny)
                ),
            }
        )

    record = {
        "benchmark": "filon_real_correlation",
        "label": args.label,
        "correlation": str(args.correlation.resolve()),
        "correlation_count": correlation_count,
        "kernel": str(args.kernel.resolve()),
        "degree": args.degree,
        "node_index": args.node_index,
        "eta": args.eta,
        "length": args.length,
        "output_count": args.output_count,
        "cutoff": args.cutoff,
        "term_count": args.term_count,
        "chunk_size": args.chunk_size,
        "repeat": args.repeat,
        "exact_seconds": exact_seconds,
        "exact_best_seconds": min(exact_seconds),
        "exact_median_seconds": float(np.median(exact_seconds)),
        "exact_value": [exact_value.real, exact_value.imag],
        "native": native_rows,
    }
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
