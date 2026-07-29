#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import finufft
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from fast_math import Type1Plan1D, Type3Plan1D  # noqa: E402


def timed(function):
    started = time.perf_counter()
    result = function()
    return result, time.perf_counter() - started


def type3_benchmark(
    source_count: int,
    target_count: int,
    n_trans: int,
    executions: int,
    threads: int,
) -> list[dict]:
    sources = np.log(
        np.arange(1, source_count + 1, dtype=np.float64)
    )
    targets = np.linspace(-0.025, 0.025, target_count)
    base = np.exp(-0.03 * sources * sources + 0.2j * sources)
    strengths = [
        np.stack(
            [
                base * sources**order / math.factorial(order)
                for order in range(n_trans)
            ]
        )
        * np.exp(1e-4j * step * sources)
        for step in range(executions)
    ]

    simple_outputs = []
    simple_started = time.perf_counter()
    for coefficients in strengths:
        simple_outputs.append(
            finufft.nufft1d3(
                sources,
                coefficients,
                targets,
                eps=1e-12,
                isign=1,
                nthreads=threads,
            )
        )
    simple_seconds = time.perf_counter() - simple_started

    simple_output = np.empty(
        (n_trans, target_count), dtype=np.complex128
    )
    simple_preallocated_started = time.perf_counter()
    for coefficients in strengths:
        finufft.nufft1d3(
            sources,
            coefficients,
            targets,
            out=simple_output,
            eps=1e-12,
            isign=1,
            nthreads=threads,
        )
    simple_preallocated_seconds = (
        time.perf_counter() - simple_preallocated_started
    )

    plan, setup_seconds = timed(
        lambda: Type3Plan1D(
            sources,
            targets,
            n_trans=n_trans,
            eps=1e-12,
            isign=1,
            nthreads=threads,
        )
    )
    planned_output = np.empty(
        (n_trans, target_count), dtype=np.complex128
    )
    execute_started = time.perf_counter()
    for coefficients in strengths:
        plan.execute(coefficients, out=planned_output)
    execute_seconds = time.perf_counter() - execute_started
    for coefficients, expected in zip(
        strengths, simple_outputs, strict=True
    ):
        actual = plan.execute(coefficients, out=planned_output)
        np.testing.assert_allclose(
            actual, expected, rtol=3e-12, atol=3e-12
        )

    return [
        {
            "benchmark": "finufft_type3_repeated",
            "backend": "simple",
            "source_count": source_count,
            "target_count": target_count,
            "n_trans": n_trans,
            "executions": executions,
            "threads": threads,
            "wall_seconds": simple_seconds,
        },
        {
            "benchmark": "finufft_type3_repeated",
            "backend": "simple_preallocated",
            "source_count": source_count,
            "target_count": target_count,
            "n_trans": n_trans,
            "executions": executions,
            "threads": threads,
            "wall_seconds": simple_preallocated_seconds,
        },
        {
            "benchmark": "finufft_type3_repeated",
            "backend": "persistent_plan",
            "source_count": source_count,
            "target_count": target_count,
            "n_trans": n_trans,
            "executions": executions,
            "threads": threads,
            "setup_seconds": setup_seconds,
            "execute_seconds": execute_seconds,
            "wall_seconds": setup_seconds + execute_seconds,
            "execute_only_speedup_vs_simple": simple_seconds
            / execute_seconds,
            "execute_only_speedup_vs_preallocated_simple":
            simple_preallocated_seconds / execute_seconds,
            "amortized_speedup_vs_simple": simple_seconds
            / (setup_seconds + execute_seconds),
            "amortized_speedup_vs_preallocated_simple":
            simple_preallocated_seconds
            / (setup_seconds + execute_seconds),
        },
    ]


def type1_benchmark(
    source_count: int,
    mode_count: int,
    executions: int,
    threads: int,
) -> list[dict]:
    nodes = 2.0 * np.pi * np.remainder(
        np.log(np.arange(1, source_count + 1, dtype=np.float64)),
        1.0,
    )
    base = np.exp(-0.02 * np.log1p(np.arange(source_count))) * np.exp(
        0.17j * nodes
    )
    strengths = [
        base * np.exp(2j * np.pi * step * nodes / executions)
        for step in range(executions)
    ]

    simple_outputs = []
    simple_started = time.perf_counter()
    for coefficients in strengths:
        simple_outputs.append(
            finufft.nufft1d1(
                nodes,
                coefficients,
                mode_count,
                eps=1e-12,
                isign=1,
                nthreads=threads,
            )
        )
    simple_seconds = time.perf_counter() - simple_started

    simple_output = np.empty(mode_count, dtype=np.complex128)
    simple_preallocated_started = time.perf_counter()
    for coefficients in strengths:
        finufft.nufft1d1(
            nodes,
            coefficients,
            mode_count,
            out=simple_output,
            eps=1e-12,
            isign=1,
            nthreads=threads,
        )
    simple_preallocated_seconds = (
        time.perf_counter() - simple_preallocated_started
    )

    plan, setup_seconds = timed(
        lambda: Type1Plan1D(
            nodes,
            mode_count,
            eps=1e-12,
            isign=1,
            nthreads=threads,
        )
    )
    output = np.empty(mode_count, dtype=np.complex128)
    execute_started = time.perf_counter()
    for coefficients in strengths:
        plan.execute(coefficients, out=output)
    execute_seconds = time.perf_counter() - execute_started
    for coefficients, expected in zip(
        strengths, simple_outputs, strict=True
    ):
        actual = plan.execute(coefficients, out=output)
        np.testing.assert_allclose(
            actual, expected, rtol=3e-12, atol=3e-12
        )

    return [
        {
            "benchmark": "finufft_type1_repeated",
            "backend": "simple",
            "source_count": source_count,
            "mode_count": mode_count,
            "executions": executions,
            "threads": threads,
            "wall_seconds": simple_seconds,
        },
        {
            "benchmark": "finufft_type1_repeated",
            "backend": "simple_preallocated",
            "source_count": source_count,
            "mode_count": mode_count,
            "executions": executions,
            "threads": threads,
            "wall_seconds": simple_preallocated_seconds,
        },
        {
            "benchmark": "finufft_type1_repeated",
            "backend": "persistent_plan",
            "source_count": source_count,
            "mode_count": mode_count,
            "executions": executions,
            "threads": threads,
            "setup_seconds": setup_seconds,
            "execute_seconds": execute_seconds,
            "wall_seconds": setup_seconds + execute_seconds,
            "execute_only_speedup_vs_simple": simple_seconds
            / execute_seconds,
            "execute_only_speedup_vs_preallocated_simple":
            simple_preallocated_seconds / execute_seconds,
            "amortized_speedup_vs_simple": simple_seconds
            / (setup_seconds + execute_seconds),
            "amortized_speedup_vs_preallocated_simple":
            simple_preallocated_seconds
            / (setup_seconds + execute_seconds),
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, default=200_000)
    parser.add_argument("--targets", type=int, default=4_096)
    parser.add_argument("--modes", type=int, default=65_536)
    parser.add_argument("--transforms", type=int, default=5)
    parser.add_argument("--executions", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = type3_benchmark(
        args.sources,
        args.targets,
        args.transforms,
        args.executions,
        args.threads,
    )
    records.extend(
        type1_benchmark(
            args.sources,
            args.modes,
            args.executions,
            args.threads,
        )
    )
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
