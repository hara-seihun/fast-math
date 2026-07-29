#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import finufft
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from fast_math import (  # noqa: E402
    Type3FixedPlan1D,
    Type3Plan1D,
    Type3SignPairPlan1D,
)


def benchmark(
    source_count: int,
    target_count: int,
    chunk_count: int,
    n_trans: int,
    threads: int,
    eps: float,
    repeats: int,
) -> dict:
    sources = np.log(
        np.arange(1, source_count + 1, dtype=np.float64)
    )
    positive_base = np.exp(
        -0.03 * sources * sources + 0.2j * sources
    )
    negative_base = np.exp(
        -0.025 * sources * sources - 0.17j * sources
    )
    positive_strengths = np.empty(
        (n_trans, source_count), dtype=np.complex128
    )
    negative_strengths = np.empty_like(positive_strengths)
    positive_strengths[0] = positive_base
    negative_strengths[0] = negative_base
    for degree in range(1, n_trans):
        positive_strengths[degree] = (
            positive_strengths[degree - 1] * sources / degree
        )
        negative_strengths[degree] = (
            negative_strengths[degree - 1] * sources / degree
        )
    target_chunks = [
        np.linspace(
            -0.022 + 0.011 * chunk,
            -0.011 + 0.011 * chunk,
            target_count,
        )
        for chunk in range(chunk_count)
    ]

    finufft.nufft1d3(
        sources[:1000],
        positive_strengths[:, :1000].copy(order="C"),
        target_chunks[0][:1000],
        eps=eps,
        isign=1,
        nthreads=threads,
    )

    simple_times = []
    expected_outputs = []
    for repeat in range(repeats):
        outputs = []
        started = time.perf_counter()
        for targets in target_chunks:
            outputs.append(
                finufft.nufft1d3(
                    sources,
                    positive_strengths,
                    targets,
                    eps=eps,
                    isign=1,
                    nthreads=threads,
                )
            )
            outputs.append(
                finufft.nufft1d3(
                    sources,
                    negative_strengths,
                    -targets,
                    eps=eps,
                    isign=1,
                    nthreads=threads,
                )
            )
        simple_times.append(time.perf_counter() - started)
        if repeat == 0:
            expected_outputs = outputs

    setup_started = time.perf_counter()
    positive_plan = Type3Plan1D(
        sources,
        target_chunks[0],
        n_trans=n_trans,
        eps=eps,
        isign=1,
        nthreads=threads,
    )
    negative_plan = Type3Plan1D(
        sources,
        -target_chunks[0],
        n_trans=n_trans,
        eps=eps,
        isign=1,
        nthreads=threads,
    )
    setup_seconds = time.perf_counter() - setup_started
    positive_output = np.empty(
        (n_trans, target_count), dtype=np.complex128
    )
    negative_output = np.empty(
        (n_trans, target_count), dtype=np.complex128
    )

    persistent_times = []
    max_absolute_difference = 0.0
    max_relative_difference = 0.0
    for repeat in range(repeats):
        output_index = 0
        started = time.perf_counter()
        for targets in target_chunks:
            positive_plan.set_targets(targets)
            positive_plan.execute(
                positive_strengths, out=positive_output
            )
            if repeat == 0:
                expected = expected_outputs[output_index]
                difference = np.abs(positive_output - expected)
                max_absolute_difference = max(
                    max_absolute_difference,
                    float(np.max(difference)),
                )
                max_relative_difference = max(
                    max_relative_difference,
                    float(
                        np.max(
                            difference
                            / np.maximum(np.abs(expected), 1e-300)
                        )
                    ),
                )
            output_index += 1

            negative_plan.set_targets(-targets)
            negative_plan.execute(
                negative_strengths, out=negative_output
            )
            if repeat == 0:
                expected = expected_outputs[output_index]
                difference = np.abs(negative_output - expected)
                max_absolute_difference = max(
                    max_absolute_difference,
                    float(np.max(difference)),
                )
                max_relative_difference = max(
                    max_relative_difference,
                    float(
                        np.max(
                            difference
                            / np.maximum(np.abs(expected), 1e-300)
                        )
                    ),
                )
            output_index += 1
        persistent_times.append(time.perf_counter() - started)

    single_simple_times = []
    single_persistent_times = []
    single_plan = Type3FixedPlan1D(
        sources,
        positive_strengths,
        target_chunks[0],
        n_trans=n_trans,
        eps=eps,
        nthreads=threads,
    )
    for _ in range(repeats):
        started = time.perf_counter()
        for targets in target_chunks:
            finufft.nufft1d3(
                sources,
                positive_strengths,
                targets,
                eps=eps,
                isign=1,
                nthreads=threads,
            )
        single_simple_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        for targets in target_chunks:
            single_plan.execute(targets=targets)
        single_persistent_times.append(
            time.perf_counter() - started
        )

    fused_setup_started = time.perf_counter()
    fused_plan = Type3SignPairPlan1D(
        sources,
        positive_strengths,
        negative_strengths,
        target_chunks[0],
        n_trans=n_trans,
        eps=eps,
        nthreads=threads,
    )
    fused_setup_seconds = time.perf_counter() - fused_setup_started
    fused_times = []
    fused_max_absolute_difference = 0.0
    fused_max_relative_difference = 0.0
    for repeat in range(repeats):
        output_index = 0
        started = time.perf_counter()
        for targets in target_chunks:
            fused_positive, fused_negative = fused_plan.execute(
                targets=targets
            )
            if repeat == 0:
                for actual in (
                    fused_positive,
                    fused_negative,
                ):
                    expected = expected_outputs[output_index]
                    difference = np.abs(actual - expected)
                    fused_max_absolute_difference = max(
                        fused_max_absolute_difference,
                        float(np.max(difference)),
                    )
                    fused_max_relative_difference = max(
                        fused_max_relative_difference,
                        float(
                            np.max(
                                difference
                                / np.maximum(
                                    np.abs(expected), 1e-300
                                )
                            )
                        ),
                    )
                    output_index += 1
        fused_times.append(time.perf_counter() - started)

    simple_median = float(np.median(simple_times))
    persistent_median = float(np.median(persistent_times))
    fused_median = float(np.median(fused_times))
    single_simple_median = float(np.median(single_simple_times))
    single_persistent_median = float(
        np.median(single_persistent_times)
    )
    return {
        "benchmark": "finufft_type3_changing_target_cells",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "sys_platform": sys.platform,
        "source_count": source_count,
        "target_count": target_count,
        "chunk_count": chunk_count,
        "n_trans": n_trans,
        "executions": 2 * chunk_count,
        "threads": threads,
        "eps": eps,
        "repeats": repeats,
        "simple_seconds": simple_times,
        "persistent_setup_seconds": setup_seconds,
        "persistent_seconds": persistent_times,
        "simple_median_seconds": simple_median,
        "persistent_median_seconds": persistent_median,
        "persistent_speedup": simple_median / persistent_median,
        "max_absolute_difference": max_absolute_difference,
        "max_relative_difference": max_relative_difference,
        "single_simple_seconds": single_simple_times,
        "single_persistent_seconds": single_persistent_times,
        "single_simple_median_seconds": single_simple_median,
        "single_persistent_median_seconds": (
            single_persistent_median
        ),
        "single_persistent_speedup": (
            single_simple_median / single_persistent_median
        ),
        "fused_setup_seconds": fused_setup_seconds,
        "fused_seconds": fused_times,
        "fused_median_seconds": fused_median,
        "fused_speedup": simple_median / fused_median,
        "fused_speedup_vs_separate_plans": (
            persistent_median / fused_median
        ),
        "fused_max_absolute_difference": (
            fused_max_absolute_difference
        ),
        "fused_max_relative_difference": (
            fused_max_relative_difference
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, default=690_988)
    parser.add_argument("--targets", type=int, default=500_000)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--transforms", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.transforms <= 128:
        raise ValueError("transforms must lie in 1..128")
    if args.sources < 1 or args.targets < 1 or args.chunks < 1:
        raise ValueError("fixture dimensions must be positive")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")

    record = benchmark(
        source_count=args.sources,
        target_count=args.targets,
        chunk_count=args.chunks,
        n_trans=args.transforms,
        threads=args.threads,
        eps=args.eps,
        repeats=args.repeats,
    )
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
