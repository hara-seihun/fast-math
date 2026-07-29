#!/usr/bin/env python3
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import statistics
import sys
import time

import finufft
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PROJECT_ROOT.parent
CANOPY_SOURCE = (
    RESEARCH_ROOT
    / "problems"
    / "riemann-hypothesis"
    / "scratch"
    / "proof-system--debruijn-newman-l00999-grouped-seam-independent-replay"
    / "source-canopy"
)
sys.path.insert(0, str(PROJECT_ROOT / "python"))
sys.path.insert(0, str(CANOPY_SOURCE))

from fast_math import Type1Plan1D  # noqa: E402
from scout_hilbert_e2_partial import truncated_inverse  # noqa: E402


def build_fixture(
    *,
    source_count: int,
    inverse_count: int,
    mode_count: int,
) -> dict[str, np.ndarray | int]:
    target = Fraction(999, 10_000)
    y_value = Fraction(7, 50)
    heat_time = target - y_value * y_value / 2
    t_value = float(heat_time)
    y_float = float(y_value)
    n_value = (mode_count // 4) - 1
    u_start = n_value * n_value - t_value / 16.0
    beta_start = (
        (1.0 + y_float) / 2.0 + t_value * np.log(u_start) / 4.0
    )

    integers = np.arange(1, source_count + 1, dtype=np.float64)
    logarithms = np.log(integers)
    source_amplitudes = np.exp(
        t_value * logarithms * logarithms / 4.0
        - beta_start * logarithms
    )
    source_coefficients = source_amplitudes * np.exp(
        2j * np.pi * np.remainder(u_start * logarithms, 1.0)
    )
    source_nodes = 2.0 * np.pi * np.remainder(logarithms, 1.0)

    inverse = truncated_inverse(inverse_count, t_value)
    inverse_logs = logarithms[:inverse_count]
    inverse_amplitudes = inverse[1:] * np.exp(
        -beta_start * inverse_logs
    )
    inverse_coefficients = inverse_amplitudes * np.exp(
        2j
        * np.pi
        * np.remainder(u_start * inverse_logs, 1.0)
    )
    inverse_nodes = 2.0 * np.pi * np.remainder(inverse_logs, 1.0)
    return {
        "mode_count": mode_count,
        "source_nodes": source_nodes,
        "source_coefficients": source_coefficients,
        "source_logs": logarithms,
        "inverse_nodes": inverse_nodes,
        "inverse_coefficients": inverse_coefficients,
        "inverse_logs": inverse_logs,
    }


def fill_coefficient_stack(
    output: np.ndarray,
    coefficients: np.ndarray,
    logarithms: np.ndarray,
    fractional_offset: float,
) -> None:
    for order, row in enumerate(output):
        np.copyto(row, coefficients)
        for _ in range(order):
            np.multiply(row, logarithms, out=row)
        np.multiply(
            row,
            np.exp(
                2j
                * np.pi
                * fractional_offset
                * logarithms
            ),
            out=row,
        )


def simple_transform(
    nodes: np.ndarray,
    coefficients: np.ndarray,
    logarithms: np.ndarray,
    fractional_offset: float,
    mode_count: int,
    eps: float,
) -> np.ndarray:
    return np.stack(
        (
            finufft.nufft1d1(
                nodes,
                coefficients
                * np.exp(
                    2j
                    * np.pi
                    * fractional_offset
                    * logarithms
                ),
                mode_count,
                eps=eps,
                isign=1,
                nthreads=1,
            ),
            finufft.nufft1d1(
                nodes,
                coefficients
                * logarithms
                * np.exp(
                    2j
                    * np.pi
                    * fractional_offset
                    * logarithms
                ),
                mode_count,
                eps=eps,
                isign=1,
                nthreads=1,
            ),
            finufft.nufft1d1(
                nodes,
                coefficients
                * logarithms
                * logarithms
                * np.exp(
                    2j
                    * np.pi
                    * fractional_offset
                    * logarithms
                ),
                mode_count,
                eps=eps,
                isign=1,
                nthreads=1,
            ),
        )
    )


def time_simple(
    fixture: dict[str, np.ndarray | int],
    *,
    mesh: int,
    source_eps: float,
    inverse_eps: float,
) -> float:
    started = time.perf_counter()
    for offset in range(mesh):
        fractional_offset = offset / mesh
        simple_transform(
            fixture["source_nodes"],
            fixture["source_coefficients"],
            fixture["source_logs"],
            fractional_offset,
            fixture["mode_count"],
            source_eps,
        )
        simple_transform(
            fixture["inverse_nodes"],
            fixture["inverse_coefficients"],
            fixture["inverse_logs"],
            fractional_offset,
            fixture["mode_count"],
            inverse_eps,
        )
    return time.perf_counter() - started


def time_persistent(
    fixture: dict[str, np.ndarray | int],
    *,
    mesh: int,
    source_eps: float,
    inverse_eps: float,
) -> tuple[float, float, float]:
    started = time.perf_counter()
    source_plan = Type1Plan1D(
        fixture["source_nodes"],
        fixture["mode_count"],
        n_trans=3,
        eps=source_eps,
        isign=1,
        nthreads=1,
    )
    inverse_plan = Type1Plan1D(
        fixture["inverse_nodes"],
        fixture["mode_count"],
        n_trans=3,
        eps=inverse_eps,
        isign=1,
        nthreads=1,
    )
    source_strengths = np.empty(
        (3, fixture["source_nodes"].size),
        dtype=np.complex128,
    )
    inverse_strengths = np.empty(
        (3, fixture["inverse_nodes"].size),
        dtype=np.complex128,
    )
    source_output = np.empty(
        (3, fixture["mode_count"]),
        dtype=np.complex128,
    )
    inverse_output = np.empty(
        (3, fixture["mode_count"]),
        dtype=np.complex128,
    )
    setup_seconds = time.perf_counter() - started
    execute_started = time.perf_counter()
    for offset in range(mesh):
        fractional_offset = offset / mesh
        fill_coefficient_stack(
            source_strengths,
            fixture["source_coefficients"],
            fixture["source_logs"],
            fractional_offset,
        )
        fill_coefficient_stack(
            inverse_strengths,
            fixture["inverse_coefficients"],
            fixture["inverse_logs"],
            fractional_offset,
        )
        source_plan.execute(
            source_strengths,
            out=source_output,
        )
        inverse_plan.execute(
            inverse_strengths,
            out=inverse_output,
        )
    execute_seconds = time.perf_counter() - execute_started
    return setup_seconds + execute_seconds, setup_seconds, execute_seconds


def parity_metrics(
    fixture: dict[str, np.ndarray | int],
    *,
    mesh: int,
    source_eps: float,
    inverse_eps: float,
) -> dict[str, float]:
    source_plan = Type1Plan1D(
        fixture["source_nodes"],
        fixture["mode_count"],
        n_trans=3,
        eps=source_eps,
        isign=1,
        nthreads=1,
    )
    inverse_plan = Type1Plan1D(
        fixture["inverse_nodes"],
        fixture["mode_count"],
        n_trans=3,
        eps=inverse_eps,
        isign=1,
        nthreads=1,
    )
    source_strengths = np.empty(
        (3, fixture["source_nodes"].size),
        dtype=np.complex128,
    )
    inverse_strengths = np.empty(
        (3, fixture["inverse_nodes"].size),
        dtype=np.complex128,
    )
    source_output = np.empty(
        (3, fixture["mode_count"]),
        dtype=np.complex128,
    )
    inverse_output = np.empty(
        (3, fixture["mode_count"]),
        dtype=np.complex128,
    )
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for offset in range(mesh):
        fractional_offset = offset / mesh
        for (
            nodes_key,
            coefficients_key,
            logs_key,
            strengths,
            output,
            plan,
            eps,
        ) in (
            (
                "source_nodes",
                "source_coefficients",
                "source_logs",
                source_strengths,
                source_output,
                source_plan,
                source_eps,
            ),
            (
                "inverse_nodes",
                "inverse_coefficients",
                "inverse_logs",
                inverse_strengths,
                inverse_output,
                inverse_plan,
                inverse_eps,
            ),
        ):
            fill_coefficient_stack(
                strengths,
                fixture[coefficients_key],
                fixture[logs_key],
                fractional_offset,
            )
            expected = simple_transform(
                fixture[nodes_key],
                fixture[coefficients_key],
                fixture[logs_key],
                fractional_offset,
                fixture["mode_count"],
                eps,
            )
            actual = plan.execute(strengths, out=output)
            difference = np.abs(actual - expected)
            scale = np.maximum(np.abs(expected), 1.0)
            maximum_absolute = max(
                maximum_absolute,
                float(np.max(difference)),
            )
            maximum_relative = max(
                maximum_relative,
                float(np.max(difference / scale)),
            )
    return {
        "maximum_absolute_difference": maximum_absolute,
        "maximum_scaled_difference": maximum_relative,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, default=500_000)
    parser.add_argument("--inverse-sources", type=int, default=32_768)
    parser.add_argument("--modes", type=int, default=131_072)
    parser.add_argument("--mesh", type=int, default=8)
    parser.add_argument("--source-eps", type=float, default=1e-10)
    parser.add_argument("--inverse-eps", type=float, default=1e-12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.inverse_sources <= args.sources:
        raise ValueError("require 1 <= inverse-sources <= sources")
    if args.modes < 4 or args.modes % 4:
        raise ValueError("modes must be a multiple of four")
    if args.mesh < 1 or args.repeats < 1:
        raise ValueError("mesh and repeats must be positive")

    fixture = build_fixture(
        source_count=args.sources,
        inverse_count=args.inverse_sources,
        mode_count=args.modes,
    )
    time_persistent(
        fixture,
        mesh=1,
        source_eps=args.source_eps,
        inverse_eps=args.inverse_eps,
    )
    simple_samples = [
        time_simple(
            fixture,
            mesh=args.mesh,
            source_eps=args.source_eps,
            inverse_eps=args.inverse_eps,
        )
        for _ in range(args.repeats)
    ]
    persistent_samples = []
    setup_samples = []
    execute_samples = []
    for _ in range(args.repeats):
        wall_seconds, setup_seconds, execute_seconds = time_persistent(
            fixture,
            mesh=args.mesh,
            source_eps=args.source_eps,
            inverse_eps=args.inverse_eps,
        )
        persistent_samples.append(wall_seconds)
        setup_samples.append(setup_seconds)
        execute_samples.append(execute_seconds)
    parity = parity_metrics(
        fixture,
        mesh=args.mesh,
        source_eps=args.source_eps,
        inverse_eps=args.inverse_eps,
    )
    simple_median = statistics.median(simple_samples)
    persistent_median = statistics.median(persistent_samples)
    record = {
        "benchmark": "finufft_fixed_R_product_canopy_transforms",
        "source_count": args.sources,
        "inverse_source_count": args.inverse_sources,
        "mode_count": args.modes,
        "mesh": args.mesh,
        "transforms_per_factor": 3,
        "source_eps": args.source_eps,
        "inverse_eps": args.inverse_eps,
        "repeats": args.repeats,
        "simple_wall_seconds_all": simple_samples,
        "simple_wall_seconds_median": simple_median,
        "persistent_wall_seconds_all": persistent_samples,
        "persistent_wall_seconds_median": persistent_median,
        "persistent_setup_seconds_all": setup_samples,
        "persistent_execute_seconds_all": execute_samples,
        "speedup_median": simple_median / persistent_median,
        **parity,
    }
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
