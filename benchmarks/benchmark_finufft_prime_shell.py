#!/usr/bin/env python3
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import statistics
import sys
import time

import finufft
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from fast_math import Type1Plan1D  # noqa: E402


def primes_through(limit: int) -> np.ndarray:
    sieve = np.ones(limit + 1, dtype=np.bool_)
    sieve[:2] = False
    for prime in range(2, math.isqrt(limit) + 1):
        if sieve[prime]:
            sieve[prime * prime : limit + 1 : prime] = False
    return np.flatnonzero(sieve)


def build_shells(
    *,
    n_value: int,
    lower_limit: int,
    upper_limit: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
    heat_time = Fraction(999, 10_000) - Fraction(7, 50) ** 2 / 2
    t_value = float(heat_time)
    y_value = 7.0 / 50.0
    u_start = n_value * n_value - t_value / 16.0
    beta_start = (
        (1.0 + y_value) / 2.0 + t_value * math.log(u_start) / 4.0
    )
    primes = primes_through(upper_limit)
    shells = []
    lower = lower_limit
    while lower < upper_limit:
        upper = min(upper_limit, 2 * lower)
        block = primes[
            (primes > lower) & (primes <= upper)
        ].astype(np.float64)
        logarithms = np.log(block)
        amplitudes = np.exp(
            t_value * logarithms * logarithms / 4.0
            - beta_start * logarithms
        )
        coefficients = amplitudes * np.exp(
            2j
            * np.pi
            * np.remainder(u_start * logarithms, 1.0)
        )
        nodes = 2.0 * np.pi * np.remainder(logarithms, 1.0)
        shells.append((logarithms, coefficients, nodes))
        lower = upper
    return shells, int(sum(len(shell[0]) for shell in shells))


def simple_outputs(
    shell: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    mode_count: int,
    mesh: int,
    eps: float,
) -> list[np.ndarray]:
    logarithms, coefficients, nodes = shell
    return [
        finufft.nufft1d1(
            nodes,
            coefficients
            * np.exp(
                2j
                * np.pi
                * (offset / mesh)
                * logarithms
            ),
            mode_count,
            eps=eps,
            isign=1,
            nthreads=1,
        )
        for offset in range(mesh)
    ]


def time_simple(
    shells: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    mode_count: int,
    mesh: int,
    eps: float,
) -> float:
    started = time.perf_counter()
    for shell in shells:
        simple_outputs(
            shell,
            mode_count=mode_count,
            mesh=mesh,
            eps=eps,
        )
    return time.perf_counter() - started


def time_persistent(
    shells: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    mode_count: int,
    mesh: int,
    eps: float,
) -> float:
    started = time.perf_counter()
    for logarithms, coefficients, nodes in shells:
        plan = Type1Plan1D(
            nodes,
            mode_count,
            eps=eps,
            isign=1,
            nthreads=1,
        )
        shifted = np.empty_like(coefficients)
        modes = np.empty(mode_count, dtype=np.complex128)
        for offset in range(mesh):
            np.multiply(
                coefficients,
                np.exp(
                    2j
                    * np.pi
                    * (offset / mesh)
                    * logarithms
                ),
                out=shifted,
            )
            plan.execute(shifted, out=modes)
    return time.perf_counter() - started


def parity_metrics(
    shells: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    mode_count: int,
    mesh: int,
    eps: float,
) -> dict[str, float | bool]:
    maximum_absolute = 0.0
    maximum_scaled = 0.0
    witnesses_equal = True
    for shell in shells:
        logarithms, coefficients, nodes = shell
        expected_outputs = simple_outputs(
            shell,
            mode_count=mode_count,
            mesh=mesh,
            eps=eps,
        )
        plan = Type1Plan1D(
            nodes,
            mode_count,
            eps=eps,
            isign=1,
            nthreads=1,
        )
        shifted = np.empty_like(coefficients)
        actual = np.empty(mode_count, dtype=np.complex128)
        for offset, expected in enumerate(expected_outputs):
            np.multiply(
                coefficients,
                np.exp(
                    2j
                    * np.pi
                    * (offset / mesh)
                    * logarithms
                ),
                out=shifted,
            )
            plan.execute(shifted, out=actual)
            difference = np.abs(actual - expected)
            scale = np.maximum(np.abs(expected), 1.0)
            maximum_absolute = max(
                maximum_absolute,
                float(np.max(difference)),
            )
            maximum_scaled = max(
                maximum_scaled,
                float(np.max(difference / scale)),
            )
            witnesses_equal &= int(np.argmax(np.abs(actual))) == int(
                np.argmax(np.abs(expected))
            )
    return {
        "maximum_absolute_difference": maximum_absolute,
        "maximum_scaled_difference": maximum_scaled,
        "full_mode_witnesses_equal": witnesses_equal,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=32_767)
    parser.add_argument("--P", type=int, default=32_768)
    parser.add_argument("--R", type=int, default=2_000_000)
    parser.add_argument("--mesh", type=int, default=8)
    parser.add_argument("--nufft-eps", type=float, default=1e-10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.N < 1 or args.P < 1 or args.R <= args.P:
        raise ValueError("require N >= 1 and 1 <= P < R")
    if args.mesh < 1 or args.repeats < 1:
        raise ValueError("mesh and repeats must be positive")

    mode_count = 2 * (2 * args.N + 2)
    shells, prime_count = build_shells(
        n_value=args.N,
        lower_limit=args.P,
        upper_limit=args.R,
    )
    time_persistent(
        shells,
        mode_count=mode_count,
        mesh=1,
        eps=args.nufft_eps,
    )
    simple_samples = []
    persistent_samples = []
    for repeat in range(args.repeats):
        if repeat % 2 == 0:
            simple_samples.append(
                time_simple(
                    shells,
                    mode_count=mode_count,
                    mesh=args.mesh,
                    eps=args.nufft_eps,
                )
            )
            persistent_samples.append(
                time_persistent(
                    shells,
                    mode_count=mode_count,
                    mesh=args.mesh,
                    eps=args.nufft_eps,
                )
            )
        else:
            persistent_samples.append(
                time_persistent(
                    shells,
                    mode_count=mode_count,
                    mesh=args.mesh,
                    eps=args.nufft_eps,
                )
            )
            simple_samples.append(
                time_simple(
                    shells,
                    mode_count=mode_count,
                    mesh=args.mesh,
                    eps=args.nufft_eps,
                )
            )
    parity = parity_metrics(
        shells,
        mode_count=mode_count,
        mesh=args.mesh,
        eps=args.nufft_eps,
    )
    simple_median = statistics.median(simple_samples)
    persistent_median = statistics.median(persistent_samples)
    record = {
        "benchmark": "finufft_dyadic_prime_shell_transforms",
        "N": args.N,
        "P": args.P,
        "R": args.R,
        "band_count": len(shells),
        "prime_count": prime_count,
        "mode_count": mode_count,
        "mesh": args.mesh,
        "transforms": len(shells) * args.mesh,
        "nufft_eps": args.nufft_eps,
        "repeats": args.repeats,
        "simple_seconds_all": simple_samples,
        "simple_seconds_median": simple_median,
        "persistent_seconds_all": persistent_samples,
        "persistent_seconds_median": persistent_median,
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
