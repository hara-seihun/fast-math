#!/usr/bin/env python3
"""Benchmark the base-p codec against the hand-written loops it replaces.

Fleet scratch routes over encoded F_p^n points (for example
``math/a3_feistel_proper_search.py``) build a digit table, a negation-code
table, dense negation-class ids, and projective point representatives with
plain Python loops before any census starts. This script times that exact
baseline workspace construction against the equivalent fast-math calls on
the shapes those routes actually use.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

import fast_math


def baseline_workspace(prime: int, dim: int) -> dict[str, object]:
    """Transcribed scratch-route construction (the program replaced)."""
    n = prime**dim
    weights = (prime ** np.arange(dim)).astype(np.int64)

    # Digit table, exactly the double loop the scratch files write.
    vec = np.empty((n, dim), dtype=np.int64)
    for z in range(n):
        x = z
        for j in range(dim):
            vec[z, j] = x % prime
            x //= prime

    # Negation codes through the NumPy dot trick the routes use.
    negation = (((-vec) % prime) @ weights).astype(np.int64)

    # Dense negation-class ids and representatives, first-come loop.
    atom = np.full(n, -1, dtype=np.int64)
    reps: list[int] = []
    for z in range(1, n):
        r = min(z, int(negation[z]))
        if atom[r] < 0:
            atom[r] = len(reps)
            atom[int(negation[z])] = len(reps)
            reps.append(r)

    # Projective point representatives by tuple-set deduplication.
    points: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for x in vec[1:]:
        j = int(np.flatnonzero(x)[0])
        y = (x * pow(int(x[j]), -1, prime)) % prime
        key = tuple(int(v) for v in y)
        if key not in seen:
            seen.add(key)
            points.append(y)

    return {
        "digits": vec,
        "negation": negation,
        "atom": atom,
        "reps": reps,
        "points": points,
    }


def fastmath_workspace(prime: int, dim: int) -> dict[str, object]:
    """The same workspace through the shipped kernels."""
    n = prime**dim
    codes = np.arange(n, dtype=np.uint64)
    digits = fast_math.base_p_digits(codes, prime, dim)
    negation = fast_math.base_p_negation_codes(codes, prime, dim)
    neg_classes = fast_math.base_p_class_table(
        prime, dim, classes="negation"
    )
    scalars = fast_math.base_p_scalar_normals(codes, prime, dim)
    scalar_classes = fast_math.base_p_class_table(
        prime, dim, classes="scalar"
    )
    return {
        "codes": codes,
        "digits": digits,
        "negation": negation,
        "neg_classes": neg_classes,
        "scalars": scalars,
        "scalar_classes": scalar_classes,
    }


def assert_workspaces_agree(prime: int, dim: int) -> None:
    baseline = baseline_workspace(prime, dim)
    native = fastmath_workspace(prime, dim)
    np.testing.assert_array_equal(
        baseline["digits"], native["digits"].astype(np.int64)
    )
    np.testing.assert_array_equal(
        baseline["negation"], native["negation"].astype(np.int64)
    )
    # Baseline skips the zero vector, so its class ids are shifted by one;
    # both assign ids in ascending representative order.
    np.testing.assert_array_equal(
        native["neg_classes"].class_ids[1:].astype(np.int64),
        baseline["atom"][1:] + 1,
    )
    np.testing.assert_array_equal(
        native["neg_classes"].representatives[1:],
        np.array(sorted(baseline["reps"]), dtype=np.uint64),
    )
    baseline_points = sorted(tuple(row) for row in baseline["points"])
    kernel_points = sorted(
        tuple(int(d) for d in row)
        for row in fast_math.base_p_digits(
            native["scalar_classes"].representatives[1:], prime, dim
        )
    )
    assert baseline_points == kernel_points
    # Scalar normals agree with brute force over the whole space here.
    normals = native["scalars"]
    for code in range(prime**dim):
        digits = [(code // prime**j) % prime for j in range(dim)]
        lead = next((j for j, d in enumerate(digits) if d), None)
        if lead is None:
            expected = 0
        else:
            scale = pow(digits[lead], -1, prime)
            expected = sum(
                d * scale % prime * prime**j for j, d in enumerate(digits)
            )
        assert int(normals[code]) == expected


def measure(function, repeats: int) -> list[float]:
    function()
    walls = []
    for _ in range(repeats):
        started = perf_counter()
        function()
        walls.append(perf_counter() - started)
    return walls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fixtures = [
        ("f3_dim6_feistel", 3, 6),
        ("f5_dim6_henon_depth2", 5, 6),
        ("f7_dim6_projective_wl", 7, 6),
        ("f5_dim8_stress", 5, 8),
    ]
    for _, prime, dim in fixtures:
        assert_workspaces_agree(prime, dim)

    results: dict[str, object] = {
        "machine": platform.platform(),
        "cpu_count": os.cpu_count(),
        "repeats": args.repeats,
        "routes_verified_identical": True,
        "fixtures": {},
    }
    for name, prime, dim in fixtures:
        baseline_walls = measure(
            lambda: baseline_workspace(prime, dim), args.repeats
        )
        native_walls = measure(
            lambda: fastmath_workspace(prime, dim), args.repeats
        )
        baseline_median = median(baseline_walls)
        native_median = median(native_walls)
        results["fixtures"][name] = {
            "prime": prime,
            "width": dim,
            "space_size": prime**dim,
            "baseline_wall_seconds": baseline_walls,
            "fastmath_wall_seconds": native_walls,
            "baseline_median_seconds": baseline_median,
            "fastmath_median_seconds": native_median,
            "speedup": baseline_median / native_median,
        }
        print(
            f"{name}: baseline {baseline_median:.6f}s, "
            f"fast-math {native_median:.6f}s, "
            f"speedup {baseline_median / native_median:.2f}x",
            flush=True,
        )

    text = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
