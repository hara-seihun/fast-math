#!/usr/bin/env python3
"""Benchmark exact fixed-weight subset orbit enumeration end to end."""

from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from fast_math.ci import enumerate_fixed_weight_subset_orbits


def cyclic_action(degree: int) -> np.ndarray:
    return np.asarray(
        [
            [(point + shift) % degree for point in range(degree)]
            for shift in range(degree)
        ],
        dtype=np.uint32,
    )


def gl33_vector_action() -> np.ndarray:
    """Complete GL(3,3) action on the 27 vectors of F_3^3."""
    vectors = list(product(range(3), repeat=3))
    rows = []
    for matrix in product(range(3), repeat=9):
        determinant = (
            matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
            - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
            + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
        ) % 3
        if determinant == 0:
            continue
        rows.append(
            [
                vectors.index(
                    tuple(
                        sum(matrix[3 * i + j] * vector[j] for j in range(3)) % 3
                        for i in range(3)
                    )
                )
                for vector in vectors
            ]
        )
    action = np.asarray(rows, dtype=np.uint32)
    assert action.shape == (11_232, 27)
    return action


def digest(values: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def timed(action: np.ndarray, weight: int, backend: str, repeats: int):
    walls = []
    result = None
    for _ in range(repeats):
        started = perf_counter()
        result = enumerate_fixed_weight_subset_orbits(
            action,
            weight,
            max_subsets=100_000_000,
            backend=backend,
        )
        walls.append(perf_counter() - started)
    assert result is not None
    return result, walls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    medium_action = cyclic_action(20)
    reference, reference_walls = timed(medium_action, 6, "reference", args.repeats)
    native, native_walls = timed(medium_action, 6, "native", args.repeats)
    np.testing.assert_array_equal(native.representatives, reference.representatives)
    np.testing.assert_array_equal(native.orbit_sizes, reference.orbit_sizes)

    target_action = cyclic_action(41)
    target, target_walls = timed(target_action, 8, "native", args.repeats)

    # This is the complete action and end-to-end validation shape used by the
    # C4 x F3^3 ordinary undirected CI route. Weight zero isolates validation:
    # the native orbit kernel itself has one input and one output.
    large_action = gl33_vector_action()
    large, large_walls = timed(large_action, 0, "native", args.repeats)
    record = {
        "schema": 1,
        "operation": "fixed_weight_subset_orbits",
        "medium": {
            "degree": 20,
            "weight": 6,
            "action_size": 20,
            "domain_subsets": native.subset_count,
            "orbits": len(native.representatives),
            "reference_seconds": reference_walls,
            "native_seconds": native_walls,
            "best_speedup": min(reference_walls) / min(native_walls),
            "representatives_sha256": digest(native.representatives),
            "orbit_sizes_sha256": digest(native.orbit_sizes),
            "exact_reference_parity": True,
        },
        "target": {
            "degree": 41,
            "weight": 8,
            "action_size": 41,
            "domain_subsets": target.subset_count,
            "orbits": len(target.representatives),
            "native_seconds": target_walls,
            "representatives_sha256": digest(target.representatives),
            "orbit_sizes_sha256": digest(target.orbit_sizes),
        },
        "large_complete_action": {
            "degree": 27,
            "weight": 0,
            "action_size": len(large_action),
            "domain_subsets": large.subset_count,
            "orbits": len(large.representatives),
            "native_seconds": large_walls,
            "representatives_sha256": digest(large.representatives),
            "orbit_sizes_sha256": digest(large.orbit_sizes),
        },
    }
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="ascii")
    print(text, end="")


if __name__ == "__main__":
    main()
