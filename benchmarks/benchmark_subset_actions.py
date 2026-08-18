#!/usr/bin/env python3
"""Benchmark retained exact packed-subset actions on CPU and HIP."""

from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from fast_math import PermutationActionPlan
from fast_math.hip import hip_subset_actions_available


def cyclic_action(degree: int) -> np.ndarray:
    return np.asarray(
        [[(point + shift) % degree for point in range(degree)] for shift in range(degree)],
        dtype=np.uint32,
    )


def binary_cyclic_product_action() -> np.ndarray:
    """GL(3,2) x Aut(C9) on the 39 inverse atoms of F2^3 x C9."""
    vectors = list(product(range(2), repeat=3))
    vector_index = {value: index for index, value in enumerate(vectors)}
    matrices = []
    for entries in product(range(2), repeat=9):
        images = []
        for vector in vectors:
            image = tuple(
                sum(entries[3 * row + column] * vector[column] for column in range(3)) % 2
                for row in range(3)
            )
            images.append(vector_index[image])
        if len(set(images)) == 8:
            matrices.append(images)
    atoms = [(vector, kind) for vector in range(8) for kind in range(5) if vector or kind]
    index = {atom: position for position, atom in enumerate(atoms)}
    units = (1, 2, 4, 5, 7, 8)
    rows = set()
    for matrix in matrices:
        for unit in units:
            row = []
            for vector, kind in atoms:
                if kind == 0:
                    target_kind = 0
                else:
                    residue = unit * kind % 9
                    target_kind = min(residue, 9 - residue)
                row.append(index[(matrix[vector], target_kind)])
            rows.add(tuple(row))
    result = np.asarray(sorted(rows), dtype=np.uint32)
    assert result.shape == (504, 39)
    return result


def digest(values: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def measure(action: np.ndarray, masks: np.ndarray, threads: int, repeats: int) -> dict:
    with PermutationActionPlan(action) as plan:
        sample = masks[: min(2048, len(masks))]
        reference = plan.canonicalize(sample, backend="reference")
        native_sample = plan.canonicalize(sample, threads=threads, backend="native")
        np.testing.assert_array_equal(
            native_sample.canonical_masks, reference.canonical_masks
        )

        native_setup_started = perf_counter()
        native_first = plan.canonicalize(masks, threads=threads, backend="native")
        native_setup_wall = perf_counter() - native_setup_started
        native_walls = []
        native = native_first
        for _ in range(repeats):
            started = perf_counter()
            native = plan.canonicalize(masks, threads=threads, backend="native")
            native_walls.append(perf_counter() - started)

        native_flag_walls = []
        native_flags = native.is_canonical
        for _ in range(repeats):
            started = perf_counter()
            native_flags = plan.is_canonical(
                masks, threads=threads, backend="native"
            )
            native_flag_walls.append(perf_counter() - started)
        np.testing.assert_array_equal(native_flags, native.is_canonical)

        record = {
            "degree": action.shape[1],
            "permutations": len(action),
            "masks": len(masks),
            "native_setup_and_first_seconds": native_setup_wall,
            "native_seconds": native_walls,
            "native_median_seconds": median(native_walls),
            "native_canonical_test_seconds": native_flag_walls,
            "native_canonical_test_median_seconds": median(native_flag_walls),
            "canonical_sha256": digest(native.canonical_masks),
            "canonical_count": int(native.is_canonical.sum()),
            "sample_reference_parity": True,
        }
        if hip_subset_actions_available():
            hip_setup_started = perf_counter()
            hip_first = plan.canonicalize(masks, backend="hip")
            hip_setup_wall = perf_counter() - hip_setup_started
            np.testing.assert_array_equal(
                hip_first.canonical_masks, native.canonical_masks
            )
            hip_walls = []
            hip = hip_first
            for _ in range(repeats):
                started = perf_counter()
                hip = plan.canonicalize(masks, backend="hip")
                hip_walls.append(perf_counter() - started)
            np.testing.assert_array_equal(hip.canonical_masks, native.canonical_masks)
            np.testing.assert_array_equal(hip.is_canonical, native.is_canonical)
            hip_flag_walls = []
            hip_flags = hip.is_canonical
            for _ in range(repeats):
                started = perf_counter()
                hip_flags = plan.is_canonical(masks, backend="hip")
                hip_flag_walls.append(perf_counter() - started)
            np.testing.assert_array_equal(hip_flags, native.is_canonical)
            hip_median = median(hip_walls)
            hip_flag_median = median(hip_flag_walls)
            record.update(
                {
                    "hip_setup_and_first_seconds": hip_setup_wall,
                    "hip_seconds": hip_walls,
                    "hip_median_seconds": hip_median,
                    "hip_canonical_test_seconds": hip_flag_walls,
                    "hip_canonical_test_median_seconds": hip_flag_median,
                    "native_over_hip": median(native_walls) / hip_median,
                    "canonical_test_native_over_hip": (
                        median(native_flag_walls) / hip_flag_median
                    ),
                    "hip_exact_parity": True,
                }
            )
        return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cyclic-masks", type=int, default=1_000_000)
    parser.add_argument("--product-masks", type=int, default=262_144)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(0x51B5E7)
    cases = {
        "cyclic-41": (
            cyclic_action(41),
            rng.integers(0, 1 << 41, size=args.cyclic_masks, dtype=np.uint64),
        ),
        "product-action-39": (
            binary_cyclic_product_action(),
            rng.integers(0, 1 << 39, size=args.product_masks, dtype=np.uint64),
        ),
    }
    payload = {
        "schema": 1,
        "operation": "retained-packed-subset-action",
        "threads": args.threads,
        "repeats": args.repeats,
        "hip_subset_actions_available": hip_subset_actions_available(),
        "cases": {
            name: measure(action, masks, args.threads, args.repeats)
            for name, (action, masks) in cases.items()
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="ascii")


if __name__ == "__main__":
    main()
