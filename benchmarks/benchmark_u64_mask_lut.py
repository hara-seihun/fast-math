#!/usr/bin/env python3
"""Benchmark exact small packed-mask permutation lookup tables."""

from __future__ import annotations

import json
from time import perf_counter

import numpy as np

from fast_math.ci import u64_mask_lut


def slow_apply(mask: int, permutation: np.ndarray) -> int:
    result = 0
    active = mask
    while active:
        bit = (active & -active).bit_length() - 1
        result |= 1 << int(permutation[bit])
        active &= active - 1
    return result


def main() -> None:
    rng = np.random.default_rng(20260815)
    degree = 11
    masks = range(1 << degree)
    permutations = [rng.permutation(degree) for _ in range(64)]
    lookup_tables = [u64_mask_lut(permutation) for permutation in permutations]

    # Warm both paths and keep table construction outside the hot-loop timing.
    expected = [
        slow_apply(mask, permutation)
        for permutation in permutations
        for mask in masks
    ]
    actual = [lookup[mask] for lookup in lookup_tables for mask in masks]
    if expected != actual:
        raise AssertionError("packed-mask lookup disagrees with bit-walk reference")

    started = perf_counter()
    expected = [
        slow_apply(mask, permutation)
        for permutation in permutations
        for mask in masks
    ]
    slow_seconds = perf_counter() - started
    started = perf_counter()
    actual = [lookup[mask] for lookup in lookup_tables for mask in masks]
    lookup_seconds = perf_counter() - started
    if expected != actual:
        raise AssertionError("packed-mask lookup disagrees after timing")
    result = {
        "benchmark": "u64_mask_lut",
        "degree": degree,
        "permutation_count": len(permutations),
        "masks_permutation": 1 << degree,
        "operations": len(expected),
        "slow_seconds": slow_seconds,
        "lookup_apply_seconds": lookup_seconds,
        "speedup": slow_seconds / lookup_seconds,
        "exact_match": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
