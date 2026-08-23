from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pytest

from fast_math import colex_rank, colex_unrank, colex_visit
from fast_math._native import (
    colex_rank_native,
    colex_unrank_native,
    colex_visit_native,
)


def direct_rank(subset: tuple[int, ...]) -> int:
    return sum(comb(c, i) for i, c in enumerate(subset, start=1))


def bitmask(subset: tuple[int, ...]) -> int:
    mask = 0
    for element in subset:
        mask |= 1 << element
    return mask


def all_subset_masks(n: int, k: int) -> np.ndarray:
    return np.array(
        [bitmask(subset) for subset in combinations(range(n), k)],
        dtype=np.uint64,
    )


@pytest.mark.parametrize("n", range(1, 10))
@pytest.mark.parametrize(
    "k", [0, 1, 3, 5], ids=lambda value: f"k{value}"
)
@pytest.mark.parametrize("backend", ["reference", "native"])
def test_colex_rank_unrank_round_trip_exhaustive(
    n: int, k: int, backend: str
) -> None:
    if k > n:
        return
    masks = all_subset_masks(n, k)
    ranks = colex_rank(masks, n, backend=backend)
    expected = np.array(
        [direct_rank(tuple(c)) for c in combinations(range(n), k)],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(ranks, expected)
    assert int(ranks.max(initial=np.uint64(0))) == comb(n, k) - 1

    recovered = colex_unrank(ranks, n, k, backend=backend)
    np.testing.assert_array_equal(recovered, masks)


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_colex_unrank_covers_every_rank_in_order(backend: str) -> None:
    n, k = 12, 5
    ranks = np.arange(comb(n, k), dtype=np.uint64)
    masks = colex_unrank(ranks, n, k, backend=backend)
    np.testing.assert_array_equal(
        colex_rank(masks, n, backend=backend),
        ranks,
    )
    # Every weight-k subset appears exactly once.
    assert len(np.unique(masks)) == comb(n, k)
    np.testing.assert_array_equal(
        np.sort(masks), np.sort(all_subset_masks(n, k))
    )


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_colex_order_is_colexicographical(backend: str) -> None:
    # Colex order on 2-subsets of {0..3}: ordered by the largest element,
    # then the next.
    expected = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
    masks = colex_unrank(
        np.arange(6, dtype=np.uint64), 4, 2, backend=backend
    )
    assert [tuple(int(b) for b in range(4) if mask >> b & 1) for mask in masks] == expected


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_colex_visit_marks_once_seeded_and_repeated(backend: str) -> None:
    n, k = 8, 3
    masks = all_subset_masks(n, k)
    visited = np.zeros((comb(n, k) + 63) // 64, dtype=np.uint64)

    # Orbit-marking shape: revisit a seeding prefix, then the full batch.
    first = colex_visit(masks[:7], n, k, visited, backend=backend)
    assert first.all()
    repeat = colex_visit(masks, n, k, visited, backend=backend)
    np.testing.assert_array_equal(repeat[:7], np.zeros(7, dtype=np.bool_))
    assert repeat[7:].all()
    # Every rank marked exactly once across both batches.
    total = np.bitwise_count(visited).sum()
    assert int(total) == comb(n, k)


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_colex_large_elements(backend: str) -> None:
    n = 64
    masks = np.array(
        [
            (np.uint64(1) << np.uint64(60))
            | (np.uint64(1) << np.uint64(61))
            | (np.uint64(1) << np.uint64(63)),
            (np.uint64(1) << np.uint64(63)) - np.uint64(1),
        ],
        dtype=np.uint64,
    )
    ranks = colex_rank(masks, n, backend=backend)
    assert ranks[0] == comb(60, 1) + comb(61, 2) + comb(63, 3)
    assert ranks[1] == 0
    np.testing.assert_array_equal(
        colex_unrank(ranks[:1], n, 3, backend=backend),
        np.array([masks[0]], dtype=np.uint64),
    )


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_colex_rejects_invalid_inputs(backend: str) -> None:
    masks = np.array([np.uint64(0b1011)], dtype=np.uint64)

    with pytest.raises(ValueError):
        colex_rank(masks, 3, backend=backend)
    with pytest.raises(ValueError):
        colex_rank(masks, 0, backend=backend)
    with pytest.raises(ValueError):
        colex_rank(masks, 65, backend=backend)
    with pytest.raises(ValueError):
        colex_unrank(
            np.array([comb(5, 2)], dtype=np.uint64), 5, 2, backend=backend
        )
    with pytest.raises(ValueError):
        colex_unrank(
            np.zeros(1, dtype=np.uint64), 5, 6, backend=backend
        )
    with pytest.raises(ValueError):
        colex_visit(
            masks,
            5,
            2,
            np.zeros(1, dtype=np.uint64),
            backend=backend,
        )
    with pytest.raises(ValueError):
        # 0b1011 has weight 3, not the declared 2.
        colex_visit(
            masks,
            8,
            2,
            np.zeros(1, dtype=np.uint64),
            backend=backend,
        )


def test_colex_visit_requires_owned_uint64_bitmap() -> None:
    masks = np.array([np.uint64(0b11)], dtype=np.uint64)
    with pytest.raises(ValueError):
        colex_visit(masks, 4, 2, np.zeros(1, dtype=np.int64))
    with pytest.raises(ValueError):
        colex_visit(
            masks, 4, 2, np.zeros((2, 3), dtype=np.uint64).T
        )


def test_native_wrappers_expose_stats() -> None:
    masks = np.array([np.uint64(0b11), np.uint64(0b101)], dtype=np.uint64)
    ranks, stats = colex_rank_native(masks, 4)
    assert stats.subset_count == 2
    np.testing.assert_array_equal(
        ranks[0], colex_rank(np.array([np.uint64(0b11)]), 4, backend="reference")[0]
    )
    subset_masks, _ = colex_unrank_native(ranks, 4, 2)
    np.testing.assert_array_equal(subset_masks, masks)

    visited = np.zeros(1, dtype=np.uint64)
    newly, stats = colex_visit_native(masks, 4, 2, visited)
    assert newly.tolist() == [True, True]
    assert stats.newly_visited == 2
    newly, stats = colex_visit_native(masks, 4, 2, visited)
    assert newly.tolist() == [False, False]
    assert stats.newly_visited == 0
