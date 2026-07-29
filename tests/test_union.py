from __future__ import annotations

import numpy as np
import pytest

from fast_math import union_closed_family_masks
from lambda_fast._native import union_closed_family_masks_native


def direct_union_closed(family_mask: int, ground_size: int) -> bool:
    family = tuple(
        member
        for member in range(1 << ground_size)
        if family_mask & (1 << member)
    )
    members = set(family)
    return all(
        (left | right) in members
        for left in family
        for right in family
    )


@pytest.mark.parametrize("ground_size", range(5))
@pytest.mark.parametrize("backend", ["reference", "native"])
def test_union_closed_masks_match_exhaustive_direct_check(
    ground_size: int,
    backend: str,
) -> None:
    family_masks = np.arange(
        1 << (1 << ground_size),
        dtype=np.uint64,
    )
    expected = np.fromiter(
        (
            direct_union_closed(int(family_mask), ground_size)
            for family_mask in family_masks
        ),
        dtype=np.bool_,
        count=len(family_masks),
    )
    np.testing.assert_array_equal(
        union_closed_family_masks(
            family_masks,
            ground_size,
            backend=backend,
        ),
        expected,
    )


@pytest.mark.parametrize("ground_size", [5, 6])
@pytest.mark.parametrize("backend", ["reference", "native"])
def test_union_closed_masks_match_sampled_direct_check(
    ground_size: int,
    backend: str,
) -> None:
    rng = np.random.default_rng(7800 + ground_size)
    family_masks = rng.integers(
        0,
        np.iinfo(np.uint64).max,
        size=257,
        dtype=np.uint64,
    )
    if ground_size == 5:
        family_masks &= np.uint64((1 << 32) - 1)
    expected = np.fromiter(
        (
            direct_union_closed(int(family_mask), ground_size)
            for family_mask in family_masks
        ),
        dtype=np.bool_,
        count=len(family_masks),
    )
    np.testing.assert_array_equal(
        union_closed_family_masks(
            family_masks,
            ground_size,
            backend=backend,
        ),
        expected,
    )


def test_union_closed_masks_cover_hostile_fixtures() -> None:
    powerset = (1 << (1 << 3)) - 1
    chain = sum(1 << member for member in (0, 1, 3, 7))
    missing_union = sum(1 << member for member in (1, 2))
    flags = union_closed_family_masks(
        [0, 1, powerset, chain, missing_union],
        3,
    )
    np.testing.assert_array_equal(flags, [True, True, True, True, False])


def test_union_closed_masks_preserve_input() -> None:
    family_masks = np.array([0, 1, 3, 7, 11], dtype=np.uint64)
    original = family_masks.copy()
    union_closed_family_masks(family_masks, 2, backend="native")
    np.testing.assert_array_equal(family_masks, original)


@pytest.mark.parametrize(
    ("family_masks", "ground_size", "message"),
    [
        ([[1]], 1, "one-dimensional"),
        ([1.5], 1, "integers"),
        ([-1], 1, "nonnegative"),
        ([1 << 4], 1, "outside"),
        ([1], -1, "between zero and six"),
        ([1], 7, "between zero and six"),
        ([1], 1.5, "integer"),
    ],
)
def test_union_closed_masks_reject_invalid_inputs(
    family_masks,
    ground_size,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        union_closed_family_masks(family_masks, ground_size)


def test_union_closed_masks_reject_invalid_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        union_closed_family_masks([1], 1, backend="other")


@pytest.mark.parametrize(
    ("family_masks", "ground_size", "message"),
    [
        (np.array([1 << 4], dtype=np.uint64), 1, "outside"),
        (np.array([1], dtype=np.uint64), 7, "at most six"),
    ],
)
def test_union_closed_native_rejects_invalid_inputs(
    family_masks: np.ndarray,
    ground_size: int,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        union_closed_family_masks_native(family_masks, ground_size)
