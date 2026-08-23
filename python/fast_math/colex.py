from __future__ import annotations

from math import comb
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._native import (
    NativeUnavailable,
    colex_rank_native,
    colex_unrank_native,
    colex_visit_native,
    native_available,
)


ColexBackend = Literal["auto", "native", "reference"]


def _validate_element_count(element_count: int) -> int:
    if not isinstance(element_count, Integral):
        raise ValueError("element_count must be an integer")
    if not 1 <= int(element_count) <= 64:
        raise ValueError("element_count must be between 1 and 64")
    return int(element_count)


def _validate_weight(weight: int, element_count: int) -> int:
    if not isinstance(weight, Integral):
        raise ValueError("weight must be an integer")
    if not 0 <= int(weight) <= element_count:
        raise ValueError("weight must be between zero and element_count")
    return int(weight)


def _prepare_subset_masks(subset_masks: ArrayLike) -> NDArray[np.uint64]:
    array = np.asarray(subset_masks)
    if array.ndim != 1:
        raise ValueError("subset_masks must be one-dimensional")
    if array.dtype.kind == "O":
        for value in array:
            if not isinstance(value, Integral):
                raise ValueError("subset masks must be integers")
            if value < 0:
                raise ValueError("subset masks must be nonnegative")
            if value > np.iinfo(np.uint64).max:
                raise ValueError("subset masks must fit in uint64")
    elif array.dtype.kind == "i":
        if np.any(array < 0):
            raise ValueError("subset masks must be nonnegative")
    elif array.dtype.kind != "u":
        raise ValueError("subset masks must be integers")
    return np.ascontiguousarray(array, dtype=np.uint64)


def _prepare_ranks(ranks: ArrayLike) -> NDArray[np.uint64]:
    array = np.asarray(ranks)
    if array.ndim != 1:
        raise ValueError("ranks must be one-dimensional")
    if array.dtype.kind == "O":
        for value in array:
            if not isinstance(value, Integral):
                raise ValueError("ranks must be integers")
            if value < 0:
                raise ValueError("ranks must be nonnegative")
    elif array.dtype.kind == "i":
        if np.any(array < 0):
            raise ValueError("ranks must be nonnegative")
    elif array.dtype.kind != "u":
        raise ValueError("ranks must be integers")
    return np.ascontiguousarray(array, dtype=np.uint64)


def _check_mask_range(
    masks: NDArray[np.uint64],
    element_count: int,
) -> None:
    if element_count < 64 and np.any(masks >> np.uint64(element_count)):
        raise ValueError(
            "subset mask contains an element outside the element range"
        )


def colex_rank(
    subset_masks: ArrayLike,
    element_count: int,
    *,
    backend: ColexBackend = "auto",
) -> NDArray[np.uint64]:
    """Rank fixed-size subsets in colexicographical order.

    Each subset of ``{0, ..., element_count - 1}`` is a uint64 bit mask. The
    rank of ``{c_1 < ... < c_k}`` is ``C(c_1, 1) + ... + C(c_k, k)``, the
    position of the subset among all subsets of the same weight in colex
    order. Ranks therefore collide across different weights; use
    ``colex_visit`` with a declared weight when marking a shared bitmap.
    """
    n = _validate_element_count(element_count)
    masks = _prepare_subset_masks(subset_masks)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    _check_mask_range(masks, n)

    if backend != "reference" and native_available():
        try:
            ranks, _ = colex_rank_native(masks, n)
            return ranks
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    return _colex_rank_reference(masks)


def colex_unrank(
    ranks: ArrayLike,
    element_count: int,
    weight: int,
    *,
    backend: ColexBackend = "auto",
) -> NDArray[np.uint64]:
    """Invert :func:`colex_rank` for declared ``weight`` subsets.

    Every rank must be below ``C(element_count, weight)``; anything larger
    is an error, not a wrap.
    """
    n = _validate_element_count(element_count)
    k = _validate_weight(weight, n)
    prepared = _prepare_ranks(ranks)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    total = comb(n, k)
    if np.any(prepared >= np.uint64(total)):
        raise ValueError("colex rank is outside the valid range for the weight")

    if backend != "reference" and native_available():
        try:
            subset_masks, _ = colex_unrank_native(prepared, n, k)
            return subset_masks
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    return _colex_unrank_reference(prepared, n, k)


def colex_visit(
    subset_masks: ArrayLike,
    element_count: int,
    weight: int,
    visited_words: NDArray[np.uint64],
    *,
    backend: ColexBackend = "auto",
) -> NDArray[np.bool_]:
    """Rank fixed-weight subsets and mark their ranks in a visited bitmap.

    ``visited_words`` is caller-owned and updated in place; it must hold at
    least ``C(element_count, weight)`` bits. Returns one flag per subset
    that is ``True`` when its rank was not already marked. Every subset must
    have exactly ``weight`` elements, since ranks are only unique within a
    weight class.
    """
    n = _validate_element_count(element_count)
    k = _validate_weight(weight, n)
    masks = _prepare_subset_masks(subset_masks)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    _check_mask_range(masks, n)
    total = comb(n, k)
    if not isinstance(visited_words, np.ndarray) or (
        visited_words.dtype != np.uint64
    ):
        raise ValueError("visited_words must be a uint64 ndarray")
    if not visited_words.flags.c_contiguous:
        raise ValueError("visited_words must be C-contiguous")
    if visited_words.size * 64 < total:
        raise ValueError("visited bitmap is too small for the weight class")
    weights = np.bitwise_count(masks).astype(np.uint32)
    if np.any(weights != np.uint32(k)):
        raise ValueError("subset mask does not have the declared weight")

    if backend != "reference" and native_available():
        try:
            newly_visited, _ = colex_visit_native(masks, n, k, visited_words)
            return newly_visited
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    return _colex_visit_reference(masks, visited_words)


def _colex_rank_reference(masks: NDArray[np.uint64]) -> NDArray[np.uint64]:
    ranks = np.empty(len(masks), dtype=np.uint64)
    for index, value in enumerate(masks):
        mask = int(value)
        rank = 0
        position = 0
        while mask:
            bit = mask & -mask
            element = bit.bit_length() - 1
            mask ^= bit
            position += 1
            rank += comb(element, position)
        ranks[index] = np.uint64(rank)
    return ranks


def _colex_unrank_reference(
    ranks: NDArray[np.uint64],
    element_count: int,
    weight: int,
) -> NDArray[np.uint64]:
    masks = np.empty(len(ranks), dtype=np.uint64)
    for index, value in enumerate(ranks):
        rank = int(value)
        mask = 0
        remaining = weight
        for bit in range(element_count - 1, -1, -1):
            if remaining == 0:
                break
            below = comb(bit, remaining)
            if rank >= below:
                rank -= below
                mask |= 1 << bit
                remaining -= 1
        masks[index] = np.uint64(mask)
    return masks


def _colex_visit_reference(
    masks: NDArray[np.uint64],
    visited_words: NDArray[np.uint64],
) -> NDArray[np.bool_]:
    ranks = _colex_rank_reference(masks)
    word_index = ranks >> np.uint64(6)
    bit = np.uint64(1) << (ranks & np.uint64(63))
    newly_visited = np.empty(len(masks), dtype=np.bool_)
    for index in range(len(masks)):
        word = int(word_index[index])
        newly_visited[index] = bool(
            (int(visited_words[word]) & int(bit[index])) == 0
        )
        visited_words[word] = np.uint64(
            int(visited_words[word]) | int(bit[index])
        )
    return newly_visited
