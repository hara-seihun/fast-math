from __future__ import annotations

from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lambda_fast._native import (
    NativeUnavailable,
    native_available,
    union_closed_family_masks_native,
)


UnionBackend = Literal["auto", "native", "reference"]


def _prepare_family_masks(
    family_masks: ArrayLike,
    ground_size: int,
) -> NDArray[np.uint64]:
    if not isinstance(ground_size, Integral) or not 0 <= ground_size <= 6:
        raise ValueError("ground_size must be an integer between zero and six")

    array = np.asarray(family_masks)
    if array.dtype.kind not in {"i", "u", "O"}:
        array = np.asarray(family_masks, dtype=object)
    if array.ndim != 1:
        raise ValueError("family_masks must be one-dimensional")
    if array.dtype.kind == "O":
        maximum = np.iinfo(np.uint64).max
        for value in array:
            if not isinstance(value, Integral):
                raise ValueError("family masks must be integers")
            if value < 0:
                raise ValueError("family masks must be nonnegative")
            if value > maximum:
                raise ValueError("family masks must fit in uint64")
    elif array.dtype.kind == "i" and np.any(array < 0):
        raise ValueError("family masks must be nonnegative")

    prepared = np.ascontiguousarray(array, dtype=np.uint64)
    set_count = 1 << int(ground_size)
    if set_count < 64 and np.any(prepared >> np.uint64(set_count)):
        raise ValueError("family mask contains a set outside the ground set")
    return prepared


def _union_closed_family_masks_reference(
    families: NDArray[np.uint64],
    ground_size: int,
) -> NDArray[np.bool_]:
    closed = np.ones(families.shape, dtype=np.bool_)
    set_count = 1 << ground_size

    for left in range(set_count):
        left_present = (families & np.uint64(1 << left)) != 0
        for right in range(left + 1, set_count):
            union = left | right
            if union == right:
                continue
            closed &= ~(
                left_present
                & ((families & np.uint64(1 << right)) != 0)
                & ((families & np.uint64(1 << union)) == 0)
            )
    return closed


def union_closed_family_masks(
    family_masks: ArrayLike,
    ground_size: int,
    *,
    backend: UnionBackend = "auto",
) -> NDArray[np.bool_]:
    """Check union closure for families packed into uint64 membership masks.

    Bit ``s`` of each input marks whether subset mask ``s`` belongs to the
    family. Ground sizes through six fit all subsets in one uint64 word.
    """
    families = _prepare_family_masks(family_masks, ground_size)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")

    if backend != "reference" and native_available():
        try:
            closed, _ = union_closed_family_masks_native(
                families,
                int(ground_size),
            )
            return closed
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    return _union_closed_family_masks_reference(
        families,
        int(ground_size),
    )
