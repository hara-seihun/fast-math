from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


def prime_u32(value: int) -> int:
    if not isinstance(value, Integral) or not 2 <= int(value) <= 0xFFFF_FFFF:
        raise ValueError("prime must be a uint32 prime")
    prime = int(value)
    if prime % 2 == 0:
        if prime != 2:
            raise ValueError("prime must be prime")
        return prime
    divisor = 3
    while divisor <= prime // divisor:
        if prime % divisor == 0:
            raise ValueError("prime must be prime")
        divisor += 2
    return prime


def field_array_u32(
    values: ArrayLike,
    prime: int,
    *,
    dimensions: int,
    own: bool,
) -> NDArray[np.uint32]:
    raw = np.asarray(values)
    if raw.ndim != dimensions or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(
            f"field values must be a {dimensions}-dimensional integer array"
        )
    if np.issubdtype(raw.dtype, np.signedinteger) and np.any(raw < 0):
        raise ValueError("field values must be nonnegative")
    if raw.size and np.any(raw >= prime):
        raise ValueError("field value is outside the prime field")
    if own:
        result = np.array(raw, dtype=np.uint32, order="C", copy=True)
        result.flags.writeable = False
        return result
    return np.ascontiguousarray(raw, dtype=np.uint32)
