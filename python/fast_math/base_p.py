"""Batched base-p digit codec with canonical class representatives over F_p.

Census loops over ``F_p^n`` tuples repeatedly re-implement the same pieces:
mixed-radix encode/decode between an index and its digit tuple, the canonical
minimum of a negation pair ``{v, -v}``, and dense labels for scalar-multiplication
classes (projective points). This module owns those pieces behind one validated,
batch-first surface with reference and native backends.

Conventions:

- Digit ``i`` is the coefficient of ``p**i`` (little-endian); row-major
  ``(element_count, width)`` arrays hold one tuple per element.
- The negation representative of index ``z`` is ``min(z, enc(-v))``, where
  ``-v`` negates every nonzero digit modulo ``p``.
- Scalar classes are orbits of ``F_p^*`` on tuples by coordinatewise scalar
  multiplication. The zero tuple is its own class. A class id is the rank of
  the class's minimal mixed-radix representative among distinct
  representatives, so ids are dense from zero and identical across backends,
  thread counts, and input orderings. ``np.unique(result.representatives)``
  recovers the representative table in id order.
- The canonical scalar-class representative has a closed form: scaling never
  changes which digits are zero, so all multiples share the highest nonzero
  position ``t``, and exactly one multiple carries digit one there; that
  multiple minimizes the encoding. Equivalently, scale the tuple by the
  modular inverse of its highest nonzero digit. The reference backend checks
  the same representative against the min-over-multiples definition.
"""

from __future__ import annotations

from numbers import Integral
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._native import (
    NativeUnavailable,
    base_p_decode_native,
    base_p_encode_native,
    base_p_negation_representatives_native,
    base_p_scalar_classes_native,
    native_available,
)

__all__ = [
    "MAX_BASE_P_PRIME",
    "MAX_BASE_P_WIDTH",
    "BasePBackend",
    "BasePScalarClasses",
    "base_p_decode",
    "base_p_encode",
    "base_p_negation_representatives",
    "base_p_scalar_classes",
]

MAX_BASE_P_PRIME = 251
MAX_BASE_P_WIDTH = 16

BasePBackend = Literal["auto", "native", "reference"]


class BasePScalarClasses(NamedTuple):
    """Dense scalar-class ids and per-element canonical representatives."""

    class_ids: NDArray[np.uint32]
    representatives: NDArray[np.uint64]


def _is_small_prime(value: int) -> bool:
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


def _prepare_prime(prime: ArrayLike) -> int:
    if not isinstance(prime, Integral):
        raise ValueError("prime must be an integer")
    prime_int = int(prime)
    if not 2 <= prime_int <= MAX_BASE_P_PRIME:
        raise ValueError(
            f"prime must be between two and {MAX_BASE_P_PRIME}"
        )
    if not _is_small_prime(prime_int):
        raise ValueError("prime must be prime")
    return prime_int


def _prepare_width(width: ArrayLike) -> int:
    if not isinstance(width, Integral):
        raise ValueError("width must be an integer")
    width_int = int(width)
    if not 1 <= width_int <= MAX_BASE_P_WIDTH:
        raise ValueError(
            f"width must be between one and {MAX_BASE_P_WIDTH}"
        )
    return width_int


def _index_space_size(prime: int, width: int) -> int:
    space = prime**width
    maximum = np.iinfo(np.uint64).max
    if space > maximum:
        raise ValueError("base-p index space p^width exceeds uint64 range")
    return space


def _prepare_indices(
    indices: ArrayLike,
    space: int,
) -> NDArray[np.uint64]:
    array = np.asarray(indices)
    if array.dtype.kind not in {"i", "u", "O"}:
        array = np.asarray(indices, dtype=object)
    if array.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if array.dtype.kind == "O":
        for value in array:
            if not isinstance(value, Integral):
                raise ValueError("indices must be integers")
            if value < 0:
                raise ValueError("indices must be nonnegative")
            if value >= space:
                raise ValueError(
                    "index is outside the base-p index space p^width"
                )
    elif array.dtype.kind == "i" and np.any(array < 0):
        raise ValueError("indices must be nonnegative")

    prepared = np.ascontiguousarray(array, dtype=np.uint64)
    if prepared.size and np.any(prepared >= np.uint64(space)):
        raise ValueError("index is outside the base-p index space p^width")
    return prepared


def _prepare_digit_rows(
    digits: ArrayLike,
    prime: int,
) -> tuple[NDArray[np.uint32], int]:
    array = np.asarray(digits)
    if array.dtype.kind not in {"i", "u", "O"}:
        array = np.asarray(digits, dtype=object)
    if array.ndim != 2:
        raise ValueError("digits must be a two-dimensional batch")
    if array.dtype.kind == "O":
        for value in array.ravel():
            if not isinstance(value, Integral):
                raise ValueError("digits must be integers")
            if value < 0 or value >= prime:
                raise ValueError("digit is outside the field F_p")
    else:
        # Bound-check before narrowing conversion so oversized signed or
        # unsigned values fail instead of wrapping modulo 2**32.
        if array.size:
            minimum = int(array.min())
            maximum = int(array.max())
            if minimum < 0 or maximum >= prime:
                raise ValueError("digit is outside the field F_p")

    prepared = np.ascontiguousarray(array, dtype=np.uint32)
    return prepared, prepared.shape[1]


def _encode_reference(
    digit_rows: NDArray[np.uint32],
    prime: int,
) -> NDArray[np.uint64]:
    count, width = digit_rows.shape
    indices = np.empty(count, dtype=np.uint64)
    for element in range(count):
        index = 0
        for dimension in range(width - 1, -1, -1):
            index = index * prime + int(digit_rows[element, dimension])
        indices[element] = index
    return indices


def _decode_reference(
    indices: NDArray[np.uint64],
    prime: int,
    width: int,
) -> NDArray[np.uint32]:
    digits = np.empty((len(indices), width), dtype=np.uint32)
    for element, value in enumerate(indices):
        remaining = int(value)
        for dimension in range(width):
            digits[element, dimension] = remaining % prime
            remaining //= prime
    return digits


def _negation_representatives_reference(
    indices: NDArray[np.uint64],
    prime: int,
    width: int,
) -> NDArray[np.uint64]:
    representatives = np.empty(len(indices), dtype=np.uint64)
    powers = [prime**dimension for dimension in range(width)]
    for element, value in enumerate(indices):
        remaining = int(value)
        negated = 0
        for dimension in range(width):
            digit = remaining % prime
            remaining //= prime
            if digit:
                negated += (prime - digit) * powers[dimension]
        representatives[element] = min(negated, int(value))
    return representatives


def _scalar_representatives_reference(
    indices: NDArray[np.uint64],
    prime: int,
    width: int,
) -> NDArray[np.uint64]:
    """Minimum mixed-radix index over each orbit of F_p* scalar multiples."""
    representatives = np.empty(len(indices), dtype=np.uint64)
    powers = [prime**dimension for dimension in range(width)]
    for element, value in enumerate(indices):
        remaining = int(value)
        digits = []
        for _ in range(width):
            digits.append(remaining % prime)
            remaining //= prime
        if not any(digits):
            representatives[element] = 0
            continue
        best = None
        for factor in range(1, prime):
            candidate = 0
            for dimension in range(width - 1, -1, -1):
                candidate = (
                    candidate * prime + digits[dimension] * factor % prime
                )
            if best is None or candidate < best:
                best = candidate
        representatives[element] = best
    return representatives


def _scalar_classes_reference(
    indices: NDArray[np.uint64],
    prime: int,
    width: int,
) -> BasePScalarClasses:
    representatives = _scalar_representatives_reference(
        indices,
        prime,
        width,
    )
    distinct = sorted(set(int(value) for value in representatives))
    ranks = {value: rank for rank, value in enumerate(distinct)}
    class_ids = np.array(
        [ranks[int(value)] for value in representatives],
        dtype=np.uint32,
    )
    return BasePScalarClasses(class_ids, representatives)


def _prepare_threads(threads: ArrayLike) -> int:
    if not isinstance(threads, Integral):
        raise ValueError("threads must be an integer")
    threads_int = int(threads)
    if threads_int < 0:
        raise ValueError("threads must be nonnegative")
    return threads_int


def _dispatch_or_reference(backend: str) -> bool:
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    if backend != "reference":
        if native_available():
            return True
        if backend == "native":
            raise NativeUnavailable("fast-math native library is unavailable")
    return False


def base_p_encode(
    digits: ArrayLike,
    prime: ArrayLike,
    *,
    backend: BasePBackend = "auto",
    threads: int = 0,
) -> NDArray[np.uint64]:
    """Encode little-endian base-p digit rows into mixed-radix uint64 indices."""
    prime_int = _prepare_prime(prime)
    rows, width = _prepare_digit_rows(digits, prime_int)
    threads_int = _prepare_threads(threads)
    if _dispatch_or_reference(backend):
        try:
            encoded, _ = base_p_encode_native(rows, prime_int, threads=threads)
            return encoded
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    return _encode_reference(rows, prime_int)


def base_p_decode(
    indices: ArrayLike,
    prime: ArrayLike,
    width: ArrayLike,
    *,
    backend: BasePBackend = "auto",
    threads: int = 0,
) -> NDArray[np.uint32]:
    """Decode uint64 indices into little-endian base-p digit rows."""
    prime_int = _prepare_prime(prime)
    width_int = _prepare_width(width)
    space = _index_space_size(prime_int, width_int)
    prepared = _prepare_indices(indices, space)
    threads_int = _prepare_threads(threads)
    if _dispatch_or_reference(backend):
        try:
            decoded, _ = base_p_decode_native(
                prepared,
                prime_int,
                width_int,
                threads=threads,
            )
            return decoded
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    return _decode_reference(prepared, prime_int, width_int)


def base_p_negation_representatives(
    indices: ArrayLike,
    prime: ArrayLike,
    width: ArrayLike,
    *,
    backend: BasePBackend = "auto",
    threads: int = 0,
) -> NDArray[np.uint64]:
    """Canonical minimum index of every vector-negation pair ``{v, -v}``."""
    prime_int = _prepare_prime(prime)
    width_int = _prepare_width(width)
    space = _index_space_size(prime_int, width_int)
    prepared = _prepare_indices(indices, space)
    threads_int = _prepare_threads(threads)
    if _dispatch_or_reference(backend):
        try:
            representatives, _ = base_p_negation_representatives_native(
                prepared,
                prime_int,
                width_int,
                threads=threads,
            )
            return representatives
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    return _negation_representatives_reference(prepared, prime_int, width_int)


def base_p_scalar_classes(
    indices: ArrayLike,
    prime: ArrayLike,
    width: ArrayLike,
    *,
    backend: BasePBackend = "auto",
    threads: int = 0,
) -> BasePScalarClasses:
    """Dense scalar-multiplication class ids and canonical representatives."""
    prime_int = _prepare_prime(prime)
    width_int = _prepare_width(width)
    space = _index_space_size(prime_int, width_int)
    prepared = _prepare_indices(indices, space)
    threads_int = _prepare_threads(threads)
    if _dispatch_or_reference(backend):
        try:
            class_ids, representatives, _ = base_p_scalar_classes_native(
                prepared,
                prime_int,
                width_int,
                threads=threads,
            )
            return BasePScalarClasses(class_ids, representatives)
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    return _scalar_classes_reference(prepared, prime_int, width_int)
