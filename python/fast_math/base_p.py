"""Batched base-p codec and scalar/negation class tables for encoded F_p^n.

Research routes hold points of ``F_p^n`` as plain base-p integers and keep
rewriting the same loops: index/digit conversion, digit-wise negation,
projective normalization, and dense class ids over scalar or sign pairs.
This module owns those contracts for primes through 251 and widths through
sixteen, with executable NumPy reference backends.
"""

from __future__ import annotations

import dataclasses
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._field import prime_u32
from ._native import (
    NativeUnavailable,
    base_p_class_table_native,
    base_p_codes_native,
    base_p_digits_native,
    base_p_negation_codes_native,
    base_p_scalar_normals_native,
    native_available,
)

__all__ = [
    "BasePClassTable",
    "BasePClasses",
    "BasePBackend",
    "base_p_class_table",
    "base_p_codes",
    "base_p_digits",
    "base_p_negation_codes",
    "base_p_scalar_normals",
]

BasePClasses = Literal["negation", "scalar"]
BasePBackend = Literal["auto", "native", "reference"]

_MAX_PRIME = 251
_MAX_WIDTH = 16


def _validate_prime_width(prime: object, width: object) -> tuple[int, int]:
    checked_prime = prime_u32(prime)
    if checked_prime > _MAX_PRIME:
        raise ValueError(f"prime must be at most {_MAX_PRIME} for byte digits")
    if not isinstance(width, Integral) or not 1 <= int(width) <= _MAX_WIDTH:
        raise ValueError(
            f"width must be an integer between one and {_MAX_WIDTH}"
        )
    return checked_prime, int(width)


def _space_size(prime: int, width: int) -> int:
    size = prime**width
    if size > np.iinfo(np.uint64).max:
        raise ValueError("p^width does not fit in unsigned 64-bit codes")
    return size


def _prepare_codes(codes: ArrayLike, space_size: int) -> NDArray[np.uint64]:
    array = np.asarray(codes)
    if array.dtype.kind not in {"i", "u", "O"}:
        raise ValueError("codes must be integers")
    if array.ndim != 1:
        raise ValueError("codes must be one-dimensional")
    if array.dtype.kind == "O":
        maximum = np.iinfo(np.uint64).max
        for value in array:
            if not isinstance(value, Integral):
                raise ValueError("codes must be integers")
            if value < 0:
                raise ValueError("codes must be nonnegative")
            if value >= space_size or value > maximum:
                raise ValueError("encoded point is outside the p^width space")
    elif array.dtype.kind == "i" and array.size and np.any(array < 0):
        raise ValueError("codes must be nonnegative")
    prepared = np.ascontiguousarray(array, dtype=np.uint64)
    if prepared.size and np.any(prepared >= np.uint64(space_size)):
        raise ValueError("encoded point is outside the p^width space")
    return prepared


def _prepare_digit_matrix(
    digits: ArrayLike,
    prime: int,
    width: int | None,
) -> NDArray[np.uint8]:
    array = np.asarray(digits)
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError("digits must be integers")
    if array.ndim != 2:
        raise ValueError("digits must be a two-dimensional row matrix")
    if width is not None and array.shape[1] != width:
        raise ValueError("digit rows must have one column per coordinate")
    if array.dtype.kind == "i" and array.size and np.any(array < 0):
        raise ValueError("digits must be nonnegative")
    if array.size and np.any(array >= prime):
        raise ValueError("digit is outside the prime field")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _weights(prime: int, width: int) -> NDArray[np.object_]:
    return np.array(
        [prime**position for position in range(width)], dtype=np.object_
    )


def _encode_rows(
    digits: NDArray[np.integer],
    prime: int,
    weights: NDArray[np.object_],
) -> NDArray[np.uint64]:
    result = np.zeros(len(digits), dtype=np.object_)
    for position in range(digits.shape[1]):
        result += digits[:, position].astype(np.object_) * weights[position]
    maximum = np.iinfo(np.uint64).max
    if result.size and int(result.max()) > maximum:
        raise ValueError("p^width does not fit in unsigned 64-bit codes")
    return result.astype(np.uint64)


def _digits_reference(
    codes: NDArray[np.uint64],
    prime: int,
    width: int,
) -> NDArray[np.uint8]:
    digits = np.empty((len(codes), width), dtype=np.uint8)
    remaining = codes.copy()
    for position in range(width):
        digits[:, position] = remaining % np.uint64(prime)
        remaining //= np.uint64(prime)
    return digits


def _codes_reference(
    digits: NDArray[np.uint8],
    prime: int,
) -> NDArray[np.uint64]:
    return _encode_rows(
        digits, prime, _weights(prime, digits.shape[1])
    )


def _negation_codes_reference(
    codes: NDArray[np.uint64],
    prime: int,
    width: int,
) -> NDArray[np.uint64]:
    negated_digits = (
        prime - _digits_reference(codes, prime, width).astype(np.int64)
    ) % prime
    return _encode_rows(negated_digits, prime, _weights(prime, width))


def _inverse_table(prime: int) -> NDArray[np.int64]:
    table = np.zeros(prime, dtype=np.int64)
    for value in range(1, prime):
        for candidate in range(1, prime):
            if value * candidate % prime == 1:
                table[value] = candidate
                break
    return table


def _scalar_normals_reference(
    codes: NDArray[np.uint64],
    prime: int,
    width: int,
) -> NDArray[np.uint64]:
    digits = _digits_reference(codes, prime, width).astype(np.int64)
    present = digits != 0
    has_nonzero = present.any(axis=1)
    lead = np.where(has_nonzero, np.argmax(present, axis=1), 0)
    lead_digits = digits[np.arange(len(digits)), lead]
    scale = np.where(has_nonzero, _inverse_table(prime)[lead_digits], 0)
    scaled = digits * scale[:, None] % prime
    scaled[~has_nonzero] = 0
    return _encode_rows(scaled, prime, _weights(prime, width))


def _class_table_reference(
    prime: int,
    width: int,
    classes: str,
) -> tuple[NDArray[np.uint32], NDArray[np.uint64], NDArray[np.uint32]]:
    """Dense class ids built straight from the canonical forms.

    Each code's id is the rank of its canonical form among all
    representatives in ascending order. This construction is independent of
    the native marking walk, so parity between them is a real check.
    """
    all_codes = np.arange(prime**width, dtype=np.uint64)
    if classes == "negation":
        canonical = np.minimum(
            all_codes, _negation_codes_reference(all_codes, prime, width)
        )
    else:
        canonical = _scalar_normals_reference(all_codes, prime, width)
    representatives, ids = np.unique(canonical, return_inverse=True)
    counts = np.bincount(ids, minlength=len(representatives))
    return (
        ids.astype(np.uint32),
        representatives.astype(np.uint64),
        counts.astype(np.uint32),
    )


def base_p_digits(
    codes: ArrayLike,
    prime: int,
    width: int,
    *,
    backend: BasePBackend = "auto",
) -> NDArray[np.uint8]:
    """Split encoded F_p^n points into little-endian digit rows.

    Entry ``[i, j]`` holds coefficient ``j`` of ``codes[i]``, the multiple of
    ``p^j``, so coordinate zero is least significant. Every code must lie
    below ``p**width``.
    """
    checked_prime, checked_width = _validate_prime_width(prime, width)
    prepared = _prepare_codes(codes, _space_size(checked_prime, checked_width))
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")

    if backend != "reference" and native_available():
        try:
            result, _ = base_p_digits_native(
                prepared, checked_prime, checked_width
            )
            return result
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    return _digits_reference(prepared, checked_prime, checked_width)


def base_p_codes(
    digits: ArrayLike,
    prime: int,
    *,
    backend: BasePBackend = "auto",
) -> NDArray[np.uint64]:
    """Encode little-endian digit rows of F_p^n points as base-p integers.

    Rows carry one column per coordinate with every entry below ``p``; row
    count equals output length. This is the exact inverse of
    :func:`base_p_digits`.
    """
    array = np.asarray(digits)
    if array.ndim != 2:
        raise ValueError("digits must be a two-dimensional row matrix")
    checked_prime, checked_width = _validate_prime_width(
        prime, array.shape[1]
    )
    _space_size(checked_prime, checked_width)
    prepared = _prepare_digit_matrix(array, checked_prime, checked_width)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")

    if backend != "reference" and native_available():
        try:
            result, _ = base_p_codes_native(prepared, checked_prime, checked_width)
            return result
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    return _codes_reference(prepared, checked_prime)


def base_p_negation_codes(
    codes: ArrayLike,
    prime: int,
    width: int,
    *,
    backend: BasePBackend = "auto",
) -> NDArray[np.uint64]:
    """Negate encoded points digit-wise modulo ``p`` without digit rows.

    Every nonzero coefficient ``d`` of a code becomes ``p - d``, which is the
    additive inverse in F_p^n and not plain integer negation of the code.
    """
    checked_prime, checked_width = _validate_prime_width(prime, width)
    prepared = _prepare_codes(codes, _space_size(checked_prime, checked_width))
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")

    if backend != "reference" and native_available():
        try:
            result, _ = base_p_negation_codes_native(
                prepared, checked_prime, checked_width
            )
            return result
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    return _negation_codes_reference(prepared, checked_prime, checked_width)


def base_p_scalar_normals(
    codes: ArrayLike,
    prime: int,
    width: int,
    *,
    backend: BasePBackend = "auto",
) -> NDArray[np.uint64]:
    """Scale points so their least-significant nonzero digit becomes one.

    One canonical representative per projective point: the zero vector maps
    to zero, and multiplying a point by any unit of F_p lands on the same
    normal form. Negation-class representatives compose differently: take
    ``np.minimum(codes, base_p_negation_codes(...))``.
    """
    checked_prime, checked_width = _validate_prime_width(prime, width)
    prepared = _prepare_codes(codes, _space_size(checked_prime, checked_width))
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")

    if backend != "reference" and native_available():
        try:
            result, _ = base_p_scalar_normals_native(
                prepared, checked_prime, checked_width
            )
            return result
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    return _scalar_normals_reference(prepared, checked_prime, checked_width)


@dataclasses.dataclass(frozen=True)
class BasePClassTable:
    """Whole-space class tables for one equivalence over encoded F_p^n."""

    classes: str
    prime: int
    width: int
    class_ids: NDArray[np.uint32]
    representatives: NDArray[np.uint64]
    counts: NDArray[np.uint32]


def base_p_class_table(
    prime: int,
    width: int,
    *,
    classes: BasePClasses = "negation",
    backend: BasePBackend = "auto",
) -> BasePClassTable:
    """Classify every point of F_p^n into dense, representative-ordered classes.

    ``classes="negation"`` pairs each nonzero code with its digit-wise
    negation (each code alone when ``p=2``); ``classes="scalar"`` groups
    projective points under unit multiples. Class ids rank representatives
    in ascending code order, so the zero vector forms class zero.
    ``representatives`` and ``counts`` are dense over the classes and their
    lengths equal ``(p**width + 1) / 2`` for odd-prime negation classes and
    ``(p**width - 1) // (p - 1) + 1`` for scalar classes.
    """
    checked_prime, checked_width = _validate_prime_width(prime, width)
    space_size = _space_size(checked_prime, checked_width)
    if classes not in {"negation", "scalar"}:
        raise ValueError("classes must be 'negation' or 'scalar'")
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")

    if backend != "reference" and native_available():
        try:
            ids, representatives, counts, _ = base_p_class_table_native(
                checked_prime,
                checked_width,
                0 if classes == "negation" else 1,
                space_size,
                _expected_class_count(classes, checked_prime, space_size),
            )
            return BasePClassTable(
                classes=classes,
                prime=checked_prime,
                width=checked_width,
                class_ids=ids,
                representatives=representatives,
                counts=counts,
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    ids, representatives, counts = _class_table_reference(
        checked_prime, checked_width, classes
    )
    return BasePClassTable(
        classes=classes,
        prime=checked_prime,
        width=checked_width,
        class_ids=ids,
        representatives=representatives,
        counts=counts,
    )


def _expected_class_count(classes: str, prime: int, space_size: int) -> int:
    """Exact number of classes for one equivalence over the whole space."""
    if classes == "negation":
        return space_size if prime == 2 else (space_size + 1) // 2
    return (space_size - 1) // (prime - 1) + 1
