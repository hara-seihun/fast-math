"""Exact spans of integer-encoded points over small prime fields.

Points use the same little-endian base-p encoding as :mod:`fast_math.base_p`.
The reference and native backends return the canonical reduced row echelon
basis, classify query points, and reduce every query to a canonical coordinate
in the quotient by the span.
"""

from __future__ import annotations

import dataclasses
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._native import (
    NativeUnavailable,
    fp_point_span_native,
    fp_span_ranks_native,
    native_available,
)
from .base_p import _prepare_codes, _space_size, _validate_prime_width

__all__ = [
    "FpPointSpan",
    "FpSpanBackend",
    "fp_point_span",
    "fp_span_ranks",
]

FpSpanBackend = Literal["auto", "native", "reference"]


@dataclasses.dataclass(frozen=True)
class FpPointSpan:
    """Canonical RREF data for one span of encoded ``F_p^width`` points.

    ``pivot_indices`` identifies the input points that increased the rank.
    The returned basis rows are ordered by ``pivot_columns``. For query ``q``,
    the result satisfies
    ``q = query_coordinates @ reduced_basis + query_quotient`` over ``F_p``.
    The quotient code is zero exactly when the query belongs to the span.
    """

    prime: int
    width: int
    rank: int
    pivot_indices: NDArray[np.uint64]
    pivot_columns: NDArray[np.uint32]
    reduced_basis_codes: NDArray[np.uint64]
    independent_points: NDArray[np.bool_]
    query_members: NDArray[np.bool_]
    query_coordinates: NDArray[np.uint32]
    query_quotient_codes: NDArray[np.uint64]
    elapsed_seconds: float
    backend: str


def _check_backend(backend: str) -> None:
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")


def _prepare_offsets(
    offsets: ArrayLike, point_count: int
) -> NDArray[np.uint64]:
    raw = np.asarray(offsets)
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u", "O"}:
        raise ValueError("span_offsets must be a one-dimensional integer array")
    if len(raw) == 0:
        raise ValueError("span_offsets must contain at least the initial zero")
    if raw.dtype.kind == "O":
        values = []
        maximum = np.iinfo(np.uint64).max
        for value in raw:
            if (
                not isinstance(value, (int, np.integer))
                or value < 0
                or value > maximum
            ):
                raise ValueError("span_offsets must be nonnegative uint64 integers")
            values.append(int(value))
        prepared = np.asarray(values, dtype=np.uint64)
    else:
        if raw.dtype.kind == "i" and np.any(raw < 0):
            raise ValueError("span_offsets must be nonnegative integers")
        prepared = np.ascontiguousarray(raw, dtype=np.uint64)
    if prepared[0] != 0:
        raise ValueError("span_offsets must start at zero")
    if np.any(prepared[1:] < prepared[:-1]):
        raise ValueError("span_offsets must be nondecreasing")
    if int(prepared[-1]) != point_count:
        raise ValueError("the final span offset must equal the point count")
    return prepared


def _decode(code: int, prime: int, width: int) -> list[int]:
    row = []
    for _ in range(width):
        row.append(code % prime)
        code //= prime
    return row


def _encode(row: list[int], prime: int) -> int:
    code = 0
    for value in reversed(row):
        code = code * prime + value
    return code


def _subtract_scaled(
    left: list[int], right: list[int], scale: int, prime: int
) -> list[int]:
    return [
        (left[column] - scale * right[column]) % prime
        for column in range(len(left))
    ]


@dataclasses.dataclass
class _ReferenceBasis:
    prime: int
    width: int
    rows: list[list[int]] = dataclasses.field(default_factory=list)
    pivot_columns: list[int] = dataclasses.field(default_factory=list)
    pivot_indices: list[int] = dataclasses.field(default_factory=list)

    def add(self, code: int, input_index: int) -> bool:
        row = _decode(code, self.prime, self.width)
        for basis, pivot in zip(self.rows, self.pivot_columns):
            factor = row[pivot]
            if factor:
                row = _subtract_scaled(row, basis, factor, self.prime)
        pivot = next(
            (column for column, value in enumerate(row) if value), None
        )
        if pivot is None:
            return False
        inverse = pow(row[pivot], -1, self.prime)
        row = [value * inverse % self.prime for value in row]
        for index, basis in enumerate(self.rows):
            factor = basis[pivot]
            if factor:
                self.rows[index] = _subtract_scaled(
                    basis, row, factor, self.prime
                )
        position = int(np.searchsorted(self.pivot_columns, pivot))
        self.rows.insert(position, row)
        self.pivot_columns.insert(position, pivot)
        self.pivot_indices.insert(position, input_index)
        return True

    def reduce(self, code: int) -> tuple[list[int], int]:
        row = _decode(code, self.prime, self.width)
        coordinates = []
        for basis, pivot in zip(self.rows, self.pivot_columns):
            factor = row[pivot]
            coordinates.append(factor)
            if factor:
                row = _subtract_scaled(row, basis, factor, self.prime)
        return coordinates, _encode(row, self.prime)


def _build_reference_basis(
    point_codes: NDArray[np.uint64], prime: int, width: int
) -> tuple[_ReferenceBasis, NDArray[np.bool_]]:
    basis = _ReferenceBasis(prime, width)
    independent = np.zeros(len(point_codes), dtype=np.bool_)
    for index, code in enumerate(point_codes):
        independent[index] = basis.add(int(code), index)
    return basis, independent


def _span_ranks_reference(
    point_codes: NDArray[np.uint64],
    span_offsets: NDArray[np.uint64],
    prime: int,
    width: int,
) -> NDArray[np.uint32]:
    ranks = np.empty(len(span_offsets) - 1, dtype=np.uint32)
    for span in range(len(ranks)):
        begin = int(span_offsets[span])
        end = int(span_offsets[span + 1])
        basis, _ = _build_reference_basis(
            point_codes[begin:end], prime, width
        )
        ranks[span] = len(basis.rows)
    return ranks


def _point_span_reference(
    point_codes: NDArray[np.uint64],
    query_codes: NDArray[np.uint64],
    prime: int,
    width: int,
) -> FpPointSpan:
    started = perf_counter()
    basis, independent = _build_reference_basis(point_codes, prime, width)
    rank = len(basis.rows)
    coordinates = np.empty((len(query_codes), rank), dtype=np.uint32)
    quotient_codes = np.empty(len(query_codes), dtype=np.uint64)
    for index, code in enumerate(query_codes):
        coordinate, quotient = basis.reduce(int(code))
        coordinates[index] = coordinate
        quotient_codes[index] = quotient
    basis_codes = np.asarray(
        [_encode(row, prime) for row in basis.rows], dtype=np.uint64
    )
    return FpPointSpan(
        prime=prime,
        width=width,
        rank=rank,
        pivot_indices=np.asarray(basis.pivot_indices, dtype=np.uint64),
        pivot_columns=np.asarray(basis.pivot_columns, dtype=np.uint32),
        reduced_basis_codes=basis_codes,
        independent_points=independent,
        query_members=quotient_codes == 0,
        query_coordinates=coordinates,
        query_quotient_codes=quotient_codes,
        elapsed_seconds=perf_counter() - started,
        backend="reference",
    )


def fp_span_ranks(
    point_codes: ArrayLike,
    span_offsets: ArrayLike,
    prime: int,
    width: int,
    *,
    backend: FpSpanBackend = "auto",
) -> NDArray[np.uint32]:
    """Return one exact rank per ragged batch of encoded field points.

    ``span_offsets`` contains the start of each contiguous point batch and one
    final offset equal to the point count. Empty batches are valid. This shape
    moves repeated small rank calls into one native dispatch.
    """
    checked_prime, checked_width = _validate_prime_width(prime, width)
    points = _prepare_codes(
        point_codes, _space_size(checked_prime, checked_width)
    )
    offsets = _prepare_offsets(span_offsets, len(points))
    _check_backend(backend)
    if backend != "reference" and native_available():
        try:
            ranks, _ = fp_span_ranks_native(
                points, offsets, checked_prime, checked_width
            )
            return ranks
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    return _span_ranks_reference(
        points, offsets, checked_prime, checked_width
    )


def fp_point_span(
    point_codes: ArrayLike,
    query_codes: ArrayLike,
    prime: int,
    width: int,
    *,
    backend: FpSpanBackend = "auto",
) -> FpPointSpan:
    """Build an encoded-point span and reduce a batch of query points.

    The basis is the canonical RREF of the generating points. Query
    coordinates give the component in that basis; quotient codes give the
    canonical residual with every pivot coordinate cleared.
    """
    checked_prime, checked_width = _validate_prime_width(prime, width)
    space_size = _space_size(checked_prime, checked_width)
    points = _prepare_codes(point_codes, space_size)
    queries = _prepare_codes(query_codes, space_size)
    _check_backend(backend)
    if backend != "reference" and native_available():
        try:
            (
                pivot_indices,
                pivot_columns,
                basis_codes,
                independent,
                members,
                coordinates,
                quotient_codes,
                stats,
            ) = fp_point_span_native(
                points, queries, checked_prime, checked_width
            )
            return FpPointSpan(
                prime=checked_prime,
                width=checked_width,
                rank=int(stats.rank_sum),
                pivot_indices=pivot_indices,
                pivot_columns=pivot_columns,
                reduced_basis_codes=basis_codes,
                independent_points=independent,
                query_members=members,
                query_coordinates=coordinates,
                query_quotient_codes=quotient_codes,
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    return _point_span_reference(
        points, queries, checked_prime, checked_width
    )
