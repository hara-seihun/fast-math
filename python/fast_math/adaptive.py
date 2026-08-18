"""Exact adaptive-oracle area kernels on the ternary restriction lattice.

A target is a real function ``g`` on the Rademacher cube ``{-1,+1}^n`` stored as
a table of length ``2**n``. Bit ``i`` of the point index carries coordinate
``i``: zero means ``X_i = -1`` and one means ``X_i = +1``.

A restriction is a code in ``range(3**n)``. Base-three digit ``i`` carries
coordinate ``i``: ``0`` free, ``1`` fixed to ``-1``, ``2`` fixed to ``+1``.
Fixing a free coordinate strictly increases the code, so a descending scan of
the lattice visits every child before its parent.

The root-inclusive adaptive oracle area of the optimal legal adaptive policy is

    A(rho) = 0                                        when Var(g | rho) = 0
    A(rho) = Var(g | rho) + min over fresh i of
             ( A(rho, X_i = -1) + A(rho, X_i = +1) ) / 2

and ``area`` below reports ``A`` at the empty restriction.

The exact backends run entirely in integers. Variance numerators use
denominator ``4**n`` and area numerators use denominator ``2**(3*n)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._native import (
    NativeUnavailable,
    adaptive_area_native,
    native_available,
)

AdaptiveBackend = Literal["auto", "native", "reference"]

MAX_COORDINATES = 20


@dataclass(frozen=True)
class AdaptiveAreaBatch:
    """Float adaptive areas for a batch of targets on one restriction lattice.

    ``first_coordinates`` holds the optimal first query per target, or ``-1``
    when the target is already constant. ``variances``, ``areas_by_restriction``
    and ``policies`` are populated only when ``restrictions=True``.
    """

    coordinate_count: int
    areas: NDArray[np.float64]
    first_coordinates: NDArray[np.int32]
    variances: NDArray[np.float64] | None = None
    areas_by_restriction: NDArray[np.float64] | None = None
    policies: NDArray[np.int8] | None = None


@dataclass(frozen=True)
class ExactAdaptiveAreaBatch:
    """Exact adaptive areas as integer numerators over fixed denominators."""

    coordinate_count: int
    area_numerators: NDArray[np.int64]
    first_coordinates: NDArray[np.int32]
    variance_numerators: NDArray[np.int64] | None = None
    area_numerators_by_restriction: NDArray[np.int64] | None = None
    policies: NDArray[np.int8] | None = None

    @property
    def area_denominator(self) -> int:
        return 1 << (3 * self.coordinate_count)

    @property
    def variance_denominator(self) -> int:
        return 1 << (2 * self.coordinate_count)

    def areas(self) -> list[Fraction]:
        denominator = self.area_denominator
        return [
            Fraction(int(value), denominator) for value in self.area_numerators
        ]


def restriction_count(coordinate_count: int) -> int:
    return 3**coordinate_count


def restriction_code(assignment: ArrayLike) -> int:
    """Map a per-coordinate assignment in ``{-1, 0, +1}`` to a lattice code."""
    code = 0
    power = 1
    for value in np.asarray(assignment).tolist():
        if value == 0:
            digit = 0
        elif value == -1:
            digit = 1
        elif value == 1:
            digit = 2
        else:
            raise ValueError("assignment entries must be -1, 0, or 1")
        code += power * digit
        power *= 3
    return code


def restriction_assignment(code: int, coordinate_count: int) -> tuple[int, ...]:
    """Inverse of :func:`restriction_code`."""
    if not 0 <= code < 3**coordinate_count:
        raise ValueError("code is outside the restriction lattice")
    digits = []
    for _ in range(coordinate_count):
        digit = code % 3
        code //= 3
        digits.append((0, -1, 1)[digit])
    return tuple(digits)


def quadratic_target_tables(
    linear: ArrayLike,
    quadratic: ArrayLike,
) -> NDArray[np.float64]:
    """Expand degree-two Walsh data into full ``2**n`` target tables.

    ``linear`` has shape ``(batch, n)`` and ``quadratic`` shape ``(batch, n, n)``
    with the strictly upper triangle carrying the pair coefficients. The result
    is ``sum_j a_j X_j + sum_{j<k} b_jk X_j X_k`` evaluated on every point.
    """
    linear_array = np.ascontiguousarray(linear, dtype=np.float64)
    if linear_array.ndim == 1:
        linear_array = linear_array[None, :]
    quadratic_array = np.ascontiguousarray(quadratic, dtype=np.float64)
    if quadratic_array.ndim == 2:
        quadratic_array = quadratic_array[None, :, :]
    batch, coordinate_count = linear_array.shape
    if quadratic_array.shape != (batch, coordinate_count, coordinate_count):
        raise ValueError("quadratic must have shape (batch, n, n)")
    _validate_coordinate_count(coordinate_count)

    points = np.arange(1 << coordinate_count, dtype=np.int64)
    signs = np.where(
        ((points[:, None] >> np.arange(coordinate_count)) & 1) == 1, 1.0, -1.0
    )
    tables = signs @ linear_array.T
    upper = np.triu(quadratic_array, 1) + np.transpose(
        np.tril(quadratic_array, -1), (0, 2, 1)
    )
    pairs = np.einsum("xj,xk->xjk", signs, signs)
    tables = tables + np.einsum("xjk,bjk->xb", pairs, upper)
    return np.ascontiguousarray(tables.T)


def _validate_coordinate_count(coordinate_count: int) -> None:
    if (
        not isinstance(coordinate_count, Integral)
        or not 1 <= coordinate_count <= MAX_COORDINATES
    ):
        raise ValueError(
            f"coordinate_count must be an integer in 1..{MAX_COORDINATES}"
        )


def _prepare_tables(tables: ArrayLike, dtype) -> NDArray:
    array = np.ascontiguousarray(tables, dtype=dtype)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError("tables must be one- or two-dimensional")
    point_count = array.shape[1]
    coordinate_count = int(point_count).bit_length() - 1
    if point_count != 1 << coordinate_count:
        raise ValueError("table length must be a power of two")
    _validate_coordinate_count(coordinate_count)
    if dtype is np.float64 and not np.all(np.isfinite(array)):
        raise ValueError("tables must be finite")
    return array


def _free_coordinate_transform(values: NDArray, coordinate_count: int) -> None:
    total = values.shape[1]
    power = 1
    for _ in range(coordinate_count):
        block = power * 3
        view = values.reshape(values.shape[0], total // block, 3, power)
        view[:, :, 0, :] = view[:, :, 1, :] + view[:, :, 2, :]
        power = block


def _lattice_digits(coordinate_count: int) -> NDArray[np.int8]:
    total = 3**coordinate_count
    codes = np.arange(total, dtype=np.int64)
    digits = np.empty((total, coordinate_count), dtype=np.int8)
    for index in range(coordinate_count):
        digits[:, index] = (codes // 3**index) % 3
    return digits


def _reference_areas(
    tables: NDArray,
    coordinate_count: int,
    zero_tolerance: float,
    exact: bool,
):
    total = 3**coordinate_count
    batch = tables.shape[0]
    dtype = np.int64 if exact else np.float64
    sums = np.zeros((batch, total), dtype=object if exact else np.float64)
    squares = np.zeros((batch, total), dtype=object if exact else np.float64)
    for point in range(1 << coordinate_count):
        code = sum(
            3**index * (2 if (point >> index) & 1 else 1)
            for index in range(coordinate_count)
        )
        column = tables[:, point]
        sums[:, code] = column
        squares[:, code] = column * column
    _free_coordinate_transform(sums, coordinate_count)
    _free_coordinate_transform(squares, coordinate_count)

    digits = _lattice_digits(coordinate_count)
    free_counts = (digits == 0).sum(axis=1)
    strides = [3**index for index in range(coordinate_count)]

    values = np.zeros((batch, total), dtype=object if exact else np.float64)
    variances = np.zeros((batch, total), dtype=object if exact else np.float64)
    policies = np.full((batch, total), -1, dtype=np.int8)
    for code in range(total - 1, -1, -1):
        free_count = int(free_counts[code])
        fixed_count = coordinate_count - free_count
        cell = 1 << free_count
        if exact:
            raw = squares[:, code] * cell - sums[:, code] ** 2
            variance = raw * (4**fixed_count)
        else:
            mean = sums[:, code] / cell
            variance = squares[:, code] / cell - mean * mean
        variances[:, code] = variance
        if free_count == 0:
            continue
        free = [
            index for index in range(coordinate_count) if digits[code, index] == 0
        ]
        best = None
        choice = np.full(batch, -1, dtype=np.int8)
        for index in free:
            offset = strides[index]
            child = values[:, code + offset] + values[:, code + 2 * offset]
            if best is None:
                best = child.copy()
                choice[:] = index
            else:
                better = child < best
                best = np.where(better, child, best)
                choice = np.where(better, np.int8(index), choice)
        if exact:
            active = variance != 0
            values[:, code] = np.where(
                active, variance * (2**free_count) + best, 0
            )
        else:
            active = variance > zero_tolerance
            values[:, code] = np.where(active, variance + 0.5 * best, 0.0)
        policies[:, code] = np.where(active, choice, np.int8(-1))

    areas = values[:, 0]
    if exact:
        scale = np.array(
            [2 ** int(coordinate_count - free_counts[code]) for code in range(total)],
            dtype=object,
        )
        values = values * scale
        return (
            np.array([int(value) for value in areas], dtype=np.int64),
            policies[:, 0].astype(np.int32),
            np.array(variances.tolist(), dtype=np.int64),
            np.array(values.tolist(), dtype=np.int64),
            policies,
        )
    return (
        np.ascontiguousarray(areas, dtype=np.float64),
        policies[:, 0].astype(np.int32),
        np.ascontiguousarray(variances, dtype=np.float64),
        np.ascontiguousarray(values, dtype=np.float64),
        policies,
    )


def adaptive_areas(
    tables: ArrayLike,
    *,
    zero_tolerance: float = 0.0,
    threads: int = 0,
    restrictions: bool = False,
    backend: AdaptiveBackend = "auto",
) -> AdaptiveAreaBatch:
    """Optimal adaptive oracle areas for a batch of real targets.

    Each target costs ``O(n * 3**n)`` and each worker holds two ``3**n`` float64
    arrays, so a lattice of ``n`` coordinates needs ``16 * 3**n`` bytes per
    thread.
    """
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    prepared = _prepare_tables(tables, np.float64)
    coordinate_count = int(prepared.shape[1]).bit_length() - 1
    total = 3**coordinate_count
    batch = prepared.shape[0]

    if backend != "reference" and native_available():
        variances = (
            np.empty((batch, total), dtype=np.float64) if restrictions else None
        )
        by_restriction = (
            np.empty((batch, total), dtype=np.float64) if restrictions else None
        )
        policies = (
            np.empty((batch, total), dtype=np.int8) if restrictions else None
        )
        try:
            areas, first = adaptive_area_native(
                prepared,
                coordinate_count,
                zero_tolerance=float(zero_tolerance),
                threads=int(threads),
                variances=variances,
                areas_by_restriction=by_restriction,
                policies=policies,
            )[:2]
            return AdaptiveAreaBatch(
                coordinate_count=coordinate_count,
                areas=areas,
                first_coordinates=first,
                variances=variances,
                areas_by_restriction=by_restriction,
                policies=policies,
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")

    areas, first, variances, by_restriction, policies = _reference_areas(
        prepared, coordinate_count, float(zero_tolerance), exact=False
    )
    return AdaptiveAreaBatch(
        coordinate_count=coordinate_count,
        areas=areas,
        first_coordinates=first,
        variances=variances if restrictions else None,
        areas_by_restriction=by_restriction if restrictions else None,
        policies=policies if restrictions else None,
    )


def exact_adaptive_areas(
    tables: ArrayLike,
    *,
    threads: int = 0,
    restrictions: bool = False,
    backend: AdaptiveBackend = "auto",
) -> ExactAdaptiveAreaBatch:
    """Exact integer adaptive oracle areas for integer-valued targets.

    Entries must satisfy ``2**n * max(entry**2) < 2**63`` so that the lattice
    moment transform stays inside int64. Overflow of an area numerator is an
    error, never a wrapped result.
    """
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    array = np.asarray(tables)
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError("exact adaptive areas require integer tables")
    prepared = _prepare_tables(array, np.int64)
    coordinate_count = int(prepared.shape[1]).bit_length() - 1
    total = 3**coordinate_count
    batch = prepared.shape[0]

    largest = int(np.abs(prepared).max(initial=0))
    if largest**2 * (1 << coordinate_count) >= 1 << 62:
        raise ValueError("table entries are too large for the int64 lattice")

    if backend != "reference" and native_available():
        variances = (
            np.empty((batch, total), dtype=np.int64) if restrictions else None
        )
        by_restriction = (
            np.empty((batch, total), dtype=np.int64) if restrictions else None
        )
        policies = (
            np.empty((batch, total), dtype=np.int8) if restrictions else None
        )
        try:
            areas, first = adaptive_area_native(
                prepared,
                coordinate_count,
                threads=int(threads),
                variances=variances,
                areas_by_restriction=by_restriction,
                policies=policies,
                exact=True,
            )[:2]
            return ExactAdaptiveAreaBatch(
                coordinate_count=coordinate_count,
                area_numerators=areas,
                first_coordinates=first,
                variance_numerators=variances,
                area_numerators_by_restriction=by_restriction,
                policies=policies,
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")

    areas, first, variances, by_restriction, policies = _reference_areas(
        prepared, coordinate_count, 0.0, exact=True
    )
    return ExactAdaptiveAreaBatch(
        coordinate_count=coordinate_count,
        area_numerators=areas,
        first_coordinates=first,
        variance_numerators=variances if restrictions else None,
        area_numerators_by_restriction=by_restriction if restrictions else None,
        policies=policies if restrictions else None,
    )
