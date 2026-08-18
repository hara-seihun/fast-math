"""Deterministic exact rank for sparse matrices over 32-bit prime fields."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral
import time
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._native import (
    NativeUnavailable,
    sparse_block_coloops_mod_u32_native,
    sparse_rank_mod_u32_batch_native,
    sparse_rank_mod_u32_native,
)


Backend = Literal["auto", "native", "reference"]


@dataclass(frozen=True)
class SparseRankResult:
    rank: int
    pivot_rows: NDArray[np.uint64]
    pivot_columns: NDArray[np.uint32]
    row_count: int
    column_count: int
    input_nonzeros: int
    active_rows: int
    processed_rows: int
    dependent_rows: int
    elimination_steps: int
    basis_nonzeros: int
    maximum_basis_size: int
    maximum_working_size: int
    peeled_pivots: int
    residual_rows: int
    residual_columns: int
    residual_nonzeros: int
    prime: int
    target_reached: bool
    preprocessing_seconds: float
    elapsed_seconds: float
    backend: str


@dataclass(frozen=True)
class SparseRankBatchResult:
    results: tuple[SparseRankResult, ...]
    prime_count: int
    thread_count: int
    elapsed_seconds: float
    backend: str


@dataclass(frozen=True)
class SparseBlockColoopResult:
    residual_mask: NDArray[np.bool_]
    removed_columns: NDArray[np.uint32]
    certificate_row_starts: NDArray[np.uint64]
    certificate_coefficients: NDArray[np.uint32]
    row_count: int
    column_count: int
    input_nonzeros: int
    block_count: int
    block_incidences: int
    active_column_count: int
    residual_column_count: int
    blocks_processed: int
    maximum_block_columns: int
    row_block_size: int
    prime: int
    elapsed_seconds: float
    backend: str


def _is_prime_u32(value: int) -> bool:
    if value < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value == small:
            return True
        if value % small == 0:
            return False
    odd_part = value - 1
    shifts = 0
    while odd_part & 1 == 0:
        odd_part >>= 1
        shifts += 1
    for base in (2, 3, 5, 7, 11):
        if base >= value:
            continue
        witness = pow(base, odd_part, value)
        if witness in (1, value - 1):
            continue
        for _ in range(1, shifts):
            witness = witness * witness % value
            if witness == value - 1:
                break
        else:
            return False
    return True


def _validate_prime(prime: int) -> int:
    if (
        not isinstance(prime, Integral)
        or isinstance(prime, bool)
        or not 2 <= prime <= np.iinfo(np.uint32).max
        or not _is_prime_u32(int(prime))
    ):
        raise ValueError("prime must be a 32-bit prime")
    return int(prime)


def _prepare_unsigned(
    source: ArrayLike,
    *,
    name: str,
    dtype: np.dtype,
    maximum: int,
) -> np.ndarray:
    array = np.asarray(source)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    if array.size == 0:
        return np.empty(0, dtype=dtype)
    if array.dtype.kind == "O":
        for value in array:
            if not isinstance(value, Integral):
                raise ValueError(f"{name} must contain integers")
            if value < 0 or value > maximum:
                raise ValueError(f"{name} values are out of range")
    elif array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain integers")
    else:
        if array.dtype.kind == "i" and np.any(array < 0):
            raise ValueError(f"{name} values are out of range")
        if array.size and int(np.max(array)) > maximum:
            raise ValueError(f"{name} values are out of range")
    return np.ascontiguousarray(array, dtype=dtype)


def _prepare_values(
    source: ArrayLike,
    *,
    prime: int,
) -> NDArray[np.uint32]:
    array = np.asarray(source)
    if array.ndim != 1:
        raise ValueError("values must be a vector")
    if array.size == 0:
        return np.empty(0, dtype=np.uint32)
    if array.dtype.kind == "O":
        reduced = []
        for value in array:
            if not isinstance(value, Integral):
                raise ValueError("values must contain integers")
            reduced.append(int(value) % prime)
        return np.ascontiguousarray(reduced, dtype=np.uint32)
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError("values must contain integers")
    reduced = np.remainder(array, prime)
    return np.ascontiguousarray(reduced, dtype=np.uint32)


def _prepare(
    row_offsets: ArrayLike,
    column_indices: ArrayLike,
    values: ArrayLike,
    *,
    column_count: int,
    prime: int,
    target_rank: int,
) -> tuple[
    NDArray[np.uint64],
    NDArray[np.uint32],
    NDArray[np.uint32],
]:
    if (
        not isinstance(column_count, Integral)
        or isinstance(column_count, bool)
        or not 1 <= column_count <= np.iinfo(np.uint32).max
    ):
        raise ValueError("column_count must be a positive uint32")
    prime = _validate_prime(prime)
    if (
        not isinstance(target_rank, Integral)
        or isinstance(target_rank, bool)
        or target_rank < 0
    ):
        raise ValueError("target_rank must be a nonnegative integer")

    offsets = _prepare_unsigned(
        row_offsets,
        name="row_offsets",
        dtype=np.dtype(np.uint64),
        maximum=np.iinfo(np.uint64).max,
    )
    columns = _prepare_unsigned(
        column_indices,
        name="column_indices",
        dtype=np.dtype(np.uint32),
        maximum=np.iinfo(np.uint32).max,
    )
    reduced_values = _prepare_values(values, prime=int(prime))
    if len(offsets) < 2:
        raise ValueError("row_offsets must describe at least one row")
    if len(columns) != len(reduced_values):
        raise ValueError("column_indices and values must have equal length")
    if (
        int(offsets[0]) != 0
        or int(offsets[-1]) != len(columns)
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("row_offsets must delimit the entry arrays")
    row_count = len(offsets) - 1
    maximum_rank = min(row_count, int(column_count))
    if target_rank > maximum_rank:
        raise ValueError("target_rank exceeds matrix dimensions")
    if len(columns) and int(np.max(columns)) >= column_count:
        raise ValueError("column index is out of range")
    for row in range(row_count):
        begin = int(offsets[row])
        end = int(offsets[row + 1])
        if (
            end - begin > 1
            and np.any(
                columns[begin + 1 : end]
                <= columns[begin : end - 1]
            )
        ):
            raise ValueError("columns must increase within each row")
    return offsets, columns, reduced_values


def _from_native(
    pivot_rows: NDArray[np.uint64],
    pivot_columns: NDArray[np.uint32],
    stats,
) -> SparseRankResult:
    return SparseRankResult(
        rank=int(stats.rank),
        pivot_rows=pivot_rows,
        pivot_columns=pivot_columns,
        row_count=int(stats.row_count),
        column_count=int(stats.column_count),
        input_nonzeros=int(stats.input_nonzeros),
        active_rows=int(stats.active_rows),
        processed_rows=int(stats.processed_rows),
        dependent_rows=int(stats.dependent_rows),
        elimination_steps=int(stats.elimination_steps),
        basis_nonzeros=int(stats.basis_nonzeros),
        maximum_basis_size=int(stats.maximum_basis_size),
        maximum_working_size=int(stats.maximum_working_size),
        peeled_pivots=int(stats.peeled_pivots),
        residual_rows=int(stats.residual_rows),
        residual_columns=int(stats.residual_columns),
        residual_nonzeros=int(stats.residual_nonzeros),
        prime=int(stats.prime),
        target_reached=bool(stats.target_reached),
        preprocessing_seconds=float(stats.preprocessing_seconds),
        elapsed_seconds=float(stats.elapsed_seconds),
        backend="native",
    )


def _block_coloop_from_native(
    residual_mask: NDArray[np.bool_],
    removed_columns: NDArray[np.uint32],
    certificate_row_starts: NDArray[np.uint64],
    certificate_coefficients: NDArray[np.uint32],
    stats,
) -> SparseBlockColoopResult:
    return SparseBlockColoopResult(
        residual_mask=residual_mask,
        removed_columns=removed_columns,
        certificate_row_starts=certificate_row_starts,
        certificate_coefficients=certificate_coefficients,
        row_count=int(stats.row_count),
        column_count=int(stats.column_count),
        input_nonzeros=int(stats.input_nonzeros),
        block_count=int(stats.block_count),
        block_incidences=int(stats.block_incidences),
        active_column_count=int(stats.active_columns),
        residual_column_count=int(stats.residual_columns),
        blocks_processed=int(stats.blocks_processed),
        maximum_block_columns=int(stats.maximum_block_columns),
        row_block_size=int(stats.row_block_size),
        prime=int(stats.prime),
        elapsed_seconds=float(stats.elapsed_seconds),
        backend="native",
    )


def _invert_small_square(
    matrix: list[list[int]],
    prime: int,
) -> list[list[int]]:
    order = len(matrix)
    left = [row.copy() for row in matrix]
    inverse = [
        [int(row == column) for column in range(order)]
        for row in range(order)
    ]
    for column in range(order):
        pivot = next(
            (
                row
                for row in range(column, order)
                if left[row][column]
            ),
            None,
        )
        if pivot is None:
            raise RuntimeError("local coloop basis minor is singular")
        left[column], left[pivot] = left[pivot], left[column]
        inverse[column], inverse[pivot] = (
            inverse[pivot],
            inverse[column],
        )
        reciprocal = pow(left[column][column], prime - 2, prime)
        left[column] = [
            value * reciprocal % prime for value in left[column]
        ]
        inverse[column] = [
            value * reciprocal % prime for value in inverse[column]
        ]
        for row in range(order):
            if row == column or left[row][column] == 0:
                continue
            factor = left[row][column]
            left[row] = [
                (value - factor * pivot_value) % prime
                for value, pivot_value in zip(
                    left[row],
                    left[column],
                    strict=True,
                )
            ]
            inverse[row] = [
                (value - factor * pivot_value) % prime
                for value, pivot_value in zip(
                    inverse[row],
                    inverse[column],
                    strict=True,
                )
            ]
    return inverse


def _local_block_coloops(
    entries: list[tuple[int, list[int]]],
    active_columns: list[bool],
    *,
    row_block_size: int,
    prime: int,
) -> list[tuple[int, list[int]]]:
    basis: list[dict[str, object]] = []
    for column, original in entries:
        if not active_columns[column]:
            continue
        working = original.copy()
        for vector in basis:
            pivot_row = int(vector["pivot_row"])
            factor = working[pivot_row]
            if not factor:
                continue
            echelon = vector["echelon"]
            assert isinstance(echelon, list)
            working = [
                (value - factor * pivot_value) % prime
                for value, pivot_value in zip(
                    working,
                    echelon,
                    strict=True,
                )
            ]
        pivot_row = next(
            (
                row
                for row, value in enumerate(working)
                if value
            ),
            None,
        )
        if pivot_row is None:
            continue
        reciprocal = pow(working[pivot_row], prime - 2, prime)
        basis.append(
            {
                "column": column,
                "original": original,
                "echelon": [
                    value * reciprocal % prime
                    for value in working
                ],
                "pivot_row": pivot_row,
                "replaceable": False,
            }
        )

    if not basis:
        return []
    basis_minor = [
        [
            int(vector["original"][int(row_vector["pivot_row"])])
            for vector in basis
        ]
        for row_vector in basis
    ]
    inverse = _invert_small_square(basis_minor, prime)
    basis_columns = {
        int(vector["column"]) for vector in basis
    }
    pivot_rows = [
        int(vector["pivot_row"]) for vector in basis
    ]
    for column, original in entries:
        if not active_columns[column] or column in basis_columns:
            continue
        selected_values = [original[row] for row in pivot_rows]
        for index in range(len(basis)):
            coordinate = sum(
                inverse[index][selected] * value
                for selected, value in enumerate(selected_values)
            ) % prime
            if coordinate:
                basis[index]["replaceable"] = True
    coloops = []
    for index, vector in enumerate(basis):
        if bool(vector["replaceable"]):
            continue
        coefficients = [0] * row_block_size
        for coordinate, row_vector in enumerate(basis):
            coefficients[int(row_vector["pivot_row"])] = (
                inverse[index][coordinate]
            )
        coloops.append((int(vector["column"]), coefficients))
    return coloops


def _block_coloop_reference(
    row_offsets: NDArray[np.uint64],
    column_indices: NDArray[np.uint32],
    values: NDArray[np.uint32],
    *,
    column_count: int,
    prime: int,
    row_block_size: int,
) -> SparseBlockColoopResult:
    started = time.perf_counter()
    row_count = len(row_offsets) - 1
    rows: list[list[tuple[int, int]]] = []
    retained_nonzeros = 0
    for row in range(row_count):
        entries = []
        for offset in range(
            int(row_offsets[row]),
            int(row_offsets[row + 1]),
        ):
            value = int(values[offset])
            if value:
                entries.append((int(column_indices[offset]), value))
                retained_nonzeros += 1
        rows.append(entries)

    block_count = (
        row_count + row_block_size - 1
    ) // row_block_size
    blocks: list[list[tuple[int, list[int]]]] = []
    incident: list[list[int]] = [
        [] for _ in range(column_count)
    ]
    maximum_block_columns = 0
    for block in range(block_count):
        local: dict[int, list[int]] = {}
        first_row = block * row_block_size
        for local_row, entries in enumerate(
            rows[first_row : first_row + row_block_size]
        ):
            for column, value in entries:
                vector = local.setdefault(
                    column,
                    [0] * row_block_size,
                )
                vector[local_row] = value
        block_entries = sorted(local.items())
        blocks.append(block_entries)
        maximum_block_columns = max(
            maximum_block_columns,
            len(block_entries),
        )
        for column, _ in block_entries:
            incident[column].append(block)

    active = [bool(cards) for cards in incident]
    active_column_count = sum(active)
    queue = deque(range(block_count))
    queued = [True] * block_count
    removed_columns = []
    row_starts = []
    certificates = []
    blocks_processed = 0
    while queue:
        block = queue.popleft()
        queued[block] = False
        blocks_processed += 1
        coloops = _local_block_coloops(
            blocks[block],
            active,
            row_block_size=row_block_size,
            prime=prime,
        )
        for column, coefficients in coloops:
            if not active[column]:
                continue
            active[column] = False
            removed_columns.append(column)
            row_starts.append(block * row_block_size)
            certificates.append(coefficients)
            for affected in incident[column]:
                if not queued[affected]:
                    queue.append(affected)
                    queued[affected] = True

    return SparseBlockColoopResult(
        residual_mask=np.asarray(active, dtype=np.bool_),
        removed_columns=np.asarray(
            removed_columns,
            dtype=np.uint32,
        ),
        certificate_row_starts=np.asarray(
            row_starts,
            dtype=np.uint64,
        ),
        certificate_coefficients=np.asarray(
            certificates,
            dtype=np.uint32,
        ).reshape((-1, row_block_size)),
        row_count=row_count,
        column_count=column_count,
        input_nonzeros=retained_nonzeros,
        block_count=block_count,
        block_incidences=sum(map(len, blocks)),
        active_column_count=active_column_count,
        residual_column_count=sum(active),
        blocks_processed=blocks_processed,
        maximum_block_columns=maximum_block_columns,
        row_block_size=row_block_size,
        prime=prime,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def _subtract_pivot(
    row: list[tuple[int, int]],
    pivot: list[tuple[int, int]],
    factor: int,
    prime: int,
) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    left = 1
    right = 1
    while left < len(row) or right < len(pivot):
        if right == len(pivot) or (
            left < len(row) and row[left][0] < pivot[right][0]
        ):
            output.append(row[left])
            left += 1
            continue
        product = factor * pivot[right][1] % prime
        if left == len(row) or pivot[right][0] < row[left][0]:
            output.append((pivot[right][0], -product % prime))
            right += 1
            continue
        value = (row[left][1] - product) % prime
        if value:
            output.append((row[left][0], value))
        left += 1
        right += 1
    return output


def _reference(
    row_offsets: NDArray[np.uint64],
    column_indices: NDArray[np.uint32],
    values: NDArray[np.uint32],
    *,
    column_count: int,
    prime: int,
    target_rank: int,
) -> SparseRankResult:
    started = time.perf_counter()
    row_count = len(row_offsets) - 1
    rows: list[list[tuple[int, int]]] = []
    column_rows: list[list[int]] = [
        [] for _ in range(column_count)
    ]
    column_degree = [0] * column_count
    retained_nonzeros = 0
    for row in range(row_count):
        entries = []
        for offset in range(
            int(row_offsets[row]),
            int(row_offsets[row + 1]),
        ):
            value = int(values[offset])
            if value:
                column = int(column_indices[offset])
                entries.append((column, value))
                column_rows[column].append(row)
                column_degree[column] += 1
                retained_nonzeros += 1
        rows.append(entries)

    requested_rank = target_rank or min(row_count, column_count)
    row_degree = [len(row) for row in rows]
    row_active = [degree != 0 for degree in row_degree]
    column_active = [degree != 0 for degree in column_degree]
    active_row_count = sum(row_active)
    peel_queue = deque(
        (0, row)
        for row, degree in enumerate(row_degree)
        if degree == 1
    )
    peel_queue.extend(
        (1, column)
        for column, degree in enumerate(column_degree)
        if degree == 1
    )
    pivot_rows: list[int] = []
    pivot_columns: list[int] = []
    residual_nonzeros = retained_nonzeros
    while peel_queue and len(pivot_rows) < requested_rank:
        kind, index = peel_queue.popleft()
        row = -1
        column = -1
        if kind == 0:
            row = index
            if not row_active[row] or row_degree[row] != 1:
                continue
            column = next(
                (
                    candidate
                    for candidate, _ in rows[row]
                    if column_active[candidate]
                ),
                -1,
            )
        else:
            column = index
            if (
                not column_active[column]
                or column_degree[column] != 1
            ):
                continue
            row = next(
                (
                    candidate
                    for candidate in column_rows[column]
                    if row_active[candidate]
                ),
                -1,
            )
        if (
            row < 0
            or column < 0
            or not row_active[row]
            or not column_active[column]
        ):
            continue

        residual_nonzeros -= (
            row_degree[row] + column_degree[column] - 1
        )
        row_active[row] = False
        column_active[column] = False
        pivot_rows.append(row)
        pivot_columns.append(column)
        row_degree[row] = 0
        column_degree[column] = 0
        for neighbor, _ in rows[row]:
            if not column_active[neighbor]:
                continue
            column_degree[neighbor] -= 1
            if column_degree[neighbor] == 1:
                peel_queue.append((1, neighbor))
        for neighbor in column_rows[column]:
            if not row_active[neighbor]:
                continue
            row_degree[neighbor] -= 1
            if row_degree[neighbor] == 1:
                peel_queue.append((0, neighbor))

    preprocessing_seconds = time.perf_counter() - started
    column_order = sorted(
        (
            column
            for column in range(column_count)
            if column_active[column] and column_degree[column]
        ),
        key=lambda column: (column_degree[column], column),
    )
    reordered_column = [-1] * column_count
    for position, column in enumerate(column_order):
        reordered_column[column] = position
    active_rows = []
    first_reordered_column = [-1] * row_count
    row_markowitz_cost = [0] * row_count
    for row, entries in enumerate(rows):
        if not row_active[row] or row_degree[row] == 0:
            continue
        rows[row] = sorted(
            (
                (reordered_column[column], value)
                for column, value in entries
                if column_active[column]
            ),
        )
        first_reordered_column[row] = rows[row][0][0]
        pivot_column = column_order[first_reordered_column[row]]
        row_markowitz_cost[row] = (
            (row_degree[row] - 1)
            * (column_degree[pivot_column] - 1)
        )
        active_rows.append(row)
    active_rows.sort(
        key=lambda row: (
            row_markowitz_cost[row],
            row_degree[row],
            first_reordered_column[row],
            row,
        )
    )

    basis: list[list[tuple[int, int]] | None] = [
        None
    ] * len(column_order)
    peeled_pivots = len(pivot_rows)
    processed_rows = peeled_pivots
    dependent_rows = 0
    elimination_steps = 0
    maximum_working_size = 0
    for row_id in active_rows:
        if len(pivot_rows) >= requested_rank:
            break
        working = rows[row_id]
        independent = False
        maximum_working_size = max(maximum_working_size, len(working))
        while working:
            pivot_column = working[0][0]
            pivot = basis[pivot_column]
            if pivot is None:
                inverse = pow(working[0][1], prime - 2, prime)
                working = [
                    (column, value * inverse % prime)
                    for column, value in working
                ]
                basis[pivot_column] = working
                pivot_rows.append(row_id)
                pivot_columns.append(column_order[pivot_column])
                independent = True
                break
            working = _subtract_pivot(
                working,
                pivot,
                working[0][1],
                prime,
            )
            elimination_steps += 1
            maximum_working_size = max(
                maximum_working_size,
                len(working),
            )
        if not independent:
            dependent_rows += 1
        processed_rows += 1

    basis_rows = [row for row in basis if row is not None]
    rank = len(pivot_rows)
    return SparseRankResult(
        rank=rank,
        pivot_rows=np.asarray(pivot_rows, dtype=np.uint64),
        pivot_columns=np.asarray(pivot_columns, dtype=np.uint32),
        row_count=row_count,
        column_count=column_count,
        input_nonzeros=retained_nonzeros,
        active_rows=active_row_count,
        processed_rows=processed_rows,
        dependent_rows=dependent_rows,
        elimination_steps=elimination_steps,
        basis_nonzeros=sum(map(len, basis_rows)),
        maximum_basis_size=max(map(len, basis_rows), default=0),
        maximum_working_size=maximum_working_size,
        peeled_pivots=peeled_pivots,
        residual_rows=len(active_rows),
        residual_columns=len(column_order),
        residual_nonzeros=residual_nonzeros,
        prime=prime,
        target_reached=rank >= requested_rank,
        preprocessing_seconds=preprocessing_seconds,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def sparse_block_coloops_mod_u32(
    row_offsets: ArrayLike,
    column_indices: ArrayLike,
    values: ArrayLike,
    *,
    column_count: int,
    prime: int,
    row_block_size: int,
    backend: Backend = "auto",
) -> SparseBlockColoopResult:
    """Peel exact column coloops exposed by contiguous row blocks.

    Each certificate is a row functional supported on one block. In removal
    order it maps its own column to one and every later removed or residual
    column to zero, proving exact rank decomposition.
    """
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    if (
        not isinstance(row_block_size, Integral)
        or isinstance(row_block_size, bool)
        or not 1 <= row_block_size <= 16
    ):
        raise ValueError("row_block_size must be in [1, 16]")
    offsets, columns, reduced_values = _prepare(
        row_offsets,
        column_indices,
        values,
        column_count=column_count,
        prime=prime,
        target_rank=0,
    )
    if backend in {"auto", "native"}:
        try:
            native_result = sparse_block_coloops_mod_u32_native(
                offsets,
                columns,
                reduced_values,
                column_count=column_count,
                prime=prime,
                row_block_size=int(row_block_size),
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return _block_coloop_from_native(*native_result)
    return _block_coloop_reference(
        offsets,
        columns,
        reduced_values,
        column_count=column_count,
        prime=prime,
        row_block_size=int(row_block_size),
    )


def sparse_rank_mod_u32(
    row_offsets: ArrayLike,
    column_indices: ArrayLike,
    values: ArrayLike,
    *,
    column_count: int,
    prime: int,
    target_rank: int = 0,
    backend: Backend = "auto",
) -> SparseRankResult:
    """Compute exact sparse rank and deterministic pivot witnesses.

    The inputs use zero-based CSR. A target of zero computes the full rank;
    a positive target stops as soon as that many independent rows are found.
    """
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    offsets, columns, reduced_values = _prepare(
        row_offsets,
        column_indices,
        values,
        column_count=column_count,
        prime=prime,
        target_rank=target_rank,
    )
    if backend in {"auto", "native"}:
        try:
            pivot_rows, pivot_columns, stats = (
                sparse_rank_mod_u32_native(
                    offsets,
                    columns,
                    reduced_values,
                    column_count=column_count,
                    prime=prime,
                    target_rank=target_rank,
                )
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return _from_native(pivot_rows, pivot_columns, stats)
    return _reference(
        offsets,
        columns,
        reduced_values,
        column_count=column_count,
        prime=prime,
        target_rank=target_rank,
    )


def sparse_rank_mod_u32_batch(
    row_offsets: ArrayLike,
    column_indices: ArrayLike,
    values: ArrayLike,
    *,
    column_count: int,
    primes: tuple[int, ...] | list[int],
    target_rank: int = 0,
    threads: int = 0,
    backend: Backend = "auto",
) -> SparseRankBatchResult:
    """Compute independent exact-rank certificates across several primes."""
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    if (
        not isinstance(threads, Integral)
        or isinstance(threads, bool)
        or threads < 0
    ):
        raise ValueError("threads must be a nonnegative integer")
    prime_values = tuple(_validate_prime(prime) for prime in primes)
    if not prime_values:
        raise ValueError("primes must contain at least one prime")

    offsets, columns, first_values = _prepare(
        row_offsets,
        column_indices,
        values,
        column_count=column_count,
        prime=prime_values[0],
        target_rank=target_rank,
    )
    value_rows = [first_values]
    value_rows.extend(
        _prepare_values(values, prime=prime)
        for prime in prime_values[1:]
    )
    values_by_prime = np.ascontiguousarray(
        np.stack(value_rows),
        dtype=np.uint32,
    )
    prime_array = np.ascontiguousarray(prime_values, dtype=np.uint32)

    if backend in {"auto", "native"}:
        try:
            witnesses, native_stats, batch_stats = (
                sparse_rank_mod_u32_batch_native(
                    offsets,
                    columns,
                    values_by_prime,
                    prime_array,
                    column_count=column_count,
                    target_rank=target_rank,
                    threads=int(threads),
                )
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            results = tuple(
                _from_native(
                    pivot_rows,
                    pivot_columns,
                    stats,
                )
                for (pivot_rows, pivot_columns), stats in zip(
                    witnesses,
                    native_stats,
                    strict=True,
                )
            )
            return SparseRankBatchResult(
                results=results,
                prime_count=int(batch_stats.prime_count),
                thread_count=int(batch_stats.thread_count),
                elapsed_seconds=float(batch_stats.elapsed_seconds),
                backend="native",
            )

    started = time.perf_counter()
    results = tuple(
        _reference(
            offsets,
            columns,
            value_row,
            column_count=column_count,
            prime=prime,
            target_rank=target_rank,
        )
        for prime, value_row in zip(
            prime_values,
            value_rows,
            strict=True,
        )
    )
    return SparseRankBatchResult(
        results=results,
        prime_count=len(results),
        thread_count=1,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )
