from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from fast_math import sparse_block_coloops_mod_u32
from lambda_fast import available_backends


NATIVE_AVAILABLE = "native" in available_backends()


def csr_from_dense(
    matrix: Sequence[Sequence[int]],
    *,
    retain_explicit: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = [0]
    columns = []
    values = []
    for row in matrix:
        for column, value in enumerate(row):
            if value != 0 or retain_explicit:
                columns.append(column)
                values.append(value)
        offsets.append(len(columns))
    return (
        np.asarray(offsets, dtype=np.uint64),
        np.asarray(columns, dtype=np.uint32),
        np.asarray(values, dtype=object),
    )


def dense_rank_mod(matrix: np.ndarray, prime: int) -> int:
    rows = [
        [int(value) % prime for value in row]
        for row in np.asarray(matrix, dtype=object)
    ]
    if not rows:
        return 0
    row_count = len(rows)
    column_count = len(rows[0])
    rank = 0
    for column in range(column_count):
        pivot = next(
            (
                row
                for row in range(rank, row_count)
                if rows[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        reciprocal = pow(rows[rank][column], prime - 2, prime)
        rows[rank] = [
            value * reciprocal % prime for value in rows[rank]
        ]
        for row in range(row_count):
            if row == rank or rows[row][column] == 0:
                continue
            factor = rows[row][column]
            rows[row] = [
                (left - factor * right) % prime
                for left, right in zip(
                    rows[row],
                    rows[rank],
                    strict=True,
                )
            ]
        rank += 1
        if rank == row_count:
            break
    return rank


def verify_reduction(
    matrix: np.ndarray,
    result,
    prime: int,
) -> None:
    matrix_mod = np.asarray(matrix, dtype=object)
    residual_columns = np.flatnonzero(result.residual_mask)
    removed = result.removed_columns.astype(np.intp)
    for index, column in enumerate(removed):
        functional = np.zeros(matrix.shape[0], dtype=object)
        start = int(result.certificate_row_starts[index])
        stop = min(start + result.row_block_size, matrix.shape[0])
        functional[start:stop] = result.certificate_coefficients[
            index,
            : stop - start,
        ].astype(object)
        images = np.asarray(
            functional @ matrix_mod,
            dtype=object,
        )
        assert int(images[column]) % prime == 1
        later = np.concatenate((removed[index + 1 :], residual_columns))
        assert all(int(images[target]) % prime == 0 for target in later)

    residual = matrix_mod[:, residual_columns]
    assert len(removed) + dense_rank_mod(residual, prime) == dense_rank_mod(
        matrix_mod,
        prime,
    )


@pytest.mark.parametrize(
    "backend",
    ["reference"] + (["native"] if NATIVE_AVAILABLE else []),
)
def test_block_coloops_produce_triangular_certificates(
    backend: str,
) -> None:
    matrix = np.asarray(
        [
            [2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 1],
        ],
        dtype=object,
    )
    offsets, columns, values = csr_from_dense(matrix)
    result = sparse_block_coloops_mod_u32(
        offsets,
        columns,
        values,
        column_count=6,
        prime=101,
        row_block_size=2,
        backend=backend,
    )
    np.testing.assert_array_equal(result.removed_columns, [0, 1, 2])
    np.testing.assert_array_equal(
        np.flatnonzero(result.residual_mask),
        [3, 4, 5],
    )
    assert result.active_column_count == 6
    assert result.residual_column_count == 3
    assert result.blocks_processed > result.block_count
    verify_reduction(matrix, result, 101)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
@pytest.mark.parametrize("row_block_size", [1, 2, 3, 4])
@pytest.mark.parametrize("prime", [2, 101, 1_000_003])
def test_native_and_reference_block_coloops_match_exactly(
    row_block_size: int,
    prime: int,
) -> None:
    rng = np.random.default_rng(8100 + row_block_size + prime)
    matrix = rng.integers(
        -2 * prime,
        2 * prime,
        size=(12, 9),
        dtype=np.int64,
    ).astype(object)
    matrix[rng.random(matrix.shape) < 0.72] = 0
    matrix[0, 0] = prime
    offsets, columns, values = csr_from_dense(
        matrix,
        retain_explicit=True,
    )
    reference = sparse_block_coloops_mod_u32(
        offsets,
        columns,
        values,
        column_count=9,
        prime=prime,
        row_block_size=row_block_size,
        backend="reference",
    )
    native = sparse_block_coloops_mod_u32(
        offsets,
        columns,
        values,
        column_count=9,
        prime=prime,
        row_block_size=row_block_size,
        backend="native",
    )
    np.testing.assert_array_equal(
        native.residual_mask,
        reference.residual_mask,
    )
    np.testing.assert_array_equal(
        native.removed_columns,
        reference.removed_columns,
    )
    np.testing.assert_array_equal(
        native.certificate_row_starts,
        reference.certificate_row_starts,
    )
    np.testing.assert_array_equal(
        native.certificate_coefficients,
        reference.certificate_coefficients,
    )
    assert native.block_incidences == reference.block_incidences
    assert native.blocks_processed == reference.blocks_processed
    verify_reduction(matrix, native, prime)


@pytest.mark.parametrize(
    "backend",
    ["reference"] + (["native"] if NATIVE_AVAILABLE else []),
)
def test_block_coloops_reduce_coefficients_before_structure(
    backend: str,
) -> None:
    matrix = np.asarray(
        [
            [101, 7, 0],
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=object,
    )
    offsets, columns, values = csr_from_dense(
        matrix,
        retain_explicit=True,
    )
    result = sparse_block_coloops_mod_u32(
        offsets,
        columns,
        values,
        column_count=3,
        prime=101,
        row_block_size=2,
        backend=backend,
    )
    assert result.input_nonzeros == 5
    verify_reduction(matrix, result, 101)


@pytest.mark.parametrize("row_block_size", [0, 17, -1, True])
def test_block_coloops_reject_invalid_block_size(
    row_block_size: int,
) -> None:
    with pytest.raises(ValueError, match="row_block_size"):
        sparse_block_coloops_mod_u32(
            [0, 1],
            [0],
            [1],
            column_count=1,
            prime=101,
            row_block_size=row_block_size,
            backend="reference",
        )
