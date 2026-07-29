from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from fast_math import sparse_rank_mod_u32, sparse_rank_mod_u32_batch
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
        inverse = pow(rows[rank][column], prime - 2, prime)
        rows[rank] = [
            value * inverse % prime for value in rows[rank]
        ]
        for row in range(row_count):
            if row == rank or rows[row][column] == 0:
                continue
            factor = rows[row][column]
            rows[row] = [
                (left - factor * right) % prime
                for left, right in zip(rows[row], rows[rank], strict=True)
            ]
        rank += 1
        if rank == row_count:
            break
    return rank


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
@pytest.mark.parametrize(
    "prime",
    [
        2,
        3,
        101,
        1_000_003,
        1_000_000_007,
        2_147_483_647,
        4_294_967_291,
    ],
)
def test_sparse_rank_matches_independent_dense_oracle(prime: int) -> None:
    rng = np.random.default_rng(7700 + prime)
    for row_count, column_count in ((1, 1), (4, 7), (9, 5), (11, 13)):
        matrix = rng.integers(
            -2 * prime,
            2 * prime,
            size=(row_count, column_count),
            dtype=np.int64,
        ).astype(object)
        matrix[rng.random(matrix.shape) < 0.72] = 0
        matrix[0, 0] = prime - 1
        if column_count > 1:
            matrix[0, 1] = prime
        offsets, columns, values = csr_from_dense(
            matrix,
            retain_explicit=True,
        )
        result = sparse_rank_mod_u32(
            offsets,
            columns,
            values,
            column_count=column_count,
            prime=prime,
            backend="native",
        )
        expected_rank = dense_rank_mod(matrix, prime)
        assert result.rank == expected_rank
        assert result.target_reached == (
            expected_rank == min(row_count, column_count)
        )

        witness = matrix[
            np.ix_(
                result.pivot_rows.astype(np.intp),
                result.pivot_columns.astype(np.intp),
            )
        ]
        assert dense_rank_mod(witness, prime) == result.rank


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_native_and_reference_choose_identical_deterministic_witnesses() -> None:
    rng = np.random.default_rng(7711)
    matrix = rng.integers(-211, 212, size=(37, 29), dtype=np.int64)
    matrix[rng.random(matrix.shape) < 0.82] = 0
    offsets, columns, values = csr_from_dense(matrix)
    reference = sparse_rank_mod_u32(
        offsets,
        columns,
        values,
        column_count=matrix.shape[1],
        prime=211,
        backend="reference",
    )
    first = sparse_rank_mod_u32(
        offsets,
        columns,
        values,
        column_count=matrix.shape[1],
        prime=211,
        backend="native",
    )
    second = sparse_rank_mod_u32(
        offsets,
        columns,
        values,
        column_count=matrix.shape[1],
        prime=211,
        backend="native",
    )
    assert first.rank == reference.rank
    assert first.elimination_steps == reference.elimination_steps
    assert first.basis_nonzeros == reference.basis_nonzeros
    assert first.peeled_pivots == reference.peeled_pivots
    assert first.residual_rows == reference.residual_rows
    assert first.residual_columns == reference.residual_columns
    assert first.residual_nonzeros == reference.residual_nonzeros
    np.testing.assert_array_equal(first.pivot_rows, reference.pivot_rows)
    np.testing.assert_array_equal(
        first.pivot_columns,
        reference.pivot_columns,
    )
    np.testing.assert_array_equal(first.pivot_rows, second.pivot_rows)
    np.testing.assert_array_equal(first.pivot_columns, second.pivot_columns)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_markowitz_order_precedes_lower_degree_when_fill_cost_is_smaller() -> None:
    matrix = np.asarray(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 2, 0],
            [0, 1, 1, 1],
            [0, 1, 2, 1],
        ],
        dtype=np.int64,
    )
    offsets, columns, values = csr_from_dense(matrix)
    reference = sparse_rank_mod_u32(
        offsets,
        columns,
        values,
        column_count=matrix.shape[1],
        prime=101,
        backend="reference",
    )
    native = sparse_rank_mod_u32(
        offsets,
        columns,
        values,
        column_count=matrix.shape[1],
        prime=101,
        backend="native",
    )

    # Row zero has lower degree, but row one has lower predicted fill.
    assert reference.pivot_rows.tolist() == [1, 2, 3, 0]
    assert reference.pivot_columns.tolist() == [0, 2, 3, 1]
    assert reference.elimination_steps == 3
    np.testing.assert_array_equal(native.pivot_rows, reference.pivot_rows)
    np.testing.assert_array_equal(
        native.pivot_columns,
        reference.pivot_columns,
    )
    assert native.elimination_steps == reference.elimination_steps


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_multi_prime_batch_matches_serial_and_is_deterministic() -> None:
    rng = np.random.default_rng(7712)
    matrix = rng.integers(-500, 501, size=(73, 61), dtype=np.int64)
    matrix[rng.random(matrix.shape) < 0.9] = 0
    offsets, columns, values = csr_from_dense(matrix)
    primes = (101, 211, 1_000_003)
    serial = tuple(
        sparse_rank_mod_u32(
            offsets,
            columns,
            values,
            column_count=matrix.shape[1],
            prime=prime,
            backend="native",
        )
        for prime in primes
    )
    first = sparse_rank_mod_u32_batch(
        offsets,
        columns,
        values,
        column_count=matrix.shape[1],
        primes=primes,
        threads=3,
        backend="native",
    )
    second = sparse_rank_mod_u32_batch(
        offsets,
        columns,
        values,
        column_count=matrix.shape[1],
        primes=primes,
        threads=3,
        backend="native",
    )
    assert first.prime_count == len(primes)
    assert first.thread_count == len(primes)
    for expected, actual, repeated in zip(
        serial,
        first.results,
        second.results,
        strict=True,
    ):
        assert actual.rank == expected.rank
        assert actual.elimination_steps == expected.elimination_steps
        assert actual.peeled_pivots == expected.peeled_pivots
        assert actual.residual_nonzeros == expected.residual_nonzeros
        np.testing.assert_array_equal(
            actual.pivot_rows,
            expected.pivot_rows,
        )
        np.testing.assert_array_equal(
            actual.pivot_columns,
            expected.pivot_columns,
        )
        np.testing.assert_array_equal(
            actual.pivot_rows,
            repeated.pivot_rows,
        )
        np.testing.assert_array_equal(
            actual.pivot_columns,
            repeated.pivot_columns,
        )


def test_multi_prime_reference_matches_individual_calls() -> None:
    matrix = np.asarray(
        [
            [1, -1, 0, 0],
            [0, 2, -2, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 7],
        ],
        dtype=object,
    )
    offsets, columns, values = csr_from_dense(matrix)
    primes = (2, 3, 101)
    batch = sparse_rank_mod_u32_batch(
        offsets,
        columns,
        values,
        column_count=4,
        primes=primes,
        backend="reference",
    )
    assert tuple(result.rank for result in batch.results) == tuple(
        sparse_rank_mod_u32(
            offsets,
            columns,
            values,
            column_count=4,
            prime=prime,
            backend="reference",
        ).rank
        for prime in primes
    )


@pytest.mark.parametrize(
    "backend",
    ["reference"] + (["native"] if NATIVE_AVAILABLE else []),
)
def test_early_target_stops_with_valid_partial_witness(backend: str) -> None:
    matrix = np.asarray(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 1, 2, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=object,
    )
    offsets, columns, values = csr_from_dense(matrix)
    result = sparse_rank_mod_u32(
        offsets,
        columns,
        values,
        column_count=5,
        prime=101,
        target_rank=3,
        backend=backend,
    )
    assert result.rank == 3
    assert result.target_reached
    assert result.processed_rows < len(matrix)
    witness = matrix[
        np.ix_(
            result.pivot_rows.astype(np.intp),
            result.pivot_columns.astype(np.intp),
        )
    ]
    assert dense_rank_mod(witness, 101) == 3


@pytest.mark.parametrize(
    "backend",
    ["reference"] + (["native"] if NATIVE_AVAILABLE else []),
)
def test_degree_one_peeling_cascades_to_the_empty_core(
    backend: str,
) -> None:
    matrix = np.asarray(
        [
            [2, 0, 0, 0],
            [3, 5, 0, 0],
            [0, 7, 11, 0],
            [0, 0, 13, 17],
        ],
        dtype=object,
    )
    offsets, columns, values = csr_from_dense(matrix)
    result = sparse_rank_mod_u32(
        offsets,
        columns,
        values,
        column_count=4,
        prime=101,
        backend=backend,
    )
    assert result.rank == 4
    assert result.peeled_pivots == 4
    assert result.residual_rows == 0
    assert result.residual_columns == 0
    assert result.residual_nonzeros == 0
    assert result.processed_rows == 4
    assert result.elimination_steps == 0
    assert result.basis_nonzeros == 0
    witness = matrix[
        np.ix_(
            result.pivot_rows.astype(np.intp),
            result.pivot_columns.astype(np.intp),
        )
    ]
    assert dense_rank_mod(witness, 101) == 4


@pytest.mark.parametrize(
    "backend",
    ["reference"] + (["native"] if NATIVE_AVAILABLE else []),
)
def test_degree_one_peeling_preserves_a_dense_residual_core(
    backend: str,
) -> None:
    matrix = np.asarray(
        [
            [7, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 2, 3],
            [0, 1, 4, 9],
        ],
        dtype=object,
    )
    offsets, columns, values = csr_from_dense(matrix)
    result = sparse_rank_mod_u32(
        offsets,
        columns,
        values,
        column_count=4,
        prime=101,
        backend=backend,
    )
    assert result.rank == 4
    assert result.peeled_pivots == 1
    assert result.residual_rows == 3
    assert result.residual_columns == 3
    assert result.residual_nonzeros == 9
    witness = matrix[
        np.ix_(
            result.pivot_rows.astype(np.intp),
            result.pivot_columns.astype(np.intp),
        )
    ]
    assert dense_rank_mod(witness, 101) == 4


@pytest.mark.parametrize(
    "backend",
    ["reference"] + (["native"] if NATIVE_AVAILABLE else []),
)
def test_modular_zeros_drive_peeling_after_reduction(backend: str) -> None:
    matrix = np.asarray(
        [
            [101, 3],
            [5, 7],
        ],
        dtype=object,
    )
    offsets, columns, values = csr_from_dense(
        matrix,
        retain_explicit=True,
    )
    result = sparse_rank_mod_u32(
        offsets,
        columns,
        values,
        column_count=2,
        prime=101,
        backend=backend,
    )
    assert result.rank == 2
    assert result.input_nonzeros == 3
    assert result.peeled_pivots == 2
    assert result.residual_nonzeros == 0
    witness = matrix[
        np.ix_(
            result.pivot_rows.astype(np.intp),
            result.pivot_columns.astype(np.intp),
        )
    ]
    assert dense_rank_mod(witness, 101) == 2


@pytest.mark.parametrize(
    "backend",
    ["reference"] + (["native"] if NATIVE_AVAILABLE else []),
)
def test_explicit_modular_zeros_and_empty_rows_are_ignored(
    backend: str,
) -> None:
    offsets = [0, 2, 2, 4, 5]
    columns = [0, 2, 1, 3, 2]
    values = [101, -202, 1, 102, 0]
    result = sparse_rank_mod_u32(
        offsets,
        columns,
        values,
        column_count=4,
        prime=101,
        backend=backend,
    )
    assert result.rank == 1
    assert result.input_nonzeros == 2
    assert result.active_rows == 1
    assert result.peeled_pivots == 1
    assert result.residual_nonzeros == 0
    assert not result.target_reached


@pytest.mark.parametrize(
    "backend",
    ["reference"] + (["native"] if NATIVE_AVAILABLE else []),
)
def test_initial_empty_row_is_valid_csr(backend: str) -> None:
    result = sparse_rank_mod_u32(
        [0, 0, 1],
        [0],
        [1],
        column_count=1,
        prime=101,
        backend=backend,
    )
    assert result.rank == 1
    np.testing.assert_array_equal(result.pivot_rows, [1])
    np.testing.assert_array_equal(result.pivot_columns, [0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"row_offsets": [0], "column_indices": [], "values": []}, "row"),
        (
            {
                "row_offsets": [1, 1],
                "column_indices": [],
                "values": [],
            },
            "delimit",
        ),
        (
            {
                "row_offsets": [0, 2],
                "column_indices": [0],
                "values": [1],
            },
            "delimit",
        ),
        (
            {
                "row_offsets": [0, 2],
                "column_indices": [1, 1],
                "values": [1, 2],
            },
            "increase",
        ),
        (
            {
                "row_offsets": [0, 1],
                "column_indices": [3],
                "values": [1],
            },
            "range",
        ),
        (
            {
                "row_offsets": [0, 1],
                "column_indices": [0],
                "values": [1.5],
            },
            "integers",
        ),
    ],
)
def test_sparse_rank_rejects_malformed_csr(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sparse_rank_mod_u32(
            **kwargs,
            column_count=3,
            prime=101,
            backend="reference",
        )


@pytest.mark.parametrize("prime", [0, 1, 4, 341, 4_294_967_295])
def test_sparse_rank_rejects_nonprime_modulus(prime: int) -> None:
    with pytest.raises(ValueError, match="prime"):
        sparse_rank_mod_u32(
            [0, 1],
            [0],
            [1],
            column_count=1,
            prime=prime,
            backend="reference",
        )


def test_sparse_rank_rejects_invalid_options() -> None:
    with pytest.raises(ValueError, match="column_count"):
        sparse_rank_mod_u32(
            [0, 0],
            [],
            [],
            column_count=0,
            prime=2,
        )
    with pytest.raises(ValueError, match="target_rank"):
        sparse_rank_mod_u32(
            [0, 0],
            [],
            [],
            column_count=1,
            prime=2,
            target_rank=2,
        )
    with pytest.raises(ValueError, match="backend"):
        sparse_rank_mod_u32(
            [0, 0],
            [],
            [],
            column_count=1,
            prime=2,
            backend="mystery",
        )
    with pytest.raises(ValueError, match="primes"):
        sparse_rank_mod_u32_batch(
            [0, 0],
            [],
            [],
            column_count=1,
            primes=[],
        )
    with pytest.raises(ValueError, match="threads"):
        sparse_rank_mod_u32_batch(
            [0, 0],
            [],
            [],
            column_count=1,
            primes=[2],
            threads=-1,
        )
