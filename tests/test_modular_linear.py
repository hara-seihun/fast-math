from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from fast_math.hip import hip_modular_linear_available
from fast_math.modular_linear import (
    ModularLinearSystemPlan,
    _factor_linear_system_reference,
)


def exact_product(
    left: np.ndarray, right: np.ndarray, prime: int
) -> np.ndarray:
    output = np.empty((len(left), right.shape[1]), dtype=np.uint32)
    for row in range(len(left)):
        for column in range(right.shape[1]):
            output[row, column] = sum(
                int(left[row, index]) * int(right[index, column])
                for index in range(left.shape[1])
            ) % prime
    return output


def test_rank_nullspaces_and_solve_certificates() -> None:
    matrix = np.asarray([[1, 2, 3], [2, 4, 6]], dtype=np.uint32)
    right_hand_sides = np.asarray(
        [[1, 2], [1, 3], [0, 0]], dtype=np.uint32
    )
    with ModularLinearSystemPlan(matrix, prime=7) as plan:
        assert plan.rank == 1
        np.testing.assert_array_equal(
            plan.reduced_row_echelon, [[1, 2, 3], [0, 0, 0]]
        )
        np.testing.assert_array_equal(plan.pivot_columns, [0])
        np.testing.assert_array_equal(plan.free_columns, [1, 2])
        np.testing.assert_array_equal(
            plan.right_nullspace, [[5, 1, 0], [4, 0, 1]]
        )
        np.testing.assert_array_equal(plan.left_nullspace, [[5, 1]])
        np.testing.assert_array_equal(plan.row_transform, [[1, 0], [5, 1]])
        np.testing.assert_array_equal(
            exact_product(plan.row_transform, matrix, 7),
            plan.reduced_row_echelon,
        )
        result = plan.solve(
            right_hand_sides, backend="native", threads=2
        )
        np.testing.assert_array_equal(result.consistent, [True, False, True])
        np.testing.assert_array_equal(result.inconsistency_rows, [-1, 0, -1])
        np.testing.assert_array_equal(
            result.solutions, [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        )
        np.testing.assert_array_equal(plan.verify(right_hand_sides, result), True)


@pytest.mark.parametrize("prime", [2, 3, 65521, 1_000_000_007, 4_294_967_291])
def test_native_structure_matches_independent_reference(prime: int) -> None:
    rng = np.random.default_rng(prime)
    matrix = rng.integers(0, prime, size=(7, 11), dtype=np.uint64)
    matrix[6] = matrix[1]
    matrix = matrix.astype(np.uint32)
    reference = _factor_linear_system_reference(matrix, prime)
    with ModularLinearSystemPlan(matrix, prime=prime) as plan:
        np.testing.assert_array_equal(
            plan.reduced_row_echelon, reference.reduced_row_echelon
        )
        np.testing.assert_array_equal(
            plan.pivot_columns, reference.pivot_columns
        )
        np.testing.assert_array_equal(
            plan.solution_operator, reference.solution_operator
        )
        np.testing.assert_array_equal(
            plan.right_nullspace, reference.right_nullspace
        )
        np.testing.assert_array_equal(
            plan.left_nullspace, reference.left_nullspace
        )
        np.testing.assert_array_equal(
            exact_product(plan.row_transform, matrix, prime),
            plan.reduced_row_echelon,
        )
        if len(plan.right_nullspace):
            zeros = exact_product(
                matrix, plan.right_nullspace.T, prime
            )
            assert not np.any(zeros)
        if len(plan.left_nullspace):
            zeros = exact_product(plan.left_nullspace, matrix, prime)
            assert not np.any(zeros)


def test_zero_system_and_empty_batch() -> None:
    matrix = np.zeros((3, 2), dtype=np.uint32)
    with ModularLinearSystemPlan(matrix, prime=2) as plan:
        assert plan.rank == 0
        np.testing.assert_array_equal(
            plan.right_nullspace, np.eye(2, dtype=np.uint32)
        )
        np.testing.assert_array_equal(
            plan.left_nullspace, np.eye(3, dtype=np.uint32)
        )
        result = plan.solve(
            [[0, 0, 0], [1, 0, 0]], backend="native"
        )
        np.testing.assert_array_equal(result.consistent, [True, False])
        np.testing.assert_array_equal(result.inconsistency_rows, [-1, 0])
        empty = plan.solve(np.empty((0, 3), dtype=np.uint32), backend="native")
        assert empty.solutions.shape == (0, 2)


def test_batched_shapes_inverse_and_reference_native_parity() -> None:
    prime = 65521
    rng = np.random.default_rng(71)
    matrix = rng.integers(0, prime, size=(8, 8), dtype=np.uint32)
    right_hand_sides = rng.integers(
        0, prime, size=(3, 5, 8), dtype=np.uint32
    )
    with ModularLinearSystemPlan(matrix, prime=prime) as plan:
        assert plan.rank == 8
        identity = exact_product(matrix, plan.inverse, prime)
        np.testing.assert_array_equal(identity, np.eye(8, dtype=np.uint32))
        reference = plan.solve(right_hand_sides, backend="reference")
        native = plan.solve(
            right_hand_sides, backend="native", threads=3
        )
        assert native.solutions.shape == (3, 5, 8)
        np.testing.assert_array_equal(native.solutions, reference.solutions)
        np.testing.assert_array_equal(native.consistent, True)
        np.testing.assert_array_equal(plan.verify(right_hand_sides, native), True)


@pytest.mark.skipif(
    not hip_modular_linear_available(),
    reason="HIP modular linear-system backend unavailable",
)
@pytest.mark.parametrize(
    ("prime", "row_count", "column_count"),
    [(1_000_000_007, 32, 20), (4_294_967_291, 10, 7)],
)
def test_hip_solve_matches_native_at_hostile_primes(
    prime: int, row_count: int, column_count: int
) -> None:
    rng = np.random.default_rng(83 + row_count)
    matrix = rng.integers(
        0, prime, size=(row_count, column_count), dtype=np.uint64
    )
    matrix[column_count:] = matrix[: row_count - column_count]
    matrix = matrix.astype(np.uint32)
    right_hand_sides = rng.integers(
        0, prime, size=(257, row_count), dtype=np.uint64
    ).astype(np.uint32)
    planted = rng.integers(
        0, prime, size=(column_count, 4), dtype=np.uint64
    ).astype(np.uint32)
    right_hand_sides[:4] = exact_product(matrix, planted, prime).T
    with ModularLinearSystemPlan(matrix, prime=prime) as plan:
        native = plan.solve(
            right_hand_sides, backend="native", threads=4
        )
        first = plan.solve(right_hand_sides, backend="hip")
        second = plan.solve(right_hand_sides, backend="hip")
        for result in (first, second):
            np.testing.assert_array_equal(result.solutions, native.solutions)
            np.testing.assert_array_equal(result.consistent, native.consistent)
            np.testing.assert_array_equal(
                result.inconsistency_rows, native.inconsistency_rows
            )
            np.testing.assert_array_equal(
                plan.verify(right_hand_sides, result), True
            )


def test_concurrent_native_batches_share_retained_flint_safely() -> None:
    prime = 65521
    rng = np.random.default_rng(97)
    matrix = rng.integers(0, prime, size=(8, 8), dtype=np.uint32)
    batches = [
        rng.integers(0, prime, size=(257, 8), dtype=np.uint32)
        for _ in range(2)
    ]
    with ModularLinearSystemPlan(matrix, prime=prime) as plan:
        expected = [
            plan.solve(batch, backend="reference") for batch in batches
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(plan.solve, batch, backend="native")
                for batch in batches
            ]
            actual = [future.result() for future in futures]
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result.solutions, reference.solutions)
        np.testing.assert_array_equal(result.consistent, reference.consistent)


def test_linear_system_storage_validation_and_close() -> None:
    matrix = np.asarray([[1, 2], [3, 4]], dtype=np.uint32)
    plan = ModularLinearSystemPlan(matrix, prime=7)
    matrix.fill(0)
    np.testing.assert_array_equal(plan.matrix, [[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="last axis"):
        plan.solve([[1, 2, 3]])
    with pytest.raises(ValueError, match="outside"):
        plan.solve([[1, 7]])
    plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        plan.solve([[1, 2]])
