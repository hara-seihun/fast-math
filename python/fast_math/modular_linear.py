"""Retained exact dense linear systems over uint32 prime fields."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._field import field_array_u32, prime_u32
from ._modular_native import (
    NativeModularLinearSystem,
    modular_linear_native_available,
)
from .modular import ModularBackend


@dataclass(frozen=True)
class ModularSolveBatch:
    solutions: NDArray[np.uint32]
    consistent: NDArray[np.bool_]
    inconsistency_rows: NDArray[np.int64]
    elapsed_seconds: float
    backend: str


@dataclass(frozen=True)
class _LinearStructure:
    reduced_row_echelon: NDArray[np.uint32]
    pivot_columns: NDArray[np.uint32]
    solution_operator: NDArray[np.uint32]
    right_nullspace: NDArray[np.uint32]
    left_nullspace: NDArray[np.uint32]


def _factor_linear_system_reference(
    matrix: NDArray[np.uint32],
    prime: int,
) -> _LinearStructure:
    row_count, column_count = matrix.shape
    reduced = [[int(value) for value in row] for row in matrix]
    transformation = [
        [int(row == column) for column in range(row_count)]
        for row in range(row_count)
    ]
    pivots: list[int] = []
    for column in range(column_count):
        pivot_row = len(pivots)
        if pivot_row == row_count:
            break
        pivot = next(
            (
                row
                for row in range(pivot_row, row_count)
                if reduced[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        reduced[pivot_row], reduced[pivot] = reduced[pivot], reduced[pivot_row]
        transformation[pivot_row], transformation[pivot] = (
            transformation[pivot],
            transformation[pivot_row],
        )
        inverse = pow(reduced[pivot_row][column], prime - 2, prime)
        reduced[pivot_row] = [
            value * inverse % prime for value in reduced[pivot_row]
        ]
        transformation[pivot_row] = [
            value * inverse % prime for value in transformation[pivot_row]
        ]
        for row in range(row_count):
            if row == pivot_row:
                continue
            factor = reduced[row][column]
            if factor == 0:
                continue
            reduced[row] = [
                (left - factor * right) % prime
                for left, right in zip(reduced[row], reduced[pivot_row])
            ]
            transformation[row] = [
                (left - factor * right) % prime
                for left, right in zip(
                    transformation[row], transformation[pivot_row]
                )
            ]
        pivots.append(column)
    rank = len(pivots)
    solution_operator = np.zeros(
        (column_count, row_count), dtype=np.uint32
    )
    for row, column in enumerate(pivots):
        solution_operator[column] = transformation[row]
    pivot_set = set(pivots)
    free_columns = [
        column for column in range(column_count) if column not in pivot_set
    ]
    right_nullspace = np.zeros(
        (len(free_columns), column_count), dtype=np.uint32
    )
    for basis_row, free_column in enumerate(free_columns):
        right_nullspace[basis_row, free_column] = 1 % prime
        for row, pivot_column in enumerate(pivots):
            value = reduced[row][free_column]
            right_nullspace[basis_row, pivot_column] = (
                0 if value == 0 else prime - value
            )
    return _LinearStructure(
        reduced_row_echelon=np.asarray(reduced, dtype=np.uint32),
        pivot_columns=np.asarray(pivots, dtype=np.uint32),
        solution_operator=solution_operator,
        right_nullspace=right_nullspace,
        left_nullspace=np.asarray(
            transformation[rank:], dtype=np.uint32
        ).reshape(row_count - rank, row_count),
    )


def _dot_reference(
    left: NDArray[np.uint32],
    right: NDArray[np.uint32],
    prime: int,
) -> int:
    return sum(
        int(left[index]) * int(right[index]) for index in range(len(left))
    ) % prime


def _solve_reference(
    solution_operator: NDArray[np.uint32],
    left_nullspace: NDArray[np.uint32],
    right_hand_sides: NDArray[np.uint32],
    prime: int,
) -> tuple[NDArray[np.uint32], NDArray[np.int64]]:
    solutions = np.zeros(
        (len(right_hand_sides), len(solution_operator)), dtype=np.uint32
    )
    inconsistency_rows = np.full(len(right_hand_sides), -1, dtype=np.int64)
    for batch, right in enumerate(right_hand_sides):
        for row, obstruction in enumerate(left_nullspace):
            if _dot_reference(obstruction, right, prime):
                inconsistency_rows[batch] = row
                break
        if inconsistency_rows[batch] >= 0:
            continue
        for column, operator_row in enumerate(solution_operator):
            solutions[batch, column] = _dot_reference(
                operator_row, right, prime
            )
    return solutions, inconsistency_rows


class ModularLinearSystemPlan:
    """Retain a canonical RREF and exact solve/certificate operators."""

    def __init__(self, matrix: ArrayLike, *, prime: int) -> None:
        self.prime = prime_u32(prime)
        self._matrix = field_array_u32(
            matrix, self.prime, dimensions=2, own=True
        )
        if self._matrix.shape[0] == 0 or self._matrix.shape[1] == 0:
            raise ValueError("linear-system matrix must have nonzero shape")
        self._native: NativeModularLinearSystem | None = None
        self._hip = None
        self._closed = False
        started = perf_counter()
        if modular_linear_native_available():
            self._native = NativeModularLinearSystem(
                self._matrix, prime=self.prime
            )
            structure = _LinearStructure(
                reduced_row_echelon=self._native.reduced_row_echelon,
                pivot_columns=self._native.pivot_columns,
                solution_operator=self._native.solution_operator,
                right_nullspace=self._native.right_nullspace,
                left_nullspace=self._native.left_nullspace,
            )
            self.setup_elapsed_seconds = float(
                self._native.setup_stats.elapsed_seconds
            )
        else:
            structure = _factor_linear_system_reference(
                self._matrix, self.prime
            )
            self.setup_elapsed_seconds = perf_counter() - started
        self._reduced_row_echelon = structure.reduced_row_echelon
        self._pivot_columns = structure.pivot_columns
        self._solution_operator = structure.solution_operator
        self._right_nullspace = structure.right_nullspace
        self._left_nullspace = structure.left_nullspace
        for value in (
            self._reduced_row_echelon,
            self._pivot_columns,
            self._solution_operator,
            self._right_nullspace,
            self._left_nullspace,
        ):
            value.flags.writeable = False

    @property
    def matrix(self) -> NDArray[np.uint32]:
        return self._matrix

    @property
    def row_count(self) -> int:
        return self._matrix.shape[0]

    @property
    def column_count(self) -> int:
        return self._matrix.shape[1]

    @property
    def rank(self) -> int:
        return len(self._pivot_columns)

    @property
    def reduced_row_echelon(self) -> NDArray[np.uint32]:
        return self._reduced_row_echelon

    @property
    def pivot_columns(self) -> NDArray[np.uint32]:
        return self._pivot_columns

    @property
    def free_columns(self) -> NDArray[np.uint32]:
        mask = np.ones(self.column_count, dtype=np.bool_)
        mask[self._pivot_columns] = False
        result = np.flatnonzero(mask).astype(np.uint32)
        result.flags.writeable = False
        return result

    @property
    def solution_operator(self) -> NDArray[np.uint32]:
        return self._solution_operator

    @property
    def row_transform(self) -> NDArray[np.uint32]:
        result = np.concatenate(
            (
                self._solution_operator[self._pivot_columns],
                self._left_nullspace,
            )
        )
        result.flags.writeable = False
        return result

    @property
    def right_nullspace(self) -> NDArray[np.uint32]:
        return self._right_nullspace

    @property
    def left_nullspace(self) -> NDArray[np.uint32]:
        return self._left_nullspace

    @property
    def inverse(self) -> NDArray[np.uint32]:
        if self.row_count != self.column_count or self.rank != self.row_count:
            raise ValueError("only a nonsingular square system has an inverse")
        return self._solution_operator

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("modular linear system plan is closed")

    def _prepare_right_hand_sides(
        self, values: ArrayLike
    ) -> tuple[NDArray[np.uint32], tuple[int, ...]]:
        raw = np.asarray(values)
        if raw.ndim == 0 or raw.shape[-1] != self.row_count:
            raise ValueError(
                "right-hand sides must have the system row count on the last axis"
            )
        prepared = field_array_u32(
            raw, self.prime, dimensions=raw.ndim, own=False
        )
        return prepared.reshape(-1, self.row_count), raw.shape[:-1]

    def _hip_plan(self):
        if self._hip is None:
            from .hip import ModularLinearSystemHipPlan

            self._hip = ModularLinearSystemHipPlan(
                self._solution_operator,
                self._pivot_columns,
                self._left_nullspace,
                prime=self.prime,
            )
        return self._hip

    def solve(
        self,
        right_hand_sides: ArrayLike,
        *,
        threads: int = 1,
        backend: ModularBackend = "auto",
    ) -> ModularSolveBatch:
        self._check_open()
        if backend not in {"auto", "reference", "native", "hip"}:
            raise ValueError(f"unknown modular backend: {backend}")
        if not isinstance(threads, Integral) or int(threads) < 1:
            raise ValueError("threads must be positive")
        prepared, batch_shape = self._prepare_right_hand_sides(
            right_hand_sides
        )
        selected = backend
        if selected == "auto":
            obstruction_count = self.row_count - self.rank
            work = (
                len(prepared)
                * self.row_count
                * (self.column_count + obstruction_count)
            )
            hip_threshold = (
                10_000_000 if self._hip is not None else 100_000_000
            )
            if work >= hip_threshold:
                from .hip import hip_modular_linear_available

                selected = (
                    "hip"
                    if hip_modular_linear_available()
                    else "native"
                    if self._native is not None
                    else "reference"
                )
            else:
                selected = "native" if self._native is not None else "reference"
        if selected == "reference":
            started = perf_counter()
            solutions, inconsistency_rows = _solve_reference(
                self._solution_operator,
                self._left_nullspace,
                prepared,
                self.prime,
            )
            elapsed = perf_counter() - started
        elif selected == "native":
            if self._native is None:
                raise RuntimeError("native modular linear systems are unavailable")
            solutions, inconsistency_rows, stats = self._native.solve(
                prepared, threads=int(threads)
            )
            elapsed = float(stats.elapsed_seconds)
        else:
            solutions, inconsistency_rows, elapsed = self._hip_plan().solve(
                prepared
            )
        return ModularSolveBatch(
            solutions=solutions.reshape(batch_shape + (self.column_count,)),
            consistent=(inconsistency_rows < 0).reshape(batch_shape),
            inconsistency_rows=inconsistency_rows.reshape(batch_shape),
            elapsed_seconds=elapsed,
            backend=selected,
        )

    def verify(
        self,
        right_hand_sides: ArrayLike,
        result: ModularSolveBatch,
    ) -> NDArray[np.bool_]:
        self._check_open()
        prepared, batch_shape = self._prepare_right_hand_sides(
            right_hand_sides
        )
        solutions = np.asarray(result.solutions)
        consistent = np.asarray(result.consistent)
        inconsistency_rows = np.asarray(result.inconsistency_rows)
        if (
            solutions.shape != batch_shape + (self.column_count,)
            or consistent.shape != batch_shape
            or inconsistency_rows.shape != batch_shape
        ):
            raise ValueError("solve result shape does not match right-hand sides")
        flat_solutions = field_array_u32(
            solutions, self.prime, dimensions=solutions.ndim, own=False
        ).reshape(-1, self.column_count)
        flat_consistent = consistent.reshape(-1)
        flat_inconsistency = inconsistency_rows.reshape(-1)
        verified = np.zeros(len(prepared), dtype=np.bool_)
        for batch, right in enumerate(prepared):
            if bool(flat_consistent[batch]):
                if flat_inconsistency[batch] != -1:
                    continue
                verified[batch] = all(
                    _dot_reference(row, flat_solutions[batch], self.prime)
                    == int(right[index])
                    for index, row in enumerate(self._matrix)
                )
                continue
            witness_index = int(flat_inconsistency[batch])
            if not 0 <= witness_index < len(self._left_nullspace):
                continue
            witness = self._left_nullspace[witness_index]
            annihilates_matrix = all(
                sum(
                    int(witness[row]) * int(self._matrix[row, column])
                    for row in range(self.row_count)
                )
                % self.prime
                == 0
                for column in range(self.column_count)
            )
            verified[batch] = annihilates_matrix and bool(
                _dot_reference(witness, right, self.prime)
            )
        return verified.reshape(batch_shape)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._native is not None:
            self._native.close()
        if self._hip is not None:
            self._hip.close()

    def __enter__(self) -> "ModularLinearSystemPlan":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["ModularLinearSystemPlan", "ModularSolveBatch"]
