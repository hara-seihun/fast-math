"""Exact batched arithmetic over uint32 prime fields."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._field import field_array_u32, prime_u32
from ._modular_native import (
    determinants_native,
    modular_native_available,
    polynomial_evaluate_native,
)


ModularBackend = Literal["auto", "reference", "native", "hip"]


@dataclass(frozen=True)
class ModularPolynomialEvaluation:
    values: NDArray[np.uint32]
    derivatives: NDArray[np.uint32] | None
    elapsed_seconds: float
    backend: str


@dataclass(frozen=True)
class ModularDeterminantBatch:
    determinants: NDArray[np.uint32]
    elapsed_seconds: float
    backend: str


def _polynomial_reference(
    coefficients: NDArray[np.uint32],
    points: NDArray[np.uint32],
    prime_value: int,
    derivative: bool,
) -> tuple[NDArray[np.uint32], NDArray[np.uint32] | None]:
    prime = np.uint64(prime_value)
    point_rows = points.astype(np.uint64, copy=False)[np.newaxis, :]
    values = np.broadcast_to(
        coefficients[:, -1, np.newaxis],
        (len(coefficients), len(points)),
    ).astype(np.uint64, copy=True)
    derivatives = np.zeros_like(values) if derivative else None
    for column in range(coefficients.shape[1] - 2, -1, -1):
        if derivatives is not None:
            derivatives = (derivatives * point_rows + values) % prime
        values = (
            values * point_rows
            + coefficients[:, column, np.newaxis].astype(np.uint64)
        ) % prime
    return (
        values.astype(np.uint32, copy=False),
        derivatives.astype(np.uint32, copy=False)
        if derivatives is not None
        else None,
    )


def _determinants_reference(
    matrices: NDArray[np.uint32],
    prime: int,
) -> NDArray[np.uint32]:
    output = np.empty(len(matrices), dtype=np.uint32)
    order = matrices.shape[1]
    for matrix_index, source in enumerate(matrices):
        work = [[int(value) for value in row] for row in source]
        determinant = 1 % prime
        negate = False
        for column in range(order):
            pivot = next(
                (row for row in range(column, order) if work[row][column]),
                None,
            )
            if pivot is None:
                determinant = 0
                break
            if pivot != column:
                work[column], work[pivot] = work[pivot], work[column]
                negate = not negate
            pivot_value = work[column][column]
            determinant = determinant * pivot_value % prime
            inverse = pow(pivot_value, prime - 2, prime)
            for row in range(column + 1, order):
                if work[row][column] == 0:
                    continue
                factor = work[row][column] * inverse % prime
                for entry in range(column + 1, order):
                    work[row][entry] = (
                        work[row][entry] - factor * work[column][entry]
                    ) % prime
        if negate and determinant:
            determinant = prime - determinant
        output[matrix_index] = determinant
    return output


class ModularPolynomialPlan:
    """Retain low-to-high coefficients for repeated finite-field evaluation."""

    def __init__(self, coefficients: ArrayLike, *, prime: int) -> None:
        self.prime = prime_u32(prime)
        self._coefficients = field_array_u32(
            coefficients, self.prime, dimensions=2, own=True
        )
        if self._coefficients.shape[0] == 0 or self._coefficients.shape[1] == 0:
            raise ValueError("coefficients must have nonzero shape")
        self._hip = None
        self._closed = False

    @property
    def coefficients(self) -> NDArray[np.uint32]:
        return self._coefficients

    @property
    def polynomial_count(self) -> int:
        return self._coefficients.shape[0]

    @property
    def coefficient_count(self) -> int:
        return self._coefficients.shape[1]

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("modular polynomial plan is closed")

    def _hip_plan(self):
        if self._hip is None:
            from .hip import ModularPolynomialHipPlan

            self._hip = ModularPolynomialHipPlan(
                self._coefficients, self.prime
            )
        return self._hip

    def evaluate(
        self,
        points: ArrayLike,
        *,
        derivative: bool = False,
        threads: int = 1,
        backend: ModularBackend = "auto",
    ) -> ModularPolynomialEvaluation:
        self._check_open()
        if backend not in {"auto", "reference", "native", "hip"}:
            raise ValueError(f"unknown modular backend: {backend}")
        if not isinstance(threads, Integral) or int(threads) < 1:
            raise ValueError("threads must be positive")
        if not isinstance(derivative, bool):
            raise ValueError("derivative must be Boolean")
        prepared = field_array_u32(
            points, self.prime, dimensions=1, own=False
        )
        selected = backend
        if selected == "auto":
            work = len(prepared) * self.polynomial_count * self.coefficient_count
            hip_threshold = 10_000_000 if self._hip is not None else 100_000_000
            if work >= hip_threshold:
                from .hip import hip_modular_available

                if hip_modular_available():
                    selected = "hip"
                else:
                    selected = (
                        "native" if modular_native_available() else "reference"
                    )
            else:
                selected = (
                    "native" if modular_native_available() else "reference"
                )
        if selected == "reference":
            started = perf_counter()
            values, derivatives = _polynomial_reference(
                self._coefficients, prepared, self.prime, derivative
            )
            elapsed = perf_counter() - started
        elif selected == "native":
            values, derivatives, stats = polynomial_evaluate_native(
                self._coefficients,
                prepared,
                prime=self.prime,
                threads=int(threads),
                with_derivative=derivative,
            )
            elapsed = float(stats.elapsed_seconds)
        else:
            values, derivatives, elapsed = self._hip_plan().evaluate(
                prepared, derivative=derivative
            )
        return ModularPolynomialEvaluation(
            values=values,
            derivatives=derivatives,
            elapsed_seconds=elapsed,
            backend=selected,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._hip is not None:
            self._hip.close()

    def __enter__(self) -> "ModularPolynomialPlan":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class ModularDeterminantPlan:
    """Retain a field and matrix order across dense determinant batches."""

    def __init__(self, order: int, *, prime: int) -> None:
        if not isinstance(order, Integral) or not 1 <= int(order) <= 64:
            raise ValueError("determinant order must be between one and 64")
        self.order = int(order)
        self.prime = prime_u32(prime)
        self._hip = None
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("modular determinant plan is closed")

    def _hip_plan(self):
        if self.order > 32:
            raise ValueError("HIP determinant order is limited to 32")
        if self._hip is None:
            from .hip import ModularDeterminantHipPlan

            self._hip = ModularDeterminantHipPlan(self.order, self.prime)
        return self._hip

    def determinants(
        self,
        matrices: ArrayLike,
        *,
        threads: int = 1,
        backend: ModularBackend = "auto",
    ) -> ModularDeterminantBatch:
        self._check_open()
        if backend not in {"auto", "reference", "native", "hip"}:
            raise ValueError(f"unknown modular backend: {backend}")
        if not isinstance(threads, Integral) or int(threads) < 1:
            raise ValueError("threads must be positive")
        prepared = field_array_u32(
            matrices, self.prime, dimensions=3, own=False
        )
        if prepared.shape[1:] != (self.order, self.order):
            raise ValueError("matrix shape does not match the determinant plan")
        selected = backend
        if selected == "auto":
            work = len(prepared) * self.order * self.order * self.order
            hip_threshold = 2_000_000 if self._hip is not None else 80_000_000
            if self.order <= 32 and work >= hip_threshold:
                from .hip import hip_modular_available

                if hip_modular_available():
                    selected = "hip"
                else:
                    selected = (
                        "native" if modular_native_available() else "reference"
                    )
            else:
                selected = (
                    "native" if modular_native_available() else "reference"
                )
        if selected == "reference":
            started = perf_counter()
            determinants = _determinants_reference(prepared, self.prime)
            elapsed = perf_counter() - started
        elif selected == "native":
            determinants, stats = determinants_native(
                prepared, prime=self.prime, threads=int(threads)
            )
            elapsed = float(stats.elapsed_seconds)
        else:
            determinants, elapsed = self._hip_plan().determinants(prepared)
        return ModularDeterminantBatch(
            determinants=determinants,
            elapsed_seconds=elapsed,
            backend=selected,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._hip is not None:
            self._hip.close()

    def __enter__(self) -> "ModularDeterminantPlan":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def determinants_mod_u32(
    matrices: ArrayLike,
    *,
    prime: int,
    threads: int = 1,
    backend: ModularBackend = "auto",
) -> ModularDeterminantBatch:
    raw = np.asarray(matrices)
    if raw.ndim != 3 or raw.shape[1] != raw.shape[2]:
        raise ValueError("matrices must have shape (count, order, order)")
    with ModularDeterminantPlan(raw.shape[1], prime=prime) as plan:
        return plan.determinants(raw, threads=threads, backend=backend)


__all__ = [
    "ModularBackend",
    "ModularDeterminantBatch",
    "ModularDeterminantPlan",
    "ModularPolynomialEvaluation",
    "ModularPolynomialPlan",
    "determinants_mod_u32",
]
