from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._native import (
    NativeUnavailable,
    taylor_coefficients_native,
    taylor_evaluate_native,
)


Backend = Literal["auto", "native", "reference"]


@dataclass(frozen=True)
class TaylorCoefficientResult:
    coefficients: NDArray[np.complex128]
    sample_count: int
    order_count: int
    elapsed_seconds: float
    backend: str


@dataclass(frozen=True)
class TaylorEvaluationResult:
    values: NDArray[np.complex128]
    log_moments: NDArray[np.complex128]
    sample_count: int
    order_count: int
    elapsed_seconds: float
    backend: str


def _complex_vector(values: ArrayLike, name: str) -> NDArray[np.complex128]:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    return np.ascontiguousarray(array)


def taylor_coefficients(
    base: ArrayLike,
    logarithms: ArrayLike,
    *,
    maximum_order: int = 4,
    chunk_size: int = 1 << 16,
    threads: int = 1,
    backend: Backend = "auto",
) -> TaylorCoefficientResult:
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    base_array = _complex_vector(base, "base")
    log_array = np.asarray(logarithms, dtype=np.float64)
    if log_array.ndim != 1 or len(log_array) != len(base_array):
        raise ValueError("logarithms must match the base vector")
    log_array = np.ascontiguousarray(log_array)
    if maximum_order < 0 or chunk_size < 1 or threads < 0:
        raise ValueError("invalid Taylor order, chunk size, or thread count")
    if backend in {"auto", "native"}:
        try:
            coefficients, stats = taylor_coefficients_native(
                base_array,
                log_array,
                maximum_order=maximum_order,
                chunk_size=chunk_size,
                threads=threads,
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return TaylorCoefficientResult(
                coefficients=coefficients,
                sample_count=int(stats.sample_count),
                order_count=int(stats.order_count),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )

    started = time.perf_counter()
    coefficients = np.empty(
        (maximum_order + 1, len(base_array)),
        dtype=np.complex128,
    )
    scale = np.ones(len(base_array), dtype=np.float64)
    for order in range(maximum_order + 1):
        coefficients[order] = base_array * scale
        scale *= log_array / (order + 1)
    return TaylorCoefficientResult(
        coefficients=coefficients,
        sample_count=len(base_array),
        order_count=maximum_order + 1,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def evaluate_taylor_basis(
    basis: ArrayLike,
    delta: ArrayLike,
    *,
    chunk_size: int = 1 << 16,
    threads: int = 1,
    backend: Backend = "auto",
) -> TaylorEvaluationResult:
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    basis_array = np.asarray(basis, dtype=np.complex128)
    delta_array = _complex_vector(delta, "delta")
    if (
        basis_array.ndim != 2
        or basis_array.shape[0] == 0
        or basis_array.shape[1] != len(delta_array)
    ):
        raise ValueError("basis must have shape (orders, samples)")
    basis_array = np.ascontiguousarray(basis_array)
    if chunk_size < 1 or threads < 0:
        raise ValueError("invalid chunk size or thread count")
    if backend in {"auto", "native"}:
        try:
            values, log_moments, stats = taylor_evaluate_native(
                basis_array,
                delta_array,
                chunk_size=chunk_size,
                threads=threads,
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return TaylorEvaluationResult(
                values=values,
                log_moments=log_moments,
                sample_count=int(stats.sample_count),
                order_count=int(stats.order_count),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )

    started = time.perf_counter()
    values = basis_array[0].copy()
    log_moments = np.zeros(len(delta_array), dtype=np.complex128)
    power = np.ones(len(delta_array), dtype=np.complex128)
    minus_delta = -delta_array
    for order in range(1, basis_array.shape[0]):
        log_moments += order * basis_array[order] * power
        power *= minus_delta
        values += basis_array[order] * power
    return TaylorEvaluationResult(
        values=values,
        log_moments=log_moments,
        sample_count=len(delta_array),
        order_count=basis_array.shape[0],
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )
