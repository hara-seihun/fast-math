from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import numpy as np

from fast_math._native import NativeUnavailable, dirichlet_inverse_native


Backend = Literal["auto", "native", "reference"]


@dataclass(frozen=True)
class InverseResult:
    coefficients: np.ndarray
    update_count: int
    elapsed_seconds: float
    backend: str


def _reference(source: np.ndarray) -> InverseResult:
    started = time.perf_counter()
    coefficients = np.zeros(len(source) + 1, dtype=np.float64)
    accumulated = np.zeros(len(source) + 1, dtype=np.float64)
    coefficients[1] = 1.0
    update_count = 0
    for divisor in range(1, len(source) + 1):
        if divisor > 1:
            coefficients[divisor] = -accumulated[divisor]
        maximum = len(source) // divisor
        if maximum < 2:
            continue
        indices = divisor * np.arange(
            2, maximum + 1, dtype=np.int64
        )
        accumulated[indices] += (
            coefficients[divisor] * source[1:maximum]
        )
        update_count += maximum - 1
    return InverseResult(
        coefficients=coefficients,
        update_count=update_count,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def dirichlet_inverse(
    source,
    *,
    backend: Backend = "auto",
) -> InverseResult:
    """Deterministic Dirichlet inverse coefficients of a finite source sequence."""
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    source_array = np.asarray(source, dtype=np.float64)
    if source_array.ndim != 1 or len(source_array) == 0:
        raise ValueError("source must be a nonempty vector")
    source_array = np.ascontiguousarray(source_array)
    if backend in {"auto", "native"}:
        try:
            coefficients, stats = dirichlet_inverse_native(source_array)
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return InverseResult(
                coefficients=coefficients,
                update_count=int(stats.update_count),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    return _reference(source_array)


def truncated_inverse(
    limit: int,
    heat_time: float,
    *,
    backend: Backend = "auto",
) -> InverseResult:
    """Dirichlet inverse of the heat-time source truncated at ``limit``."""
    if limit < 1:
        raise ValueError("limit must be positive")
    logarithms = np.log(np.arange(1, limit + 1, dtype=np.float64))
    source = np.exp((heat_time / 4.0) * logarithms * logarithms)
    return dirichlet_inverse(source, backend=backend)
