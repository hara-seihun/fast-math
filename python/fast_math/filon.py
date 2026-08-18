"""Compact Chebyshev-Filon contractions for long autocorrelations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._native import (
    NativeUnavailable,
    filon_chebyshev_inner_product_native,
)


Backend = Literal["auto", "native", "reference"]


@dataclass(frozen=True)
class FilonInnerProductResult:
    value: complex
    correlation_count: int
    output_count: int
    exact_count: int
    tail_count: int
    term_count: int
    chunk_count: int
    thread_count: int
    elapsed_seconds: float
    backend: str


@lru_cache(maxsize=32)
def _endpoint_derivative_matrices(
    degree: int,
    term_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if degree == 0:
        positive = np.zeros((term_count, 1), dtype=np.float64)
        negative = np.zeros_like(positive)
        positive[0, 0] = 1.0
        negative[0, 0] = 1.0
    else:
        indices = np.arange(degree + 1, dtype=np.int64)
        nodes = np.cos(np.pi * indices / degree)
        scales = np.ones(degree + 1)
        scales[[0, -1]] = 2.0
        scales *= np.where(indices % 2 == 0, 1.0, -1.0)
        differences = nodes[:, None] - nodes[None, :]
        differentiation = np.divide(
            scales[:, None],
            scales[None, :] * differences,
            out=np.zeros_like(differences),
            where=differences != 0.0,
        )
        differentiation[np.diag_indices_from(differentiation)] = (
            -differentiation.sum(axis=1)
        )
        positive = np.empty(
            (term_count, degree + 1),
            dtype=np.float64,
        )
        negative = np.empty_like(positive)
        power = np.eye(degree + 1)
        for order in range(term_count):
            positive[order] = power[0]
            negative[order] = power[-1]
            power = differentiation @ power
    positive.setflags(write=False)
    negative.setflags(write=False)
    return positive, negative


def chebyshev_lobatto_endpoint_derivatives(
    degree: int,
    node_index: int,
    *,
    term_count: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return endpoint derivatives of one Lobatto cardinal polynomial."""
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if not 0 <= node_index <= degree:
        raise ValueError("node_index must lie in [0, degree]")
    if term_count < 1:
        raise ValueError("term_count must be positive")
    positive, negative = _endpoint_derivative_matrices(
        degree,
        term_count,
    )
    return positive[:, node_index].copy(), negative[:, node_index].copy()


def _prepare_complex_vector(
    values: ArrayLike,
    name: str,
) -> NDArray[np.complex128]:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    return np.ascontiguousarray(array)


def _reference_tail_weights(
    lags: NDArray[np.float64],
    positive: NDArray[np.float64],
    negative: NDArray[np.float64],
    *,
    eta: float,
    length: float,
) -> NDArray[np.complex128]:
    phase = lags * eta / 2.0
    exp_positive = np.exp(1j * phase)
    exp_negative = np.conjugate(exp_positive)
    inverse_it = 1.0 / (1j * phase)
    factor = inverse_it.copy()
    weights = np.zeros(len(lags), dtype=np.complex128)
    sign = 1.0
    for order in range(len(positive)):
        weights += (
            sign
            * (
                positive[order] * exp_positive
                - negative[order] * exp_negative
            )
            * factor
        )
        sign = -sign
        factor *= inverse_it
    return length / 2.0 * weights


def _reference_inner_product(
    correlation: NDArray[np.complex128],
    exact_weights: NDArray[np.complex128],
    positive: NDArray[np.float64],
    negative: NDArray[np.float64],
    *,
    output_count: int,
    eta: float,
    length: float,
    conjugate_kernel: bool,
    chunk_size: int,
) -> complex:
    result = 0.0j
    exact = (
        np.conjugate(exact_weights)
        if conjugate_kernel
        else exact_weights
    )
    result += np.dot(correlation[: len(exact)], exact)
    if len(exact) > 1:
        result += np.dot(
            correlation[-1 : -len(exact) : -1],
            np.conjugate(exact[1:]),
        )
    for begin in range(len(exact), output_count, chunk_size):
        end = min(output_count, begin + chunk_size)
        lags = np.arange(begin, end, dtype=np.float64)
        weights = _reference_tail_weights(
            lags,
            positive,
            negative,
            eta=eta,
            length=length,
        )
        if conjugate_kernel:
            weights = np.conjugate(weights)
        result += np.dot(correlation[begin:end], weights)
        result += np.dot(
            correlation[-begin : -end : -1],
            np.conjugate(weights),
        )
    return complex(result)


def filon_chebyshev_inner_product(
    correlation: ArrayLike,
    exact_weights: ArrayLike,
    *,
    degree: int,
    node_index: int,
    output_count: int,
    eta: float,
    length: float,
    term_count: int = 10,
    conjugate_kernel: bool = False,
    chunk_size: int = 1 << 16,
    threads: int = 1,
    backend: Backend = "auto",
) -> FilonInnerProductResult:
    """Contract an autocorrelation with an exact-prefix Filon row."""
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    if output_count < 1:
        raise ValueError("output_count must be positive")
    if output_count > (np.iinfo(np.intp).max + 1) // 2:
        raise ValueError("output_count is too large")
    if not np.isfinite(eta) or eta == 0.0 or not np.isfinite(length):
        raise ValueError("eta and length must be finite with nonzero eta")
    if chunk_size < 1 or threads < 0:
        raise ValueError("invalid chunk size or thread count")
    correlation_array = _prepare_complex_vector(
        correlation,
        "correlation",
    )
    exact_array = _prepare_complex_vector(
        exact_weights,
        "exact_weights",
    )
    if len(exact_array) > output_count:
        raise ValueError("exact_weights exceeds output_count")
    if len(correlation_array) < 2 * output_count - 1:
        raise ValueError(
            "correlation must contain both lag directions"
        )
    positive, negative = chebyshev_lobatto_endpoint_derivatives(
        degree,
        node_index,
        term_count=term_count,
    )

    if backend in {"auto", "native"}:
        try:
            value, stats = filon_chebyshev_inner_product_native(
                correlation_array,
                exact_array,
                positive,
                negative,
                output_count=output_count,
                eta=eta,
                length=length,
                conjugate_kernel=conjugate_kernel,
                chunk_size=chunk_size,
                threads=threads,
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return FilonInnerProductResult(
                value=value,
                correlation_count=int(stats.correlation_count),
                output_count=int(stats.output_count),
                exact_count=int(stats.exact_count),
                tail_count=int(stats.tail_count),
                term_count=int(stats.term_count),
                chunk_count=int(stats.chunk_count),
                thread_count=int(stats.thread_count),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )

    started = time.perf_counter()
    value = _reference_inner_product(
        correlation_array,
        exact_array,
        positive,
        negative,
        output_count=output_count,
        eta=eta,
        length=length,
        conjugate_kernel=conjugate_kernel,
        chunk_size=chunk_size,
    )
    return FilonInnerProductResult(
        value=value,
        correlation_count=len(correlation_array),
        output_count=output_count,
        exact_count=len(exact_array),
        tail_count=output_count - len(exact_array),
        term_count=term_count,
        chunk_count=(output_count + chunk_size - 1) // chunk_size,
        thread_count=1,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )
