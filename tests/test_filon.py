from __future__ import annotations

import numpy as np
import pytest

from fast_math import (
    chebyshev_lobatto_endpoint_derivatives,
    filon_chebyshev_inner_product,
)
from lambda_fast import available_backends


NATIVE_AVAILABLE = "native" in available_backends()


def _correlation(output_count: int) -> np.ndarray:
    lags = np.arange(output_count, dtype=np.float64)
    positive = (
        np.exp(-lags / 731.0)
        * np.exp(0.017j * lags)
        * (1.0 + 0.1 * np.cos(0.031 * lags))
    )
    result = np.empty(2 * output_count - 1, dtype=np.complex128)
    result[:output_count] = positive
    result[output_count:] = np.conjugate(positive[1:][::-1])
    return result


def test_endpoint_derivatives_reproduce_polynomial_derivatives() -> None:
    degree = 12
    term_count = 7
    rng = np.random.default_rng(9127)
    coefficients = rng.normal(size=degree + 1)
    polynomial = np.polynomial.Polynomial(coefficients)
    nodes = np.cos(np.pi * np.arange(degree + 1) / degree)
    values = polynomial(nodes)
    positive_matrix = np.empty((term_count, degree + 1))
    negative_matrix = np.empty_like(positive_matrix)
    for node_index in range(degree + 1):
        node_positive, node_negative = (
            chebyshev_lobatto_endpoint_derivatives(
                degree,
                node_index,
                term_count=term_count,
            )
        )
        positive_matrix[:, node_index] = node_positive
        negative_matrix[:, node_index] = node_negative
    for order in range(term_count):
        derivative = polynomial.deriv(order)
        np.testing.assert_allclose(
            positive_matrix[order] @ values,
            derivative(1.0),
            rtol=2e-9,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            negative_matrix[order] @ values,
            derivative(-1.0),
            rtol=2e-9,
            atol=2e-7,
        )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
@pytest.mark.parametrize("conjugate_kernel", [False, True])
def test_native_filon_matches_reference(conjugate_kernel: bool) -> None:
    output_count = 20_003
    exact_count = 257
    correlation = _correlation(output_count)
    exact_lags = np.arange(exact_count, dtype=np.float64)
    exact_weights = (
        np.sinc(exact_lags / 13.0)
        * np.exp(0.03j * exact_lags)
    ).astype(np.complex128)
    common = dict(
        degree=24,
        node_index=7,
        output_count=output_count,
        eta=32.0,
        length=11989041.291992188,
        term_count=10,
        conjugate_kernel=conjugate_kernel,
        chunk_size=251,
    )
    reference = filon_chebyshev_inner_product(
        correlation,
        exact_weights,
        backend="reference",
        **common,
    )
    native = filon_chebyshev_inner_product(
        correlation,
        exact_weights,
        threads=2,
        backend="native",
        **common,
    )
    np.testing.assert_allclose(
        native.value,
        reference.value,
        rtol=3e-13,
        atol=3e-9,
    )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_filon_is_thread_deterministic() -> None:
    output_count = 30_001
    correlation = _correlation(output_count)
    exact_weights = np.exp(
        0.01j * np.arange(513, dtype=np.float64)
    )
    common = dict(
        degree=32,
        node_index=11,
        output_count=output_count,
        eta=16.0,
        length=7.5,
        term_count=9,
        chunk_size=383,
        backend="native",
    )
    serial = filon_chebyshev_inner_product(
        correlation,
        exact_weights,
        threads=1,
        **common,
    )
    parallel = filon_chebyshev_inner_product(
        correlation,
        exact_weights,
        threads=2,
        **common,
    )
    assert serial.value == parallel.value


@pytest.mark.parametrize(
    ("correlation", "exact", "kwargs", "message"),
    [
        (
            np.ones(9, dtype=np.complex128),
            np.ones(2, dtype=np.complex128),
            {"output_count": 6},
            "both lag directions",
        ),
        (
            np.ones(9, dtype=np.complex128),
            np.ones(6, dtype=np.complex128),
            {"output_count": 5},
            "exceeds output_count",
        ),
    ],
)
def test_filon_rejects_invalid_shapes(
    correlation: np.ndarray,
    exact: np.ndarray,
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        filon_chebyshev_inner_product(
            correlation,
            exact,
            degree=4,
            node_index=2,
            eta=8.0,
            length=3.0,
            backend="reference",
            **kwargs,
        )
