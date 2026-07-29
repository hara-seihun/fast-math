from __future__ import annotations

import numpy as np
import pytest

from fast_math import evaluate_taylor_basis, taylor_coefficients
from lambda_fast import available_backends


NATIVE_AVAILABLE = "native" in available_backends()


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_taylor_coefficients_match_reference() -> None:
    rng = np.random.default_rng(7401)
    base = rng.normal(size=10_003) + 1j * rng.normal(size=10_003)
    logarithms = rng.normal(size=len(base))
    reference = taylor_coefficients(
        base,
        logarithms,
        maximum_order=7,
        backend="reference",
    )
    native = taylor_coefficients(
        base,
        logarithms,
        maximum_order=7,
        chunk_size=257,
        threads=5,
        backend="native",
    )
    np.testing.assert_allclose(
        native.coefficients,
        reference.coefficients,
        rtol=3e-15,
        atol=3e-15,
    )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_taylor_evaluation_matches_phase_model_formula() -> None:
    rng = np.random.default_rng(7402)
    basis = (
        rng.normal(size=(5, 20_003))
        + 1j * rng.normal(size=(5, 20_003))
    )
    delta = (
        0.01 * rng.normal(size=basis.shape[1])
        + 0.01j * rng.normal(size=basis.shape[1])
    )
    reference = evaluate_taylor_basis(
        basis,
        delta,
        backend="reference",
    )
    native = evaluate_taylor_basis(
        basis,
        delta,
        chunk_size=1_003,
        threads=5,
        backend="native",
    )
    np.testing.assert_allclose(
        native.values,
        reference.values,
        rtol=3e-15,
        atol=3e-15,
    )
    np.testing.assert_allclose(
        native.log_moments,
        reference.log_moments,
        rtol=3e-15,
        atol=3e-15,
    )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_taylor_horner_matches_direct_high_order_evaluation() -> None:
    rng = np.random.default_rng(7423)
    basis = (
        rng.normal(size=(13, 4_097))
        + 1j * rng.normal(size=(13, 4_097))
    )
    delta = (
        0.2 * rng.normal(size=basis.shape[1])
        + 0.2j * rng.normal(size=basis.shape[1])
    )
    native = evaluate_taylor_basis(
        basis,
        delta,
        chunk_size=131,
        threads=5,
        backend="native",
    )
    powers = (-delta[None, :]) ** np.arange(basis.shape[0])[:, None]
    direct_values = np.sum(basis * powers, axis=0)
    direct_moments = np.sum(
        np.arange(1, basis.shape[0])[:, None]
        * basis[1:]
        * powers[:-1],
        axis=0,
    )
    np.testing.assert_allclose(
        native.values,
        direct_values,
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        native.log_moments,
        direct_moments,
        rtol=2e-13,
        atol=2e-13,
    )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_taylor_outputs_are_thread_deterministic() -> None:
    base = np.exp(0.001j * np.arange(30_000))
    logarithms = np.log(np.arange(1, len(base) + 1))
    first = taylor_coefficients(
        base,
        logarithms,
        maximum_order=4,
        threads=1,
        backend="native",
    )
    second = taylor_coefficients(
        base,
        logarithms,
        maximum_order=4,
        threads=5,
        backend="native",
    )
    np.testing.assert_array_equal(first.coefficients, second.coefficients)


@pytest.mark.parametrize(
    "backend",
    ["reference"] + (["native"] if NATIVE_AVAILABLE else []),
)
def test_zero_order_taylor_is_identity_and_preserves_inputs(
    backend: str,
) -> None:
    base = np.array([1 + 2j, -3 + 0.5j, 7 - 4j])
    logarithms = np.array([0.0, 2.0, -5.0])
    base_snapshot = base.copy()
    log_snapshot = logarithms.copy()
    coefficients = taylor_coefficients(
        base,
        logarithms,
        maximum_order=0,
        backend=backend,
    )
    np.testing.assert_array_equal(coefficients.coefficients[0], base)
    evaluated = evaluate_taylor_basis(
        coefficients.coefficients,
        np.array([2 + 3j, -1j, 9 - 7j]),
        backend=backend,
    )
    np.testing.assert_array_equal(evaluated.values, base)
    np.testing.assert_array_equal(
        evaluated.log_moments,
        np.zeros_like(base),
    )
    np.testing.assert_array_equal(base, base_snapshot)
    np.testing.assert_array_equal(logarithms, log_snapshot)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_taylor_evaluation_is_thread_deterministic() -> None:
    rng = np.random.default_rng(7411)
    basis = (
        rng.normal(size=(7, 12_003))
        + 1j * rng.normal(size=(7, 12_003))
    )
    delta = (
        rng.normal(scale=0.01, size=basis.shape[1])
        + 1j * rng.normal(scale=0.01, size=basis.shape[1])
    )
    first = evaluate_taylor_basis(
        basis,
        delta,
        chunk_size=251,
        threads=1,
        backend="native",
    )
    second = evaluate_taylor_basis(
        basis,
        delta,
        chunk_size=251,
        threads=5,
        backend="native",
    )
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(
        first.log_moments,
        second.log_moments,
    )
