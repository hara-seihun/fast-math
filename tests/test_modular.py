from __future__ import annotations

import numpy as np
import pytest

from fast_math.modular import (
    ModularDeterminantPlan,
    ModularPolynomialPlan,
    determinants_mod_u32,
)
from fast_math.hip import hip_modular_available


def test_polynomial_values_and_derivatives() -> None:
    coefficients = np.asarray([[1, 2, 3], [5, 0, 0]], dtype=np.uint32)
    points = np.asarray([0, 1, 2, 6], dtype=np.uint32)
    with ModularPolynomialPlan(coefficients, prime=7) as plan:
        result = plan.evaluate(points, derivative=True, backend="reference")
    np.testing.assert_array_equal(
        result.values,
        [[1, 6, 3, 2], [5, 5, 5, 5]],
    )
    np.testing.assert_array_equal(
        result.derivatives,
        [[2, 1, 0, 3], [0, 0, 0, 0]],
    )


@pytest.mark.parametrize("prime", [2, 3, 65521, 1_000_000_007, 4_294_967_291])
def test_native_polynomial_matches_reference_at_hostile_primes(prime: int) -> None:
    rng = np.random.default_rng(prime)
    coefficients = rng.integers(0, prime, size=(9, 17), dtype=np.uint64)
    points = rng.integers(0, prime, size=67, dtype=np.uint64)
    with ModularPolynomialPlan(coefficients, prime=prime) as plan:
        reference = plan.evaluate(points, derivative=True, backend="reference")
        native = plan.evaluate(
            points, derivative=True, threads=3, backend="native"
        )
    np.testing.assert_array_equal(native.values, reference.values)
    np.testing.assert_array_equal(native.derivatives, reference.derivatives)


@pytest.mark.skipif(
    not hip_modular_available(),
    reason="HIP modular backend unavailable",
)
def test_hip_polynomial_matches_reference() -> None:
    prime = 4_294_967_291
    rng = np.random.default_rng(91)
    coefficients = rng.integers(0, prime, size=(31, 33), dtype=np.uint64)
    points = rng.integers(0, prime, size=257, dtype=np.uint64)
    with ModularPolynomialPlan(coefficients, prime=prime) as plan:
        reference = plan.evaluate(points, derivative=True, backend="reference")
        first = plan.evaluate(points, derivative=True, backend="hip")
        second = plan.evaluate(points, derivative=True, backend="hip")
    for result in (first, second):
        np.testing.assert_array_equal(result.values, reference.values)
        np.testing.assert_array_equal(result.derivatives, reference.derivatives)


def test_modular_determinants_match_reference_and_flint() -> None:
    pytest.importorskip("flint", reason="python-flint is unavailable")
    from flint import nmod_mat

    prime = 1_000_000_007
    rng = np.random.default_rng(29)
    matrices = rng.integers(0, prime, size=(64, 7, 7), dtype=np.uint32)
    matrices[0, 1] = matrices[0, 0]
    reference = determinants_mod_u32(
        matrices, prime=prime, backend="reference"
    )
    native = determinants_mod_u32(
        matrices, prime=prime, threads=4, backend="native"
    )
    flint_values = np.asarray(
        [
            int(nmod_mat(7, 7, matrix.reshape(-1).tolist(), prime).det())
            for matrix in matrices
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(native.determinants, reference.determinants)
    np.testing.assert_array_equal(native.determinants, flint_values)
    assert native.determinants[0] == 0


@pytest.mark.skipif(
    not hip_modular_available(),
    reason="HIP modular backend unavailable",
)
def test_hip_determinants_match_native() -> None:
    prime = 1_000_000_007
    rng = np.random.default_rng(31)
    matrices = rng.integers(0, prime, size=(2048, 8, 8), dtype=np.uint32)
    with ModularDeterminantPlan(8, prime=prime) as plan:
        native = plan.determinants(matrices, threads=4, backend="native")
        hip = plan.determinants(matrices, backend="hip")
    np.testing.assert_array_equal(hip.determinants, native.determinants)


def test_modular_validation_storage_and_closed_plans() -> None:
    coefficients = np.asarray([[1, 2]], dtype=np.uint32)
    plan = ModularPolynomialPlan(coefficients, prime=7)
    coefficients.fill(0)
    np.testing.assert_array_equal(plan.coefficients, [[1, 2]])
    with pytest.raises(ValueError, match="prime"):
        ModularPolynomialPlan([[1]], prime=9)
    with pytest.raises(ValueError, match="outside"):
        plan.evaluate([7])
    plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        plan.evaluate([1])
