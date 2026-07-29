from __future__ import annotations

import numpy as np
import pytest

from lambda_fast import (
    available_backends,
    dirichlet_inverse,
    truncated_inverse,
)


NATIVE_AVAILABLE = "native" in available_backends()


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
@pytest.mark.parametrize("limit", [1, 2, 17, 257, 2_003])
def test_dirichlet_inverse_matches_reference(limit: int) -> None:
    indices = np.arange(1, limit + 1, dtype=np.float64)
    source = np.exp(-0.03 * np.log(indices) ** 2)
    reference = dirichlet_inverse(source, backend="reference")
    native = dirichlet_inverse(source, backend="native")
    np.testing.assert_allclose(
        native.coefficients,
        reference.coefficients,
        rtol=2e-13,
        atol=2e-13,
    )
    assert native.update_count == reference.update_count


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_truncated_inverse_matches_current_heat_source() -> None:
    reference = truncated_inverse(1_003, 0.142_908_675_2, backend="reference")
    native = truncated_inverse(1_003, 0.142_908_675_2, backend="native")
    np.testing.assert_allclose(
        native.coefficients,
        reference.coefficients,
        rtol=2e-13,
        atol=2e-13,
    )


def test_dirichlet_inverse_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        dirichlet_inverse([])


@pytest.mark.parametrize(
    "backend",
    ["reference"] + (["native"] if NATIVE_AVAILABLE else []),
)
@pytest.mark.parametrize("seed", [8101, 8102, 8103])
def test_dirichlet_inverse_satisfies_convolution_identity(
    backend: str,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    source = rng.normal(scale=0.15, size=127)
    source[0] = 1.0
    result = dirichlet_inverse(source, backend=backend)
    for value in range(1, len(source) + 1):
        convolution = sum(
            result.coefficients[divisor] * source[value // divisor - 1]
            for divisor in range(1, value + 1)
            if value % divisor == 0
        )
        assert convolution == pytest.approx(
            1.0 if value == 1 else 0.0,
            abs=2e-13,
        )


def test_dirichlet_inverse_does_not_mutate_source() -> None:
    source = np.linspace(1.0, 2.0, 73)
    snapshot = source.copy()
    dirichlet_inverse(source, backend="reference")
    np.testing.assert_array_equal(source, snapshot)
