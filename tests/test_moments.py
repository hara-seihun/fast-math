from __future__ import annotations

import numpy as np
import pytest

from lambda_fast import available_backends, power_moments


NATIVE_AVAILABLE = "native" in available_backends()


def sample_values(seed: int = 43, count: int = 1_003):
    rng = np.random.default_rng(seed)
    values = (
        0.3 + rng.normal(scale=0.2, size=count)
        + 1j * rng.normal(scale=0.2, size=count)
    )
    derivatives = (
        rng.normal(scale=0.1, size=count)
        + 1j * rng.normal(scale=0.1, size=count)
    )
    return values, derivatives


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
@pytest.mark.parametrize("minimum_power,maximum_power", [(1, 3), (3, 12)])
def test_power_moments_match_reference(
    minimum_power: int, maximum_power: int
) -> None:
    values, derivatives = sample_values()
    arguments = dict(
        mesh_step=0.25,
        minimum_power=minimum_power,
        maximum_power=maximum_power,
        chunk_size=127,
    )
    reference = power_moments(
        values, derivatives, backend="reference", **arguments
    )
    native = power_moments(
        values,
        derivatives,
        backend="native",
        threads=5,
        **arguments,
    )

    assert native.sample_count == reference.sample_count
    assert native.maximum_modulus == pytest.approx(
        reference.maximum_modulus, rel=2e-15
    )
    assert native.maximum_derivative == pytest.approx(
        reference.maximum_derivative, rel=2e-15
    )
    for actual, expected in zip(
        native.moments, reference.moments, strict=True
    ):
        assert actual.power == expected.power
        assert actual.value == pytest.approx(
            expected.value, rel=3e-13, abs=1e-14
        )
        assert actual.ordinary == pytest.approx(
            expected.ordinary, rel=3e-13, abs=1e-14
        )
        assert actual.phase_current == pytest.approx(
            expected.phase_current, rel=3e-13, abs=1e-14
        )
        assert actual.radial == pytest.approx(
            expected.radial, rel=3e-13, abs=1e-14
        )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_power_moments_are_bitwise_deterministic() -> None:
    values, derivatives = sample_values(count=10_003)
    results = [
        power_moments(
            values,
            derivatives,
            mesh_step=4.0,
            minimum_power=3,
            maximum_power=8,
            chunk_size=257,
            threads=threads,
            backend="native",
        )
        for threads in (1, 4, 5)
    ]
    for result in results[1:]:
        assert result.moments == results[0].moments
        assert result.maximum_modulus == results[0].maximum_modulus
        assert result.maximum_derivative == results[0].maximum_derivative


def test_power_moments_reject_mismatched_vectors() -> None:
    values, derivatives = sample_values()
    with pytest.raises(ValueError, match="equal nonzero"):
        power_moments(values, derivatives[:-1], mesh_step=1.0)
