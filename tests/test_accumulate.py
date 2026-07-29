from __future__ import annotations

import numpy as np
import pytest

from lambda_fast import accumulate_coefficients, available_backends


NATIVE_AVAILABLE = "native" in available_backends()


def sample_inputs(seed: int = 17):
    rng = np.random.default_rng(seed)
    inverse = rng.normal(size=24)
    inverse[0] = 0.0
    inverse[::5] = 0.0
    primary = rng.normal(size=37)
    transformed = (
        rng.normal(size=11) + 1j * rng.normal(size=11)
    ).astype(np.complex128)
    low = rng.normal(size=29)
    return inverse, primary, transformed, low


def test_reference_matches_manual_small_example() -> None:
    inverse = np.array([0.0, 1.0, -0.5])
    primary = np.array([2.0, 3.0, 5.0])
    transformed = np.array([1.0 + 2.0j])
    low = np.array([7.0, 11.0])

    result = accumulate_coefficients(
        inverse,
        primary,
        transformed,
        low,
        transformed_first=2,
        output_limit=6,
        backend="reference",
    )

    expected_common = np.zeros(7, dtype=np.complex128)
    expected_common[1] = 2.0
    expected_common[2] = 3.0 + (1.0 + 2.0j) - 1.0
    expected_common[3] = 5.0
    expected_common[4] = -1.5 - 0.5 * (1.0 + 2.0j)
    expected_common[6] = -2.5
    expected_low = np.zeros(7)
    expected_low[1] = 7.0
    expected_low[2] = 11.0 - 3.5
    expected_low[4] = -5.5

    np.testing.assert_array_equal(result.common, expected_common)
    np.testing.assert_array_equal(result.low, expected_low)
    assert result.pairs.primary == 6
    assert result.pairs.transformed == 2
    assert result.pairs.low == 4


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
@pytest.mark.parametrize("seed", [1, 7, 19, 101])
@pytest.mark.parametrize("tile_size", [7, 31, 256])
def test_native_matches_reference(seed: int, tile_size: int) -> None:
    inverse, primary, transformed, low = sample_inputs(seed)
    arguments = dict(
        transformed_first=9,
        output_limit=211,
        tile_size=tile_size,
    )
    reference = accumulate_coefficients(
        inverse,
        primary,
        transformed,
        low,
        backend="reference",
        **arguments,
    )
    native = accumulate_coefficients(
        inverse,
        primary,
        transformed,
        low,
        backend="native",
        threads=4,
        **arguments,
    )

    np.testing.assert_allclose(native.common, reference.common, rtol=0, atol=2e-15)
    np.testing.assert_allclose(native.low, reference.low, rtol=0, atol=2e-15)
    assert native.pairs == reference.pairs


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_native_is_bitwise_deterministic_across_thread_counts() -> None:
    inverse, primary, transformed, low = sample_inputs()
    results = [
        accumulate_coefficients(
            inverse,
            primary,
            transformed,
            low,
            transformed_first=4,
            output_limit=513,
            tile_size=19,
            threads=threads,
            backend="native",
        )
        for threads in (1, 2, 5)
    ]

    for result in results[1:]:
        np.testing.assert_array_equal(result.common, results[0].common)
        np.testing.assert_array_equal(result.low, results[0].low)
        assert result.pairs == results[0].pairs


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("transformed_first", 0, "transformed_first"),
        ("output_limit", 0, "output_limit"),
        ("tile_size", 0, "tile_size"),
        ("threads", -1, "threads"),
    ],
)
def test_invalid_scalar_arguments(
    field: str, value: int, message: str
) -> None:
    inverse, primary, transformed, low = sample_inputs()
    arguments = {
        "transformed_first": 2,
        "output_limit": 100,
        "tile_size": 16,
        "threads": 1,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        accumulate_coefficients(
            inverse,
            primary,
            transformed,
            low,
            backend="reference",
            **arguments,
        )


def test_rejects_multidimensional_sources() -> None:
    inverse, primary, transformed, low = sample_inputs()
    with pytest.raises(ValueError, match="primary"):
        accumulate_coefficients(
            inverse,
            primary.reshape(1, -1),
            transformed,
            low,
            transformed_first=2,
            output_limit=100,
        )


@pytest.mark.parametrize(
    "backend",
    ["reference"] + (["native"] if NATIVE_AVAILABLE else []),
)
def test_empty_optional_sources(backend: str) -> None:
    inverse = np.array([0.0, 1.0, -0.25])
    result = accumulate_coefficients(
        inverse,
        np.array([2.0, 3.0]),
        np.array([], dtype=np.complex128),
        np.array([], dtype=np.float64),
        transformed_first=5,
        output_limit=8,
        backend=backend,
    )
    assert result.pairs.transformed == 0
    assert result.pairs.low == 0
    assert result.common[1] == 2.0
    assert result.common[2] == 2.5
    assert result.common[4] == -0.75


@pytest.mark.parametrize(
    "backend",
    ["reference"] + (["native"] if NATIVE_AVAILABLE else []),
)
def test_accumulation_does_not_mutate_inputs(backend: str) -> None:
    inputs = sample_inputs(9021)
    snapshots = tuple(value.copy() for value in inputs)
    accumulate_coefficients(
        *inputs,
        transformed_first=4,
        output_limit=197,
        tile_size=13,
        threads=3,
        backend=backend,
    )
    for actual, expected in zip(inputs, snapshots, strict=True):
        np.testing.assert_array_equal(actual, expected)
