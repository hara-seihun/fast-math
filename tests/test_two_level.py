from __future__ import annotations

import numpy as np
import pytest

from lambda_fast import available_backends, fused_two_level
from lambda_fast.two_level import WEIGHT_DTYPE


NATIVE_AVAILABLE = "native" in available_backends()


def sample_inputs(seed: int = 31):
    rng = np.random.default_rng(seed)
    inverse = rng.normal(size=29)
    inverse[0] = 0.0
    inverse[::7] = 0.0
    primary = rng.normal(size=43)
    transformed = (
        rng.normal(size=9) + 1j * rng.normal(size=9)
    ).astype(np.complex128)
    low = rng.normal(size=37)
    return inverse, primary, transformed, low


def weight_intervals(output_limit: int, sigma: float) -> np.ndarray:
    records = []
    left = 2
    width = 1
    while left <= output_limit:
        right = min(output_limit, left + width - 1)
        records.append(
            (
                left,
                right,
                right ** (-sigma) * (1.0 - 1e-14),
                left ** (-sigma) * (1.0 + 1e-14),
            )
        )
        left = right + 1
        width = width % 11 + 1
    return np.array(records, dtype=WEIGHT_DTYPE)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
@pytest.mark.parametrize("seed", [3, 17, 91])
@pytest.mark.parametrize("tile_size", [17, 53, 257])
def test_fused_two_level_matches_materialized_reference(
    seed: int, tile_size: int
) -> None:
    inverse, primary, transformed, low = sample_inputs(seed)
    sigma = 1.57
    weights = weight_intervals(257, sigma)
    arguments = dict(
        transformed_first=13,
        output_limit=257,
        gamma_abs=0.137,
        sigma=sigma,
        q_primary=complex(sigma + 0.001, -0.056),
        q_dual=complex(sigma + 0.001, 0.056),
        outer_ratio=4.0,
        tile_size=tile_size,
    )
    reference = fused_two_level(
        inverse,
        primary,
        transformed,
        low,
        weights,
        backend="reference",
        **arguments,
    )
    native = fused_two_level(
        inverse,
        primary,
        transformed,
        low,
        weights,
        backend="native",
        threads=4,
        **arguments,
    )

    assert native.pairs == reference.pairs
    assert native.fine_weight_block_count == reference.fine_weight_block_count
    assert native.fine_piece_count == reference.fine_piece_count
    scalar_fields = (
        "constant_common_error",
        "constant_low_error",
        "center_cost",
        "weight_variation_upper",
        "fine_phase_drift_upper",
        "common_weighted_l1_upper",
        "low_weighted_l1_upper",
        "weighted_l1_upper",
        "two_level_upper",
    )
    for field in scalar_fields:
        assert getattr(native, field) == pytest.approx(
            getattr(reference, field), rel=2e-13, abs=2e-13
        )
    assert len(native.records) == len(reference.records)
    for actual, expected in zip(native.records, reference.records, strict=True):
        assert actual.left == expected.left
        assert actual.right == expected.right
        assert actual.fine_piece_count == expected.fine_piece_count
        assert actual.first_center == pytest.approx(
            expected.first_center, rel=2e-13, abs=2e-13
        )
        assert actual.second_center == pytest.approx(
            expected.second_center, rel=2e-13, abs=2e-13
        )
        assert actual.two_level_upper == pytest.approx(
            expected.two_level_upper, rel=2e-13, abs=2e-13
        )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_fused_two_level_is_bitwise_deterministic() -> None:
    inverse, primary, transformed, low = sample_inputs()
    sigma = 1.61
    weights = weight_intervals(1_003, sigma)
    results = [
        fused_two_level(
            inverse,
            primary,
            transformed,
            low,
            weights,
            transformed_first=7,
            output_limit=1_003,
            gamma_abs=0.12,
            sigma=sigma,
            q_primary=complex(sigma + 0.001, -0.04),
            q_dual=complex(sigma + 0.001, 0.04),
            outer_ratio=6.0,
            tile_size=61,
            threads=threads,
            backend="native",
        )
        for threads in (1, 3, 5)
    ]
    baseline = results[0]
    for result in results[1:]:
        assert result.records == baseline.records
        assert result.pairs == baseline.pairs
        assert result.two_level_upper == baseline.two_level_upper
        assert result.weighted_l1_upper == baseline.weighted_l1_upper


def test_fused_two_level_rejects_incomplete_weight_cover() -> None:
    inverse, primary, transformed, low = sample_inputs()
    weights = weight_intervals(50, 1.5)[:-1]
    with pytest.raises(ValueError, match="cover"):
        fused_two_level(
            inverse,
            primary,
            transformed,
            low,
            weights,
            transformed_first=4,
            output_limit=50,
            gamma_abs=0.1,
            sigma=1.5,
            q_primary=1.501 - 0.03j,
            q_dual=1.501 + 0.03j,
            outer_ratio=4.0,
        )
