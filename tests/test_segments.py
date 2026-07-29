from __future__ import annotations

import numpy as np
import pytest

from fast_math import segmented_complex_stats
from lambda_fast import available_backends


NATIVE_AVAILABLE = "native" in available_backends()


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_segmented_complex_stats_match_reference() -> None:
    rng = np.random.default_rng(7301)
    values = (
        rng.normal(size=10_003)
        + 1j * rng.normal(size=10_003)
    )
    offsets = np.array([0, 1, 17, 511, 4_096, len(values)])
    reference = segmented_complex_stats(
        values,
        offsets,
        backend="reference",
    )
    native = segmented_complex_stats(
        values,
        offsets,
        threads=5,
        backend="native",
    )
    np.testing.assert_allclose(native.sums, reference.sums, rtol=2e-14)
    np.testing.assert_allclose(native.l1, reference.l1, rtol=2e-14)
    np.testing.assert_allclose(
        native.variation,
        reference.variation,
        rtol=2e-14,
    )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_segmented_complex_stats_are_thread_deterministic() -> None:
    values = np.exp(0.003j * np.arange(20_000))
    offsets = np.arange(0, 20_001, 200, dtype=np.uint64)
    first = segmented_complex_stats(
        values,
        offsets,
        threads=1,
        backend="native",
    )
    second = segmented_complex_stats(
        values,
        offsets,
        threads=5,
        backend="native",
    )
    np.testing.assert_array_equal(first.sums, second.sums)
    np.testing.assert_array_equal(first.l1, second.l1)
    np.testing.assert_array_equal(first.variation, second.variation)


def test_segmented_complex_stats_reject_bad_offsets() -> None:
    with pytest.raises(ValueError, match="partition"):
        segmented_complex_stats([1, 2, 3], [0, 2, 2, 3])


@pytest.mark.parametrize(
    "backend",
    ["reference"] + (["native"] if NATIVE_AVAILABLE else []),
)
def test_singleton_and_constant_segments(backend: str) -> None:
    values = np.array(
        [3 + 4j, 1 - 2j, 1 - 2j, 1 - 2j, -5j],
        dtype=np.complex128,
    )
    snapshot = values.copy()
    result = segmented_complex_stats(
        values,
        [0, 1, 4, 5],
        threads=4,
        backend=backend,
    )
    np.testing.assert_array_equal(result.sums, [3 + 4j, 3 - 6j, -5j])
    np.testing.assert_array_equal(result.l1, [5.0, 3 * np.sqrt(5), 5.0])
    np.testing.assert_array_equal(result.variation, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(values, snapshot)
