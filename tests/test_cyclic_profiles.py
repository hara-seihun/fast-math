from __future__ import annotations

import numpy as np
import pytest

from fast_math import cyclic_correlation_profiles
from fast_math._cyclic_native import cyclic_correlation_profiles_native
from fast_math._native import native_available


BACKENDS = ["reference", "native"]


def direct_profiles(mask: int, width: int) -> tuple[list[int], list[int]]:
    bits = [(mask >> position) & 1 for position in range(width)]
    signs = [1 - 2 * bit for bit in bits]
    intersections = []
    correlations = []
    for lag in range(width):
        intersections.append(
            sum(bits[position] * bits[(position - lag) % width]
                for position in range(width))
        )
        correlations.append(
            sum(signs[position] * signs[(position - lag) % width]
                for position in range(width))
        )
    return intersections, correlations


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("width", [1, 2, 3, 5, 8, 31, 32, 63, 64])
def test_matches_independent_periodic_definition(
    backend: str, width: int
) -> None:
    maximum = (1 << width) - 1
    masks = np.asarray(
        [0, 1, maximum, maximum // 3, maximum // 5], dtype=np.uint64
    )
    result = cyclic_correlation_profiles(
        masks, width, backend=backend, threads=1
    )
    for index, mask in enumerate(masks):
        intersections, correlations = direct_profiles(int(mask), width)
        np.testing.assert_array_equal(
            result.intersection_counts[index], intersections
        )
        np.testing.assert_array_equal(
            result.signed_correlations[index], correlations
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_profiles_have_expected_extreme_rows(backend: str) -> None:
    width = 12
    maximum = (1 << width) - 1
    result = cyclic_correlation_profiles(
        [0, maximum, 1], width, backend=backend
    )
    assert np.all(result.intersection_counts[0] == 0)
    assert np.all(result.intersection_counts[1] == width)
    assert np.all(result.signed_correlations[:2] == width)
    assert result.intersection_counts[2].tolist() == [1] + [0] * 11
    assert result.signed_correlations[2].tolist() == [width] + [
        width - 4
    ] * 11


@pytest.mark.parametrize("width", range(1, 65))
def test_native_and_reference_match_random_batches(width: int) -> None:
    random = np.random.default_rng(20260823 + width)
    high = 1 << width
    masks = np.asarray(
        [int(random.integers(0, 1 << min(width, 63))) for _ in range(257)],
        dtype=np.uint64,
    )
    if width == 64:
        masks |= random.integers(
            0, 2, size=len(masks), dtype=np.uint64
        ) << np.uint64(63)
    reference = cyclic_correlation_profiles(
        masks, width, backend="reference"
    )
    native = cyclic_correlation_profiles(
        masks, width, backend="native", threads=4
    )
    np.testing.assert_array_equal(
        native.intersection_counts, reference.intersection_counts
    )
    np.testing.assert_array_equal(
        native.signed_correlations, reference.signed_correlations
    )
    assert int(masks.max(initial=0)) < high


@pytest.mark.parametrize("backend", BACKENDS)
def test_rotation_and_complement_invariance(backend: str) -> None:
    width = 17
    mask = 0b10110010100101101
    valid = (1 << width) - 1
    rotated = ((mask << 6) | (mask >> (width - 6))) & valid
    complement = (~mask) & valid
    result = cyclic_correlation_profiles(
        [mask, rotated, complement], width, backend=backend
    )
    np.testing.assert_array_equal(
        result.intersection_counts[0], result.intersection_counts[1]
    )
    np.testing.assert_array_equal(
        result.signed_correlations[0], result.signed_correlations[1]
    )
    np.testing.assert_array_equal(
        result.signed_correlations[0], result.signed_correlations[2]
    )
    np.testing.assert_array_equal(
        result.intersection_counts[:, 1:],
        result.intersection_counts[:, :0:-1],
    )


def test_native_output_is_thread_stable() -> None:
    random = np.random.default_rng(19)
    masks = random.integers(0, 1 << 32, size=4096, dtype=np.uint64)
    serial = cyclic_correlation_profiles(
        masks, 32, backend="native", threads=1
    )
    parallel = cyclic_correlation_profiles(
        masks, 32, backend="native", threads=8
    )
    np.testing.assert_array_equal(
        parallel.intersection_counts, serial.intersection_counts
    )
    np.testing.assert_array_equal(
        parallel.signed_correlations, serial.signed_correlations
    )
    assert serial.worker_count == 1
    assert parallel.worker_count == 8
    assert serial.popcount_evaluations == len(masks) * 17


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_batch_has_declared_shape(backend: str) -> None:
    result = cyclic_correlation_profiles([], 11, backend=backend)
    assert result.intersection_counts.shape == (0, 11)
    assert result.signed_correlations.shape == (0, 11)
    assert result.mask_count == 0


@pytest.mark.parametrize(
    ("masks", "width", "threads", "message"),
    [
        ([0], 0, 0, "bit_width"),
        ([0], 65, 0, "bit_width"),
        ([0], True, 0, "bit_width"),
        ([8], 3, 0, "outside"),
        ([-1], 3, 0, "nonnegative"),
        ([1.5], 3, 0, "integers"),
        ([[1]], 3, 0, "one-dimensional"),
        ([1], 3, -1, "threads"),
        ([1], 3, 1025, "threads"),
        ([1], 3, True, "threads"),
    ],
)
def test_rejects_invalid_public_inputs(
    masks, width: int, threads: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cyclic_correlation_profiles(masks, width, threads=threads)


def test_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        cyclic_correlation_profiles([1], 3, backend="gpu")


@pytest.mark.parametrize(
    ("masks", "width", "threads", "message"),
    [
        ([8], 3, 0, "outside"),
        ([0], 0, 0, "between one"),
        ([0], 3, 1025, "at most 1024"),
    ],
)
def test_native_abi_rejects_hostile_inputs(
    masks: list[int], width: int, threads: int, message: str
) -> None:
    if not native_available():
        pytest.skip("native library is unavailable")
    with pytest.raises(RuntimeError, match=message):
        cyclic_correlation_profiles_native(
            np.asarray(masks, dtype=np.uint64), width, threads
        )
