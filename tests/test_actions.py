from __future__ import annotations

import numpy as np
import pytest

from fast_math import PermutationActionPlan
from fast_math.hip import (
    SubsetActionHipPlan,
    hip_subset_actions_available,
)


def cyclic_action(degree: int) -> np.ndarray:
    return np.asarray(
        [[(point + shift) % degree for point in range(degree)] for shift in range(degree)],
        dtype=np.uint32,
    )


def test_cyclic_action_canonical_masks_and_partition() -> None:
    masks = np.asarray([0, 1, 2, 3, 5, 9, 17, 31], dtype=np.uint64)
    with PermutationActionPlan(cyclic_action(5)) as plan:
        result = plan.canonicalize(masks, backend="reference")
        np.testing.assert_array_equal(
            result.canonical_masks,
            np.asarray([0, 1, 1, 3, 5, 5, 3, 31], dtype=np.uint64),
        )
        np.testing.assert_array_equal(
            result.is_canonical,
            np.asarray([True, True, False, True, True, False, False, True]),
        )
        partition = plan.partition(masks, backend="reference")
        np.testing.assert_array_equal(
            partition.representatives,
            np.asarray([0, 1, 3, 5, 31], dtype=np.uint64),
        )
        assert int(partition.class_sizes.sum()) == len(masks)


def test_native_matches_reference() -> None:
    rng = np.random.default_rng(17)
    masks = rng.integers(0, 1 << 13, size=4096, dtype=np.uint64)
    with PermutationActionPlan(cyclic_action(13)) as plan:
        reference = plan.canonicalize(masks, backend="reference")
        native = plan.canonicalize(masks, threads=3, backend="native")
    np.testing.assert_array_equal(native.canonical_masks, reference.canonical_masks)
    np.testing.assert_array_equal(native.is_canonical, reference.is_canonical)


@pytest.mark.skipif(
    not hip_subset_actions_available(),
    reason="HIP subset-action backend unavailable",
)
def test_hip_matches_native() -> None:
    rng = np.random.default_rng(23)
    masks = rng.integers(0, 1 << 20, size=16384, dtype=np.uint64)
    with PermutationActionPlan(cyclic_action(20)) as plan:
        native = plan.canonicalize(masks, threads=4, backend="native")
        hip = plan.canonicalize(masks, backend="hip")
        hip_flags = plan.is_canonical(masks, backend="hip")
    np.testing.assert_array_equal(hip.canonical_masks, native.canonical_masks)
    np.testing.assert_array_equal(hip.is_canonical, native.is_canonical)
    np.testing.assert_array_equal(hip_flags, native.is_canonical)


@pytest.mark.skipif(
    not hip_subset_actions_available(),
    reason="HIP subset-action backend unavailable",
)
def test_degree_64_hip_is_exact_and_deterministic() -> None:
    identity = np.arange(64, dtype=np.uint32)
    action = np.stack((identity, identity[::-1], np.roll(identity, 13)))
    masks = np.asarray(
        [0, 1, 1 << 63, (1 << 64) - 1, 0xA53C_79E1_0246_8BDF],
        dtype=np.uint64,
    )
    with PermutationActionPlan(action) as plan:
        reference = plan.canonicalize(masks, backend="reference")
        native = plan.canonicalize(masks, backend="native", threads=2)
        first = plan.canonicalize(masks, backend="hip")
        second = plan.canonicalize(masks, backend="hip")
    for result in (native, first, second):
        np.testing.assert_array_equal(
            result.canonical_masks, reference.canonical_masks
        )
        np.testing.assert_array_equal(
            result.is_canonical, reference.is_canonical
        )


def test_identity_is_implicit_and_action_storage_is_owned() -> None:
    action = np.asarray([[1, 0]], dtype=np.uint32)
    with PermutationActionPlan(action) as plan:
        assert action.flags.writeable
        action[:] = [[0, 1]]
        np.testing.assert_array_equal(plan.permutations, [[1, 0]])
        result = plan.canonicalize([1, 2], backend="reference")
    np.testing.assert_array_equal(result.canonical_masks, [1, 1])
    np.testing.assert_array_equal(result.is_canonical, [True, False])


def test_action_validation_and_closed_plan() -> None:
    with pytest.raises(ValueError, match="permutation"):
        PermutationActionPlan([[0, 0]])
    with pytest.raises(TypeError, match="integers"):
        SubsetActionHipPlan([[0.25, 1.25]])
    plan = PermutationActionPlan([[0, 1]])
    with pytest.raises(ValueError, match="out-of-range"):
        plan.canonicalize([4], backend="reference")
    plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        plan.canonicalize([0], backend="reference")
