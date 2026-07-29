from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from fast_math import (
    Type1Plan1D,
    Type3FixedPlan1D,
    Type3Plan1D,
    Type3SignPairPlan1D,
)


FINUFFT_AVAILABLE = importlib.util.find_spec("finufft") is not None
pytestmark = pytest.mark.skipif(
    not FINUFFT_AVAILABLE, reason="FINUFFT is not installed"
)


def test_type3_many_transform_plan_matches_simple_interface() -> None:
    import finufft

    rng = np.random.default_rng(8101)
    sources = rng.uniform(-2.0, 2.0, size=701)
    targets = rng.uniform(-3.0, 3.0, size=389)
    strengths = (
        rng.normal(size=(5, sources.size))
        + 1j * rng.normal(size=(5, sources.size))
    )
    plan = Type3Plan1D(
        sources,
        targets,
        n_trans=5,
        eps=1e-12,
        nthreads=1,
    )
    actual = plan.execute(strengths)
    expected = finufft.nufft1d3(
        sources,
        strengths,
        targets,
        eps=1e-12,
        isign=1,
        nthreads=1,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    assert plan.timings.execute_count == 1
    assert plan.timings.point_set_count == 1


def test_type3_plan_reuses_sources_and_updates_targets() -> None:
    import finufft

    rng = np.random.default_rng(8102)
    sources = rng.uniform(-1.0, 1.0, size=503)
    original_sources = sources.copy()
    targets = rng.uniform(-2.0, 2.0, size=211)
    strengths = rng.normal(size=sources.size) + 1j * rng.normal(
        size=sources.size
    )
    plan = Type3Plan1D(
        sources, targets, eps=1e-11, isign=-1, nthreads=1
    )
    sources[:] = 0.0
    first = plan.execute(strengths)
    expected_first = finufft.nufft1d3(
        original_sources,
        strengths,
        targets,
        eps=1e-11,
        isign=-1,
        nthreads=1,
    )
    np.testing.assert_allclose(
        first, expected_first, rtol=2e-11, atol=2e-11
    )

    next_targets = rng.uniform(-4.0, 4.0, size=307)
    plan.set_targets(next_targets)
    second = plan.execute(strengths)
    expected_second = finufft.nufft1d3(
        original_sources,
        strengths,
        next_targets,
        eps=1e-11,
        isign=-1,
        nthreads=1,
    )
    np.testing.assert_allclose(
        second, expected_second, rtol=2e-11, atol=2e-11
    )
    assert plan.timings.execute_count == 2
    assert plan.timings.point_set_count == 2


def test_type3_sign_pair_plan_matches_separate_signs() -> None:
    import finufft

    rng = np.random.default_rng(8106)
    sources = rng.uniform(-2.0, 2.0, size=809)
    targets = rng.uniform(-3.0, 3.0, size=431)
    positive_strengths = (
        rng.normal(size=(3, sources.size))
        + 1j * rng.normal(size=(3, sources.size))
    )
    negative_strengths = (
        rng.normal(size=(3, sources.size))
        + 1j * rng.normal(size=(3, sources.size))
    )
    plan = Type3SignPairPlan1D(
        sources,
        positive_strengths,
        negative_strengths,
        targets,
        n_trans=3,
        eps=1e-12,
        nthreads=1,
    )
    actual_positive, actual_negative = plan.execute()
    expected_positive = finufft.nufft1d3(
        sources,
        positive_strengths,
        targets,
        eps=1e-12,
        isign=1,
        nthreads=1,
    )
    expected_negative = finufft.nufft1d3(
        sources,
        negative_strengths,
        -targets,
        eps=1e-12,
        isign=1,
        nthreads=1,
    )
    np.testing.assert_allclose(
        actual_positive,
        expected_positive,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        actual_negative,
        expected_negative,
        rtol=2e-12,
        atol=2e-12,
    )
    assert plan.timings.execute_count == 1
    assert plan.timings.point_set_count == 1


def test_type3_fixed_plan_retains_strengths_and_updates_targets() -> None:
    import finufft

    rng = np.random.default_rng(8108)
    sources = rng.uniform(-2.0, 2.0, size=607)
    original_sources = sources.copy()
    strengths = (
        rng.normal(size=(3, sources.size))
        + 1j * rng.normal(size=(3, sources.size))
    )
    original_strengths = strengths.copy()
    targets = rng.uniform(-3.0, 3.0, size=313)
    plan = Type3FixedPlan1D(
        sources,
        strengths,
        targets,
        n_trans=3,
        eps=1e-12,
        nthreads=1,
    )
    sources[:] = 0.0
    strengths[:] = 0.0

    next_targets = rng.uniform(-4.0, 4.0, size=419)
    actual = plan.execute(targets=next_targets)
    expected = finufft.nufft1d3(
        original_sources,
        original_strengths,
        next_targets,
        eps=1e-12,
        isign=1,
        nthreads=1,
    )
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=2e-12,
        atol=2e-12,
    )
    assert actual.shape == (3, next_targets.size)
    assert not plan.strengths.flags.writeable
    assert plan.timings.execute_count == 1
    assert plan.timings.point_set_count == 2


def test_type3_fixed_plan_single_transform_shape() -> None:
    sources = np.arange(7, dtype=np.float64)
    plan = Type3FixedPlan1D(
        sources,
        np.ones(7, dtype=np.complex128),
        np.arange(5, dtype=np.float64),
        nthreads=1,
    )
    assert plan.execute().shape == (5,)


def test_type3_sign_pair_plan_retains_inputs_and_resizes_targets() -> None:
    import finufft

    rng = np.random.default_rng(8107)
    sources = rng.uniform(-1.0, 1.0, size=503)
    original_sources = sources.copy()
    positive_strengths = (
        rng.normal(size=sources.size)
        + 1j * rng.normal(size=sources.size)
    )
    negative_strengths = (
        rng.normal(size=sources.size)
        + 1j * rng.normal(size=sources.size)
    )
    original_positive = positive_strengths.copy()
    original_negative = negative_strengths.copy()
    targets = rng.uniform(-2.0, 2.0, size=211)
    plan = Type3SignPairPlan1D(
        sources,
        positive_strengths,
        negative_strengths,
        targets,
        eps=1e-11,
        nthreads=1,
    )
    sources[:] = 0.0
    positive_strengths[:] = 0.0
    negative_strengths[:] = 0.0

    next_targets = rng.uniform(-4.0, 4.0, size=307)
    actual_positive, actual_negative = plan.execute(
        targets=next_targets
    )
    assert actual_positive.shape == (next_targets.size,)
    assert actual_negative.shape == (next_targets.size,)
    expected_positive = finufft.nufft1d3(
        original_sources,
        original_positive,
        next_targets,
        eps=1e-11,
        isign=1,
        nthreads=1,
    )
    expected_negative = finufft.nufft1d3(
        original_sources,
        original_negative,
        -next_targets,
        eps=1e-11,
        isign=1,
        nthreads=1,
    )
    np.testing.assert_allclose(
        actual_positive,
        expected_positive,
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        actual_negative,
        expected_negative,
        rtol=2e-11,
        atol=2e-11,
    )
    assert not plan.positive_strengths.flags.writeable
    assert not plan.negative_strengths.flags.writeable
    assert plan.timings.point_set_count == 2


def test_type3_sign_pair_plan_rejects_wrong_strength_shape() -> None:
    with pytest.raises(
        ValueError, match="strengths must have shape"
    ):
        Type3SignPairPlan1D(
            np.arange(7, dtype=np.float64),
            np.ones((2, 7), dtype=np.complex128),
            np.ones((3, 7), dtype=np.complex128),
            np.arange(5, dtype=np.float64),
            n_trans=2,
            nthreads=1,
        )


def test_type1_plan_matches_simple_interface_and_reuses_output() -> None:
    import finufft

    rng = np.random.default_rng(8103)
    nodes = rng.uniform(-np.pi, np.pi, size=997)
    strengths = rng.normal(size=nodes.size) + 1j * rng.normal(
        size=nodes.size
    )
    plan = Type1Plan1D(
        nodes, 512, eps=1e-12, isign=1, nthreads=1
    )
    out = np.empty(512, dtype=np.complex128)
    actual = plan.execute(strengths, out=out)
    expected = finufft.nufft1d1(
        nodes,
        strengths,
        512,
        eps=1e-12,
        isign=1,
        nthreads=1,
    )
    assert actual is out
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_single_precision_plan_preserves_dtype() -> None:
    import finufft

    rng = np.random.default_rng(8104)
    sources = rng.uniform(-1.0, 1.0, size=401).astype(np.float32)
    targets = rng.uniform(-2.0, 2.0, size=233).astype(np.float32)
    strengths = (
        rng.normal(size=sources.size)
        + 1j * rng.normal(size=sources.size)
    ).astype(np.complex64)
    plan = Type3Plan1D(
        sources,
        targets,
        eps=1e-5,
        dtype=np.complex64,
        nthreads=1,
    )
    actual = plan.execute(strengths)
    expected = finufft.nufft1d3(
        sources,
        strengths,
        targets,
        eps=1e-5,
        isign=1,
        nthreads=1,
    )
    assert actual.dtype == np.complex64
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: Type1Plan1D([], 8), "nodes must not be empty"),
        (
            lambda: Type3Plan1D([0.0], [np.nan]),
            "targets must contain only finite values",
        ),
        (
            lambda: Type3Plan1D([0.0], [1.0], n_trans=0),
            "n_trans must be positive",
        ),
        (
            lambda: Type1Plan1D([0.0], 8, dtype=np.float64),
            "dtype must be complex64 or complex128",
        ),
        (
            lambda: Type1Plan1D([0.0], 8.5),
            "n_modes must be an integer",
        ),
        (
            lambda: Type3Plan1D([0.0], [1.0], n_trans=2.5),
            "n_trans must be an integer",
        ),
        (
            lambda: Type3Plan1D([0.0], [1.0], isign=True),
            "isign must be -1 or 1",
        ),
    ],
)
def test_plan_rejects_invalid_configuration(factory, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_plan_rejects_wrong_strength_shape() -> None:
    plan = Type3Plan1D(
        np.arange(7, dtype=np.float64),
        np.arange(5, dtype=np.float64),
        n_trans=3,
        nthreads=1,
    )
    with pytest.raises(ValueError, match="strengths must have shape"):
        plan.execute(np.ones(7, dtype=np.complex128))


def test_plan_does_not_mutate_strengths() -> None:
    strengths = np.arange(7, dtype=np.complex128) * (1.0 + 2.0j)
    snapshot = strengths.copy()
    plan = Type3Plan1D(
        np.arange(7, dtype=np.float64),
        np.arange(5, dtype=np.float64),
        nthreads=1,
    )
    plan.execute(strengths)
    np.testing.assert_array_equal(strengths, snapshot)


def test_type1_many_transform_plan_matches_simple_interface() -> None:
    import finufft

    rng = np.random.default_rng(8105)
    nodes = rng.uniform(-np.pi, np.pi, size=601)
    strengths = (
        rng.normal(size=(3, nodes.size))
        + 1j * rng.normal(size=(3, nodes.size))
    )
    plan = Type1Plan1D(
        nodes,
        384,
        n_trans=3,
        eps=1e-11,
        isign=-1,
        nthreads=1,
    )
    actual = plan.execute(strengths)
    expected = finufft.nufft1d1(
        nodes,
        strengths,
        384,
        eps=1e-11,
        isign=-1,
        nthreads=1,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
