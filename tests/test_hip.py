from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from fast_math import AffineHipPlan, HipUnavailable, affine_plan
from fast_math.affine import numpy_contour_metrics


@pytest.fixture(scope="module")
def hip_plan():
    if importlib.util.find_spec("fast_math.hip") is None:
        pytest.skip("HIP module is unavailable")
    rng = np.random.default_rng(9011)
    base = (
        rng.normal(size=193) + 1j * rng.normal(size=193)
    ).astype(np.complex64)
    basis = (
        rng.normal(size=(5, base.size))
        + 1j * rng.normal(size=(5, base.size))
    ).astype(np.complex64)
    try:
        plan = AffineHipPlan(base, basis)
    except (HipUnavailable, OSError, RuntimeError) as error:
        pytest.skip(str(error))
    yield plan, base, basis
    plan.close()


def test_hip_matches_numpy(hip_plan) -> None:
    plan, base, basis = hip_plan
    rng = np.random.default_rng(9012)
    steps = rng.normal(size=(17, basis.shape[0])).astype(np.float32)
    expected = base[None, :] + steps @ basis
    actual = plan.evaluate(steps)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
    metrics = plan.contour_metrics(steps, edge_slice=slice(19, 171))
    expected_metrics = numpy_contour_metrics(
        expected, edge_start=19, edge_stop=171
    )
    np.testing.assert_array_equal(metrics.windings, expected_metrics[0])
    np.testing.assert_allclose(
        metrics.maximum_phases, expected_metrics[1], rtol=3e-5, atol=3e-5
    )
    np.testing.assert_allclose(
        metrics.edge_floors, expected_metrics[2], rtol=3e-5, atol=3e-5
    )
    assert metrics.backend == "hip"


def test_hip_batching_preserves_results(hip_plan) -> None:
    plan, _, basis = hip_plan
    rng = np.random.default_rng(9013)
    steps = rng.normal(size=(23, basis.shape[0])).astype(np.float32)
    np.testing.assert_allclose(
        plan.evaluate(steps, batch_size=7), plan.evaluate(steps),
        rtol=3e-5, atol=3e-5,
    )
    full = plan.contour_metrics(steps, edge_slice=slice(11, 180))
    batched = plan.contour_metrics(
        steps, edge_slice=slice(11, 180), batch_size=7
    )
    np.testing.assert_array_equal(batched.windings, full.windings)
    np.testing.assert_allclose(
        batched.maximum_phases, full.maximum_phases,
        rtol=3e-5, atol=3e-5,
    )
    np.testing.assert_allclose(
        batched.edge_floors, full.edge_floors,
        rtol=3e-5, atol=3e-5,
    )


def test_auto_prefers_hip_when_available(hip_plan) -> None:
    _, base, basis = hip_plan
    plan = affine_plan(base, basis)
    try:
        assert plan.backend == "hip"
    finally:
        close = getattr(plan, "close", None)
        if close is not None:
            close()
