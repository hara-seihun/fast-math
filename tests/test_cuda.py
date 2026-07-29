from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from fast_math import AffineCudaPlan, affine_plan


CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None
pytestmark = pytest.mark.skipif(
    not CUPY_AVAILABLE, reason="CuPy is not installed"
)


def numpy_metrics(
    values: np.ndarray,
    edge_slice: slice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    products = values[:, 1:] * np.conj(values[:, :-1])
    phases = np.arctan2(products.imag, products.real)
    return (
        np.rint(np.sum(phases, axis=1) / (2 * np.pi)).astype(np.int64),
        np.max(np.abs(phases), axis=1),
        np.min(np.abs(values[:, edge_slice]), axis=1),
    )


def test_affine_cuda_plan_matches_numpy_evaluation_and_metrics() -> None:
    rng = np.random.default_rng(8301)
    base = (
        rng.normal(size=257) + 1j * rng.normal(size=257)
    ).astype(np.complex64)
    basis = (
        rng.normal(size=(7, base.size))
        + 1j * rng.normal(size=(7, base.size))
    ).astype(np.complex64)
    steps = rng.normal(size=(23, basis.shape[0])).astype(np.float32)
    expected_values = base[None, :] + steps @ basis
    expected_metrics = numpy_metrics(expected_values, slice(31, 199))

    plan = AffineCudaPlan(base, basis)
    actual_values = plan.evaluate(steps)
    actual_metrics = plan.contour_metrics(
        steps, edge_slice=slice(31, 199)
    )

    assert plan.point_count == base.size
    assert plan.direction_count == basis.shape[0]
    assert plan.setup_seconds >= 0.0
    assert actual_metrics.backend == "cuda"
    assert actual_values.dtype == np.complex64
    np.testing.assert_allclose(
        actual_values, expected_values, rtol=3e-5, atol=3e-5
    )
    np.testing.assert_array_equal(
        actual_metrics.windings, expected_metrics[0]
    )
    np.testing.assert_allclose(
        actual_metrics.maximum_phases,
        expected_metrics[1],
        rtol=3e-5,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        actual_metrics.edge_floors,
        expected_metrics[2],
        rtol=3e-5,
        atol=3e-5,
    )


def test_affine_cuda_plan_batches_without_changing_results() -> None:
    rng = np.random.default_rng(8302)
    base = np.exp(0.03j * np.arange(193)).astype(np.complex64)
    basis = (
        rng.normal(size=(5, base.size))
        + 1j * rng.normal(size=(5, base.size))
    ).astype(np.complex64)
    steps = rng.normal(size=(29, basis.shape[0])).astype(np.float32)
    plan = AffineCudaPlan(base, basis)

    full_values = plan.evaluate(steps)
    batched_values = plan.evaluate(steps, batch_size=7)
    full_metrics = plan.contour_metrics(
        steps, edge_slice=slice(20, 170)
    )
    batched_metrics = plan.contour_metrics(
        steps, edge_slice=slice(20, 170), batch_size=7
    )

    np.testing.assert_allclose(
        batched_values, full_values, rtol=3e-5, atol=3e-5
    )
    np.testing.assert_array_equal(
        batched_metrics.windings, full_metrics.windings
    )
    np.testing.assert_allclose(
        batched_metrics.maximum_phases,
        full_metrics.maximum_phases,
        rtol=3e-5,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        batched_metrics.edge_floors,
        full_metrics.edge_floors,
        rtol=3e-5,
        atol=3e-5,
    )


def test_affine_cuda_plan_copies_retained_inputs() -> None:
    base = np.ones(17, dtype=np.complex64)
    basis = np.ones((3, 17), dtype=np.complex64)
    plan = AffineCudaPlan(base, basis)
    base[:] = 100
    basis[:] = 100
    actual = plan.evaluate(np.ones((2, 3), dtype=np.float32))
    np.testing.assert_array_equal(actual, np.full((2, 17), 4 + 0j))


def test_affine_factory_selects_cuda_explicitly() -> None:
    plan = affine_plan(
        np.ones(17, dtype=np.complex64),
        np.ones((3, 17), dtype=np.complex64),
        backend="cuda",
    )
    assert isinstance(plan, AffineCudaPlan)
    assert plan.backend == "cuda"


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: AffineCudaPlan([], np.ones((2, 3))), "base must not"),
        (
            lambda: AffineCudaPlan([1], np.ones(3)),
            "basis must be 2-dimensional",
        ),
        (
            lambda: AffineCudaPlan([1, 2], np.ones((3, 4))),
            "basis must have shape",
        ),
        (
            lambda: AffineCudaPlan([np.nan], np.ones((1, 1))),
            "base must contain only finite",
        ),
    ],
)
def test_affine_cuda_plan_rejects_invalid_setup(
    factory, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_affine_cuda_plan_rejects_invalid_execution() -> None:
    plan = AffineCudaPlan(
        np.ones(11, dtype=np.complex64),
        np.ones((3, 11), dtype=np.complex64),
    )
    with pytest.raises(ValueError, match="steps must have shape"):
        plan.evaluate(np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="steps must have shape"):
        plan.evaluate(np.ones((4, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="steps must not be empty"):
        plan.evaluate(np.ones((0, 3), dtype=np.float32))
    with pytest.raises(TypeError, match="steps must be real"):
        plan.evaluate(np.ones((4, 3), dtype=np.complex64))
    with pytest.raises(ValueError, match="batch_size must be positive"):
        plan.evaluate(np.ones((4, 3), dtype=np.float32), batch_size=0)
    with pytest.raises(TypeError, match="edge_slice must be a slice"):
        plan.contour_metrics(
            np.ones((4, 3), dtype=np.float32), edge_slice=(0, 3)
        )
    with pytest.raises(ValueError, match="nonempty contiguous"):
        plan.contour_metrics(
            np.ones((4, 3), dtype=np.float32), edge_slice=slice(3, 3)
        )
