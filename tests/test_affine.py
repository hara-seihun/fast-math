from __future__ import annotations

import numpy as np
import pytest

from fast_math import AffineNumpyPlan, affine_plan


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


def test_affine_numpy_plan_matches_direct_evaluation_and_metrics() -> None:
    rng = np.random.default_rng(8101)
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

    plan = AffineNumpyPlan(base, basis)
    actual_values = plan.evaluate(steps)
    actual_metrics = plan.contour_metrics(
        steps, edge_slice=slice(31, 199)
    )

    assert plan.backend == "numpy"
    assert plan.point_count == base.size
    assert plan.direction_count == basis.shape[0]
    assert plan.setup_seconds >= 0.0
    assert actual_metrics.backend == "numpy"
    np.testing.assert_array_equal(actual_values, expected_values)
    np.testing.assert_array_equal(
        actual_metrics.windings, expected_metrics[0]
    )
    np.testing.assert_array_equal(
        actual_metrics.maximum_phases, expected_metrics[1]
    )
    np.testing.assert_array_equal(
        actual_metrics.edge_floors, expected_metrics[2]
    )


def test_affine_numpy_plan_batches_and_copies_inputs() -> None:
    rng = np.random.default_rng(8102)
    base = np.exp(0.03j * np.arange(193)).astype(np.complex64)
    basis = (
        rng.normal(size=(5, base.size))
        + 1j * rng.normal(size=(5, base.size))
    ).astype(np.complex64)
    retained_base = base.copy()
    retained_basis = basis.copy()
    steps = rng.normal(size=(29, basis.shape[0])).astype(np.float32)
    plan = AffineNumpyPlan(base, basis)
    base[:] = 100
    basis[:] = 100

    expected = retained_base[None, :] + steps @ retained_basis
    np.testing.assert_allclose(
        plan.evaluate(steps, batch_size=7),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )
    full_metrics = plan.contour_metrics(
        steps, edge_slice=slice(20, 170)
    )
    batched_metrics = plan.contour_metrics(
        steps, edge_slice=slice(20, 170), batch_size=7
    )
    np.testing.assert_array_equal(
        batched_metrics.windings, full_metrics.windings
    )
    np.testing.assert_allclose(
        batched_metrics.maximum_phases,
        full_metrics.maximum_phases,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        batched_metrics.edge_floors,
        full_metrics.edge_floors,
        rtol=1e-6,
        atol=1e-6,
    )


def test_affine_plan_factory_supports_portable_and_auto_backends() -> None:
    base = np.ones(17, dtype=np.complex64)
    basis = np.ones((3, 17), dtype=np.complex64)

    portable = affine_plan(base, basis, backend="numpy")
    automatic = affine_plan(base, basis)

    assert isinstance(portable, AffineNumpyPlan)
    assert automatic.backend in {"numpy", "metal", "cuda"}
    np.testing.assert_allclose(
        automatic.evaluate(np.ones((2, 3), dtype=np.float32)),
        np.full((2, 17), 4 + 0j, dtype=np.complex64),
        rtol=3e-5,
        atol=3e-5,
    )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: AffineNumpyPlan([], np.ones((2, 3))), "base must not"),
        (
            lambda: AffineNumpyPlan([1], np.ones(3)),
            "basis must be 2-dimensional",
        ),
        (
            lambda: AffineNumpyPlan([1, 2], np.ones((3, 4))),
            "basis must have shape",
        ),
        (
            lambda: AffineNumpyPlan([np.nan], np.ones((1, 1))),
            "base must contain only finite",
        ),
    ],
)
def test_affine_numpy_plan_rejects_invalid_setup(
    factory, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_affine_numpy_plan_rejects_invalid_execution() -> None:
    plan = AffineNumpyPlan(
        np.ones(11, dtype=np.complex64),
        np.ones((3, 11), dtype=np.complex64),
    )
    with pytest.raises(ValueError, match="steps must have shape"):
        plan.evaluate(np.ones(3, dtype=np.float32))
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
    with pytest.raises(ValueError, match="at least two points"):
        AffineNumpyPlan(
            np.ones(1, dtype=np.complex64),
            np.ones((3, 1), dtype=np.complex64),
        ).contour_metrics(np.ones((4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="unknown affine backend"):
        affine_plan(
            np.ones(3),
            np.ones((1, 3)),
            backend="vulkan",  # type: ignore[arg-type]
        )
