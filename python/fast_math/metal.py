from __future__ import annotations

import time

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .affine import (
    AffineContourMetrics,
    affine_batches,
    affine_inputs,
    affine_steps,
    edge_bounds,
)


class MetalUnavailable(RuntimeError):
    """Raised when the optional MLX Metal backend is unavailable."""


def _mlx_module():
    try:
        import mlx.core as mx
    except ImportError as error:
        raise MetalUnavailable(
            "Metal support requires the optional 'metal' dependency"
        ) from error
    if not mx.metal.is_available():
        raise MetalUnavailable("Metal support requires an available GPU")
    return mx


class AffineMetalPlan:
    """Persistent Metal plan for ``base + steps @ basis`` workloads."""

    backend = "metal"

    def __init__(self, base: ArrayLike, basis: ArrayLike) -> None:
        prepared_base, prepared_basis = affine_inputs(base, basis)
        mx = _mlx_module()
        started = time.perf_counter()
        with mx.stream(mx.gpu):
            self._base_gpu = mx.array(prepared_base)
            self._basis_gpu = mx.array(prepared_basis)
            mx.eval(self._base_gpu, self._basis_gpu)
        self._setup_seconds = time.perf_counter() - started
        self._base = prepared_base
        self._basis = prepared_basis

    @property
    def point_count(self) -> int:
        return self._base.size

    @property
    def direction_count(self) -> int:
        return self._basis.shape[0]

    @property
    def setup_seconds(self) -> float:
        return self._setup_seconds

    def _steps(self, values: ArrayLike) -> NDArray[np.float32]:
        return affine_steps(values, direction_count=self.direction_count)

    def _evaluate_batch(self, steps: NDArray[np.float32]):
        mx = _mlx_module()
        with mx.stream(mx.gpu):
            values = (
                self._base_gpu[None, :]
                + mx.array(steps) @ self._basis_gpu
            )
            mx.eval(values)
        return values

    def evaluate(
        self,
        steps: ArrayLike,
        *,
        batch_size: int | None = None,
    ) -> NDArray[np.complex64]:
        """Evaluate an affine population and return the full complex matrix."""
        prepared_steps = self._steps(steps)
        if batch_size is None:
            return np.asarray(self._evaluate_batch(prepared_steps)).copy()
        outputs = [
            np.asarray(self._evaluate_batch(batch)).copy()
            for batch in affine_batches(prepared_steps, batch_size)
        ]
        return np.concatenate(outputs, axis=0)

    def contour_metrics(
        self,
        steps: ArrayLike,
        *,
        edge_slice: slice = slice(None),
        batch_size: int | None = None,
    ) -> AffineContourMetrics:
        """Compute winding, phase-increment, and edge-floor ranking metrics."""
        edge_start, edge_stop = edge_bounds(
            edge_slice, point_count=self.point_count
        )
        mx = _mlx_module()
        prepared_steps = self._steps(steps)
        started = time.perf_counter()
        winding_batches = []
        phase_batches = []
        floor_batches = []
        for batch in affine_batches(prepared_steps, batch_size):
            with mx.stream(mx.gpu):
                values = (
                    self._base_gpu[None, :]
                    + mx.array(batch) @ self._basis_gpu
                )
                products = values[:, 1:] * mx.conj(values[:, :-1])
                phases = mx.arctan2(
                    mx.imag(products), mx.real(products)
                )
                windings = mx.round(
                    mx.sum(phases, axis=1) / (2 * np.pi)
                )
                maximum_phases = mx.max(mx.abs(phases), axis=1)
                edge_floors = mx.min(
                    mx.abs(values[:, edge_start:edge_stop]), axis=1
                )
                mx.eval(windings, maximum_phases, edge_floors)
            winding_batches.append(
                np.asarray(windings).astype(np.int64, copy=True)
            )
            phase_batches.append(np.asarray(maximum_phases).copy())
            floor_batches.append(np.asarray(edge_floors).copy())
        return AffineContourMetrics(
            windings=np.concatenate(winding_batches),
            maximum_phases=np.concatenate(phase_batches),
            edge_floors=np.concatenate(floor_batches),
            elapsed_seconds=time.perf_counter() - started,
            backend=self.backend,
        )

    @staticmethod
    def clear_cache() -> None:
        """Release MLX's reusable Metal allocation cache."""
        _mlx_module().clear_cache()
