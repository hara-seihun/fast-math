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


class CudaUnavailable(RuntimeError):
    """Raised when the optional CuPy CUDA backend is unavailable."""


def _cupy_module():
    try:
        import cupy as cp
    except ImportError as error:
        raise CudaUnavailable(
            "CUDA support requires the optional 'cuda' dependency"
        ) from error
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise CudaUnavailable("CUDA support requires an available GPU")
    except CudaUnavailable:
        raise
    except Exception as error:
        raise CudaUnavailable("CUDA initialization failed") from error
    return cp


class AffineCudaPlan:
    """Persistent CUDA plan for ``base + steps @ basis`` workloads."""

    backend = "cuda"

    def __init__(self, base: ArrayLike, basis: ArrayLike) -> None:
        prepared_base, prepared_basis = affine_inputs(base, basis)
        cp = _cupy_module()
        started = time.perf_counter()
        self._base_gpu = cp.asarray(prepared_base)
        self._basis_gpu = cp.asarray(prepared_basis)
        cp.cuda.get_current_stream().synchronize()
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
        cp = _cupy_module()
        return self._base_gpu[None, :] + cp.asarray(steps) @ self._basis_gpu

    def evaluate(
        self,
        steps: ArrayLike,
        *,
        batch_size: int | None = None,
    ) -> NDArray[np.complex64]:
        """Evaluate an affine population and return the full complex matrix."""
        cp = _cupy_module()
        prepared_steps = self._steps(steps)
        outputs = [
            cp.asnumpy(self._evaluate_batch(batch))
            for batch in affine_batches(prepared_steps, batch_size)
        ]
        if len(outputs) == 1:
            return outputs[0]
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
        cp = _cupy_module()
        prepared_steps = self._steps(steps)
        started = time.perf_counter()
        winding_batches = []
        phase_batches = []
        floor_batches = []
        for batch in affine_batches(prepared_steps, batch_size):
            values = self._evaluate_batch(batch)
            products = values[:, 1:] * cp.conj(values[:, :-1])
            phases = cp.arctan2(cp.imag(products), cp.real(products))
            windings = cp.rint(
                cp.sum(phases, axis=1) / (2 * np.pi)
            ).astype(cp.int64)
            maximum_phases = cp.max(cp.abs(phases), axis=1)
            edge_floors = cp.min(
                cp.abs(values[:, edge_start:edge_stop]), axis=1
            )
            winding_batches.append(cp.asnumpy(windings))
            phase_batches.append(cp.asnumpy(maximum_phases))
            floor_batches.append(cp.asnumpy(edge_floors))
        cp.cuda.get_current_stream().synchronize()
        return AffineContourMetrics(
            windings=np.concatenate(winding_batches),
            maximum_phases=np.concatenate(phase_batches),
            edge_floors=np.concatenate(floor_batches),
            elapsed_seconds=time.perf_counter() - started,
            backend=self.backend,
        )

    @staticmethod
    def clear_cache() -> None:
        """Release CuPy's reusable device and pinned-memory caches."""
        cp = _cupy_module()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
