from __future__ import annotations

from dataclasses import dataclass
import operator
import platform
import time
from typing import Any, Iterator, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


AffineBackend = Literal["auto", "numpy", "metal", "cuda"]


@dataclass(frozen=True)
class AffineContourMetrics:
    windings: NDArray[np.int64]
    maximum_phases: NDArray[np.float32]
    edge_floors: NDArray[np.float32]
    elapsed_seconds: float
    backend: str


def positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def complex_array(
    values: ArrayLike,
    *,
    name: str,
    dimensions: int,
) -> NDArray[np.complex64]:
    result = np.array(values, dtype=np.complex64, order="C", copy=True)
    if result.ndim != dimensions:
        raise ValueError(f"{name} must be {dimensions}-dimensional")
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.flags.writeable = False
    return result


def affine_inputs(
    base: ArrayLike,
    basis: ArrayLike,
) -> tuple[NDArray[np.complex64], NDArray[np.complex64]]:
    prepared_base = complex_array(base, name="base", dimensions=1)
    prepared_basis = complex_array(basis, name="basis", dimensions=2)
    if prepared_basis.shape[1] != prepared_base.size:
        raise ValueError(
            "basis must have shape (direction_count, point_count)"
        )
    return prepared_base, prepared_basis


def affine_steps(
    values: ArrayLike,
    *,
    direction_count: int,
) -> NDArray[np.float32]:
    if np.iscomplexobj(values):
        raise TypeError("steps must be real")
    result = np.asarray(values, dtype=np.float32, order="C")
    if result.ndim != 2 or result.shape[1] != direction_count:
        raise ValueError(
            "steps must have shape (population, direction_count)"
        )
    if result.shape[0] == 0:
        raise ValueError("steps must not be empty")
    return result


def affine_batches(
    steps: NDArray[np.float32],
    batch_size: int | None,
) -> Iterator[NDArray[np.float32]]:
    if batch_size is None:
        yield steps
        return
    normalized_size = positive_integer(batch_size, name="batch_size")
    for begin in range(0, steps.shape[0], normalized_size):
        yield steps[begin : begin + normalized_size]


def edge_bounds(edge_slice: slice, *, point_count: int) -> tuple[int, int]:
    if point_count < 2:
        raise ValueError("contour metrics require at least two points")
    if not isinstance(edge_slice, slice):
        raise TypeError("edge_slice must be a slice")
    edge_start, edge_stop, edge_step = edge_slice.indices(point_count)
    if edge_step != 1 or edge_stop <= edge_start:
        raise ValueError("edge_slice must select a nonempty contiguous range")
    return edge_start, edge_stop


def numpy_contour_metrics(
    values: NDArray[np.complex64],
    *,
    edge_start: int,
    edge_stop: int,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    products = values[:, 1:] * np.conj(values[:, :-1])
    phases = np.arctan2(products.imag, products.real)
    windings = np.rint(
        np.sum(phases, axis=1) / (2 * np.pi)
    ).astype(np.int64)
    maximum_phases = np.max(np.abs(phases), axis=1)
    edge_floors = np.min(
        np.abs(values[:, edge_start:edge_stop]), axis=1
    )
    return windings, maximum_phases, edge_floors


class AffineNumpyPlan:
    """Portable CPU plan for ``base + steps @ basis`` workloads."""

    backend = "numpy"

    def __init__(self, base: ArrayLike, basis: ArrayLike) -> None:
        started = time.perf_counter()
        self._base, self._basis = affine_inputs(base, basis)
        self._setup_seconds = time.perf_counter() - started

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

    def _evaluate_batch(
        self,
        steps: NDArray[np.float32],
    ) -> NDArray[np.complex64]:
        return np.asarray(
            self._base[None, :] + steps @ self._basis,
            dtype=np.complex64,
        )

    def evaluate(
        self,
        steps: ArrayLike,
        *,
        batch_size: int | None = None,
    ) -> NDArray[np.complex64]:
        """Evaluate an affine population and return the full complex matrix."""
        prepared_steps = self._steps(steps)
        outputs = [
            self._evaluate_batch(batch)
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
        prepared_steps = self._steps(steps)
        started = time.perf_counter()
        winding_batches = []
        phase_batches = []
        floor_batches = []
        for batch in affine_batches(prepared_steps, batch_size):
            windings, maximum_phases, edge_floors = numpy_contour_metrics(
                self._evaluate_batch(batch),
                edge_start=edge_start,
                edge_stop=edge_stop,
            )
            winding_batches.append(windings)
            phase_batches.append(maximum_phases)
            floor_batches.append(edge_floors)
        return AffineContourMetrics(
            windings=np.concatenate(winding_batches),
            maximum_phases=np.concatenate(phase_batches),
            edge_floors=np.concatenate(floor_batches),
            elapsed_seconds=time.perf_counter() - started,
            backend=self.backend,
        )

    @staticmethod
    def clear_cache() -> None:
        """Match the GPU plan cache-management contract."""
        return None


def affine_plan(
    base: ArrayLike,
    basis: ArrayLike,
    *,
    backend: AffineBackend = "auto",
):
    """Create the fastest available affine plan for the requested backend."""
    if backend not in {"auto", "numpy", "metal", "cuda"}:
        raise ValueError(f"unknown affine backend: {backend}")
    if backend == "numpy":
        return AffineNumpyPlan(base, basis)
    if backend == "metal":
        from .metal import AffineMetalPlan

        return AffineMetalPlan(base, basis)
    if backend == "cuda":
        from .cuda import AffineCudaPlan

        return AffineCudaPlan(base, basis)

    if platform.system() == "Darwin":
        from .metal import AffineMetalPlan, MetalUnavailable

        try:
            return AffineMetalPlan(base, basis)
        except MetalUnavailable:
            pass
    else:
        from .cuda import AffineCudaPlan, CudaUnavailable

        try:
            return AffineCudaPlan(base, basis)
        except CudaUnavailable:
            pass
    return AffineNumpyPlan(base, basis)
