from __future__ import annotations

import ctypes
import os
from pathlib import Path
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


class HipUnavailable(RuntimeError):
    """Raised when the optional ROCm/HIP backend is unavailable."""


def _library() -> ctypes.CDLL:
    candidates = []
    configured = os.environ.get("FAST_MATH_HIP_LIBRARY")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path(__file__).resolve().parents[2] / "build" / "libfast_math_hip.so",
            Path(__file__).resolve().parents[2] / "build" / "libfast_math_hip.dylib",
        ]
    )
    for path in candidates:
        if path.exists():
            try:
                library = ctypes.CDLL(str(path))
            except OSError as error:
                raise HipUnavailable(
                    f"could not load HIP library {path}: {error}"
                ) from error
            library.fast_math_hip_create.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            library.fast_math_hip_create.restype = ctypes.c_int
            library.fast_math_hip_destroy.argtypes = [ctypes.c_void_p]
            library.fast_math_hip_destroy.restype = ctypes.c_int
            library.fast_math_hip_evaluate.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
            ]
            library.fast_math_hip_evaluate.restype = ctypes.c_int
            library.fast_math_hip_metrics.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            library.fast_math_hip_metrics.restype = ctypes.c_int
            return library
    raise HipUnavailable(
        "HIP support requires build/libfast_math_hip.so; run `make hip`"
    )


def _pointer(array: NDArray[np.generic], dtype) -> ctypes.Array:
    return array.ctypes.data_as(ctypes.POINTER(dtype))


def _check(code: int, operation: str) -> None:
    if code != 0:
        raise RuntimeError(f"HIP {operation} failed with status {code}")


class AffineHipPlan:
    """Persistent ROCm plan for ``base + steps @ basis`` workloads."""

    backend = "hip"

    def __init__(self, base: ArrayLike, basis: ArrayLike) -> None:
        prepared_base, prepared_basis = affine_inputs(base, basis)
        library = _library()
        started = time.perf_counter()
        handle = ctypes.c_void_p()
        _check(
            library.fast_math_hip_create(
                _pointer(prepared_base.view(np.float32), ctypes.c_float),
                prepared_base.size,
                _pointer(prepared_basis.view(np.float32), ctypes.c_float),
                prepared_basis.shape[0],
                ctypes.byref(handle),
            ),
            "plan creation",
        )
        self._library = library
        self._handle = handle
        self._base = prepared_base
        self._basis = prepared_basis
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

    def _evaluate_batch(self, steps: NDArray[np.float32]) -> NDArray[np.complex64]:
        output = np.empty((steps.shape[0], self.point_count), dtype=np.complex64)
        _check(
            self._library.fast_math_hip_evaluate(
                self._handle,
                _pointer(steps, ctypes.c_float),
                steps.shape[0],
                _pointer(output.view(np.float32), ctypes.c_float),
            ),
            "evaluation",
        )
        return output

    def evaluate(
        self,
        steps: ArrayLike,
        *,
        batch_size: int | None = None,
    ) -> NDArray[np.complex64]:
        prepared_steps = self._steps(steps)
        outputs = [
            self._evaluate_batch(batch)
            for batch in affine_batches(prepared_steps, batch_size)
        ]
        return outputs[0] if len(outputs) == 1 else np.concatenate(outputs, axis=0)

    def contour_metrics(
        self,
        steps: ArrayLike,
        *,
        edge_slice: slice = slice(None),
        batch_size: int | None = None,
    ) -> AffineContourMetrics:
        edge_start, edge_stop = edge_bounds(
            edge_slice, point_count=self.point_count
        )
        prepared_steps = self._steps(steps)
        started = time.perf_counter()
        windings = []
        maximum_phases = []
        edge_floors = []
        for batch in affine_batches(prepared_steps, batch_size):
            rows = batch.shape[0]
            batch_windings = np.empty(rows, dtype=np.int64)
            batch_phases = np.empty(rows, dtype=np.float32)
            batch_floors = np.empty(rows, dtype=np.float32)
            _check(
                self._library.fast_math_hip_metrics(
                    self._handle,
                    _pointer(batch, ctypes.c_float),
                    rows,
                    edge_start,
                    edge_stop,
                    _pointer(batch_windings, ctypes.c_int64),
                    _pointer(batch_phases, ctypes.c_float),
                    _pointer(batch_floors, ctypes.c_float),
                ),
                "contour metrics",
            )
            windings.append(batch_windings)
            maximum_phases.append(batch_phases)
            edge_floors.append(batch_floors)
        return AffineContourMetrics(
            windings=np.concatenate(windings),
            maximum_phases=np.concatenate(maximum_phases),
            edge_floors=np.concatenate(edge_floors),
            elapsed_seconds=time.perf_counter() - started,
            backend=self.backend,
        )

    def close(self) -> None:
        handle, self._handle = self._handle, ctypes.c_void_p()
        if handle and handle.value:
            _check(self._library.fast_math_hip_destroy(handle), "plan destruction")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def clear_cache() -> None:
        # HIP's allocator is released when each persistent plan is closed.
        return None
