from __future__ import annotations

import ctypes
from functools import lru_cache
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


@lru_cache(maxsize=1)
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
            if hasattr(library, "fast_math_hip_square_cover_create"):
                library.fast_math_hip_square_cover_create.argtypes = [
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                library.fast_math_hip_square_cover_create.restype = ctypes.c_int
                library.fast_math_hip_square_cover_destroy.argtypes = [
                    ctypes.c_void_p
                ]
                library.fast_math_hip_square_cover_destroy.restype = ctypes.c_int
                library.fast_math_hip_square_cover_evaluate.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_uint64),
                ]
                library.fast_math_hip_square_cover_evaluate.restype = ctypes.c_int
                library.fast_math_hip_square_weighted_evaluate.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                ]
                library.fast_math_hip_square_weighted_evaluate.restype = ctypes.c_int
            if hasattr(library, "fast_math_hip_subset_action_create"):
                library.fast_math_hip_subset_action_create.argtypes = [
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_size_t,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                library.fast_math_hip_subset_action_create.restype = ctypes.c_int
                library.fast_math_hip_subset_action_destroy.argtypes = [
                    ctypes.c_void_p
                ]
                library.fast_math_hip_subset_action_destroy.restype = ctypes.c_int
                library.fast_math_hip_subset_action_canonicalize.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_uint8),
                ]
                library.fast_math_hip_subset_action_canonicalize.restype = ctypes.c_int
                library.fast_math_hip_subset_action_is_canonical.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_uint8),
                ]
                library.fast_math_hip_subset_action_is_canonical.restype = ctypes.c_int
            return library
    raise HipUnavailable(
        "HIP support requires build/libfast_math_hip.so; run `make hip`"
    )


def hip_available() -> bool:
    try:
        _library()
    except HipUnavailable:
        return False
    return True


def hip_subset_actions_available() -> bool:
    try:
        return hasattr(_library(), "fast_math_hip_subset_action_create")
    except HipUnavailable:
        return False


def _pointer(array: NDArray[np.generic], dtype) -> ctypes.Array:
    return array.ctypes.data_as(ctypes.POINTER(dtype))


def _check(code: int, operation: str) -> None:
    if code != 0:
        raise RuntimeError(f"HIP {operation} failed with status {code}")


def _subset_permutations(values: ArrayLike) -> NDArray[np.uint32]:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[0] == 0 or not 1 <= raw.shape[1] <= 64:
        raise ValueError("permutations must have shape (count, degree), degree 1..64")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("permutations must contain integers")
    if np.any(raw < 0) or np.any(raw >= raw.shape[1]):
        raise ValueError("permutation image is out of range")
    result = np.array(raw, dtype=np.uint32, order="C", copy=True)
    if np.any(np.sort(result, axis=1) != np.arange(result.shape[1])):
        raise ValueError("every action row must be a permutation")
    result.flags.writeable = False
    return result


def _subset_masks(values: ArrayLike, degree: int) -> NDArray[np.uint64]:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("masks must be one-dimensional")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("masks must contain integers")
    if np.issubdtype(raw.dtype, np.signedinteger) and np.any(raw < 0):
        raise ValueError("masks must be nonnegative")
    result = np.ascontiguousarray(raw, dtype=np.uint64)
    if degree < 64 and np.any(result >> np.uint64(degree)):
        raise ValueError("mask contains an out-of-range bit")
    return result


class SubsetActionHipPlan:
    """Persistent exact HIP plan for packed subset images."""

    backend = "hip"

    def __init__(self, permutations: ArrayLike) -> None:
        values = _subset_permutations(permutations)
        library = _library()
        if not hasattr(library, "fast_math_hip_subset_action_create"):
            raise HipUnavailable("HIP packed subset actions are not built")
        handle = ctypes.c_void_p()
        _check(
            library.fast_math_hip_subset_action_create(
                _pointer(values.reshape(-1), ctypes.c_uint32),
                len(values),
                values.shape[1],
                ctypes.byref(handle),
            ),
            "subset action plan creation",
        )
        if not handle.value:
            raise RuntimeError("HIP subset action returned a null handle")
        self._library = library
        self._handle = handle
        self._permutations = values

    @property
    def degree(self) -> int:
        return self._permutations.shape[1]

    def canonicalize(
        self,
        masks: ArrayLike,
    ) -> tuple[NDArray[np.uint64], NDArray[np.bool_], float]:
        if not self._handle.value:
            raise RuntimeError("HIP subset action plan is closed")
        values = _subset_masks(masks, self.degree)
        canonical = np.empty(len(values), dtype=np.uint64)
        flags = np.empty(len(values), dtype=np.uint8)
        started = time.perf_counter()
        _check(
            self._library.fast_math_hip_subset_action_canonicalize(
                self._handle,
                _pointer(values, ctypes.c_uint64),
                len(values),
                _pointer(canonical, ctypes.c_uint64),
                _pointer(flags, ctypes.c_uint8),
            ),
            "subset action canonicalization",
        )
        return canonical, flags.view(np.bool_), time.perf_counter() - started

    def is_canonical(
        self,
        masks: ArrayLike,
    ) -> tuple[NDArray[np.bool_], float]:
        if not self._handle.value:
            raise RuntimeError("HIP subset action plan is closed")
        values = _subset_masks(masks, self.degree)
        flags = np.empty(len(values), dtype=np.uint8)
        started = time.perf_counter()
        _check(
            self._library.fast_math_hip_subset_action_is_canonical(
                self._handle,
                _pointer(values, ctypes.c_uint64),
                len(values),
                _pointer(flags, ctypes.c_uint8),
            ),
            "subset action canonical test",
        )
        return flags.view(np.bool_), time.perf_counter() - started

    def close(self) -> None:
        handle, self._handle = self._handle, ctypes.c_void_p()
        if handle.value:
            _check(
                self._library.fast_math_hip_subset_action_destroy(handle),
                "subset action plan destruction",
            )

    def __enter__(self) -> "SubsetActionHipPlan":
        if not self._handle.value:
            raise RuntimeError("HIP subset action plan is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class SquareCoverHipPlan:
    """Persistent FP64 HIP plan for oriented-square point incidence."""

    backend = "hip"

    def __init__(self, points: ArrayLike) -> None:
        values = np.asarray(points, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != 2
            or len(values) == 0
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("points must have finite shape (point_count, 2)")
        values = np.ascontiguousarray(values)
        library = _library()
        if not hasattr(library, "fast_math_hip_square_cover_create"):
            raise HipUnavailable("HIP square-cover support is not built")
        handle = ctypes.c_void_p()
        _check(
            library.fast_math_hip_square_cover_create(
                _pointer(values.reshape(-1), ctypes.c_double),
                len(values),
                ctypes.byref(handle),
            ),
            "square-cover plan creation",
        )
        self._library = library
        self._handle = handle
        self._points = values

    @property
    def point_count(self) -> int:
        return len(self._points)

    def evaluate(
        self,
        poses: ArrayLike,
        *,
        half_extent: float = 0.5,
        uncertainty: float = 0.0,
    ) -> tuple[NDArray[np.uint64], NDArray[np.uint64], float]:
        values = np.asarray(poses, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != 4
            or len(values) == 0
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("poses must have finite shape (pose_count, 4)")
        if (
            not np.isfinite(half_extent)
            or half_extent <= 0
            or not np.isfinite(uncertainty)
            or uncertainty < 0
            or uncertainty >= half_extent
        ):
            raise ValueError("invalid half_extent or uncertainty")
        values = np.ascontiguousarray(values)
        outputs = np.empty(
            ((self.point_count + 63) // 64, len(values)), dtype=np.uint64
        )
        uncertain = np.empty_like(outputs)
        started = time.perf_counter()
        _check(
            self._library.fast_math_hip_square_cover_evaluate(
                self._handle,
                _pointer(values.reshape(-1), ctypes.c_double),
                len(values),
                half_extent,
                uncertainty,
                _pointer(outputs.reshape(-1), ctypes.c_uint64),
                _pointer(uncertain.reshape(-1), ctypes.c_uint64),
            ),
            "square-cover evaluation",
        )
        return outputs, uncertain, time.perf_counter() - started

    def weighted_scores(
        self,
        poses: ArrayLike,
        weights: ArrayLike,
        *,
        half_extent: float = 0.5,
        uncertainty: float = 0.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        values = np.asarray(poses, dtype=np.float64)
        weight_values = np.asarray(weights, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != 4
            or len(values) == 0
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("poses must have finite shape (pose_count, 4)")
        if (
            weight_values.shape != (self.point_count,)
            or not np.all(np.isfinite(weight_values))
            or np.any(weight_values < 0)
        ):
            raise ValueError("weights must be finite and nonnegative")
        if (
            not np.isfinite(half_extent)
            or half_extent <= 0
            or not np.isfinite(uncertainty)
            or uncertainty < 0
            or uncertainty >= half_extent
        ):
            raise ValueError("invalid half_extent or uncertainty")
        values = np.ascontiguousarray(values)
        weight_values = np.ascontiguousarray(weight_values)
        definite = np.empty(len(values), dtype=np.float64)
        possible = np.empty_like(definite)
        started = time.perf_counter()
        _check(
            self._library.fast_math_hip_square_weighted_evaluate(
                self._handle,
                _pointer(values.reshape(-1), ctypes.c_double),
                len(values),
                _pointer(weight_values, ctypes.c_double),
                half_extent,
                uncertainty,
                _pointer(definite, ctypes.c_double),
                _pointer(possible, ctypes.c_double),
            ),
            "square weighted evaluation",
        )
        return definite, possible, time.perf_counter() - started

    def __enter__(self) -> "SquareCoverHipPlan":
        if not self._handle.value:
            raise RuntimeError("square-cover plan is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        handle, self._handle = self._handle, ctypes.c_void_p()
        if handle and handle.value:
            _check(
                self._library.fast_math_hip_square_cover_destroy(handle),
                "square-cover plan destruction",
            )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


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
