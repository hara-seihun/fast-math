"""ctypes binding for incremental planar collinearity scores."""

from __future__ import annotations

import ctypes
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from ._native import NativeUnavailable, _check, _uint64_pointer, load_library


class NativePlanarCollinearityStats(ctypes.Structure):
    _fields_ = [
        ("base_point_count", ctypes.c_uint64),
        ("edit_count", ctypes.c_uint64),
        ("base_score", ctypes.c_uint64),
        ("base_determinant_evaluations", ctypes.c_uint64),
        ("edit_determinant_evaluations", ctypes.c_uint64),
        ("worker_count", ctypes.c_uint32),
        ("elapsed_seconds", ctypes.c_double),
    ]


@lru_cache(maxsize=1)
def _native_function():
    library = load_library()
    if not hasattr(library, "fast_math_planar_collinearity_edits_i32"):
        raise NativeUnavailable(
            "fast-math was built without planar collinearity scores"
        )
    function = library.fast_math_planar_collinearity_edits_i32
    function.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(NativePlanarCollinearityStats),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int
    return function


def _pointer(array: np.ndarray, c_type):
    return array.ctypes.data_as(ctypes.POINTER(c_type))


def planar_collinearity_edits_native(
    base_points: NDArray[np.int32],
    delete_indices: NDArray[np.uint32],
    delete_offsets: NDArray[np.uint64],
    added_points: NDArray[np.int32],
    add_offsets: NDArray[np.uint64],
    score_cutoff: int,
    threads: int,
) -> tuple[
    int,
    NDArray[np.uint64],
    NDArray[np.uint64],
    NDArray[np.int64],
    NDArray[np.bool_],
    NativePlanarCollinearityStats,
]:
    if len(delete_offsets) == 0 or len(delete_offsets) != len(add_offsets):
        raise ValueError(
            "native delete_offsets and add_offsets must have equal nonzero lengths"
        )
    edit_count = len(delete_offsets) - 1
    base_score = ctypes.c_uint64()
    degrees = np.empty(len(base_points), dtype=np.uint64)
    edit_scores = np.empty(edit_count, dtype=np.uint64)
    edit_deltas = np.empty(edit_count, dtype=np.int64)
    cutoff_reached = np.empty(edit_count, dtype=np.uint8)
    stats = NativePlanarCollinearityStats()
    error = ctypes.create_string_buffer(1024)
    status = _native_function()(
        _pointer(base_points, ctypes.c_int32),
        len(base_points),
        _pointer(delete_indices, ctypes.c_uint32),
        len(delete_indices),
        _uint64_pointer(delete_offsets),
        _pointer(added_points, ctypes.c_int32),
        len(added_points),
        _uint64_pointer(add_offsets),
        edit_count,
        score_cutoff,
        threads,
        ctypes.byref(base_score),
        _uint64_pointer(degrees),
        _uint64_pointer(edit_scores),
        _pointer(edit_deltas, ctypes.c_int64),
        _pointer(cutoff_reached, ctypes.c_uint8),
        ctypes.byref(stats),
        error,
        len(error),
    )
    _check(status, error)
    return (
        int(base_score.value),
        degrees,
        edit_scores,
        edit_deltas,
        cutoff_reached.view(np.bool_),
        stats,
    )
