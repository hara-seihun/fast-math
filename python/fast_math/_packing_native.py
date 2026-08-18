from __future__ import annotations

import ctypes

import numpy as np
from numpy.typing import NDArray

from ._native import load_library


class NativeSquareCoverStats(ctypes.Structure):
    _fields_ = [
        ("point_count", ctypes.c_uint64),
        ("pose_count", ctypes.c_uint64),
        ("word_count", ctypes.c_uint64),
        ("incidence_tests", ctypes.c_uint64),
        ("thread_count", ctypes.c_uint32),
        ("simd_lanes", ctypes.c_uint32),
        ("elapsed_seconds", ctypes.c_double),
    ]


def _pointer(array: NDArray[np.generic], dtype):
    return array.ctypes.data_as(ctypes.POINTER(dtype))


def square_cover_words_native(
    points: NDArray[np.float64],
    center_x: NDArray[np.float64],
    center_y: NDArray[np.float64],
    direction_x: NDArray[np.float64],
    direction_y: NDArray[np.float64],
    *,
    half_extent: float,
    uncertainty: float,
    threads: int,
) -> tuple[NDArray[np.uint64], NDArray[np.uint64], NativeSquareCoverStats]:
    library = load_library()
    function = library.fast_math_square_cover_words_f64
    function.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(NativeSquareCoverStats),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int
    pose_count = len(center_x)
    word_count = (len(points) + 63) // 64
    inside = np.empty((word_count, pose_count), dtype=np.uint64)
    uncertain = np.empty_like(inside)
    stats = NativeSquareCoverStats()
    error = ctypes.create_string_buffer(512)
    status = function(
        _pointer(points.reshape(-1), ctypes.c_double),
        len(points),
        _pointer(center_x, ctypes.c_double),
        _pointer(center_y, ctypes.c_double),
        _pointer(direction_x, ctypes.c_double),
        _pointer(direction_y, ctypes.c_double),
        pose_count,
        half_extent,
        uncertainty,
        threads,
        _pointer(inside.reshape(-1), ctypes.c_uint64),
        _pointer(uncertain.reshape(-1), ctypes.c_uint64),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(message or f"native square cover failed ({status})")
    return inside, uncertain, stats


def square_weighted_scores_native(
    points: NDArray[np.float64],
    weights: NDArray[np.float64],
    center_x: NDArray[np.float64],
    center_y: NDArray[np.float64],
    direction_x: NDArray[np.float64],
    direction_y: NDArray[np.float64],
    *,
    half_extent: float,
    uncertainty: float,
    threads: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NativeSquareCoverStats]:
    library = load_library()
    function = library.fast_math_square_weighted_scores_f64
    function.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(NativeSquareCoverStats),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int
    definite = np.empty(len(center_x), dtype=np.float64)
    possible = np.empty_like(definite)
    stats = NativeSquareCoverStats()
    error = ctypes.create_string_buffer(512)
    status = function(
        _pointer(points.reshape(-1), ctypes.c_double),
        _pointer(weights, ctypes.c_double),
        len(points),
        _pointer(center_x, ctypes.c_double),
        _pointer(center_y, ctypes.c_double),
        _pointer(direction_x, ctypes.c_double),
        _pointer(direction_y, ctypes.c_double),
        len(center_x),
        half_extent,
        uncertainty,
        threads,
        _pointer(definite, ctypes.c_double),
        _pointer(possible, ctypes.c_double),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(message or f"native square scores failed ({status})")
    return definite, possible, stats
