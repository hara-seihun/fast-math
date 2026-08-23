"""ctypes binding for packed cyclic correlation profiles."""

from __future__ import annotations

import ctypes
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from ._native import NativeUnavailable, _check, _uint64_pointer, load_library


class NativeCyclicProfileStats(ctypes.Structure):
    _fields_ = [
        ("mask_count", ctypes.c_uint64),
        ("popcount_evaluations", ctypes.c_uint64),
        ("bit_width", ctypes.c_uint32),
        ("lag_count", ctypes.c_uint32),
        ("worker_count", ctypes.c_uint32),
        ("elapsed_seconds", ctypes.c_double),
    ]


@lru_cache(maxsize=1)
def _native_function():
    library = load_library()
    if not hasattr(library, "fast_math_cyclic_correlation_profiles_u64"):
        raise NativeUnavailable(
            "fast-math was built without cyclic correlation profiles"
        )
    function = library.fast_math_cyclic_correlation_profiles_u64
    function.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(NativeCyclicProfileStats),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int
    return function


def cyclic_correlation_profiles_native(
    masks: NDArray[np.uint64],
    bit_width: int,
    threads: int,
) -> tuple[
    NDArray[np.uint8],
    NDArray[np.int16],
    NativeCyclicProfileStats,
]:
    intersections = np.empty((len(masks), bit_width), dtype=np.uint8)
    correlations = np.empty((len(masks), bit_width), dtype=np.int16)
    stats = NativeCyclicProfileStats()
    error = ctypes.create_string_buffer(1024)
    status = _native_function()(
        _uint64_pointer(masks),
        len(masks),
        bit_width,
        threads,
        intersections.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        correlations.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        ctypes.byref(stats),
        error,
        len(error),
    )
    _check(status, error)
    return intersections, correlations, stats
