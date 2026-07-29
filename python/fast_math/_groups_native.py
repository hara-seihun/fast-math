from __future__ import annotations

import ctypes

import numpy as np
from numpy.typing import NDArray

from lambda_fast._native import NativeUnavailable, load_library


class NativeGroupStats(ctypes.Structure):
    _fields_ = [
        ("degree", ctypes.c_uint32),
        ("generator_count", ctypes.c_uint64),
        ("item_count", ctypes.c_uint64),
        ("orbit_count", ctypes.c_uint64),
        ("chain_level_count", ctypes.c_uint64),
        ("strong_generator_count", ctypes.c_uint64),
        ("thread_count", ctypes.c_uint32),
        ("elapsed_seconds", ctypes.c_double),
    ]


def _library() -> ctypes.CDLL:
    library = load_library()
    required = (
        "fast_math_permutation_orbits_u32",
        "fast_math_schreier_sims_u32",
        "fast_math_permutation_group_contains_u32",
    )
    if not all(hasattr(library, name) for name in required):
        raise NativeUnavailable(
            "fast-math group kernels are unavailable in the native library"
        )
    if not getattr(library, "_fast_math_groups_configured", False):
        uint32_pointer = ctypes.POINTER(ctypes.c_uint32)
        uint64_pointer = ctypes.POINTER(ctypes.c_uint64)
        uint8_pointer = ctypes.POINTER(ctypes.c_uint8)
        char_pointer = ctypes.POINTER(ctypes.c_char)

        library.fast_math_permutation_orbits_u32.argtypes = [
            uint32_pointer,
            ctypes.c_size_t,
            ctypes.c_uint32,
            uint32_pointer,
            uint32_pointer,
            ctypes.POINTER(NativeGroupStats),
            char_pointer,
            ctypes.c_size_t,
        ]
        library.fast_math_permutation_orbits_u32.restype = ctypes.c_int
        library.fast_math_schreier_sims_u32.argtypes = [
            uint32_pointer,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_size_t,
            uint32_pointer,
            uint32_pointer,
            uint64_pointer,
            ctypes.c_size_t,
            uint32_pointer,
            uint64_pointer,
            uint64_pointer,
            ctypes.POINTER(NativeGroupStats),
            char_pointer,
            ctypes.c_size_t,
        ]
        library.fast_math_schreier_sims_u32.restype = ctypes.c_int
        library.fast_math_permutation_group_contains_u32.argtypes = [
            uint32_pointer,
            ctypes.c_size_t,
            ctypes.c_uint32,
            uint32_pointer,
            ctypes.c_size_t,
            ctypes.c_uint32,
            uint8_pointer,
            ctypes.POINTER(NativeGroupStats),
            char_pointer,
            ctypes.c_size_t,
        ]
        library.fast_math_permutation_group_contains_u32.restype = ctypes.c_int
        library._fast_math_groups_configured = True
    return library


def _u32_pointer(array: NDArray[np.uint32]) -> ctypes.POINTER(ctypes.c_uint32):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))


def _u64_pointer(array: NDArray[np.uint64]) -> ctypes.POINTER(ctypes.c_uint64):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


def _raise_native(status: int, error: ctypes.Array[ctypes.c_char]) -> None:
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")


def permutation_orbits_native(
    generators: NDArray[np.uint32],
    degree: int,
) -> tuple[NDArray[np.uint32], int, NativeGroupStats]:
    labels = np.empty(degree, dtype=np.uint32)
    orbit_count = ctypes.c_uint32()
    stats = NativeGroupStats()
    error = ctypes.create_string_buffer(1024)
    status = _library().fast_math_permutation_orbits_u32(
        _u32_pointer(generators),
        len(generators),
        degree,
        _u32_pointer(labels),
        ctypes.byref(orbit_count),
        ctypes.byref(stats),
        error,
        len(error),
    )
    _raise_native(status, error)
    return labels, int(orbit_count.value), stats


def schreier_sims_native(
    generators: NDArray[np.uint32],
    degree: int,
) -> tuple[
    NDArray[np.uint32],
    NDArray[np.uint32],
    NDArray[np.uint64],
    NDArray[np.uint32],
    NativeGroupStats,
]:
    library = _library()
    base_count = ctypes.c_uint64()
    strong_count = ctypes.c_uint64()
    stats = NativeGroupStats()
    error = ctypes.create_string_buffer(1024)
    null_u32 = ctypes.POINTER(ctypes.c_uint32)()
    null_u64 = ctypes.POINTER(ctypes.c_uint64)()
    status = library.fast_math_schreier_sims_u32(
        _u32_pointer(generators),
        len(generators),
        degree,
        0,
        null_u32,
        null_u32,
        null_u64,
        0,
        null_u32,
        ctypes.byref(base_count),
        ctypes.byref(strong_count),
        ctypes.byref(stats),
        error,
        len(error),
    )
    _raise_native(status, error)

    base = np.empty(int(base_count.value), dtype=np.uint32)
    orbit_sizes = np.empty(int(base_count.value), dtype=np.uint32)
    offsets = np.empty(int(base_count.value) + 1, dtype=np.uint64)
    strong = np.empty(
        (int(strong_count.value), degree),
        dtype=np.uint32,
    )
    status = library.fast_math_schreier_sims_u32(
        _u32_pointer(generators),
        len(generators),
        degree,
        len(base),
        _u32_pointer(base),
        _u32_pointer(orbit_sizes),
        _u64_pointer(offsets),
        len(strong),
        _u32_pointer(strong),
        ctypes.byref(base_count),
        ctypes.byref(strong_count),
        ctypes.byref(stats),
        error,
        len(error),
    )
    _raise_native(status, error)
    return base, orbit_sizes, offsets, strong, stats


def permutation_group_contains_native(
    generators: NDArray[np.uint32],
    elements: NDArray[np.uint32],
    degree: int,
    threads: int,
) -> tuple[NDArray[np.bool_], NativeGroupStats]:
    output = np.empty(len(elements), dtype=np.uint8)
    stats = NativeGroupStats()
    error = ctypes.create_string_buffer(1024)
    status = _library().fast_math_permutation_group_contains_u32(
        _u32_pointer(generators),
        len(generators),
        degree,
        _u32_pointer(elements),
        len(elements),
        threads,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.byref(stats),
        error,
        len(error),
    )
    _raise_native(status, error)
    return output.view(np.bool_), stats
