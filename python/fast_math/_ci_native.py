from __future__ import annotations

import ctypes

import numpy as np
from numpy.typing import NDArray

from lambda_fast._native import NativeUnavailable, load_library


class NativeCIStats(ctypes.Structure):
    _fields_ = [
        ("degree", ctypes.c_uint32),
        ("generator_count", ctypes.c_uint64),
        ("item_count", ctypes.c_uint64),
        ("class_count", ctypes.c_uint64),
        ("relation_count", ctypes.c_uint64),
        ("iteration_count", ctypes.c_uint64),
        ("thread_count", ctypes.c_uint32),
        ("elapsed_seconds", ctypes.c_double),
    ]


def _library() -> ctypes.CDLL:
    library = load_library()
    required = (
        "fast_math_permutation_double_cosets_u32",
        "fast_math_subset_orbits_u64",
        "fast_math_cayley_graphs_u32",
        "fast_math_derivative_orbits_u32",
        "fast_math_wl2_refine_u32",
        "fast_math_intersection_numbers_u64",
    )
    if not all(hasattr(library, name) for name in required):
        raise NativeUnavailable(
            "fast-math CI kernels are unavailable in the native library"
        )
    if not getattr(library, "_fast_math_ci_configured", False):
        u8 = ctypes.POINTER(ctypes.c_uint8)
        u32 = ctypes.POINTER(ctypes.c_uint32)
        u64 = ctypes.POINTER(ctypes.c_uint64)
        char = ctypes.POINTER(ctypes.c_char)
        stats = ctypes.POINTER(NativeCIStats)

        library.fast_math_permutation_double_cosets_u32.argtypes = [
            u32,
            ctypes.c_size_t,
            u32,
            ctypes.c_size_t,
            u32,
            ctypes.c_size_t,
            ctypes.c_uint32,
            u64,
            u64,
            u64,
            u64,
            stats,
            char,
            ctypes.c_size_t,
        ]
        library.fast_math_permutation_double_cosets_u32.restype = ctypes.c_int
        library.fast_math_subset_orbits_u64.argtypes = [
            u64,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            u32,
            ctypes.c_size_t,
            u64,
            u64,
            u64,
            u64,
            stats,
            char,
            ctypes.c_size_t,
        ]
        library.fast_math_subset_orbits_u64.restype = ctypes.c_int
        library.fast_math_cayley_graphs_u32.argtypes = [
            u32,
            ctypes.c_uint32,
            u64,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            u64,
            stats,
            char,
            ctypes.c_size_t,
        ]
        library.fast_math_cayley_graphs_u32.restype = ctypes.c_int
        library.fast_math_derivative_orbits_u32.argtypes = [
            u32,
            u32,
            u32,
            ctypes.c_uint32,
            u32,
            u32,
            u32,
            stats,
            char,
            ctypes.c_size_t,
        ]
        library.fast_math_derivative_orbits_u32.restype = ctypes.c_int
        library.fast_math_wl2_refine_u32.argtypes = [
            u32,
            ctypes.c_uint32,
            u32,
            u32,
            stats,
            char,
            ctypes.c_size_t,
        ]
        library.fast_math_wl2_refine_u32.restype = ctypes.c_int
        library.fast_math_intersection_numbers_u64.argtypes = [
            u32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            u64,
            stats,
            char,
            ctypes.c_size_t,
        ]
        library.fast_math_intersection_numbers_u64.restype = ctypes.c_int
        library._fast_math_ci_configured = True
    return library


def _u32(array: NDArray[np.uint32]) -> ctypes.POINTER(ctypes.c_uint32):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))


def _u64(array: NDArray[np.uint64]) -> ctypes.POINTER(ctypes.c_uint64):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


def _raise(status: int, error: ctypes.Array[ctypes.c_char]) -> None:
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")


def permutation_double_cosets_native(
    candidates: NDArray[np.uint32],
    left_generators: NDArray[np.uint32],
    right_generators: NDArray[np.uint32],
) -> tuple[
    NDArray[np.uint64],
    NDArray[np.uint64],
    NDArray[np.uint64],
    NativeCIStats,
]:
    count, degree = candidates.shape
    class_ids = np.empty(count, dtype=np.uint64)
    representatives = np.empty(count, dtype=np.uint64)
    sizes = np.empty(count, dtype=np.uint64)
    class_count = ctypes.c_uint64()
    stats = NativeCIStats()
    error = ctypes.create_string_buffer(1024)
    status = _library().fast_math_permutation_double_cosets_u32(
        _u32(candidates),
        count,
        _u32(left_generators),
        len(left_generators),
        _u32(right_generators),
        len(right_generators),
        degree,
        _u64(class_ids),
        _u64(representatives),
        _u64(sizes),
        ctypes.byref(class_count),
        ctypes.byref(stats),
        error,
        len(error),
    )
    _raise(status, error)
    classes = int(class_count.value)
    return class_ids, representatives[:classes], sizes[:classes], stats


def subset_orbits_native(
    subset_words: NDArray[np.uint64],
    action_generators: NDArray[np.uint32],
    atom_count: int,
) -> tuple[
    NDArray[np.uint64],
    NDArray[np.uint64],
    NDArray[np.uint64],
    NativeCIStats,
]:
    subset_count, word_count = subset_words.shape
    class_ids = np.empty(subset_count, dtype=np.uint64)
    representatives = np.empty(subset_count, dtype=np.uint64)
    sizes = np.empty(subset_count, dtype=np.uint64)
    class_count = ctypes.c_uint64()
    stats = NativeCIStats()
    error = ctypes.create_string_buffer(1024)
    status = _library().fast_math_subset_orbits_u64(
        _u64(subset_words),
        subset_count,
        word_count,
        atom_count,
        _u32(action_generators),
        len(action_generators),
        _u64(class_ids),
        _u64(representatives),
        _u64(sizes),
        ctypes.byref(class_count),
        ctypes.byref(stats),
        error,
        len(error),
    )
    _raise(status, error)
    classes = int(class_count.value)
    return class_ids, representatives[:classes], sizes[:classes], stats


def cayley_graphs_native(
    multiplication_table: NDArray[np.uint32],
    connection_words: NDArray[np.uint64],
    threads: int,
) -> tuple[NDArray[np.uint64], NativeCIStats]:
    order = len(multiplication_table)
    word_count = connection_words.shape[1]
    adjacency = np.empty(
        (len(connection_words), order, word_count),
        dtype=np.uint64,
    )
    stats = NativeCIStats()
    error = ctypes.create_string_buffer(1024)
    status = _library().fast_math_cayley_graphs_u32(
        _u32(multiplication_table),
        order,
        _u64(connection_words),
        len(connection_words),
        word_count,
        threads,
        _u64(adjacency),
        ctypes.byref(stats),
        error,
        len(error),
    )
    _raise(status, error)
    return adjacency, stats


def derivative_orbits_native(
    multiplication_table: NDArray[np.uint32],
    inverse_indices: NDArray[np.uint32],
    bijection: NDArray[np.uint32],
) -> tuple[
    NDArray[np.uint32],
    NDArray[np.uint32],
    int,
    NativeCIStats,
]:
    order = len(multiplication_table)
    generators = np.empty((order, order), dtype=np.uint32)
    labels = np.empty(order, dtype=np.uint32)
    orbit_count = ctypes.c_uint32()
    stats = NativeCIStats()
    error = ctypes.create_string_buffer(1024)
    status = _library().fast_math_derivative_orbits_u32(
        _u32(multiplication_table),
        _u32(inverse_indices),
        _u32(bijection),
        order,
        _u32(generators),
        _u32(labels),
        ctypes.byref(orbit_count),
        ctypes.byref(stats),
        error,
        len(error),
    )
    _raise(status, error)
    return generators, labels, int(orbit_count.value), stats


def wl2_refine_native(
    initial_relations: NDArray[np.uint32],
) -> tuple[NDArray[np.uint32], int, NativeCIStats]:
    vertex_count = len(initial_relations)
    stable = np.empty_like(initial_relations)
    relation_count = ctypes.c_uint32()
    stats = NativeCIStats()
    error = ctypes.create_string_buffer(1024)
    status = _library().fast_math_wl2_refine_u32(
        _u32(initial_relations),
        vertex_count,
        _u32(stable),
        ctypes.byref(relation_count),
        ctypes.byref(stats),
        error,
        len(error),
    )
    _raise(status, error)
    return stable, int(relation_count.value), stats


def intersection_numbers_native(
    relations: NDArray[np.uint32],
    relation_count: int,
) -> tuple[NDArray[np.uint64], NativeCIStats]:
    tensor = np.empty(
        (relation_count, relation_count, relation_count),
        dtype=np.uint64,
    )
    stats = NativeCIStats()
    error = ctypes.create_string_buffer(1024)
    status = _library().fast_math_intersection_numbers_u64(
        _u32(relations),
        len(relations),
        relation_count,
        _u64(tensor),
        ctypes.byref(stats),
        error,
        len(error),
    )
    _raise(status, error)
    return tensor, stats
