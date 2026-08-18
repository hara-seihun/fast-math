from __future__ import annotations

import ctypes

import numpy as np
from numpy.typing import NDArray

from ._native import NativeUnavailable, load_library


class NativeModularStats(ctypes.Structure):
    _fields_ = [
        ("batch_count", ctypes.c_uint64),
        ("item_count", ctypes.c_uint64),
        ("operation_count", ctypes.c_uint64),
        ("prime", ctypes.c_uint32),
        ("thread_count", ctypes.c_uint32),
        ("elapsed_seconds", ctypes.c_double),
    ]


def modular_native_available() -> bool:
    try:
        return hasattr(load_library(), "fast_math_polynomial_evaluate_mod_u32")
    except (NativeUnavailable, OSError):
        return False


def _u32(array: NDArray[np.uint32]):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))


def _library() -> ctypes.CDLL:
    library = load_library()
    if not hasattr(library, "fast_math_polynomial_evaluate_mod_u32"):
        raise NativeUnavailable("native modular batch kernels are unavailable")
    if not getattr(library, "_fast_math_modular_configured", False):
        u32 = ctypes.POINTER(ctypes.c_uint32)
        stats = ctypes.POINTER(NativeModularStats)
        char = ctypes.POINTER(ctypes.c_char)
        library.fast_math_polynomial_evaluate_mod_u32.argtypes = [
            u32,
            ctypes.c_size_t,
            ctypes.c_size_t,
            u32,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            u32,
            u32,
            stats,
            char,
            ctypes.c_size_t,
        ]
        library.fast_math_polynomial_evaluate_mod_u32.restype = ctypes.c_int
        library.fast_math_determinants_mod_u32.argtypes = [
            u32,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            u32,
            stats,
            char,
            ctypes.c_size_t,
        ]
        library.fast_math_determinants_mod_u32.restype = ctypes.c_int
        library._fast_math_modular_configured = True
    return library


def polynomial_evaluate_native(
    coefficients: NDArray[np.uint32],
    points: NDArray[np.uint32],
    *,
    prime: int,
    threads: int,
    with_derivative: bool,
) -> tuple[
    NDArray[np.uint32],
    NDArray[np.uint32] | None,
    NativeModularStats,
]:
    polynomial_count, coefficient_count = coefficients.shape
    values = np.empty((polynomial_count, len(points)), dtype=np.uint32)
    derivatives = np.empty_like(values) if with_derivative else None
    derivative_pointer = (
        _u32(derivatives)
        if derivatives is not None
        else ctypes.POINTER(ctypes.c_uint32)()
    )
    stats = NativeModularStats()
    error = ctypes.create_string_buffer(1024)
    status = _library().fast_math_polynomial_evaluate_mod_u32(
        _u32(coefficients),
        polynomial_count,
        coefficient_count,
        _u32(points),
        len(points),
        prime,
        threads,
        _u32(values),
        derivative_pointer,
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return values, derivatives, stats


def determinants_native(
    matrices: NDArray[np.uint32],
    *,
    prime: int,
    threads: int,
) -> tuple[NDArray[np.uint32], NativeModularStats]:
    matrix_count, order, _ = matrices.shape
    determinants = np.empty(matrix_count, dtype=np.uint32)
    stats = NativeModularStats()
    error = ctypes.create_string_buffer(1024)
    status = _library().fast_math_determinants_mod_u32(
        _u32(matrices),
        matrix_count,
        order,
        prime,
        threads,
        _u32(determinants),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return determinants, stats
