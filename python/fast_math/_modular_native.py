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


class NativeModularLinearStats(ctypes.Structure):
    _fields_ = [
        ("row_count", ctypes.c_uint64),
        ("column_count", ctypes.c_uint64),
        ("rank", ctypes.c_uint64),
        ("batch_count", ctypes.c_uint64),
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


def modular_linear_native_available() -> bool:
    try:
        return hasattr(
            load_library(), "fast_math_modular_linear_system_create_u32"
        )
    except (NativeUnavailable, OSError):
        return False


def _u32(array: NDArray[np.uint32]):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))


def _i64(array: NDArray[np.int64]):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))


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
        if hasattr(library, "fast_math_modular_linear_system_create_u32"):
            linear_stats = ctypes.POINTER(NativeModularLinearStats)
            void_handle = ctypes.POINTER(ctypes.c_void_p)
            library.fast_math_modular_linear_system_create_u32.argtypes = [
                u32,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_uint32,
                void_handle,
                linear_stats,
                char,
                ctypes.c_size_t,
            ]
            library.fast_math_modular_linear_system_create_u32.restype = (
                ctypes.c_int
            )
            library.fast_math_modular_linear_system_destroy.argtypes = [
                ctypes.c_void_p
            ]
            library.fast_math_modular_linear_system_destroy.restype = None
            library.fast_math_modular_linear_system_export_u32.argtypes = [
                ctypes.c_void_p,
                u32,
                u32,
                u32,
                u32,
                u32,
                char,
                ctypes.c_size_t,
            ]
            library.fast_math_modular_linear_system_export_u32.restype = (
                ctypes.c_int
            )
            library.fast_math_modular_linear_system_solve_u32.argtypes = [
                ctypes.c_void_p,
                u32,
                ctypes.c_size_t,
                ctypes.c_uint32,
                u32,
                ctypes.POINTER(ctypes.c_int64),
                linear_stats,
                char,
                ctypes.c_size_t,
            ]
            library.fast_math_modular_linear_system_solve_u32.restype = (
                ctypes.c_int
            )
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


class NativeModularLinearSystem:
    def __init__(self, matrix: NDArray[np.uint32], *, prime: int) -> None:
        self.row_count, self.column_count = matrix.shape
        self.prime = prime
        self._library = _library()
        if not hasattr(
            self._library, "fast_math_modular_linear_system_create_u32"
        ):
            raise NativeUnavailable(
                "native modular linear-system kernels are unavailable"
            )
        handle = ctypes.c_void_p()
        stats = NativeModularLinearStats()
        error = ctypes.create_string_buffer(1024)
        status = self._library.fast_math_modular_linear_system_create_u32(
            _u32(matrix),
            self.row_count,
            self.column_count,
            self.prime,
            ctypes.byref(handle),
            ctypes.byref(stats),
            error,
            len(error),
        )
        if status != 0:
            message = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(f"fast-math native error {status}: {message}")
        if not handle.value:
            raise RuntimeError("native modular linear system returned a null handle")
        self._handle = handle
        self.setup_stats = stats
        self.rank = int(stats.rank)
        self.reduced_row_echelon = np.empty_like(matrix)
        self.pivot_columns = np.empty(self.rank, dtype=np.uint32)
        self.solution_operator = np.empty(
            (self.column_count, self.row_count), dtype=np.uint32
        )
        self.right_nullspace = np.empty(
            (self.column_count - self.rank, self.column_count),
            dtype=np.uint32,
        )
        self.left_nullspace = np.empty(
            (self.row_count - self.rank, self.row_count), dtype=np.uint32
        )
        error = ctypes.create_string_buffer(1024)
        status = self._library.fast_math_modular_linear_system_export_u32(
            self._handle,
            _u32(self.reduced_row_echelon),
            _u32(self.pivot_columns),
            _u32(self.solution_operator),
            _u32(self.right_nullspace),
            _u32(self.left_nullspace),
            error,
            len(error),
        )
        if status != 0:
            self.close()
            message = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(f"fast-math native error {status}: {message}")

    def solve(
        self,
        right_hand_sides: NDArray[np.uint32],
        *,
        threads: int,
    ) -> tuple[
        NDArray[np.uint32],
        NDArray[np.int64],
        NativeModularLinearStats,
    ]:
        if not self._handle.value:
            raise RuntimeError("native modular linear system is closed")
        solutions = np.empty(
            (len(right_hand_sides), self.column_count), dtype=np.uint32
        )
        inconsistency_rows = np.empty(len(right_hand_sides), dtype=np.int64)
        stats = NativeModularLinearStats()
        error = ctypes.create_string_buffer(1024)
        status = self._library.fast_math_modular_linear_system_solve_u32(
            self._handle,
            _u32(right_hand_sides),
            len(right_hand_sides),
            threads,
            _u32(solutions),
            _i64(inconsistency_rows),
            ctypes.byref(stats),
            error,
            len(error),
        )
        if status != 0:
            message = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(f"fast-math native error {status}: {message}")
        return solutions, inconsistency_rows, stats

    def close(self) -> None:
        handle, self._handle = self._handle, ctypes.c_void_p()
        if handle.value:
            self._library.fast_math_modular_linear_system_destroy(handle)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
