from __future__ import annotations

import ctypes

import numpy as np
from numpy.typing import NDArray

from ._native import NativeUnavailable, load_library


class NativeCnfStats(ctypes.Structure):
    _fields_ = [
        ("variable_count", ctypes.c_uint32),
        ("clause_count", ctypes.c_uint64),
        ("literal_count", ctypes.c_uint64),
        ("assignment_count", ctypes.c_uint64),
        ("inspected_literal_count", ctypes.c_uint64),
        ("thread_count", ctypes.c_uint32),
        ("elapsed_seconds", ctypes.c_double),
    ]


def cnf_native_available() -> bool:
    try:
        return hasattr(load_library(), "fast_math_cnf_create_i32")
    except (NativeUnavailable, OSError):
        return False


def _u64(array: NDArray[np.uint64]):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


def _i32(array: NDArray[np.int32]):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _i64(array: NDArray[np.int64]):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))


def _u8(array: NDArray[np.uint8]):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


class NativeCnfPlan:
    def __init__(
        self,
        clause_offsets: NDArray[np.uint64],
        literals: NDArray[np.int32],
        variable_count: int,
    ) -> None:
        library = load_library()
        if not hasattr(library, "fast_math_cnf_create_i32"):
            raise NativeUnavailable("native CNF plans are unavailable")
        if not getattr(library, "_fast_math_cnf_configured", False):
            library.fast_math_cnf_create_i32.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            library.fast_math_cnf_create_i32.restype = ctypes.c_int
            library.fast_math_cnf_destroy.argtypes = [ctypes.c_void_p]
            library.fast_math_cnf_destroy.restype = None
            library.fast_math_cnf_evaluate_u64.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(NativeCnfStats),
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            library.fast_math_cnf_evaluate_u64.restype = ctypes.c_int
            library._fast_math_cnf_configured = True
        handle = ctypes.c_void_p()
        error = ctypes.create_string_buffer(1024)
        literal_pointer = (
            _i32(literals)
            if len(literals)
            else ctypes.POINTER(ctypes.c_int32)()
        )
        status = library.fast_math_cnf_create_i32(
            _u64(clause_offsets),
            len(clause_offsets) - 1,
            literal_pointer,
            len(literals),
            variable_count,
            ctypes.byref(handle),
            error,
            len(error),
        )
        if status != 0:
            raise RuntimeError(error.value.decode() or "native CNF plan creation failed")
        if not handle.value:
            raise RuntimeError("native CNF plan returned a null handle")
        self._library = library
        self._handle = handle

    def evaluate(
        self,
        assignments: NDArray[np.uint64],
        *,
        threads: int,
    ) -> tuple[NDArray[np.bool_], NDArray[np.int64], NativeCnfStats]:
        if not self._handle.value:
            raise RuntimeError("native CNF plan is closed")
        satisfied = np.empty(len(assignments), dtype=np.uint8)
        first = np.empty(len(assignments), dtype=np.int64)
        stats = NativeCnfStats()
        error = ctypes.create_string_buffer(1024)
        status = self._library.fast_math_cnf_evaluate_u64(
            self._handle,
            _u64(assignments),
            len(assignments),
            assignments.shape[1],
            threads,
            _u8(satisfied),
            _i64(first),
            ctypes.byref(stats),
            error,
            len(error),
        )
        if status != 0:
            raise RuntimeError(error.value.decode() or "native CNF evaluation failed")
        return satisfied.view(np.bool_), first, stats

    def close(self) -> None:
        handle, self._handle = self._handle, ctypes.c_void_p()
        if handle.value:
            self._library.fast_math_cnf_destroy(handle)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
