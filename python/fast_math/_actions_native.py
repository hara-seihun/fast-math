from __future__ import annotations

import ctypes

import numpy as np
from numpy.typing import NDArray

from ._native import load_library


class NativeActionStats(ctypes.Structure):
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


def _u32(array: NDArray[np.uint32]):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))


def _u64(array: NDArray[np.uint64]):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


def _u8(array: NDArray[np.uint8]):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


class NativeSubsetActionPlan:
    def __init__(self, permutations: NDArray[np.uint32]) -> None:
        library = load_library()
        if not hasattr(library, "fast_math_subset_action_create_u32"):
            raise RuntimeError("native packed subset actions are unavailable")
        if not getattr(library, "_fast_math_subset_action_configured", False):
            library.fast_math_subset_action_create_u32.argtypes = [
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(NativeActionStats),
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            library.fast_math_subset_action_create_u32.restype = ctypes.c_int
            library.fast_math_subset_action_destroy.argtypes = [ctypes.c_void_p]
            library.fast_math_subset_action_destroy.restype = None
            library.fast_math_subset_action_canonicalize_u64.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(NativeActionStats),
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            library.fast_math_subset_action_canonicalize_u64.restype = ctypes.c_int
            library._fast_math_subset_action_configured = True
        handle = ctypes.c_void_p()
        stats = NativeActionStats()
        error = ctypes.create_string_buffer(1024)
        status = library.fast_math_subset_action_create_u32(
            _u32(permutations.reshape(-1)),
            len(permutations),
            permutations.shape[1],
            ctypes.byref(handle),
            ctypes.byref(stats),
            error,
            len(error),
        )
        if status != 0:
            raise RuntimeError(error.value.decode() or "native action plan creation failed")
        if not handle.value:
            raise RuntimeError("native action plan returned a null handle")
        self._library = library
        self._handle = handle
        self.setup_seconds = float(stats.elapsed_seconds)

    def canonicalize(
        self,
        masks: NDArray[np.uint64],
        *,
        threads: int,
        canonical_masks: bool = True,
    ) -> tuple[NDArray[np.uint64] | None, NDArray[np.bool_], NativeActionStats]:
        if not self._handle.value:
            raise RuntimeError("native action plan is closed")
        canonical = np.empty(len(masks), dtype=np.uint64) if canonical_masks else None
        flags = np.empty(len(masks), dtype=np.uint8)
        stats = NativeActionStats()
        error = ctypes.create_string_buffer(1024)
        canonical_pointer = (
            _u64(canonical)
            if canonical is not None
            else ctypes.POINTER(ctypes.c_uint64)()
        )
        status = self._library.fast_math_subset_action_canonicalize_u64(
            self._handle,
            _u64(masks),
            len(masks),
            threads,
            canonical_pointer,
            _u8(flags),
            ctypes.byref(stats),
            error,
            len(error),
        )
        if status != 0:
            raise RuntimeError(error.value.decode() or "native action application failed")
        return canonical, flags.view(np.bool_), stats

    def close(self) -> None:
        handle, self._handle = self._handle, ctypes.c_void_p()
        if handle.value:
            self._library.fast_math_subset_action_destroy(handle)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
