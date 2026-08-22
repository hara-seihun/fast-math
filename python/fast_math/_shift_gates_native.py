"""Native bridge for the shift-gate scan (see shift_gates.py)."""

from __future__ import annotations

import ctypes
import os

import numpy as np

from ._native import NativeUnavailable, load_library

_DECLARED = False


def _declare(library: ctypes.CDLL) -> None:
    global _DECLARED
    if _DECLARED:
        return
    library.fast_math_shift_gate_scan_u64.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    library.fast_math_shift_gate_scan_u64.restype = ctypes.c_int
    _DECLARED = True


def _u64_pointer(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


def scan_native(plan, v_start: int, v_count: int, *, survivor_capacity: int = 1 << 20):
    from .shift_gates import ShiftGateStats

    library = load_library()
    _declare(library)
    gate = plan.gate

    form_a = np.array([f.a for f in gate.forms], dtype=np.uint64)
    form_b = np.array([f.b for f in gate.forms], dtype=np.uint64)
    smooth = np.array(gate.smooth_primes, dtype=np.uint32)
    lut_moduli, lut_offsets, lut_bits = plan.packed_luts()
    classes = plan.wheel_classes

    survivors = np.zeros(survivor_capacity, dtype=np.uint64)
    count = ctypes.c_size_t(0)
    stats = np.zeros(4, dtype=np.uint64)
    error = ctypes.create_string_buffer(256)
    thread_count = int(os.environ.get("FAST_MATH_THREADS", "0")) or (os.cpu_count() or 1)

    status = library.fast_math_shift_gate_scan_u64(
        _u64_pointer(form_a),
        _u64_pointer(form_b),
        len(form_a),
        smooth.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        len(smooth),
        _u64_pointer(lut_moduli),
        _u64_pointer(lut_offsets),
        len(lut_moduli),
        _u64_pointer(lut_bits),
        ctypes.c_uint64(plan.wheel),
        _u64_pointer(classes),
        len(classes),
        ctypes.c_uint32(plan.sieve_low),
        ctypes.c_uint32(plan.sieve_bound),
        ctypes.c_uint64(v_start),
        ctypes.c_uint64(v_count),
        ctypes.c_uint32(thread_count),
        _u64_pointer(survivors),
        survivor_capacity,
        ctypes.byref(count),
        _u64_pointer(stats),
        error,
        ctypes.sizeof(error),
    )
    if status != 0:
        raise NativeUnavailable(error.value.decode() or f"scan failed with status {status}")
    return (
        survivors[: count.value].copy(),
        ShiftGateStats(
            scanned=int(stats[0]),
            wheel_alive=int(stats[2]),
            sieve_survivors=int(stats[1]),
            survivors=int(stats[3]),
        ),
    )
