from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import time

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .affine import (
    AffineContourMetrics,
    affine_batches,
    affine_inputs,
    affine_steps,
    edge_bounds,
)
from ._field import field_array_u32, prime_u32


class HipUnavailable(RuntimeError):
    """Raised when the optional ROCm/HIP backend is unavailable."""


@lru_cache(maxsize=1)
def _library() -> ctypes.CDLL:
    candidates = []
    configured = os.environ.get("FAST_MATH_HIP_LIBRARY")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path(__file__).resolve().parents[2] / "build" / "libfast_math_hip.so",
            Path(__file__).resolve().parents[2] / "build" / "libfast_math_hip.dylib",
        ]
    )
    for path in candidates:
        if path.exists():
            try:
                library = ctypes.CDLL(str(path))
            except OSError as error:
                raise HipUnavailable(
                    f"could not load HIP library {path}: {error}"
                ) from error
            library.fast_math_hip_create.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            library.fast_math_hip_create.restype = ctypes.c_int
            library.fast_math_hip_destroy.argtypes = [ctypes.c_void_p]
            library.fast_math_hip_destroy.restype = ctypes.c_int
            library.fast_math_hip_evaluate.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
            ]
            library.fast_math_hip_evaluate.restype = ctypes.c_int
            library.fast_math_hip_metrics.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            library.fast_math_hip_metrics.restype = ctypes.c_int
            if hasattr(library, "fast_math_hip_square_cover_create"):
                library.fast_math_hip_square_cover_create.argtypes = [
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                library.fast_math_hip_square_cover_create.restype = ctypes.c_int
                library.fast_math_hip_square_cover_destroy.argtypes = [
                    ctypes.c_void_p
                ]
                library.fast_math_hip_square_cover_destroy.restype = ctypes.c_int
                library.fast_math_hip_square_cover_evaluate.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_uint64),
                ]
                library.fast_math_hip_square_cover_evaluate.restype = ctypes.c_int
                library.fast_math_hip_square_weighted_evaluate.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                ]
                library.fast_math_hip_square_weighted_evaluate.restype = ctypes.c_int
            if hasattr(library, "fast_math_hip_cnf_create"):
                library.fast_math_hip_cnf_create.argtypes = [
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_int32),
                    ctypes.c_size_t,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                library.fast_math_hip_cnf_create.restype = ctypes.c_int
                library.fast_math_hip_cnf_destroy.argtypes = [ctypes.c_void_p]
                library.fast_math_hip_cnf_destroy.restype = ctypes.c_int
                library.fast_math_hip_cnf_evaluate.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.c_size_t,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_int64),
                ]
                library.fast_math_hip_cnf_evaluate.restype = ctypes.c_int
            if hasattr(library, "fast_math_hip_polynomial_create"):
                library.fast_math_hip_polynomial_create.argtypes = [
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                library.fast_math_hip_polynomial_create.restype = ctypes.c_int
                library.fast_math_hip_polynomial_destroy.argtypes = [ctypes.c_void_p]
                library.fast_math_hip_polynomial_destroy.restype = ctypes.c_int
                library.fast_math_hip_polynomial_evaluate.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_size_t,
                    ctypes.c_uint8,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_uint32),
                ]
                library.fast_math_hip_polynomial_evaluate.restype = ctypes.c_int
                library.fast_math_hip_determinant_create.argtypes = [
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                library.fast_math_hip_determinant_create.restype = ctypes.c_int
                library.fast_math_hip_determinant_destroy.argtypes = [ctypes.c_void_p]
                library.fast_math_hip_determinant_destroy.restype = ctypes.c_int
                library.fast_math_hip_determinants.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_uint32),
                ]
                library.fast_math_hip_determinants.restype = ctypes.c_int
            if hasattr(library, "fast_math_hip_linear_system_create"):
                library.fast_math_hip_linear_system_create.argtypes = [
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                library.fast_math_hip_linear_system_create.restype = ctypes.c_int
                library.fast_math_hip_linear_system_destroy.argtypes = [
                    ctypes.c_void_p
                ]
                library.fast_math_hip_linear_system_destroy.restype = ctypes.c_int
                library.fast_math_hip_linear_system_solve.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_int64),
                ]
                library.fast_math_hip_linear_system_solve.restype = ctypes.c_int
            if hasattr(library, "fast_math_hip_subset_action_create"):
                library.fast_math_hip_subset_action_create.argtypes = [
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_size_t,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                library.fast_math_hip_subset_action_create.restype = ctypes.c_int
                library.fast_math_hip_subset_action_destroy.argtypes = [
                    ctypes.c_void_p
                ]
                library.fast_math_hip_subset_action_destroy.restype = ctypes.c_int
                library.fast_math_hip_subset_action_canonicalize.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.POINTER(ctypes.c_uint8),
                ]
                library.fast_math_hip_subset_action_canonicalize.restype = ctypes.c_int
                library.fast_math_hip_subset_action_is_canonical.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_uint8),
                ]
                library.fast_math_hip_subset_action_is_canonical.restype = ctypes.c_int
            return library
    raise HipUnavailable(
        "HIP support requires build/libfast_math_hip.so; run `make hip`"
    )


def hip_available() -> bool:
    try:
        _library()
    except HipUnavailable:
        return False
    return True


def hip_cnf_available() -> bool:
    try:
        return hasattr(_library(), "fast_math_hip_cnf_create")
    except HipUnavailable:
        return False


def hip_modular_available() -> bool:
    try:
        return hasattr(_library(), "fast_math_hip_polynomial_create")
    except HipUnavailable:
        return False


def hip_modular_linear_available() -> bool:
    try:
        return hasattr(_library(), "fast_math_hip_linear_system_create")
    except HipUnavailable:
        return False


def hip_subset_actions_available() -> bool:
    try:
        return hasattr(_library(), "fast_math_hip_subset_action_create")
    except HipUnavailable:
        return False


def _pointer(array: NDArray[np.generic], dtype) -> ctypes.Array:
    return array.ctypes.data_as(ctypes.POINTER(dtype))


def _check(code: int, operation: str) -> None:
    if code != 0:
        raise RuntimeError(f"HIP {operation} failed with status {code}")


class CnfHipPlan:
    """Persistent HIP clause storage for exact assignment checking."""

    backend = "hip"

    def __init__(
        self,
        clause_offsets: ArrayLike,
        literals: ArrayLike,
        variable_count: int,
    ) -> None:
        if (
            not isinstance(variable_count, (int, np.integer))
            or not 1 <= int(variable_count) <= np.iinfo(np.uint32).max
        ):
            raise ValueError("variable_count must be a positive uint32")
        self.variable_count = int(variable_count)
        offsets_raw = np.asarray(clause_offsets)
        literals_raw = np.asarray(literals)
        if offsets_raw.ndim != 1 or not np.issubdtype(offsets_raw.dtype, np.integer):
            raise ValueError("clause_offsets must be a one-dimensional integer array")
        if literals_raw.ndim != 1 or not np.issubdtype(literals_raw.dtype, np.integer):
            raise ValueError("literals must be a one-dimensional integer array")
        if np.issubdtype(offsets_raw.dtype, np.signedinteger) and np.any(offsets_raw < 0):
            raise ValueError("clause offsets must be nonnegative")
        if (
            np.any(literals_raw < np.iinfo(np.int32).min)
            or np.any(literals_raw > np.iinfo(np.int32).max)
        ):
            raise ValueError("CNF literal must fit int32")
        offsets = np.array(offsets_raw, dtype=np.uint64, order="C", copy=True)
        literals_array = np.array(literals_raw, dtype=np.int64, order="C", copy=True)
        if (
            len(offsets) == 0
            or offsets[0] != 0
            or offsets[-1] != len(literals_array)
            or np.any(offsets[1:] < offsets[:-1])
            or np.any(literals_array == 0)
            or np.any(np.abs(literals_array) > self.variable_count)
        ):
            raise ValueError("CNF offsets or literals are invalid")
        literals_i32 = literals_array.astype(np.int32)
        library = _library()
        if not hasattr(library, "fast_math_hip_cnf_create"):
            raise HipUnavailable("HIP CNF support is not built")
        handle = ctypes.c_void_p()
        literal_pointer = (
            _pointer(literals_i32, ctypes.c_int32)
            if len(literals_i32)
            else ctypes.POINTER(ctypes.c_int32)()
        )
        _check(
            library.fast_math_hip_cnf_create(
                _pointer(offsets, ctypes.c_uint64),
                len(offsets) - 1,
                literal_pointer,
                len(literals_i32),
                self.variable_count,
                ctypes.byref(handle),
            ),
            "CNF plan creation",
        )
        if not handle.value:
            raise RuntimeError("HIP CNF plan returned a null handle")
        self._library = library
        self._handle = handle
        self._clause_offsets = offsets
        self._literals = literals_i32

    @property
    def word_count(self) -> int:
        return (self.variable_count + 63) // 64

    def evaluate(
        self,
        assignments: ArrayLike,
    ) -> tuple[NDArray[np.int64], float]:
        if not self._handle.value:
            raise RuntimeError("HIP CNF plan is closed")
        raw = np.asarray(assignments)
        if (
            raw.ndim != 2
            or raw.shape[1] != self.word_count
            or not np.issubdtype(raw.dtype, np.integer)
        ):
            raise ValueError("assignments have an invalid packed shape")
        if np.issubdtype(raw.dtype, np.signedinteger) and np.any(raw < 0):
            raise ValueError("assignment words must be nonnegative")
        prepared = np.ascontiguousarray(raw, dtype=np.uint64)
        final_bits = self.variable_count % 64
        if final_bits and np.any(prepared[:, -1] >> np.uint64(final_bits)):
            raise ValueError("assignment contains an out-of-range bit")
        first = np.empty(len(prepared), dtype=np.int64)
        started = time.perf_counter()
        _check(
            self._library.fast_math_hip_cnf_evaluate(
                self._handle,
                _pointer(prepared, ctypes.c_uint64),
                len(prepared),
                self.word_count,
                _pointer(first, ctypes.c_int64),
            ),
            "CNF evaluation",
        )
        return first, time.perf_counter() - started

    def close(self) -> None:
        handle, self._handle = self._handle, ctypes.c_void_p()
        if handle.value:
            _check(
                self._library.fast_math_hip_cnf_destroy(handle),
                "CNF plan destruction",
            )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class ModularPolynomialHipPlan:
    """Persistent HIP coefficients for exact prime-field evaluation."""

    backend = "hip"

    def __init__(self, coefficients: ArrayLike, prime: int) -> None:
        self.prime = prime_u32(prime)
        values = field_array_u32(
            coefficients, self.prime, dimensions=2, own=True
        )
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("coefficients must have nonzero shape")
        library = _library()
        if not hasattr(library, "fast_math_hip_polynomial_create"):
            raise HipUnavailable("HIP modular polynomial support is not built")
        handle = ctypes.c_void_p()
        _check(
            library.fast_math_hip_polynomial_create(
                _pointer(values.reshape(-1), ctypes.c_uint32),
                values.shape[0],
                values.shape[1],
                self.prime,
                ctypes.byref(handle),
            ),
            "modular polynomial plan creation",
        )
        if not handle.value:
            raise RuntimeError("HIP modular polynomial plan returned a null handle")
        self._library = library
        self._handle = handle
        self._coefficients = values

    @property
    def polynomial_count(self) -> int:
        return self._coefficients.shape[0]

    def evaluate(
        self,
        points: ArrayLike,
        *,
        derivative: bool,
    ) -> tuple[NDArray[np.uint32], NDArray[np.uint32] | None, float]:
        if not self._handle.value:
            raise RuntimeError("HIP modular polynomial plan is closed")
        prepared = field_array_u32(
            points, self.prime, dimensions=1, own=False
        )
        values = np.empty((self.polynomial_count, len(prepared)), dtype=np.uint32)
        derivatives = np.empty_like(values) if derivative else None
        derivative_pointer = (
            _pointer(derivatives, ctypes.c_uint32)
            if derivatives is not None
            else ctypes.POINTER(ctypes.c_uint32)()
        )
        started = time.perf_counter()
        _check(
            self._library.fast_math_hip_polynomial_evaluate(
                self._handle,
                _pointer(prepared, ctypes.c_uint32),
                len(prepared),
                int(derivative),
                _pointer(values, ctypes.c_uint32),
                derivative_pointer,
            ),
            "modular polynomial evaluation",
        )
        return values, derivatives, time.perf_counter() - started

    def close(self) -> None:
        handle, self._handle = self._handle, ctypes.c_void_p()
        if handle.value:
            _check(
                self._library.fast_math_hip_polynomial_destroy(handle),
                "modular polynomial plan destruction",
            )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class ModularDeterminantHipPlan:
    """Persistent HIP workspace for exact small dense determinants."""

    backend = "hip"

    def __init__(self, order: int, prime: int) -> None:
        self.prime = prime_u32(prime)
        if not isinstance(order, (int, np.integer)) or not 1 <= int(order) <= 32:
            raise ValueError("HIP determinant order must be between one and 32")
        self.order = int(order)
        library = _library()
        if not hasattr(library, "fast_math_hip_determinant_create"):
            raise HipUnavailable("HIP modular determinant support is not built")
        handle = ctypes.c_void_p()
        _check(
            library.fast_math_hip_determinant_create(
                self.order, self.prime, ctypes.byref(handle)
            ),
            "modular determinant plan creation",
        )
        if not handle.value:
            raise RuntimeError("HIP modular determinant plan returned a null handle")
        self._library = library
        self._handle = handle

    def determinants(
        self,
        matrices: ArrayLike,
    ) -> tuple[NDArray[np.uint32], float]:
        if not self._handle.value:
            raise RuntimeError("HIP modular determinant plan is closed")
        prepared = field_array_u32(
            matrices, self.prime, dimensions=3, own=False
        )
        if prepared.shape[1:] != (self.order, self.order):
            raise ValueError("matrix shape does not match the determinant plan")
        determinants = np.empty(len(prepared), dtype=np.uint32)
        started = time.perf_counter()
        _check(
            self._library.fast_math_hip_determinants(
                self._handle,
                _pointer(prepared, ctypes.c_uint32),
                len(prepared),
                _pointer(determinants, ctypes.c_uint32),
            ),
            "modular determinants",
        )
        return determinants, time.perf_counter() - started

    def close(self) -> None:
        handle, self._handle = self._handle, ctypes.c_void_p()
        if handle.value:
            _check(
                self._library.fast_math_hip_determinant_destroy(handle),
                "modular determinant plan destruction",
            )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _subset_permutations(values: ArrayLike) -> NDArray[np.uint32]:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[0] == 0 or not 1 <= raw.shape[1] <= 64:
        raise ValueError("permutations must have shape (count, degree), degree 1..64")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("permutations must contain integers")
    if np.any(raw < 0) or np.any(raw >= raw.shape[1]):
        raise ValueError("permutation image is out of range")
    result = np.array(raw, dtype=np.uint32, order="C", copy=True)
    if np.any(np.sort(result, axis=1) != np.arange(result.shape[1])):
        raise ValueError("every action row must be a permutation")
    result.flags.writeable = False
    return result


def _subset_masks(values: ArrayLike, degree: int) -> NDArray[np.uint64]:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("masks must be one-dimensional")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("masks must contain integers")
    if np.issubdtype(raw.dtype, np.signedinteger) and np.any(raw < 0):
        raise ValueError("masks must be nonnegative")
    result = np.ascontiguousarray(raw, dtype=np.uint64)
    if degree < 64 and np.any(result >> np.uint64(degree)):
        raise ValueError("mask contains an out-of-range bit")
    return result


class ModularLinearSystemHipPlan:
    """Persistent exact HIP operators for fixed-matrix solve batches."""

    backend = "hip"

    def __init__(
        self,
        solution_operator: ArrayLike,
        pivot_columns: ArrayLike,
        left_nullspace: ArrayLike,
        *,
        prime: int,
    ) -> None:
        self.prime = prime_u32(prime)
        operator = field_array_u32(
            solution_operator, self.prime, dimensions=2, own=True
        )
        pivots_raw = np.asarray(pivot_columns)
        if pivots_raw.ndim != 1 or not np.issubdtype(
            pivots_raw.dtype, np.integer
        ):
            raise ValueError("pivot columns must be one-dimensional integers")
        if np.issubdtype(pivots_raw.dtype, np.signedinteger) and np.any(
            pivots_raw < 0
        ):
            raise ValueError("pivot columns must be nonnegative")
        pivots = np.ascontiguousarray(pivots_raw, dtype=np.uint32)
        obstructions = field_array_u32(
            left_nullspace, self.prime, dimensions=2, own=True
        )
        if operator.shape[0] == 0 or operator.shape[1] == 0:
            raise ValueError("solution operator must have nonzero shape")
        if obstructions.shape[1] != operator.shape[1]:
            raise ValueError("left nullspace and solution operator row mismatch")
        if (
            len(pivots) + len(obstructions) != operator.shape[1]
            or (len(pivots) and np.any(pivots >= operator.shape[0]))
            or (len(pivots) > 1 and np.any(pivots[1:] <= pivots[:-1]))
        ):
            raise ValueError("pivot columns do not describe the solution operator")
        library = _library()
        if not hasattr(library, "fast_math_hip_linear_system_create"):
            raise HipUnavailable("HIP modular linear-system support is not built")
        handle = ctypes.c_void_p()
        left_pointer = (
            _pointer(obstructions, ctypes.c_uint32)
            if obstructions.size
            else ctypes.POINTER(ctypes.c_uint32)()
        )
        _check(
            library.fast_math_hip_linear_system_create(
                _pointer(operator, ctypes.c_uint32),
                _pointer(pivots, ctypes.c_uint32),
                left_pointer,
                operator.shape[1],
                operator.shape[0],
                len(pivots),
                len(obstructions),
                self.prime,
                ctypes.byref(handle),
            ),
            "modular linear-system plan creation",
        )
        if not handle.value:
            raise RuntimeError(
                "HIP modular linear-system plan returned a null handle"
            )
        self._library = library
        self._handle = handle
        self._solution_operator = operator
        self._pivot_columns = pivots
        self._left_nullspace = obstructions

    @property
    def row_count(self) -> int:
        return self._solution_operator.shape[1]

    @property
    def column_count(self) -> int:
        return self._solution_operator.shape[0]

    def solve(
        self,
        right_hand_sides: ArrayLike,
    ) -> tuple[NDArray[np.uint32], NDArray[np.int64], float]:
        if not self._handle.value:
            raise RuntimeError("HIP modular linear-system plan is closed")
        prepared = field_array_u32(
            right_hand_sides, self.prime, dimensions=2, own=False
        )
        if prepared.shape[1] != self.row_count:
            raise ValueError("right-hand side width does not match the system")
        solutions = np.empty(
            (len(prepared), self.column_count), dtype=np.uint32
        )
        inconsistency_rows = np.empty(len(prepared), dtype=np.int64)
        started = time.perf_counter()
        _check(
            self._library.fast_math_hip_linear_system_solve(
                self._handle,
                _pointer(prepared, ctypes.c_uint32),
                len(prepared),
                _pointer(solutions, ctypes.c_uint32),
                _pointer(inconsistency_rows, ctypes.c_int64),
            ),
            "modular linear-system solve",
        )
        return solutions, inconsistency_rows, time.perf_counter() - started

    def close(self) -> None:
        handle, self._handle = self._handle, ctypes.c_void_p()
        if handle.value:
            _check(
                self._library.fast_math_hip_linear_system_destroy(handle),
                "modular linear-system plan destruction",
            )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class SubsetActionHipPlan:
    """Persistent exact HIP plan for packed subset images."""

    backend = "hip"

    def __init__(self, permutations: ArrayLike) -> None:
        values = _subset_permutations(permutations)
        library = _library()
        if not hasattr(library, "fast_math_hip_subset_action_create"):
            raise HipUnavailable("HIP packed subset actions are not built")
        handle = ctypes.c_void_p()
        _check(
            library.fast_math_hip_subset_action_create(
                _pointer(values.reshape(-1), ctypes.c_uint32),
                len(values),
                values.shape[1],
                ctypes.byref(handle),
            ),
            "subset action plan creation",
        )
        if not handle.value:
            raise RuntimeError("HIP subset action returned a null handle")
        self._library = library
        self._handle = handle
        self._permutations = values

    @property
    def degree(self) -> int:
        return self._permutations.shape[1]

    def canonicalize(
        self,
        masks: ArrayLike,
    ) -> tuple[NDArray[np.uint64], NDArray[np.bool_], float]:
        if not self._handle.value:
            raise RuntimeError("HIP subset action plan is closed")
        values = _subset_masks(masks, self.degree)
        canonical = np.empty(len(values), dtype=np.uint64)
        flags = np.empty(len(values), dtype=np.uint8)
        started = time.perf_counter()
        _check(
            self._library.fast_math_hip_subset_action_canonicalize(
                self._handle,
                _pointer(values, ctypes.c_uint64),
                len(values),
                _pointer(canonical, ctypes.c_uint64),
                _pointer(flags, ctypes.c_uint8),
            ),
            "subset action canonicalization",
        )
        return canonical, flags.view(np.bool_), time.perf_counter() - started

    def is_canonical(
        self,
        masks: ArrayLike,
    ) -> tuple[NDArray[np.bool_], float]:
        if not self._handle.value:
            raise RuntimeError("HIP subset action plan is closed")
        values = _subset_masks(masks, self.degree)
        flags = np.empty(len(values), dtype=np.uint8)
        started = time.perf_counter()
        _check(
            self._library.fast_math_hip_subset_action_is_canonical(
                self._handle,
                _pointer(values, ctypes.c_uint64),
                len(values),
                _pointer(flags, ctypes.c_uint8),
            ),
            "subset action canonical test",
        )
        return flags.view(np.bool_), time.perf_counter() - started

    def close(self) -> None:
        handle, self._handle = self._handle, ctypes.c_void_p()
        if handle.value:
            _check(
                self._library.fast_math_hip_subset_action_destroy(handle),
                "subset action plan destruction",
            )

    def __enter__(self) -> "SubsetActionHipPlan":
        if not self._handle.value:
            raise RuntimeError("HIP subset action plan is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class SquareCoverHipPlan:
    """Persistent FP64 HIP plan for oriented-square point incidence."""

    backend = "hip"

    def __init__(self, points: ArrayLike) -> None:
        values = np.asarray(points, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != 2
            or len(values) == 0
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("points must have finite shape (point_count, 2)")
        values = np.ascontiguousarray(values)
        library = _library()
        if not hasattr(library, "fast_math_hip_square_cover_create"):
            raise HipUnavailable("HIP square-cover support is not built")
        handle = ctypes.c_void_p()
        _check(
            library.fast_math_hip_square_cover_create(
                _pointer(values.reshape(-1), ctypes.c_double),
                len(values),
                ctypes.byref(handle),
            ),
            "square-cover plan creation",
        )
        self._library = library
        self._handle = handle
        self._points = values

    @property
    def point_count(self) -> int:
        return len(self._points)

    def evaluate(
        self,
        poses: ArrayLike,
        *,
        half_extent: float = 0.5,
        uncertainty: float = 0.0,
    ) -> tuple[NDArray[np.uint64], NDArray[np.uint64], float]:
        values = np.asarray(poses, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != 4
            or len(values) == 0
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("poses must have finite shape (pose_count, 4)")
        if (
            not np.isfinite(half_extent)
            or half_extent <= 0
            or not np.isfinite(uncertainty)
            or uncertainty < 0
            or uncertainty >= half_extent
        ):
            raise ValueError("invalid half_extent or uncertainty")
        values = np.ascontiguousarray(values)
        outputs = np.empty(
            ((self.point_count + 63) // 64, len(values)), dtype=np.uint64
        )
        uncertain = np.empty_like(outputs)
        started = time.perf_counter()
        _check(
            self._library.fast_math_hip_square_cover_evaluate(
                self._handle,
                _pointer(values.reshape(-1), ctypes.c_double),
                len(values),
                half_extent,
                uncertainty,
                _pointer(outputs.reshape(-1), ctypes.c_uint64),
                _pointer(uncertain.reshape(-1), ctypes.c_uint64),
            ),
            "square-cover evaluation",
        )
        return outputs, uncertain, time.perf_counter() - started

    def weighted_scores(
        self,
        poses: ArrayLike,
        weights: ArrayLike,
        *,
        half_extent: float = 0.5,
        uncertainty: float = 0.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        values = np.asarray(poses, dtype=np.float64)
        weight_values = np.asarray(weights, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != 4
            or len(values) == 0
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("poses must have finite shape (pose_count, 4)")
        if (
            weight_values.shape != (self.point_count,)
            or not np.all(np.isfinite(weight_values))
            or np.any(weight_values < 0)
        ):
            raise ValueError("weights must be finite and nonnegative")
        if (
            not np.isfinite(half_extent)
            or half_extent <= 0
            or not np.isfinite(uncertainty)
            or uncertainty < 0
            or uncertainty >= half_extent
        ):
            raise ValueError("invalid half_extent or uncertainty")
        values = np.ascontiguousarray(values)
        weight_values = np.ascontiguousarray(weight_values)
        definite = np.empty(len(values), dtype=np.float64)
        possible = np.empty_like(definite)
        started = time.perf_counter()
        _check(
            self._library.fast_math_hip_square_weighted_evaluate(
                self._handle,
                _pointer(values.reshape(-1), ctypes.c_double),
                len(values),
                _pointer(weight_values, ctypes.c_double),
                half_extent,
                uncertainty,
                _pointer(definite, ctypes.c_double),
                _pointer(possible, ctypes.c_double),
            ),
            "square weighted evaluation",
        )
        return definite, possible, time.perf_counter() - started

    def __enter__(self) -> "SquareCoverHipPlan":
        if not self._handle.value:
            raise RuntimeError("square-cover plan is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        handle, self._handle = self._handle, ctypes.c_void_p()
        if handle and handle.value:
            _check(
                self._library.fast_math_hip_square_cover_destroy(handle),
                "square-cover plan destruction",
            )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class AffineHipPlan:
    """Persistent ROCm plan for ``base + steps @ basis`` workloads."""

    backend = "hip"

    def __init__(self, base: ArrayLike, basis: ArrayLike) -> None:
        prepared_base, prepared_basis = affine_inputs(base, basis)
        library = _library()
        started = time.perf_counter()
        handle = ctypes.c_void_p()
        _check(
            library.fast_math_hip_create(
                _pointer(prepared_base.view(np.float32), ctypes.c_float),
                prepared_base.size,
                _pointer(prepared_basis.view(np.float32), ctypes.c_float),
                prepared_basis.shape[0],
                ctypes.byref(handle),
            ),
            "plan creation",
        )
        self._library = library
        self._handle = handle
        self._base = prepared_base
        self._basis = prepared_basis
        self._setup_seconds = time.perf_counter() - started

    @property
    def point_count(self) -> int:
        return self._base.size

    @property
    def direction_count(self) -> int:
        return self._basis.shape[0]

    @property
    def setup_seconds(self) -> float:
        return self._setup_seconds

    def _steps(self, values: ArrayLike) -> NDArray[np.float32]:
        return affine_steps(values, direction_count=self.direction_count)

    def _evaluate_batch(self, steps: NDArray[np.float32]) -> NDArray[np.complex64]:
        output = np.empty((steps.shape[0], self.point_count), dtype=np.complex64)
        _check(
            self._library.fast_math_hip_evaluate(
                self._handle,
                _pointer(steps, ctypes.c_float),
                steps.shape[0],
                _pointer(output.view(np.float32), ctypes.c_float),
            ),
            "evaluation",
        )
        return output

    def evaluate(
        self,
        steps: ArrayLike,
        *,
        batch_size: int | None = None,
    ) -> NDArray[np.complex64]:
        prepared_steps = self._steps(steps)
        outputs = [
            self._evaluate_batch(batch)
            for batch in affine_batches(prepared_steps, batch_size)
        ]
        return outputs[0] if len(outputs) == 1 else np.concatenate(outputs, axis=0)

    def contour_metrics(
        self,
        steps: ArrayLike,
        *,
        edge_slice: slice = slice(None),
        batch_size: int | None = None,
    ) -> AffineContourMetrics:
        edge_start, edge_stop = edge_bounds(
            edge_slice, point_count=self.point_count
        )
        prepared_steps = self._steps(steps)
        started = time.perf_counter()
        windings = []
        maximum_phases = []
        edge_floors = []
        for batch in affine_batches(prepared_steps, batch_size):
            rows = batch.shape[0]
            batch_windings = np.empty(rows, dtype=np.int64)
            batch_phases = np.empty(rows, dtype=np.float32)
            batch_floors = np.empty(rows, dtype=np.float32)
            _check(
                self._library.fast_math_hip_metrics(
                    self._handle,
                    _pointer(batch, ctypes.c_float),
                    rows,
                    edge_start,
                    edge_stop,
                    _pointer(batch_windings, ctypes.c_int64),
                    _pointer(batch_phases, ctypes.c_float),
                    _pointer(batch_floors, ctypes.c_float),
                ),
                "contour metrics",
            )
            windings.append(batch_windings)
            maximum_phases.append(batch_phases)
            edge_floors.append(batch_floors)
        return AffineContourMetrics(
            windings=np.concatenate(windings),
            maximum_phases=np.concatenate(maximum_phases),
            edge_floors=np.concatenate(edge_floors),
            elapsed_seconds=time.perf_counter() - started,
            backend=self.backend,
        )

    def close(self) -> None:
        handle, self._handle = self._handle, ctypes.c_void_p()
        if handle and handle.value:
            _check(self._library.fast_math_hip_destroy(handle), "plan destruction")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def clear_cache() -> None:
        # HIP's allocator is released when each persistent plan is closed.
        return None
