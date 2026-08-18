"""Retained packed-subset actions under explicit permutation collections."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._actions_native import NativeSubsetActionPlan
from ._native import native_available


ActionBackend = Literal["auto", "reference", "native", "hip"]
U64MaskLUT = tuple[int, ...]


def u64_mask_lut(
    permutation: ArrayLike,
    *,
    degree: int | None = None,
) -> U64MaskLUT:
    """Precompute an exact lookup for a packed-mask permutation through degree 16."""
    array = np.asarray(permutation)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("permutation must be a one-dimensional integer array")
    if degree is None:
        degree = int(array.size)
    if not isinstance(degree, Integral) or not 0 < int(degree) <= 16:
        raise ValueError("degree must be an integer between one and 16")
    degree = int(degree)
    if array.size != degree:
        raise ValueError("permutation length must equal degree")
    prepared = np.ascontiguousarray(array, dtype=np.int64)
    if np.any(prepared < 0) or np.any(prepared >= degree):
        raise ValueError("permutation values are outside the degree")
    if not np.array_equal(np.sort(prepared), np.arange(degree)):
        raise ValueError("permutation must contain each point exactly once")
    image_bits = tuple(1 << int(image) for image in prepared)
    lookup = [0] * (1 << degree)
    for mask in range(1, len(lookup)):
        least = mask & -mask
        lookup[mask] = lookup[mask ^ least] | image_bits[least.bit_length() - 1]
    return tuple(lookup)


def compose_u64_mask_luts(
    first: U64MaskLUT,
    second: U64MaskLUT,
) -> U64MaskLUT:
    """Compose two packed-mask lookup tables as ``first(second(mask))``."""
    if len(first) == 0 or len(first) != len(second):
        raise ValueError("mask lookup tables must have the same nonzero size")
    limit = len(first)
    if limit & (limit - 1):
        raise ValueError("mask lookup table size must be a power of two")
    if any(
        value < 0 or value >= limit
        for table in (first, second)
        for value in table
    ):
        raise ValueError("mask lookup table contains an out-of-range value")
    return tuple(first[second[mask]] for mask in range(limit))


@dataclass(frozen=True)
class CanonicalMaskBatch:
    canonical_masks: NDArray[np.uint64]
    is_canonical: NDArray[np.bool_]
    elapsed_seconds: float
    backend: str


@dataclass(frozen=True)
class MaskOrbitPartition:
    canonical_masks: NDArray[np.uint64]
    class_ids: NDArray[np.uint64]
    representatives: NDArray[np.uint64]
    class_sizes: NDArray[np.uint64]
    elapsed_seconds: float
    backend: str


def _permutations(values: ArrayLike) -> NDArray[np.uint32]:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[0] == 0 or not 1 <= raw.shape[1] <= 64:
        raise ValueError("permutations must have shape (count, degree), degree 1..64")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("permutations must contain integers")
    if np.any(raw < 0) or np.any(raw >= raw.shape[1]):
        raise ValueError("permutation image is out of range")
    result = np.array(raw, dtype=np.uint32, order="C", copy=True)
    expected = np.arange(result.shape[1], dtype=np.uint32)
    if np.any(np.sort(result, axis=1) != expected):
        raise ValueError("every action row must be a permutation")
    result.flags.writeable = False
    return result


def _masks(values: ArrayLike, degree: int) -> NDArray[np.uint64]:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("masks must be one-dimensional")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("masks must contain integers")
    if np.issubdtype(raw.dtype, np.signedinteger) and np.any(raw < 0):
        raise ValueError("masks must be nonnegative")
    result = np.ascontiguousarray(raw, dtype=np.uint64)
    valid = (
        np.uint64(np.iinfo(np.uint64).max)
        if degree == 64
        else np.uint64((1 << degree) - 1)
    )
    if np.any(result & ~valid):
        raise ValueError("mask contains an out-of-range bit")
    return result


def _byte_luts(permutations: NDArray[np.uint32]) -> NDArray[np.uint64]:
    degree = permutations.shape[1]
    byte_count = (degree + 7) // 8
    tables = np.zeros((len(permutations), byte_count, 256), dtype=np.uint64)
    for row, permutation in enumerate(permutations):
        for byte in range(byte_count):
            table = tables[row, byte]
            for value in range(1, 256):
                low = value & -value
                point = byte * 8 + low.bit_length() - 1
                table[value] = table[value ^ low]
                if point < degree:
                    table[value] |= np.uint64(1) << np.uint64(permutation[point])
    return tables


def _canonicalize_reference(
    masks: NDArray[np.uint64],
    tables: NDArray[np.uint64],
) -> tuple[NDArray[np.uint64], NDArray[np.bool_]]:
    best = masks.copy()
    for action in tables:
        remaining = masks.copy()
        image = np.zeros(len(masks), dtype=np.uint64)
        for table in action:
            image |= table[(remaining & np.uint64(255)).astype(np.uint8)]
            remaining >>= np.uint64(8)
        np.minimum(best, image, out=best)
    return best, best == masks


class PermutationActionPlan:
    """Retain permutations and packed-mask workspaces across action batches.

    The original mask is an implicit identity image. All supplied permutation
    images are minimized with it. Pass a complete finite group action when
    canonical masks are intended as orbit representatives; a generator list
    alone tests and minimizes only its one-step images.
    """

    def __init__(self, permutations: ArrayLike) -> None:
        self._permutations = _permutations(permutations)
        self._reference_tables: NDArray[np.uint64] | None = None
        self._native: NativeSubsetActionPlan | None = None
        self._hip = None
        self._closed = False

    @property
    def degree(self) -> int:
        return self._permutations.shape[1]

    @property
    def permutation_count(self) -> int:
        return len(self._permutations)

    @property
    def permutations(self) -> NDArray[np.uint32]:
        return self._permutations

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("permutation action plan is closed")

    def _reference(self) -> NDArray[np.uint64]:
        if self._reference_tables is None:
            self._reference_tables = _byte_luts(self._permutations)
        return self._reference_tables

    def _native_plan(self) -> NativeSubsetActionPlan:
        if self._native is None:
            self._native = NativeSubsetActionPlan(self._permutations)
        return self._native

    def _hip_plan(self):
        if self._hip is None:
            from .hip import SubsetActionHipPlan

            self._hip = SubsetActionHipPlan(self._permutations)
        return self._hip

    def _automatic_backend(
        self,
        mask_count: int,
        *,
        canonical_masks: bool,
    ) -> str:
        if not canonical_masks:
            if self._hip is not None and mask_count >= 500_000:
                return "hip"
            return "native" if native_available() else "reference"
        work = (
            mask_count
            * self.permutation_count
            * ((self.degree + 7) // 8)
        )
        if work >= 300_000_000:
            from .hip import hip_subset_actions_available

            if hip_subset_actions_available():
                return "hip"
        return "native" if native_available() else "reference"

    def canonicalize(
        self,
        masks: ArrayLike,
        *,
        threads: int = 1,
        backend: ActionBackend = "auto",
    ) -> CanonicalMaskBatch:
        self._check_open()
        if backend not in {"auto", "reference", "native", "hip"}:
            raise ValueError(f"unknown action backend: {backend}")
        if threads < 1:
            raise ValueError("threads must be positive")
        prepared = _masks(masks, self.degree)
        selected = backend
        if selected == "auto":
            selected = self._automatic_backend(
                len(prepared), canonical_masks=True
            )
        started = perf_counter()
        if selected == "reference":
            canonical, flags = _canonicalize_reference(prepared, self._reference())
            elapsed = perf_counter() - started
        elif selected == "native":
            canonical, flags, stats = self._native_plan().canonicalize(
                prepared, threads=threads
            )
            assert canonical is not None
            elapsed = float(stats.elapsed_seconds)
        else:
            canonical, flags, elapsed = self._hip_plan().canonicalize(prepared)
        return CanonicalMaskBatch(
            canonical_masks=canonical,
            is_canonical=flags,
            elapsed_seconds=elapsed,
            backend=selected,
        )

    def is_canonical(
        self,
        masks: ArrayLike,
        *,
        threads: int = 1,
        backend: ActionBackend = "auto",
    ) -> NDArray[np.bool_]:
        self._check_open()
        prepared = _masks(masks, self.degree)
        if backend not in {"auto", "reference", "native", "hip"}:
            raise ValueError(f"unknown action backend: {backend}")
        selected = (
            self._automatic_backend(
                len(prepared), canonical_masks=False
            )
            if backend == "auto"
            else backend
        )
        if selected == "native":
            _, flags, _ = self._native_plan().canonicalize(
                prepared, threads=threads, canonical_masks=False
            )
            return flags
        if selected == "hip":
            flags, _ = self._hip_plan().is_canonical(prepared)
            return flags
        return self.canonicalize(
            prepared, threads=threads, backend="reference"
        ).is_canonical

    def partition(
        self,
        masks: ArrayLike,
        *,
        threads: int = 1,
        backend: ActionBackend = "auto",
    ) -> MaskOrbitPartition:
        result = self.canonicalize(masks, threads=threads, backend=backend)
        representatives, class_ids, sizes = np.unique(
            result.canonical_masks,
            return_inverse=True,
            return_counts=True,
        )
        return MaskOrbitPartition(
            canonical_masks=result.canonical_masks,
            class_ids=class_ids.astype(np.uint64, copy=False),
            representatives=representatives,
            class_sizes=sizes.astype(np.uint64, copy=False),
            elapsed_seconds=result.elapsed_seconds,
            backend=result.backend,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._native is not None:
            self._native.close()
        if self._hip is not None:
            self._hip.close()

    def __enter__(self) -> "PermutationActionPlan":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "ActionBackend",
    "CanonicalMaskBatch",
    "MaskOrbitPartition",
    "PermutationActionPlan",
    "U64MaskLUT",
    "compose_u64_mask_luts",
    "u64_mask_lut",
]
