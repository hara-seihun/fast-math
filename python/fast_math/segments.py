from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._native import (
    NativeUnavailable,
    segmented_complex_stats_native,
)


Backend = Literal["auto", "native", "reference"]


@dataclass(frozen=True)
class SegmentedComplexStats:
    sums: NDArray[np.complex128]
    l1: NDArray[np.float64]
    variation: NDArray[np.float64]
    sample_count: int
    segment_count: int
    elapsed_seconds: float
    backend: str


def _prepare(
    values: ArrayLike,
    offsets: ArrayLike,
) -> tuple[NDArray[np.complex128], NDArray[np.uint64]]:
    value_array = np.ascontiguousarray(values, dtype=np.complex128)
    offset_array = np.ascontiguousarray(offsets, dtype=np.uint64)
    if value_array.ndim != 1 or len(value_array) == 0:
        raise ValueError("values must be a nonempty vector")
    if offset_array.ndim != 1 or len(offset_array) < 2:
        raise ValueError("offsets must contain at least two entries")
    if (
        int(offset_array[0]) != 0
        or int(offset_array[-1]) != len(value_array)
        or np.any(offset_array[1:] <= offset_array[:-1])
    ):
        raise ValueError("offsets must partition the value vector")
    return value_array, offset_array


def _reference(
    values: NDArray[np.complex128],
    offsets: NDArray[np.uint64],
) -> SegmentedComplexStats:
    started = time.perf_counter()
    segment_count = len(offsets) - 1
    sums = np.empty(segment_count, dtype=np.complex128)
    l1 = np.empty(segment_count, dtype=np.float64)
    variation = np.empty(segment_count, dtype=np.float64)
    for segment in range(segment_count):
        begin = int(offsets[segment])
        end = int(offsets[segment + 1])
        local = values[begin:end]
        sums[segment] = np.sum(local, dtype=np.complex128)
        l1[segment] = np.sum(np.abs(local), dtype=np.float64)
        variation[segment] = np.sum(
            np.abs(np.diff(local)),
            dtype=np.float64,
        )
    return SegmentedComplexStats(
        sums=sums,
        l1=l1,
        variation=variation,
        sample_count=len(values),
        segment_count=segment_count,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def segmented_complex_stats(
    values: ArrayLike,
    offsets: ArrayLike,
    *,
    threads: int = 1,
    backend: Backend = "auto",
) -> SegmentedComplexStats:
    """Per-segment complex sum, L1 mass, and total variation over one value array."""
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    if threads < 0:
        raise ValueError("threads must be nonnegative")
    value_array, offset_array = _prepare(values, offsets)
    if backend in {"auto", "native"}:
        try:
            sums, l1, variation, stats = segmented_complex_stats_native(
                value_array,
                offset_array,
                threads=threads,
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return SegmentedComplexStats(
                sums=sums,
                l1=l1,
                variation=variation,
                sample_count=int(stats.sample_count),
                segment_count=int(stats.segment_count),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    return _reference(value_array, offset_array)
