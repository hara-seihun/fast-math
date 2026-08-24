"""Periodic correlation profiles for packed binary cycles."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cyclic_native import cyclic_correlation_profiles_native
from ._native import NativeUnavailable, native_available

__all__ = [
    "CyclicCorrelationBackend",
    "CyclicCorrelationProfiles",
    "cyclic_correlation_profiles",
]

CyclicCorrelationBackend = Literal["auto", "native", "reference"]


@dataclass(frozen=True)
class CyclicCorrelationProfiles:
    """Exact mask intersections and signed correlations by periodic lag.

    Rows follow the input mask order and columns are lags
    ``0, ..., bit_width - 1``. A set bit represents ``-1`` and a clear bit
    represents ``+1`` in ``signed_correlations``.
    """

    intersection_counts: NDArray[np.uint8]
    signed_correlations: NDArray[np.int16]
    bit_width: int
    mask_count: int
    popcount_evaluations: int
    worker_count: int
    elapsed_seconds: float
    backend: str


def _validate_width(bit_width: int) -> int:
    if (
        not isinstance(bit_width, Integral)
        or isinstance(bit_width, bool)
        or not 1 <= int(bit_width) <= 64
    ):
        raise ValueError("bit_width must be an integer between one and 64")
    return int(bit_width)


def _validate_threads(threads: int) -> int:
    if (
        not isinstance(threads, Integral)
        or isinstance(threads, bool)
        or not 0 <= int(threads) <= 1024
    ):
        raise ValueError("threads must be an integer between zero and 1024")
    return int(threads)


def _prepare_masks(
    masks: ArrayLike,
    bit_width: int,
) -> NDArray[np.uint64]:
    raw = np.asarray(masks)
    if raw.ndim != 1:
        raise ValueError("masks must be one-dimensional")
    if raw.size == 0:
        return np.empty(0, dtype=np.uint64)
    if raw.dtype.kind not in {"i", "u", "O"}:
        raise ValueError("masks must contain integers")
    maximum = (1 << bit_width) - 1
    if raw.dtype.kind == "O":
        values = []
        for value in raw:
            if (
                not isinstance(value, Integral)
                or isinstance(value, bool)
                or not 0 <= int(value) <= maximum
            ):
                raise ValueError("mask has bits outside bit_width")
            values.append(int(value))
        return np.asarray(values, dtype=np.uint64)
    if raw.dtype.kind == "i" and np.any(raw < 0):
        raise ValueError("masks must be nonnegative")
    prepared = np.ascontiguousarray(raw, dtype=np.uint64)
    if bit_width < 64 and np.any(
        prepared > np.uint64(maximum)
    ):
        raise ValueError("mask has bits outside bit_width")
    return prepared


def _reference_profiles(
    masks: NDArray[np.uint64],
    bit_width: int,
) -> tuple[NDArray[np.uint8], NDArray[np.int16]]:
    intersections = np.empty((len(masks), bit_width), dtype=np.uint8)
    correlations = np.empty((len(masks), bit_width), dtype=np.int16)
    valid_bits = (1 << bit_width) - 1
    for index, raw in enumerate(masks):
        value = int(raw)
        weight = value.bit_count()
        for lag in range(bit_width):
            rotated = value if lag == 0 else (
                (value << lag) | (value >> (bit_width - lag))
            ) & valid_bits
            overlap = (value & rotated).bit_count()
            intersections[index, lag] = overlap
            correlations[index, lag] = (
                bit_width - 4 * (weight - overlap)
            )
    return intersections, correlations


def cyclic_correlation_profiles(
    masks: ArrayLike,
    bit_width: int,
    *,
    threads: int = 0,
    backend: CyclicCorrelationBackend = "auto",
) -> CyclicCorrelationProfiles:
    """Return exact periodic profiles for a batch of packed masks.

    ``intersection_counts[i, k]`` is
    ``popcount(mask[i] & rotate_left(mask[i], k))`` within ``bit_width``
    bits. ``signed_correlations`` is the corresponding periodic dot product
    after mapping set bits to ``-1`` and clear bits to ``+1``.
    """
    width = _validate_width(bit_width)
    worker_request = _validate_threads(threads)
    prepared = _prepare_masks(masks, width)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")

    if backend != "reference" and native_available():
        try:
            intersections, correlations, stats = (
                cyclic_correlation_profiles_native(
                    prepared, width, worker_request
                )
            )
        except (NativeUnavailable, OSError, AttributeError):
            if backend == "native":
                raise
        else:
            return CyclicCorrelationProfiles(
                intersection_counts=intersections,
                signed_correlations=correlations,
                bit_width=width,
                mask_count=len(prepared),
                popcount_evaluations=int(stats.popcount_evaluations),
                worker_count=int(stats.worker_count),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")

    started = perf_counter()
    intersections, correlations = _reference_profiles(prepared, width)
    return CyclicCorrelationProfiles(
        intersection_counts=intersections,
        signed_correlations=correlations,
        bit_width=width,
        mask_count=len(prepared),
        popcount_evaluations=len(prepared) * width,
        worker_count=1,
        elapsed_seconds=perf_counter() - started,
        backend="reference",
    )
