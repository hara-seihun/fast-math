from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._inputs import prepare_inputs
from fast_math._native import NativeUnavailable, fused_two_level_native
from .api import PairCounts
from .reference import accumulate_reference


Backend = Literal["auto", "native", "reference"]
WEIGHT_DTYPE = np.dtype(
    [
        ("left", "<u8"),
        ("right", "<u8"),
        ("lower", "<f8"),
        ("upper", "<f8"),
    ]
)


@dataclass(frozen=True)
class TwoLevelRecord:
    left: int
    right: int
    fine_piece_count: int
    first_center: complex
    second_center: complex
    center_cost: float
    weight_variation_upper: float
    fine_phase_drift_upper: float
    two_level_upper: float


@dataclass(frozen=True)
class TwoLevelResult:
    records: tuple[TwoLevelRecord, ...]
    pairs: PairCounts
    fine_weight_block_count: int
    fine_piece_count: int
    constant_common_error: float
    constant_low_error: float
    center_cost: float
    weight_variation_upper: float
    fine_phase_drift_upper: float
    common_weighted_l1_upper: float
    low_weighted_l1_upper: float
    weighted_l1_upper: float
    two_level_upper: float
    elapsed_seconds: float
    backend: str


@dataclass(frozen=True)
class PreparedWeights:
    left: NDArray[np.uint64]
    right: NDArray[np.uint64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]


def multiplicative_blocks(
    first: int, last: int, ratio: float
) -> list[tuple[int, int]]:
    blocks = []
    left = first
    while left <= last:
        right = min(last, max(left, math.floor(left * ratio)))
        blocks.append((left, right))
        left = right + 1
    return blocks


def prepare_weights(
    weight_intervals: np.ndarray, output_limit: int
) -> PreparedWeights:
    intervals = np.asarray(weight_intervals)
    if intervals.ndim != 1 or intervals.dtype.names is None:
        raise ValueError("weight_intervals must be a structured vector")
    required = {"left", "right", "lower", "upper"}
    if not required.issubset(intervals.dtype.names):
        raise ValueError("weight_intervals have missing fields")
    left = np.ascontiguousarray(intervals["left"], dtype=np.uint64)
    right = np.ascontiguousarray(intervals["right"], dtype=np.uint64)
    lower = np.ascontiguousarray(intervals["lower"], dtype=np.float64)
    upper = np.ascontiguousarray(intervals["upper"], dtype=np.float64)
    if (
        len(left) == 0
        or int(left[0]) != 2
        or int(right[-1]) != output_limit
        or np.any(right < left)
        or np.any(lower <= 0.0)
        or np.any(upper < lower)
        or np.any(left[1:] != right[:-1] + 1)
    ):
        raise ValueError("weight intervals do not cover the output range")
    return PreparedWeights(left, right, lower, upper)


def _reference_two_level(
    inputs,
    weights: PreparedWeights,
    *,
    gamma_abs: float,
    sigma: float,
    q_primary: complex,
    q_dual: complex,
    outer_ratio: float,
) -> TwoLevelResult:
    started = time.perf_counter()
    accumulated = accumulate_reference(inputs)
    common = accumulated.common[2:]
    low = accumulated.low[2:]
    denominator = 1.0 + gamma_abs
    primary_exponent = q_primary - sigma
    dual_exponent = q_dual - sigma
    outer_blocks = multiplicative_blocks(2, inputs.output_limit, outer_ratio)
    fine_index = 0
    center_total = 0.0
    variation_total = 0.0
    drift_total = 0.0
    common_l1_total = 0.0
    low_l1_total = 0.0
    piece_count = 0
    records = []

    for outer_left, outer_right in outer_blocks:
        first_center = 0.0j
        second_center = 0.0j
        outer_variation = 0.0
        outer_drift = 0.0
        local_piece_count = 0
        while fine_index < len(weights.left) and weights.right[fine_index] < outer_left:
            fine_index += 1
        scan_index = fine_index
        while scan_index < len(weights.left) and weights.left[scan_index] <= outer_right:
            piece_left = max(outer_left, int(weights.left[scan_index]))
            piece_right = min(outer_right, int(weights.right[scan_index]))
            start = piece_left - 2
            stop = piece_right - 1
            common_piece = common[start:stop]
            low_piece = low[start:stop]
            lower = float(weights.lower[scan_index])
            upper = float(weights.upper[scan_index])
            midpoint = 0.5 * (lower + upper)
            radius = 0.5 * (upper - lower)
            common_sum = midpoint * np.sum(common_piece, dtype=np.complex128)
            low_sum = midpoint * np.sum(low_piece, dtype=np.complex128)
            common_l1 = float(np.sum(np.abs(common_piece), dtype=np.float64))
            low_l1 = float(np.sum(np.abs(low_piece), dtype=np.float64))

            log_left = math.log(piece_left)
            primary_anchor = np.exp(-primary_exponent * log_left)
            dual_anchor = np.exp(-dual_exponent * log_left)
            first_center += (
                primary_anchor * common_sum
                - gamma_abs * dual_anchor * low_sum
            )
            second_center += (
                dual_anchor * low_sum
                - gamma_abs * primary_anchor * common_sum
            )
            outer_variation += radius * (common_l1 + low_l1)
            log_ratio = math.log(piece_right / piece_left)
            primary_chord = abs(
                np.exp(-primary_exponent * log_ratio) - 1.0
            )
            dual_chord = abs(
                np.exp(-dual_exponent * log_ratio) - 1.0
            )
            outer_drift += (
                primary_chord * upper * common_l1
                + dual_chord * upper * low_l1
            )
            common_l1_total += upper * common_l1
            low_l1_total += upper * low_l1
            local_piece_count += 1
            if weights.right[scan_index] > outer_right:
                break
            scan_index += 1
        fine_index = scan_index
        center_cost = (
            abs(first_center) + abs(second_center)
        ) / denominator
        center_total += center_cost
        variation_total += outer_variation
        drift_total += outer_drift
        piece_count += local_piece_count
        records.append(
            TwoLevelRecord(
                left=outer_left,
                right=outer_right,
                fine_piece_count=local_piece_count,
                first_center=first_center,
                second_center=second_center,
                center_cost=center_cost,
                weight_variation_upper=outer_variation,
                fine_phase_drift_upper=outer_drift,
                two_level_upper=(
                    center_cost + outer_variation + outer_drift
                ),
            )
        )

    return TwoLevelResult(
        records=tuple(records),
        pairs=PairCounts(
            accumulated.primary_pairs,
            accumulated.transformed_pairs,
            accumulated.low_pairs,
        ),
        fine_weight_block_count=len(weights.left),
        fine_piece_count=piece_count,
        constant_common_error=abs(accumulated.common[1] - 1.0),
        constant_low_error=abs(accumulated.low[1] - gamma_abs),
        center_cost=center_total,
        weight_variation_upper=variation_total,
        fine_phase_drift_upper=drift_total,
        common_weighted_l1_upper=common_l1_total,
        low_weighted_l1_upper=low_l1_total,
        weighted_l1_upper=common_l1_total + low_l1_total,
        two_level_upper=center_total + variation_total + drift_total,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def fused_two_level(
    inverse: ArrayLike,
    primary: ArrayLike,
    transformed: ArrayLike,
    low: ArrayLike,
    weight_intervals: np.ndarray,
    *,
    transformed_first: int,
    output_limit: int,
    gamma_abs: float,
    sigma: float,
    q_primary: complex,
    q_dual: complex,
    outer_ratio: float,
    tile_size: int = 1 << 13,
    threads: int = 1,
    backend: Backend = "auto",
) -> TwoLevelResult:
    """Stream truncated Dirichlet products into fine sums, L1 envelopes, and two-level records."""
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    if not 0.0 <= gamma_abs < 1.0:
        raise ValueError("gamma_abs must lie in [0, 1)")
    if not math.isfinite(outer_ratio) or outer_ratio <= 1.0:
        raise ValueError("outer_ratio must be finite and above one")
    inputs = prepare_inputs(
        inverse,
        primary,
        transformed,
        low,
        transformed_first=transformed_first,
        output_limit=output_limit,
        tile_size=tile_size,
        threads=threads,
    )
    weights = prepare_weights(weight_intervals, output_limit)
    outer_blocks = multiplicative_blocks(2, output_limit, outer_ratio)

    if backend in {"auto", "native"}:
        try:
            raw_records, stats = fused_two_level_native(
                inputs,
                weight_left=weights.left,
                weight_right=weights.right,
                weight_lower=weights.lower,
                weight_upper=weights.upper,
                gamma_abs=gamma_abs,
                sigma=sigma,
                q_primary=q_primary,
                q_dual=q_dual,
                outer_ratio=outer_ratio,
                record_count=len(outer_blocks),
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            records = tuple(
                TwoLevelRecord(
                    left=int(record.left),
                    right=int(record.right),
                    fine_piece_count=int(record.fine_piece_count),
                    first_center=complex(
                        record.first_real, record.first_imag
                    ),
                    second_center=complex(
                        record.second_real, record.second_imag
                    ),
                    center_cost=float(record.center_cost),
                    weight_variation_upper=float(
                        record.weight_variation_upper
                    ),
                    fine_phase_drift_upper=float(
                        record.fine_phase_drift_upper
                    ),
                    two_level_upper=float(record.two_level_upper),
                )
                for record in raw_records
            )
            return TwoLevelResult(
                records=records,
                pairs=PairCounts(
                    int(stats.primary_pairs),
                    int(stats.transformed_pairs),
                    int(stats.low_pairs),
                ),
                fine_weight_block_count=int(
                    stats.fine_weight_block_count
                ),
                fine_piece_count=int(stats.fine_piece_count),
                constant_common_error=float(
                    stats.constant_common_error
                ),
                constant_low_error=float(stats.constant_low_error),
                center_cost=float(stats.center_cost),
                weight_variation_upper=float(
                    stats.weight_variation_upper
                ),
                fine_phase_drift_upper=float(
                    stats.fine_phase_drift_upper
                ),
                common_weighted_l1_upper=float(
                    stats.common_weighted_l1_upper
                ),
                low_weighted_l1_upper=float(
                    stats.low_weighted_l1_upper
                ),
                weighted_l1_upper=float(stats.weighted_l1_upper),
                two_level_upper=float(stats.two_level_upper),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )

    return _reference_two_level(
        inputs,
        weights,
        gamma_abs=gamma_abs,
        sigma=sigma,
        q_primary=q_primary,
        q_dual=q_dual,
        outer_ratio=outer_ratio,
    )
