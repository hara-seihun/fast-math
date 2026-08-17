from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import time

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._packing_native import (
    square_cover_words_native,
    square_weighted_scores_native,
)


PackingBackend = Literal["auto", "native", "reference", "hip"]


@dataclass(frozen=True)
class SquareScoreBatch:
    definite_scores: NDArray[np.float64]
    possible_scores: NDArray[np.float64]
    point_count: int
    pose_count: int
    incidence_tests: int
    thread_count: int
    simd_lanes: int
    elapsed_seconds: float
    backend: str

    @property
    def minimum_definite_index(self) -> int:
        return int(np.argmin(self.definite_scores))

    @property
    def minimum_definite_score(self) -> float:
        return float(self.definite_scores[self.minimum_definite_index])


@dataclass(frozen=True)
class SquareCoverBatch:
    """Ternary point/pose incidence, stored word-major for zero-copy scans.

    Bit ``j % 64`` in row ``j // 64`` describes support point ``j``. A bit in
    ``inside_words`` is definitely inside the oriented square; a bit in
    ``uncertain_words`` is within ``uncertainty`` of its boundary. All other
    bits are definitely outside under the caller's numerical margin.
    """

    inside_words: NDArray[np.uint64]
    uncertain_words: NDArray[np.uint64]
    point_count: int
    pose_count: int
    incidence_tests: int
    thread_count: int
    simd_lanes: int
    elapsed_seconds: float
    backend: str

    @property
    def word_count(self) -> int:
        return self.inside_words.shape[0]

    def inside_matrix(self) -> NDArray[np.bool_]:
        shifts = np.arange(64, dtype=np.uint64)
        bits = ((self.inside_words[:, :, None] >> shifts) & 1).astype(bool)
        return bits.transpose(1, 0, 2).reshape(self.pose_count, -1)[
            :, : self.point_count
        ]

    def uncertain_matrix(self) -> NDArray[np.bool_]:
        shifts = np.arange(64, dtype=np.uint64)
        bits = ((self.uncertain_words[:, :, None] >> shifts) & 1).astype(bool)
        return bits.transpose(1, 0, 2).reshape(self.pose_count, -1)[
            :, : self.point_count
        ]


def _points(values: ArrayLike) -> NDArray[np.float64]:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("points must have shape (point_count, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    return np.ascontiguousarray(points)


def _poses(
    centers: ArrayLike,
    *,
    angles: ArrayLike | None,
    directions: ArrayLike | None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    center_values = np.asarray(centers, dtype=np.float64)
    if (
        center_values.ndim != 2
        or center_values.shape[1] != 2
        or len(center_values) == 0
        or not np.all(np.isfinite(center_values))
    ):
        raise ValueError("centers must have finite shape (pose_count, 2)")
    if (angles is None) == (directions is None):
        raise ValueError("provide exactly one of angles or directions")
    if angles is not None:
        angle_values = np.asarray(angles, dtype=np.float64)
        if (
            angle_values.ndim != 1
            or len(angle_values) != len(center_values)
            or not np.all(np.isfinite(angle_values))
        ):
            raise ValueError("angles must have one finite value per pose")
        direction_x = np.cos(angle_values)
        direction_y = np.sin(angle_values)
    else:
        direction_values = np.asarray(directions, dtype=np.float64)
        if (
            direction_values.shape != center_values.shape
            or not np.all(np.isfinite(direction_values))
        ):
            raise ValueError("directions must have finite shape (pose_count, 2)")
        norms = np.hypot(direction_values[:, 0], direction_values[:, 1])
        if np.any(np.abs(norms - 1.0) > 2e-12):
            raise ValueError("directions must be unit vectors")
        direction_x = direction_values[:, 0]
        direction_y = direction_values[:, 1]
    return tuple(
        np.ascontiguousarray(value, dtype=np.float64)
        for value in (
            center_values[:, 0],
            center_values[:, 1],
            direction_x,
            direction_y,
        )
    )


def _reference(
    points: NDArray[np.float64],
    center_x: NDArray[np.float64],
    center_y: NDArray[np.float64],
    direction_x: NDArray[np.float64],
    direction_y: NDArray[np.float64],
    *,
    half_extent: float,
    uncertainty: float,
) -> SquareCoverBatch:
    started = time.perf_counter()
    pose_count = len(center_x)
    word_count = (len(points) + 63) // 64
    inside_words = np.zeros((word_count, pose_count), dtype=np.uint64)
    uncertain_words = np.zeros_like(inside_words)
    chunk_size = max(1, min(16384, 2_000_000 // len(points)))
    for begin in range(0, pose_count, chunk_size):
        end = min(pose_count, begin + chunk_size)
        dx = points[:, 0, None] - center_x[None, begin:end]
        dy = points[:, 1, None] - center_y[None, begin:end]
        first = np.abs(
            dx * direction_x[None, begin:end]
            + dy * direction_y[None, begin:end]
        )
        second = np.abs(
            dy * direction_x[None, begin:end]
            - dx * direction_y[None, begin:end]
        )
        definite = (first <= half_extent - uncertainty) & (
            second <= half_extent - uncertainty
        )
        possible = (first <= half_extent + uncertainty) & (
            second <= half_extent + uncertainty
        )
        for word in range(word_count):
            point_begin = 64 * word
            point_end = min(len(points), point_begin + 64)
            bits = np.left_shift(
                np.uint64(1),
                np.arange(point_end - point_begin, dtype=np.uint64),
            )[:, None]
            inside_words[word, begin:end] = np.bitwise_or.reduce(
                definite[point_begin:point_end] * bits,
                axis=0,
            )
            uncertain_words[word, begin:end] = np.bitwise_or.reduce(
                (possible[point_begin:point_end]
                 & ~definite[point_begin:point_end])
                * bits,
                axis=0,
            )
    return SquareCoverBatch(
        inside_words=inside_words,
        uncertain_words=uncertain_words,
        point_count=len(points),
        pose_count=pose_count,
        incidence_tests=len(points) * pose_count,
        thread_count=1,
        simd_lanes=1,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def oriented_square_cover_words(
    points: ArrayLike,
    centers: ArrayLike,
    *,
    angles: ArrayLike | None = None,
    directions: ArrayLike | None = None,
    side_length: float = 1.0,
    uncertainty: float = 0.0,
    threads: int = 0,
    backend: PackingBackend = "auto",
) -> SquareCoverBatch:
    """Classify fixed points against a batch of freely oriented squares.

    The calculation uses square-local coordinates. ``uncertainty`` is an
    absolute geometric margin: points at least that far inside both slabs are
    definite, points no farther than that outside both slabs are possible, and
    their difference is reported as uncertain. This makes approximate scouts
    explicit rather than silently treating last-bit boundary decisions as
    theorem certificates.
    """

    if backend not in {"auto", "native", "reference", "hip"}:
        raise ValueError(f"unknown backend: {backend}")
    if threads < 0:
        raise ValueError("threads must be nonnegative")
    if (
        not np.isfinite(side_length)
        or side_length <= 0
        or not np.isfinite(uncertainty)
        or uncertainty < 0
        or uncertainty >= side_length / 2
    ):
        raise ValueError("invalid side_length or uncertainty")
    point_values = _points(points)
    center_x, center_y, direction_x, direction_y = _poses(
        centers,
        angles=angles,
        directions=directions,
    )
    if backend == "hip":
        from .hip import SquareCoverHipPlan

        poses = np.column_stack(
            (center_x, center_y, direction_x, direction_y)
        )
        plan = SquareCoverHipPlan(point_values)
        try:
            inside, uncertain_words, elapsed = plan.evaluate(
                poses,
                half_extent=side_length / 2,
                uncertainty=uncertainty,
            )
        finally:
            plan.close()
        return SquareCoverBatch(
            inside_words=inside,
            uncertain_words=uncertain_words,
            point_count=len(point_values),
            pose_count=len(center_x),
            incidence_tests=len(point_values) * len(center_x),
            thread_count=0,
            simd_lanes=1,
            elapsed_seconds=elapsed,
            backend="hip",
        )
    if backend in {"auto", "native"}:
        inside, uncertain_words, stats = square_cover_words_native(
            point_values,
            center_x,
            center_y,
            direction_x,
            direction_y,
            half_extent=side_length / 2,
            uncertainty=uncertainty,
            threads=threads,
        )
        return SquareCoverBatch(
            inside_words=inside,
            uncertain_words=uncertain_words,
            point_count=int(stats.point_count),
            pose_count=int(stats.pose_count),
            incidence_tests=int(stats.incidence_tests),
            thread_count=int(stats.thread_count),
            simd_lanes=int(stats.simd_lanes),
            elapsed_seconds=float(stats.elapsed_seconds),
            backend="native",
        )
    return _reference(
        point_values,
        center_x,
        center_y,
        direction_x,
        direction_y,
        half_extent=side_length / 2,
        uncertainty=uncertainty,
    )


def oriented_square_weighted_scores(
    points: ArrayLike,
    weights: ArrayLike,
    centers: ArrayLike,
    *,
    angles: ArrayLike | None = None,
    directions: ArrayLike | None = None,
    side_length: float = 1.0,
    uncertainty: float = 0.0,
    threads: int = 0,
    backend: PackingBackend = "auto",
) -> SquareScoreBatch:
    """Sum nonnegative point weights captured by each oriented square."""
    if backend not in {"auto", "native", "reference", "hip"}:
        raise ValueError(f"unknown backend: {backend}")
    if threads < 0:
        raise ValueError("threads must be nonnegative")
    if (
        not np.isfinite(side_length)
        or side_length <= 0
        or not np.isfinite(uncertainty)
        or uncertainty < 0
        or uncertainty >= side_length / 2
    ):
        raise ValueError("invalid side_length or uncertainty")
    point_values = _points(points)
    weight_values = np.asarray(weights, dtype=np.float64)
    if (
        weight_values.shape != (len(point_values),)
        or not np.all(np.isfinite(weight_values))
        or np.any(weight_values < 0)
    ):
        raise ValueError("weights must be one finite nonnegative value per point")
    weight_values = np.ascontiguousarray(weight_values)
    center_x, center_y, direction_x, direction_y = _poses(
        centers, angles=angles, directions=directions
    )
    if backend == "hip":
        from .hip import SquareCoverHipPlan

        poses = np.column_stack(
            (center_x, center_y, direction_x, direction_y)
        )
        plan = SquareCoverHipPlan(point_values)
        try:
            definite, possible, elapsed = plan.weighted_scores(
                poses,
                weight_values,
                half_extent=side_length / 2,
                uncertainty=uncertainty,
            )
        finally:
            plan.close()
        return SquareScoreBatch(
            definite, possible, len(point_values), len(center_x),
            len(point_values) * len(center_x), 0, 1, elapsed, "hip"
        )
    if backend in {"auto", "native"}:
        definite, possible, stats = square_weighted_scores_native(
            point_values,
            weight_values,
            center_x,
            center_y,
            direction_x,
            direction_y,
            half_extent=side_length / 2,
            uncertainty=uncertainty,
            threads=threads,
        )
        return SquareScoreBatch(
            definite_scores=definite,
            possible_scores=possible,
            point_count=int(stats.point_count),
            pose_count=int(stats.pose_count),
            incidence_tests=int(stats.incidence_tests),
            thread_count=int(stats.thread_count),
            simd_lanes=int(stats.simd_lanes),
            elapsed_seconds=float(stats.elapsed_seconds),
            backend="native",
        )
    cover = _reference(
        point_values,
        center_x,
        center_y,
        direction_x,
        direction_y,
        half_extent=side_length / 2,
        uncertainty=uncertainty,
    )
    definite = np.zeros(len(center_x), dtype=np.float64)
    possible = np.zeros(len(center_x), dtype=np.float64)
    for point, weight in enumerate(weight_values):
        bit = np.uint64(1) << np.uint64(point % 64)
        inside = (cover.inside_words[point // 64] & bit) != 0
        uncertain_bit = (cover.uncertain_words[point // 64] & bit) != 0
        definite += weight * inside
        possible += weight * (inside | uncertain_bit)
    return SquareScoreBatch(
        definite_scores=definite,
        possible_scores=possible,
        point_count=len(point_values),
        pose_count=len(center_x),
        incidence_tests=len(point_values) * len(center_x),
        thread_count=1,
        simd_lanes=1,
        elapsed_seconds=cover.elapsed_seconds,
        backend="reference",
    )
