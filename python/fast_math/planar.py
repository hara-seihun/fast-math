"""Exact collinear-triple scores under small planar point-set edits."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._native import NativeUnavailable, native_available
from ._planar_native import planar_collinearity_edits_native

__all__ = [
    "PlanarCollinearityBackend",
    "PlanarCollinearityScores",
    "planar_collinearity_scores",
]

PlanarCollinearityBackend = Literal["auto", "native", "reference"]

_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class PlanarCollinearityScores:
    """Base conflicts and edit scores, with explicit early-cutoff flags."""

    base_score: int
    point_degrees: NDArray[np.uint64]
    edit_scores: NDArray[np.uint64]
    edit_deltas: NDArray[np.int64]
    cutoff_reached: NDArray[np.bool_]
    score_cutoff: int | None
    base_point_count: int
    edit_count: int
    base_determinant_evaluations: int
    edit_determinant_evaluations: int
    worker_count: int
    elapsed_seconds: float
    backend: str


def _prepare_points(points: ArrayLike, name: str) -> NDArray[np.int32]:
    raw = np.asarray(points)
    if raw.size == 0 and raw.ndim == 1:
        return np.empty((0, 2), dtype=np.int32)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError(f"{name} must have shape (count, 2)")
    if raw.dtype.kind not in {"i", "u", "O"}:
        raise ValueError(f"{name} must contain integers")
    values: list[tuple[int, int]] = []
    for row in raw:
        pair = []
        for value in row:
            if (
                not isinstance(value, Integral)
                or isinstance(value, bool)
                or not _INT32_MIN <= int(value) <= _INT32_MAX
            ):
                raise ValueError(f"{name} coordinates must fit in int32")
            pair.append(int(value))
        values.append((pair[0], pair[1]))
    return np.ascontiguousarray(values, dtype=np.int32).reshape((-1, 2))


def _prepare_nonnegative_vector(
    values: ArrayLike,
    name: str,
    maximum: int,
    dtype,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    prepared = []
    for value in raw:
        if (
            not isinstance(value, Integral)
            or isinstance(value, bool)
            or not 0 <= int(value) <= maximum
        ):
            raise ValueError(f"{name} contains an out-of-range value")
        prepared.append(int(value))
    return np.ascontiguousarray(prepared, dtype=dtype)


def _prepare_offsets(
    offsets: ArrayLike,
    name: str,
    item_count: int,
) -> NDArray[np.uint64]:
    prepared = _prepare_nonnegative_vector(
        offsets, name, (1 << 64) - 1, np.uint64
    )
    if len(prepared) == 0:
        raise ValueError(f"{name} must contain at least the initial zero")
    if int(prepared[0]) != 0 or int(prepared[-1]) != item_count:
        raise ValueError(
            f"{name} must start at zero and end at the item count"
        )
    if np.any(prepared[1:] < prepared[:-1]):
        raise ValueError(f"{name} must be nondecreasing")
    return prepared


def _validate_cutoff(score_cutoff: int | None) -> int:
    if score_cutoff is None:
        return 0
    if (
        not isinstance(score_cutoff, Integral)
        or isinstance(score_cutoff, bool)
        or not 1 <= int(score_cutoff) <= (1 << 64) - 1
    ):
        raise ValueError("score_cutoff must be None or a positive uint64 integer")
    return int(score_cutoff)


def _validate_threads(threads: int) -> int:
    if (
        not isinstance(threads, Integral)
        or isinstance(threads, bool)
        or not 0 <= int(threads) <= 1024
    ):
        raise ValueError("threads must be an integer between zero and 1024")
    return int(threads)


def _collinear(
    first: tuple[int, int],
    second: tuple[int, int],
    third: tuple[int, int],
) -> bool:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        == (second[1] - first[1]) * (third[0] - first[0])
    )


def _score_and_degrees(
    points: list[tuple[int, int]],
) -> tuple[int, NDArray[np.uint64]]:
    degrees = np.zeros(len(points), dtype=np.uint64)
    score = 0
    for first, second, third in combinations(range(len(points)), 3):
        if _collinear(points[first], points[second], points[third]):
            score += 1
            degrees[first] += 1
            degrees[second] += 1
            degrees[third] += 1
    return score, degrees


def _validate_edits(
    base_points: NDArray[np.int32],
    delete_indices: NDArray[np.uint32],
    delete_offsets: NDArray[np.uint64],
    added_points: NDArray[np.int32],
    add_offsets: NDArray[np.uint64],
) -> None:
    base = [tuple(map(int, row)) for row in base_points]
    if len(set(base)) != len(base):
        raise ValueError("base_points must be unique")
    if len(base) > 512:
        raise ValueError("at most 512 base points are supported")
    base_index = {point: index for index, point in enumerate(base)}
    for edit in range(len(delete_offsets) - 1):
        deleted = {
            int(index)
            for index in delete_indices[
                int(delete_offsets[edit]) : int(delete_offsets[edit + 1])
            ]
        }
        delete_count = int(delete_offsets[edit + 1] - delete_offsets[edit])
        if len(deleted) != delete_count:
            raise ValueError("delete_indices repeat within an edit")
        additions = [
            tuple(map(int, point))
            for point in added_points[
                int(add_offsets[edit]) : int(add_offsets[edit + 1])
            ]
        ]
        if len(additions) > 64:
            raise ValueError("at most 64 additions per edit are supported")
        if len(set(additions)) != len(additions):
            raise ValueError("added_points repeat within an edit")
        for point in additions:
            if point in base_index and base_index[point] not in deleted:
                raise ValueError("an edit produces duplicate points")


def _score_with_cutoff(
    points: list[tuple[int, int]],
    score_cutoff: int,
) -> tuple[int, bool, int]:
    score = 0
    evaluations = 0
    for first, second, third in combinations(range(len(points)), 3):
        evaluations += 1
        if not _collinear(points[first], points[second], points[third]):
            continue
        score += 1
        if score_cutoff and score >= score_cutoff:
            return score_cutoff, True, evaluations
    return score, False, evaluations


def _reference_scores(
    base_points: NDArray[np.int32],
    delete_indices: NDArray[np.uint32],
    delete_offsets: NDArray[np.uint64],
    added_points: NDArray[np.int32],
    add_offsets: NDArray[np.uint64],
    score_cutoff: int,
) -> tuple[
    int,
    NDArray[np.uint64],
    NDArray[np.uint64],
    NDArray[np.int64],
    NDArray[np.bool_],
    int,
]:
    base = [tuple(map(int, row)) for row in base_points]
    base_score, degrees = _score_and_degrees(base)
    scores = np.empty(len(delete_offsets) - 1, dtype=np.uint64)
    deltas = np.empty(len(delete_offsets) - 1, dtype=np.int64)
    cutoff_reached = np.empty(len(delete_offsets) - 1, dtype=np.bool_)
    evaluations = 0
    for edit in range(len(scores)):
        deleted = {
            int(index)
            for index in delete_indices[
                int(delete_offsets[edit]) : int(delete_offsets[edit + 1])
            ]
        }
        final_points = [
            point for index, point in enumerate(base) if index not in deleted
        ]
        final_points.extend(
            tuple(map(int, point))
            for point in added_points[
                int(add_offsets[edit]) : int(add_offsets[edit + 1])
            ]
        )
        score, reached, used = _score_with_cutoff(final_points, score_cutoff)
        scores[edit] = score
        deltas[edit] = score - base_score
        cutoff_reached[edit] = reached
        evaluations += used
    return base_score, degrees, scores, deltas, cutoff_reached, evaluations


def planar_collinearity_scores(
    base_points: ArrayLike,
    *,
    delete_indices: ArrayLike = (),
    delete_offsets: ArrayLike = (0,),
    added_points: ArrayLike = (),
    add_offsets: ArrayLike = (0,),
    score_cutoff: int | None = None,
    threads: int = 0,
    backend: PlanarCollinearityBackend = "auto",
) -> PlanarCollinearityScores:
    """Score collinear triples before and after ragged point edits.

    Edit ``i`` removes ``delete_indices[delete_offsets[i]:delete_offsets[i+1]]``
    and appends ``added_points[add_offsets[i]:add_offsets[i+1]]``. Both offset
    arrays must therefore describe the same number of edits. When
    ``score_cutoff`` is set, an edit stops at that many conflicts and its
    ``cutoff_reached`` flag is true; scores without that flag remain exact.
    """
    points = _prepare_points(base_points, "base_points")
    additions = _prepare_points(added_points, "added_points")
    deletes = _prepare_nonnegative_vector(
        delete_indices,
        "delete_indices",
        len(points) - 1,
        np.uint32,
    )
    delete_starts = _prepare_offsets(
        delete_offsets, "delete_offsets", len(deletes)
    )
    add_starts = _prepare_offsets(add_offsets, "add_offsets", len(additions))
    if len(delete_starts) != len(add_starts):
        raise ValueError(
            "delete_offsets and add_offsets must describe the same edits"
        )
    native_cutoff = _validate_cutoff(score_cutoff)
    worker_request = _validate_threads(threads)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    _validate_edits(points, deletes, delete_starts, additions, add_starts)

    if backend != "reference" and native_available():
        try:
            base_score, degrees, scores, deltas, reached, stats = (
                planar_collinearity_edits_native(
                    points,
                    deletes,
                    delete_starts,
                    additions,
                    add_starts,
                    native_cutoff,
                    worker_request,
                )
            )
        except (NativeUnavailable, OSError, AttributeError):
            if backend == "native":
                raise
        else:
            return PlanarCollinearityScores(
                base_score=base_score,
                point_degrees=degrees,
                edit_scores=scores,
                edit_deltas=deltas,
                cutoff_reached=reached,
                score_cutoff=score_cutoff,
                base_point_count=len(points),
                edit_count=len(scores),
                base_determinant_evaluations=int(
                    stats.base_determinant_evaluations
                ),
                edit_determinant_evaluations=int(
                    stats.edit_determinant_evaluations
                ),
                worker_count=int(stats.worker_count),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")

    started = perf_counter()
    (
        base_score,
        degrees,
        scores,
        deltas,
        reached,
        edit_evaluations,
    ) = _reference_scores(
        points,
        deletes,
        delete_starts,
        additions,
        add_starts,
        native_cutoff,
    )
    return PlanarCollinearityScores(
        base_score=base_score,
        point_degrees=degrees,
        edit_scores=scores,
        edit_deltas=deltas,
        cutoff_reached=reached,
        score_cutoff=score_cutoff,
        base_point_count=len(points),
        edit_count=len(scores),
        base_determinant_evaluations=(
            len(points) * (len(points) - 1) * (len(points) - 2) // 6
        ),
        edit_determinant_evaluations=edit_evaluations,
        worker_count=1,
        elapsed_seconds=perf_counter() - started,
        backend="reference",
    )
