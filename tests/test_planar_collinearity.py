from __future__ import annotations

import numpy as np
import pytest

from fast_math import planar_collinearity_scores
from fast_math._native import native_available
from fast_math._planar_native import planar_collinearity_edits_native


BACKENDS = ["reference", "native"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_base_score_and_point_degrees(backend: str) -> None:
    points = [(0, 0), (1, 1), (2, 2), (3, 3), (0, 1)]
    result = planar_collinearity_scores(points, backend=backend)
    assert result.base_score == 4
    assert result.point_degrees.tolist() == [3, 3, 3, 3, 0]
    assert int(result.point_degrees.sum()) == 3 * result.base_score
    assert result.edit_scores.shape == (0,)
    assert result.edit_deltas.shape == (0,)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ragged_add_delete_and_swap_edits(backend: str) -> None:
    points = [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (3, 1)]
    # identity; delete 0; add two; replace 1 and 2; delete all
    deletes = [0, 1, 2, 0, 1, 2, 3, 4, 5]
    delete_offsets = [0, 0, 1, 1, 3, 9]
    additions = [(3, 3), (4, 4), (5, 0), (6, 2)]
    add_offsets = [0, 0, 0, 2, 4, 4]
    result = planar_collinearity_scores(
        points,
        delete_indices=deletes,
        delete_offsets=delete_offsets,
        added_points=additions,
        add_offsets=add_offsets,
        backend=backend,
    )
    assert result.edit_scores[0] == result.base_score
    assert result.edit_scores[-1] == 0
    np.testing.assert_array_equal(
        result.edit_deltas,
        result.edit_scores.astype(np.int64) - result.base_score,
    )


def test_native_matches_independent_reference_on_random_edits() -> None:
    random = np.random.default_rng(20260823)
    points = np.asarray(
        [(index, int(random.integers(-20, 21))) for index in range(18)],
        dtype=np.int32,
    )
    deletes: list[int] = []
    delete_offsets = [0]
    additions: list[tuple[int, int]] = []
    add_offsets = [0]
    for edit in range(40):
        chosen = random.choice(len(points), size=edit % 4, replace=False)
        deletes.extend(map(int, chosen))
        delete_offsets.append(len(deletes))
        for added in range((edit + 1) % 4):
            additions.append((100 + 4 * edit + added, edit - added))
        add_offsets.append(len(additions))
    arguments = dict(
        delete_indices=deletes,
        delete_offsets=delete_offsets,
        added_points=additions,
        add_offsets=add_offsets,
    )
    reference = planar_collinearity_scores(
        points, backend="reference", **arguments
    )
    native = planar_collinearity_scores(
        points, backend="native", threads=4, **arguments
    )
    assert native.base_score == reference.base_score
    np.testing.assert_array_equal(native.point_degrees, reference.point_degrees)
    np.testing.assert_array_equal(native.edit_scores, reference.edit_scores)
    np.testing.assert_array_equal(native.edit_deltas, reference.edit_deltas)
    np.testing.assert_array_equal(
        native.cutoff_reached, reference.cutoff_reached
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_int32_extremes_use_overflow_safe_determinants(backend: str) -> None:
    low = -(1 << 31)
    high = (1 << 31) - 1
    points = [(low, low), (0, 0), (high, high), (low, high), (high, low)]
    result = planar_collinearity_scores(points, backend=backend)
    assert result.base_score == 1
    assert result.point_degrees.tolist() == [1, 1, 1, 0, 0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_base_supports_pure_additions(backend: str) -> None:
    result = planar_collinearity_scores(
        [],
        delete_offsets=[0, 0],
        added_points=[(0, 0), (1, 1), (2, 2), (0, 1)],
        add_offsets=[0, 4],
        backend=backend,
    )
    assert result.base_score == 0
    assert result.point_degrees.shape == (0,)
    assert result.edit_scores.tolist() == [1]
    assert result.edit_deltas.tolist() == [1]


@pytest.mark.parametrize("backend", BACKENDS)
def test_optional_cutoff_caps_only_flagged_edits(backend: str) -> None:
    points = [(index, 0) for index in range(8)]
    result = planar_collinearity_scores(
        points,
        delete_offsets=[0, 0, 0],
        added_points=[(0, 1)],
        add_offsets=[0, 0, 1],
        score_cutoff=5,
        backend=backend,
    )
    assert result.base_score == 56
    assert result.edit_scores.tolist() == [5, 5]
    assert result.cutoff_reached.tolist() == [True, True]
    assert result.score_cutoff == 5
    exact = planar_collinearity_scores(
        [(0, 0), (0, 1), (1, 0)],
        delete_offsets=[0, 0],
        add_offsets=[0, 0],
        score_cutoff=5,
        backend=backend,
    )
    assert exact.edit_scores.tolist() == [0]
    assert exact.cutoff_reached.tolist() == [False]


@pytest.mark.parametrize("backend", BACKENDS)
def test_deleted_point_may_be_readded(backend: str) -> None:
    points = [(0, 0), (1, 1), (2, 2)]
    result = planar_collinearity_scores(
        points,
        delete_indices=[1],
        delete_offsets=[0, 1],
        added_points=[(1, 1)],
        add_offsets=[0, 1],
        backend=backend,
    )
    assert result.edit_scores.tolist() == [1]
    assert result.edit_deltas.tolist() == [0]


def test_native_output_is_thread_stable() -> None:
    points = [(index, (index * index + 3) % 31) for index in range(24)]
    deletes = list(range(24)) * 16
    additions = [(1000 + edit, edit % 37) for edit in range(len(deletes))]
    offsets = list(range(len(deletes) + 1))
    arguments = dict(
        delete_indices=deletes,
        delete_offsets=offsets,
        added_points=additions,
        add_offsets=offsets,
        backend="native",
    )
    serial = planar_collinearity_scores(points, threads=1, **arguments)
    parallel = planar_collinearity_scores(points, threads=8, **arguments)
    np.testing.assert_array_equal(parallel.edit_scores, serial.edit_scores)
    np.testing.assert_array_equal(parallel.edit_deltas, serial.edit_deltas)
    assert serial.worker_count == 1
    assert parallel.worker_count == 8


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base_points": [(0, 0), (0, 0)]}, "unique"),
        ({"base_points": [(0,)]}, "shape"),
        ({"base_points": [(0.5, 1)]}, "integers"),
        ({"base_points": [(1 << 31, 0)]}, "int32"),
        (
            {
                "base_points": [(0, 0)],
                "delete_indices": [1],
                "delete_offsets": [0, 1],
                "add_offsets": [0, 0],
            },
            "out-of-range",
        ),
        (
            {
                "base_points": [(0, 0), (1, 1)],
                "delete_indices": [0, 0],
                "delete_offsets": [0, 2],
                "add_offsets": [0, 0],
            },
            "repeat",
        ),
        (
            {
                "base_points": [(0, 0)],
                "delete_offsets": [1],
            },
            "start at zero",
        ),
        (
            {
                "base_points": [(0, 0)],
                "delete_offsets": [0, 0],
                "add_offsets": [0],
            },
            "same edits",
        ),
        (
            {
                "base_points": [(0, 0)],
                "delete_offsets": [0, 0],
                "added_points": [(0, 0)],
                "add_offsets": [0, 1],
            },
            "duplicate",
        ),
        (
            {
                "base_points": [],
                "delete_offsets": [0, 0],
                "added_points": [(1, 1), (1, 1)],
                "add_offsets": [0, 2],
            },
            "repeat",
        ),
        ({"base_points": [(0, 0)], "score_cutoff": 0}, "score_cutoff"),
        ({"base_points": [(0, 0)], "score_cutoff": -1}, "score_cutoff"),
        ({"base_points": [(0, 0)], "score_cutoff": True}, "score_cutoff"),
        ({"base_points": [(0, 0)], "score_cutoff": 1 << 64}, "score_cutoff"),
        ({"base_points": [(0, 0)], "threads": -1}, "threads"),
        ({"base_points": [(0, 0)], "threads": 1025}, "threads"),
        ({"base_points": [(0, 0)], "threads": True}, "threads"),
        ({"base_points": [(0, 0)], "backend": "gpu"}, "backend"),
    ],
)
def test_rejects_invalid_public_inputs(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        planar_collinearity_scores(**kwargs)


def test_rejects_too_many_base_points_and_additions() -> None:
    with pytest.raises(ValueError, match="512"):
        planar_collinearity_scores([(index, 0) for index in range(513)])
    with pytest.raises(ValueError, match="64 additions"):
        planar_collinearity_scores(
            [],
            delete_offsets=[0, 0],
            added_points=[(index, 0) for index in range(65)],
            add_offsets=[0, 65],
        )


@pytest.mark.parametrize(
    ("base", "added", "threads", "message"),
    [
        ([(0, 0), (0, 0)], [], 0, "not unique"),
        ([(0, 0)], [(0, 0)], 0, "duplicate"),
        ([(0, 0)], [], 1025, "at most 1024"),
    ],
)
def test_native_abi_rejects_hostile_inputs(
    base, added, threads: int, message: str
) -> None:
    if not native_available():
        pytest.skip("native library is unavailable")
    add_offsets = np.asarray([0, len(added)], dtype=np.uint64)
    with pytest.raises(RuntimeError, match=message):
        planar_collinearity_edits_native(
            np.asarray(base, dtype=np.int32).reshape((-1, 2)),
            np.empty(0, dtype=np.uint32),
            np.asarray([0, 0], dtype=np.uint64),
            np.asarray(added, dtype=np.int32).reshape((-1, 2)),
            add_offsets,
            0,
            threads,
        )
