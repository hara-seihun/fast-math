from __future__ import annotations

import numpy as np
import pytest

from fast_math import (
    oriented_square_cover_words,
    oriented_square_weighted_scores,
)


def test_square_cover_known_axis_aligned_and_rotated() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [0.5001, 0.0],
            [0.7, 0.0],
            [0.0, 0.7],
            [1.0, 1.0],
        ]
    )
    centers = np.zeros((2, 2))
    angles = np.array([0.0, np.pi / 4])
    result = oriented_square_cover_words(
        points,
        centers,
        angles=angles,
        backend="reference",
    )
    assert result.inside_words.shape == (1, 2)
    assert int(result.inside_words[0, 0]) == 0b000011
    assert int(result.inside_words[0, 1]) == 0b011101
    assert not np.any(result.uncertain_words)


def test_square_cover_uncertainty_is_explicit() -> None:
    points = np.array([[0.49, 0.0], [0.5, 0.0], [0.51, 0.0]])
    result = oriented_square_cover_words(
        points,
        [[0.0, 0.0]],
        directions=[[1.0, 0.0]],
        uncertainty=0.005,
        backend="reference",
    )
    assert int(result.inside_words[0, 0]) == 0b001
    assert int(result.uncertain_words[0, 0]) == 0b010


@pytest.mark.parametrize("point_count", [1, 63, 64, 65, 137])
def test_square_cover_native_reference_parity(point_count: int) -> None:
    rng = np.random.default_rng(1000 + point_count)
    points = rng.uniform(-2.0, 2.0, size=(point_count, 2))
    centers = rng.uniform(-1.0, 1.0, size=(503, 2))
    angles = rng.uniform(-np.pi, np.pi, size=len(centers))
    reference = oriented_square_cover_words(
        points,
        centers,
        angles=angles,
        uncertainty=1e-10,
        backend="reference",
    )
    native = oriented_square_cover_words(
        points,
        centers,
        angles=angles,
        uncertainty=1e-10,
        threads=3,
        backend="native",
    )
    np.testing.assert_array_equal(native.inside_words, reference.inside_words)
    np.testing.assert_array_equal(
        native.uncertain_words, reference.uncertain_words
    )
    assert native.incidence_tests == point_count * len(centers)


def test_square_weighted_native_reference_parity() -> None:
    rng = np.random.default_rng(811)
    points = rng.uniform(-2, 2, size=(137, 2))
    weights = rng.uniform(0, 1, size=len(points))
    centers = rng.uniform(-1, 1, size=(509, 2))
    angles = rng.uniform(-np.pi, np.pi, size=len(centers))
    reference = oriented_square_weighted_scores(
        points,
        weights,
        centers,
        angles=angles,
        uncertainty=1e-10,
        backend="reference",
    )
    native = oriented_square_weighted_scores(
        points,
        weights,
        centers,
        angles=angles,
        uncertainty=1e-10,
        threads=3,
        backend="native",
    )
    np.testing.assert_allclose(
        native.definite_scores, reference.definite_scores, atol=2e-14
    )
    np.testing.assert_allclose(
        native.possible_scores, reference.possible_scores, atol=2e-14
    )
    assert native.minimum_definite_index == int(
        np.argmin(reference.definite_scores)
    )


def test_square_cover_matrix_coordinate_order() -> None:
    points = np.array([[0.0, 0.0], [2.0, 2.0], [0.1, 0.1]])
    result = oriented_square_cover_words(
        points,
        [[0.0, 0.0], [2.0, 2.0]],
        angles=[0.0, 0.0],
        backend="reference",
    )
    np.testing.assert_array_equal(
        result.inside_matrix(),
        [[True, False, True], [False, True, False]],
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"points": [], "centers": [[0, 0]], "angles": [0]},
        {"points": [[0, 0]], "centers": [], "angles": []},
        {
            "points": [[0, 0]],
            "centers": [[0, 0]],
            "angles": [0],
            "directions": [[1, 0]],
        },
        {
            "points": [[0, 0]],
            "centers": [[0, 0]],
            "directions": [[2, 0]],
        },
        {
            "points": [[0, 0]],
            "centers": [[0, 0]],
            "angles": [0],
            "side_length": 0,
        },
        {
            "points": [[0, 0]],
            "centers": [[0, 0]],
            "angles": [0],
            "uncertainty": 0.5,
        },
    ],
)
def test_square_cover_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        oriented_square_cover_words(**kwargs)
