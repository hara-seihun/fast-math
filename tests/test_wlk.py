from __future__ import annotations

from collections import Counter
from itertools import product

import numpy as np
import pytest

from fast_math import (
    graph_wl3_refinement,
    graph_wl4_refinement,
    graph_wlk_refinement,
)
from fast_math._ci_native import graph_wlk_refine_native
from fast_math._native import native_available


def graph_matrix(
    vertex_count: int,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    adjacency = np.zeros((vertex_count, vertex_count), dtype=np.uint8)
    for left, right in edges:
        adjacency[left, right] = 1
        adjacency[right, left] = 1
    return adjacency


def pack_graph(adjacency: np.ndarray) -> np.ndarray:
    vertex_count = len(adjacency)
    words = np.zeros(
        (vertex_count, (vertex_count + 63) // 64), dtype=np.uint64
    )
    for right in range(vertex_count):
        words[:, right // 64] |= (
            adjacency[:, right].astype(np.uint64)
            << np.uint64(right % 64)
        )
    return words


def naive_wlk(adjacency: np.ndarray, dimension: int) -> np.ndarray:
    vertices = tuple(product(range(len(adjacency)), repeat=dimension))
    initial = []
    for vertex_tuple in vertices:
        initial.append(
            tuple(
                (
                    0
                    if vertex_tuple[left] == vertex_tuple[right]
                    else 1
                    if adjacency[vertex_tuple[left], vertex_tuple[right]]
                    else 2
                )
                for left in range(dimension)
                for right in range(dimension)
                if left != right
            )
        )
    canonical = {value: color for color, value in enumerate(sorted(set(initial)))}
    colors = [canonical[value] for value in initial]
    tuple_index = {value: index for index, value in enumerate(vertices)}
    while True:
        signatures = []
        for index, vertex_tuple in enumerate(vertices):
            parts = [colors[index]]
            for coordinate in range(dimension):
                replacement_colors = []
                for replacement in range(len(adjacency)):
                    changed = list(vertex_tuple)
                    changed[coordinate] = replacement
                    replacement_colors.append(
                        colors[tuple_index[tuple(changed)]]
                    )
                counts = Counter(replacement_colors)
                parts.append(len(counts))
                for color, count in sorted(counts.items()):
                    parts.extend((color, count))
            signatures.append(tuple(parts))
        canonical = {
            value: color
            for color, value in enumerate(sorted(set(signatures)))
        }
        refined = [canonical[value] for value in signatures]
        if refined == colors:
            return np.asarray(refined, dtype=np.uint32).reshape(
                (len(adjacency),) * dimension
            )
        colors = refined


@pytest.mark.parametrize("dimension", [3, 4])
def test_native_and_reference_match_independent_definition(
    dimension: int,
) -> None:
    adjacency = graph_matrix(
        5,
        [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (2, 4)],
    )
    expected = naive_wlk(adjacency, dimension)
    reference = graph_wlk_refinement(
        adjacency, dimension, backend="reference"
    )
    np.testing.assert_array_equal(reference.colors, expected)
    np.testing.assert_array_equal(
        reference.color_sizes,
        np.bincount(expected.ravel()).astype(np.uint64),
    )
    if native_available():
        native = graph_wlk_refinement(
            adjacency, dimension, backend="native"
        )
        np.testing.assert_array_equal(native.colors, expected)
        np.testing.assert_array_equal(
            native.color_sizes, reference.color_sizes
        )
        assert native.iterations == reference.iterations


@pytest.mark.parametrize("dimension", [3, 4])
@pytest.mark.parametrize("kind", ["empty", "complete"])
def test_extreme_graphs_stabilize_to_equality_types(
    dimension: int, kind: str
) -> None:
    vertex_count = 5
    edges = (
        []
        if kind == "empty"
        else [
            (left, right)
            for left in range(vertex_count)
            for right in range(left + 1, vertex_count)
        ]
    )
    result = graph_wlk_refinement(
        graph_matrix(vertex_count, edges),
        dimension,
        backend="native" if native_available() else "reference",
    )
    assert result.color_count == (5 if dimension == 3 else 15)
    assert int(result.color_sizes.sum()) == vertex_count**dimension
    assert result.iterations == 1


@pytest.mark.parametrize("dimension", [3, 4])
def test_colors_and_histogram_are_canonical_under_vertex_relabeling(
    dimension: int,
) -> None:
    adjacency = graph_matrix(
        6,
        [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5)],
    )
    permutation = np.asarray([4, 2, 5, 1, 0, 3], dtype=np.uint32)
    relabeled = adjacency[np.ix_(permutation, permutation)]
    backend = "native" if native_available() else "reference"
    original = graph_wlk_refinement(adjacency, dimension, backend=backend)
    changed = graph_wlk_refinement(relabeled, dimension, backend=backend)
    np.testing.assert_array_equal(changed.color_sizes, original.color_sizes)
    expected = original.colors
    for coordinate in range(dimension):
        expected = np.take(expected, permutation, axis=coordinate)
    np.testing.assert_array_equal(changed.colors, expected)


def test_tuple_index_contract_is_c_order_lexicographic() -> None:
    adjacency = graph_matrix(4, [(0, 1), (1, 2), (2, 3)])
    result = graph_wl3_refinement(
        adjacency,
        backend="native" if native_available() else "reference",
    )
    assert result.colors.shape == (4, 4, 4)
    for index in range(result.colors.size):
        vertices = result.tuple_at(index)
        assert result.tuple_index(vertices) == index
        assert result.color(vertices) == int(result.colors[vertices])
    with pytest.raises(ValueError, match="coordinates"):
        result.tuple_index((0, 1))
    with pytest.raises(ValueError, match="outside"):
        result.tuple_index((0, 1, 4))
    with pytest.raises(ValueError, match="outside"):
        result.tuple_at(result.colors.size)


@pytest.mark.parametrize("dimension", [3, 4])
def test_packed_and_matrix_graphs_match(dimension: int) -> None:
    adjacency = graph_matrix(
        7,
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0)],
    )
    backend = "native" if native_available() else "reference"
    matrix = graph_wlk_refinement(adjacency, dimension, backend=backend)
    packed = graph_wlk_refinement(
        pack_graph(adjacency), dimension, backend=backend
    )
    np.testing.assert_array_equal(packed.colors, matrix.colors)
    np.testing.assert_array_equal(packed.color_sizes, matrix.color_sizes)


def test_convenience_functions_match_dimension_dispatch() -> None:
    adjacency = graph_matrix(4, [(0, 1), (1, 2), (2, 3)])
    backend = "native" if native_available() else "reference"
    three = graph_wl3_refinement(adjacency, backend=backend)
    four = graph_wl4_refinement(adjacency, backend=backend)
    np.testing.assert_array_equal(
        three.colors,
        graph_wlk_refinement(adjacency, 3, backend=backend).colors,
    )
    np.testing.assert_array_equal(
        four.colors,
        graph_wlk_refinement(adjacency, 4, backend=backend).colors,
    )


@pytest.mark.parametrize(
    ("adjacency", "dimension", "max_tuples", "message"),
    [
        (np.zeros((2, 2), dtype=np.uint8), 2, 10_000_000, "dimension"),
        (np.zeros((2, 2), dtype=np.uint8), 5, 10_000_000, "dimension"),
        (np.zeros((4, 4), dtype=np.uint8), 4, 100, "needs 256"),
        (np.zeros((2, 2), dtype=np.uint8), 3, 0, "max_tuples"),
        (np.asarray([[1]], dtype=np.uint8), 3, 10_000_000, "loopless"),
        (np.asarray([[0, 2], [0, 0]]), 3, 10_000_000, "Boolean"),
        (np.asarray([[0.5], [0.0]]), 3, 10_000_000, "integers"),
        (np.asarray([[-1], [0]], dtype=np.int64), 3, 10_000_000, "nonnegative"),
        (
            np.asarray([[4], [0]], dtype=np.uint64),
            3,
            10_000_000,
            "out-of-range",
        ),
    ],
)
def test_rejects_hostile_inputs(
    adjacency, dimension: int, max_tuples: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        graph_wlk_refinement(
            adjacency,
            dimension,
            max_tuples=max_tuples,
            backend="reference",
        )


@pytest.mark.parametrize(
    ("packed", "dimension", "message"),
    [
        (np.asarray([[0], [0]], dtype=np.uint64), 2, "dimension"),
        (np.asarray([[1]], dtype=np.uint64), 3, "loopless"),
        (np.asarray([[4], [0]], dtype=np.uint64), 3, "out-of-range"),
    ],
)
def test_native_abi_rejects_hostile_inputs(
    packed: np.ndarray, dimension: int, message: str
) -> None:
    if not native_available():
        pytest.skip("native library is unavailable")
    with pytest.raises(RuntimeError, match=message):
        graph_wlk_refine_native(packed, dimension)
