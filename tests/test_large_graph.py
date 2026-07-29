from __future__ import annotations

import numpy as np
import pytest

from fast_math import (
    csr_common_neighbors,
    enumerate_csr_triangles,
    undirected_csr,
)
from lambda_fast import available_backends
from lambda_fast._native import load_library


NATIVE_AVAILABLE = (
    "native" in available_backends()
    and hasattr(load_library(), "fast_math_graph_triangles_csr_u32")
)
COMMON_NEIGHBOR_NATIVE_AVAILABLE = (
    "native" in available_backends()
    and hasattr(
        load_library(),
        "fast_math_graph_common_neighbors_csr_u32",
    )
)


def test_undirected_csr_is_sorted_and_symmetric() -> None:
    graph = undirected_csr(
        5,
        [(3, 1), (0, 4), (1, 0), (3, 3)],
        edge_color_masks=[4, 2, 1, 8],
    )
    np.testing.assert_array_equal(graph.row_offsets, [0, 2, 4, 4, 5, 6])
    np.testing.assert_array_equal(graph.column_indices, [1, 4, 0, 3, 1, 0])
    np.testing.assert_array_equal(graph.edge_color_masks, [1, 2, 1, 4, 4, 2])
    np.testing.assert_array_equal(graph.vertex_loop_color_masks, [0, 0, 0, 8, 0])


def test_csr_common_neighbors_reference_counts_and_materialization() -> None:
    graph = undirected_csr(
        5,
        [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)],
    )
    pairs = [(0, 1), (0, 3), (1, 1), (4, 0)]
    counts = csr_common_neighbors(
        graph.row_offsets,
        graph.column_indices,
        pairs,
        backend="reference",
    )
    materialized = csr_common_neighbors(
        graph.row_offsets,
        graph.column_indices,
        pairs,
        materialize=True,
        backend="reference",
    )
    np.testing.assert_array_equal(counts.pair_offsets, [0, 1, 3, 6, 6])
    np.testing.assert_array_equal(counts.counts, [1, 2, 3, 0])
    assert counts.common_neighbors is None
    np.testing.assert_array_equal(
        materialized.pair_offsets,
        counts.pair_offsets,
    )
    np.testing.assert_array_equal(
        materialized.common_neighbors,
        [2, 1, 2, 0, 2, 3],
    )


@pytest.mark.skipif(
    not COMMON_NEIGHBOR_NATIVE_AVAILABLE,
    reason="native common-neighbor kernel is not built",
)
@pytest.mark.parametrize("materialize", [False, True])
def test_csr_common_neighbors_native_matches_reference(
    materialize: bool,
) -> None:
    graph = undirected_csr(
        7,
        [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
            (5, 6),
        ],
    )
    pairs = np.asarray(
        [(0, 1), (0, 3), (4, 4), (5, 6), (6, 5)],
        dtype=np.uint32,
    )
    reference = csr_common_neighbors(
        graph.row_offsets,
        graph.column_indices,
        pairs,
        materialize=materialize,
        backend="reference",
    )
    native = csr_common_neighbors(
        graph.row_offsets,
        graph.column_indices,
        pairs,
        materialize=materialize,
        backend="native",
    )
    np.testing.assert_array_equal(native.pair_offsets, reference.pair_offsets)
    if materialize:
        np.testing.assert_array_equal(
            native.common_neighbors,
            reference.common_neighbors,
        )
    else:
        assert native.common_neighbors is None
    assert native.intersection_steps == reference.intersection_steps


@pytest.mark.parametrize(
    ("pairs", "message"),
    [
        ([0, 1], "shape"),
        ([[0.0, 1.0]], "integers"),
        ([[-1, 1]], "nonnegative"),
        ([[0, 3]], "outside"),
    ],
)
def test_csr_common_neighbors_rejects_invalid_pairs(
    pairs,
    message: str,
) -> None:
    graph = undirected_csr(3, [(0, 1)])
    with pytest.raises(ValueError, match=message):
        csr_common_neighbors(
            graph.row_offsets,
            graph.column_indices,
            pairs,
            backend="reference",
        )


def test_csr_common_neighbors_rejects_nonboolean_materialize() -> None:
    graph = undirected_csr(2, [(0, 1)])
    with pytest.raises(TypeError, match="boolean"):
        csr_common_neighbors(
            graph.row_offsets,
            graph.column_indices,
            [(0, 1)],
            materialize=1,
            backend="reference",
        )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_large_colored_triangles_match_reference() -> None:
    graph = undirected_csr(
        8,
        [
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 3),
            (4, 5),
            (4, 6),
            (5, 6),
            (5, 7),
            (1, 1),
        ],
        edge_color_masks=[1, 2, 4, 8, 16, 3, 5, 9, 17, 32],
    )
    reference = enumerate_csr_triangles(
        graph.row_offsets,
        graph.column_indices,
        edge_color_masks=graph.edge_color_masks,
        vertex_loop_color_masks=graph.vertex_loop_color_masks,
        backend="reference",
    )
    native = enumerate_csr_triangles(
        graph.row_offsets,
        graph.column_indices,
        edge_color_masks=graph.edge_color_masks,
        vertex_loop_color_masks=graph.vertex_loop_color_masks,
        backend="native",
    )
    np.testing.assert_array_equal(native.triangles, reference.triangles)
    np.testing.assert_array_equal(
        native.edge_color_masks,
        reference.edge_color_masks,
    )
    assert tuple(map(tuple, native.triangles)) == (
        (0, 1, 1),
        (0, 1, 2),
        (1, 1, 1),
        (1, 1, 2),
        (1, 1, 3),
        (1, 2, 3),
        (4, 5, 6),
    )


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        ([(0, 3)], "outside"),
        ([(0, 1), (1, 0)], "duplicate"),
    ],
)
def test_undirected_csr_rejects_invalid_edges(edges, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        undirected_csr(3, edges)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_large_triangle_native_rejects_asymmetric_csr() -> None:
    with pytest.raises(RuntimeError, match="undirected"):
        enumerate_csr_triangles(
            [0, 1, 1],
            [1],
            backend="native",
        )


@pytest.mark.skipif(
    not COMMON_NEIGHBOR_NATIVE_AVAILABLE,
    reason="native common-neighbor kernel is not built",
)
def test_common_neighbor_native_rejects_asymmetric_csr() -> None:
    with pytest.raises(RuntimeError, match="undirected"):
        csr_common_neighbors(
            [0, 1, 1],
            [1],
            [(0, 1)],
            backend="native",
        )


def test_large_triangle_rejects_column_narrowing() -> None:
    with pytest.raises(ValueError, match="outside"):
        enumerate_csr_triangles(
            [0, 1, 1],
            [2**32 + 1],
            backend="reference",
        )
