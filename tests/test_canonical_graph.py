from __future__ import annotations

import numpy as np
import pytest

from fast_math import (
    canonicalize_colored_digraphs,
    pack_digraph_adjacency,
)
from lambda_fast import available_backends
from lambda_fast._native import load_library


NATIVE_AVAILABLE = (
    "native" in available_backends()
    and hasattr(load_library(), "fast_math_canonical_digraphs_nauty_u64")
)


def relabel(
    adjacency: np.ndarray,
    colors: np.ndarray,
    permutation: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        adjacency[np.ix_(permutation, permutation)],
        colors[list(permutation)],
    )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_nauty_classes_and_automorphisms_match_reference() -> None:
    first = np.zeros((6, 6), dtype=np.uint8)
    for left, right in (
        (0, 2),
        (1, 2),
        (2, 3),
        (2, 4),
        (3, 5),
        (4, 5),
    ):
        first[left, right] = 1
    colors = np.asarray([0, 0, 1, 2, 2, 3], dtype=np.uint32)
    second, second_colors = relabel(
        first,
        colors,
        (1, 0, 2, 4, 3, 5),
    )
    third = first.copy()
    third[4, 5] = 0
    third[5, 4] = 1
    dense = np.stack((first, second, third))
    packed = pack_digraph_adjacency(dense)
    color_batch = np.stack((colors, second_colors, colors))

    reference = canonicalize_colored_digraphs(
        packed,
        color_batch,
        backend="reference",
    )
    native = canonicalize_colored_digraphs(
        packed,
        color_batch,
        backend="native",
    )
    assert native.class_ids[0] == native.class_ids[1]
    assert native.class_ids[0] != native.class_ids[2]
    assert reference.class_ids[0] == reference.class_ids[1]
    assert reference.class_ids[0] != reference.class_ids[2]
    np.testing.assert_array_equal(
        native.automorphism_group_mantissas,
        reference.automorphism_group_mantissas,
    )
    np.testing.assert_array_equal(
        native.automorphism_group_exponents,
        reference.automorphism_group_exponents,
    )
    np.testing.assert_array_equal(native.orbit_counts, reference.orbit_counts)
    for graph_index in range(len(packed)):
        begin = int(native.automorphism_generator_offsets[graph_index])
        end = int(native.automorphism_generator_offsets[graph_index + 1])
        for generator in native.automorphism_generators[begin:end]:
            permutation = tuple(map(int, generator))
            reconstructed = relabel(
                dense[graph_index],
                color_batch[graph_index],
                permutation,
            )
            np.testing.assert_array_equal(
                reconstructed[0],
                dense[graph_index],
            )
            np.testing.assert_array_equal(
                reconstructed[1],
                color_batch[graph_index],
            )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_nauty_permutation_reconstructs_canonical_graph() -> None:
    adjacency = np.zeros((1, 5, 5), dtype=np.uint8)
    adjacency[0, 0, 1] = 1
    adjacency[0, 1, 2] = 1
    adjacency[0, 0, 3] = 1
    adjacency[0, 3, 4] = 1
    colors = np.asarray([[0, 1, 2, 1, 2]], dtype=np.uint32)
    packed = pack_digraph_adjacency(adjacency)
    result = canonicalize_colored_digraphs(
        packed,
        colors,
        backend="native",
    )
    permutation = tuple(map(int, result.permutations[0]))
    reconstructed_dense = adjacency[0][np.ix_(permutation, permutation)]
    reconstructed = pack_digraph_adjacency(reconstructed_dense)[0]
    np.testing.assert_array_equal(
        reconstructed,
        result.adjacency_words[0],
    )
    np.testing.assert_array_equal(
        colors[0, list(permutation)],
        result.vertex_colors[0],
    )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_nauty_multiword_isomorphism_at_seventy_vertices() -> None:
    rng = np.random.default_rng(7821)
    vertex_count = 70
    first = np.zeros((vertex_count, vertex_count), dtype=np.uint8)
    for left in range(vertex_count):
        present = rng.random(vertex_count - left - 1) < 0.08
        first[left, left + 1 :] = present
    colors = rng.integers(0, 7, size=vertex_count, dtype=np.uint32)
    permutation = tuple(map(int, rng.permutation(vertex_count)))
    second, second_colors = relabel(first, colors, permutation)
    result = canonicalize_colored_digraphs(
        pack_digraph_adjacency(np.stack((first, second))),
        np.stack((colors, second_colors)),
        backend="native",
    )
    assert result.class_ids[0] == result.class_ids[1]
    np.testing.assert_array_equal(
        result.adjacency_words[0],
        result.adjacency_words[1],
    )
    np.testing.assert_array_equal(
        result.vertex_colors[0],
        result.vertex_colors[1],
    )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_parallel_nauty_matches_serial_exactly() -> None:
    rng = np.random.default_rng(8294)
    dense = np.zeros((32, 12, 12), dtype=np.uint8)
    for graph in dense:
        graph[:] = rng.random((12, 12)) < 0.18
        np.fill_diagonal(graph, 0)
    colors = rng.integers(0, 3, size=(32, 12), dtype=np.uint32)
    packed = pack_digraph_adjacency(dense)
    serial = canonicalize_colored_digraphs(
        packed,
        colors,
        threads=1,
        backend="native",
    )
    parallel = canonicalize_colored_digraphs(
        packed,
        colors,
        threads=4,
        backend="native",
    )
    for field in (
        "permutations",
        "adjacency_words",
        "vertex_colors",
        "class_ids",
        "automorphism_group_mantissas",
        "automorphism_group_exponents",
        "orbit_counts",
        "automorphism_generator_offsets",
        "automorphism_generators",
    ):
        np.testing.assert_array_equal(
            getattr(parallel, field),
            getattr(serial, field),
        )
    assert parallel.search_nodes == serial.search_nodes


@pytest.mark.parametrize(
    "backend",
    [
        "reference",
        pytest.param(
            "native",
            marks=pytest.mark.skipif(
                not NATIVE_AVAILABLE,
                reason="native library is not built",
            ),
        ),
    ],
)
def test_canonicalization_can_skip_automorphism_generators(
    backend: str,
) -> None:
    dense = np.zeros((2, 6, 6), dtype=np.uint8)
    for graph in dense:
        for vertex in range(6):
            graph[vertex, (vertex + 1) % 6] = 1
    packed = pack_digraph_adjacency(dense)
    colors = np.zeros((2, 6), dtype=np.uint32)
    full = canonicalize_colored_digraphs(
        packed,
        colors,
        threads=2,
        backend=backend,
    )
    compact = canonicalize_colored_digraphs(
        packed,
        colors,
        threads=2,
        collect_automorphism_generators=False,
        backend=backend,
    )
    np.testing.assert_array_equal(
        compact.adjacency_words,
        full.adjacency_words,
    )
    np.testing.assert_array_equal(
        compact.automorphism_group_mantissas,
        full.automorphism_group_mantissas,
    )
    np.testing.assert_array_equal(compact.orbit_counts, full.orbit_counts)
    np.testing.assert_array_equal(
        compact.automorphism_generator_offsets,
        np.zeros(3, dtype=np.uint64),
    )
    assert compact.automorphism_generators.shape == (0, 6)


def test_pack_digraph_rejects_self_loops() -> None:
    adjacency = np.eye(3, dtype=np.uint8)
    with pytest.raises(ValueError, match="self-loop"):
        pack_digraph_adjacency(adjacency)


def test_canonicalization_rejects_color_narrowing() -> None:
    with pytest.raises(ValueError, match="fit uint32"):
        canonicalize_colored_digraphs(
            pack_digraph_adjacency(np.zeros((2, 2), dtype=np.uint8)),
            [0, 2**32],
            backend="reference",
        )
