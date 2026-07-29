from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations
from math import comb

import numpy as np
import pytest

from fast_math import (
    decode_graph6,
    encode_graph6 as encode_graph6_batch,
    find_cliques,
    find_independent_sets,
    graph_invariants,
    graph_pair_profiles,
    induced_subgraph_profile_stack,
    induced_subgraph_profiles,
)
from lambda_fast import available_backends


NATIVE_AVAILABLE = "native" in available_backends()


def graph_masks(
    vertex_count: int,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    adjacency = np.zeros(vertex_count, dtype=np.uint64)
    for left, right in edges:
        adjacency[left] |= np.uint64(1 << right)
        adjacency[right] |= np.uint64(1 << left)
    return adjacency


def random_graphs(
    seed: int,
    graph_count: int,
    vertex_count: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    graphs = np.zeros((graph_count, vertex_count), dtype=np.uint64)
    for left in range(vertex_count):
        for right in range(left + 1, vertex_count):
            present = rng.integers(0, 2, size=graph_count, dtype=np.uint64)
            graphs[:, left] |= present << np.uint64(right)
            graphs[:, right] |= present << np.uint64(left)
    return graphs


def random_graphs_density(
    seed: int,
    graph_count: int,
    vertex_count: int,
    density: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    graphs = np.zeros((graph_count, vertex_count), dtype=np.uint64)
    for left in range(vertex_count):
        for right in range(left + 1, vertex_count):
            present = (rng.random(graph_count) < density).astype(np.uint64)
            graphs[:, left] |= present << np.uint64(right)
            graphs[:, right] |= present << np.uint64(left)
    return graphs


def all_labeled_graphs(vertex_count: int) -> np.ndarray:
    edges = tuple(combinations(range(vertex_count), 2))
    graph_count = 1 << len(edges)
    edge_masks = np.arange(graph_count, dtype=np.uint64)
    graphs = np.zeros((graph_count, vertex_count), dtype=np.uint64)
    for edge, (left, right) in enumerate(edges):
        present = (edge_masks >> np.uint64(edge)) & np.uint64(1)
        graphs[:, left] |= present << np.uint64(right)
        graphs[:, right] |= present << np.uint64(left)
    return graphs


def encode_graph6(adjacency: np.ndarray) -> bytes:
    vertex_count = len(adjacency)
    bits = [
        int(bool(int(adjacency[left]) & (1 << right)))
        for right in range(1, vertex_count)
        for left in range(right)
    ]
    bits.extend([0] * ((-len(bits)) % 6))
    payload = bytearray([vertex_count + 63])
    for begin in range(0, len(bits), 6):
        value = 0
        for bit in bits[begin : begin + 6]:
            value = (value << 1) | bit
        payload.append(value + 63)
    return bytes(payload)


@lru_cache(maxsize=None)
def canonical_class_lookup(order: int) -> np.ndarray:
    edges = tuple(combinations(range(order), 2))
    edge_index = {edge: index for index, edge in enumerate(edges)}
    edge_maps = []
    for permutation in permutations(range(order)):
        edge_maps.append(
            tuple(
                edge_index[
                    tuple(sorted((permutation[left], permutation[right])))
                ]
                for left, right in edges
            )
        )

    def relabel(mask: int, edge_map: tuple[int, ...]) -> int:
        result = 0
        while mask:
            bit = mask & -mask
            source = bit.bit_length() - 1
            result |= 1 << edge_map[source]
            mask ^= bit
        return result

    canonical = [
        min(relabel(mask, edge_map) for edge_map in edge_maps)
        for mask in range(1 << len(edges))
    ]
    representatives = {
        value: index
        for index, value in enumerate(sorted(set(canonical)))
    }
    return np.array(
        [representatives[value] for value in canonical],
        dtype=np.uint32,
    )


@lru_cache(maxsize=None)
def edge_count_class_lookup(order: int) -> np.ndarray:
    edge_count = order * (order - 1) // 2
    masks = np.arange(1 << edge_count, dtype=np.uint32)
    return np.bitwise_count(masks).astype(np.uint32)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_graph_pair_profiles_match_reference() -> None:
    graphs = random_graphs(7501, 37, 12)
    reference = graph_pair_profiles(graphs, backend="reference")
    native = graph_pair_profiles(
        graphs,
        threads=5,
        backend="native",
    )
    np.testing.assert_array_equal(native.left, reference.left)
    np.testing.assert_array_equal(native.right, reference.right)
    np.testing.assert_array_equal(native.adjacent, reference.adjacent)
    np.testing.assert_array_equal(
        native.common_neighbors,
        reference.common_neighbors,
    )
    np.testing.assert_array_equal(
        native.common_nonneighbors,
        reference.common_nonneighbors,
    )
    np.testing.assert_array_equal(native.only_left, reference.only_left)
    np.testing.assert_array_equal(native.only_right, reference.only_right)


def assert_witness(
    adjacency: np.ndarray,
    witness: int,
    order: int,
    *,
    complement: bool,
) -> None:
    vertices = [
        vertex
        for vertex in range(len(adjacency))
        if witness & (1 << vertex)
    ]
    assert len(vertices) == order
    for left_index, left in enumerate(vertices):
        for right in vertices[left_index + 1 :]:
            adjacent = bool(int(adjacency[left]) & (1 << right))
            assert adjacent != complement


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
@pytest.mark.parametrize("complement", [False, True])
def test_clique_search_matches_reference(complement: bool) -> None:
    graphs = random_graphs(7502, 41, 14)
    reference = find_cliques(
        graphs,
        4,
        complement=complement,
        backend="reference",
    )
    native = find_cliques(
        graphs,
        4,
        complement=complement,
        threads=5,
        backend="native",
    )
    np.testing.assert_array_equal(native.found, reference.found)
    for adjacency, witness in zip(
        graphs,
        native.witness_masks,
        strict=True,
    ):
        if witness:
            assert_witness(
                adjacency,
                int(witness),
                4,
                complement=complement,
            )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_complete_and_empty_graph_extremes() -> None:
    complete = graph_masks(
        8,
        [
            (left, right)
            for left in range(8)
            for right in range(left + 1, 8)
        ],
    )
    empty = graph_masks(8, [])
    graphs = np.stack([complete, empty])
    cliques = find_cliques(graphs, 5, backend="native")
    independent = find_independent_sets(graphs, 5, backend="native")
    np.testing.assert_array_equal(cliques.found, [True, False])
    np.testing.assert_array_equal(independent.found, [False, True])


def test_graph_masks_reject_noninteger_values() -> None:
    with pytest.raises(ValueError, match="integers"):
        graph_pair_profiles([[0.0, 0.0]])


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_pair_profiles_are_exhaustive_through_order_five() -> None:
    graphs = all_labeled_graphs(5)
    reference = graph_pair_profiles(graphs, backend="reference")
    native = graph_pair_profiles(graphs, threads=5, backend="native")
    np.testing.assert_array_equal(native.adjacent, reference.adjacent)
    np.testing.assert_array_equal(
        native.common_neighbors,
        reference.common_neighbors,
    )
    np.testing.assert_array_equal(
        native.common_nonneighbors,
        reference.common_nonneighbors,
    )
    np.testing.assert_array_equal(native.only_left, reference.only_left)
    np.testing.assert_array_equal(native.only_right, reference.only_right)
    np.testing.assert_array_equal(
        native.common_neighbors
        + native.common_nonneighbors
        + native.only_left
        + native.only_right,
        np.full_like(native.common_neighbors, 3),
    )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_pair_profiles_match_reference_at_graph64_boundary() -> None:
    graphs = random_graphs_density(7793, 9, 64, 0.5)
    reference = graph_pair_profiles(graphs, backend="reference")
    native = graph_pair_profiles(graphs, threads=5, backend="native")
    np.testing.assert_array_equal(native.adjacent, reference.adjacent)
    np.testing.assert_array_equal(
        native.common_neighbors,
        reference.common_neighbors,
    )
    np.testing.assert_array_equal(
        native.common_nonneighbors,
        reference.common_nonneighbors,
    )
    np.testing.assert_array_equal(native.only_left, reference.only_left)
    np.testing.assert_array_equal(native.only_right, reference.only_right)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
@pytest.mark.parametrize("complement", [False, True])
def test_clique_search_is_exhaustive_through_order_five(
    complement: bool,
) -> None:
    graphs = all_labeled_graphs(5)
    for order in range(1, 6):
        reference = find_cliques(
            graphs,
            order,
            complement=complement,
            backend="reference",
        )
        native = find_cliques(
            graphs,
            order,
            complement=complement,
            threads=5,
            backend="native",
        )
        np.testing.assert_array_equal(native.found, reference.found)
        np.testing.assert_array_equal(
            native.nodes_visited,
            reference.nodes_visited,
        )
        for adjacency, witness in zip(
            graphs[native.found],
            native.witness_masks[native.found],
            strict=True,
        ):
            assert_witness(
                adjacency,
                int(witness),
                order,
                complement=complement,
            )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_graph_invariants_are_exhaustive_through_order_six() -> None:
    graphs = all_labeled_graphs(6)
    reference = graph_invariants(graphs, backend="reference")
    native = graph_invariants(graphs, threads=5, backend="native")
    np.testing.assert_array_equal(native.degrees, reference.degrees)
    np.testing.assert_array_equal(native.edge_counts, reference.edge_counts)
    np.testing.assert_array_equal(
        native.triangle_counts,
        reference.triangle_counts,
    )
    np.testing.assert_array_equal(native.wedge_counts, reference.wedge_counts)
    np.testing.assert_array_equal(
        native.induced_path3_counts,
        reference.induced_path3_counts,
    )
    np.testing.assert_array_equal(
        np.sum(native.degrees, axis=1, dtype=np.uint64),
        2 * native.edge_counts,
    )
    np.testing.assert_array_equal(
        native.induced_path3_counts,
        native.wedge_counts - 3 * native.triangle_counts,
    )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
@pytest.mark.parametrize("vertex_count", [31, 32, 43, 64])
@pytest.mark.parametrize("density", [0.1, 0.5, 0.9])
def test_graph_invariant_density_paths_match_reference(
    vertex_count: int,
    density: float,
) -> None:
    graphs = random_graphs_density(
        7700 + vertex_count + int(10 * density),
        17,
        vertex_count,
        density,
    )
    reference = graph_invariants(graphs, backend="reference")
    native = graph_invariants(graphs, threads=5, backend="native")
    np.testing.assert_array_equal(native.degrees, reference.degrees)
    np.testing.assert_array_equal(native.edge_counts, reference.edge_counts)
    np.testing.assert_array_equal(
        native.triangle_counts,
        reference.triangle_counts,
    )
    np.testing.assert_array_equal(native.wedge_counts, reference.wedge_counts)
    np.testing.assert_array_equal(
        native.induced_path3_counts,
        reference.induced_path3_counts,
    )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
@pytest.mark.parametrize("order", range(1, 6))
def test_induced_profiles_are_exhaustive_through_order_five(
    order: int,
) -> None:
    graphs = all_labeled_graphs(5)
    lookup = canonical_class_lookup(order)
    reference = induced_subgraph_profiles(
        graphs,
        order,
        lookup,
        backend="reference",
    )
    native = induced_subgraph_profiles(
        graphs,
        order,
        lookup,
        threads=5,
        backend="native",
    )
    np.testing.assert_array_equal(native.counts, reference.counts)
    np.testing.assert_array_equal(
        np.sum(native.counts, axis=1, dtype=np.uint64),
        np.full(len(graphs), comb(5, order), dtype=np.uint64),
    )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_induced_profile_stack_matches_individual_kernels() -> None:
    graphs = all_labeled_graphs(5)
    orders = (1, 2, 3, 4, 5)
    lookups = tuple(canonical_class_lookup(order) for order in orders)
    stack = induced_subgraph_profile_stack(
        graphs,
        orders,
        lookups,
        threads=2,
        backend="native",
    )
    assert stack.induced_orders == orders
    assert stack.subsets_per_graph == tuple(comb(5, order) for order in orders)
    for order, lookup in zip(orders, lookups, strict=True):
        individual = induced_subgraph_profiles(
            graphs,
            order,
            lookup,
            threads=2,
            backend="native",
        )
        np.testing.assert_array_equal(
            stack.counts_for_order(order),
            individual.counts,
        )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_induced_profile_stack_matches_reference_and_is_deterministic() -> None:
    graphs = random_graphs(7614, 307, 7)
    orders = (2, 3, 5)
    lookups = tuple(canonical_class_lookup(order) for order in orders)
    reference = induced_subgraph_profile_stack(
        graphs,
        orders,
        lookups,
        backend="reference",
    )
    single = induced_subgraph_profile_stack(
        graphs,
        orders,
        lookups,
        threads=1,
        backend="native",
    )
    parallel = induced_subgraph_profile_stack(
        graphs,
        orders,
        lookups,
        threads=2,
        backend="native",
    )
    np.testing.assert_array_equal(single.counts, reference.counts)
    np.testing.assert_array_equal(parallel.counts, reference.counts)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
@pytest.mark.parametrize("vertex_count", [22, 23])
def test_induced_profile_large_subset_boundaries(vertex_count: int) -> None:
    graph = np.zeros((1, vertex_count), dtype=np.uint64)
    result = induced_subgraph_profiles(
        graph,
        7,
        edge_count_class_lookup(7),
        threads=1,
        backend="native",
    )
    expected = comb(vertex_count, 7)
    assert int(result.counts[0, 0]) == expected
    assert int(np.sum(result.counts[0, 1:], dtype=np.uint64)) == 0


def test_induced_profile_stack_rejects_invalid_contracts() -> None:
    graph = graph_masks(4, [])
    with pytest.raises(ValueError, match="equal nonzero length"):
        induced_subgraph_profile_stack(
            graph,
            [1, 2],
            [[0]],
            backend="reference",
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        induced_subgraph_profile_stack(
            graph,
            [2, 1],
            [[0, 1], [0]],
            backend="reference",
        )
    with pytest.raises(ValueError, match="integers"):
        induced_subgraph_profile_stack(
            graph,
            [1.5],
            [[0]],
            backend="reference",
        )
    with pytest.raises(ValueError, match="not in this stack"):
        induced_subgraph_profile_stack(
            graph,
            [1],
            [[0]],
            backend="reference",
        ).counts_for_order(2)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
@pytest.mark.parametrize("vertex_count", [1, 2, 5, 9, 43, 62])
def test_graph6_round_trip_matches_reference(vertex_count: int) -> None:
    graphs = random_graphs(7600 + vertex_count, 19, vertex_count)
    records = [encode_graph6(graph) for graph in graphs]
    reference_encoding = encode_graph6_batch(graphs, backend="reference")
    native_encoding = encode_graph6_batch(
        graphs,
        threads=5,
        backend="native",
    )
    assert reference_encoding.records == tuple(
        record.decode("ascii") for record in records
    )
    assert native_encoding.records == reference_encoding.records
    reference = decode_graph6(records, backend="reference")
    native = decode_graph6(records, threads=5, backend="native")
    np.testing.assert_array_equal(reference.adjacency_masks, graphs)
    np.testing.assert_array_equal(native.adjacency_masks, graphs)


@pytest.mark.parametrize(
    ("record", "vertex_count", "edges"),
    [
        (b"@", 1, []),
        (b"Ch", 4, [(0, 1), (1, 2), (2, 3)]),
        (b"Dhc", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]),
    ],
)
def test_graph6_matches_standard_known_encodings(
    record: bytes,
    vertex_count: int,
    edges: list[tuple[int, int]],
) -> None:
    decoded = decode_graph6(record, backend="reference")
    np.testing.assert_array_equal(
        decoded.adjacency_masks[0],
        graph_masks(vertex_count, edges),
    )
    assert encode_graph6_batch(
        decoded.adjacency_masks,
        backend="reference",
    ).records == (record.decode("ascii"),)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_new_graph_kernels_are_thread_deterministic() -> None:
    graphs = random_graphs(7609, 701, 9)
    records = [encode_graph6(graph) for graph in graphs]
    first_decode = decode_graph6(records, threads=1, backend="native")
    second_decode = decode_graph6(records, threads=5, backend="native")
    np.testing.assert_array_equal(
        first_decode.adjacency_masks,
        second_decode.adjacency_masks,
    )
    first_encode = encode_graph6_batch(
        graphs,
        threads=1,
        backend="native",
    )
    second_encode = encode_graph6_batch(
        graphs,
        threads=5,
        backend="native",
    )
    assert first_encode.records == second_encode.records

    first_invariants = graph_invariants(
        graphs,
        threads=1,
        backend="native",
    )
    second_invariants = graph_invariants(
        graphs,
        threads=5,
        backend="native",
    )
    np.testing.assert_array_equal(
        first_invariants.degrees,
        second_invariants.degrees,
    )
    np.testing.assert_array_equal(
        first_invariants.triangle_counts,
        second_invariants.triangle_counts,
    )

    lookup = canonical_class_lookup(6)
    first_profiles = induced_subgraph_profiles(
        graphs,
        6,
        lookup,
        threads=1,
        backend="native",
    )
    second_profiles = induced_subgraph_profiles(
        graphs,
        6,
        lookup,
        threads=5,
        backend="native",
    )
    np.testing.assert_array_equal(
        first_profiles.counts,
        second_profiles.counts,
    )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_graph64_boundary_invariants() -> None:
    complete = graph_masks(
        64,
        [
            (left, right)
            for left in range(64)
            for right in range(left + 1, 64)
        ],
    )
    result = graph_invariants(complete, backend="native")
    np.testing.assert_array_equal(result.degrees, np.full((1, 64), 63))
    assert int(result.edge_counts[0]) == comb(64, 2)
    assert int(result.triangle_counts[0]) == comb(64, 3)
    assert int(result.induced_path3_counts[0]) == 0


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_single_vertex_graph_boundary() -> None:
    graph = np.zeros((1, 1), dtype=np.uint64)
    pairs = graph_pair_profiles(graph, backend="native")
    assert pairs.adjacent.shape == (1, 0)
    clique = find_cliques(graph, 1, backend="native")
    assert clique.found.tolist() == [True]
    assert clique.witness_masks.tolist() == [1]
    invariants = graph_invariants(graph, backend="native")
    np.testing.assert_array_equal(invariants.degrees, [[0]])
    np.testing.assert_array_equal(invariants.edge_counts, [0])
    profiles = induced_subgraph_profiles(
        graph,
        1,
        [0],
        backend="native",
    )
    np.testing.assert_array_equal(profiles.counts, [[1]])


@pytest.mark.parametrize(
    "graphs",
    [
        [[2, 0]],
        [[0, 1]],
        [[4, 0]],
    ],
)
def test_graph_kernels_reject_invalid_adjacency(graphs) -> None:
    with pytest.raises(ValueError):
        graph_invariants(graphs, backend="reference")


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
@pytest.mark.parametrize(
    ("vertex_count", "kind", "message"),
    [
        (31, "asymmetric", "undirected"),
        (32, "asymmetric", "undirected"),
        (64, "asymmetric", "undirected"),
        (32, "self_loop", "self-loop"),
        (32, "out_of_range", "invalid vertex"),
    ],
)
def test_native_validation_rejects_invalid_adjacency(
    vertex_count: int,
    kind: str,
    message: str,
) -> None:
    graph = np.zeros((1, vertex_count), dtype=np.uint64)
    if kind == "asymmetric":
        graph[0, 0] = np.uint64(1 << (vertex_count - 1))
    elif kind == "self_loop":
        graph[0, vertex_count - 1] = np.uint64(1 << (vertex_count - 1))
    else:
        graph[0, 0] = np.uint64(1 << vertex_count)
    with pytest.raises(RuntimeError, match=message):
        graph_invariants(graph, backend="native")


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
@pytest.mark.parametrize(
    ("first_kind", "message"),
    [
        ("asymmetric", "undirected"),
        ("self_loop", "self-loop"),
    ],
)
def test_parallel_validation_reports_first_invalid_graph(
    first_kind: str,
    message: str,
) -> None:
    graphs = random_graphs_density(7791, 307, 43, 0.5)
    if first_kind == "asymmetric":
        graphs[17, 0] ^= np.uint64(1 << 42)
        graphs[211, 42] |= np.uint64(1 << 42)
    else:
        graphs[17, 42] |= np.uint64(1 << 42)
        graphs[211, 0] ^= np.uint64(1 << 42)
    with pytest.raises(RuntimeError, match=message):
        graph_invariants(graphs, threads=5, backend="native")


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native library is not built")
def test_parallel_swar_validation_matches_single_thread() -> None:
    graphs = random_graphs_density(7792, 307, 43, 0.5)
    single = graph_invariants(graphs, threads=1, backend="native")
    parallel = graph_invariants(graphs, threads=5, backend="native")
    np.testing.assert_array_equal(parallel.degrees, single.degrees)
    np.testing.assert_array_equal(
        parallel.triangle_counts,
        single.triangle_counts,
    )
    np.testing.assert_array_equal(
        parallel.induced_path3_counts,
        single.induced_path3_counts,
    )


@pytest.mark.parametrize(
    "records",
    [
        [],
        [b">>graph6<<D??"],
        [b"D?"],
        [b"D??", b"E???"],
        ["\N{SNOWMAN}"],
    ],
)
def test_graph6_rejects_invalid_records(records) -> None:
    with pytest.raises(ValueError):
        decode_graph6(records, backend="reference")


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_graph6_encode_rejects_invalid_adjacency(backend: str) -> None:
    if backend == "native" and not NATIVE_AVAILABLE:
        pytest.skip("native library is not built")
    invalid = np.asarray([[2, 0]], dtype=np.uint64)
    with pytest.raises((ValueError, RuntimeError), match="undirected"):
        encode_graph6_batch(invalid, backend=backend)


@pytest.mark.parametrize(
    ("order", "lookup", "message"),
    [
        (0, [0], "induced_order"),
        (3, [0, 1], "one integer class"),
        (2, [0, 2], "dense"),
        (2, [0, 4_000_000_000], "dense"),
        (2, [0, -1], "nonnegative"),
    ],
)
def test_induced_profiles_reject_bad_lookup(
    order: int,
    lookup,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        induced_subgraph_profiles(
            graph_masks(3, []),
            order,
            lookup,
            backend="reference",
        )
