from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from operator import index
import time
from typing import Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._native import (
    NativeUnavailable,
    graph6_decode_native,
    graph6_encode_native,
    graph_delete_vertices_native,
    graph_rooted_leaf_features_native,
    graph_find_clique_native,
    graph_induced_profiles_native,
    graph_induced_profile_stack_native,
    graph_invariants_native,
    graph_pair_profiles_native,
)


Backend = Literal["auto", "native", "reference"]


@dataclass(frozen=True)
class GraphPairProfiles:
    left: NDArray[np.uint32]
    right: NDArray[np.uint32]
    adjacent: NDArray[np.bool_]
    common_neighbors: NDArray[np.uint32]
    common_nonneighbors: NDArray[np.uint32]
    only_left: NDArray[np.uint32]
    only_right: NDArray[np.uint32]
    graph_count: int
    vertex_count: int
    elapsed_seconds: float
    backend: str

    @property
    def neighborhood_distance(self) -> NDArray[np.uint32]:
        return self.only_left + self.only_right


@dataclass(frozen=True)
class CliqueBatchResult:
    witness_masks: NDArray[np.uint64]
    nodes_visited: NDArray[np.uint64]
    graph_count: int
    vertex_count: int
    order: int
    complement: bool
    elapsed_seconds: float
    backend: str

    @property
    def found(self) -> NDArray[np.bool_]:
        return self.witness_masks != 0


@dataclass(frozen=True)
class DecodedGraphBatch:
    adjacency_masks: NDArray[np.uint64]
    graph_count: int
    vertex_count: int
    elapsed_seconds: float
    backend: str


@dataclass(frozen=True)
class EncodedGraphBatch:
    records: tuple[str, ...]
    graph_count: int
    vertex_count: int
    elapsed_seconds: float
    backend: str


@dataclass(frozen=True)
class VertexDeletedGraphBatch:
    adjacency_masks: NDArray[np.uint64]
    source_graphs: NDArray[np.uint64]
    deleted_vertices: NDArray[np.uint32]
    graph_count: int
    vertex_count: int
    elapsed_seconds: float
    backend: str


@dataclass(frozen=True)
class RootedLeafFeatureBatch:
    root_degrees: NDArray[np.uint32]
    two_star_counts: NDArray[np.uint32]
    two_step_counts: NDArray[np.uint32]
    source_graphs: NDArray[np.uint64]
    leaf_vertices: NDArray[np.uint32]
    request_count: int
    vertex_count: int
    elapsed_seconds: float
    backend: str


@dataclass(frozen=True)
class GraphInvariantBatch:
    degrees: NDArray[np.uint32]
    edge_counts: NDArray[np.uint64]
    triangle_counts: NDArray[np.uint64]
    wedge_counts: NDArray[np.uint64]
    induced_path3_counts: NDArray[np.uint64]
    graph_count: int
    vertex_count: int
    elapsed_seconds: float
    backend: str


@dataclass(frozen=True)
class InducedProfileBatch:
    counts: NDArray[np.uint64]
    graph_count: int
    vertex_count: int
    induced_order: int
    class_count: int
    subsets_per_graph: int
    elapsed_seconds: float
    backend: str


@dataclass(frozen=True)
class InducedProfileStackBatch:
    counts: NDArray[np.uint64]
    graph_count: int
    vertex_count: int
    induced_orders: tuple[int, ...]
    class_counts: tuple[int, ...]
    offsets: NDArray[np.uint64]
    subsets_per_graph: tuple[int, ...]
    elapsed_seconds: float
    backend: str

    def counts_for_order(self, induced_order: int) -> NDArray[np.uint64]:
        try:
            index = self.induced_orders.index(induced_order)
        except ValueError as error:
            raise ValueError(
                f"induced order {induced_order} is not in this stack"
            ) from error
        begin = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return self.counts[:, begin:end]


def _prepare_graphs(adjacency_masks: ArrayLike) -> NDArray[np.uint64]:
    array = np.asarray(adjacency_masks)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if (
        array.ndim != 2
        or array.shape[0] == 0
        or not 1 <= array.shape[1] <= 64
    ):
        raise ValueError(
            "adjacency_masks must have shape (graphs, 1..64 vertices)"
        )
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError("adjacency masks must be integers")
    if array.dtype.kind == "i" and np.any(array < 0):
        raise ValueError("adjacency masks must be nonnegative")
    return np.ascontiguousarray(array, dtype=np.uint64)


def _prepare_graph6(
    records: str | bytes | Iterable[str | bytes],
) -> tuple[NDArray[np.uint8], NDArray[np.uint64], int]:
    if isinstance(records, (str, bytes)):
        values = (records,)
    else:
        values = tuple(records)
    if not values:
        raise ValueError("graph6 records must be nonempty")

    encoded: list[bytes] = []
    vertex_count = -1
    for record in values:
        if isinstance(record, str):
            try:
                data = record.encode("ascii")
            except UnicodeEncodeError as error:
                raise ValueError("graph6 records must be ASCII") from error
        elif isinstance(record, bytes):
            data = record
        else:
            raise ValueError("graph6 records must be strings or bytes")
        data = data.strip()
        if not data or data.startswith(b">>"):
            raise ValueError("expected unheaded short-form graph6 records")
        order = data[0] - 63
        if not 1 <= order <= 62:
            raise ValueError("short-form graph6 supports 1-62 vertices")
        expected_size = 1 + (order * (order - 1) // 2 + 5) // 6
        if len(data) != expected_size or any(
            byte < 63 or byte > 126 for byte in data
        ):
            raise ValueError("invalid short-form graph6 record")
        if vertex_count < 0:
            vertex_count = order
        elif order != vertex_count:
            raise ValueError("graph6 records must share one vertex count")
        encoded.append(data)

    offsets = np.empty(len(encoded) + 1, dtype=np.uint64)
    offsets[0] = 0
    for index, data in enumerate(encoded, start=1):
        offsets[index] = offsets[index - 1] + len(data)
    joined = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()
    return joined, offsets, vertex_count


def _validate_graphs(adjacency_masks: NDArray[np.uint64]) -> None:
    vertex_count = adjacency_masks.shape[1]
    all_vertices = (
        (1 << vertex_count) - 1
        if vertex_count < 64
        else (1 << 64) - 1
    )
    for adjacency in adjacency_masks:
        for left in range(vertex_count):
            mask = int(adjacency[left])
            if mask & ~all_vertices or mask & (1 << left):
                raise ValueError(
                    "adjacency masks contain an invalid vertex or self-loop"
                )
            for right in range(left + 1, vertex_count):
                if bool(mask & (1 << right)) != bool(
                    int(adjacency[right]) & (1 << left)
                ):
                    raise ValueError(
                        "adjacency masks must describe undirected graphs"
                    )


def _prepare_deletion_requests(
    source_graphs: ArrayLike,
    deleted_vertices: ArrayLike,
    *,
    graph_count: int,
    vertex_count: int,
) -> tuple[NDArray[np.uint64], NDArray[np.uint32]]:
    sources = np.asarray(source_graphs)
    deleted = np.asarray(deleted_vertices)
    for name, values in (
        ("source_graphs", sources),
        ("deleted_vertices", deleted),
    ):
        if values.ndim != 1 or len(values) == 0:
            raise ValueError(f"{name} must be a nonempty one-dimensional array")
        if values.dtype.kind not in {"i", "u"}:
            raise ValueError(f"{name} must contain integers")
        if values.dtype.kind == "i" and np.any(values < 0):
            raise ValueError(f"{name} must be nonnegative")
    if len(sources) != len(deleted):
        raise ValueError(
            "source_graphs and deleted_vertices must have equal length"
        )
    if np.any(sources >= graph_count):
        raise ValueError("source graph is out of range")
    if np.any(deleted >= vertex_count):
        raise ValueError("deleted vertex is out of range")
    return (
        np.ascontiguousarray(sources, dtype=np.uint64),
        np.ascontiguousarray(deleted, dtype=np.uint32),
    )


def _pair_vertices(
    vertex_count: int,
) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    left, right = np.triu_indices(vertex_count, 1)
    return left.astype(np.uint32), right.astype(np.uint32)


def _decode_graph6_reference(
    data: NDArray[np.uint8],
    offsets: NDArray[np.uint64],
    vertex_count: int,
) -> DecodedGraphBatch:
    started = time.perf_counter()
    graph_count = len(offsets) - 1
    adjacency_masks = np.zeros(
        (graph_count, vertex_count),
        dtype=np.uint64,
    )
    for graph in range(graph_count):
        begin = int(offsets[graph])
        bit_index = 0
        for right in range(1, vertex_count):
            for left in range(right):
                encoded = int(data[begin + 1 + bit_index // 6]) - 63
                present = (encoded >> (5 - bit_index % 6)) & 1
                bit_index += 1
                if present:
                    adjacency_masks[graph, left] |= np.uint64(1 << right)
                    adjacency_masks[graph, right] |= np.uint64(1 << left)
    return DecodedGraphBatch(
        adjacency_masks=adjacency_masks,
        graph_count=graph_count,
        vertex_count=vertex_count,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def _encoded_records(
    data: NDArray[np.uint8],
    graph_count: int,
    record_size: int,
) -> tuple[str, ...]:
    payload = memoryview(data)
    return tuple(
        bytes(payload[index * record_size : (index + 1) * record_size]).decode(
            "ascii"
        )
        for index in range(graph_count)
    )


def _encode_graph6_reference(
    adjacency_masks: NDArray[np.uint64],
) -> EncodedGraphBatch:
    started = time.perf_counter()
    _validate_graphs(adjacency_masks)
    graph_count, vertex_count = adjacency_masks.shape
    edge_bits = vertex_count * (vertex_count - 1) // 2
    record_size = 1 + (edge_bits + 5) // 6
    data = np.full(graph_count * record_size, 63, dtype=np.uint8)
    for graph, adjacency in enumerate(adjacency_masks):
        begin = graph * record_size
        data[begin] = vertex_count + 63
        bit_index = 0
        for right in range(1, vertex_count):
            for left in range(right):
                if int(adjacency[left]) & (1 << right):
                    data[begin + 1 + bit_index // 6] += 1 << (
                        5 - bit_index % 6
                    )
                bit_index += 1
    return EncodedGraphBatch(
        records=_encoded_records(data, graph_count, record_size),
        graph_count=graph_count,
        vertex_count=vertex_count,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def encode_graph6(
    adjacency_masks: ArrayLike,
    *,
    threads: int = 1,
    backend: Backend = "auto",
) -> EncodedGraphBatch:
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    if threads < 0:
        raise ValueError("threads must be nonnegative")
    graphs = _prepare_graphs(adjacency_masks)
    if graphs.shape[1] > 62:
        raise ValueError("short-form graph6 supports 1-62 vertices")
    if backend in {"auto", "native"}:
        try:
            data, stats = graph6_encode_native(graphs, threads=threads)
        except NativeUnavailable:
            if backend == "native":
                raise
        else:
            edge_bits = graphs.shape[1] * (graphs.shape[1] - 1) // 2
            record_size = 1 + (edge_bits + 5) // 6
            return EncodedGraphBatch(
                records=_encoded_records(data, len(graphs), record_size),
                graph_count=int(stats.graph_count),
                vertex_count=int(stats.vertex_count),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    return _encode_graph6_reference(graphs)


def decode_graph6(
    records: str | bytes | Iterable[str | bytes],
    *,
    threads: int = 1,
    backend: Backend = "auto",
) -> DecodedGraphBatch:
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    if threads < 0:
        raise ValueError("threads must be nonnegative")
    data, offsets, vertex_count = _prepare_graph6(records)
    if backend in {"auto", "native"}:
        try:
            adjacency_masks, stats = graph6_decode_native(
                data,
                offsets,
                vertex_count=vertex_count,
                threads=threads,
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return DecodedGraphBatch(
                adjacency_masks=adjacency_masks,
                graph_count=int(stats.graph_count),
                vertex_count=int(stats.vertex_count),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    return _decode_graph6_reference(data, offsets, vertex_count)


def _delete_graph_vertices_reference(
    adjacency_masks: NDArray[np.uint64],
    source_graphs: NDArray[np.uint64],
    deleted_vertices: NDArray[np.uint32],
) -> VertexDeletedGraphBatch:
    _validate_graphs(adjacency_masks)
    started = time.perf_counter()
    vertex_count = adjacency_masks.shape[1]
    output = np.empty(
        (len(source_graphs), vertex_count - 1),
        dtype=np.uint64,
    )
    for request, (source_graph, deleted_vertex) in enumerate(
        zip(source_graphs, deleted_vertices, strict=True)
    ):
        source = adjacency_masks[int(source_graph)]
        deleted = int(deleted_vertex)
        low_mask = (1 << deleted) - 1
        for old_vertex in range(vertex_count):
            if old_vertex == deleted:
                continue
            new_vertex = old_vertex - int(old_vertex > deleted)
            row = int(source[old_vertex])
            output[request, new_vertex] = (
                (row & low_mask) |
                ((row >> (deleted + 1)) << deleted)
            )
    return VertexDeletedGraphBatch(
        adjacency_masks=output,
        source_graphs=source_graphs,
        deleted_vertices=deleted_vertices,
        graph_count=len(output),
        vertex_count=vertex_count - 1,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def delete_graph_vertices(
    adjacency_masks: ArrayLike,
    source_graphs: ArrayLike,
    deleted_vertices: ArrayLike,
    *,
    threads: int = 1,
    backend: Backend = "auto",
) -> VertexDeletedGraphBatch:
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    if threads < 0:
        raise ValueError("threads must be nonnegative")
    graphs = _prepare_graphs(adjacency_masks)
    if graphs.shape[1] < 2:
        raise ValueError("vertex deletion requires graphs of order 2-64")
    sources, deleted = _prepare_deletion_requests(
        source_graphs,
        deleted_vertices,
        graph_count=len(graphs),
        vertex_count=graphs.shape[1],
    )
    if backend in {"auto", "native"}:
        try:
            output, stats = graph_delete_vertices_native(
                graphs,
                sources,
                deleted,
                threads=threads,
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return VertexDeletedGraphBatch(
                adjacency_masks=output,
                source_graphs=sources,
                deleted_vertices=deleted,
                graph_count=int(stats.graph_count),
                vertex_count=int(stats.vertex_count),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    return _delete_graph_vertices_reference(graphs, sources, deleted)


def _rooted_leaf_features_reference(
    adjacency_masks: NDArray[np.uint64],
    source_graphs: NDArray[np.uint64],
    leaf_vertices: NDArray[np.uint32],
) -> RootedLeafFeatureBatch:
    _validate_graphs(adjacency_masks)
    started = time.perf_counter()
    root_degrees = np.empty(len(source_graphs), dtype=np.uint32)
    two_star_counts = np.empty(len(source_graphs), dtype=np.uint32)
    two_step_counts = np.empty(len(source_graphs), dtype=np.uint32)
    for request, (source_graph, leaf_vertex) in enumerate(
        zip(source_graphs, leaf_vertices, strict=True)
    ):
        adjacency = adjacency_masks[int(source_graph)]
        leaf = int(leaf_vertex)
        leaf_neighbors = int(adjacency[leaf])
        if leaf_neighbors.bit_count() != 1:
            raise ValueError(
                "rooted-leaf request vertex must have degree one"
            )
        root = leaf_neighbors.bit_length() - 1
        other_neighbors = int(adjacency[root]) & ~(1 << leaf)
        degree = other_neighbors.bit_count()
        two_step = 0
        while other_neighbors:
            bit = other_neighbors & -other_neighbors
            neighbor = bit.bit_length() - 1
            two_step += int(adjacency[neighbor]).bit_count() - 1
            other_neighbors ^= bit
        root_degrees[request] = degree
        two_star_counts[request] = degree * (degree - 1) // 2
        two_step_counts[request] = two_step
    return RootedLeafFeatureBatch(
        root_degrees=root_degrees,
        two_star_counts=two_star_counts,
        two_step_counts=two_step_counts,
        source_graphs=source_graphs,
        leaf_vertices=leaf_vertices,
        request_count=len(source_graphs),
        vertex_count=adjacency_masks.shape[1],
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def rooted_leaf_attachment_features(
    adjacency_masks: ArrayLike,
    source_graphs: ArrayLike,
    leaf_vertices: ArrayLike,
    *,
    threads: int = 1,
    backend: Backend = "auto",
) -> RootedLeafFeatureBatch:
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    if threads < 0:
        raise ValueError("threads must be nonnegative")
    graphs = _prepare_graphs(adjacency_masks)
    sources, leaves = _prepare_deletion_requests(
        source_graphs,
        leaf_vertices,
        graph_count=len(graphs),
        vertex_count=graphs.shape[1],
    )
    if backend in {"auto", "native"}:
        try:
            degree, two_star, two_step, stats = (
                graph_rooted_leaf_features_native(
                    graphs,
                    sources,
                    leaves,
                    threads=threads,
                )
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return RootedLeafFeatureBatch(
                root_degrees=degree,
                two_star_counts=two_star,
                two_step_counts=two_step,
                source_graphs=sources,
                leaf_vertices=leaves,
                request_count=int(stats.graph_count),
                vertex_count=int(stats.vertex_count),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    return _rooted_leaf_features_reference(graphs, sources, leaves)


def _pair_profiles_reference(
    adjacency_masks: NDArray[np.uint64],
) -> GraphPairProfiles:
    _validate_graphs(adjacency_masks)
    started = time.perf_counter()
    graph_count, vertex_count = adjacency_masks.shape
    left, right = _pair_vertices(vertex_count)
    shape = (graph_count, len(left))
    adjacent = np.empty(shape, dtype=np.bool_)
    common_neighbors = np.empty(shape, dtype=np.uint32)
    common_nonneighbors = np.empty(shape, dtype=np.uint32)
    only_left = np.empty(shape, dtype=np.uint32)
    only_right = np.empty(shape, dtype=np.uint32)
    all_vertices = (
        (1 << vertex_count) - 1
        if vertex_count < 64
        else (1 << 64) - 1
    )
    for graph, adjacency_masks_for_graph in enumerate(adjacency_masks):
        for pair, (left_vertex, right_vertex) in enumerate(
            zip(left, right, strict=True)
        ):
            left_int = int(left_vertex)
            right_int = int(right_vertex)
            external = all_vertices & ~(
                (1 << left_int) | (1 << right_int)
            )
            left_neighbors = (
                int(adjacency_masks_for_graph[left_int]) & external
            )
            right_neighbors = (
                int(adjacency_masks_for_graph[right_int]) & external
            )
            adjacent[graph, pair] = bool(
                int(adjacency_masks_for_graph[left_int])
                & (1 << right_int)
            )
            common_neighbors[graph, pair] = (
                left_neighbors & right_neighbors
            ).bit_count()
            common_nonneighbors[graph, pair] = (
                external & ~left_neighbors & ~right_neighbors
            ).bit_count()
            only_left[graph, pair] = (
                left_neighbors & ~right_neighbors
            ).bit_count()
            only_right[graph, pair] = (
                right_neighbors & ~left_neighbors
            ).bit_count()
    return GraphPairProfiles(
        left=left,
        right=right,
        adjacent=adjacent,
        common_neighbors=common_neighbors,
        common_nonneighbors=common_nonneighbors,
        only_left=only_left,
        only_right=only_right,
        graph_count=graph_count,
        vertex_count=vertex_count,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def graph_pair_profiles(
    adjacency_masks: ArrayLike,
    *,
    threads: int = 1,
    backend: Backend = "auto",
) -> GraphPairProfiles:
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    if threads < 0:
        raise ValueError("threads must be nonnegative")
    graphs = _prepare_graphs(adjacency_masks)
    left, right = _pair_vertices(graphs.shape[1])
    if backend in {"auto", "native"}:
        try:
            (
                adjacent,
                common_neighbors,
                common_nonneighbors,
                only_left,
                only_right,
                stats,
            ) = graph_pair_profiles_native(graphs, threads=threads)
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return GraphPairProfiles(
                left=left,
                right=right,
                adjacent=adjacent,
                common_neighbors=common_neighbors,
                common_nonneighbors=common_nonneighbors,
                only_left=only_left,
                only_right=only_right,
                graph_count=int(stats.graph_count),
                vertex_count=int(stats.vertex_count),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    return _pair_profiles_reference(graphs)


def _find_cliques_reference(
    adjacency_masks: NDArray[np.uint64],
    order: int,
    complement: bool,
) -> CliqueBatchResult:
    _validate_graphs(adjacency_masks)
    started = time.perf_counter()
    graph_count, vertex_count = adjacency_masks.shape
    all_vertices = (
        (1 << vertex_count) - 1
        if vertex_count < 64
        else (1 << 64) - 1
    )
    witnesses = np.zeros(graph_count, dtype=np.uint64)
    nodes_visited = np.zeros(graph_count, dtype=np.uint64)
    for graph, adjacency in enumerate(adjacency_masks):
        witness = 0
        visits = 0

        def search(candidates: int, remaining: int, chosen: int) -> bool:
            nonlocal witness, visits
            visits += 1
            if remaining == 0:
                witness = chosen
                return True
            while candidates.bit_count() >= remaining:
                bit = candidates & -candidates
                vertex = bit.bit_length() - 1
                candidates ^= bit
                neighbors = int(adjacency[vertex])
                if complement:
                    neighbors = all_vertices & ~neighbors & ~bit
                if search(
                    candidates & neighbors,
                    remaining - 1,
                    chosen | bit,
                ):
                    return True
            return False

        search(all_vertices, order, 0)
        witnesses[graph] = witness
        nodes_visited[graph] = visits
    return CliqueBatchResult(
        witness_masks=witnesses,
        nodes_visited=nodes_visited,
        graph_count=graph_count,
        vertex_count=vertex_count,
        order=order,
        complement=complement,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def find_cliques(
    adjacency_masks: ArrayLike,
    order: int,
    *,
    complement: bool = False,
    threads: int = 1,
    backend: Backend = "auto",
) -> CliqueBatchResult:
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    if threads < 0:
        raise ValueError("threads must be nonnegative")
    graphs = _prepare_graphs(adjacency_masks)
    if not 1 <= order <= graphs.shape[1]:
        raise ValueError("clique order is outside the graph")
    if backend in {"auto", "native"}:
        try:
            witnesses, nodes_visited, stats = graph_find_clique_native(
                graphs,
                order=order,
                complement=complement,
                threads=threads,
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return CliqueBatchResult(
                witness_masks=witnesses,
                nodes_visited=nodes_visited,
                graph_count=int(stats.graph_count),
                vertex_count=int(stats.vertex_count),
                order=order,
                complement=complement,
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    return _find_cliques_reference(graphs, order, complement)


def find_independent_sets(
    adjacency_masks: ArrayLike,
    order: int,
    *,
    threads: int = 1,
    backend: Backend = "auto",
) -> CliqueBatchResult:
    return find_cliques(
        adjacency_masks,
        order,
        complement=True,
        threads=threads,
        backend=backend,
    )


def _graph_invariants_reference(
    adjacency_masks: NDArray[np.uint64],
) -> GraphInvariantBatch:
    _validate_graphs(adjacency_masks)
    started = time.perf_counter()
    graph_count, vertex_count = adjacency_masks.shape
    degrees = np.empty((graph_count, vertex_count), dtype=np.uint32)
    edge_counts = np.empty(graph_count, dtype=np.uint64)
    triangle_counts = np.empty(graph_count, dtype=np.uint64)
    wedge_counts = np.empty(graph_count, dtype=np.uint64)
    induced_path3_counts = np.empty(graph_count, dtype=np.uint64)
    all_vertices = (
        (1 << vertex_count) - 1
        if vertex_count < 64
        else (1 << 64) - 1
    )
    for graph, adjacency in enumerate(adjacency_masks):
        graph_degrees = np.fromiter(
            (int(mask).bit_count() for mask in adjacency),
            dtype=np.uint32,
            count=vertex_count,
        )
        degrees[graph] = graph_degrees
        edge_counts[graph] = int(np.sum(graph_degrees, dtype=np.uint64)) // 2
        wedges = sum(
            int(degree) * (int(degree) - 1) // 2
            for degree in graph_degrees
        )
        triangles = 0
        for left in range(vertex_count):
            later = int(adjacency[left]) & (
                all_vertices & ~((1 << (left + 1)) - 1)
            )
            while later:
                bit = later & -later
                right = bit.bit_length() - 1
                later ^= bit
                after_right = all_vertices & ~((1 << (right + 1)) - 1)
                triangles += (
                    int(adjacency[left])
                    & int(adjacency[right])
                    & after_right
                ).bit_count()
        triangle_counts[graph] = triangles
        wedge_counts[graph] = wedges
        induced_path3_counts[graph] = wedges - 3 * triangles
    return GraphInvariantBatch(
        degrees=degrees,
        edge_counts=edge_counts,
        triangle_counts=triangle_counts,
        wedge_counts=wedge_counts,
        induced_path3_counts=induced_path3_counts,
        graph_count=graph_count,
        vertex_count=vertex_count,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def graph_invariants(
    adjacency_masks: ArrayLike,
    *,
    threads: int = 1,
    backend: Backend = "auto",
) -> GraphInvariantBatch:
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    if threads < 0:
        raise ValueError("threads must be nonnegative")
    graphs = _prepare_graphs(adjacency_masks)
    if backend in {"auto", "native"}:
        try:
            (
                degrees,
                edge_counts,
                triangle_counts,
                wedge_counts,
                induced_path3_counts,
                stats,
            ) = graph_invariants_native(graphs, threads=threads)
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return GraphInvariantBatch(
                degrees=degrees,
                edge_counts=edge_counts,
                triangle_counts=triangle_counts,
                wedge_counts=wedge_counts,
                induced_path3_counts=induced_path3_counts,
                graph_count=int(stats.graph_count),
                vertex_count=int(stats.vertex_count),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    return _graph_invariants_reference(graphs)


def _prepare_class_lookup(
    class_lookup: ArrayLike,
    induced_order: int,
) -> tuple[NDArray[np.uint32], int]:
    lookup = np.asarray(class_lookup)
    expected_size = 1 << (induced_order * (induced_order - 1) // 2)
    if (
        lookup.ndim != 1
        or len(lookup) != expected_size
        or lookup.dtype.kind not in {"i", "u"}
    ):
        raise ValueError(
            "class_lookup must contain one integer class per labeled mask"
        )
    if lookup.dtype.kind == "i" and np.any(lookup < 0):
        raise ValueError("class_lookup must be nonnegative")
    if np.any(lookup >= np.iinfo(np.uint32).max):
        raise ValueError("class_lookup class IDs are too large")
    lookup = np.ascontiguousarray(lookup, dtype=np.uint32)
    classes = np.unique(lookup)
    class_count = int(classes[-1]) + 1
    if class_count > len(lookup):
        raise ValueError("class_lookup IDs must be dense from zero")
    if not np.array_equal(
        classes,
        np.arange(class_count, dtype=np.uint32),
    ):
        raise ValueError("class_lookup IDs must be dense from zero")
    return lookup, class_count


def _induced_profiles_reference(
    adjacency_masks: NDArray[np.uint64],
    induced_order: int,
    class_lookup: NDArray[np.uint32],
    class_count: int,
) -> InducedProfileBatch:
    _validate_graphs(adjacency_masks)
    started = time.perf_counter()
    graph_count, vertex_count = adjacency_masks.shape
    counts = np.zeros((graph_count, class_count), dtype=np.uint64)
    for graph, adjacency in enumerate(adjacency_masks):
        for vertices in combinations(range(vertex_count), induced_order):
            raw_mask = 0
            edge = 0
            for left_index, left in enumerate(vertices):
                for right in vertices[left_index + 1 :]:
                    if int(adjacency[left]) & (1 << right):
                        raw_mask |= 1 << edge
                    edge += 1
            counts[graph, int(class_lookup[raw_mask])] += 1
    return InducedProfileBatch(
        counts=counts,
        graph_count=graph_count,
        vertex_count=vertex_count,
        induced_order=induced_order,
        class_count=class_count,
        subsets_per_graph=comb(vertex_count, induced_order),
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def induced_subgraph_profiles(
    adjacency_masks: ArrayLike,
    induced_order: int,
    class_lookup: ArrayLike,
    *,
    threads: int = 1,
    backend: Backend = "auto",
) -> InducedProfileBatch:
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    if threads < 0:
        raise ValueError("threads must be nonnegative")
    graphs = _prepare_graphs(adjacency_masks)
    if not 1 <= induced_order <= min(graphs.shape[1], 7):
        raise ValueError(
            "induced_order must be between 1 and min(vertices, 7)"
        )
    lookup, class_count = _prepare_class_lookup(
        class_lookup,
        induced_order,
    )
    if backend in {"auto", "native"}:
        try:
            counts, stats = graph_induced_profiles_native(
                graphs,
                induced_order=induced_order,
                class_lookup=lookup,
                class_count=class_count,
                threads=threads,
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return InducedProfileBatch(
                counts=counts,
                graph_count=int(stats.graph_count),
                vertex_count=int(stats.vertex_count),
                induced_order=int(stats.induced_order),
                class_count=int(stats.class_count),
                subsets_per_graph=int(stats.subsets_per_graph),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    return _induced_profiles_reference(
        graphs,
        induced_order,
        lookup,
        class_count,
    )


def induced_subgraph_profile_stack(
    adjacency_masks: ArrayLike,
    induced_orders: Iterable[int],
    class_lookups: Iterable[ArrayLike],
    *,
    threads: int = 1,
    backend: Backend = "auto",
) -> InducedProfileStackBatch:
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    if threads < 0:
        raise ValueError("threads must be nonnegative")
    graphs = _prepare_graphs(adjacency_masks)
    try:
        orders = tuple(index(order) for order in induced_orders)
    except TypeError as error:
        raise ValueError("induced_orders must contain integers") from error
    lookup_values = tuple(class_lookups)
    if not orders or len(orders) != len(lookup_values):
        raise ValueError(
            "induced_orders and class_lookups must have equal nonzero length"
        )
    if tuple(sorted(set(orders))) != orders:
        raise ValueError("induced_orders must be strictly increasing")

    prepared_lookups = []
    class_count_values = []
    subset_counts = []
    for order, lookup_value in zip(orders, lookup_values, strict=True):
        if not 1 <= order <= min(graphs.shape[1], 7):
            raise ValueError(
                "induced orders must be between 1 and min(vertices, 7)"
            )
        lookup, class_count = _prepare_class_lookup(lookup_value, order)
        prepared_lookups.append(lookup)
        class_count_values.append(class_count)
        subset_counts.append(comb(graphs.shape[1], order))

    class_counts_array = np.asarray(
        class_count_values,
        dtype=np.uint32,
    )
    offsets = np.empty(len(orders) + 1, dtype=np.uint64)
    offsets[0] = 0
    np.cumsum(class_counts_array, dtype=np.uint64, out=offsets[1:])
    induced_orders_array = np.asarray(orders, dtype=np.uint32)

    lookup_offsets = np.empty(len(orders) + 1, dtype=np.uint64)
    lookup_offsets[0] = 0
    lookup_sizes = np.asarray(
        [len(lookup) for lookup in prepared_lookups],
        dtype=np.uint64,
    )
    np.cumsum(lookup_sizes, dtype=np.uint64, out=lookup_offsets[1:])
    joined_lookups = np.concatenate(prepared_lookups)

    if backend in {"auto", "native"}:
        try:
            counts, stats = graph_induced_profile_stack_native(
                graphs,
                induced_orders_array,
                joined_lookups,
                lookup_offsets,
                class_counts_array,
                threads=threads,
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return InducedProfileStackBatch(
                counts=counts,
                graph_count=int(stats.graph_count),
                vertex_count=int(stats.vertex_count),
                induced_orders=orders,
                class_counts=tuple(class_count_values),
                offsets=offsets,
                subsets_per_graph=tuple(subset_counts),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )

    started = time.perf_counter()
    parts = [
        _induced_profiles_reference(
            graphs,
            order,
            lookup,
            class_count,
        ).counts
        for order, lookup, class_count in zip(
            orders,
            prepared_lookups,
            class_count_values,
            strict=True,
        )
    ]
    return InducedProfileStackBatch(
        counts=np.concatenate(parts, axis=1),
        graph_count=len(graphs),
        vertex_count=graphs.shape[1],
        induced_orders=orders,
        class_counts=tuple(class_count_values),
        offsets=offsets,
        subsets_per_graph=tuple(subset_counts),
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )
