from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from numbers import Integral
import time
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lambda_fast._native import (
    NativeUnavailable,
    canonical_digraphs_nauty_native,
)


Backend = Literal["auto", "native", "reference"]


@dataclass(frozen=True)
class CanonicalDigraphBatch:
    permutations: NDArray[np.uint32]
    adjacency_words: NDArray[np.uint64]
    vertex_colors: NDArray[np.uint32]
    class_ids: NDArray[np.uint32]
    automorphism_group_mantissas: NDArray[np.float64]
    automorphism_group_exponents: NDArray[np.int32]
    orbit_counts: NDArray[np.uint32]
    automorphism_generator_offsets: NDArray[np.uint64]
    automorphism_generators: NDArray[np.uint32]
    search_nodes: int
    elapsed_seconds: float
    backend: str


def pack_digraph_adjacency(adjacency: ArrayLike) -> NDArray[np.uint64]:
    """Pack one or more square Boolean adjacency matrices into uint64 rows."""
    array = np.asarray(adjacency)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError(
            "adjacency must have shape (graphs, vertices, vertices)"
        )
    if array.shape[1] == 0:
        raise ValueError("digraphs require at least one vertex")
    if np.any(np.diagonal(array, axis1=1, axis2=2)):
        raise ValueError("self-loops are not supported")
    graph_count, vertex_count, _ = array.shape
    words = np.zeros(
        (graph_count, vertex_count, (vertex_count + 63) // 64),
        dtype=np.uint64,
    )
    present = np.asarray(array, dtype=np.bool_)
    for neighbor in range(vertex_count):
        words[:, :, neighbor // 64] |= (
            present[:, :, neighbor].astype(np.uint64)
            << np.uint64(neighbor % 64)
        )
    return words


def _prepare(
    adjacency_words: ArrayLike,
    vertex_colors: ArrayLike,
) -> tuple[NDArray[np.uint64], NDArray[np.uint32]]:
    adjacency = np.asarray(adjacency_words)
    colors = np.asarray(vertex_colors)
    if adjacency.ndim != 3:
        raise ValueError(
            "adjacency_words must have shape (graphs, vertices, words)"
        )
    graph_count, vertex_count, word_count = adjacency.shape
    if vertex_count == 0 or word_count != (vertex_count + 63) // 64:
        raise ValueError("adjacency_words has an invalid packed shape")
    if not np.issubdtype(adjacency.dtype, np.integer):
        raise ValueError("adjacency_words must contain integers")
    if np.issubdtype(adjacency.dtype, np.signedinteger) and np.any(
        adjacency < 0
    ):
        raise ValueError("adjacency_words must be nonnegative")
    if colors.shape == (vertex_count,):
        colors = np.broadcast_to(colors, (graph_count, vertex_count))
    if colors.shape != (graph_count, vertex_count):
        raise ValueError(
            "vertex_colors must have shape (graphs, vertices)"
        )
    if not np.issubdtype(colors.dtype, np.integer):
        raise ValueError("vertex_colors must contain integers")
    if np.issubdtype(colors.dtype, np.signedinteger) and np.any(colors < 0):
        raise ValueError("vertex_colors must be nonnegative")
    if colors.size and np.any(colors > np.iinfo(np.uint32).max):
        raise ValueError("vertex_colors must fit uint32")
    return (
        np.ascontiguousarray(adjacency, dtype=np.uint64),
        np.ascontiguousarray(colors, dtype=np.uint32),
    )


def _relabel(
    adjacency: NDArray[np.uint64],
    lab: tuple[int, ...],
) -> NDArray[np.uint64]:
    vertex_count = len(lab)
    result = np.zeros_like(adjacency)
    for output_left, input_left in enumerate(lab):
        input_row = adjacency[input_left]
        for output_right, input_right in enumerate(lab):
            if int(input_row[input_right // 64]) & (
                1 << (input_right % 64)
            ):
                result[output_left, output_right // 64] |= np.uint64(
                    1 << (output_right % 64)
                )
    return result


def _class_ids(
    adjacency: NDArray[np.uint64],
    colors: NDArray[np.uint32],
) -> NDArray[np.uint32]:
    row_count = len(adjacency)
    if row_count == 0:
        return np.empty(0, dtype=np.uint32)
    adjacency_bytes = adjacency.reshape(row_count, -1).view(np.uint8)
    color_bytes = colors.reshape(row_count, -1).view(np.uint8)
    keys = np.concatenate((adjacency_bytes, color_bytes), axis=1)
    packed = keys.view(np.dtype((np.void, keys.shape[1]))).reshape(-1)
    _, first_indices, sorted_ids = np.unique(
        packed,
        return_index=True,
        return_inverse=True,
    )
    first_occurrence_order = np.argsort(first_indices)
    stable_ids = np.empty(len(first_occurrence_order), dtype=np.uint32)
    stable_ids[first_occurrence_order] = np.arange(
        len(first_occurrence_order), dtype=np.uint32
    )
    return stable_ids[sorted_ids]


def _reference(
    adjacency: NDArray[np.uint64],
    colors: NDArray[np.uint32],
    *,
    collect_automorphism_generators: bool,
) -> CanonicalDigraphBatch:
    vertex_count = adjacency.shape[1]
    if vertex_count > 9:
        raise ValueError(
            "reference canonicalization is limited to nine vertices"
        )
    started = time.perf_counter()
    canonical = np.empty_like(adjacency)
    canonical_colors = np.empty_like(colors)
    canonical_labs = np.empty(colors.shape, dtype=np.uint32)
    group_mantissas = np.empty(len(adjacency), dtype=np.float64)
    group_exponents = np.zeros(len(adjacency), dtype=np.int32)
    orbit_counts = np.empty(len(adjacency), dtype=np.uint32)
    generator_offsets = np.zeros(len(adjacency) + 1, dtype=np.uint64)
    all_generators: list[tuple[int, ...]] = []
    search_nodes = 0

    for graph_index, (graph, graph_colors) in enumerate(
        zip(adjacency, colors, strict=True)
    ):
        color_values = sorted(set(map(int, graph_colors)))
        groups = [
            tuple(
                vertex
                for vertex, color in enumerate(graph_colors)
                if int(color) == value
            )
            for value in color_values
        ]
        candidates = product(*(permutations(group) for group in groups))
        best_key = None
        best_graph = None
        best_lab = None
        base_lab = tuple(vertex for group in groups for vertex in group)
        base_graph = _relabel(graph, base_lab)
        automorphisms: list[tuple[int, ...]] = []
        for group_permutations in candidates:
            lab = tuple(
                vertex
                for group in group_permutations
                for vertex in group
            )
            relabeled = _relabel(graph, lab)
            key = relabeled.tobytes(order="C")
            if best_key is None or key < best_key:
                best_key = key
                best_graph = relabeled
                best_lab = lab
            if np.array_equal(relabeled, base_graph):
                mapping = list(range(vertex_count))
                for source, target in zip(base_lab, lab, strict=True):
                    mapping[source] = target
                automorphisms.append(tuple(mapping))
            search_nodes += 1
        assert best_graph is not None and best_lab is not None
        canonical[graph_index] = best_graph
        canonical_labs[graph_index] = best_lab
        canonical_colors[graph_index] = graph_colors[list(best_lab)]
        group_mantissas[graph_index] = len(automorphisms)
        nonidentity_generators = [
            mapping
            for mapping in automorphisms
            if any(
                source != target
                for source, target in enumerate(mapping)
            )
        ]
        if collect_automorphism_generators:
            all_generators.extend(nonidentity_generators)
        generator_offsets[graph_index + 1] = len(all_generators)
        parent = list(range(vertex_count))

        def find(vertex: int) -> int:
            while parent[vertex] != vertex:
                parent[vertex] = parent[parent[vertex]]
                vertex = parent[vertex]
            return vertex

        for mapping in automorphisms:
            for source, target in enumerate(mapping):
                left = find(source)
                right = find(target)
                if left != right:
                    parent[right] = left
        orbit_counts[graph_index] = len(
            {find(vertex) for vertex in range(vertex_count)}
        )

    return CanonicalDigraphBatch(
        permutations=canonical_labs,
        adjacency_words=canonical,
        vertex_colors=canonical_colors,
        class_ids=_class_ids(canonical, canonical_colors),
        automorphism_group_mantissas=group_mantissas,
        automorphism_group_exponents=group_exponents,
        orbit_counts=orbit_counts,
        automorphism_generator_offsets=generator_offsets,
        automorphism_generators=np.asarray(
            all_generators,
            dtype=np.uint32,
        ).reshape((-1, vertex_count)),
        search_nodes=search_nodes,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def canonicalize_colored_digraphs(
    adjacency_words: ArrayLike,
    vertex_colors: ArrayLike,
    *,
    threads: int = 0,
    collect_automorphism_generators: bool = True,
    backend: Backend = "auto",
) -> CanonicalDigraphBatch:
    """Canonicalize a batch of directed vertex-colored simple graphs."""
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    if not isinstance(threads, Integral) or int(threads) < 0:
        raise ValueError("threads must be a nonnegative integer")
    if not isinstance(collect_automorphism_generators, bool):
        raise ValueError(
            "collect_automorphism_generators must be a Boolean"
        )
    adjacency, colors = _prepare(adjacency_words, vertex_colors)
    if backend in {"auto", "native"}:
        try:
            (
                canonical_labs,
                canonical,
                canonical_colors,
                group_mantissas,
                group_exponents,
                orbit_counts,
                generator_offsets,
                generators,
                stats,
            ) = canonical_digraphs_nauty_native(
                adjacency,
                colors,
                threads=int(threads),
                collect_automorphism_generators=(
                    collect_automorphism_generators
                ),
            )
        except (NativeUnavailable, OSError, AttributeError):
            if backend == "native":
                raise
        else:
            return CanonicalDigraphBatch(
                permutations=canonical_labs,
                adjacency_words=canonical,
                vertex_colors=canonical_colors,
                class_ids=_class_ids(canonical, canonical_colors),
                automorphism_group_mantissas=group_mantissas,
                automorphism_group_exponents=group_exponents,
                orbit_counts=orbit_counts,
                automorphism_generator_offsets=generator_offsets,
                automorphism_generators=generators,
                search_nodes=int(stats.search_nodes),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    return _reference(
        adjacency,
        colors,
        collect_automorphism_generators=collect_automorphism_generators,
    )
