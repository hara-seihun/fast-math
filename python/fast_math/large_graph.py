from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import time
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._native import (
    NativeUnavailable,
    graph_common_neighbors_csr_native,
    graph_triangles_csr_native,
)


Backend = Literal["auto", "native", "reference"]


@dataclass(frozen=True)
class UndirectedCSR:
    row_offsets: NDArray[np.uint64]
    column_indices: NDArray[np.uint32]
    edge_color_masks: NDArray[np.uint64] | None = None
    vertex_loop_color_masks: NDArray[np.uint64] | None = None

    @property
    def vertex_count(self) -> int:
        return len(self.row_offsets) - 1


@dataclass(frozen=True)
class TriangleBatch:
    triangles: NDArray[np.uint32]
    edge_color_masks: NDArray[np.uint64] | None
    vertex_count: int
    directed_edge_count: int
    intersection_steps: int
    elapsed_seconds: float
    backend: str


@dataclass(frozen=True)
class CommonNeighborBatch:
    pair_offsets: NDArray[np.uint64]
    common_neighbors: NDArray[np.uint32] | None
    vertex_count: int
    directed_edge_count: int
    pair_count: int
    intersection_steps: int
    common_neighbor_count: int
    elapsed_seconds: float
    backend: str

    @property
    def counts(self) -> NDArray[np.uint64]:
        return np.diff(self.pair_offsets)


def undirected_csr(
    vertex_count: int,
    edges: ArrayLike,
    *,
    edge_color_masks: ArrayLike | None = None,
) -> UndirectedCSR:
    """Build deterministic symmetric CSR from one row per undirected edge."""
    if (
        not isinstance(vertex_count, Integral)
        or isinstance(vertex_count, bool)
        or not 0 <= vertex_count <= np.iinfo(np.uint32).max
    ):
        raise ValueError("vertex_count must fit uint32")
    edge_array = np.asarray(edges)
    if edge_array.size == 0:
        edge_array = np.empty((0, 2), dtype=np.uint32)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("edges must have shape (edge_count, 2)")
    if not np.issubdtype(edge_array.dtype, np.integer):
        raise ValueError("edges must contain integers")
    if np.issubdtype(edge_array.dtype, np.signedinteger) and np.any(
        edge_array < 0
    ):
        raise ValueError("edge endpoints must be nonnegative")
    edge_array_u64 = np.asarray(edge_array, dtype=np.uint64)
    if np.any(edge_array_u64 >= vertex_count):
        raise ValueError("edge endpoint is outside vertex_count")
    colors = None
    if edge_color_masks is not None:
        raw_colors = np.asarray(edge_color_masks)
        if raw_colors.ndim != 1 or len(raw_colors) != len(edge_array_u64):
            raise ValueError(
                "edge_color_masks must have one entry per edge"
            )
        if not np.issubdtype(raw_colors.dtype, np.integer):
            raise ValueError("edge_color_masks must contain integers")
        if np.issubdtype(raw_colors.dtype, np.signedinteger) and np.any(
            raw_colors < 0
        ):
            raise ValueError("edge_color_masks must be nonnegative")
        colors = np.asarray(raw_colors, dtype=np.uint64)

    canonical_edges = np.sort(edge_array_u64, axis=1)
    order = np.lexsort(
        (canonical_edges[:, 1], canonical_edges[:, 0])
    )
    canonical_edges = canonical_edges[order]
    if colors is not None:
        colors = colors[order]
    if len(canonical_edges) > 1 and np.any(
        np.all(canonical_edges[1:] == canonical_edges[:-1], axis=1)
    ):
        raise ValueError("duplicate undirected edge")

    loop_entries = canonical_edges[:, 0] == canonical_edges[:, 1]
    loop_masks = np.zeros(vertex_count, dtype=np.uint64)
    for edge_index in np.flatnonzero(loop_entries):
        vertex = int(canonical_edges[edge_index, 0])
        loop_masks[vertex] = (
            colors[edge_index] if colors is not None else 1
        )
    nonloop_edges = canonical_edges[~loop_entries]
    nonloop_colors = colors[~loop_entries] if colors is not None else None
    degree = np.zeros(vertex_count, dtype=np.uint64)
    if len(nonloop_edges):
        np.add.at(degree, nonloop_edges[:, 0], 1)
        np.add.at(degree, nonloop_edges[:, 1], 1)
    row_offsets = np.empty(vertex_count + 1, dtype=np.uint64)
    row_offsets[0] = 0
    np.cumsum(degree, out=row_offsets[1:])
    columns = np.empty(2 * len(nonloop_edges), dtype=np.uint32)
    directed_colors = (
        np.empty(2 * len(nonloop_edges), dtype=np.uint64)
        if colors is not None
        else None
    )
    cursor = row_offsets[:-1].copy()
    for edge_index, (left, right) in enumerate(nonloop_edges):
        left_offset = int(cursor[left])
        right_offset = int(cursor[right])
        columns[left_offset] = right
        columns[right_offset] = left
        if directed_colors is not None:
            directed_colors[left_offset] = nonloop_colors[edge_index]
            directed_colors[right_offset] = nonloop_colors[edge_index]
        cursor[left] += 1
        cursor[right] += 1
    for vertex in range(vertex_count):
        begin = int(row_offsets[vertex])
        end = int(row_offsets[vertex + 1])
        local_order = np.argsort(columns[begin:end], kind="stable")
        columns[begin:end] = columns[begin:end][local_order]
        if directed_colors is not None:
            directed_colors[begin:end] = directed_colors[begin:end][
                local_order
            ]
    return UndirectedCSR(
        row_offsets,
        columns,
        directed_colors,
        loop_masks if np.any(loop_masks) else None,
    )


def _prepare_csr(
    row_offsets: ArrayLike,
    column_indices: ArrayLike,
    edge_color_masks: ArrayLike | None,
    vertex_loop_color_masks: ArrayLike | None,
) -> UndirectedCSR:
    offsets = np.asarray(row_offsets)
    columns = np.asarray(column_indices)
    if offsets.ndim != 1 or len(offsets) == 0:
        raise ValueError("row_offsets must be a nonempty vector")
    if columns.ndim != 1:
        raise ValueError("column_indices must be a vector")
    if not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError("row_offsets must contain integers")
    if not np.issubdtype(columns.dtype, np.integer):
        raise ValueError("column_indices must contain integers")
    if np.issubdtype(offsets.dtype, np.signedinteger) and np.any(offsets < 0):
        raise ValueError("row_offsets must be nonnegative")
    if np.issubdtype(columns.dtype, np.signedinteger) and np.any(columns < 0):
        raise ValueError("column_indices must be nonnegative")
    if columns.size and np.any(columns >= len(offsets) - 1):
        raise ValueError("column index is outside the CSR vertex range")
    prepared_offsets = np.ascontiguousarray(offsets, dtype=np.uint64)
    prepared_columns = np.ascontiguousarray(columns, dtype=np.uint32)
    colors = None
    if edge_color_masks is not None:
        raw_colors = np.asarray(edge_color_masks)
        if raw_colors.ndim != 1 or len(raw_colors) != len(prepared_columns):
            raise ValueError(
                "edge_color_masks must align with column_indices"
            )
        if not np.issubdtype(raw_colors.dtype, np.integer):
            raise ValueError("edge_color_masks must contain integers")
        if np.issubdtype(raw_colors.dtype, np.signedinteger) and np.any(
            raw_colors < 0
        ):
            raise ValueError("edge_color_masks must be nonnegative")
        colors = np.ascontiguousarray(raw_colors, dtype=np.uint64)
    loops = None
    if vertex_loop_color_masks is not None:
        raw_loops = np.asarray(vertex_loop_color_masks)
        if raw_loops.ndim != 1 or len(raw_loops) != len(offsets) - 1:
            raise ValueError(
                "vertex_loop_color_masks must have one entry per vertex"
            )
        if not np.issubdtype(raw_loops.dtype, np.integer):
            raise ValueError(
                "vertex_loop_color_masks must contain integers"
            )
        if np.issubdtype(raw_loops.dtype, np.signedinteger) and np.any(
            raw_loops < 0
        ):
            raise ValueError(
                "vertex_loop_color_masks must be nonnegative"
            )
        loops = np.ascontiguousarray(raw_loops, dtype=np.uint64)
    return UndirectedCSR(
        prepared_offsets,
        prepared_columns,
        colors,
        loops,
    )


def _prepare_pairs(
    pairs: ArrayLike,
    *,
    vertex_count: int,
) -> NDArray[np.uint32]:
    pair_array = np.asarray(pairs)
    if pair_array.size == 0:
        pair_array = np.empty((0, 2), dtype=np.uint32)
    if pair_array.ndim != 2 or pair_array.shape[1] != 2:
        raise ValueError("pairs must have shape (pair_count, 2)")
    if not np.issubdtype(pair_array.dtype, np.integer):
        raise ValueError("pairs must contain integers")
    if np.issubdtype(pair_array.dtype, np.signedinteger) and np.any(
        pair_array < 0
    ):
        raise ValueError("pair endpoints must be nonnegative")
    pair_array_u64 = np.asarray(pair_array, dtype=np.uint64)
    if np.any(pair_array_u64 >= vertex_count):
        raise ValueError("pair endpoint is outside the CSR vertex range")
    return np.ascontiguousarray(pair_array_u64, dtype=np.uint32)


def _reference_common_neighbors(
    graph: UndirectedCSR,
    pairs: NDArray[np.uint32],
    *,
    materialize: bool,
) -> CommonNeighborBatch:
    started = time.perf_counter()
    pair_offsets = np.empty(len(pairs) + 1, dtype=np.uint64)
    pair_offsets[0] = 0
    neighbors: list[int] | None = [] if materialize else None
    count = 0
    steps = 0
    for pair_index, (left, right) in enumerate(pairs):
        left_offset = int(graph.row_offsets[left])
        left_end = int(graph.row_offsets[int(left) + 1])
        right_offset = int(graph.row_offsets[right])
        right_end = int(graph.row_offsets[int(right) + 1])
        while left_offset < left_end and right_offset < right_end:
            steps += 1
            left_neighbor = int(graph.column_indices[left_offset])
            right_neighbor = int(graph.column_indices[right_offset])
            if left_neighbor < right_neighbor:
                left_offset += 1
            elif right_neighbor < left_neighbor:
                right_offset += 1
            else:
                if neighbors is not None:
                    neighbors.append(left_neighbor)
                count += 1
                left_offset += 1
                right_offset += 1
        pair_offsets[pair_index + 1] = count
    return CommonNeighborBatch(
        pair_offsets=pair_offsets,
        common_neighbors=(
            np.asarray(neighbors, dtype=np.uint32)
            if neighbors is not None
            else None
        ),
        vertex_count=graph.vertex_count,
        directed_edge_count=len(graph.column_indices),
        pair_count=len(pairs),
        intersection_steps=steps,
        common_neighbor_count=count,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def csr_common_neighbors(
    row_offsets: ArrayLike,
    column_indices: ArrayLike,
    pairs: ArrayLike,
    *,
    materialize: bool = False,
    backend: Backend = "auto",
) -> CommonNeighborBatch:
    """Count or materialize sorted common neighbors for explicit CSR pairs."""
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    if not isinstance(materialize, (bool, np.bool_)):
        raise TypeError("materialize must be boolean")
    graph = _prepare_csr(row_offsets, column_indices, None, None)
    prepared_pairs = _prepare_pairs(
        pairs,
        vertex_count=graph.vertex_count,
    )
    if backend in {"auto", "native"}:
        try:
            pair_offsets, common_neighbors, stats = (
                graph_common_neighbors_csr_native(
                    graph.row_offsets,
                    graph.column_indices,
                    prepared_pairs,
                    materialize=bool(materialize),
                )
            )
        except (NativeUnavailable, OSError, AttributeError):
            if backend == "native":
                raise
        else:
            return CommonNeighborBatch(
                pair_offsets=pair_offsets,
                common_neighbors=common_neighbors,
                vertex_count=int(stats.vertex_count),
                directed_edge_count=int(stats.directed_edge_count),
                pair_count=int(stats.pair_count),
                intersection_steps=int(stats.intersection_steps),
                common_neighbor_count=int(stats.common_neighbor_count),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    return _reference_common_neighbors(
        graph,
        prepared_pairs,
        materialize=bool(materialize),
    )


def _reference(graph: UndirectedCSR) -> TriangleBatch:
    started = time.perf_counter()
    offsets = graph.row_offsets
    columns = graph.column_indices
    colors = graph.edge_color_masks
    triangles: list[tuple[int, int, int]] = []
    triangle_colors: list[tuple[int, int, int]] = []
    steps = 0
    for left in range(graph.vertex_count):
        left_loop = (
            int(graph.vertex_loop_color_masks[left])
            if graph.vertex_loop_color_masks is not None
            else 0
        )
        if left_loop:
            triangles.append((left, left, left))
            if colors is not None:
                triangle_colors.append((left_loop, left_loop, left_loop))
        left_begin = int(offsets[left])
        left_end = int(offsets[left + 1])
        left_neighbors = columns[left_begin:left_end]
        first_middle = int(
            np.searchsorted(left_neighbors, left + 1, side="left")
        )
        if left_loop:
            for local_right in range(first_middle, len(left_neighbors)):
                right = int(left_neighbors[local_right])
                edge_color = (
                    int(colors[left_begin + local_right])
                    if colors is not None
                    else 0
                )
                triangles.append((left, left, right))
                if colors is not None:
                    triangle_colors.append(
                        (left_loop, edge_color, edge_color)
                    )
        for local_middle in range(first_middle, len(left_neighbors)):
            middle = int(left_neighbors[local_middle])
            edge_color = (
                int(colors[left_begin + local_middle])
                if colors is not None
                else 0
            )
            middle_loop = (
                int(graph.vertex_loop_color_masks[middle])
                if graph.vertex_loop_color_masks is not None
                else 0
            )
            if middle_loop:
                triangles.append((left, middle, middle))
                if colors is not None:
                    triangle_colors.append(
                        (edge_color, edge_color, middle_loop)
                    )
            middle_begin = int(offsets[middle])
            middle_end = int(offsets[middle + 1])
            middle_neighbors = columns[middle_begin:middle_end]
            left_index = local_middle + 1
            middle_index = int(
                np.searchsorted(middle_neighbors, middle + 1, side="left")
            )
            while (
                left_index < len(left_neighbors)
                and middle_index < len(middle_neighbors)
            ):
                steps += 1
                left_right = int(left_neighbors[left_index])
                middle_right = int(middle_neighbors[middle_index])
                if left_right < middle_right:
                    left_index += 1
                elif middle_right < left_right:
                    middle_index += 1
                else:
                    triangles.append((left, middle, left_right))
                    if colors is not None:
                        triangle_colors.append(
                            (
                                int(colors[left_begin + local_middle]),
                                int(colors[left_begin + left_index]),
                                int(colors[middle_begin + middle_index]),
                            )
                        )
                    left_index += 1
                    middle_index += 1
    return TriangleBatch(
        triangles=np.asarray(triangles, dtype=np.uint32).reshape((-1, 3)),
        edge_color_masks=(
            np.asarray(triangle_colors, dtype=np.uint64).reshape((-1, 3))
            if colors is not None
            else None
        ),
        vertex_count=graph.vertex_count,
        directed_edge_count=len(columns),
        intersection_steps=steps,
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def enumerate_csr_triangles(
    row_offsets: ArrayLike,
    column_indices: ArrayLike,
    *,
    edge_color_masks: ArrayLike | None = None,
    vertex_loop_color_masks: ArrayLike | None = None,
    backend: Backend = "auto",
) -> TriangleBatch:
    """Enumerate each undirected triangle once as ``left < middle < right``."""
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    graph = _prepare_csr(
        row_offsets,
        column_indices,
        edge_color_masks,
        vertex_loop_color_masks,
    )
    if backend in {"auto", "native"}:
        try:
            triangles, triangle_colors, stats = graph_triangles_csr_native(
                graph.row_offsets,
                graph.column_indices,
                graph.edge_color_masks,
                graph.vertex_loop_color_masks,
            )
        except (NativeUnavailable, OSError, AttributeError):
            if backend == "native":
                raise
        else:
            return TriangleBatch(
                triangles=triangles,
                edge_color_masks=triangle_colors,
                vertex_count=int(stats.vertex_count),
                directed_edge_count=int(stats.directed_edge_count),
                intersection_steps=int(stats.intersection_steps),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    return _reference(graph)
