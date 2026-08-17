#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from fast_math import (  # noqa: E402
    decode_graph6,
    delete_graph_vertices,
    find_cliques,
    graph_invariants,
    graph_pair_profiles,
    induced_subgraph_profiles,
)


def random_graphs(
    graph_count: int,
    vertex_count: int,
    seed: int,
    density: float = 0.5,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    graphs = np.zeros((graph_count, vertex_count), dtype=np.uint64)
    for left in range(vertex_count):
        for right in range(left + 1, vertex_count):
            if density == 0.5:
                present = rng.integers(
                    0,
                    2,
                    size=graph_count,
                    dtype=np.uint64,
                )
            else:
                present = (
                    rng.random(graph_count) < density
                ).astype(np.uint64)
            graphs[:, left] |= present << np.uint64(right)
            graphs[:, right] |= present << np.uint64(left)
    return graphs


def invariant_bytes_per_graph(vertex_count: int) -> int:
    input_bytes = vertex_count * np.dtype(np.uint64).itemsize
    output_bytes = (
        vertex_count * np.dtype(np.uint32).itemsize
        + 4 * np.dtype(np.uint64).itemsize
    )
    return input_bytes + output_bytes


def timed(function):
    started = time.perf_counter()
    result = function()
    return result, time.perf_counter() - started


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


def edge_count_lookup(order: int) -> np.ndarray:
    edge_count = order * (order - 1) // 2
    return np.fromiter(
        (mask.bit_count() for mask in range(1 << edge_count)),
        dtype=np.uint32,
        count=1 << edge_count,
    )


def turan_graphs(
    graph_count: int,
    vertex_count: int,
    part_count: int,
) -> np.ndarray:
    graph = np.zeros(vertex_count, dtype=np.uint64)
    parts = np.arange(vertex_count) % part_count
    for left in range(vertex_count):
        for right in range(left + 1, vertex_count):
            if parts[left] == parts[right]:
                continue
            graph[left] |= np.uint64(1 << right)
            graph[right] |= np.uint64(1 << left)
    return np.repeat(graph[np.newaxis, :], graph_count, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=int, default=2_000)
    parser.add_argument("--vertices", type=int, default=43)
    parser.add_argument("--profile-graphs", type=int, default=20_000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    graphs = random_graphs(args.graphs, args.vertices, 7601)
    records = []

    pair_results = {}
    for backend in ("reference", "native"):
        result, wall = timed(
            lambda backend=backend: graph_pair_profiles(
                graphs,
                threads=args.threads,
                backend=backend,
            )
        )
        pair_results[backend] = result
        records.append(
            {
                "benchmark": "graph64_pair_profiles",
                "backend": backend,
                "graph_count": args.graphs,
                "vertex_count": args.vertices,
                "pair_count_per_graph": len(result.left),
                "total_pair_count": args.graphs * len(result.left),
                "threads": args.threads,
                "wall_seconds": wall,
                "kernel_seconds": result.elapsed_seconds,
                "pairs_per_second": args.graphs * len(result.left) / wall,
            }
        )
    np.testing.assert_array_equal(
        pair_results["native"].common_neighbors,
        pair_results["reference"].common_neighbors,
    )
    np.testing.assert_array_equal(
        pair_results["native"].common_nonneighbors,
        pair_results["reference"].common_nonneighbors,
    )

    invariant_results = {}
    invariant_bytes = invariant_bytes_per_graph(args.vertices)
    for backend in ("reference", "native"):
        result, wall = timed(
            lambda backend=backend: graph_invariants(
                graphs,
                threads=args.threads,
                backend=backend,
            )
        )
        invariant_results[backend] = result
        records.append(
            {
                "benchmark": "graph64_invariants",
                "backend": backend,
                "graph_count": args.graphs,
                "vertex_count": args.vertices,
                "threads": args.threads,
                "wall_seconds": wall,
                "kernel_seconds": result.elapsed_seconds,
                "graphs_per_second": args.graphs / wall,
                "bytes_per_graph": invariant_bytes,
                "gigabytes_per_second": (
                    args.graphs * invariant_bytes / wall / 1e9
                ),
                "density": 0.5,
                "edge_checksum": int(np.sum(result.edge_counts)),
                "triangle_checksum": int(
                    np.sum(result.triangle_counts)
                ),
            }
        )
    np.testing.assert_array_equal(
        invariant_results["native"].degrees,
        invariant_results["reference"].degrees,
    )
    np.testing.assert_array_equal(
        invariant_results["native"].triangle_counts,
        invariant_results["reference"].triangle_counts,
    )

    deletion_sources = np.repeat(
        np.arange(args.graphs, dtype=np.uint64), args.vertices
    )
    deleted_vertices = np.tile(
        np.arange(args.vertices, dtype=np.uint32), args.graphs
    )
    deletion_results = {}
    for backend in ("reference", "native"):
        result, wall = timed(
            lambda backend=backend: delete_graph_vertices(
                graphs,
                deletion_sources,
                deleted_vertices,
                threads=args.threads,
                backend=backend,
            )
        )
        deletion_results[backend] = result
        records.append(
            {
                "benchmark": "graph64_vertex_deletion",
                "backend": backend,
                "graph_count": args.graphs,
                "vertex_count": args.vertices,
                "request_count": len(deletion_sources),
                "threads": args.threads,
                "wall_seconds": wall,
                "kernel_seconds": result.elapsed_seconds,
                "requests_per_second": len(deletion_sources) / wall,
                "output_checksum": int(
                    np.bitwise_xor.reduce(
                        result.adjacency_masks.reshape(-1),
                        dtype=np.uint64,
                    )
                ),
            }
        )
    np.testing.assert_array_equal(
        deletion_results["native"].adjacency_masks,
        deletion_results["reference"].adjacency_masks,
    )

    for density, fixture, seed in (
        (0.1, "erdos_renyi_sparse", 7610),
        (0.9, "erdos_renyi_dense", 7611),
    ):
        density_graphs = random_graphs(
            args.graphs,
            args.vertices,
            seed,
            density,
        )
        result, wall = timed(
            lambda: graph_invariants(
                density_graphs,
                threads=args.threads,
                backend="native",
            )
        )
        records.append(
            {
                "benchmark": "graph64_invariants",
                "backend": "native",
                "graph_count": args.graphs,
                "vertex_count": args.vertices,
                "threads": args.threads,
                "wall_seconds": wall,
                "kernel_seconds": result.elapsed_seconds,
                "graphs_per_second": args.graphs / wall,
                "bytes_per_graph": invariant_bytes,
                "gigabytes_per_second": (
                    args.graphs * invariant_bytes / wall / 1e9
                ),
                "density": density,
                "fixture": fixture,
                "edge_checksum": int(np.sum(result.edge_counts)),
                "triangle_checksum": int(
                    np.sum(result.triangle_counts)
                ),
            }
        )

    clique_results = {}
    for backend in ("reference", "native"):
        result, wall = timed(
            lambda backend=backend: find_cliques(
                graphs,
                5,
                threads=args.threads,
                backend=backend,
            )
        )
        clique_results[backend] = result
        records.append(
            {
                "benchmark": "graph64_find_clique",
                "backend": backend,
                "graph_count": args.graphs,
                "vertex_count": args.vertices,
                "order": 5,
                "threads": args.threads,
                "wall_seconds": wall,
                "kernel_seconds": result.elapsed_seconds,
                "graphs_per_second": args.graphs / wall,
                "found_count": int(np.count_nonzero(result.found)),
                "nodes_visited": int(np.sum(result.nodes_visited)),
            }
        )
    np.testing.assert_array_equal(
        clique_results["native"].found,
        clique_results["reference"].found,
    )

    negative_graphs = turan_graphs(args.graphs, args.vertices, 4)
    negative_results = {}
    for backend in ("reference", "native"):
        result, wall = timed(
            lambda backend=backend: find_cliques(
                negative_graphs,
                5,
                threads=args.threads,
                backend=backend,
            )
        )
        negative_results[backend] = result
        records.append(
            {
                "benchmark": "graph64_find_clique_negative",
                "backend": backend,
                "graph_count": args.graphs,
                "vertex_count": args.vertices,
                "order": 5,
                "threads": args.threads,
                "wall_seconds": wall,
                "kernel_seconds": result.elapsed_seconds,
                "graphs_per_second": args.graphs / wall,
                "found_count": int(np.count_nonzero(result.found)),
                "nodes_visited": int(np.sum(result.nodes_visited)),
                "fixture": "balanced_complete_4_partite",
            }
        )
    np.testing.assert_array_equal(
        negative_results["native"].found,
        negative_results["reference"].found,
    )
    if np.any(negative_results["native"].found):
        raise AssertionError("the K5-free fixture produced a K5 witness")

    profile_graphs = random_graphs(args.profile_graphs, 9, 7602)
    graph6_records = [encode_graph6(graph) for graph in profile_graphs]
    decode_results = {}
    for backend in ("reference", "native"):
        result, wall = timed(
            lambda backend=backend: decode_graph6(
                graph6_records,
                threads=args.threads,
                backend=backend,
            )
        )
        decode_results[backend] = result
        records.append(
            {
                "benchmark": "graph6_decode_order9",
                "backend": backend,
                "graph_count": args.profile_graphs,
                "vertex_count": 9,
                "threads": args.threads,
                "wall_seconds": wall,
                "kernel_seconds": result.elapsed_seconds,
                "graphs_per_second": args.profile_graphs / wall,
            }
        )
    np.testing.assert_array_equal(
        decode_results["native"].adjacency_masks,
        decode_results["reference"].adjacency_masks,
    )
    np.testing.assert_array_equal(
        decode_results["native"].adjacency_masks,
        profile_graphs,
    )

    lookup = edge_count_lookup(6)
    profile_results = {}
    for backend in ("reference", "native"):
        result, wall = timed(
            lambda backend=backend: induced_subgraph_profiles(
                profile_graphs,
                6,
                lookup,
                threads=args.threads,
                backend=backend,
            )
        )
        profile_results[backend] = result
        records.append(
            {
                "benchmark": "graph64_induced_profile_order9_choose6",
                "backend": backend,
                "graph_count": args.profile_graphs,
                "vertex_count": 9,
                "induced_order": 6,
                "class_count": result.class_count,
                "subsets_per_graph": result.subsets_per_graph,
                "total_subsets": (
                    args.profile_graphs * result.subsets_per_graph
                ),
                "threads": args.threads,
                "wall_seconds": wall,
                "kernel_seconds": result.elapsed_seconds,
                "subsets_per_second": (
                    args.profile_graphs * result.subsets_per_graph / wall
                ),
                "count_checksum": int(np.sum(result.counts)),
                "fixture": "dense_class_lookup_by_induced_edge_count",
            }
        )
    np.testing.assert_array_equal(
        profile_results["native"].counts,
        profile_results["reference"].counts,
    )

    for record in records:
        print(json.dumps(record, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
