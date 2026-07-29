from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from fast_math import (
    canonicalize_colored_digraphs,
    enumerate_csr_triangles,
    undirected_csr,
)


def factor_graph_batch(
    families: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    graph_count = len(families)
    vertex_count = len(families[0])
    adjacency = np.zeros(
        (graph_count, vertex_count, 1),
        dtype=np.uint64,
    )
    colors = np.zeros((graph_count, vertex_count), dtype=np.uint32)
    for graph_index, family in enumerate(families):
        if len(family) != vertex_count:
            raise ValueError("factor families do not have one common size")
        indegrees = [0] * vertex_count
        outdegrees = [0] * vertex_count
        for lower_index, lower in enumerate(family):
            for upper_index, upper in enumerate(family):
                if lower_index == upper_index or lower & upper != lower:
                    continue
                if any(
                    middle_index not in (lower_index, upper_index)
                    and lower & middle == lower
                    and middle & upper == middle
                    for middle_index, middle in enumerate(family)
                ):
                    continue
                adjacency[graph_index, lower_index, 0] |= np.uint64(
                    1 << upper_index
                )
                outdegrees[lower_index] += 1
                indegrees[upper_index] += 1
        signatures = sorted(
            set(zip(indegrees, outdegrees, strict=True))
        )
        color_ids = {
            signature: index
            for index, signature in enumerate(signatures)
        }
        colors[graph_index] = [
            color_ids[signature]
            for signature in zip(indegrees, outdegrees, strict=True)
        ]
    return adjacency, colors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = json.loads(arguments.report.read_text())
    edges = np.asarray(report["edge_types"], dtype=np.uint32)
    edge_colors = np.ones(len(edges), dtype=np.uint64)
    build_started = time.perf_counter()
    graph = undirected_csr(
        report["factor_types"],
        edges,
        edge_color_masks=edge_colors,
    )
    csr_build_seconds = time.perf_counter() - build_started
    triangle_started = time.perf_counter()
    triangles = enumerate_csr_triangles(
        graph.row_offsets,
        graph.column_indices,
        edge_color_masks=graph.edge_color_masks,
        vertex_loop_color_masks=graph.vertex_loop_color_masks,
        backend="native",
    )
    triangle_wall_seconds = time.perf_counter() - triangle_started
    expected_triangles = np.asarray(
        report["allowed_type_triples"],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(
        triangles.triangles,
        expected_triangles,
    )
    logical_triangle_bytes = (
        triangles.intersection_steps * 2 * np.dtype(np.uint32).itemsize
        + triangles.directed_edge_count
        * (
            np.dtype(np.uint32).itemsize
            + np.dtype(np.uint64).itemsize
        )
    )

    factor_started = time.perf_counter()
    factor_adjacency, factor_colors = factor_graph_batch(
        report["type_examples"]
    )
    factor_build_seconds = time.perf_counter() - factor_started
    canonical_started = time.perf_counter()
    canonical = canonicalize_colored_digraphs(
        factor_adjacency,
        factor_colors,
        backend="native",
    )
    canonical_wall_seconds = time.perf_counter() - canonical_started
    class_count = len(set(map(int, canonical.class_ids)))
    if class_count != report["factor_types"]:
        raise AssertionError(
            f"nauty found {class_count} classes, "
            f"expected {report['factor_types']}"
        )
    canonical_input_bytes = (
        factor_adjacency.nbytes + factor_colors.nbytes
    )

    result = {
        "source": str(arguments.report),
        "factor_types": report["factor_types"],
        "undirected_edges_including_loops": len(edges),
        "directed_nonloop_edges": triangles.directed_edge_count,
        "allowed_type_triples": len(triangles.triangles),
        "triangle_intersection_steps": triangles.intersection_steps,
        "csr_build_seconds": csr_build_seconds,
        "triangle_wall_seconds": triangle_wall_seconds,
        "triangle_second_pass_kernel_seconds": triangles.elapsed_seconds,
        "triangle_logical_gb_s": (
            logical_triangle_bytes / triangles.elapsed_seconds / 1e9
        ),
        "factor_graph_build_seconds": factor_build_seconds,
        "canonical_wall_seconds": canonical_wall_seconds,
        "canonical_kernel_seconds": canonical.elapsed_seconds,
        "canonical_graphs_per_second": (
            report["factor_types"] / canonical.elapsed_seconds
        ),
        "canonical_input_gb_s": (
            canonical_input_bytes / canonical.elapsed_seconds / 1e9
        ),
        "canonical_class_count": class_count,
        "canonical_search_nodes": canonical.search_nodes,
        "automorphism_group_exponent_max": int(
            np.max(canonical.automorphism_group_exponents)
        ),
    }
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
