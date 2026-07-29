from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from fast_math import csr_common_neighbors, undirected_csr


def _best_run(
    graph,
    pairs: np.ndarray,
    *,
    materialize: bool,
    repetitions: int,
):
    best_wall = float("inf")
    best_result = None
    for _ in range(repetitions):
        started = time.perf_counter()
        result = csr_common_neighbors(
            graph.row_offsets,
            graph.column_indices,
            pairs,
            materialize=materialize,
            backend="native",
        )
        wall = time.perf_counter() - started
        if wall < best_wall:
            best_wall = wall
            best_result = result
    return best_result, best_wall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--source-label")
    arguments = parser.parse_args()

    report = json.loads(arguments.report.read_text())
    edges = np.asarray(report["edge_types"], dtype=np.uint32)
    graph = undirected_csr(report["factor_types"], edges)
    pairs = np.ascontiguousarray(
        np.sort(edges[edges[:, 0] != edges[:, 1]], axis=1),
        dtype=np.uint32,
    )

    reference_counts_started = time.perf_counter()
    reference_counts = csr_common_neighbors(
        graph.row_offsets,
        graph.column_indices,
        pairs,
        backend="reference",
    )
    reference_counts_wall = time.perf_counter() - reference_counts_started
    reference_materialized_started = time.perf_counter()
    reference = csr_common_neighbors(
        graph.row_offsets,
        graph.column_indices,
        pairs,
        materialize=True,
        backend="reference",
    )
    reference_materialized_wall = (
        time.perf_counter() - reference_materialized_started
    )

    native_counts, counts_wall = _best_run(
        graph,
        pairs,
        materialize=False,
        repetitions=arguments.repetitions,
    )
    native_materialized, materialized_wall = _best_run(
        graph,
        pairs,
        materialize=True,
        repetitions=arguments.repetitions,
    )
    np.testing.assert_array_equal(
        native_counts.pair_offsets,
        reference_counts.pair_offsets,
    )
    np.testing.assert_array_equal(
        native_materialized.pair_offsets,
        reference.pair_offsets,
    )
    np.testing.assert_array_equal(
        native_materialized.common_neighbors,
        reference.common_neighbors,
    )

    digest = hashlib.sha256()
    digest.update(reference.pair_offsets.tobytes())
    digest.update(reference.common_neighbors.tobytes())
    logical_bytes = (
        reference.intersection_steps
        * 2
        * np.dtype(np.uint32).itemsize
        + reference.common_neighbor_count
        * np.dtype(np.uint32).itemsize
    )
    degrees = np.diff(graph.row_offsets)
    allocation_upper_bound = int(
        np.minimum(
            degrees[pairs[:, 0]],
            degrees[pairs[:, 1]],
        ).sum(dtype=np.uint64)
    )
    result = {
        "source": arguments.source_label or str(arguments.report),
        "vertex_count": graph.vertex_count,
        "directed_edge_count": len(graph.column_indices),
        "pair_count": len(pairs),
        "intersection_steps": reference.intersection_steps,
        "common_neighbor_count": reference.common_neighbor_count,
        "allocation_upper_bound": allocation_upper_bound,
        "allocation_upper_bound_ratio": (
            allocation_upper_bound / reference.common_neighbor_count
        ),
        "output_sha256": digest.hexdigest(),
        "reference_counts_wall_seconds": reference_counts_wall,
        "reference_materialized_wall_seconds": (
            reference_materialized_wall
        ),
        "native_counts_wall_seconds": counts_wall,
        "native_counts_kernel_seconds": native_counts.elapsed_seconds,
        "native_materialized_wall_seconds": materialized_wall,
        "native_materialized_output_kernel_seconds": (
            native_materialized.elapsed_seconds
        ),
        "counts_speedup": reference_counts_wall / counts_wall,
        "materialized_speedup": (
            reference_materialized_wall / materialized_wall
        ),
        "materialized_logical_gb_s": (
            logical_bytes
            / native_materialized.elapsed_seconds
            / 1e9
        ),
        "repetitions": arguments.repetitions,
    }
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
