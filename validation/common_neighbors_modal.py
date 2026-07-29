from __future__ import annotations

import json

import numpy as np

from fast_math import csr_common_neighbors, undirected_csr


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
    materialize=True,
    backend="reference",
)
native = csr_common_neighbors(
    graph.row_offsets,
    graph.column_indices,
    pairs,
    materialize=True,
    backend="native",
)
np.testing.assert_array_equal(native.pair_offsets, reference.pair_offsets)
np.testing.assert_array_equal(
    native.common_neighbors,
    reference.common_neighbors,
)
print(
    json.dumps(
        {
            "event": "fast_math_common_neighbors_modal_validation",
            "backend": native.backend,
            "pair_count": native.pair_count,
            "common_neighbor_count": native.common_neighbor_count,
            "intersection_steps": native.intersection_steps,
            "parity": True,
        },
        sort_keys=True,
    )
)
