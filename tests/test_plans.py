from __future__ import annotations

import numpy as np
import pytest

from fast_math import CayleyGraphPlan, FiniteGroupPlan, GraphCanonicalPlan


def cyclic_table(order: int) -> np.ndarray:
    values = np.arange(order, dtype=np.uint32)
    return (values[:, None] + values[None, :]) % order


def test_finite_group_plan_derives_inverses() -> None:
    table = cyclic_table(5)
    group = FiniteGroupPlan(table)
    assert table.flags.writeable
    table.fill(0)
    assert group.order == 5
    np.testing.assert_array_equal(group.multiplication_table[1], [1, 2, 3, 4, 0])
    np.testing.assert_array_equal(group.inverse_indices, [0, 4, 3, 2, 1])
    with pytest.raises(ValueError, match="identity"):
        FiniteGroupPlan(cyclic_table(5), identity=1)
    nonassociative = np.asarray(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 3, 4, 2],
            [2, 3, 4, 0, 1],
            [3, 4, 1, 2, 0],
            [4, 2, 0, 1, 3],
        ],
        dtype=np.uint32,
    )
    with pytest.raises(ValueError, match="associative"):
        FiniteGroupPlan(nonassociative)


def test_cayley_plan_reuses_group() -> None:
    group = FiniteGroupPlan(cyclic_table(4))
    plan = CayleyGraphPlan(group)
    connections = np.asarray([[0b1010]], dtype=np.uint64)
    reference = plan.graphs(connections, backend="reference")
    native = plan.graphs(connections, threads=2, backend="native")
    np.testing.assert_array_equal(native.adjacency_words, reference.adjacency_words)


def test_fixed_graph_canonical_plan_varies_colors() -> None:
    adjacency = np.asarray(
        [[0b0010], [0b0101], [0b1010], [0b0100]], dtype=np.uint64
    )
    plan = GraphCanonicalPlan(adjacency)
    colors = np.asarray([[0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.uint32)
    result = plan.canonicalize_colors(colors, backend="reference", threads=2)
    assert result.adjacency_words.shape == (2, 4, 1)
    assert result.vertex_colors.shape == (2, 4)


def test_fixed_graph_plan_rejects_invalid_packed_graphs() -> None:
    with pytest.raises(TypeError, match="integers"):
        GraphCanonicalPlan([[0.0], [0.0]])
    with pytest.raises(ValueError, match="self-loops"):
        GraphCanonicalPlan([[0b01], [0b00]])
    with pytest.raises(ValueError, match="out-of-range"):
        GraphCanonicalPlan([[0b100], [0b000]])
