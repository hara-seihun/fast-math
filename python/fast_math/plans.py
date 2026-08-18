"""Composable retained plans for repeated finite-group and graph work."""

from __future__ import annotations

from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .canonical import CanonicalDigraphBatch, canonicalize_colored_digraphs
from .ci import (
    CanonicalCayleyBatch,
    CayleyGraphBatch,
    canonicalize_cayley_graphs,
    cayley_graphs,
)


class FiniteGroupPlan:
    """Retain a validated finite multiplication table and inverse map."""

    def __init__(
        self,
        multiplication_table: ArrayLike,
        *,
        inverse_indices: ArrayLike | None = None,
        identity: int = 0,
    ) -> None:
        raw = np.asarray(multiplication_table)
        if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or len(raw) == 0:
            raise ValueError("multiplication_table must be a nonempty square matrix")
        if not np.issubdtype(raw.dtype, np.integer):
            raise TypeError("multiplication_table must contain integers")
        order = len(raw)
        if order > 512 or np.any(raw < 0) or np.any(raw >= order):
            raise ValueError("multiplication_table entry or order is out of range")
        if not isinstance(identity, Integral) or not 0 <= int(identity) < order:
            raise ValueError("identity is out of range")
        table = np.array(raw, dtype=np.uint32, order="C", copy=True)
        element = np.arange(order, dtype=np.uint32)
        if not (
            np.array_equal(table[int(identity)], element)
            and np.array_equal(table[:, int(identity)], element)
        ):
            raise ValueError("declared identity is not an identity in the table")
        if np.any(np.sort(table, axis=0) != element[:, np.newaxis]) or np.any(
            np.sort(table, axis=1) != element
        ):
            raise ValueError("multiplication_table is not a Latin square")
        for left in range(order):
            if not np.array_equal(table[table[left]], table[left, table]):
                raise ValueError("multiplication_table is not associative")
        if inverse_indices is None:
            inverses = np.empty(order, dtype=np.uint32)
            for value in range(order):
                candidates = np.flatnonzero(
                    (table[value] == int(identity))
                    & (table[:, value] == int(identity))
                )
                if len(candidates) != 1:
                    raise ValueError("multiplication table lacks unique two-sided inverses")
                inverses[value] = candidates[0]
        else:
            raw_inverses = np.asarray(inverse_indices)
            if (
                raw_inverses.shape != (order,)
                or not np.issubdtype(raw_inverses.dtype, np.integer)
                or np.any(raw_inverses < 0)
                or np.any(raw_inverses >= order)
            ):
                raise ValueError("inverse_indices must contain one valid index per element")
            inverses = np.array(
                raw_inverses, dtype=np.uint32, order="C", copy=True
            )
            if np.any(table[element, inverses] != int(identity)) or np.any(
                table[inverses, element] != int(identity)
            ):
                raise ValueError("inverse_indices do not give two-sided inverses")
        table.flags.writeable = False
        inverses.flags.writeable = False
        self._table = table
        self._inverses = inverses
        self._identity = int(identity)

    @property
    def order(self) -> int:
        return len(self._table)

    @property
    def multiplication_table(self) -> NDArray[np.uint32]:
        return self._table

    @property
    def inverse_indices(self) -> NDArray[np.uint32]:
        return self._inverses

    @property
    def identity(self) -> int:
        return self._identity


class CayleyGraphPlan:
    """Retain one finite group across repeated Cayley graph batches."""

    def __init__(
        self,
        group: FiniteGroupPlan | ArrayLike,
        *,
        inverse_indices: ArrayLike | None = None,
        identity: int = 0,
    ) -> None:
        self.group = (
            group
            if isinstance(group, FiniteGroupPlan)
            else FiniteGroupPlan(
                group, inverse_indices=inverse_indices, identity=identity
            )
        )

    def graphs(
        self,
        connection_sets: ArrayLike,
        *,
        require_inverse_closed: bool = True,
        threads: int = 1,
        backend: Literal["auto", "native", "reference"] = "auto",
    ) -> CayleyGraphBatch:
        return cayley_graphs(
            self.group.multiplication_table,
            connection_sets,
            inverse_indices=self.group.inverse_indices,
            identity=self.group.identity,
            require_inverse_closed=require_inverse_closed,
            threads=threads,
            backend=backend,
        )

    def canonicalize(
        self,
        connection_sets: ArrayLike,
        *,
        require_inverse_closed: bool = True,
        threads: int = 1,
        collect_automorphism_generators: bool = False,
        graph_backend: Literal["auto", "native", "reference"] = "auto",
        canonical_backend: Literal["auto", "native", "reference"] = "auto",
    ) -> CanonicalCayleyBatch:
        return canonicalize_cayley_graphs(
            self.group.multiplication_table,
            connection_sets,
            inverse_indices=self.group.inverse_indices,
            identity=self.group.identity,
            require_inverse_closed=require_inverse_closed,
            threads=threads,
            collect_automorphism_generators=collect_automorphism_generators,
            graph_backend=graph_backend,
            canonical_backend=canonical_backend,
        )


class GraphCanonicalPlan:
    """Retain fixed packed digraph adjacency while vertex colors vary."""

    def __init__(self, adjacency_words: ArrayLike) -> None:
        raw = np.asarray(adjacency_words)
        if raw.ndim == 2:
            raw = raw[np.newaxis, :, :]
        if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] == 0:
            raise ValueError(
                "adjacency_words must describe one fixed packed graph"
            )
        if not np.issubdtype(raw.dtype, np.integer):
            raise TypeError("adjacency_words must contain integers")
        if np.issubdtype(raw.dtype, np.signedinteger) and np.any(raw < 0):
            raise ValueError("adjacency_words must be nonnegative")
        vertex_count = raw.shape[1]
        expected_words = (vertex_count + 63) // 64
        if raw.shape[2] != expected_words:
            raise ValueError("packed adjacency word count is inconsistent")
        adjacency = np.array(raw, dtype=np.uint64, order="C", copy=True)
        final_bits = vertex_count % 64
        if final_bits and np.any(adjacency[:, :, -1] >> np.uint64(final_bits)):
            raise ValueError("adjacency_words contains an out-of-range bit")
        vertices = np.arange(vertex_count)
        loop_bits = (
            adjacency[0, vertices, vertices // 64]
            >> (vertices % 64).astype(np.uint64)
        ) & np.uint64(1)
        if np.any(loop_bits):
            raise ValueError("self-loops are not supported")
        adjacency.flags.writeable = False
        self._adjacency = adjacency

    @property
    def vertex_count(self) -> int:
        return self._adjacency.shape[1]

    def canonicalize_colors(
        self,
        vertex_colors: ArrayLike,
        *,
        threads: int = 1,
        collect_automorphism_generators: bool = False,
        backend: Literal["auto", "native", "reference"] = "auto",
    ) -> CanonicalDigraphBatch:
        colors = np.asarray(vertex_colors)
        if colors.ndim == 1:
            colors = colors[np.newaxis, :]
        if colors.ndim != 2 or colors.shape[1] != self.vertex_count:
            raise ValueError("vertex_colors must have shape (graphs, vertices)")
        adjacency = np.broadcast_to(
            self._adjacency,
            (len(colors), self.vertex_count, self._adjacency.shape[2]),
        )
        return canonicalize_colored_digraphs(
            adjacency,
            colors,
            threads=threads,
            collect_automorphism_generators=collect_automorphism_generators,
            backend=backend,
        )


__all__ = ["CayleyGraphPlan", "FiniteGroupPlan", "GraphCanonicalPlan"]
