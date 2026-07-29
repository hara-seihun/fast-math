from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from lambda_fast._native import load_library

from fast_math.ci import (
    canonicalize_cayley_graphs,
    cayley_graphs,
    coherent_configuration,
    deduplicate_subset_orbits,
    derivative_orbit_partitions,
    double_cosets,
    enumerate_subset_orbits,
    expand_atom_masks,
    generalized_dihedral_automorphisms,
    generalized_dihedral_group,
    graph_coherent_configuration,
    induced_atom_generators,
    inverse_closed_atoms,
    pack_subsets,
)

try:
    NAUTY_AVAILABLE = hasattr(
        load_library(),
        "fast_math_canonical_digraphs_nauty_u64",
    )
except (OSError, RuntimeError):
    NAUTY_AVAILABLE = False


def cyclic_automorphisms(modulus: int) -> np.ndarray:
    units = [
        unit
        for unit in range(modulus)
        if np.gcd(unit, modulus) == 1
    ]
    return np.asarray(
        [
            [(unit * value) % modulus for value in range(modulus)]
            for unit in units
        ],
        dtype=np.uint32,
    )


def dihedral_four_data():
    group = generalized_dihedral_group([4])
    automorphisms = generalized_dihedral_automorphisms(
        group,
        cyclic_automorphisms(4),
    )
    atoms = inverse_closed_atoms(
        group.inverse_indices,
        identity=group.identity,
    )
    action = induced_atom_generators(atoms, automorphisms)
    return group, automorphisms, atoms, action


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_d8_inverse_closed_connection_orbits_match_atlas(
    backend: str,
) -> None:
    _, automorphisms, atoms, action = dihedral_four_data()
    partition = enumerate_subset_orbits(action, backend=backend)
    assert len(atoms.atoms) == 6
    assert len(automorphisms) == 8
    assert len(partition.representatives) == 24
    assert int(partition.orbit_sizes.sum()) == 64
    assert partition.representatives.tolist()[:8] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]


def test_native_and_reference_subset_orbits_match_exactly() -> None:
    action = np.asarray(
        [
            [1, 2, 3, 4, 0, 5],
            [1, 0, 2, 3, 4, 5],
        ],
        dtype=np.uint32,
    )
    reference = enumerate_subset_orbits(action, backend="reference")
    native = enumerate_subset_orbits(action, backend="native")
    np.testing.assert_array_equal(
        native.representatives, reference.representatives
    )
    np.testing.assert_array_equal(native.orbit_sizes, reference.orbit_sizes)
    np.testing.assert_array_equal(
        native.representative_of, reference.representative_of
    )


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_batched_cayley_graphs_match_d8_group_law(backend: str) -> None:
    group, _, atoms, action = dihedral_four_data()
    partition = enumerate_subset_orbits(action, backend="reference")
    masks = partition.representatives[:12]
    connections = expand_atom_masks(masks, atoms)
    batch = cayley_graphs(
        group.multiplication_table,
        connections,
        identity=group.identity,
        threads=3,
        backend=backend,
    )
    adjacency = batch.adjacency_words
    for row_index, mask in enumerate(map(int, masks)):
        steps = {
            int(element)
            for atom_index, atom in enumerate(atoms.atoms)
            if mask & (1 << atom_index)
            for element in atom
        }
        for vertex in range(len(group.elements)):
            actual = {
                target
                for target in range(len(group.elements))
                if int(adjacency[row_index, vertex, target // 64])
                & (1 << (target % 64))
            }
            expected = {
                int(group.multiplication_table[step, vertex])
                for step in steps
            }
            assert actual == expected


@pytest.mark.skipif(
    not NAUTY_AVAILABLE,
    reason="native nauty canonicalization is not built",
)
def test_cayley_native_and_reference_batches_match() -> None:
    group, _, atoms, action = dihedral_four_data()
    partition = enumerate_subset_orbits(action, backend="reference")
    connections = expand_atom_masks(partition.representatives, atoms)
    reference = cayley_graphs(
        group.multiplication_table,
        connections,
        identity=group.identity,
        backend="reference",
    )
    native = cayley_graphs(
        group.multiplication_table,
        connections,
        identity=group.identity,
        backend="native",
        threads=4,
    )
    np.testing.assert_array_equal(
        native.adjacency_words,
        reference.adjacency_words,
    )

    canonical = canonicalize_cayley_graphs(
        group.multiplication_table,
        connections,
        identity=group.identity,
        construction_backend="native",
        canonical_backend="native",
    )
    assert len(np.unique(canonical.class_ids)) == 14


def prism_map_d8(group) -> np.ndarray:
    image = []
    for vector, reflection in group.elements:
        exponent = vector[0]
        if reflection == 0 and exponent % 2 == 0:
            mapped = ((exponent,), 0)
        elif reflection == 0:
            mapped = (((1 - exponent) % 4,), 1)
        elif exponent % 2 == 0:
            mapped = (((exponent + 1) % 4,), 1)
        else:
            mapped = (((-exponent) % 4,), 0)
        image.append(group.elements.index(mapped))
    return np.asarray(image, dtype=np.uint32)


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_derivative_orbits_cover_identity_automorphism_and_prism_map(
    backend: str,
) -> None:
    group, automorphisms, atoms, _ = dihedral_four_data()
    identity = np.arange(len(group.elements), dtype=np.uint32)
    nonlinear = prism_map_d8(group)
    bijections = np.stack([identity, automorphisms[3], nonlinear])
    result = derivative_orbit_partitions(
        group.multiplication_table,
        group.inverse_indices,
        bijections,
        identity=group.identity,
        threads=3,
        backend=backend,
    )
    assert result.orbit_counts[0] == len(group.elements)
    assert result.orbit_counts[1] == len(group.elements)

    standard_mask = 0
    for atom_index, atom in enumerate(atoms.atoms):
        atom_set = set(map(int, atom))
        if atom_set in ({1, 3}, {4}):
            standard_mask |= 1 << atom_index
    connection = {
        int(element)
        for atom_index, atom in enumerate(atoms.atoms)
        if standard_mask & (1 << atom_index)
        for element in atom
    }
    labels = result.orbit_labels[2]
    for orbit in range(int(result.orbit_counts[2])):
        members = set(map(int, np.flatnonzero(labels == orbit)))
        assert members <= connection or members.isdisjoint(connection)


def test_native_and_reference_derivative_partitions_match() -> None:
    group, automorphisms, _, _ = dihedral_four_data()
    bijections = np.vstack([automorphisms, prism_map_d8(group)])
    reference = derivative_orbit_partitions(
        group.multiplication_table,
        group.inverse_indices,
        bijections,
        backend="reference",
    )
    native = derivative_orbit_partitions(
        group.multiplication_table,
        group.inverse_indices,
        bijections,
        backend="native",
        threads=4,
    )
    np.testing.assert_array_equal(native.orbit_labels, reference.orbit_labels)
    np.testing.assert_array_equal(native.orbit_counts, reference.orbit_counts)


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_s3_double_cosets_have_sizes_two_and_four(backend: str) -> None:
    elements = np.asarray(list(permutations(range(3))), dtype=np.uint32)
    transposition = np.asarray([[1, 0, 2]], dtype=np.uint32)
    result = double_cosets(
        elements,
        transposition,
        transposition,
        backend=backend,
    )
    assert sorted(map(int, result.class_sizes)) == [2, 4]
    assert len(result.representative_indices) == 2
    assert sorted(np.bincount(result.class_ids).tolist()) == [2, 4]


def test_native_and_reference_double_cosets_match_exactly() -> None:
    elements = np.asarray(list(permutations(range(4))), dtype=np.uint32)
    left = np.asarray([[1, 0, 2, 3]], dtype=np.uint32)
    right = np.asarray([[0, 1, 3, 2]], dtype=np.uint32)
    reference = double_cosets(
        elements, left, right, backend="reference"
    )
    native = double_cosets(elements, left, right, backend="native")
    np.testing.assert_array_equal(native.class_ids, reference.class_ids)
    np.testing.assert_array_equal(
        native.representative_indices,
        reference.representative_indices,
    )
    np.testing.assert_array_equal(native.class_sizes, reference.class_sizes)


def test_packed_subsets_support_degree_above_one_word() -> None:
    packed = pack_subsets([[0, 64, 129], [1, 65, 127]], 130)
    assert packed.shape == (2, 3)
    assert int(packed[0, 0]) == 1
    assert int(packed[0, 1]) == 1
    assert int(packed[0, 2]) == 2


def test_native_and_reference_multiword_subset_orbits_match() -> None:
    action = np.arange(70, dtype=np.uint32)
    action[[0, 64]] = action[[64, 0]]
    subsets = pack_subsets([[], [0], [64], [0, 64], [69]], 70)
    reference = deduplicate_subset_orbits(
        subsets,
        [action],
        atom_count=70,
        backend="reference",
    )
    native = deduplicate_subset_orbits(
        subsets,
        [action],
        atom_count=70,
        backend="native",
    )
    np.testing.assert_array_equal(native.class_ids, reference.class_ids)
    np.testing.assert_array_equal(
        native.representative_words,
        reference.representative_words,
    )
    np.testing.assert_array_equal(native.orbit_sizes, [1, 2, 1, 1])


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_c5_coherent_configuration_has_three_basic_relations(
    backend: str,
) -> None:
    adjacency = np.zeros((5, 5), dtype=np.uint8)
    for vertex in range(5):
        adjacency[vertex, (vertex + 1) % 5] = 1
        adjacency[(vertex + 1) % 5, vertex] = 1
    result = graph_coherent_configuration(adjacency, backend=backend)
    assert result.relation_count == 3
    assert result.relation_sizes.tolist() == [5, 10, 10]
    assert result.intersection_numbers.shape == (3, 3, 3)
    np.testing.assert_array_equal(
        result.intersection_numbers.sum(axis=(0, 1)),
        np.full(3, 5, dtype=np.uint64),
    )


def test_native_and_reference_wl2_match_exactly() -> None:
    relations = np.asarray(
        [
            [0, 1, 2, 1],
            [3, 0, 1, 2],
            [2, 3, 0, 1],
            [3, 2, 3, 0],
        ],
        dtype=np.uint32,
    )
    reference = coherent_configuration(relations, backend="reference")
    native = coherent_configuration(relations, backend="native")
    np.testing.assert_array_equal(native.relations, reference.relations)
    np.testing.assert_array_equal(
        native.intersection_numbers,
        reference.intersection_numbers,
    )
    np.testing.assert_array_equal(
        native.relation_sizes,
        reference.relation_sizes,
    )


def test_wl2_handles_a_graph_of_order_a_few_hundred() -> None:
    order = 257
    adjacency = np.ones((order, order), dtype=np.uint8)
    np.fill_diagonal(adjacency, 0)
    result = graph_coherent_configuration(adjacency, backend="native")
    assert result.relation_count == 2
    np.testing.assert_array_equal(
        result.relation_sizes,
        [order, order * (order - 1)],
    )
    assert result.intersection_numbers.shape == (2, 2, 2)


def test_ci_kernels_reject_hostile_inputs() -> None:
    group, _, atoms, action = dihedral_four_data()
    with pytest.raises(ValueError, match="max_subsets"):
        enumerate_subset_orbits(action, max_subsets=32)
    with pytest.raises(ValueError, match="out-of-range atom"):
        expand_atom_masks([1 << len(atoms.atoms)], atoms)
    with pytest.raises(ValueError, match="duplicate"):
        pack_subsets([[1, 1]], len(group.elements))

    directed = pack_subsets([[1]], len(group.elements))
    with pytest.raises(ValueError, match="inverse-closed"):
        cayley_graphs(
            group.multiplication_table,
            directed,
            backend="reference",
        )

    bad_bijection = np.arange(len(group.elements), dtype=np.uint32)
    bad_bijection[[0, 1]] = bad_bijection[[1, 0]]
    with pytest.raises(ValueError, match="fix the identity"):
        derivative_orbit_partitions(
            group.multiplication_table,
            group.inverse_indices,
            [bad_bijection],
        )

    elements = np.asarray(list(permutations(range(3)))[:3], dtype=np.uint32)
    with pytest.raises(ValueError, match="not closed"):
        double_cosets(
            elements,
            [[1, 0, 2]],
            [[0, 1, 2]],
            backend="reference",
        )
    with pytest.raises(ValueError, match="intersection tensor"):
        graph_coherent_configuration(
            np.zeros((2, 2), dtype=np.uint8),
            max_tensor_entries=0,
            backend="reference",
        )
    with pytest.raises(ValueError, match="non-automorphism"):
        generalized_dihedral_automorphisms(
            group,
            [[0, 2, 1, 3]],
        )
