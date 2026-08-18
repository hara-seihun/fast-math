from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from fast_math._native import load_library

from fast_math.ci import (
    atom_subsets_to_element_words,
    compose_u64_mask_luts,
    canonicalize_cayley_graphs,
    cayley_graphs,
    coherent_configuration,
    deduplicate_subset_orbits,
    derivative_group_orbits,
    derivative_orbit_partitions,
    double_cosets,
    enumerate_fixed_weight_subset_orbits,
    enumerate_subset_orbits,
    expand_atom_masks,
    generalized_dihedral_automorphisms,
    generalized_dihedral_group,
    graph_coherent_configuration,
    graph_wl2_refinement,
    induced_atom_generators,
    inverse_closed_atoms,
    pack_subsets,
    u64_mask_lut,
)

try:
    NAUTY_AVAILABLE = hasattr(
        load_library(),
        "fast_math_canonical_digraphs_nauty_u64",
    )
except (OSError, RuntimeError):
    NAUTY_AVAILABLE = False


def test_small_mask_luts_compose_exactly() -> None:
    permutation = np.asarray([2, 0, 3, 1], dtype=np.uint32)
    inverse = np.asarray([1, 3, 0, 2], dtype=np.uint32)
    direct = u64_mask_lut(permutation)
    inverse_lut = u64_mask_lut(inverse)
    identity = compose_u64_mask_luts(direct, inverse_lut)
    assert identity == tuple(range(16))
    assert direct[0b0011] == 0b0101
    assert direct[0b1100] == 0b1010


def test_small_mask_luts_reject_oversized_or_invalid_actions() -> None:
    with pytest.raises(ValueError, match="between one and 16"):
        u64_mask_lut(np.arange(17, dtype=np.uint32))
    with pytest.raises(ValueError, match="each point"):
        u64_mask_lut(np.asarray([0, 0, 1], dtype=np.uint32))


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
def test_complete_action_subset_orbits_match_generator_mode(
    backend: str,
) -> None:
    action = np.asarray(
        list(permutations(range(3))),
        dtype=np.uint32,
    )
    complete = enumerate_subset_orbits(
        action,
        action_is_group=True,
        backend=backend,
    )
    generated = enumerate_subset_orbits(
        np.asarray(
            [
                [1, 0, 2],
                [1, 2, 0],
            ],
            dtype=np.uint32,
        ),
        action_is_group=False,
        backend=backend,
    )
    automatic = enumerate_subset_orbits(action, backend=backend)
    np.testing.assert_array_equal(complete.class_ids, generated.class_ids)
    np.testing.assert_array_equal(
        complete.representative_indices,
        generated.representative_indices,
    )
    np.testing.assert_array_equal(automatic.class_ids, complete.class_ids)
    np.testing.assert_array_equal(automatic.orbit_sizes, complete.orbit_sizes)


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_complete_action_subset_orbits_reject_non_group(
    backend: str,
) -> None:
    incomplete = np.asarray(
        [
            [0, 1, 2],
            [1, 0, 2],
            [0, 2, 1],
        ],
        dtype=np.uint32,
    )
    with pytest.raises((ValueError, RuntimeError), match="not closed"):
        enumerate_subset_orbits(
            incomplete,
            action_is_group=True,
            backend=backend,
        )


def test_native_and_reference_fixed_weight_subset_orbits_match_exactly() -> None:
    action = np.asarray(
        [
            [(point + shift) % 8 for point in range(8)]
            for shift in range(8)
        ],
        dtype=np.uint32,
    )
    reference = enumerate_fixed_weight_subset_orbits(
        action,
        3,
        backend="reference",
    )
    native = enumerate_fixed_weight_subset_orbits(
        action,
        3,
        backend="native",
    )
    assert native.subset_count == 56
    assert int(native.orbit_sizes.sum()) == 56
    np.testing.assert_array_equal(
        native.representative_words,
        reference.representative_words,
    )
    np.testing.assert_array_equal(native.orbit_sizes, reference.orbit_sizes)
    repeated = enumerate_fixed_weight_subset_orbits(
        action,
        3,
        backend="native",
    )
    np.testing.assert_array_equal(repeated.representatives, native.representatives)
    np.testing.assert_array_equal(repeated.orbit_sizes, native.orbit_sizes)


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_fixed_weight_subset_orbits_cover_zero_weight_and_bit_63(
    backend: str,
) -> None:
    identity = np.arange(64, dtype=np.uint32)[np.newaxis, :]
    empty = enumerate_fixed_weight_subset_orbits(
        identity,
        0,
        max_subsets=1,
        backend=backend,
    )
    assert empty.representatives.tolist() == [0]
    assert empty.orbit_sizes.tolist() == [1]
    singletons = enumerate_fixed_weight_subset_orbits(
        identity,
        1,
        max_subsets=64,
        backend=backend,
    )
    assert len(singletons.representatives) == 64
    assert int(singletons.representatives[-1]) == 1 << 63
    assert singletons.orbit_sizes.tolist() == [1] * 64


def test_fixed_weight_subset_orbits_reject_invalid_domains() -> None:
    incomplete = np.asarray(
        [
            [0, 1, 2],
            [1, 0, 2],
            [0, 2, 1],
        ],
        dtype=np.uint32,
    )
    with pytest.raises(ValueError, match="not closed"):
        enumerate_fixed_weight_subset_orbits(incomplete, 1)
    identity = np.arange(10, dtype=np.uint32)[np.newaxis, :]
    with pytest.raises(ValueError, match="above max_subsets"):
        enumerate_fixed_weight_subset_orbits(identity, 5, max_subsets=100)
    with pytest.raises(ValueError, match="between zero and atom_count"):
        enumerate_fixed_weight_subset_orbits(identity, 11)


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_complete_action_subset_orbits_support_multiword_masks(
    backend: str,
) -> None:
    identity = np.arange(65, dtype=np.uint32)
    swap = identity.copy()
    swap[[0, 64]] = swap[[64, 0]]
    subsets = pack_subsets(
        [[], [0], [64], [0, 64]],
        65,
    )
    complete = deduplicate_subset_orbits(
        subsets,
        np.stack((identity, swap)),
        atom_count=65,
        action_is_group=True,
        backend=backend,
    )
    generated = deduplicate_subset_orbits(
        subsets,
        swap[np.newaxis, :],
        atom_count=65,
        action_is_group=False,
        backend=backend,
    )
    np.testing.assert_array_equal(complete.class_ids, generated.class_ids)
    np.testing.assert_array_equal(
        complete.representative_words,
        generated.representative_words,
    )
    np.testing.assert_array_equal(
        complete.orbit_sizes,
        generated.orbit_sizes,
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


def test_derivative_partitions_support_group_order_above_512() -> None:
    order = 513
    elements = np.arange(order, dtype=np.uint32)
    table = (elements[:, None] + elements[None, :]) % order
    inverses = (-elements.astype(np.int64) % order).astype(np.uint32)
    bijection = elements.copy()
    bijection[1], bijection[2] = bijection[2], bijection[1]
    reference = derivative_group_orbits(
        table,
        inverses,
        bijection,
        backend="reference",
    )
    native = derivative_group_orbits(
        table,
        inverses,
        bijection,
        backend="native",
    )
    np.testing.assert_array_equal(native.orbit_labels, reference.orbit_labels)
    assert len(native.orbits) == len(reference.orbits)


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


def test_native_and_reference_atom_subset_expansion_match() -> None:
    atoms = (
        np.asarray([0, 64], dtype=np.uint32),
        np.asarray([1, 65, 129], dtype=np.uint32),
        np.asarray([], dtype=np.uint32),
    )
    subsets = np.asarray([0, 1, 2, 3, 4, 7], dtype=np.uint64)
    reference = atom_subsets_to_element_words(
        subsets,
        atoms,
        group_order=130,
        backend="reference",
    )
    native = atom_subsets_to_element_words(
        subsets,
        atoms,
        group_order=130,
        threads=3,
        backend="native",
    )
    np.testing.assert_array_equal(native, reference)


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


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_wl2_can_skip_intersection_numbers(backend: str) -> None:
    adjacency = np.zeros((7, 7), dtype=np.uint8)
    for vertex in range(7):
        adjacency[vertex, (vertex + 1) % 7] = 1
        adjacency[(vertex + 1) % 7, vertex] = 1
    full = graph_coherent_configuration(adjacency, backend=backend)
    refined = graph_wl2_refinement(
        adjacency,
        backend=backend,
    )
    np.testing.assert_array_equal(refined.relations, full.relations)
    np.testing.assert_array_equal(
        refined.relation_sizes,
        full.relation_sizes,
    )
    assert refined.relation_count == full.relation_count
    assert refined.iterations == full.iterations


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
