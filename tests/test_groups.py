from __future__ import annotations

from collections import deque
from itertools import permutations
from math import factorial

import numpy as np
import pytest

from fast_math.groups import (
    PermutationGroup,
    compose_permutations,
    group_order,
    invert_permutation,
    permutation_group_contains,
    permutation_orbits,
    schreier_sims,
)


def compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(left[right[point]] for point in range(len(left)))


def generated_group(
    generators: list[tuple[int, ...]],
    degree: int,
) -> set[tuple[int, ...]]:
    identity = tuple(range(degree))
    group = {identity}
    queue = deque([identity])
    while queue:
        current = queue.popleft()
        for generator in generators:
            product = compose(generator, current)
            if product not in group:
                group.add(product)
                queue.append(product)
    return group


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_schreier_sims_symmetric_group(backend: str) -> None:
    degree = 7
    cycle = np.roll(np.arange(degree, dtype=np.uint32), -1)
    transposition = np.arange(degree, dtype=np.uint32)
    transposition[[0, 1]] = transposition[[1, 0]]
    generators = np.stack([cycle, transposition])
    chain = schreier_sims(generators, backend=backend)
    assert chain.order == factorial(degree)
    assert len(chain.base) == degree - 1
    assert np.prod(chain.orbit_sizes, dtype=object) == factorial(degree)
    assert chain.level_generator_offsets[0] == 0
    assert chain.level_generator_offsets[-1] == len(
        chain.strong_generators
    )


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_reusable_permutation_group(backend: str) -> None:
    degree = 7
    cycle = np.roll(np.arange(degree, dtype=np.uint32), -1)
    transposition = np.arange(degree, dtype=np.uint32)
    transposition[[0, 1]] = transposition[[1, 0]]
    candidates = np.stack(
        [
            np.arange(degree, dtype=np.uint32),
            cycle,
            transposition,
        ]
    )
    with PermutationGroup(
        np.stack([cycle, transposition]), backend=backend
    ) as group:
        assert group.order == factorial(degree)
        assert len(group.orbits) == 1
        np.testing.assert_array_equal(
            group.contains(candidates, threads=3),
            [True, True, True],
        )
        np.testing.assert_array_equal(
            group.contains(candidates, threads=2),
            [True, True, True],
        )
        assert not group.closed
    assert group.closed
    with pytest.raises(RuntimeError, match="closed"):
        group.contains(candidates)


def test_native_and_reference_reusable_groups_match() -> None:
    generators = np.asarray(
        [
            [1, 2, 3, 4, 0, 5],
            [1, 0, 2, 3, 4, 5],
            [0, 1, 2, 4, 5, 3],
        ],
        dtype=np.uint32,
    )
    candidates = np.asarray(list(permutations(range(6))), dtype=np.uint32)
    with PermutationGroup(generators, backend="reference") as reference:
        with PermutationGroup(generators, backend="native") as native:
            assert native.order == reference.order
            np.testing.assert_array_equal(native.base, reference.base)
            np.testing.assert_array_equal(
                native.orbit_sizes, reference.orbit_sizes
            )
            assert [orbit.tolist() for orbit in native.orbits] == [
                orbit.tolist() for orbit in reference.orbits
            ]
            np.testing.assert_array_equal(
                native.contains(candidates, threads=4),
                reference.contains(candidates),
            )


def test_native_and_reference_chains_match_exactly() -> None:
    generators = np.asarray(
        [
            [1, 2, 3, 4, 0, 5],
            [1, 0, 2, 3, 4, 5],
            [0, 1, 2, 4, 5, 3],
        ],
        dtype=np.uint32,
    )
    reference = schreier_sims(generators, backend="reference")
    native = schreier_sims(generators, backend="native")
    assert native.order == reference.order
    np.testing.assert_array_equal(native.base, reference.base)
    np.testing.assert_array_equal(native.orbit_sizes, reference.orbit_sizes)
    np.testing.assert_array_equal(
        native.level_generator_offsets,
        reference.level_generator_offsets,
    )
    np.testing.assert_array_equal(
        native.strong_generators,
        reference.strong_generators,
    )


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_membership_matches_complete_small_group(backend: str) -> None:
    generators = [
        (1, 2, 3, 0),
        (1, 0, 2, 3),
    ]
    group = generated_group(generators, 4)
    candidates = np.asarray(list(permutations(range(4))), dtype=np.uint32)
    expected = np.asarray(
        [tuple(map(int, row)) in group for row in candidates],
        dtype=np.bool_,
    )
    actual = permutation_group_contains(
        generators,
        candidates,
        threads=3,
        backend=backend,
    )
    np.testing.assert_array_equal(actual, expected)

    duplicate_query = np.stack([candidates[0], candidates[0], candidates[-1]])
    duplicate_expected = np.asarray(
        [
            tuple(map(int, row)) in group
            for row in duplicate_query
        ],
        dtype=np.bool_,
    )
    np.testing.assert_array_equal(
        permutation_group_contains(
            generators,
            duplicate_query,
            backend=backend,
        ),
        duplicate_expected,
    )


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_orbits_and_trivial_group(backend: str) -> None:
    generators = np.asarray(
        [[1, 0, 2, 4, 3, 5]],
        dtype=np.uint32,
    )
    actual = permutation_orbits(generators, backend=backend)
    assert [orbit.tolist() for orbit in actual] == [
        [0, 1],
        [2],
        [3, 4],
        [5],
    ]

    trivial = schreier_sims(
        np.empty((0, 6), dtype=np.uint32),
        degree=6,
        backend=backend,
    )
    assert trivial.order == 1
    assert len(trivial.base) == 0
    np.testing.assert_array_equal(
        trivial.level_generator_offsets,
        [0],
    )


@pytest.mark.parametrize("backend", ["reference", "native"])
@pytest.mark.parametrize("degree", [512, 4096])
def test_degree_boundary_cyclic(backend: str, degree: int) -> None:
    cycle = np.roll(np.arange(degree, dtype=np.uint32), -1)
    assert group_order([cycle], backend=backend) == degree
    inverse = invert_permutation(cycle)
    queries = np.stack(
        [
            np.arange(degree, dtype=np.uint32),
            cycle,
            inverse,
        ]
    )
    np.testing.assert_array_equal(
        permutation_group_contains(
            [cycle],
            queries,
            backend=backend,
        ),
        [True, True, True],
    )
    orbits = permutation_orbits([cycle], backend=backend)
    assert len(orbits) == 1 and orbits[0].size == degree


def test_degree_above_boundary_rejected() -> None:
    degree = 4097
    cycle = np.roll(np.arange(degree, dtype=np.uint32), -1)
    with pytest.raises(ValueError):
        permutation_orbits([cycle])


def test_permutation_helpers_use_left_after_right_convention() -> None:
    left = np.asarray([1, 0, 2, 3], dtype=np.uint32)
    right = np.asarray([0, 2, 1, 3], dtype=np.uint32)
    np.testing.assert_array_equal(
        compose_permutations(left, right),
        [1, 2, 0, 3],
    )
    np.testing.assert_array_equal(
        compose_permutations(left, invert_permutation(left)),
        np.arange(4),
    )


@pytest.mark.parametrize(
    ("generators", "degree", "message"),
    [
        ([[0, 0]], None, "not a permutation"),
        ([[0, 2]], None, "out-of-range"),
        ([], None, "degree is required"),
        (np.empty((0, 0), dtype=np.uint32), 0, "between one and 4096"),
        (np.empty((0, 4097), dtype=np.uint32), 4097, "between one and 4096"),
    ],
)
def test_group_kernels_reject_invalid_inputs(
    generators,
    degree,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        schreier_sims(generators, degree=degree, backend="reference")


def test_group_kernels_reject_invalid_backend_and_threads() -> None:
    with pytest.raises(ValueError, match="backend"):
        group_order([[0]], backend="other")
    with pytest.raises(ValueError, match="threads"):
        permutation_group_contains([[0]], [[0]], threads=-1)
