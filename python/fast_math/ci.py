from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from itertools import combinations, product
from math import comb
from numbers import Integral
from time import perf_counter
from typing import Iterable, Iterator, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lambda_fast._native import NativeUnavailable

from ._ci_native import (
    cayley_graphs_native,
    derivative_orbits_native,
    fixed_weight_subset_orbits_native,
    expand_atom_subsets_native,
    intersection_numbers_native,
    subset_orbits_native,
    wl2_refine_native,
)
from .canonical import CanonicalDigraphBatch, canonicalize_colored_digraphs
from .groups import DoubleCosetPartition, group_order, permutation_double_cosets


CIBackend = Literal["auto", "native", "reference"]
PermutationDoubleCosetPartition = DoubleCosetPartition
U64MaskLUT = tuple[int, ...]


def u64_mask_lut(
    permutation: ArrayLike,
    *,
    degree: int | None = None,
) -> U64MaskLUT:
    """Precompute an exact lookup table for a small packed-mask action.

    The table maps bit ``i`` to bit ``permutation[i]``.  It is deliberately
    bounded at degree 16: the resulting table is a hot-loop microkernel for
    small finite-group searches, not a general replacement for the packed
    native orbit kernels.  Applying a tuple lookup is substantially cheaper
    than rebuilding a Python integer mask one set bit at a time.
    """
    array = np.asarray(permutation)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("permutation must be a one-dimensional integer array")
    if degree is None:
        degree = int(array.size)
    if not isinstance(degree, Integral) or not 0 < int(degree) <= 16:
        raise ValueError("degree must be an integer between one and 16")
    degree = int(degree)
    if array.size != degree:
        raise ValueError("permutation length must equal degree")
    prepared = np.ascontiguousarray(array, dtype=np.int64)
    if np.any(prepared < 0) or np.any(prepared >= degree):
        raise ValueError("permutation values are outside the degree")
    if not np.array_equal(np.sort(prepared), np.arange(degree)):
        raise ValueError("permutation must contain each point exactly once")
    image_bits = tuple(1 << int(image) for image in prepared)
    lookup = [0] * (1 << degree)
    for mask in range(1, len(lookup)):
        least = mask & -mask
        lookup[mask] = lookup[mask ^ least] | image_bits[least.bit_length() - 1]
    return tuple(lookup)


def compose_u64_mask_luts(
    first: U64MaskLUT,
    second: U64MaskLUT,
) -> U64MaskLUT:
    """Compose two packed-mask lookup tables as ``first(second(mask))``."""
    if len(first) == 0 or len(first) != len(second):
        raise ValueError("mask lookup tables must have the same nonzero size")
    limit = len(first)
    if limit & (limit - 1):
        raise ValueError("mask lookup table size must be a power of two")
    if any(
        value < 0 or value >= limit
        for table in (first, second)
        for value in table
    ):
        raise ValueError("mask lookup table contains an out-of-range value")
    return tuple(first[second[mask]] for mask in range(limit))


@dataclass(frozen=True)
class InverseClosedAtoms:
    order: int
    identity: int
    atoms: tuple[NDArray[np.uint32], ...]
    atom_of_element: NDArray[np.int32]

    def __len__(self) -> int:
        return len(self.atoms)

    def __iter__(self) -> Iterator[NDArray[np.uint32]]:
        return iter(self.atoms)

    def __getitem__(self, index: int) -> NDArray[np.uint32]:
        return self.atoms[index]


@dataclass(frozen=True)
class SubsetOrbitPartition:
    subset_words: NDArray[np.uint64]
    class_ids: NDArray[np.uint64]
    representative_indices: NDArray[np.uint64]
    representative_words: NDArray[np.uint64]
    orbit_sizes: NDArray[np.uint64]
    atom_count: int
    elapsed_seconds: float
    backend: str

    @property
    def representatives(self) -> NDArray[np.uint64]:
        if self.representative_words.shape[1] == 1:
            return self.representative_words[:, 0]
        return self.representative_words

    @property
    def representative_of(self) -> NDArray[np.uint64]:
        words = self.representative_words[self.class_ids]
        if words.shape[1] == 1:
            return words[:, 0]
        return words


@dataclass(frozen=True)
class FixedWeightSubsetOrbits:
    representative_words: NDArray[np.uint64]
    orbit_sizes: NDArray[np.uint64]
    atom_count: int
    subset_weight: int
    subset_count: int
    elapsed_seconds: float
    backend: str

    @property
    def representatives(self) -> NDArray[np.uint64]:
        return self.representative_words[:, 0]


@dataclass(frozen=True)
class CayleyGraphBatch:
    adjacency_words: NDArray[np.uint64]
    connection_words: NDArray[np.uint64]
    group_order: int
    graph_count: int
    elapsed_seconds: float
    backend: str

    @property
    def graph64_masks(self) -> NDArray[np.uint64]:
        if self.adjacency_words.shape[2] != 1:
            raise ValueError("graph64 masks require group order at most 64")
        return self.adjacency_words[:, :, 0]


@dataclass(frozen=True)
class CanonicalCayleyBatch:
    graphs: CayleyGraphBatch
    canonical: CanonicalDigraphBatch

    @property
    def class_ids(self) -> NDArray[np.uint64]:
        return self.canonical.class_ids


@dataclass(frozen=True)
class DerivativeOrbitPartition:
    generators: NDArray[np.uint32]
    orbit_labels: NDArray[np.uint32]
    orbits: tuple[NDArray[np.uint32], ...]
    elapsed_seconds: float
    backend: str


@dataclass(frozen=True)
class DerivativeOrbitBatch:
    orbit_labels: NDArray[np.uint32]
    orbit_counts: NDArray[np.uint32]
    elapsed_seconds: float
    backend: str

    def orbits(self, index: int) -> tuple[NDArray[np.uint32], ...]:
        labels = self.orbit_labels[index]
        return tuple(
            np.flatnonzero(labels == orbit).astype(np.uint32)
            for orbit in range(int(self.orbit_counts[index]))
        )


@dataclass(frozen=True)
class CoherentConfiguration:
    relations: NDArray[np.uint32]
    relation_count: int
    relation_sizes: NDArray[np.uint64]
    intersection_numbers: NDArray[np.uint64]
    iterations: int
    elapsed_seconds: float
    backend: str

    @property
    def basic_sets(self) -> tuple[NDArray[np.uint32], ...]:
        return tuple(
            np.argwhere(self.relations == relation).astype(np.uint32)
            for relation in range(self.relation_count)
        )


@dataclass(frozen=True)
class WL2Refinement:
    relations: NDArray[np.uint32]
    relation_count: int
    relation_sizes: NDArray[np.uint64]
    iterations: int
    elapsed_seconds: float
    backend: str

    @property
    def basic_sets(self) -> tuple[NDArray[np.uint32], ...]:
        return tuple(
            np.argwhere(self.relations == relation).astype(np.uint32)
            for relation in range(self.relation_count)
        )


@dataclass(frozen=True)
class GeneralizedDihedralGroup:
    moduli: tuple[int, ...]
    elements: tuple[tuple[tuple[int, ...], int], ...]
    multiplication_table: NDArray[np.uint32]
    inverse_indices: NDArray[np.uint32]
    identity: int
    abelian_order: int


def _validate_backend(backend: str) -> None:
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")


def _prepare_action(
    generators: ArrayLike,
    atom_count: int,
) -> NDArray[np.uint32]:
    array = np.asarray(generators)
    if array.ndim == 1 and array.size:
        array = array[np.newaxis, :]
    if array.ndim == 1 and array.size == 0:
        array = np.empty((0, atom_count), dtype=np.uint32)
    if array.ndim != 2 or array.shape[1] != atom_count:
        raise ValueError("action generators must have shape (count, atoms)")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("action generators must contain integers")
    if np.issubdtype(array.dtype, np.signedinteger) and np.any(array < 0):
        raise ValueError("action generators must be nonnegative")
    prepared = np.ascontiguousarray(array, dtype=np.uint32)
    expected = np.arange(atom_count, dtype=np.uint32)
    for generator in prepared:
        if not np.array_equal(np.sort(generator), expected):
            raise ValueError("action generator is not a permutation")
    return prepared


def _prepare_subset_words(
    subsets: ArrayLike,
    atom_count: int,
    *,
    require_unique: bool = True,
) -> NDArray[np.uint64]:
    if not isinstance(atom_count, Integral) or not 0 <= int(atom_count) <= 512:
        raise ValueError("atom_count must be an integer between zero and 512")
    atom_count = int(atom_count)
    word_count = (atom_count + 63) // 64
    array = np.asarray(subsets)
    if atom_count == 0:
        if array.ndim != 1:
            raise ValueError("zero-atom subsets must be one-dimensional")
        if any(not isinstance(value, Integral) or value != 0 for value in array):
            raise ValueError("the only zero-atom subset is zero")
        return np.empty((len(array), 0), dtype=np.uint64)
    if array.ndim == 1:
        if word_count > 1:
            raise ValueError("multiword subsets require a two-dimensional array")
        if array.dtype.kind == "O":
            maximum = np.iinfo(np.uint64).max
            if any(
                not isinstance(value, Integral)
                or value < 0
                or value > maximum
                for value in array
            ):
                raise ValueError("subset masks must fit uint64")
        elif not np.issubdtype(array.dtype, np.integer):
            raise ValueError("subset masks must contain integers")
        elif np.issubdtype(array.dtype, np.signedinteger) and np.any(array < 0):
            raise ValueError("subset masks must be nonnegative")
        array = np.asarray(array, dtype=np.uint64).reshape((-1, word_count))
    if array.ndim != 2 or array.shape[1] != word_count:
        raise ValueError("subset words have an invalid packed shape")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("subset words must contain integers")
    if np.issubdtype(array.dtype, np.signedinteger) and np.any(array < 0):
        raise ValueError("subset words must be nonnegative")
    prepared = np.ascontiguousarray(array, dtype=np.uint64)
    if atom_count:
        final_bits = atom_count % 64
        if final_bits and np.any(prepared[:, -1] >> np.uint64(final_bits)):
            raise ValueError("subset contains an out-of-range atom")
    if require_unique and len(prepared) > 1:
        if word_count == 1:
            unique_count = len(np.unique(prepared[:, 0]))
        else:
            unique_count = len(np.unique(prepared, axis=0))
        if unique_count != len(prepared):
            raise ValueError("subset rows must be unique")
    return prepared


def _subset_key(words: NDArray[np.uint64]) -> tuple[int, ...]:
    return tuple(reversed(tuple(map(int, words))))


def _permute_subset_reference(
    words: NDArray[np.uint64],
    permutation: NDArray[np.uint32],
    atom_count: int,
) -> tuple[int, ...]:
    output = [0] * len(words)
    for atom in range(atom_count):
        if int(words[atom // 64]) & (1 << (atom % 64)):
            image = int(permutation[atom])
            output[image // 64] |= 1 << (image % 64)
    return tuple(output)


def _subset_orbits_reference(
    subsets: NDArray[np.uint64],
    generators: NDArray[np.uint32],
    atom_count: int,
    action_is_group: bool | None,
) -> tuple[NDArray[np.uint64], NDArray[np.uint64], NDArray[np.uint64]]:
    index_of = {
        tuple(map(int, row)): index
        for index, row in enumerate(subsets)
    }
    unseen = set(range(len(subsets)))
    raw_orbits: list[list[int]] = []
    complete_action = _is_complete_action_reference(
        generators,
        atom_count,
        required=action_is_group is True,
    )
    if action_is_group is False:
        complete_action = False
    while unseen:
        seed = min(unseen)
        if complete_action:
            orbit = set()
            for generator in generators:
                image = _permute_subset_reference(
                    subsets[seed],
                    generator,
                    atom_count,
                )
                if image not in index_of:
                    raise ValueError(
                        "subset collection is not invariant under the action"
                    )
                orbit.add(index_of[image])
        else:
            orbit = {seed}
            queue = deque([seed])
            while queue:
                current = queue.popleft()
                for generator in generators:
                    image = _permute_subset_reference(
                        subsets[current],
                        generator,
                        atom_count,
                    )
                    if image not in index_of:
                        raise ValueError(
                            "subset collection is not invariant under the action"
                        )
                    image_index = index_of[image]
                    if image_index not in orbit:
                        orbit.add(image_index)
                        queue.append(image_index)
        unseen.difference_update(orbit)
        raw_orbits.append(sorted(orbit))
    raw_orbits.sort(
        key=lambda orbit: min(_subset_key(subsets[index]) for index in orbit)
    )
    class_ids = np.empty(len(subsets), dtype=np.uint64)
    representatives = np.empty(len(raw_orbits), dtype=np.uint64)
    sizes = np.empty(len(raw_orbits), dtype=np.uint64)
    for class_id, orbit in enumerate(raw_orbits):
        representative = min(orbit, key=lambda index: _subset_key(subsets[index]))
        representatives[class_id] = representative
        sizes[class_id] = len(orbit)
        class_ids[orbit] = class_id
    return class_ids, representatives, sizes


def _is_complete_action_reference(
    action: NDArray[np.uint32],
    degree: int,
    *,
    required: bool,
) -> bool:
    elements = {tuple(map(int, row)) for row in action}
    if len(elements) != len(action):
        if required:
            raise ValueError("complete action rows must be unique")
        return False
    identity = tuple(range(degree))
    if identity not in elements:
        if required:
            raise ValueError("complete action must contain the identity")
        return False
    # The rows generate a permutation group containing every row.  Therefore
    # they are the complete group exactly when that generated group's order is
    # the number of unique rows.  Schreier--Sims establishes this without the
    # quadratic all-pairs closure scan, which dominates large complete actions.
    if group_order(action, degree=degree, backend="reference") != len(action):
        if required:
            raise ValueError("complete action is not closed under composition")
        return False
    return True


def deduplicate_subset_orbits(
    subsets: ArrayLike,
    action_generators: ArrayLike,
    *,
    atom_count: int,
    action_is_group: bool | None = None,
    backend: CIBackend = "auto",
) -> SubsetOrbitPartition:
    """Partition invariant subsets under generators or a complete group action."""
    _validate_backend(backend)
    if action_is_group not in {None, False, True}:
        raise ValueError("action_is_group must be true, false, or None")
    words = _prepare_subset_words(
        subsets,
        atom_count,
        require_unique=backend == "reference",
    )
    generators = _prepare_action(action_generators, int(atom_count))
    if atom_count == 0:
        if len(words) != 1:
            raise ValueError("zero atoms have exactly one subset")
        return SubsetOrbitPartition(
            subset_words=words,
            class_ids=np.asarray([0], dtype=np.uint64),
            representative_indices=np.asarray([0], dtype=np.uint64),
            representative_words=words.copy(),
            orbit_sizes=np.asarray([1], dtype=np.uint64),
            atom_count=0,
            elapsed_seconds=0.0,
            backend="reference",
        )
    if backend != "reference":
        try:
            class_ids, indices, sizes, stats = subset_orbits_native(
                words,
                generators,
                int(atom_count),
                0 if action_is_group is None else (2 if action_is_group else 1),
            )
        except (NativeUnavailable, OSError, AttributeError):
            if backend == "native":
                raise
        else:
            return SubsetOrbitPartition(
                subset_words=words,
                class_ids=class_ids,
                representative_indices=indices,
                representative_words=words[indices],
                orbit_sizes=sizes,
                atom_count=int(atom_count),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    if len(words) > 1:
        if words.shape[1] == 1:
            unique_count = len(np.unique(words[:, 0]))
        else:
            unique_count = len(np.unique(words, axis=0))
        if unique_count != len(words):
            raise ValueError("subset rows must be unique")
    started = perf_counter()
    class_ids, indices, sizes = _subset_orbits_reference(
        words,
        generators,
        int(atom_count),
        action_is_group,
    )
    return SubsetOrbitPartition(
        subset_words=words,
        class_ids=class_ids,
        representative_indices=indices,
        representative_words=words[indices],
        orbit_sizes=sizes,
        atom_count=int(atom_count),
        elapsed_seconds=perf_counter() - started,
        backend="reference",
    )


def enumerate_subset_orbits(
    action_generators: ArrayLike,
    *,
    atom_count: int | None = None,
    max_subsets: int = 1 << 24,
    action_is_group: bool | None = None,
    backend: CIBackend = "auto",
) -> SubsetOrbitPartition:
    """Enumerate the complete powerset quotient when it fits a bounded batch."""
    if not isinstance(max_subsets, Integral) or int(max_subsets) <= 0:
        raise ValueError("max_subsets must be a positive integer")
    action = np.asarray(action_generators)
    if atom_count is None:
        if action.ndim != 2:
            raise ValueError(
                "atom_count is required when the action shape is ambiguous"
            )
        atom_count = int(action.shape[1])
    if not isinstance(atom_count, Integral) or not 0 <= int(atom_count) <= 62:
        raise ValueError("complete enumeration requires zero to 62 atoms")
    total = 1 << int(atom_count)
    if total > int(max_subsets):
        raise ValueError(
            f"complete powerset has {total} subsets, above max_subsets"
        )
    subsets = np.arange(total, dtype=np.uint64)
    return deduplicate_subset_orbits(
        subsets,
        action,
        atom_count=int(atom_count),
        action_is_group=action_is_group,
        backend=backend,
    )


def _fixed_weight_invariant_count(
    permutation: NDArray[np.uint32],
    subset_weight: int,
) -> int:
    unseen = set(range(len(permutation)))
    cycle_lengths: list[int] = []
    while unseen:
        point = min(unseen)
        length = 0
        while point in unseen:
            unseen.remove(point)
            point = int(permutation[point])
            length += 1
        cycle_lengths.append(length)
    coefficients = [0] * (subset_weight + 1)
    coefficients[0] = 1
    for length in cycle_lengths:
        for degree in range(subset_weight, length - 1, -1):
            coefficients[degree] += coefficients[degree - length]
    return coefficients[subset_weight]


def _fixed_weight_orbit_count(
    complete_action: NDArray[np.uint32],
    subset_weight: int,
) -> int:
    total = sum(
        _fixed_weight_invariant_count(permutation, subset_weight)
        for permutation in complete_action
    )
    if total % len(complete_action):
        raise RuntimeError("Burnside fixed-set sum is not divisible by action size")
    return total // len(complete_action)


def _fixed_weight_subset_orbits_reference(
    complete_action: NDArray[np.uint32],
    atom_count: int,
    subset_weight: int,
) -> tuple[NDArray[np.uint64], NDArray[np.uint64]]:
    seen: set[int] = set()
    representatives: list[int] = []
    sizes: list[int] = []
    image_bits = tuple(
        tuple(1 << int(image) for image in permutation)
        for permutation in complete_action
    )
    for subset in combinations(range(atom_count), subset_weight):
        mask = sum(1 << point for point in subset)
        if mask in seen:
            continue
        images = {
            sum(bits[point] for point in subset)
            for bits in image_bits
        }
        if images & seen:
            raise RuntimeError("complete action crossed an earlier orbit")
        representatives.append(mask)
        sizes.append(len(images))
        seen.update(images)
    if len(seen) != comb(atom_count, subset_weight):
        raise RuntimeError("fixed-weight orbit enumeration did not cover its domain")
    return (
        np.asarray(representatives, dtype=np.uint64),
        np.asarray(sizes, dtype=np.uint64),
    )


def enumerate_fixed_weight_subset_orbits(
    complete_action: ArrayLike,
    subset_weight: int,
    *,
    atom_count: int | None = None,
    max_subsets: int = 1 << 28,
    backend: CIBackend = "auto",
) -> FixedWeightSubsetOrbits:
    """Enumerate one representative per fixed-weight subset orbit.

    ``complete_action`` must contain every distinct element of a finite
    permutation group.  Unlike ``deduplicate_subset_orbits``, this route does
    not materialize the complete fixed-weight subset domain.
    """
    _validate_backend(backend)
    action_array = np.asarray(complete_action)
    if atom_count is None:
        if action_array.ndim != 2:
            raise ValueError(
                "atom_count is required when the action shape is ambiguous"
            )
        atom_count = int(action_array.shape[1])
    if not isinstance(atom_count, Integral) or not 1 <= int(atom_count) <= 64:
        raise ValueError("fixed-weight atom_count must be between one and 64")
    atom_count = int(atom_count)
    if not isinstance(subset_weight, Integral) or not 0 <= int(
        subset_weight
    ) <= atom_count:
        raise ValueError("subset_weight must be between zero and atom_count")
    subset_weight = int(subset_weight)
    if not isinstance(max_subsets, Integral) or int(max_subsets) <= 0:
        raise ValueError("max_subsets must be a positive integer")
    max_subsets = int(max_subsets)
    action = _prepare_action(action_array, atom_count)
    _is_complete_action_reference(action, atom_count, required=True)
    subset_count = comb(atom_count, subset_weight)
    if subset_count > max_subsets:
        raise ValueError(
            f"fixed-weight domain has {subset_count} subsets, above max_subsets"
        )
    expected_orbits = _fixed_weight_orbit_count(action, subset_weight)
    if backend != "reference":
        try:
            representatives, sizes, stats = fixed_weight_subset_orbits_native(
                action,
                atom_count,
                subset_weight,
                min(max_subsets, np.iinfo(np.uint64).max),
                expected_orbits,
            )
        except (NativeUnavailable, OSError, AttributeError):
            if backend == "native":
                raise
        else:
            if len(representatives) != expected_orbits:
                raise RuntimeError(
                    "native fixed-weight orbit count disagrees with Burnside"
                )
            return FixedWeightSubsetOrbits(
                representative_words=representatives.reshape((-1, 1)),
                orbit_sizes=sizes,
                atom_count=atom_count,
                subset_weight=subset_weight,
                subset_count=subset_count,
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    if backend == "native":
        raise NativeUnavailable(
            "fast-math native fixed-weight subset orbit kernel is unavailable"
        )
    started = perf_counter()
    representatives, sizes = _fixed_weight_subset_orbits_reference(
        action,
        atom_count,
        subset_weight,
    )
    if len(representatives) != expected_orbits:
        raise RuntimeError(
            "reference fixed-weight orbit count disagrees with Burnside"
        )
    return FixedWeightSubsetOrbits(
        representative_words=representatives.reshape((-1, 1)),
        orbit_sizes=sizes,
        atom_count=atom_count,
        subset_weight=subset_weight,
        subset_count=subset_count,
        elapsed_seconds=perf_counter() - started,
        backend="reference",
    )


def inverse_closed_atoms(
    inverse_indices: ArrayLike,
    *,
    identity: int = 0,
) -> InverseClosedAtoms:
    inverses = np.asarray(inverse_indices)
    if inverses.ndim != 1 or not np.issubdtype(inverses.dtype, np.integer):
        raise ValueError("inverse_indices must be a one-dimensional integer array")
    order = len(inverses)
    if not 0 <= identity < order:
        raise ValueError("identity is out of range")
    if np.any(inverses < 0) or np.any(inverses >= order):
        raise ValueError("inverse index is out of range")
    inverses = np.ascontiguousarray(inverses, dtype=np.uint32)
    if any(int(inverses[int(inverses[index])]) != index for index in range(order)):
        raise ValueError("inverse_indices is not an involution")
    unseen = set(range(order))
    unseen.remove(identity)
    atoms = []
    atom_of = np.full(order, -1, dtype=np.int32)
    while unseen:
        element = min(unseen)
        atom = np.asarray(
            sorted({element, int(inverses[element])}),
            dtype=np.uint32,
        )
        atom_of[atom] = len(atoms)
        atoms.append(atom)
        unseen.difference_update(map(int, atom))
    return InverseClosedAtoms(
        order=order,
        identity=int(identity),
        atoms=tuple(atoms),
        atom_of_element=atom_of,
    )


def induced_atom_action(
    atoms: Sequence[ArrayLike] | InverseClosedAtoms,
    element_generators: ArrayLike,
    *,
    group_order: int | None = None,
) -> NDArray[np.uint32]:
    prepared_atoms = tuple(
        np.ascontiguousarray(atom, dtype=np.uint32)
        for atom in atoms
    )
    if group_order is None:
        if isinstance(atoms, InverseClosedAtoms):
            group_order = atoms.order
        else:
            group_order = 1 + max(
                (int(atom.max()) for atom in prepared_atoms if len(atom)),
                default=-1,
            )
    element_action = _prepare_action(element_generators, int(group_order))
    atom_of = np.full(int(group_order), -1, dtype=np.int64)
    for atom_index, atom in enumerate(prepared_atoms):
        if np.any(atom >= group_order) or np.any(atom_of[atom] >= 0):
            raise ValueError("atoms overlap or contain an out-of-range element")
        atom_of[atom] = atom_index
    actions = []
    for generator in element_action:
        action = []
        for atom in prepared_atoms:
            images = {int(atom_of[int(generator[element])]) for element in atom}
            if len(images) != 1 or -1 in images:
                raise ValueError("element generator does not preserve the atoms")
            action.append(images.pop())
        actions.append(tuple(action))
    return np.asarray(sorted(set(actions)), dtype=np.uint32).reshape(
        (-1, len(atoms))
    )


def induced_atom_generators(
    atoms: InverseClosedAtoms,
    automorphism_generators: ArrayLike,
) -> NDArray[np.uint32]:
    return induced_atom_action(
        atoms,
        automorphism_generators,
        group_order=atoms.order,
    )


def pack_subsets(
    subsets: Iterable[Iterable[int]],
    universe_size: int,
) -> NDArray[np.uint64]:
    """Pack explicit subsets into little-endian uint64 words."""
    if not isinstance(universe_size, Integral) or not 0 <= int(
        universe_size
    ) <= 512:
        raise ValueError("universe_size must be between zero and 512")
    universe_size = int(universe_size)
    rows = [tuple(subset) for subset in subsets]
    output = np.zeros(
        (len(rows), (universe_size + 63) // 64),
        dtype=np.uint64,
    )
    for row_index, subset in enumerate(rows):
        seen: set[int] = set()
        for value in subset:
            if not isinstance(value, Integral) or not 0 <= int(
                value
            ) < universe_size:
                raise ValueError("subset contains an out-of-range element")
            value = int(value)
            if value in seen:
                raise ValueError("subset contains a duplicate element")
            seen.add(value)
            output[row_index, value // 64] |= np.uint64(
                1 << (value % 64)
            )
    return output


def atom_subsets_to_element_words(
    subset_words: ArrayLike,
    atoms: Sequence[ArrayLike] | InverseClosedAtoms,
    *,
    group_order: int,
    threads: int = 0,
    backend: CIBackend = "auto",
) -> NDArray[np.uint64]:
    _validate_backend(backend)
    if not isinstance(group_order, Integral) or not 1 <= int(
        group_order
    ) <= 512:
        raise ValueError("group_order must be an integer between one and 512")
    if not isinstance(threads, Integral) or int(threads) < 0:
        raise ValueError("threads must be a nonnegative integer")
    group_order = int(group_order)
    subsets = _prepare_subset_words(
        subset_words,
        len(atoms),
        require_unique=False,
    )
    atom_rows: list[NDArray[np.uint32]] = []
    offsets = np.zeros(len(atoms) + 1, dtype=np.uint64)
    for atom_index, atom_values in enumerate(atoms):
        values = np.asarray(atom_values)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("atoms must be one-dimensional integer arrays")
        if np.any(values < 0) or np.any(values >= group_order):
            raise ValueError("atom contains an out-of-range element")
        row = np.ascontiguousarray(values, dtype=np.uint32)
        atom_rows.append(row)
        offsets[atom_index + 1] = offsets[atom_index] + len(row)
    elements = np.ascontiguousarray(
        np.concatenate(atom_rows)
        if atom_rows
        else np.empty(0, dtype=np.uint32),
        dtype=np.uint32,
    )
    if backend != "reference":
        try:
            element_words, _ = expand_atom_subsets_native(
                subsets,
                offsets,
                elements,
                group_order,
                int(threads),
            )
        except (NativeUnavailable, OSError, AttributeError):
            if backend == "native":
                raise
        else:
            return element_words
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    element_words = np.zeros(
        (len(subsets), (group_order + 63) // 64),
        dtype=np.uint64,
    )
    for subset_index, subset in enumerate(subsets):
        for atom_index, atom_values in enumerate(atom_rows):
            if not int(subset[atom_index // 64]) & (1 << (atom_index % 64)):
                continue
            for element in atom_values:
                value = int(element)
                element_words[subset_index, value // 64] |= np.uint64(
                    1 << (value % 64)
                )
    return element_words


def expand_atom_masks(
    atom_masks: ArrayLike,
    atoms: InverseClosedAtoms,
    *,
    threads: int = 0,
    backend: CIBackend = "auto",
) -> NDArray[np.uint64]:
    return atom_subsets_to_element_words(
        atom_masks,
        atoms,
        group_order=atoms.order,
        threads=threads,
        backend=backend,
    )


def _prepare_group_table(
    multiplication_table: ArrayLike,
) -> NDArray[np.uint32]:
    table = np.asarray(multiplication_table)
    if table.ndim != 2 or table.shape[0] != table.shape[1]:
        raise ValueError("multiplication_table must be square")
    order = len(table)
    if not 1 <= order <= 512:
        raise ValueError("group order must be between one and 512")
    if not np.issubdtype(table.dtype, np.integer):
        raise ValueError("multiplication_table must contain integers")
    if np.any(table < 0) or np.any(table >= order):
        raise ValueError("multiplication table entry is out of range")
    return np.ascontiguousarray(table, dtype=np.uint32)


def _prepare_connections(
    connection_sets: ArrayLike,
    order: int,
) -> NDArray[np.uint64]:
    return _prepare_subset_words(
        connection_sets,
        order,
        require_unique=False,
    )


def _validate_connection_sets(
    connections: NDArray[np.uint64],
    inverse_indices: ArrayLike | None,
    order: int,
    identity: int,
    require_inverse_closed: bool,
) -> NDArray[np.uint32] | None:
    if np.any(
        connections[:, identity // 64] & np.uint64(1 << (identity % 64))
    ):
        raise ValueError("connection sets must exclude the identity")
    if inverse_indices is None:
        return None
    inverses = np.asarray(inverse_indices)
    if inverses.shape != (order,) or not np.issubdtype(
        inverses.dtype, np.integer
    ):
        raise ValueError(
            "inverse_indices must contain one integer per group element"
        )
    if np.any(inverses < 0) or np.any(inverses >= order):
        raise ValueError("inverse index is out of range")
    prepared = np.ascontiguousarray(inverses, dtype=np.uint32)
    if require_inverse_closed:
        elements = np.arange(order, dtype=np.uint32)
        word_indices = elements // np.uint32(64)
        shifts = np.asarray(elements % np.uint32(64), dtype=np.uint64)
        rows_per_chunk = max(
            1,
            (32 << 20)
            // (order * np.dtype(np.uint64).itemsize),
        )
        for begin in range(0, len(connections), rows_per_chunk):
            chunk = connections[begin : begin + rows_per_chunk]
            present = (
                (chunk[:, word_indices] >> shifts[np.newaxis, :])
                & np.uint64(1)
            ).astype(np.bool_)
            if np.any(present != present[:, prepared]):
                raise ValueError("connection set is not inverse-closed")
    return prepared


def _cayley_graphs_reference(
    table: NDArray[np.uint32],
    connections: NDArray[np.uint64],
) -> NDArray[np.uint64]:
    order = len(table)
    adjacency = np.zeros(
        (len(connections), order, connections.shape[1]),
        dtype=np.uint64,
    )
    for graph_index, connection in enumerate(connections):
        for step in range(order):
            if not int(connection[step // 64]) & (1 << (step % 64)):
                continue
            for vertex in range(order):
                neighbor = int(table[step, vertex])
                adjacency[graph_index, vertex, neighbor // 64] |= np.uint64(
                    1 << (neighbor % 64)
                )
    return adjacency


def _validate_undirected_adjacency(
    adjacency: NDArray[np.uint64],
) -> None:
    for graph in adjacency:
        order = len(graph)
        for left in range(order):
            for right in range(order):
                forward = bool(
                    int(graph[left, right // 64]) & (1 << (right % 64))
                )
                reverse = bool(
                    int(graph[right, left // 64]) & (1 << (left % 64))
                )
                if forward != reverse:
                    raise ValueError("connection set is not inverse-closed")


def cayley_graphs(
    multiplication_table: ArrayLike,
    connection_sets: ArrayLike,
    *,
    inverse_indices: ArrayLike | None = None,
    identity: int = 0,
    require_inverse_closed: bool = True,
    require_undirected: bool | None = None,
    threads: int = 0,
    backend: CIBackend = "auto",
) -> CayleyGraphBatch:
    """Construct a batch of Cayley digraphs with arcs ``(g, s*g)``."""
    _validate_backend(backend)
    if not isinstance(threads, Integral) or int(threads) < 0:
        raise ValueError("threads must be a nonnegative integer")
    if require_undirected is not None:
        require_inverse_closed = bool(require_undirected)
    table = _prepare_group_table(multiplication_table)
    if not isinstance(identity, Integral) or not 0 <= int(identity) < len(table):
        raise ValueError("identity is out of range")
    connections = _prepare_connections(connection_sets, len(table))
    _validate_connection_sets(
        connections,
        inverse_indices,
        len(table),
        int(identity),
        require_inverse_closed,
    )
    if backend != "reference":
        try:
            adjacency, stats = cayley_graphs_native(
                table,
                connections,
                int(threads),
            )
        except (NativeUnavailable, OSError, AttributeError):
            if backend == "native":
                raise
        else:
            if require_inverse_closed and inverse_indices is None:
                _validate_undirected_adjacency(adjacency)
            return CayleyGraphBatch(
                adjacency_words=adjacency,
                connection_words=connections,
                group_order=len(table),
                graph_count=len(connections),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    started = perf_counter()
    adjacency = _cayley_graphs_reference(table, connections)
    if require_inverse_closed and inverse_indices is None:
        _validate_undirected_adjacency(adjacency)
    return CayleyGraphBatch(
        adjacency_words=adjacency,
        connection_words=connections,
        group_order=len(table),
        graph_count=len(connections),
        elapsed_seconds=perf_counter() - started,
        backend="reference",
    )


def canonicalize_cayley_graphs(
    multiplication_table: ArrayLike,
    connection_sets: ArrayLike,
    *,
    inverse_indices: ArrayLike | None = None,
    identity: int = 0,
    require_inverse_closed: bool = True,
    require_undirected: bool | None = None,
    threads: int = 0,
    collect_automorphism_generators: bool = True,
    graph_backend: CIBackend = "auto",
    construction_backend: CIBackend | None = None,
    canonical_backend: CIBackend = "auto",
) -> CanonicalCayleyBatch:
    if require_undirected is not None:
        require_inverse_closed = bool(require_undirected)
    if construction_backend is not None:
        if graph_backend != "auto" and graph_backend != construction_backend:
            raise ValueError("graph backend aliases disagree")
        graph_backend = construction_backend
    graphs = cayley_graphs(
        multiplication_table,
        connection_sets,
        inverse_indices=inverse_indices,
        identity=identity,
        require_inverse_closed=require_inverse_closed,
        threads=threads,
        backend=graph_backend,
    )
    colors = np.zeros(
        (graphs.graph_count, graphs.group_order),
        dtype=np.uint32,
    )
    canonical = canonicalize_colored_digraphs(
        graphs.adjacency_words,
        colors,
        threads=threads,
        collect_automorphism_generators=(
            collect_automorphism_generators
        ),
        backend=canonical_backend,
    )
    return CanonicalCayleyBatch(graphs=graphs, canonical=canonical)


def _prepare_inverses(
    inverse_indices: ArrayLike,
    order: int,
) -> NDArray[np.uint32]:
    inverses = np.asarray(inverse_indices)
    if inverses.shape != (order,) or not np.issubdtype(
        inverses.dtype, np.integer
    ):
        raise ValueError("inverse_indices must have shape (group_order,)")
    if np.any(inverses < 0) or np.any(inverses >= order):
        raise ValueError("inverse index is out of range")
    prepared = np.ascontiguousarray(inverses, dtype=np.uint32)
    if any(int(prepared[int(prepared[index])]) != index for index in range(order)):
        raise ValueError("inverse_indices is not an involution")
    return prepared


def _prepare_bijection(
    bijection: ArrayLike,
    order: int,
) -> NDArray[np.uint32]:
    mapping = _prepare_action(bijection, order)
    if len(mapping) != 1:
        raise ValueError("bijection must contain one permutation")
    return mapping[0]


def _derivative_reference(
    table: NDArray[np.uint32],
    inverses: NDArray[np.uint32],
    mapping: NDArray[np.uint32],
) -> tuple[NDArray[np.uint32], NDArray[np.uint32], int]:
    order = len(table)
    mapping_inverse = np.empty(order, dtype=np.uint32)
    mapping_inverse[mapping] = np.arange(order, dtype=np.uint32)
    generators = np.empty((order, order), dtype=np.uint32)
    for vertex in range(order):
        mapped_vertex_inverse = int(inverses[int(mapping[vertex])])
        for connection in range(order):
            product = int(table[connection, vertex])
            difference = int(
                table[int(mapping[product]), mapped_vertex_inverse]
            )
            generators[vertex, connection] = mapping_inverse[difference]
    labels = np.full(order, np.iinfo(np.uint32).max, dtype=np.uint32)
    orbit_count = 0
    for seed in range(order):
        if labels[seed] != np.iinfo(np.uint32).max:
            continue
        labels[seed] = orbit_count
        queue = deque([seed])
        while queue:
            point = queue.popleft()
            for generator in generators:
                image = int(generator[point])
                if labels[image] == np.iinfo(np.uint32).max:
                    labels[image] = orbit_count
                    queue.append(image)
        orbit_count += 1
    return generators, labels, orbit_count


def derivative_group_orbits(
    multiplication_table: ArrayLike,
    inverse_indices: ArrayLike,
    bijection: ArrayLike,
    *,
    identity: int = 0,
    backend: CIBackend = "auto",
) -> DerivativeOrbitPartition:
    """Compute the R-0805 relative derivative group and its point orbits."""
    _validate_backend(backend)
    table = _prepare_group_table(multiplication_table)
    inverses = _prepare_inverses(inverse_indices, len(table))
    mapping = _prepare_bijection(bijection, len(table))
    if not isinstance(identity, Integral) or not 0 <= int(identity) < len(table):
        raise ValueError("identity is out of range")
    if int(mapping[int(identity)]) != int(identity):
        raise ValueError("derivative bijections must fix the identity")
    if backend != "reference":
        try:
            generators, labels, count, stats = derivative_orbits_native(
                table,
                inverses,
                mapping,
            )
        except (NativeUnavailable, OSError, AttributeError):
            if backend == "native":
                raise
        else:
            return DerivativeOrbitPartition(
                generators=generators,
                orbit_labels=labels,
                orbits=tuple(
                    np.flatnonzero(labels == orbit).astype(np.uint32)
                    for orbit in range(count)
                ),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    started = perf_counter()
    generators, labels, count = _derivative_reference(
        table,
        inverses,
        mapping,
    )
    return DerivativeOrbitPartition(
        generators=generators,
        orbit_labels=labels,
        orbits=tuple(
            np.flatnonzero(labels == orbit).astype(np.uint32)
            for orbit in range(count)
        ),
        elapsed_seconds=perf_counter() - started,
        backend="reference",
    )


def derivative_orbit_partitions(
    multiplication_table: ArrayLike,
    inverse_indices: ArrayLike,
    bijections: ArrayLike,
    *,
    identity: int = 0,
    threads: int = 0,
    backend: CIBackend = "auto",
) -> DerivativeOrbitBatch:
    """Compute R-0805 derivative orbit partitions for a batch of bijections."""
    if not isinstance(threads, Integral) or int(threads) < 0:
        raise ValueError("threads must be a nonnegative integer")
    table = _prepare_group_table(multiplication_table)
    prepared = _prepare_action(bijections, len(table))
    labels = np.empty((len(prepared), len(table)), dtype=np.uint32)
    counts = np.empty(len(prepared), dtype=np.uint32)
    elapsed = 0.0
    selected_backends = set()
    for index, mapping in enumerate(prepared):
        result = derivative_group_orbits(
            table,
            inverse_indices,
            mapping,
            identity=identity,
            backend=backend,
        )
        labels[index] = result.orbit_labels
        counts[index] = len(result.orbits)
        elapsed += result.elapsed_seconds
        selected_backends.add(result.backend)
    selected = selected_backends.pop() if len(selected_backends) == 1 else "mixed"
    return DerivativeOrbitBatch(
        orbit_labels=labels,
        orbit_counts=counts,
        elapsed_seconds=elapsed,
        backend=selected,
    )


def double_cosets(
    elements: ArrayLike,
    left_generators: ArrayLike,
    right_generators: ArrayLike,
    *,
    degree: int | None = None,
    backend: CIBackend = "auto",
) -> PermutationDoubleCosetPartition:
    if degree is not None:
        candidates = np.asarray(elements)
        if candidates.ndim != 2 or candidates.shape[1] != int(degree):
            raise ValueError("candidate degree does not match degree")
    return permutation_double_cosets(
        elements,
        left_generators,
        right_generators,
        backend=backend,
    )


def generalized_dihedral_group(
    moduli: Sequence[int],
) -> GeneralizedDihedralGroup:
    """Build ``Dih(product C_moduli)`` with the atlas left-action convention."""
    normalized = tuple(int(modulus) for modulus in moduli)
    if not normalized or any(modulus < 2 for modulus in normalized):
        raise ValueError("moduli must be a nonempty sequence of integers >= 2")
    vectors = list(product(*(range(modulus) for modulus in normalized)))
    if 2 * len(vectors) > 512:
        raise ValueError("generalized dihedral group order exceeds 512")
    elements = tuple(
        (tuple(vector), layer)
        for layer in (0, 1)
        for vector in vectors
    )
    index = {element: position for position, element in enumerate(elements)}

    def add(
        left: tuple[int, ...],
        right: tuple[int, ...],
    ) -> tuple[int, ...]:
        return tuple(
            (left[position] + right[position]) % normalized[position]
            for position in range(len(normalized))
        )

    def negate(value: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            (-value[position]) % normalized[position]
            for position in range(len(normalized))
        )

    def multiply(
        left: tuple[tuple[int, ...], int],
        right: tuple[tuple[int, ...], int],
    ) -> tuple[tuple[int, ...], int]:
        signed_right = right[0] if left[1] == 0 else negate(right[0])
        return (add(left[0], signed_right), (left[1] + right[1]) % 2)

    table = np.asarray(
        [
            [index[multiply(left, right)] for right in elements]
            for left in elements
        ],
        dtype=np.uint32,
    )
    zero = (0,) * len(normalized)
    identity = index[(zero, 0)]
    inverses = np.empty(len(elements), dtype=np.uint32)
    for element_index in range(len(elements)):
        candidates = np.flatnonzero(
            (table[element_index] == identity)
            & (table[:, element_index] == identity)
        )
        if len(candidates) != 1:
            raise AssertionError("generalized-dihedral inverse is not unique")
        inverses[element_index] = candidates[0]
    return GeneralizedDihedralGroup(
        moduli=normalized,
        elements=elements,
        multiplication_table=table,
        inverse_indices=inverses,
        identity=identity,
        abelian_order=len(vectors),
    )


def generalized_dihedral_automorphisms(
    group: GeneralizedDihedralGroup,
    abelian_automorphisms: ArrayLike,
    *,
    include_all_translations: bool = True,
) -> NDArray[np.uint32]:
    """Lift an explicit action on ``A`` to the affine Hol(A) action on Dih(A)."""
    abelian = _prepare_action(
        abelian_automorphisms,
        group.abelian_order,
    )
    vectors = [element[0] for element in group.elements[: group.abelian_order]]
    vector_index = {vector: index for index, vector in enumerate(vectors)}
    zero = (0,) * len(group.moduli)

    def add(
        left: tuple[int, ...],
        right: tuple[int, ...],
    ) -> tuple[int, ...]:
        return tuple(
            (left[position] + right[position]) % group.moduli[position]
            for position in range(len(group.moduli))
        )

    translations = vectors if include_all_translations else [zero]
    element_index = {
        element: index for index, element in enumerate(group.elements)
    }
    for automorphism in abelian:
        if int(automorphism[group.identity]) != group.identity:
            raise ValueError("abelian automorphisms must fix the identity")
        for left in range(group.abelian_order):
            for right in range(group.abelian_order):
                product_index = int(group.multiplication_table[left, right])
                if int(automorphism[product_index]) != int(
                    group.multiplication_table[
                        int(automorphism[left]),
                        int(automorphism[right]),
                    ]
                ):
                    raise ValueError(
                        "abelian action contains a non-automorphism"
                    )
    output = []
    for automorphism in abelian:
        for translation in translations:
            image = []
            for vector, layer in group.elements:
                transformed = vectors[int(automorphism[vector_index[vector]])]
                if layer:
                    transformed = add(transformed, translation)
                image.append(element_index[(transformed, layer)])
            output.append(tuple(image))
    return np.asarray(sorted(set(output)), dtype=np.uint32)


def _prepare_relations(initial_relations: ArrayLike) -> NDArray[np.uint32]:
    relations = np.asarray(initial_relations)
    if relations.ndim != 2 or relations.shape[0] != relations.shape[1]:
        raise ValueError("initial_relations must be square")
    if not 1 <= len(relations) <= 512:
        raise ValueError("relation order must be between one and 512")
    if not np.issubdtype(relations.dtype, np.integer):
        raise ValueError("initial_relations must contain integers")
    if np.issubdtype(relations.dtype, np.signedinteger) and np.any(relations < 0):
        raise ValueError("initial_relations must be nonnegative")
    values = sorted(set(map(int, relations.flat)))
    mapping = {value: index for index, value in enumerate(values)}
    return np.ascontiguousarray(
        np.vectorize(mapping.__getitem__, otypes=[np.uint32])(relations),
        dtype=np.uint32,
    )


def _wl2_reference(
    relations: NDArray[np.uint32],
) -> tuple[NDArray[np.uint32], int]:
    stable = relations.copy()
    iterations = 0
    while True:
        relation_count = int(stable.max()) + 1
        signatures = []
        for left in range(len(stable)):
            for right in range(len(stable)):
                counts = Counter(
                    (
                        int(stable[left, middle]),
                        int(stable[middle, right]),
                    )
                    for middle in range(len(stable))
                )
                signatures.append(
                    (
                        int(stable[left, right]),
                        tuple(sorted(counts.items())),
                    )
                )
        unique = {
            signature: index
            for index, signature in enumerate(sorted(set(signatures)))
        }
        refined = np.asarray(
            [unique[signature] for signature in signatures],
            dtype=np.uint32,
        ).reshape(stable.shape)
        iterations += 1
        if np.array_equal(refined, stable):
            return refined, iterations
        stable = refined


def _intersection_numbers_reference(
    relations: NDArray[np.uint32],
    relation_count: int,
) -> NDArray[np.uint64]:
    tensor = np.zeros(
        (relation_count, relation_count, relation_count),
        dtype=np.uint64,
    )
    representatives = [
        tuple(map(int, np.argwhere(relations == relation)[0]))
        for relation in range(relation_count)
    ]
    for target, (left, right) in enumerate(representatives):
        for middle in range(len(relations)):
            tensor[
                int(relations[left, middle]),
                int(relations[middle, right]),
                target,
            ] += 1
    for left in range(len(relations)):
        for right in range(len(relations)):
            target = int(relations[left, right])
            counts = np.zeros((relation_count, relation_count), dtype=np.uint64)
            for middle in range(len(relations)):
                counts[
                    int(relations[left, middle]),
                    int(relations[middle, right]),
                ] += 1
            if not np.array_equal(counts, tensor[:, :, target]):
                raise ValueError("relation partition is not coherent")
    return tensor


def wl2_refinement(
    initial_relations: ArrayLike,
    *,
    backend: CIBackend = "auto",
) -> WL2Refinement:
    """Run exact stable 2-WL without constructing intersection numbers."""
    _validate_backend(backend)
    relations = _prepare_relations(initial_relations)
    started = perf_counter()
    if backend != "reference":
        try:
            stable, relation_count, refine_stats = wl2_refine_native(relations)
        except (NativeUnavailable, OSError, AttributeError):
            if backend == "native":
                raise
        else:
            return WL2Refinement(
                relations=stable,
                relation_count=relation_count,
                relation_sizes=np.bincount(
                    stable.ravel(),
                    minlength=relation_count,
                ).astype(np.uint64),
                iterations=int(refine_stats.iteration_count),
                elapsed_seconds=float(refine_stats.elapsed_seconds),
                backend="native",
            )
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    stable, iterations = _wl2_reference(relations)
    relation_count = int(stable.max()) + 1
    return WL2Refinement(
        relations=stable,
        relation_count=relation_count,
        relation_sizes=np.bincount(
            stable.ravel(),
            minlength=relation_count,
        ).astype(np.uint64),
        iterations=iterations,
        elapsed_seconds=perf_counter() - started,
        backend="reference",
    )


def coherent_configuration(
    initial_relations: ArrayLike,
    *,
    max_tensor_entries: int = 100_000_000,
    backend: CIBackend = "auto",
) -> CoherentConfiguration:
    """Run stable 2-WL and verify all intersection numbers."""
    refinement = wl2_refinement(initial_relations, backend=backend)
    entries = refinement.relation_count**3
    if entries > max_tensor_entries:
        raise ValueError(f"intersection tensor needs {entries} entries")
    if refinement.backend == "native":
        tensor, intersection_stats = intersection_numbers_native(
            refinement.relations,
            refinement.relation_count,
        )
        intersection_seconds = float(
            intersection_stats.elapsed_seconds
        )
    else:
        started = perf_counter()
        tensor = _intersection_numbers_reference(
            refinement.relations,
            refinement.relation_count,
        )
        intersection_seconds = perf_counter() - started
    return CoherentConfiguration(
        relations=refinement.relations,
        relation_count=refinement.relation_count,
        relation_sizes=refinement.relation_sizes,
        intersection_numbers=tensor,
        iterations=refinement.iterations,
        elapsed_seconds=(
            refinement.elapsed_seconds + intersection_seconds
        ),
        backend=refinement.backend,
    )


def _graph_initial_relations(
    adjacency_words: ArrayLike,
) -> NDArray[np.uint32]:
    adjacency = np.asarray(adjacency_words)
    if adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1]:
        if not np.all((adjacency == 0) | (adjacency == 1)):
            raise ValueError("adjacency matrix must be Boolean")
        present = np.asarray(adjacency, dtype=np.bool_)
    elif adjacency.ndim == 2:
        vertex_count, word_count = adjacency.shape
        if word_count != (vertex_count + 63) // 64:
            raise ValueError("adjacency words have an invalid packed shape")
        packed = np.ascontiguousarray(adjacency, dtype=np.uint64)
        present = np.zeros((vertex_count, vertex_count), dtype=np.bool_)
        for right in range(vertex_count):
            present[:, right] = (
                packed[:, right // 64] >> np.uint64(right % 64)
            ) & np.uint64(1)
    else:
        raise ValueError("adjacency must be a square matrix or packed rows")
    if np.any(np.diag(present)):
        raise ValueError("graphs must be loopless")
    relations = np.full(present.shape, 2, dtype=np.uint32)
    relations[present] = 1
    np.fill_diagonal(relations, 0)
    return relations


def graph_wl2_refinement(
    adjacency_words: ArrayLike,
    *,
    backend: CIBackend = "auto",
) -> WL2Refinement:
    """Run exact stable 2-WL on one loopless graph."""
    return wl2_refinement(
        _graph_initial_relations(adjacency_words),
        backend=backend,
    )


def graph_coherent_configuration(
    adjacency_words: ArrayLike,
    *,
    max_tensor_entries: int = 100_000_000,
    backend: CIBackend = "auto",
) -> CoherentConfiguration:
    """Generate the coherent configuration of one loopless graph."""
    return coherent_configuration(
        _graph_initial_relations(adjacency_words),
        max_tensor_entries=max_tensor_entries,
        backend=backend,
    )


__all__ = [
    "CIBackend",
    "CanonicalCayleyBatch",
    "CayleyGraphBatch",
    "CoherentConfiguration",
    "DerivativeOrbitBatch",
    "DerivativeOrbitPartition",
    "GeneralizedDihedralGroup",
    "InverseClosedAtoms",
    "PermutationDoubleCosetPartition",
    "SubsetOrbitPartition",
    "WL2Refinement",
    "atom_subsets_to_element_words",
    "compose_u64_mask_luts",
    "canonicalize_cayley_graphs",
    "cayley_graphs",
    "coherent_configuration",
    "deduplicate_subset_orbits",
    "derivative_group_orbits",
    "derivative_orbit_partitions",
    "double_cosets",
    "enumerate_subset_orbits",
    "expand_atom_masks",
    "generalized_dihedral_automorphisms",
    "generalized_dihedral_group",
    "graph_coherent_configuration",
    "graph_wl2_refinement",
    "induced_atom_action",
    "induced_atom_generators",
    "inverse_closed_atoms",
    "pack_subsets",
    "permutation_double_cosets",
    "u64_mask_lut",
    "wl2_refinement",
]
