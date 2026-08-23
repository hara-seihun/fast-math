from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import reduce
from numbers import Integral
from operator import mul
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._native import NativeUnavailable

from ._groups_native import (
    NativePermutationGroupPlan,
    permutation_group_contains_native,
    permutation_orbits_native,
    schreier_sims_native,
)
from ._ci_native import permutation_double_cosets_native


GroupBackend = Literal["auto", "native", "reference"]
Permutation = tuple[int, ...]


@dataclass(frozen=True)
class SchreierSimsChain:
    degree: int
    generators: NDArray[np.uint32]
    base: NDArray[np.uint32]
    orbit_sizes: NDArray[np.uint32]
    level_generator_offsets: NDArray[np.uint64]
    strong_generators: NDArray[np.uint32]
    order: int
    elapsed_seconds: float
    backend: str

    def contains(
        self,
        elements: ArrayLike,
        *,
        threads: int = 0,
        backend: GroupBackend | None = None,
    ) -> NDArray[np.bool_]:
        selected = self.backend if backend is None else backend
        return permutation_group_contains(
            self.generators,
            elements,
            degree=self.degree,
            threads=threads,
            backend=selected,
        )


@dataclass(frozen=True)
class DoubleCosetPartition:
    class_ids: NDArray[np.uint64]
    representative_indices: NDArray[np.uint64]
    representatives: NDArray[np.uint32]
    class_sizes: NDArray[np.uint64]
    elapsed_seconds: float
    backend: str


def _validate_backend(backend: str) -> None:
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")


def _prepare_permutations(
    permutations: ArrayLike,
    degree: int | None,
    *,
    name: str,
) -> tuple[NDArray[np.uint32], int]:
    array = np.asarray(permutations)
    if array.ndim == 1 and array.size:
        array = array[np.newaxis, :]
    if array.ndim == 1 and array.size == 0:
        if degree is None:
            raise ValueError(f"degree is required when {name} is empty")
        array = np.empty((0, degree), dtype=np.uint32)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (count, degree)")
    inferred_degree = array.shape[1]
    if degree is None:
        degree = inferred_degree
    if not isinstance(degree, Integral) or not 1 <= int(degree) <= 4096:
        raise ValueError("degree must be an integer between one and 4096")
    degree = int(degree)
    if inferred_degree != degree:
        raise ValueError(f"{name} degree does not match degree")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    if np.issubdtype(array.dtype, np.signedinteger) and np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    if array.size and np.any(array >= degree):
        raise ValueError(f"{name} contains an out-of-range image")
    prepared = np.ascontiguousarray(array, dtype=np.uint32)
    expected = np.arange(degree, dtype=np.uint32)
    for permutation in prepared:
        if not np.array_equal(np.sort(permutation), expected):
            raise ValueError(f"{name} contains a row that is not a permutation")
    return prepared, degree


def compose_permutations(left: ArrayLike, right: ArrayLike) -> NDArray[np.uint32]:
    """Return ``left after right`` using image-array permutations."""
    left_array, degree = _prepare_permutations(left, None, name="left")
    right_array, right_degree = _prepare_permutations(
        right, degree, name="right"
    )
    if len(left_array) != 1 or len(right_array) != 1:
        raise ValueError("left and right must each be one permutation")
    if degree != right_degree:
        raise ValueError("permutation degrees differ")
    return left_array[0][right_array[0]]


def invert_permutation(permutation: ArrayLike) -> NDArray[np.uint32]:
    """The inverse of one permutation."""
    array, degree = _prepare_permutations(
        permutation, None, name="permutation"
    )
    if len(array) != 1:
        raise ValueError("permutation must contain one row")
    inverse = np.empty(degree, dtype=np.uint32)
    inverse[array[0]] = np.arange(degree, dtype=np.uint32)
    return inverse


def _tuple_compose(left: Permutation, right: Permutation) -> Permutation:
    return tuple(left[right[point]] for point in range(len(left)))


def _tuple_inverse(permutation: Permutation) -> Permutation:
    result = [0] * len(permutation)
    for point, image in enumerate(permutation):
        result[image] = point
    return tuple(result)


def _deduplicate(
    permutations: list[Permutation],
    *,
    remove_identity: bool,
) -> list[Permutation]:
    if not permutations:
        return []
    identity = tuple(range(len(permutations[0])))
    return sorted(
        {
            permutation
            for permutation in permutations
            if not remove_identity or permutation != identity
        }
    )


def _symmetric_generators(
    generators: list[Permutation],
) -> list[Permutation]:
    return sorted(
        set(generators)
        | {_tuple_inverse(generator) for generator in generators}
    )


@dataclass(frozen=True)
class _ReferenceLevel:
    base: int
    generators: tuple[Permutation, ...]
    orbit: tuple[int, ...]
    orbit_index: tuple[int, ...]
    transversals: tuple[Permutation, ...]
    inverse_transversals: tuple[Permutation, ...]


def _reference_chain(
    generators: NDArray[np.uint32],
    degree: int,
) -> list[_ReferenceLevel]:
    current = _deduplicate(
        [tuple(map(int, generator)) for generator in generators],
        remove_identity=True,
    )
    identity = tuple(range(degree))
    levels: list[_ReferenceLevel] = []
    while current:
        base = next(
            point
            for point in range(degree)
            if any(generator[point] != point for generator in current)
        )
        symmetric = _symmetric_generators(current)
        orbit = [base]
        orbit_index = [-1] * degree
        orbit_index[base] = 0
        transversals = [identity]
        head = 0
        while head < len(orbit):
            point = orbit[head]
            transversal = transversals[head]
            head += 1
            for generator in symmetric:
                image = generator[point]
                if orbit_index[image] >= 0:
                    continue
                orbit_index[image] = len(orbit)
                orbit.append(image)
                transversals.append(
                    _tuple_compose(generator, transversal)
                )
        inverse_transversals = [
            _tuple_inverse(transversal)
            for transversal in transversals
        ]
        level = _ReferenceLevel(
            base=base,
            generators=tuple(current),
            orbit=tuple(orbit),
            orbit_index=tuple(orbit_index),
            transversals=tuple(transversals),
            inverse_transversals=tuple(inverse_transversals),
        )
        levels.append(level)
        schreier = []
        for orbit_position, point in enumerate(orbit):
            transversal = transversals[orbit_position]
            for generator in symmetric:
                image = generator[point]
                image_position = orbit_index[image]
                schreier.append(
                    _tuple_compose(
                        inverse_transversals[image_position],
                        _tuple_compose(generator, transversal),
                    )
                )
        current = _deduplicate(schreier, remove_identity=True)
    return levels


def _reference_contains_levels(
    levels: list[_ReferenceLevel],
    elements: NDArray[np.uint32],
    degree: int,
) -> NDArray[np.bool_]:
    identity = tuple(range(degree))
    output = np.empty(len(elements), dtype=np.bool_)
    for index, row in enumerate(elements):
        residual = tuple(map(int, row))
        for level in levels:
            image = residual[level.base]
            orbit_position = level.orbit_index[image]
            if orbit_position < 0:
                output[index] = False
                break
            residual = _tuple_compose(
                level.inverse_transversals[orbit_position],
                residual,
            )
        else:
            output[index] = residual == identity
    return output


class PermutationGroup:
    """Reusable exact permutation group with one retained stabilizer chain."""

    def __init__(
        self,
        generators: ArrayLike,
        *,
        degree: int | None = None,
        backend: GroupBackend = "auto",
    ) -> None:
        _validate_backend(backend)
        prepared, degree = _prepare_permutations(
            generators, degree, name="generators"
        )
        self.degree = degree
        self.generators = prepared.copy()
        self._native: NativePermutationGroupPlan | None = None
        self._reference_levels: list[_ReferenceLevel] | None = None
        self._closed = False

        if backend != "reference":
            try:
                native = NativePermutationGroupPlan(prepared, degree)
            except (NativeUnavailable, OSError, AttributeError):
                if backend == "native":
                    raise
            else:
                self._native = native
                self.base = native.base.copy()
                self.orbit_sizes = native.orbit_sizes.copy()
                self.orbits = tuple(
                    np.flatnonzero(native.point_orbit_labels == orbit).astype(
                        np.uint32
                    )
                    for orbit in range(native.point_orbit_count)
                )
                self.order = reduce(mul, map(int, self.orbit_sizes), 1)
                self.elapsed_seconds = native.elapsed_seconds
                self.backend = "native"
                return
        if backend == "native":
            raise NativeUnavailable("fast-math native library is unavailable")

        from time import perf_counter

        started = perf_counter()
        levels = _reference_chain(prepared, degree)
        self._reference_levels = levels
        self.base = np.asarray(
            [level.base for level in levels], dtype=np.uint32
        )
        self.orbit_sizes = np.asarray(
            [len(level.orbit) for level in levels], dtype=np.uint32
        )
        self.orbits = permutation_orbits(
            prepared, degree=degree, backend="reference"
        )
        self.order = reduce(mul, map(int, self.orbit_sizes), 1)
        self.elapsed_seconds = perf_counter() - started
        self.backend = "reference"

    @property
    def closed(self) -> bool:
        return self._closed

    def contains(
        self,
        elements: ArrayLike,
        *,
        threads: int = 0,
    ) -> NDArray[np.bool_]:
        if self._closed:
            raise RuntimeError("permutation group is closed")
        if not isinstance(threads, Integral) or int(threads) < 0:
            raise ValueError("threads must be a nonnegative integer")
        candidates, _ = _prepare_permutations(
            elements, self.degree, name="elements"
        )
        if self._native is not None:
            output, _ = self._native.contains(candidates, int(threads))
            return output
        if self._reference_levels is None:
            raise RuntimeError("permutation group has no stabilizer chain")
        return _reference_contains_levels(
            self._reference_levels, candidates, self.degree
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._native is not None:
            self._native.close()
            self._native = None
        self._reference_levels = None
        self._closed = True

    def __enter__(self) -> PermutationGroup:
        if self._closed:
            raise RuntimeError("permutation group is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.close()


def _chain_result_reference(
    generators: NDArray[np.uint32],
    degree: int,
) -> SchreierSimsChain:
    from time import perf_counter

    started = perf_counter()
    levels = _reference_chain(generators, degree)
    offsets = [0]
    strong: list[Permutation] = []
    for level in levels:
        strong.extend(level.generators)
        offsets.append(len(strong))
    orbit_sizes = np.asarray(
        [len(level.orbit) for level in levels],
        dtype=np.uint32,
    )
    return SchreierSimsChain(
        degree=degree,
        generators=generators.copy(),
        base=np.asarray([level.base for level in levels], dtype=np.uint32),
        orbit_sizes=orbit_sizes,
        level_generator_offsets=np.asarray(offsets, dtype=np.uint64),
        strong_generators=np.asarray(strong, dtype=np.uint32).reshape(
            (-1, degree)
        ),
        order=reduce(mul, map(int, orbit_sizes), 1),
        elapsed_seconds=perf_counter() - started,
        backend="reference",
    )


def permutation_orbits(
    generators: ArrayLike,
    *,
    degree: int | None = None,
    backend: GroupBackend = "auto",
) -> tuple[NDArray[np.uint32], ...]:
    """Return point orbits of the group generated by ``generators``."""
    _validate_backend(backend)
    prepared, degree = _prepare_permutations(
        generators, degree, name="generators"
    )
    if backend != "reference":
        try:
            labels, orbit_count, _ = permutation_orbits_native(
                prepared, degree
            )
        except (NativeUnavailable, OSError, AttributeError):
            if backend == "native":
                raise
        else:
            return tuple(
                np.flatnonzero(labels == orbit).astype(np.uint32)
                for orbit in range(orbit_count)
            )
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")

    tuples = [tuple(map(int, row)) for row in prepared]
    symmetric = _symmetric_generators(tuples)
    unseen = set(range(degree))
    result = []
    while unseen:
        seed = min(unseen)
        orbit = {seed}
        queue = deque([seed])
        while queue:
            point = queue.popleft()
            for generator in symmetric:
                image = generator[point]
                if image not in orbit:
                    orbit.add(image)
                    queue.append(image)
        unseen.difference_update(orbit)
        result.append(np.asarray(sorted(orbit), dtype=np.uint32))
    return tuple(result)


def schreier_sims(
    generators: ArrayLike,
    *,
    degree: int | None = None,
    backend: GroupBackend = "auto",
) -> SchreierSimsChain:
    """Build an exact deterministic stabilizer chain."""
    _validate_backend(backend)
    prepared, degree = _prepare_permutations(
        generators, degree, name="generators"
    )
    if backend != "reference":
        try:
            base, orbit_sizes, offsets, strong, stats = (
                schreier_sims_native(prepared, degree)
            )
        except (NativeUnavailable, OSError, AttributeError):
            if backend == "native":
                raise
        else:
            return SchreierSimsChain(
                degree=degree,
                generators=prepared.copy(),
                base=base,
                orbit_sizes=orbit_sizes,
                level_generator_offsets=offsets,
                strong_generators=strong,
                order=reduce(mul, map(int, orbit_sizes), 1),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    return _chain_result_reference(prepared, degree)


def group_order(
    generators: ArrayLike,
    *,
    degree: int | None = None,
    backend: GroupBackend = "auto",
) -> int:
    """Order of the permutation group these generators generate, by Schreier-Sims."""
    return schreier_sims(
        generators, degree=degree, backend=backend
    ).order


def _reference_contains(
    generators: NDArray[np.uint32],
    elements: NDArray[np.uint32],
    degree: int,
) -> NDArray[np.bool_]:
    return _reference_contains_levels(
        _reference_chain(generators, degree), elements, degree
    )


def permutation_group_contains(
    generators: ArrayLike,
    elements: ArrayLike,
    *,
    degree: int | None = None,
    threads: int = 0,
    backend: GroupBackend = "auto",
) -> NDArray[np.bool_]:
    """Test a batch of permutations for membership in a generated group."""
    _validate_backend(backend)
    if not isinstance(threads, Integral) or int(threads) < 0:
        raise ValueError("threads must be a nonnegative integer")
    prepared, degree = _prepare_permutations(
        generators, degree, name="generators"
    )
    candidates, _ = _prepare_permutations(
        elements, degree, name="elements"
    )
    if backend != "reference":
        try:
            output, _ = permutation_group_contains_native(
                prepared, candidates, degree, int(threads)
            )
        except (NativeUnavailable, OSError, AttributeError):
            if backend == "native":
                raise
        else:
            return output
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    return _reference_contains(prepared, candidates, degree)


def _double_cosets_reference(
    candidates: NDArray[np.uint32],
    left_generators: NDArray[np.uint32],
    right_generators: NDArray[np.uint32],
) -> tuple[NDArray[np.uint64], NDArray[np.uint64], NDArray[np.uint64]]:
    candidate_tuples = [tuple(map(int, row)) for row in candidates]
    index_of = {
        candidate: index
        for index, candidate in enumerate(candidate_tuples)
    }
    if len(index_of) != len(candidate_tuples):
        raise ValueError("candidates must be unique")
    left = _symmetric_generators(
        [tuple(map(int, row)) for row in left_generators]
    )
    right_inverse = [
        _tuple_inverse(generator)
        for generator in _symmetric_generators(
            [tuple(map(int, row)) for row in right_generators]
        )
    ]
    missing = np.iinfo(np.uint64).max
    class_ids = np.full(len(candidates), missing, dtype=np.uint64)
    representatives = []
    sizes = []
    for seed in range(len(candidates)):
        if class_ids[seed] != missing:
            continue
        class_id = len(representatives)
        representatives.append(seed)
        class_ids[seed] = class_id
        queue = deque([seed])
        size = 0
        while queue:
            current = queue.popleft()
            size += 1
            images = (
                [
                    _tuple_compose(generator, candidate_tuples[current])
                    for generator in left
                ]
                + [
                    _tuple_compose(candidate_tuples[current], generator)
                    for generator in right_inverse
                ]
            )
            for image in images:
                if image not in index_of:
                    raise ValueError(
                        "candidate collection is not closed under the "
                        "double-coset action"
                    )
                image_index = index_of[image]
                if class_ids[image_index] == missing:
                    class_ids[image_index] = class_id
                    queue.append(image_index)
        sizes.append(size)
    return (
        class_ids,
        np.asarray(representatives, dtype=np.uint64),
        np.asarray(sizes, dtype=np.uint64),
    )


def permutation_double_cosets(
    candidates: ArrayLike,
    left_generators: ArrayLike,
    right_generators: ArrayLike,
    *,
    degree: int | None = None,
    backend: GroupBackend = "auto",
) -> DoubleCosetPartition:
    """Partition candidates under ``beta * f * alpha^-1``.

    The candidate collection must be invariant under the groups generated by
    ``left_generators`` and ``right_generators``.
    """
    _validate_backend(backend)
    candidate_array, degree = _prepare_permutations(
        candidates,
        degree,
        name="candidates",
    )
    left, _ = _prepare_permutations(
        left_generators,
        degree,
        name="left_generators",
    )
    right, _ = _prepare_permutations(
        right_generators,
        degree,
        name="right_generators",
    )
    if backend != "reference":
        try:
            class_ids, indices, sizes, stats = (
                permutation_double_cosets_native(
                    candidate_array,
                    left,
                    right,
                )
            )
        except (NativeUnavailable, OSError, AttributeError):
            if backend == "native":
                raise
        else:
            return DoubleCosetPartition(
                class_ids=class_ids,
                representative_indices=indices,
                representatives=candidate_array[indices],
                class_sizes=sizes,
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    from time import perf_counter

    started = perf_counter()
    class_ids, indices, sizes = _double_cosets_reference(
        candidate_array,
        left,
        right,
    )
    return DoubleCosetPartition(
        class_ids=class_ids,
        representative_indices=indices,
        representatives=candidate_array[indices],
        class_sizes=sizes,
        elapsed_seconds=perf_counter() - started,
        backend="reference",
    )


__all__ = [
    "DoubleCosetPartition",
    "GroupBackend",
    "PermutationGroup",
    "SchreierSimsChain",
    "compose_permutations",
    "group_order",
    "invert_permutation",
    "permutation_group_contains",
    "permutation_double_cosets",
    "permutation_orbits",
    "schreier_sims",
]
