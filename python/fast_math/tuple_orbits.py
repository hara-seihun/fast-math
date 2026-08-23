from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._native import (
    NativeUnavailable,
    native_available,
    tuple_orbit_canonicalize_native,
    tuple_orbit_space_native,
)


TupleOrbitBackend = Literal["auto", "native", "reference"]

_MAX_SPACE_SIZE = 1 << 24


@dataclass(frozen=True)
class TupleOrbitPartition:
    """Dense orbit partition of a code batch.

    ``representatives`` are the numeric minima of each orbit in sorted
    order; ``class_ids[i]`` is the index of code ``i``'s orbit in that
    order, matching the class-id convention of ``MaskOrbitPartition``.
    """

    canonical_codes: NDArray[np.uint64]
    class_ids: NDArray[np.uint64]
    representatives: NDArray[np.uint64]
    class_sizes: NDArray[np.uint64]
    backend: str


@dataclass(frozen=True)
class TupleOrbitSpace:
    """Full-space orbit structure of a positional permutation action.

    ``orbit_ids[code]`` is the dense orbit id of every code in
    ``range(base ** width)``. ``burnside_orbit_count`` is the Burnside
    average ``sum(base ** cycles(g)) / |G|``; it is computed independently
    of the canonical pass and ``burnside_valid`` records that the two
    counts agree.
    """

    orbit_ids: NDArray[np.uint64]
    representatives: NDArray[np.uint64]
    orbit_count: int
    group_size: int
    burnside_orbit_count: int
    burnside_valid: bool
    backend: str


def _validate_shape(base: int, width: int) -> tuple[int, int]:
    if not isinstance(base, Integral) or not 2 <= int(base) <= 64:
        raise ValueError("base must be an integer between 2 and 64")
    if not isinstance(width, Integral) or not 1 <= int(width) <= 64:
        raise ValueError("width must be an integer between 1 and 64")
    return int(base), int(width)


def _prepare_generators(
    generators: ArrayLike,
    width: int,
) -> NDArray[np.uint32]:
    raw = np.asarray(generators)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] != width:
        raise ValueError(
            "generators must have shape (count, width) with count >= 1"
        )
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("generators must contain integers")
    if np.any(raw < 0) or np.any(raw >= width):
        raise ValueError("generator image is out of range")
    prepared = np.array(raw, dtype=np.uint32, order="C", copy=True)
    for row in prepared:
        if len(np.unique(row)) != width:
            raise ValueError(
                "generator rows must be permutations of 0..width-1"
            )
    return prepared


def _prepare_codes(codes: ArrayLike, limit: int) -> NDArray[np.uint64]:
    array = np.asarray(codes)
    if array.ndim != 1:
        raise ValueError("codes must be one-dimensional")
    if array.dtype.kind == "O":
        for value in array:
            if not isinstance(value, Integral) or value < 0:
                raise ValueError("codes must be nonnegative integers")
    elif array.dtype.kind == "i":
        if np.any(array < 0):
            raise ValueError("codes must be nonnegative")
    elif array.dtype.kind != "u":
        raise ValueError("codes must be integers")
    prepared = np.ascontiguousarray(array, dtype=np.uint64)
    if np.any(prepared >= np.uint64(limit)):
        raise ValueError("code is outside the base^width code range")
    return prepared


def _reference_group(
    generators: NDArray[np.uint32],
) -> list[tuple[int, ...]]:
    width = int(generators.shape[1])
    identity = tuple(range(width))
    group = {identity}
    queue = [identity]
    for row in generators:
        permutation = tuple(int(v) for v in row)
        if permutation not in group:
            group.add(permutation)
            queue.append(permutation)
    while queue:
        current = queue.pop()
        for row in generators:
            generator = tuple(int(v) for v in row)
            product = tuple(current[generator[i]] for i in range(width))
            if product not in group:
                group.add(product)
                queue.append(product)
    return list(group)


def _apply_reference(
    permutation: tuple[int, ...],
    digits: list[int],
) -> list[int]:
    out = [0] * len(digits)
    for i, target in enumerate(permutation):
        out[target] = digits[i]
    return out


def _decode(code: int, base: int, width: int) -> list[int]:
    digits = [0] * width
    for i in range(width - 1, -1, -1):
        digits[i] = code % base
        code //= base
    return digits


def _encode(digits: list[int], base: int) -> int:
    code = 0
    for digit in digits:
        code = code * base + digit
    return code


def _canonicalize_reference(
    generators: NDArray[np.uint32],
    base: int,
    width: int,
    codes: NDArray[np.uint64],
) -> tuple[NDArray[np.uint64], NDArray[np.bool_]]:
    group = _reference_group(generators)
    canonical = np.empty(len(codes), dtype=np.uint64)
    is_canonical = np.empty(len(codes), dtype=np.bool_)
    for index, raw in enumerate(codes):
        code = int(raw)
        best = code
        digits = _decode(code, base, width)
        for permutation in group:
            image = _encode(
                _apply_reference(permutation, digits), base
            )
            if image < best:
                best = image
        canonical[index] = np.uint64(best)
        is_canonical[index] = best == code
    return canonical, is_canonical


def _dispatch_canonicalize(
    generators: NDArray[np.uint32],
    base: int,
    width: int,
    codes: NDArray[np.uint64],
    backend: str,
) -> tuple[NDArray[np.uint64], NDArray[np.bool_], int, str]:
    if backend != "reference" and native_available():
        try:
            canonical, is_canonical, stats = (
                tuple_orbit_canonicalize_native(
                    generators, width, base, codes
                )
            )
            return canonical, is_canonical, int(stats.group_size), "native"
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")
    canonical, is_canonical = _canonicalize_reference(
        generators, base, width, codes
    )
    return canonical, is_canonical, len(_reference_group(generators)), (
        "reference"
    )


def tuple_orbit_canonicalize(
    generators: ArrayLike,
    base: int,
    width: int,
    codes: ArrayLike,
    *,
    backend: TupleOrbitBackend = "auto",
) -> tuple[NDArray[np.uint64], NDArray[np.bool_]]:
    """Canonicalize codes of Z_base^width under a positional group action.

    Codes are the mixed-radix indices of digit tuples, most significant
    digit first. Generator rows are image arrays: ``p[i]`` moves the digit
    at position ``i`` to position ``p[i]``. The canonical code of a tuple is
    the numeric minimum over its orbit under the generated group.
    """
    base, width = _validate_shape(base, width)
    prepared_generators = _prepare_generators(generators, width)
    prepared_codes = _prepare_codes(codes, base**width)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    canonical, is_canonical, _, _ = _dispatch_canonicalize(
        prepared_generators, base, width, prepared_codes, backend
    )
    return canonical, is_canonical


def tuple_orbit_partition(
    generators: ArrayLike,
    base: int,
    width: int,
    codes: ArrayLike,
    *,
    backend: TupleOrbitBackend = "auto",
) -> TupleOrbitPartition:
    """Partition a code batch into orbits of the positional group action.

    Formats ``tuple_orbit_canonicalize`` output into dense class ids,
    sorted representatives, and per-orbit sizes, mirroring
    ``MaskOrbitPartition`` from the packed-subset action API.
    """
    base, width = _validate_shape(base, width)
    prepared_generators = _prepare_generators(generators, width)
    prepared_codes = _prepare_codes(codes, base**width)
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")
    canonical, _, _, used = _dispatch_canonicalize(
        prepared_generators, base, width, prepared_codes, backend
    )
    representatives, class_ids, class_sizes = np.unique(
        canonical, return_inverse=True, return_counts=True
    )
    return TupleOrbitPartition(
        canonical_codes=canonical,
        class_ids=class_ids.astype(np.uint64, copy=False),
        representatives=representatives,
        class_sizes=class_sizes.astype(np.uint64, copy=False),
        backend=used,
    )


def tuple_orbit_space(
    generators: ArrayLike,
    base: int,
    width: int,
    *,
    backend: TupleOrbitBackend = "auto",
) -> TupleOrbitSpace:
    """Orbit structure of the full Z_base^width code space.

    Computes the canonical code of every tuple, partitions the space into
    dense orbit ids, and validates the orbit count against the Burnside
    average over the generated group (a position permutation with ``c``
    cycles fixes ``base ** c`` tuples). The space size ``base ** width``
    must not exceed ``2**24``.
    """
    base, width = _validate_shape(base, width)
    prepared_generators = _prepare_generators(generators, width)
    space = base**width
    if space > _MAX_SPACE_SIZE:
        raise ValueError(
            "full-space orbit computation is bounded at 2**24 codes; "
            "use tuple_orbit_partition for larger spaces"
        )
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")

    if backend != "reference" and native_available():
        try:
            canonical, stats = tuple_orbit_space_native(
                prepared_generators, width, base, space
            )
            return _space_record(
                canonical,
                int(stats.group_size),
                int(stats.burnside_orbit_count),
                bool(stats.burnside_valid),
                "native",
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")

    codes = np.arange(space, dtype=np.uint64)
    canonical, _ = _canonicalize_reference(
        prepared_generators, base, width, codes
    )
    group = _reference_group(prepared_generators)
    fixed_total = 0
    for permutation in group:
        seen = [False] * width
        cycles = 0
        for start in range(width):
            if seen[start]:
                continue
            cycles += 1
            node = start
            while not seen[node]:
                seen[node] = True
                node = permutation[node]
        fixed_total += base**cycles
    return _space_record(
        canonical,
        len(group),
        fixed_total // len(group),
        fixed_total % len(group) == 0,
        "reference",
    )


def _space_record(
    canonical: NDArray[np.uint64],
    group_size: int,
    burnside_orbit_count: int,
    burnside_valid: bool,
    backend: str,
) -> TupleOrbitSpace:
    representatives, orbit_ids = np.unique(canonical, return_inverse=True)
    return TupleOrbitSpace(
        orbit_ids=orbit_ids.astype(np.uint64, copy=False),
        representatives=representatives,
        orbit_count=len(representatives),
        group_size=group_size,
        burnside_orbit_count=burnside_orbit_count,
        burnside_valid=burnside_valid
        and len(representatives) == burnside_orbit_count,
        backend=backend,
    )


__all__ = [
    "TupleOrbitBackend",
    "TupleOrbitPartition",
    "TupleOrbitSpace",
    "tuple_orbit_canonicalize",
    "tuple_orbit_partition",
    "tuple_orbit_space",
]
