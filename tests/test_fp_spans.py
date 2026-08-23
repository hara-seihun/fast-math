from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from fast_math import fp_point_span, fp_span_ranks
from fast_math._native import (
    fp_point_span_native,
    fp_span_ranks_native,
    native_available,
)
from fast_math.base_p import base_p_codes, base_p_digits


BACKENDS = ["reference", "native"]


def encode(rows, prime: int) -> np.ndarray:
    return base_p_codes(np.asarray(rows, dtype=np.uint8), prime, backend="reference")


def brute_span(rows: np.ndarray, prime: int) -> set[tuple[int, ...]]:
    width = rows.shape[1]
    result = {tuple(0 for _ in range(width))}
    for row in rows:
        result = {
            tuple((left + coefficient * int(right)) % prime for left, right in zip(old, row))
            for old in result
            for coefficient in range(prime)
        }
    return result


def assert_same(left, right) -> None:
    assert left.prime == right.prime
    assert left.width == right.width
    assert left.rank == right.rank
    for name in (
        "pivot_indices",
        "pivot_columns",
        "reduced_basis_codes",
        "independent_points",
        "query_members",
        "query_coordinates",
        "query_quotient_codes",
    ):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


@pytest.mark.parametrize("backend", BACKENDS)
def test_canonical_rref_and_query_decomposition(backend: str) -> None:
    prime, width = 3, 3
    points = encode(
        [
            [1, 1, 1],
            [0, 1, 1],
            [2, 1, 1],
            [0, 0, 0],
        ],
        prime,
    )
    queries = np.arange(prime**width, dtype=np.uint64)
    result = fp_point_span(points, queries, prime, width, backend=backend)

    assert result.rank == 2
    assert result.pivot_columns.tolist() == [0, 1]
    assert result.pivot_indices.tolist() == [0, 1]
    assert result.independent_points.tolist() == [True, True, False, False]
    basis = base_p_digits(
        result.reduced_basis_codes, prime, width, backend="reference"
    ).astype(np.uint64)
    np.testing.assert_array_equal(basis, [[1, 0, 0], [0, 1, 1]])

    query_rows = base_p_digits(
        queries, prime, width, backend="reference"
    ).astype(np.uint64)
    quotient_rows = base_p_digits(
        result.query_quotient_codes, prime, width, backend="reference"
    ).astype(np.uint64)
    rebuilt = (
        result.query_coordinates.astype(np.uint64) @ basis + quotient_rows
    ) % prime
    np.testing.assert_array_equal(rebuilt, query_rows)
    np.testing.assert_array_equal(result.query_members, quotient_rows.sum(axis=1) == 0)
    assert np.all(quotient_rows[:, result.pivot_columns] == 0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_membership_matches_independent_brute_closure(backend: str) -> None:
    prime, width = 3, 3
    points = encode([[1, 1, 0], [0, 1, 1], [2, 2, 0]], prime)
    queries = np.arange(prime**width, dtype=np.uint64)
    result = fp_point_span(points, queries, prime, width, backend=backend)
    expected = brute_span(
        base_p_digits(points, prime, width, backend="reference"), prime
    )
    actual = {
        tuple(map(int, row))
        for row, member in zip(
            base_p_digits(queries, prime, width, backend="reference"),
            result.query_members,
        )
        if member
    }
    assert actual == expected
    assert result.rank == 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_quotient_codes_are_constant_on_cosets(backend: str) -> None:
    prime, width = 5, 4
    points = encode([[1, 2, 0, 0], [0, 1, 3, 0]], prime)
    point_rows = base_p_digits(points, prime, width, backend="reference")
    span = sorted(brute_span(point_rows, prime))
    representatives = encode([[0, 0, 1, 4], [0, 0, 2, 3]], prime)
    queries = []
    groups = []
    for representative in representatives:
        row = base_p_digits(
            [representative], prime, width, backend="reference"
        )[0]
        begin = len(queries)
        for shift in span:
            queries.append(tuple((row + shift) % prime))
        groups.append(slice(begin, len(queries)))
    result = fp_point_span(
        points, encode(queries, prime), prime, width, backend=backend
    )
    for group in groups:
        assert np.unique(result.query_quotient_codes[group]).size == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_ragged_rank_batches_match_brute_force(backend: str) -> None:
    prime, width = 3, 3
    batches = [
        [],
        [[1, 0, 0]],
        [[1, 1, 0], [2, 2, 0], [0, 0, 1]],
        list(product(range(prime), repeat=width)),
        [],
    ]
    rows = [row for batch in batches for row in batch]
    points = encode(rows, prime) if rows else np.empty(0, dtype=np.uint64)
    offsets = np.cumsum([0] + [len(batch) for batch in batches], dtype=np.uint64)
    ranks = fp_span_ranks(points, offsets, prime, width, backend=backend)
    expected = []
    for batch in batches:
        array = np.asarray(batch, dtype=np.uint8).reshape(-1, width)
        size = len(brute_span(array, prime))
        rank = 0
        while prime**rank < size:
            rank += 1
        expected.append(rank)
    np.testing.assert_array_equal(ranks, expected)


@pytest.mark.parametrize("prime", [2, 3, 5, 251])
def test_native_and_reference_match_random_batches(prime: int) -> None:
    width = 4
    rng = np.random.default_rng(20260823 + prime)
    points = rng.integers(0, prime**width, size=137, dtype=np.uint64)
    queries = rng.integers(0, prime**width, size=401, dtype=np.uint64)
    reference = fp_point_span(points, queries, prime, width, backend="reference")
    native = fp_point_span(points, queries, prime, width, backend="native")
    assert_same(reference, native)

    lengths = rng.integers(0, 12, size=53)
    offsets = np.cumsum(np.concatenate(([0], lengths)), dtype=np.uint64)
    ragged = rng.integers(0, prime**width, size=int(offsets[-1]), dtype=np.uint64)
    np.testing.assert_array_equal(
        fp_span_ranks(ragged, offsets, prime, width, backend="native"),
        fp_span_ranks(ragged, offsets, prime, width, backend="reference"),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_span_has_identity_quotient(backend: str) -> None:
    queries = np.array([0, 1, 7, 48], dtype=np.uint64)
    result = fp_point_span([], queries, 7, 2, backend=backend)
    assert result.rank == 0
    assert result.reduced_basis_codes.shape == (0,)
    assert result.query_coordinates.shape == (4, 0)
    assert result.query_members.tolist() == [True, False, False, False]
    np.testing.assert_array_equal(result.query_quotient_codes, queries)
    np.testing.assert_array_equal(
        fp_span_ranks([], [0, 0, 0], 7, 2, backend=backend), [0, 0]
    )


@pytest.mark.parametrize(
    ("points", "offsets", "prime", "width", "message"),
    [
        ([1], [0, 1], 4, 2, "prime"),
        ([1], [0, 1], 257, 2, "prime"),
        ([1], [0, 1], 5, 0, "between one"),
        ([1], [0, 1], 5, 17, "between one"),
        ([1], [0, 1], 251, 16, "does not fit"),
        ([-1], [0, 1], 5, 2, "nonnegative"),
        ([[1]], [0, 1], 5, 2, "one-dimensional"),
        ([1.5], [0, 1], 5, 2, "integers"),
        ([25], [0, 1], 5, 2, "outside"),
        ([1], [1, 1], 5, 2, "start"),
        ([1], [0, 1, 0], 5, 2, "nondecreasing"),
        ([1], [0, 0], 5, 2, "point count"),
    ],
)
def test_rejects_invalid_rank_inputs(
    points, offsets, prime: int, width: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fp_span_ranks(points, offsets, prime, width)


def test_point_span_rejects_bad_queries_and_backend() -> None:
    with pytest.raises(ValueError, match="outside"):
        fp_point_span([1], [25], 5, 2)
    with pytest.raises(ValueError, match="one-dimensional"):
        fp_point_span([1], [[1]], 5, 2)
    with pytest.raises(ValueError, match="backend"):
        fp_point_span([1], [1], 5, 2, backend="gpu")


def test_native_abi_rejects_bad_offsets() -> None:
    if not native_available():
        pytest.skip("fast-math native library is unavailable")
    points = np.array([1, 2], dtype=np.uint64)
    offsets = np.array([0, 2, 1], dtype=np.uint64)
    with pytest.raises(RuntimeError, match="offsets"):
        fp_span_ranks_native(points, offsets, 5, 2)


@pytest.mark.parametrize(
    ("prime", "width", "points", "message"),
    [
        (4, 2, [1], "prime"),
        (251, 16, [1], "does not fit"),
        (5, 2, [25], "outside"),
    ],
)
def test_native_abi_rejects_invalid_field_and_codes(
    prime: int, width: int, points: list[int], message: str
) -> None:
    if not native_available():
        pytest.skip("fast-math native library is unavailable")
    encoded = np.asarray(points, dtype=np.uint64)
    offsets = np.array([0, len(encoded)], dtype=np.uint64)
    with pytest.raises(RuntimeError, match=message):
        fp_span_ranks_native(encoded, offsets, prime, width)
    with pytest.raises(RuntimeError, match=message):
        fp_point_span_native(
            encoded, np.zeros(0, dtype=np.uint64), prime, width
        )
