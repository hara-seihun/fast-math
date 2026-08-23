from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from fast_math import (
    base_p_class_table,
    base_p_codes,
    base_p_digits,
    base_p_negation_codes,
    base_p_scalar_normals,
)
from fast_math._native import (
    NativeUnavailable,
    base_p_class_table_native,
    native_available,
)

NATIVE = pytest.param(
    "native",
    marks=pytest.mark.skipif(
        not native_available(),
        reason="fast-math native library is unavailable",
    ),
)


def brute_force_digits(codes: np.ndarray, prime: int, width: int):
    return np.array(
        [
            [(int(code) // prime**j) % prime for j in range(width)]
            for code in codes
        ],
        dtype=np.uint8,
    )


def brute_force_negation(code: int, prime: int, width: int) -> int:
    result, weight = 0, 1
    for _ in range(width):
        digit = code % prime
        code //= prime
        if digit:
            result += (prime - digit) * weight
        weight *= prime
    return result


def brute_force_normal(code: int, prime: int, width: int) -> int:
    digits = [(code // prime**j) % prime for j in range(width)]
    lead = next((j for j, d in enumerate(digits) if d), None)
    if lead is None:
        return 0
    scale = pow(digits[lead], -1, prime)
    return sum(d * scale % prime * prime**j for j, d in enumerate(digits))


@pytest.mark.parametrize("backend", ["reference", NATIVE])
@pytest.mark.parametrize(("prime", "width"), [(2, 1), (2, 6), (3, 4), (5, 3), (7, 2)])
def test_digits_match_exhaustive_brute_force(
    backend: str,
    prime: int,
    width: int,
) -> None:
    codes = np.arange(prime**width, dtype=np.uint64)
    digits = base_p_digits(codes, prime, width, backend=backend)
    np.testing.assert_array_equal(
        digits, brute_force_digits(codes, prime, width)
    )
    np.testing.assert_array_equal(
        base_p_codes(digits, prime, backend=backend), codes
    )


@pytest.mark.parametrize("backend", ["reference", NATIVE])
def test_codec_round_trips_random_batches(backend: str) -> None:
    rng = np.random.default_rng(20260823)
    # Widths chosen so every p^width stays inside uint64 codes.
    for prime, top_width in ((2, 16), (3, 16), (5, 16), (7, 16), (11, 16), (251, 8)):
        for width in {1, 5, top_width}:
            space = prime**width
            if space <= np.iinfo(np.uint64).max // 2:
                codes = rng.integers(0, space, size=129, dtype=np.uint64)
            else:
                # Draw bytes to stay inside the generator's uint64 range.
                raw = rng.integers(
                    0, 256, size=(129, 8), dtype=np.uint8
                )
                codes = raw.view(np.uint64).ravel() % np.uint64(space)
            digits = base_p_digits(codes, prime, width, backend=backend)
            assert digits.dtype == np.uint8
            assert digits.shape == (129, width)
            np.testing.assert_array_equal(
                base_p_codes(digits, prime, backend=backend), codes
            )


@pytest.mark.parametrize("backend", ["reference", NATIVE])
@pytest.mark.parametrize(("prime", "width"), [(2, 5), (3, 4), (5, 3), (13, 2)])
def test_negation_matches_digit_wise_brute_force(
    backend: str,
    prime: int,
    width: int,
) -> None:
    codes = np.arange(prime**width, dtype=np.uint64)
    negated = base_p_negation_codes(codes, prime, width, backend=backend)
    expected = np.array(
        [brute_force_negation(int(c), prime, width) for c in codes],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(negated, expected)
    # Involution and additive-inverse semantics on sampled points.
    rng = np.random.default_rng(prime * width + 1)
    sample = rng.integers(0, prime**width, size=97, dtype=np.uint64)
    once = base_p_negation_codes(sample, prime, width, backend=backend)
    twice = base_p_negation_codes(once, prime, width, backend=backend)
    np.testing.assert_array_equal(twice, sample)


@pytest.mark.parametrize("backend", ["reference", NATIVE])
def test_scalar_normals_match_brute_force_and_absorb_units(
    backend: str,
) -> None:
    prime, width = 7, 4
    codes = np.arange(prime**width, dtype=np.uint64)
    normals = base_p_scalar_normals(codes, prime, width, backend=backend)
    expected = np.array(
        [brute_force_normal(int(c), prime, width) for c in codes],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(normals, expected)
    # Every unit multiple lands on the same normal form; normal forms are
    # fixed points; the zero vector is its own class.
    digits = base_p_digits(codes, prime, width)
    for unit in range(1, prime):
        multiples = base_p_codes((digits * unit) % prime, prime)
        np.testing.assert_array_equal(
            normals[multiples], normals[codes]
        )
    np.testing.assert_array_equal(normals[normals], normals)
    assert normals[0] == 0


@pytest.mark.parametrize("backend", ["reference", NATIVE])
@pytest.mark.parametrize("classes", ["negation", "scalar"])
def test_class_tables_match_independent_unique_construction(
    backend: str,
    classes: str,
) -> None:
    prime, width = 3, 4
    table = base_p_class_table(prime, width, classes=classes, backend=backend)
    canonical = {
        "negation": lambda code: min(
            code, brute_force_negation(code, prime, width)
        ),
        "scalar": lambda code: brute_force_normal(code, prime, width),
    }[classes]
    expected_ids = {}
    representatives = sorted(
        {canonical(code) for code in range(prime**width)}
    )
    for index, representative in enumerate(representatives):
        expected_ids[representative] = index
    np.testing.assert_array_equal(
        table.representatives, np.array(representatives, dtype=np.uint64)
    )
    np.testing.assert_array_equal(
        table.class_ids,
        np.array(
            [expected_ids[canonical(code)] for code in range(prime**width)],
            dtype=np.uint32,
        ),
    )
    counts = np.bincount(table.class_ids, minlength=len(representatives))
    np.testing.assert_array_equal(table.counts, counts)


@pytest.mark.parametrize("classes", ["negation", "scalar"])
def test_class_tables_carry_exact_sizes_and_dense_ids(classes: str) -> None:
    prime, width = 5, 2
    space = prime**width
    table = base_p_class_table(prime, width, classes=classes)
    expected_count = (
        (space + 1) // 2 if classes == "negation"
        else (space - 1) // (prime - 1) + 1
    )
    assert len(table.representatives) == len(table.counts) == expected_count
    assert table.class_ids.shape == (space,)
    assert sorted(np.unique(table.class_ids).tolist()) == list(
        range(expected_count)
    )
    assert int(table.counts.sum()) == space
    assert table.counts.min() >= 1
    # Ascending representatives; zero vector forms class zero.
    assert table.representatives.tolist() == sorted(
        table.representatives.tolist()
    )
    assert table.representatives[0] == 0 and table.class_ids[0] == 0


def test_negation_table_pairs_every_code_with_its_partner() -> None:
    prime, width = 5, 3
    codes = np.arange(prime**width, dtype=np.uint64)
    table = base_p_class_table(prime, width, classes="negation")
    partners = base_p_negation_codes(codes, prime, width)
    np.testing.assert_array_equal(
        table.class_ids[codes], table.class_ids[partners]
    )
    assert set(table.counts.tolist()) <= {1, 2}
    assert table.counts[0] == 1  # zero pairs with itself


def test_scalar_table_counts_projective_points() -> None:
    # PG(2,5) has 31 projective points plus the zero class.
    prime, width = 5, 3
    table = base_p_class_table(prime, width, classes="scalar")
    assert len(table.representatives) == 32
    nonzero = table.counts[1:]
    assert set(nonzero.tolist()) == {prime - 1}


def test_two_prime_negation_classes_are_singletons() -> None:
    prime, width = 2, 4
    table = base_p_class_table(prime, width, classes="negation")
    assert len(table.representatives) == prime**width
    assert set(table.counts.tolist()) == {1}


def test_native_class_table_rejects_small_capacity() -> None:
    if not native_available():
        pytest.skip("fast-math native library is unavailable")
    with pytest.raises(RuntimeError, match="smaller than the class count"):
        base_p_class_table_native(3, 3, 0, 27, 5)


def test_empty_batch_is_accepted_on_both_backends() -> None:
    codes = np.empty(0, dtype=np.uint64)
    for backend in ("reference", "native"):
        assert len(base_p_digits(codes, 5, 3, backend=backend)) == 0
        assert len(base_p_negation_codes(codes, 5, 3, backend=backend)) == 0
        assert len(base_p_scalar_normals(codes, 5, 3, backend=backend)) == 0
        assert len(
            base_p_codes(np.empty((0, 3), dtype=np.uint8), 5, backend=backend)
        ) == 0


@pytest.mark.parametrize(
    ("codes", "prime", "width", "message"),
    [
        ([5], 4, 1, "prime"),
        ([1], 256, 1, "prime"),
        ([1], 2, 0, "between one"),
        ([1], 2, 17, "between one"),
        ([1], 2, 1.5, "integer"),
        ([-1], 5, 2, "nonnegative"),
        ([[1]], 5, 2, "one-dimensional"),
        ([1.5], 5, 2, "integers"),
        ([25], 5, 2, "outside"),
        ([1 << 64], 251, 2, "outside"),
    ],
)
def test_rejects_invalid_code_inputs(
    codes, prime: int, width: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        base_p_digits(codes, prime, width)
    with pytest.raises(ValueError, match=message):
        base_p_negation_codes(codes, prime, width)
    with pytest.raises(ValueError, match=message):
        base_p_scalar_normals(codes, prime, width)


@pytest.mark.parametrize(
    ("digits", "prime", "message"),
    [
        (np.full((2, 2), 5, dtype=np.uint8), 5, "outside the prime field"),
        (-np.ones((2, 2), dtype=np.int64), 5, "nonnegative"),
        (np.zeros((2,), dtype=np.uint8), 5, "two-dimensional"),
        (np.zeros((2, 3), dtype=np.float64), 5, "integers"),
    ],
)
def test_rejects_invalid_digit_inputs(
    digits: np.ndarray, prime: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        base_p_codes(digits, prime)


def test_rejects_invalid_class_kind_and_backend() -> None:
    with pytest.raises(ValueError, match="negation"):
        base_p_class_table(5, 2, classes="sign")
    with pytest.raises(ValueError, match="backend"):
        base_p_digits([1], 5, 2, backend="other")
    if native_available():
        with pytest.raises(RuntimeError, match="class_kind"):
            base_p_class_table_native(5, 2, 7, 25, 50)


def test_width_seventeen_is_rejected_even_when_codes_fit() -> None:
    with pytest.raises(ValueError, match="between one"):
        base_p_digits([0], 2, 17)
    with pytest.raises(ValueError, match="between one"):
        base_p_class_table(2, 17)


def test_object_dtype_codes_beyond_uint64_are_rejected() -> None:
    huge = 1 << 70
    with pytest.raises(ValueError, match="outside"):
        base_p_scalar_normals([huge], 251, 2)


def test_reference_and_native_full_space_parity_at_251() -> None:
    if not native_available():
        pytest.skip("fast-math native library is unavailable")
    prime, width = 251, 2
    codes = np.arange(prime**width, dtype=np.uint64)
    np.testing.assert_array_equal(
        base_p_digits(codes, prime, width, backend="native"),
        base_p_digits(codes, prime, width, backend="reference"),
    )
    np.testing.assert_array_equal(
        base_p_negation_codes(codes, prime, width, backend="native"),
        base_p_negation_codes(codes, prime, width, backend="reference"),
    )
    np.testing.assert_array_equal(
        base_p_scalar_normals(codes, prime, width, backend="native"),
        base_p_scalar_normals(codes, prime, width, backend="reference"),
    )
