from __future__ import annotations

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from fast_math import (
    adaptive_areas,
    exact_adaptive_areas,
    quadratic_target_tables,
    restriction_assignment,
    restriction_code,
)


def direct_area(table: list[Fraction], coordinate_count: int) -> Fraction:
    """Textbook recursion over restrictions with no lattice encoding."""

    def cell(restriction: tuple[int, ...]) -> list[Fraction]:
        return [
            table[point]
            for point in range(1 << coordinate_count)
            if all(
                fixed == 0 or fixed == (1 if (point >> index) & 1 else -1)
                for index, fixed in enumerate(restriction)
            )
        ]

    memo: dict[tuple[int, ...], Fraction] = {}

    def area(restriction: tuple[int, ...]) -> Fraction:
        values = cell(restriction)
        size = len(values)
        mean = sum(values) / size
        variance = sum(value * value for value in values) / size - mean * mean
        if variance == 0:
            return Fraction(0)
        if restriction in memo:
            return memo[restriction]
        best = min(
            (
                area(restriction[:index] + (-1,) + restriction[index + 1 :])
                + area(restriction[:index] + (1,) + restriction[index + 1 :])
            )
            / 2
            for index, fixed in enumerate(restriction)
            if fixed == 0
        )
        memo[restriction] = variance + best
        return memo[restriction]

    return area((0,) * coordinate_count)


@pytest.mark.parametrize("coordinate_count", [1, 2, 3, 4])
def test_exact_areas_match_direct_recursion(coordinate_count: int) -> None:
    rng = np.random.default_rng(coordinate_count)
    tables = rng.integers(-4, 5, size=(6, 1 << coordinate_count))
    expected = [
        direct_area([Fraction(int(v)) for v in row], coordinate_count)
        for row in tables
    ]
    for backend in ("reference", "native"):
        batch = exact_adaptive_areas(tables, backend=backend)
        assert batch.areas() == expected


def test_exact_areas_cover_every_boolean_target_through_order_three() -> None:
    for coordinate_count in (1, 2, 3):
        points = 1 << coordinate_count
        tables = np.array(
            list(product((-1, 1), repeat=points)),
            dtype=np.int64,
        )
        expected = [
            direct_area([Fraction(int(v)) for v in row], coordinate_count)
            for row in tables
        ]
        batch = exact_adaptive_areas(tables, backend="native")
        assert batch.areas() == expected
        assert exact_adaptive_areas(
            tables, backend="reference"
        ).area_numerators.tolist() == batch.area_numerators.tolist()


@pytest.mark.parametrize("coordinate_count", [1, 3, 5, 7])
def test_native_matches_reference_on_float_targets(
    coordinate_count: int,
) -> None:
    rng = np.random.default_rng(100 + coordinate_count)
    tables = rng.normal(size=(5, 1 << coordinate_count))
    native = adaptive_areas(tables, restrictions=True, backend="native")
    reference = adaptive_areas(tables, restrictions=True, backend="reference")
    assert np.allclose(native.areas, reference.areas, rtol=0, atol=1e-12)
    assert np.array_equal(native.first_coordinates, reference.first_coordinates)
    assert np.allclose(native.variances, reference.variances, atol=1e-12)
    assert np.allclose(
        native.areas_by_restriction, reference.areas_by_restriction, atol=1e-12
    )
    assert np.array_equal(native.policies, reference.policies)


def test_restriction_arrays_expose_the_documented_recursion() -> None:
    rng = np.random.default_rng(5)
    coordinate_count = 4
    tables = rng.normal(size=(3, 1 << coordinate_count))
    batch = adaptive_areas(tables, restrictions=True)
    for target in range(tables.shape[0]):
        for code in range(3**coordinate_count):
            assignment = restriction_assignment(code, coordinate_count)
            assert restriction_code(assignment) == code
            choice = int(batch.policies[target, code])
            area = batch.areas_by_restriction[target, code]
            if choice < 0:
                assert area == 0.0
                continue
            children = [
                code + digit * 3**choice for digit in (1, 2)
            ]
            expected = batch.variances[target, code] + 0.5 * sum(
                batch.areas_by_restriction[target, child] for child in children
            )
            assert area == pytest.approx(expected, abs=1e-12)


def test_constant_targets_have_zero_area_and_no_first_query() -> None:
    tables = np.array([[3.0] * 8, [0.0] * 8, [-2.0] * 8])
    batch = adaptive_areas(tables)
    assert np.array_equal(batch.areas, np.zeros(3))
    assert np.array_equal(batch.first_coordinates, np.full(3, -1, np.int32))
    exact = exact_adaptive_areas(np.array([[3] * 8], dtype=np.int64))
    assert exact.area_numerators.tolist() == [0]


def test_single_coordinate_area_is_the_root_variance() -> None:
    tables = np.array([[-1.0, 1.0], [0.0, 4.0]])
    batch = adaptive_areas(tables)
    assert batch.areas == pytest.approx([1.0, 4.0])
    assert batch.first_coordinates.tolist() == [0, 0]


def test_quadratic_tables_reproduce_the_walsh_expansion() -> None:
    rng = np.random.default_rng(9)
    coordinate_count = 5
    linear = rng.normal(size=(2, coordinate_count))
    quadratic = np.triu(rng.normal(size=(2, coordinate_count, coordinate_count)), 1)
    tables = quadratic_target_tables(linear, quadratic)
    for target in range(2):
        for point in range(1 << coordinate_count):
            signs = [
                1.0 if (point >> index) & 1 else -1.0
                for index in range(coordinate_count)
            ]
            expected = sum(
                linear[target, index] * signs[index]
                for index in range(coordinate_count)
            ) + sum(
                quadratic[target, j, k] * signs[j] * signs[k]
                for j in range(coordinate_count)
                for k in range(j + 1, coordinate_count)
            )
            assert tables[target, point] == pytest.approx(expected)


def test_thread_counts_are_bitwise_deterministic() -> None:
    rng = np.random.default_rng(21)
    tables = rng.normal(size=(64, 1 << 8))
    single = adaptive_areas(tables, threads=1, restrictions=True)
    many = adaptive_areas(tables, threads=8, restrictions=True)
    assert np.array_equal(single.areas, many.areas)
    assert np.array_equal(single.first_coordinates, many.first_coordinates)
    assert np.array_equal(single.areas_by_restriction, many.areas_by_restriction)
    assert np.array_equal(single.policies, many.policies)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        adaptive_areas(np.zeros((2, 7)))
    with pytest.raises(ValueError):
        adaptive_areas(np.array([[np.nan, 0.0]]))
    with pytest.raises(ValueError):
        adaptive_areas(np.zeros((2, 1 << 8)), backend="cuda")
    with pytest.raises(ValueError):
        exact_adaptive_areas(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        exact_adaptive_areas(np.full((1, 4), 1 << 40, dtype=np.int64))
    with pytest.raises(ValueError):
        restriction_assignment(3**3, 3)
    with pytest.raises(ValueError):
        restriction_code([2, 0, 0])


def test_inputs_are_not_mutated() -> None:
    rng = np.random.default_rng(33)
    tables = rng.normal(size=(4, 1 << 6))
    original = tables.copy()
    adaptive_areas(tables, restrictions=True)
    assert np.array_equal(tables, original)
    integers = rng.integers(-2, 3, size=(4, 1 << 6))
    integer_original = integers.copy()
    exact_adaptive_areas(integers, restrictions=True)
    assert np.array_equal(integers, integer_original)


def test_zero_tolerance_prunes_near_constant_subcubes() -> None:
    table = np.zeros((1, 1 << 4))
    table[0, 0] = 1e-8
    strict = adaptive_areas(table, zero_tolerance=0.0)
    pruned = adaptive_areas(table, zero_tolerance=1e-9)
    assert strict.areas[0] > 0.0
    assert pruned.areas[0] == 0.0
    assert pruned.first_coordinates[0] == -1
