from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from fast_math.cnf import CnfPlan, pack_boolean_assignments
from fast_math.hip import hip_cnf_available


def test_exhaustive_small_cnf_witnesses() -> None:
    clauses = [[1, -2], [2, 3], [-1, -3]]
    booleans = np.asarray(list(product([False, True], repeat=3)))
    assignments = pack_boolean_assignments(booleans)
    with CnfPlan(clauses, variable_count=3) as plan:
        reference = plan.evaluate(assignments, backend="reference")
        native = plan.evaluate(assignments, threads=3, backend="native")
    np.testing.assert_array_equal(
        reference.first_unsatisfied_clause,
        [1, -1, 0, 0, 1, 2, -1, 2],
    )
    np.testing.assert_array_equal(
        native.first_unsatisfied_clause,
        reference.first_unsatisfied_clause,
    )
    np.testing.assert_array_equal(native.satisfied, reference.satisfied)
    assert native.inspected_literal_count == reference.inspected_literal_count


@pytest.mark.skipif(not hip_cnf_available(), reason="HIP CNF backend unavailable")
def test_hip_matches_native_and_is_deterministic() -> None:
    rng = np.random.default_rng(71)
    variable_count = 129
    clauses = []
    for _ in range(300):
        variables = rng.choice(variable_count, size=4, replace=False)
        signs = rng.integers(0, 2, size=4, dtype=np.uint8)
        clauses.append(
            [
                int(variable + 1) if sign else -int(variable + 1)
                for variable, sign in zip(variables, signs)
            ]
        )
    booleans = rng.integers(
        0, 2, size=(4096, variable_count), dtype=np.uint8
    )
    assignments = pack_boolean_assignments(booleans)
    with CnfPlan(clauses, variable_count=variable_count) as plan:
        native = plan.evaluate(assignments, threads=4, backend="native")
        first = plan.evaluate(assignments, backend="hip")
        second = plan.evaluate(assignments, backend="hip")
    for result in (first, second):
        np.testing.assert_array_equal(result.satisfied, native.satisfied)
        np.testing.assert_array_equal(
            result.first_unsatisfied_clause,
            native.first_unsatisfied_clause,
        )


def test_empty_formula_and_empty_clause() -> None:
    assignments = pack_boolean_assignments([[False], [True]])
    with CnfPlan([], variable_count=1) as plan:
        empty_formula = plan.evaluate(assignments, backend="native")
    np.testing.assert_array_equal(empty_formula.satisfied, [True, True])
    np.testing.assert_array_equal(
        empty_formula.first_unsatisfied_clause, [-1, -1]
    )
    with CnfPlan([[]], variable_count=1) as plan:
        empty_clause = plan.evaluate(assignments, backend="native")
    np.testing.assert_array_equal(empty_clause.satisfied, [False, False])
    np.testing.assert_array_equal(
        empty_clause.first_unsatisfied_clause, [0, 0]
    )


def test_assignment_boundaries_and_validation() -> None:
    booleans = np.zeros((2, 65), dtype=np.bool_)
    booleans[0, 64] = True
    words = pack_boolean_assignments(booleans)
    assert words.shape == (2, 2)
    assert words[0, 1] == 1
    plan = CnfPlan([[65]], variable_count=65)
    result = plan.evaluate(words, backend="reference")
    np.testing.assert_array_equal(result.satisfied, [True, False])
    with pytest.raises(ValueError, match="out-of-range"):
        plan.evaluate([[0, 2]])
    with pytest.raises(ValueError, match="nonzero"):
        CnfPlan([[0]], variable_count=1)
    plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        plan.evaluate(words)
