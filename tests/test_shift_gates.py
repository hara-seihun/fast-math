from __future__ import annotations

import math

import numpy as np
import pytest

from flint import fmpz

from fast_math._native import native_available
from fast_math.hip import hip_shift_gates_available
from fast_math.shift_gates import (
    ShiftGateScanPlan,
    derive_shift_gate,
    verify_shift_gate,
)

needs_native = pytest.mark.skipif(not native_available(), reason="native library unavailable")
needs_hip = pytest.mark.skipif(
    not hip_shift_gates_available(), reason="HIP shift-gate backend unavailable"
)


def _tau(x: int) -> int:
    out = 1
    for _, e in fmpz(x).factor():
        out *= int(e) + 1
    return out


# --- derivation ------------------------------------------------------------


def test_erdos_647_gate_shape() -> None:
    gate = derive_shift_gate(360360)
    assert [f.shift for f in gate.forms] == [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 18, 20, 24]
    assert all(gate.modulus % f.a == 0 for f in gate.forms)
    assert gate.max_alive_smooth == 9
    # Wheel classes and tables kill most of v-space before any sieve work.
    plan = ShiftGateScanPlan(gate)
    assert len(plan.wheel_classes) / plan.wheel < 0.045


@pytest.mark.parametrize("modulus", [4, 24, 2520, 360360])
def test_derivation_verifies_against_direct_factorization(modulus: int) -> None:
    gate = derive_shift_gate(modulus)
    verify_shift_gate(gate, trials=300, small_limit=800)


def test_gate_survivor_matches_tau_condition_exactly() -> None:
    # On the gate's domain a survivor is exactly a v whose exploitable
    # shifts all satisfy their tau budgets; cross-check both directions on a
    # contiguous window against flint.
    gate = derive_shift_gate(4)
    shifts = {f.shift for f in gate.forms}
    for v in range(10_000, 12_000):
        expected = all(_tau(4 * v - j) <= j + 2 for j in shifts)
        # The gate may kill via classes even when the sieveable shifts pass
        # (a non-exploitable shift's tau bound is not modeled), so only the
        # kill direction is exact.
        if gate.survivor(v):
            assert expected
        elif expected:
            # Killed despite sieveable shifts passing: some other shift must
            # be violated.
            assert any(_tau(4 * v - j) > j + 2 for j in range(1, gate.jmax + 1))


# --- scanning --------------------------------------------------------------

M4_U64_WINDOW = (10_000_000_000, 300_000)
M4_U128_WINDOW = (5_000_000_000_000_000_000, 3_000_000)


@pytest.fixture(scope="module")
def m4_plan() -> ShiftGateScanPlan:
    return ShiftGateScanPlan(derive_shift_gate(4))


@pytest.fixture(scope="module")
def m4_reference_u64(m4_plan):
    return m4_plan.scan_reference(*M4_U64_WINDOW)


@pytest.fixture(scope="module")
def m4_reference_u128(m4_plan):
    return m4_plan.scan_reference(*M4_U128_WINDOW)


def test_reference_scan_finds_survivors(m4_reference_u64) -> None:
    survivors, stats = m4_reference_u64
    assert stats.survivors == len(survivors) > 0
    for v in survivors.tolist():
        for j in (1, 2, 4, 8):
            assert _tau(4 * v - j) <= j + 2


@needs_native
def test_native_matches_reference_u64(m4_plan, m4_reference_u64) -> None:
    survivors, _ = m4_reference_u64
    native, stats = m4_plan.scan(*M4_U64_WINDOW, backend="native")
    np.testing.assert_array_equal(native, survivors)
    assert stats.survivors == len(survivors)


@needs_native
def test_native_matches_reference_u128(m4_plan, m4_reference_u128) -> None:
    survivors, _ = m4_reference_u128
    native, _ = m4_plan.scan(*M4_U128_WINDOW, backend="native")
    np.testing.assert_array_equal(native, survivors)
    assert len(survivors) > 0  # two-word Montgomery path genuinely exercised


@needs_native
def test_survivors_independent_of_sieve_bound(m4_reference_u64) -> None:
    survivors, _ = m4_reference_u64
    weak_plan = ShiftGateScanPlan(derive_shift_gate(4), sieve_bound=64)
    weak, stats = weak_plan.scan(*M4_U64_WINDOW, backend="native")
    np.testing.assert_array_equal(weak, survivors)
    # With almost no sieving every alive candidate reaches Miller-Rabin.
    assert stats.wheel_alive > 10 * len(survivors)


@needs_hip
def test_hip_matches_reference_u64(m4_plan, m4_reference_u64) -> None:
    survivors, _ = m4_reference_u64
    hip, _ = m4_plan.scan(*M4_U64_WINDOW, backend="hip")
    np.testing.assert_array_equal(hip, survivors)


@needs_hip
def test_hip_matches_reference_u128(m4_plan, m4_reference_u128) -> None:
    survivors, _ = m4_reference_u128
    hip, _ = m4_plan.scan(*M4_U128_WINDOW, backend="hip")
    np.testing.assert_array_equal(hip, survivors)


@needs_native
def test_erdos_gate_backends_agree_at_campaign_scale() -> None:
    plan = ShiftGateScanPlan(derive_shift_gate(360360))
    v_start, v_count = 1_935_428_000_000, 1_000_000
    reference, _ = plan.scan_reference(v_start, v_count)
    native, _ = plan.scan(v_start, v_count, backend="native")
    np.testing.assert_array_equal(native, reference)
    # The window straddles a known 13-form survivor.
    assert 1_935_428_395_330 in native.tolist()


# --- preconditions ---------------------------------------------------------


def test_scan_rejects_ranges_too_low_for_exact_sieving(m4_plan) -> None:
    with pytest.raises(ValueError, match="too low"):
        m4_plan.scan(1_000, 1_000)


def test_plan_rejects_wheel_primes_dividing_modulus() -> None:
    with pytest.raises(ValueError, match="wheel primes"):
        ShiftGateScanPlan(derive_shift_gate(360360), wheel_primes=(13, 17, 19))
