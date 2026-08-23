"""Shift-divisor gates: derived, verified, and scanned at scale.

A shift-divisor problem asks for n = modulus * v whose neighbourhood satisfies
divisor bounds tau(n - j) <= budget(j) for every positive shift j (Erdos 647
is budget(j) = j + 2 with modulus 360360).  For p | modulus the exponent
v_p(n - j) is a function of v modulo a small power of p, so each shift
factors n - j = F(class) * cofactor with tau(n - j) = tau(F) * tau(cofactor)
and cofactor budget B = floor(budget(j) / tau(F)):

  B <= 1   the class is dead: no witness has v in it
  B == 2   the modulus-coprime part of A*v - b must be prime
  B == 3   ... must be prime or the square of a prime
  B >= 4   the shift cannot be exploited by the gate

``derive_shift_gate`` computes, by exact integer arithmetic, the per-prime
alive lookup tables and every shift whose alive classes all have B <= 3;
those shifts yield linear forms A*v - b (A = modulus / F_generic) whose
coprime part must be prime or a prime square for every witness.

``verify_shift_gate`` re-checks the derivation against direct factorization:
exponent tables on random v, kill necessity exhaustively on small v and per
form at mid scale, and absence of false kills.

``ShiftGateScanPlan`` scans a v-interval for gate survivors.  The reference
backend evaluates the contract directly; the native and HIP backends run a
wheel-compressed segmented sieve (kills v when a prime q <= sieve_bound
coprime to the modulus divides some form; sound because on alive classes the
form's modulus-smooth part is bounded, so a small q forces the coprime part
composite and non-square) followed by deterministic Miller-Rabin on what
remains.  Survivors are identical across backends and independent of
sieve_bound; the plan asserts the size precondition that makes this exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from flint import fmpz

__all__ = [
    "ShiftGate",
    "ShiftGateForm",
    "ShiftGateScanPlan",
    "ShiftGateStats",
    "derive_shift_gate",
    "verify_shift_gate",
]

# Deterministic Miller-Rabin base set for moduli < 3.317e24 (Sorenson &
# Webster); every value this module tests is far below that bound.
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
_MR_LIMIT = 3_317_044_064_679_887_385_961_981


def _factorize(x: int) -> dict[int, int]:
    return {int(p): int(e) for p, e in fmpz(x).factor()}


def _vp(x: int, p: int) -> int:
    e = 0
    while x % p == 0:
        x //= p
        e += 1
    return e


def _tau(x: int) -> int:
    out = 1
    for _, e in fmpz(x).factor():
        out *= int(e) + 1
    return out


def _is_prime(x: int) -> bool:
    return x >= 2 and bool(fmpz(x).is_prime())


@dataclass(frozen=True)
class ShiftGateForm:
    """One exploitable shift: coprime_part(a * v - b) must be prime or the
    square of a prime for every witness n = modulus * v."""

    shift: int
    a: int
    b: int


@dataclass(frozen=True)
class ShiftGate:
    modulus: int
    jmax: int
    forms: tuple[ShiftGateForm, ...]
    #: per prime p | modulus: class-exponent cap K with LUT over v mod p^K
    lut_exponents: dict[int, int]
    #: per prime p | modulus: alive[c] over c in range(p^K)
    alive: dict[int, NDArray[np.bool_]]
    #: bound on the modulus-smooth part of any form value on alive classes
    max_alive_smooth: int

    def budget(self, j: int) -> int:
        return j + 2

    @property
    def smooth_primes(self) -> tuple[int, ...]:
        return tuple(sorted(self.lut_exponents))

    def coprime_part(self, x: int) -> int:
        for p in self.smooth_primes:
            while x % p == 0:
                x //= p
        return x

    def class_alive(self, v: int) -> bool:
        return all(
            bool(self.alive[p][v % (p ** self.lut_exponents[p])]) for p in self.smooth_primes
        )

    def form_alive(self, form: ShiftGateForm, v: int) -> bool:
        part = self.coprime_part(form.a * v - form.b)
        if part <= 1:
            return part == 1
        if _is_prime(part):
            return True
        root = math.isqrt(part)
        return root * root == part and _is_prime(root)

    def survivor(self, v: int) -> bool:
        return self.class_alive(v) and all(self.form_alive(f, v) for f in self.forms)


def _default_lut_exponents(modulus_factors: dict[int, int]) -> dict[int, int]:
    # Deep enough that every exploitable shift provably kills its deep
    # classes (checked in derive_shift_gate), small enough to enumerate.
    caps = {}
    for p, a in modulus_factors.items():
        caps[p] = a + {2: 4, 3: 3}.get(p, 2)
    return caps


def _exponent_lut(modulus: int, j: int, p: int, cap: int) -> list[int | None]:
    """v_p(modulus * c - j) per class c mod p^cap; None marks a deep class
    (exponent >= cap, not determined by the class)."""
    mod = p**cap
    lut: list[int | None] = []
    for c in range(mod):
        val = modulus * c - j
        e = _vp(val, p) if val != 0 else cap
        lut.append(e if e < cap else None)
    return lut


def derive_shift_gate(modulus: int, *, jmax: int = 64) -> ShiftGate:
    """Derive the exact divisor gate for ``n = modulus * v`` from the modulus alone."""
    factors = _factorize(modulus)
    if any(e < 1 for e in factors.values()) or modulus < 2:
        raise ValueError("modulus must be an integer >= 2")
    caps = _default_lut_exponents(factors)
    primes = tuple(sorted(factors))

    luts: dict[int, list[list[int | None]]] = {
        p: [_exponent_lut(modulus, j, p, caps[p]) for j in range(1, jmax + 1)] for p in primes
    }

    def min_exponent(j: int, p: int) -> int:
        return min((e for e in luts[p][j - 1] if e is not None), default=caps[p])

    # Alive tables: class c mod p^cap dies when some shift's budget drops
    # below 2 with every other prime at its minimal exponent.  Deep classes
    # are judged at exponent cap; deeper members only shrink the budget, so
    # the kill extends to them.
    alive = {p: np.ones(p ** caps[p], dtype=np.bool_) for p in primes}
    for j in range(1, jmax + 1):
        budget = j + 2
        gen = {p: min_exponent(j, p) + 1 for p in primes}
        total = math.prod(gen.values())
        for p in primes:
            other = total // gen[p]
            lut = luts[p][j - 1]
            for c in range(p ** caps[p]):
                e = lut[c]
                e_eff = caps[p] if e is None else e
                if budget // (other * (e_eff + 1)) < 2:
                    alive[p][c] = False

    # Exploitable shifts: every alive joint exponent assignment has B <= 3.
    forms: list[ShiftGateForm] = []
    for j in range(1, jmax + 1):
        budget = j + 2
        per_prime: list[list[int]] = []
        for p in primes:
            lut = luts[p][j - 1]
            exps = sorted({e for e in lut if e is not None})
            if any(e is None for e in lut):
                exps.append(caps[p])
            per_prime.append(exps)

        worst = [0]  # max alive budget; 0 when nothing alive

        def rec(i: int, tf: int) -> None:
            if tf > budget:
                return
            if i == len(per_prime):
                b = budget // tf
                if b >= 2:
                    worst[0] = max(worst[0], b)
                return
            for e in per_prime[i]:
                rec(i + 1, tf * (e + 1))

        rec(0, 1)
        if 2 <= worst[0] <= 3:
            f_generic = math.prod(p ** min(_vp(j, p), factors[p]) for p in primes)
            forms.append(ShiftGateForm(shift=j, a=modulus // f_generic, b=j // f_generic))

    # Soundness data for the sieve backends: on alive classes, the
    # modulus-smooth part of each form value is bounded.  This requires that
    # every exploitable shift kills its deep classes, which the budget
    # arithmetic above guarantees only implicitly - check it explicitly.
    max_smooth = 1
    for form in forms:
        smooth = 1
        for p in primes:
            # modulus*v - j = F_generic * (a*v - b), so the exponent of p in
            # the form value is the LUT exponent minus the generic one.
            lut = luts[p][form.shift - 1]
            gen_e = min(_vp(form.shift, p), factors[p])
            best_form = 0
            for c in range(p ** caps[p]):
                if not alive[p][c]:
                    continue
                e = lut[c]
                if e is None:
                    raise AssertionError(
                        f"shift {form.shift}: alive deep class mod {p}^{caps[p]}; "
                        "smooth bound unavailable"
                    )
                best_form = max(best_form, e - gen_e)
            smooth *= p**best_form
        max_smooth = max(max_smooth, smooth)

    return ShiftGate(
        modulus=modulus,
        jmax=jmax,
        # Ascending coefficient: scan backends test forms in list order, so
        # the smallest values (cheapest primality tests, and u64 rather than
        # u128 Montgomery at campaign scale) reject most candidates first.
        forms=tuple(sorted(forms, key=lambda f: f.a)),
        lut_exponents=caps,
        alive=alive,
        max_alive_smooth=max_smooth,
    )


def verify_shift_gate(
    gate: ShiftGate,
    *,
    trials: int = 2000,
    small_limit: int = 4000,
    seed: int = 647,
) -> None:
    """Re-check the derivation against direct factorization.  Raises on any
    disagreement; passing means every gate kill is a true kill on the tested
    ranges and the exponent tables are exact."""
    import random

    rng = random.Random(seed)
    modulus, jmax = gate.modulus, gate.jmax
    caps = gate.lut_exponents

    for _ in range(trials):
        v = rng.randrange(1, 10**15)
        j = rng.randrange(1, jmax + 1)
        for p in gate.smooth_primes:
            e = _exponent_lut(modulus, j, p, caps[p])[v % (p ** caps[p])]
            if e is not None and _vp(modulus * v - j, p) != e:
                raise AssertionError(f"exponent LUT wrong: v={v} j={j} p={p}")

    # The gate's domain starts where every form value is at least 2; below
    # that (n - j nonpositive or trivially small) the shift conditions are
    # vacuous and the scan's precondition excludes the range anyway.
    v_floor = max(
        (form.b + 2 + form.a - 1) // form.a for form in gate.forms
    )
    for v in range(v_floor, v_floor + small_limit):
        if not gate.survivor(v):
            if not any(_tau(modulus * v - j) > j + 2 for j in range(1, jmax + 1)):
                raise AssertionError(f"false kill at v={v}")

    for _ in range(trials):
        v = rng.randrange(10**6, 10**9)
        if not gate.class_alive(v):
            continue
        for form in gate.forms:
            if not gate.form_alive(form, v):
                if _tau(modulus * v - form.shift) <= form.shift + 2:
                    raise AssertionError(f"form j={form.shift} false kill at v={v}")
                break


@dataclass(frozen=True)
class ShiftGateStats:
    scanned: int
    wheel_alive: int
    sieve_survivors: int
    survivors: int


class ShiftGateScanPlan:
    """Retained scan plan: wheel classes, packed alive tables, and form data
    ready for repeated segment scans over disjoint v-intervals."""

    def __init__(
        self,
        gate: ShiftGate,
        *,
        wheel_primes: tuple[int, ...] = (17, 19, 23),
        sieve_bound: int = 1 << 15,
    ) -> None:
        if any(gate.modulus % q == 0 for q in wheel_primes):
            raise ValueError("wheel primes must not divide the gate modulus")
        self._gate = gate
        self._sieve_bound = int(sieve_bound)
        self._wheel_primes = tuple(int(q) for q in wheel_primes)

        # Wheel: v mod (base * prod(wheel_primes)) restricted to classes that
        # survive the mod-base projection of the p=2 table and the per-prime
        # form-residue kills.  The base is the p=2 LUT modulus itself, so the
        # whole 2-adic table folds into the wheel.
        p2_cap = gate.lut_exponents.get(2)
        base = 2**p2_cap if p2_cap is not None else 1
        base_classes = (
            np.flatnonzero(gate.alive[2]).astype(np.uint64)
            if base > 1
            else np.zeros(1, dtype=np.uint64)
        )

        # Iterated CRT: extend the class list one wheel prime at a time.
        # Kept vectorized because the extended wheel reaches ~1e9 with ~1e7
        # classes; enumerating the full wheel is not an option.
        classes = base_classes
        modulus = base
        for q in self._wheel_primes:
            dead = {
                int((form.b * pow(form.a, -1, q)) % q)
                for form in gate.forms
                if form.a % q != 0
            }
            alive = np.array(
                [s for s in range(q) if s not in dead], dtype=np.uint64
            )
            inv_m = pow(modulus, -1, q)
            # v = c + modulus * ((s - c) * inv_m mod q) hits v = c (mod
            # modulus) and v = s (mod q).
            offset = ((alive[None, :] + (q - classes[:, None] % q)) * inv_m) % q
            classes = (classes[:, None] + modulus * offset).ravel()
            modulus *= q
        classes.sort()
        self._wheel = int(modulus)
        self._classes = classes

    @property
    def gate(self) -> ShiftGate:
        return self._gate

    @property
    def wheel(self) -> int:
        return self._wheel

    @property
    def wheel_classes(self) -> NDArray[np.uint64]:
        return self._classes

    @property
    def sieve_bound(self) -> int:
        return self._sieve_bound

    @property
    def sieve_low(self) -> int:
        """First sieve prime: past every wheel and smooth prime, whose kills
        the wheel classes and lookup tables already encode."""
        floor = max((*self._wheel_primes, *self._gate.smooth_primes))
        candidate = floor + 1
        while any(candidate % d == 0 for d in range(2, int(math.isqrt(candidate)) + 1)):
            candidate += 1
        return candidate

    def _check_precondition(self, v_start: int) -> None:
        gate = self._gate
        a_min = min(f.a for f in gate.forms)
        b_max = max(f.b for f in gate.forms)
        part_min = (a_min * v_start - b_max) // gate.max_alive_smooth
        if part_min <= self._sieve_bound**2:
            raise ValueError(
                "scan range too low for exact sieving: coprime parts could "
                f"collide with q or q^2 (part_min={part_min}, "
                f"sieve_bound^2={self._sieve_bound ** 2}); lower sieve_bound "
                "or raise v_start"
            )
        max_value = max(f.a for f in gate.forms) * (v_start) + max(f.b for f in gate.forms)
        if max_value >= _MR_LIMIT:
            raise ValueError("form values exceed the deterministic MR base-set limit")

    def scan_reference(
        self, v_start: int, v_count: int
    ) -> tuple[NDArray[np.uint64], ShiftGateStats]:
        """Direct contract evaluation; exact at any scale, slow beyond ~1e6."""
        self._check_precondition(v_start)
        gate = self._gate
        wheel_alive = 0
        survivors = []
        class_set = frozenset(int(c) for c in self._classes)
        for v in range(v_start, v_start + v_count):
            if v % self._wheel not in class_set:
                continue
            if not gate.class_alive(v):
                continue
            wheel_alive += 1
            if all(gate.form_alive(f, v) for f in gate.forms):
                survivors.append(v)
        return (
            np.array(survivors, dtype=np.uint64),
            ShiftGateStats(
                scanned=v_count,
                wheel_alive=wheel_alive,
                sieve_survivors=wheel_alive,
                survivors=len(survivors),
            ),
        )

    def scan(
        self, v_start: int, v_count: int, *, backend: str = "auto"
    ) -> tuple[NDArray[np.uint64], ShiftGateStats]:
        self._check_precondition(v_start)
        if backend == "reference":
            return self.scan_reference(v_start, v_count)
        if backend in ("auto", "hip"):
            try:
                return self._scan_hip(v_start, v_count)
            except Exception:
                if backend == "hip":
                    raise
        if backend in ("auto", "native"):
            return self._scan_native(v_start, v_count)
        raise ValueError(f"unknown backend {backend!r}")

    # Native and HIP dispatch live in _shift_gates_native.py / hip.py; the
    # imports are deferred so the pure-python surface works without them.
    def _scan_native(self, v_start: int, v_count: int):
        from ._shift_gates_native import scan_native

        return scan_native(self, v_start, v_count)

    def _scan_hip(self, v_start: int, v_count: int):
        from .hip import shift_gate_scan_hip

        return shift_gate_scan_hip(self, v_start, v_count)

    def packed_luts(self) -> tuple[NDArray[np.uint64], NDArray[np.uint64], NDArray[np.uint64]]:
        """(primes^caps moduli, bit offsets, packed alive bits) for the
        native backends."""
        gate = self._gate
        moduli = []
        offsets = []
        bits: list[int] = []
        total = 0
        for p in gate.smooth_primes:
            table = gate.alive[p]
            moduli.append(len(table))
            offsets.append(total)
            total += len(table)
            bits.extend(int(b) for b in table)
        packed = np.zeros((total + 63) // 64, dtype=np.uint64)
        for i, b in enumerate(bits):
            if b:
                packed[i // 64] |= np.uint64(1 << (i % 64))
        return (
            np.array(moduli, dtype=np.uint64),
            np.array(offsets, dtype=np.uint64),
            packed,
        )
