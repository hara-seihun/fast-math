"""Elliptic-curve rank hunting: Mestre families, Nagao sieving, quartic points.

Three contracts, all stated over mathematical data rather than one search:

``mestre_quartic``
    The exact construction.  For a sextuple ``a`` on the Mestre locus
    (``p1 = 0`` and ``12 p5 = 5 p2 p3`` in power sums) and a rational ``T``,
    ``p6(x-T) p6(x+T) = g(x)^2 - r(x)`` with ``g`` monic of degree six.  The
    locus condition is exactly what forces ``deg r <= 4``.  The fibre is the
    Jacobian of ``y^2 = r(x)``, and the twelve points ``x = a_i +- T``,
    ``y = g(x)`` are rational on it, so every fibre has rank at least twelve
    minus relations.

``MestreApTables``
    A retained plan holding ``A_p[t] = a_p`` of the fibre at ``T = t`` for every
    ``t`` in ``F_p`` and every prime up to a bound, because ``a_p`` of a fibre
    depends only on ``T mod p``.  Building costs ``O(p^2)`` per prime; scoring
    any rational ``T = m/n`` afterwards costs one gather per prime, which turns
    a Mestre-Nagao sweep over millions of fibres into gather-and-FMA.

``quartic_points``
    Rational points on ``y^2 = r(x)`` with ``x = u/w`` in a box, by quadratic
    residue bit-sieving at small primes followed by an exact integer square
    test.  Coefficients are arbitrary Python integers, which they must be: the
    interesting quartics have coefficients of thirty digits while their
    interesting points have small ``(u, w)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt, log
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._native import (
    NativeUnavailable,
    elliptic_mestre_ap_tables_native,
    elliptic_nagao_scores_native,
    elliptic_quartic_sieve_native,
    native_available,
)

EllipticBackend = Literal["auto", "native", "reference"]

__all__ = [
    "mestre_locus_sextuples",
    "MestreApTables",
    "QuarticFibre",
    "WeierstrassCover",
    "jacobian_invariants",
    "quartic_to_weierstrass",
    "mestre_ap_tables",
    "mestre_locus_residual",
    "mestre_quartic",
    "power_sums",
    "primes_upto",
    "quartic_points",
]


def _integers(values: Sequence[object]) -> list[int]:
    return [int(value) for value in values]


def _rational(value: object) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(str(value))


def power_sums(sextuple: Sequence[object], count: int = 5) -> list[int]:
    """Power sums ``p_1 .. p_count`` of a sextuple of integers."""
    roots = _integers(sextuple)
    return [sum(root**power for root in roots) for power in range(1, count + 1)]


def mestre_locus_residual(sextuple: Sequence[object]) -> tuple[int, int]:
    """``(p1, 12 p5 - 5 p2 p3)``; both zero exactly on the Mestre locus."""
    p1, p2, p3, _, p5 = power_sums(sextuple, 5)
    return p1, 12 * p5 - 5 * p2 * p3


def primes_upto(bound: int) -> NDArray[np.uint32]:
    """Primes at most ``bound`` as uint32."""
    if bound < 2:
        return np.zeros(0, dtype=np.uint32)
    sieve = np.ones(int(bound) + 1, dtype=bool)
    sieve[:2] = False
    for candidate in range(2, isqrt(int(bound)) + 1):
        if sieve[candidate]:
            sieve[candidate * candidate :: candidate] = False
    return np.flatnonzero(sieve).astype(np.uint32)


def mestre_locus_sextuples(
    bound: int,
    *,
    primitive: bool = True,
) -> NDArray[np.int64]:
    """Integer sextuples on the Mestre locus with entries bounded in size.

    Solves the locus directly rather than parametrizing it.  With ``p1 = 0``,
    write ``s = a5 + a6`` and ``m = a5 a6``; the condition ``12 p5 = 5 p2 p3``
    becomes a *quadratic* in ``m`` once ``a1..a4`` are fixed,

        30 s m^2 + (15 s A2 + 10 A3 - 35 s^3) m
            + 12 A5 + 12 s^5 - 5 (A2 + s^2)(A3 + s^3) = 0,

    where ``A_k`` is the ``k``-th power sum of ``a1..a4``.  An integer root whose
    companion ``s^2 - 4m`` is a perfect square splits off the last two entries.

    Returned sextuples are sorted, deduplicated, and by default primitive: one
    representative per scaling class, since ``a -> lambda a`` gives isomorphic
    fibres.
    """
    bound = int(bound)
    if bound < 1:
        raise ValueError("bound must be positive")

    span = np.arange(-bound, bound + 1, dtype=np.int64)
    results: set[tuple[int, ...]] = set()
    for a1 in range(-bound, 1):
        for a2 in range(a1, bound + 1):
            for a3 in range(a2, bound + 1):
                a4 = span[span >= a3]
                if len(a4) == 0:
                    continue
                head = np.array([a1, a2, a3], dtype=np.int64)
                s = -(head.sum() + a4)
                powers = [head.astype(object).sum()]
                A2 = int((head**2).sum()) + a4.astype(object) ** 2
                A3 = int((head**3).sum()) + a4.astype(object) ** 3
                A5 = int((head**5).sum()) + a4.astype(object) ** 5
                s = s.astype(object)
                quadratic = 30 * s
                linear = 15 * s * A2 + 10 * A3 - 35 * s**3
                constant = 12 * A5 + 12 * s**5 - 5 * (A2 + s**2) * (A3 + s**3)
                for index in range(len(a4)):
                    roots = _integer_roots(
                        int(quadratic[index]),
                        int(linear[index]),
                        int(constant[index]),
                    )
                    for product in roots:
                        sum_last = int(s[index])
                        square = sum_last * sum_last - 4 * product
                        if square < 0:
                            continue
                        root = isqrt(square)
                        if root * root != square or (sum_last + root) % 2 != 0:
                            continue
                        a5 = (sum_last + root) // 2
                        a6 = sum_last - a5
                        sextuple = tuple(
                            sorted([a1, a2, a3, int(a4[index]), a5, a6])
                        )
                        if max(abs(value) for value in sextuple) > bound:
                            continue
                        if all(value == 0 for value in sextuple):
                            continue
                        if primitive:
                            divisor = 0
                            for value in sextuple:
                                divisor = gcd(divisor, value)
                            sextuple = tuple(value // divisor for value in sextuple)
                            if sextuple[0] + sextuple[-1] < 0 or (
                                sextuple[0] + sextuple[-1] == 0
                                and sextuple[1] < -sextuple[-2]
                            ):
                                sextuple = tuple(sorted(-value for value in sextuple))
                        results.add(sextuple)
    if not results:
        return np.zeros((0, 6), dtype=np.int64)
    return np.array(sorted(results), dtype=np.int64)


def _integer_roots(quadratic: int, linear: int, constant: int) -> list[int]:
    """Integer roots of ``quadratic x^2 + linear x + constant``."""
    if quadratic == 0:
        if linear == 0:
            return []
        if constant % linear != 0:
            return []
        return [-constant // linear]
    square = linear * linear - 4 * quadratic * constant
    if square < 0:
        return []
    root = isqrt(square)
    if root * root != square:
        return []
    roots = []
    for sign in (root, -root):
        numerator = -linear + sign
        if numerator % (2 * quadratic) == 0:
            roots.append(numerator // (2 * quadratic))
    return roots


def _polynomial_product(left: list[int], right: list[int]) -> list[int]:
    product = [0] * (len(left) + len(right) - 1)
    for i, value in enumerate(left):
        if value:
            for j, other in enumerate(right):
                product[i + j] += value * other
    return product


def _sextic(sextuple: Sequence[object]) -> list[int]:
    polynomial = [1]
    for root in _integers(sextuple):
        polynomial = _polynomial_product(polynomial, [-root, 1])
    return polynomial


def _shifted_in_parameter(sextic: list[int], sign: int) -> list[list[int]]:
    """``p6(x + sign*T)`` as x-degree major, T-degree minor coefficients."""
    shifted = [[0] * 7 for _ in range(7)]
    for degree, coefficient in enumerate(sextic):
        binomial = 1
        for taken in range(degree + 1):
            shifted[degree - taken][taken] += coefficient * binomial * sign**taken
            binomial = binomial * (degree - taken) // (taken + 1)
    return shifted


def _add(left: list[Fraction], right: list[Fraction]) -> list[Fraction]:
    size = max(len(left), len(right))
    zero = Fraction(0)
    return [
        (left[i] if i < len(left) else zero) + (right[i] if i < len(right) else zero)
        for i in range(size)
    ]


def _multiply(left: list[Fraction], right: list[Fraction]) -> list[Fraction]:
    product = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, value in enumerate(left):
        if value:
            for j, other in enumerate(right):
                product[i + j] += value * other
    return product


def _trim(polynomial: list[Fraction]) -> list[Fraction]:
    while len(polynomial) > 1 and polynomial[-1] == 0:
        polynomial.pop()
    return polynomial


def _family_in_parameter(
    sextuple: Sequence[object],
) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    """``r`` and ``g`` coefficients, each a polynomial in the parameter ``T``."""
    sextic = _sextic(sextuple)
    left = _shifted_in_parameter(sextic, -1)
    right = _shifted_in_parameter(sextic, +1)

    product = [[0] * 13 for _ in range(13)]
    for i, left_row in enumerate(left):
        for j, right_row in enumerate(right):
            for u, left_value in enumerate(left_row):
                if left_value:
                    for v, right_value in enumerate(right_row):
                        if right_value:
                            product[i + j][u + v] += left_value * right_value

    root: list[list[Fraction]] = [[Fraction(0)] for _ in range(7)]
    root[6] = [Fraction(1)]
    for degree in range(5, -1, -1):
        accumulated = [Fraction(0)]
        for i in range(degree + 1, 7):
            j = degree + 6 - i
            if 0 <= j <= 6:
                accumulated = _add(accumulated, _multiply(root[i], root[j]))
        target = [Fraction(value) for value in product[degree + 6]]
        root[degree] = [
            value / 2 for value in _add(target, [-value for value in accumulated])
        ]

    quartic: list[list[Fraction]] = []
    for degree in range(5):
        accumulated = [Fraction(0)]
        for i in range(7):
            j = degree - i
            if 0 <= j <= 6:
                accumulated = _add(accumulated, _multiply(root[i], root[j]))
        target = [Fraction(value) for value in product[degree]]
        quartic.append(_add(accumulated, [-value for value in target]))

    return [_trim(row) for row in quartic], [_trim(row) for row in root]


@dataclass(frozen=True)
class QuarticFibre:
    """One fibre ``y^2 = sum coefficients[k] x^k`` with its twelve base points."""

    coefficients: tuple[int, int, int, int, int]
    points: tuple[tuple[Fraction, Fraction], ...]
    parameter: Fraction

    def evaluate(self, x: Fraction) -> Fraction:
        return sum(
            coefficient * x**degree
            for degree, coefficient in enumerate(self.coefficients)
        )

    def jacobian_invariants(self) -> tuple[int, int]:
        return jacobian_invariants(self.coefficients)


def jacobian_invariants(coefficients: Sequence[int]) -> tuple[int, int]:
    """``(a4, a6)`` of ``Y^2 = X^3 + a4 X + a6``, the Jacobian of the quartic."""
    r0, r1, r2, r3, r4 = (int(value) for value in coefficients)
    first = 12 * r4 * r0 - 3 * r3 * r1 + r2 * r2
    second = (
        72 * r4 * r2 * r0
        + 9 * r3 * r2 * r1
        - 27 * r4 * r1 * r1
        - 27 * r3 * r3 * r0
        - 2 * r2**3
    )
    return -27 * first, -27 * second


@dataclass(frozen=True)
class WeierstrassCover:
    """A Weierstrass model of ``y^2 = quartic`` together with the point map.

    Built from one rational point of the quartic: shifting that point to the
    origin makes the constant term a square ``q^2``, and the classical map then
    sends quartic points to the Weierstrass model, which is the Jacobian.
    """

    a_invariants: tuple[Fraction, Fraction, Fraction, Fraction, Fraction]
    shift: Fraction
    square_root: Fraction
    shifted_coefficients: tuple[Fraction, Fraction, Fraction, Fraction, Fraction]

    def image(self, x: object, y: object) -> tuple[Fraction, Fraction] | None:
        """Image of a quartic point, or ``None`` for the point over the origin."""
        u = _rational(x) - self.shift
        v = _rational(y)
        if u == 0:
            return None
        _, d, c, _, _ = self.shifted_coefficients
        q = self.square_root
        big_x = (2 * q * (v + q) + d * u) / u**2
        big_y = (
            4 * q**2 * (v + q) + 2 * q * (d * u + c * u**2) - d**2 * u**2 / (2 * q)
        ) / u**3
        return big_x, big_y

    def contains(self, x: Fraction, y: Fraction) -> bool:
        a1, a2, a3, a4, a6 = self.a_invariants
        return y * y + a1 * x * y + a3 * y == x**3 + a2 * x * x + a4 * x + a6


def quartic_to_weierstrass(
    coefficients: Sequence[object],
    base_point: tuple[object, object],
) -> WeierstrassCover:
    """Weierstrass model of ``y^2 = sum coefficients[k] x^k`` through a point.

    The quartic must be nonsingular and the base point must lie on it with
    nonzero ordinate.  The resulting curve is the Jacobian of the quartic.
    """
    if len(coefficients) != 5:
        raise ValueError("a quartic needs five coefficients, constant term first")
    values = [_rational(value) for value in coefficients]
    shift = _rational(base_point[0])
    ordinate = _rational(base_point[1])
    if ordinate == 0:
        raise ValueError("the base point must have nonzero ordinate")

    shifted = [Fraction(0)] * 5
    for degree, coefficient in enumerate(values):
        binomial = 1
        for taken in range(degree + 1):
            shifted[degree - taken] += coefficient * binomial * shift**taken
            binomial = binomial * (degree - taken) // (taken + 1)
    if shifted[0] != ordinate * ordinate:
        raise ValueError("the base point does not lie on the quartic")

    q = ordinate
    a, b, c, d = shifted[4], shifted[3], shifted[2], shifted[1]
    a1 = d / q
    a2 = c - d * d / (4 * q * q)
    a3 = 2 * q * b
    a4 = -4 * q * q * a
    a6 = a2 * a4
    return WeierstrassCover(
        (a1, a2, a3, a4, a6),
        shift,
        q,
        (shifted[0], shifted[1], shifted[2], shifted[3], shifted[4]),
    )


def mestre_quartic(sextuple: Sequence[object], parameter: object) -> QuarticFibre:
    """The fibre at ``T = parameter``, cleared to integer coefficients.

    Clearing multiplies the quartic by a square, which is an isomorphism of
    ``y^2 = r(x)``, so the returned points still satisfy the returned equation.
    """
    p1, residual = mestre_locus_residual(sextuple)
    if p1 != 0 or residual != 0:
        raise ValueError(
            "sextuple is not on the Mestre locus: need p1 = 0 and 12 p5 = 5 p2 p3"
        )

    quartic, root = _family_in_parameter(sextuple)
    parameter = _rational(parameter)
    values = [
        sum(coefficient * parameter**degree for degree, coefficient in enumerate(row))
        for row in quartic
    ]
    scale = 1
    for value in values:
        scale = scale * value.denominator // gcd(scale, value.denominator)
    coefficients = tuple(int(value * scale**2) for value in values)

    def sextic_root(x: Fraction) -> Fraction:
        total = Fraction(0)
        for degree, row in enumerate(root):
            coefficient = sum(
                value * parameter**power for power, value in enumerate(row)
            )
            total += coefficient * x**degree
        return total

    points = []
    for value in _integers(sextuple):
        for sign in (1, -1):
            x = Fraction(value) + sign * parameter
            points.append((x, sextic_root(x) * scale))
    return QuarticFibre(coefficients, tuple(points), parameter)


def _reference_ap_table(sextuple: Sequence[object], prime: int) -> NDArray[np.int32]:
    modulus = int(prime)
    sextic = np.array(_sextic(sextuple), dtype=object) % modulus
    sextic = sextic.astype(np.int64)
    parameters = np.arange(modulus, dtype=np.int64)

    binomials = np.zeros((7, 7), dtype=np.int64)
    for upper in range(7):
        value = 1
        for lower in range(upper + 1):
            binomials[upper, lower] = value
            value = value * (upper - lower) // (lower + 1)

    powers = np.ones((modulus, 7), dtype=np.int64)
    for power in range(1, 7):
        powers[:, power] = powers[:, power - 1] * parameters % modulus

    def shifted(sign: int) -> NDArray[np.int64]:
        out = np.zeros((modulus, 7), dtype=np.int64)
        for degree in range(7):
            for taken in range(degree + 1):
                weight = sextic[degree] * binomials[degree, taken] % modulus
                signed = pow(sign % modulus, taken, modulus)
                out[:, degree - taken] = (
                    out[:, degree - taken] + weight * signed % modulus * powers[:, taken]
                ) % modulus
        return out

    left, right = shifted(-1), shifted(+1)
    product = np.zeros((modulus, 13), dtype=np.int64)
    for i in range(7):
        for j in range(7):
            product[:, i + j] = (product[:, i + j] + left[:, i] * right[:, j]) % modulus

    half = pow(2, modulus - 2, modulus)
    root = np.zeros((modulus, 7), dtype=np.int64)
    root[:, 6] = 1
    for degree in range(5, -1, -1):
        accumulated = np.zeros(modulus, dtype=np.int64)
        for i in range(degree + 1, 7):
            j = degree + 6 - i
            if 0 <= j <= 6:
                accumulated = (accumulated + root[:, i] * root[:, j]) % modulus
        root[:, degree] = (product[:, degree + 6] - accumulated) % modulus * half % modulus

    quartic = np.zeros((modulus, 5), dtype=np.int64)
    for degree in range(5):
        accumulated = np.zeros(modulus, dtype=np.int64)
        for i in range(degree + 1):
            accumulated = (accumulated + root[:, i] * root[:, degree - i]) % modulus
        quartic[:, degree] = (accumulated - product[:, degree]) % modulus

    characters = -np.ones(modulus, dtype=np.int8)
    characters[0] = 0
    characters[(np.arange(1, modulus, dtype=np.int64) ** 2) % modulus] = 1

    # Score the Jacobian cubic, not the quartic: the quartic model degenerates
    # modulo p much more often than the curve does, and its character sum then
    # reaches size p instead of a trace bounded by 2 sqrt(p).
    r0, r1, r2, r3, r4 = (quartic[:, degree] for degree in range(5))
    invariant_i = (12 * r4 * r0 - 3 * r3 * r1 + r2 * r2) % modulus
    invariant_j = (
        72 * r4 * r2 % modulus * r0
        + 9 * r3 * r2 % modulus * r1
        - 27 * r4 * r1 % modulus * r1
        - 27 * r3 * r3 % modulus * r0
        - 2 * r2 * r2 % modulus * r2
    ) % modulus
    singular = (
        4 * invariant_i % modulus * invariant_i % modulus * invariant_i
        - invariant_j * invariant_j
    ) % modulus == 0
    linear = (-27 * invariant_i) % modulus
    constant = (-27 * invariant_j) % modulus

    points = np.arange(modulus, dtype=np.int64)
    cubes = points * points % modulus * points % modulus
    table = np.zeros(modulus, dtype=np.int64)
    step = max(1, (1 << 22) // max(modulus, 1))
    for low in range(0, modulus, step):
        high = min(modulus, low + step)
        value = (
            cubes[None, :]
            + linear[low:high, None] * points[None, :]
            + constant[low:high, None]
        ) % modulus
        table[low:high] = -characters[value].sum(axis=1, dtype=np.int64)
    table[singular] = 0
    return table.astype(np.int32)


@dataclass(frozen=True)
class MestreApTables:
    """Retained ``a_p`` tables for one Mestre family, indexed by ``T mod p``."""

    sextuple: tuple[int, ...]
    primes: NDArray[np.uint32]
    offsets: NDArray[np.uint64]
    tables: NDArray[np.int32]
    weights: NDArray[np.float64]
    elapsed_seconds: float
    backend: str

    @property
    def prime_bound(self) -> int:
        return int(self.primes[-1]) if len(self.primes) else 0

    def table(self, prime: int) -> NDArray[np.int32]:
        index = int(np.searchsorted(self.primes, prime))
        if index >= len(self.primes) or int(self.primes[index]) != int(prime):
            raise KeyError(f"prime {prime} is not tabulated")
        return self.tables[self.offsets[index] : self.offsets[index + 1]]

    def scores(
        self,
        numerators: ArrayLike,
        denominators: ArrayLike | None = None,
        *,
        backend: EllipticBackend = "auto",
        threads: int = 0,
        normalize: bool = False,
    ) -> NDArray[np.float64]:
        """Mestre-Nagao scores ``-sum_p A_p[m/n] log p / p`` of rational fibres.

        With ``normalize`` the score is divided by ``log`` of the prime bound,
        which is the quantity Nagao's heuristic compares against the rank.
        """
        numerator = np.ascontiguousarray(numerators, dtype=np.int64)
        if denominators is None:
            denominator = np.ones_like(numerator)
        else:
            denominator = np.ascontiguousarray(denominators, dtype=np.int64)
        if numerator.shape != denominator.shape:
            raise ValueError("numerators and denominators must have equal shape")
        if numerator.ndim != 1:
            raise ValueError("numerators must be one-dimensional")
        if np.any(denominator == 0):
            raise ValueError("denominators must be nonzero")

        scores = self._scores(numerator, denominator, backend, threads)
        if normalize:
            scores = scores / log(self.prime_bound)
        return scores

    def _scores(
        self,
        numerator: NDArray[np.int64],
        denominator: NDArray[np.int64],
        backend: EllipticBackend,
        threads: int,
    ) -> NDArray[np.float64]:
        if backend not in {"auto", "native", "reference"}:
            raise ValueError("backend must be 'auto', 'native', or 'reference'")
        if backend != "reference" and native_available():
            try:
                scores, _ = elliptic_nagao_scores_native(
                    self.tables,
                    self.primes,
                    self.weights,
                    self.offsets,
                    numerator,
                    denominator,
                    threads=threads,
                )
                return scores
            except (NativeUnavailable, OSError):
                if backend == "native":
                    raise
        if backend == "native":
            raise NativeUnavailable("fast-math native library is unavailable")

        scores = np.zeros(numerator.shape, dtype=np.float64)
        for index, prime in enumerate(self.primes):
            prime = int(prime)
            table = self.tables[self.offsets[index] : self.offsets[index + 1]]
            inverse = np.zeros(prime, dtype=np.int64)
            residues = np.arange(1, prime, dtype=np.int64)
            inverse[1:] = _power_mod(residues, prime - 2, prime)
            reduced = denominator % prime
            parameter = (numerator % prime) * inverse[reduced] % prime
            contribution = table[parameter].astype(np.float64)
            contribution[reduced == 0] = 0.0
            scores -= contribution * self.weights[index]
        return scores


def _power_mod(
    base: NDArray[np.int64], exponent: int, modulus: int
) -> NDArray[np.int64]:
    result = np.ones_like(base)
    factor = base % modulus
    while exponent:
        if exponent & 1:
            result = result * factor % modulus
        factor = factor * factor % modulus
        exponent >>= 1
    return result


def mestre_ap_tables(
    sextuple: Sequence[object],
    prime_bound: int,
    *,
    minimum_prime: int = 5,
    backend: EllipticBackend = "auto",
    threads: int = 0,
) -> MestreApTables:
    """Tabulate ``a_p`` of every fibre of a Mestre family, for all ``p <= bound``.

    ``a_p`` of the fibre at ``T`` depends only on ``T mod p``, so one table per
    prime scores unboundedly many rational fibres afterwards.
    """
    p1, residual = mestre_locus_residual(sextuple)
    if p1 != 0 or residual != 0:
        raise ValueError(
            "sextuple is not on the Mestre locus: need p1 = 0 and 12 p5 = 5 p2 p3"
        )
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")

    primes = primes_upto(prime_bound)
    primes = primes[primes >= max(3, int(minimum_prime))]
    offsets = np.zeros(len(primes) + 1, dtype=np.uint64)
    np.cumsum(primes.astype(np.uint64), out=offsets[1:])
    weights = np.array(
        [log(int(prime)) / int(prime) for prime in primes], dtype=np.float64
    )
    roots = np.ascontiguousarray(_integers(sextuple), dtype=np.int64)

    if backend != "reference" and native_available():
        try:
            tables, stats = elliptic_mestre_ap_tables_native(
                roots, primes, offsets, threads=threads
            )
            return MestreApTables(
                tuple(int(value) for value in roots),
                primes,
                offsets,
                tables,
                weights,
                float(stats.elapsed_seconds),
                "native",
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
    if backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")

    tables = np.concatenate(
        [_reference_ap_table(sextuple, int(prime)) for prime in primes]
    ) if len(primes) else np.zeros(0, dtype=np.int32)
    return MestreApTables(
        tuple(int(value) for value in roots),
        primes,
        offsets,
        tables.astype(np.int32),
        weights,
        0.0,
        "reference",
    )


def _homogeneous_value(coefficients: Sequence[int], u: int, w: int) -> int:
    total = 0
    for degree, coefficient in enumerate(coefficients):
        total += int(coefficient) * u**degree * w ** (4 - degree)
    return total


def _sieve_primes(coefficients: Sequence[int], count: int, bound: int) -> NDArray[np.uint32]:
    chosen = []
    for prime in primes_upto(bound):
        prime = int(prime)
        if prime < 5:
            continue
        residues = [int(value) % prime for value in coefficients]
        if all(residue == 0 for residue in residues):
            continue
        chosen.append(prime)
        if len(chosen) == count:
            break
    return np.array(chosen, dtype=np.uint32)


def _default_sieve_prime_count(pairs: int) -> int:
    """Enough primes that surviving candidates stay a manageable fraction.

    Each prime keeps about half the pairs, so ``k`` primes leave ``pairs / 2^k``.
    Aim to leave a few hundred thousand, and never fewer than twelve primes.
    """
    target = max(1, pairs // 200000)
    count = target.bit_length()
    return min(24, max(12, count))


def quartic_points(
    coefficients: Sequence[int],
    numerator_range: tuple[int, int],
    denominator_range: tuple[int, int] = (1, 1),
    *,
    sieve_prime_count: int | None = None,
    sieve_prime_bound: int = 400,
    capacity: int = 1 << 22,
    backend: EllipticBackend = "auto",
    threads: int = 0,
) -> list[tuple[Fraction, Fraction]]:
    """Rational points on ``y^2 = sum coefficients[k] x^k`` with ``x = u/w`` in a box.

    Small primes bit-sieve the box down to pairs where the homogenized quartic
    is a quadratic residue everywhere, and each survivor is settled by an exact
    integer square test.  Coefficients may be arbitrary Python integers.
    """
    if len(coefficients) != 5:
        raise ValueError("a quartic needs five coefficients, constant term first")
    low, high = int(numerator_range[0]), int(numerator_range[1])
    denominator_low, denominator_high = (
        int(denominator_range[0]),
        int(denominator_range[1]),
    )
    if high < low or denominator_high < denominator_low:
        raise ValueError("search ranges must be nondecreasing")
    if denominator_low < 1:
        raise ValueError("denominators must be positive")
    if backend not in {"auto", "native", "reference"}:
        raise ValueError("backend must be 'auto', 'native', or 'reference'")

    if sieve_prime_count is None:
        pairs = (high - low + 1) * (denominator_high - denominator_low + 1)
        sieve_prime_count = _default_sieve_prime_count(pairs)
    primes = _sieve_primes(coefficients, sieve_prime_count, sieve_prime_bound)
    residues = np.empty((len(primes), 5), dtype=np.uint32)
    for index, prime in enumerate(primes):
        for degree, coefficient in enumerate(coefficients):
            residues[index, degree] = int(coefficient) % int(prime)
    residues = np.ascontiguousarray(residues)

    use_native = backend != "reference" and native_available()
    if not use_native and backend == "native":
        raise NativeUnavailable("fast-math native library is unavailable")

    def sieve(first: int, last: int) -> list[NDArray[np.int64]]:
        """Candidates for denominators in [first, last], splitting on overflow."""
        if use_native:
            try:
                found, stats = elliptic_quartic_sieve_native(
                    residues,
                    primes,
                    (low, high),
                    (first, last),
                    capacity=capacity,
                    threads=threads,
                )
            except (NativeUnavailable, OSError):
                if backend == "native":
                    raise
                return [
                    _reference_quartic_sieve(residues, primes, (low, high), (first, last))
                ]
            if not stats.truncated:
                return [found]
            # The kernel reports the true count, so never drop candidates:
            # halve the denominator span until each piece fits.
            if first == last:
                raise MemoryError(
                    f"denominator {first} yields {int(stats.candidate_count)} "
                    f"candidates, above capacity {capacity}; raise capacity or "
                    "sieve_prime_count"
                )
            middle = (first + last) // 2
            return sieve(first, middle) + sieve(middle + 1, last)
        return [_reference_quartic_sieve(residues, primes, (low, high), (first, last))]

    blocks = [block for block in sieve(denominator_low, denominator_high) if len(block)]
    candidates = (
        np.concatenate(blocks) if blocks else np.zeros((0, 2), dtype=np.int64)
    )

    points: list[tuple[Fraction, Fraction]] = []
    for u, w in candidates:
        u, w = int(u), int(w)
        if gcd(u, w) != 1:
            continue
        value = _homogeneous_value(coefficients, u, w)
        if value < 0:
            continue
        root = isqrt(value)
        if root * root != value:
            continue
        x = Fraction(u, w)
        y = Fraction(root, w * w)
        points.append((x, y))
    return points


def _reference_quartic_sieve(
    residues: NDArray[np.uint32],
    primes: NDArray[np.uint32],
    numerator_range: tuple[int, int],
    denominator_range: tuple[int, int],
) -> NDArray[np.int64]:
    low, high = numerator_range
    numerators = np.arange(low, high + 1, dtype=np.int64)
    found: list[tuple[int, int]] = []
    for denominator in range(denominator_range[0], denominator_range[1] + 1):
        alive = np.ones(numerators.shape, dtype=bool)
        for index, prime in enumerate(primes):
            prime = int(prime)
            characters = -np.ones(prime, dtype=np.int8)
            characters[0] = 0
            characters[(np.arange(1, prime, dtype=np.int64) ** 2) % prime] = 1
            value = np.zeros(numerators.shape, dtype=np.int64)
            for degree in range(5):
                term = (
                    int(residues[index, degree])
                    * pow(denominator, 4 - degree, prime)
                    % prime
                )
                value = (value + term * _power_mod_scalar(numerators, degree, prime)) % prime
            alive &= characters[value] >= 0
        for numerator in numerators[alive]:
            found.append((int(numerator), denominator))
    if not found:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(found, dtype=np.int64)


def _power_mod_scalar(
    base: NDArray[np.int64], exponent: int, modulus: int
) -> NDArray[np.int64]:
    result = np.ones_like(base)
    factor = base % modulus
    for _ in range(exponent):
        result = result * factor % modulus
    return result
