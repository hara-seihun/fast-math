from fractions import Fraction

import numpy as np
import pytest

from fast_math.elliptic import (
    quartic_to_weierstrass,
    jacobian_invariants,
    mestre_ap_tables,
    mestre_locus_residual,
    mestre_quartic,
    quartic_points,
)


# The centred sextuple of the leaderboard's rank-19 record curve, whose fibre
# at T = 743 has minimal discriminant of 156.3436 nats.
RANK19 = [-557, -444, -7, 196, 311, 501]
# The rank-17 record curve is this family at T = 2454.
RANK17 = [-1146, -2304, -654, 3054, 2880, -1830]


def test_locus_residual_detects_membership():
    assert mestre_locus_residual(RANK19) == (0, 0)
    assert mestre_locus_residual(RANK17) == (0, 0)
    assert mestre_locus_residual([1, 2, 3, -4, -5, 3]) != (0, 0)


def test_off_locus_sextuple_is_rejected():
    with pytest.raises(ValueError):
        mestre_quartic([1, 2, 3, -4, -5, 3], 7)


def test_twelve_base_points_lie_on_the_fibre():
    for parameter in (1, 743, Fraction(2429, 5), Fraction(-13, 4)):
        fibre = mestre_quartic(RANK19, parameter)
        assert len(fibre.points) == 12
        for x, y in fibre.points:
            assert y * y == fibre.evaluate(x)


def _j_invariant_of_weierstrass(a1, a2, a3, a4, a6):
    b2 = a1 * a1 + 4 * a2
    b4 = 2 * a4 + a1 * a3
    b6 = a3 * a3 + 4 * a6
    b8 = (
        a1 * a1 * a6
        + 4 * a2 * a6
        - a1 * a3 * a4
        + a2 * a3 * a3
        - a4 * a4
    )
    c4 = b2 * b2 - 24 * b4
    discriminant = (
        -b2 * b2 * b8 - 8 * b4**3 - 27 * b6 * b6 + 9 * b2 * b4 * b6
    )
    return Fraction(c4**3, discriminant)


def _j_invariant_of_short(a4, a6):
    return _j_invariant_of_weierstrass(0, 0, 0, a4, a6)


def test_known_record_curves_are_reproduced():
    # Leaderboard curve #159, rank 17, in its global minimal model, and curve
    # #240, the rank-19 record.  The construction has to land on them exactly.
    cases = [
        (RANK17, 2454, (1, -1, 0, -1321749187079172070, 247663242328119893241310696)),
        (
            RANK19,
            743,
            (
                1,
                -1,
                0,
                -29015852737941337052556,
                1950033817928958623296270251397996,
            ),
        ),
    ]
    for sextuple, parameter, minimal_model in cases:
        fibre = mestre_quartic(sextuple, parameter)
        assert _j_invariant_of_short(
            *jacobian_invariants(fibre.coefficients)
        ) == _j_invariant_of_weierstrass(*minimal_model)


def test_native_tables_match_reference():
    reference = mestre_ap_tables(RANK19, 200, backend="reference")
    native = mestre_ap_tables(RANK19, 200, backend="native")
    assert native.backend == "native"
    assert np.array_equal(reference.primes, native.primes)
    assert np.array_equal(reference.tables, native.tables)


def _trace_from_exact_fibre(sextuple, parameter, prime):
    """a_p of the Jacobian by reducing the exact integer fibre, an independent path."""
    fibre = mestre_quartic(sextuple, parameter)
    a4, a6 = jacobian_invariants(fibre.coefficients)
    a4, a6 = a4 % prime, a6 % prime
    if (4 * a4**3 + 27 * a6**2) % prime == 0:
        return 0
    characters = [0] * prime
    for root in range(1, prime):
        characters[root * root % prime] = 1
    for residue in range(1, prime):
        if characters[residue] == 0:
            characters[residue] = -1
    return -sum(
        characters[(x * x * x + a4 * x + a6) % prime] for x in range(prime)
    )


def test_every_table_entry_matches_the_exact_fibre():
    tables = mestre_ap_tables(RANK19, 60)
    for prime in (11, 31, 59):
        table = tables.table(prime)
        assert len(table) == prime
        for parameter in range(prime):
            assert int(table[parameter]) == _trace_from_exact_fibre(
                RANK19, parameter, prime
            ), (prime, parameter)


def test_all_entries_obey_the_hasse_bound():
    # Scoring the Jacobian rather than the quartic makes this exact for every
    # fibre, including the ones whose quartic model degenerates modulo p.
    tables = mestre_ap_tables(RANK19, 500)
    for prime in tables.primes:
        table = tables.table(int(prime))
        assert np.all(np.abs(table) <= 2 * np.sqrt(int(prime)))


def test_scores_match_between_backends():
    tables = mestre_ap_tables(RANK19, 300)
    numerators = np.arange(1, 5000, dtype=np.int64)
    denominators = np.where(numerators % 3 == 0, 4, 1).astype(np.int64)
    native = tables.scores(numerators, denominators, backend="native")
    reference = tables.scores(numerators, denominators, backend="reference")
    assert np.allclose(native, reference, rtol=0, atol=1e-9)


def test_score_ranks_the_certified_rank_nineteen_fibre_highly():
    tables = mestre_ap_tables(RANK19, 2000)
    numerators = np.arange(1, 20001, dtype=np.int64)
    scores = tables.scores(numerators)
    position = int((scores > scores[742]).sum())
    assert position < 50


def test_quartic_points_find_the_base_points():
    fibre = mestre_quartic(RANK19, 3)
    xs = sorted({point[0] for point in fibre.points})
    low, high = int(min(xs)) - 1, int(max(xs)) + 1
    found = quartic_points(fibre.coefficients, (low, high), (1, 1))
    found_x = {x for x, _ in found}
    for x in xs:
        assert x in found_x
    for x, y in found:
        assert y * y == fibre.evaluate(x)


def test_quartic_sieve_backends_agree():
    fibre = mestre_quartic(RANK19, 3)
    native = quartic_points(fibre.coefficients, (-800, 800), (1, 3), backend="native")
    reference = quartic_points(
        fibre.coefficients, (-800, 800), (1, 3), backend="reference"
    )
    assert sorted(native) == sorted(reference)


def test_quartic_points_reject_bad_ranges():
    fibre = mestre_quartic(RANK19, 3)
    with pytest.raises(ValueError):
        quartic_points(fibre.coefficients, (10, 0))
    with pytest.raises(ValueError):
        quartic_points(fibre.coefficients, (0, 10), (0, 3))


def test_weierstrass_cover_maps_quartic_points_onto_the_jacobian():
    fibre = mestre_quartic(RANK19, 3)
    base = fibre.points[0]
    cover = quartic_to_weierstrass(fibre.coefficients, base)
    mapped = 0
    for x, y in fibre.points:
        image = cover.image(x, y)
        if image is None:
            continue
        assert cover.contains(*image)
        mapped += 1
    assert mapped >= 10

    a1, a2, a3, a4, a6 = cover.a_invariants
    b2 = a1 * a1 + 4 * a2
    b4 = 2 * a4 + a1 * a3
    b6 = a3 * a3 + 4 * a6
    b8 = a1 * a1 * a6 + 4 * a2 * a6 - a1 * a3 * a4 + a2 * a3 * a3 - a4 * a4
    c4 = b2 * b2 - 24 * b4
    discriminant = -b2 * b2 * b8 - 8 * b4**3 - 27 * b6 * b6 + 9 * b2 * b4 * b6
    assert Fraction(c4**3, discriminant) == _j_invariant_of_short(
        *jacobian_invariants(fibre.coefficients)
    )


def test_weierstrass_cover_rejects_a_point_off_the_quartic():
    fibre = mestre_quartic(RANK19, 3)
    x, y = fibre.points[0]
    with pytest.raises(ValueError):
        quartic_to_weierstrass(fibre.coefficients, (x, y + 1))


def test_point_search_is_monotone_in_the_box_and_in_sieve_primes():
    fibre = mestre_quartic(RANK19, 21)
    small = set(quartic_points(fibre.coefficients, (-20000, 20000), (1, 8)))
    wide = set(quartic_points(fibre.coefficients, (-200000, 200000), (1, 8)))
    deep = set(quartic_points(fibre.coefficients, (-200000, 200000), (1, 40)))
    assert small <= wide <= deep
    for count in (12, 16, 20):
        assert (
            set(
                quartic_points(
                    fibre.coefficients,
                    (-200000, 200000),
                    (1, 40),
                    sieve_prime_count=count,
                )
            )
            == deep
        )


def test_point_search_never_silently_truncates():
    # A capacity far below the survivor count must still return every point,
    # by splitting the denominator range rather than dropping candidates.
    fibre = mestre_quartic(RANK19, 21)
    full = set(quartic_points(fibre.coefficients, (-100000, 100000), (1, 60)))
    cramped = set(
        quartic_points(
            fibre.coefficients,
            (-100000, 100000),
            (1, 60),
            sieve_prime_count=12,
            capacity=4096,
        )
    )
    assert cramped == full
    assert len(full) > 0


def test_single_denominator_overflow_is_reported():
    fibre = mestre_quartic(RANK19, 21)
    with pytest.raises(MemoryError):
        quartic_points(
            fibre.coefficients,
            (-1000000, 1000000),
            (7, 7),
            sieve_prime_count=12,
            capacity=8,
        )
