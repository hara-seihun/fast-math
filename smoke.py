"""Deployment smoke test: the native library loads, runs, and agrees with the
executable reference model. Run it through the `fast-math` launcher."""

import numpy as np

import fast_math.graphs as graphs
from fast_math.elliptic import mestre_ap_tables, mestre_quartic

TRIANGLE = np.array([[0b110, 0b101, 0b011]], dtype=np.uint64)

native = graphs.encode_graph6(TRIANGLE, backend="native")
reference = graphs.encode_graph6(TRIANGLE, backend="reference")

assert native.backend == "native", native.backend
assert native.records == reference.records == ("Bw",), (native.records, reference.records)

# The sextuple of the leaderboard's rank-19 record curve, whose fibre at T = 743
# is that curve.  Native and reference tables must agree and stay within Hasse.
RANK19 = [-557, -444, -7, 196, 311, 501]
fibre = mestre_quartic(RANK19, 743)
assert all(y * y == fibre.evaluate(x) for x, y in fibre.points), "base points left the fibre"

tables = mestre_ap_tables(RANK19, 200, backend="native")
assert tables.backend == "native", tables.backend
assert np.array_equal(
    tables.tables, mestre_ap_tables(RANK19, 200, backend="reference").tables
), "native and reference a_p tables disagree"
for prime in tables.primes:
    assert np.all(np.abs(tables.table(int(prime))) <= 2 * np.sqrt(int(prime)))

print(f"fast-math ok: native/reference parity, library {__import__('os').environ['FAST_MATH_LIBRARY']}")
