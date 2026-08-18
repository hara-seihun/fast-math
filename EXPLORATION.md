# Exploration and certification batches

Fast Math complements, rather than replaces, specialist mathematical systems.
The canonical NixOS host includes SageMath, PARI/GP, Singular, Macaulay2,
Normaliz, cddlib, polymake, fplll, LinBox, FFLAS-FFPACK, GAP, FLINT, Z3,
cvc5, CaDiCaL, and Kissat. These systems remain the owners of general computer
algebra, Gröbner bases, Hilbert bases, polyhedral conversion, LLL, black-box
linear algebra, and SAT/SMT search.

The optimization opportunity is the repeated boundary around those systems:
small object construction, language crossings, process startup, repeated
validation, and materialization between operations. Fast Math admits a kernel
when the mathematical contract is reusable and an executable reference model
can check the complete output.

## Exact modular batches

`ModularPolynomialPlan` retains low-to-high coefficient rows over a caller-chosen
uint32 prime. One Horner pass returns values and, optionally, first derivatives
for every polynomial/point pair. Reference NumPy, native C++20, and HIP use the
same exact field contract. The HIP plan retains coefficients and output
workspaces; each call transfers only the new point row and requested outputs.

`ModularDeterminantPlan` computes exact determinants for batches of square dense
matrices. It uses pivoted finite-field elimination, not floating-point rank.
Reference, native, HIP, SageMath, and FLINT checks agree on the complete retained
fixtures. The returned determinant is directly checkable in another exact
system; this API does not claim a rank or inverse certificate.

Inputs must already be canonical field representatives in `[0, prime)`. The
library rejects composite moduli instead of silently applying field algorithms
to a ring with zero divisors.

```python
from fast_math import ModularDeterminantPlan, ModularPolynomialPlan

with ModularPolynomialPlan(coefficients, prime=1_000_000_007) as plan:
    result = plan.evaluate(points, derivative=True, backend="auto", threads=8)

with ModularDeterminantPlan(8, prime=1_000_000_007) as plan:
    determinants = plan.determinants(matrices, backend="auto", threads=8)
```

## CNF assignment certificates

`CnfPlan` retains DIMACS-style signed literals and clause offsets. It verifies
packed Boolean assignment batches and returns both a satisfaction flag and the
first unsatisfied clause. Thus every rejection carries an explicit witness;
every acceptance has scanned every clause under the supplied assignment.

The reference backend is a direct executable specification. Native C++ uses
caller-inclusive parallel batches and reports the exact number of inspected
literals. HIP retains the clause database and returns compact failure indices.
Automatic dispatch samples a bounded prefix through the native checker so that
an easy-rejection workload stays on CPU while full-certificate scans can use the
GPU.

This verifies candidate models. It is not an UNSAT proof checker and does not
replace CaDiCaL/Kissat search or DRAT/LRAT/FRAT replay.

```python
from fast_math import CnfPlan, pack_boolean_assignments

assignments = pack_boolean_assignments(boolean_matrix)
with CnfPlan(clauses, variable_count=variable_count) as plan:
    checked = plan.evaluate(assignments, backend="auto", threads=8)
```

## Reproducible comparisons

```sh
make modular-batches
make cnf-verification
```

`benchmark_modular_batches.py` checks Fast Math against SageMath and
python-flint before reporting timings. `benchmark_cnf_verification.py` checks
complete reference/native/HIP failure-index parity and validates the planted
single-assignment fixture through CaDiCaL. Machine-readable local receipts are
written under `benchmarks/results/` and summarized in the main README.

On the local Ryzen AI Max+ 395 / Radeon 8060S host, retained warm medians are:

- fused values and derivatives for 128 degree-64 polynomials at 1,024 points:
  0.000426 seconds HIP versus 0.3060 seconds SageMath, a 718.9x speedup;
- 1,000 exact 8-by-8 determinants: 0.000341 seconds native versus 0.00919
  seconds python-flint and 0.0657 seconds SageMath, 26.9x and 192.5x;
- 100,000 valid CNF model checks against 1,024 three-literal clauses: 0.00120
  seconds HIP versus 0.02857 seconds native, a 23.8x speedup;
- 1,000 valid checks of the same CNF: 0.000464 seconds native versus 0.4949
  seconds in the executable Python specification, a 1,066x speedup.

These measurements include the ordinary Python call and compact output transfer
but exclude one-time retained-plan construction. The receipts separately record
setup plus first execution, preventing warm GPU results from being mistaken for
one-shot latency.
