# Contributing

Every change lands through a pull request against `hara-seihun/fast-math`. An
agent lane reviews the queue, merges what holds up, and republishes the machine
copy; nothing else is needed from you after the PR exists.

## What belongs here

[`ARCHITECTURE.md`](ARCHITECTURE.md) owns the boundary and
[`TARGETS.md`](TARGETS.md) owns the queue. In short: a kernel belongs here when
its contract is stated over mathematical data rather than one census, and when
more than one route needs it. A script that answers one question belongs in the
work that asked the question.

If you wrote the same loop by hand twice, it is a kernel.

## What a reviewable PR carries

- The kernel, behind the existing C ABI and Python surface.
- An executable reference backend. Native output is checked against it in
  `tests/`, not asserted in prose.
- A benchmark under `benchmarks/` with the numbers in the PR body: the shape,
  the before, the after, the ratio, and the machine. `TARGETS.md` records
  measurements this way and a merge appends to it.
- `make test` green. `make portable-test` too when the change touches SIMD,
  intrinsics, or anything conditioned on the build's architecture.

A measured negative result is worth a PR against `TARGETS.md` alone. The
"Measured negative verdicts" section exists so nobody pays twice for the same
disappointment.

## From this machine

Fleet agents already hold the GitHub identity and the toolchain:

```sh
git clone https://github.com/hara-seihun/fast-math ~/work/fast-math
cd ~/work/fast-math && make test
git switch -c kernel-name && git commit -am "Add ..." && git push -u origin HEAD
gh pr create --fill
```

`fast-math script.py` on `PATH` runs the published copy at
`/srv/pi/fast-math`, which tracks `origin/main`. It is derived and disposable:
the lane replaces it after every merge, so develop from your clone and treat a
locally published tree as temporary.

Reviewing is the same repository: `gh pr diff`, read it, build it, and merge or
say in a review what is missing.

## Optimization rule

Every native kernel has an executable Python reference backend. A change is
ready only when:

1. `backend="native"` agrees with `backend="reference"`.
2. Results are bitwise stable across supported thread counts when the API
   promises deterministic output.
3. Invalid inputs fail instead of wrapping, truncating, or corrupting memory.
4. A representative JSON benchmark improves wall time, not merely an isolated
   inner-loop timer.
5. Existing `lambda_fast` Python imports, environment variables, library
   names, and C symbols still work.

Add a hostile fixture when fixing a bug. Do not weaken a tolerance to make a
performance patch pass.

For a finite-field arithmetic diagnostic build:

```sh
cmake -S . -B build-verify \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAST_MATH_NATIVE_ARCH=OFF \
  -DFAST_MATH_USE_FORKUNION=OFF \
  -DFAST_MATH_USE_COMMONCRYPTO=OFF \
  -DFAST_MATH_USE_NAUTY=OFF \
  -DFAST_MATH_VERIFY_MODULAR_ARITHMETIC=ON
cmake --build build-verify --parallel 4
```

## Exact modular and certificate conventions

- Prime-field APIs reject composite moduli and noncanonical representatives.
- Polynomial coefficients are low-to-high. Fused derivatives use the exact
  Horner recurrence and must match complete reference arrays.
- Dense determinant APIs return determinants only. Do not infer an unreturned
  rank, inverse, or nonsingular-minor certificate.
- Dense linear-system plans choose the first available pivot in each leftmost
  pivot column. Right-nullspace vectors are rows. An inconsistent solve returns
  the first retained left-nullspace row whose product with the right-hand side
  is nonzero; inconsistent solution rows are zeroed.
- A linear-system backend must preserve complete RREF, pivot, nullspace,
  solution, consistency, and witness-index parity at hostile uint32 primes.
- CNF literals use DIMACS signs and one-based variable numbers. Packed
  assignment bit zero is variable one.
- CNF acceptance scans every clause. Rejection returns the first unsatisfied
  clause in input order. This is model verification, not an UNSAT proof.
- Auto CNF dispatch samples actual inspected-literal work because random
  rejected candidates can exit orders of magnitude earlier than valid models.

## Graph conventions

- Graph batches have shape `(graph_count, vertex_count)` and `uint64`
  adjacency masks.
- Large undirected graphs use symmetric CSR with strictly increasing rows;
  loops are stored separately because Union compatibility triples include
  repeated vertices.
- Edge colors are 64-bit option masks, not one exclusive color ID.
- Directed colored canonicalization is delegated to nauty and returns
  generator permutations as well as group-size/orbit metadata.
- Graphs are simple and undirected, with no self-loops.
- Induced edge-mask bits use
  `itertools.combinations(range(induced_order), 2)` order.
- Induced-profile class lookup IDs must be dense from zero.
- Stacked induced orders must be strictly increasing; output coordinates are
  concatenated in that order.
- Canonical labeling belongs to nauty/Traces or a precomputed exact lookup;
  fast-math owns repeated extraction and counting, not graph isomorphism.
- `decode_graph6` intentionally supports unheaded short-form records only.
- Digest namespaces must identify the coordinate schema, not just the route.
- Fixed-width digest values are canonical little-endian `uint64`.

## Group and Cayley-CI conventions

- `PermutationActionPlan` minimizes the implicit identity image together with
  its supplied permutation rows. Orbit language requires a complete finite
  action and an invariant mask batch.
- GPU action backends preserve exact `uint64` semantics and deterministic
  numeric-minimum representatives; compare complete mask and flag arrays.
- Permutations are image arrays: `p[x]` is the image of `x`; composition is
  `left[right]`.
- Multiplication tables use `table[left, right] = left * right`; Cayley arcs
  are `(g, s * g)`.
- Packed subset bit `i` denotes atom or element `i`, low bit first. Identity
  is never part of a connection set.
- Complete powerset enumeration is bounded and distinct from quotienting a
  caller-supplied invariant subset collection. Fixed-weight enumeration requires
  the complete finite action, validates Burnside's orbit count, and keeps the
  combinatorial domain implicit behind an explicit `max_subsets` bound.
- Derivative maps are normalized bijections fixing the identity and use the
  exact R-0805 convention documented in `INTERFACE.md`.
- A 2-WL result is not complete without stable basic relations and verified
  intersection numbers.
- Use `wl2_refinement` when only the stable partition is required; do not
  present its result as a complete coherent configuration.
- Every group/CI native kernel retains an executable Python/NumPy reference
  path. Compare full partitions or tensors, not only aggregate counts.
- Keep atlas graph canonicalization on the existing batched nauty API; do not
  reintroduce per-graph `labelg` or GAP shellouts inside a hot loop.
- Sparse matrices use zero-based CSR with strictly increasing columns per row.
- A sparse-rank target of zero means full rank; a positive target is an exact
  early-stop certificate, not a rank estimate.
- Pivot witnesses are in original matrix coordinates and must define a
  nonsingular minor over the selected field.
- Degree-one row/column peeling is exact rank decomposition, not a heuristic;
  validate both its cascades and the residual witness minor.
- Sparse row order is deterministic Markowitz cost, row degree, first
  reordered column, then original row ID; columns use degree then original ID.
  Reference and native witnesses must stay identical.
- Bounded-block coloop certificates are ordered local duals: certificate `i`
  must map removed column `i` to one and every later/residual column to zero.
- A block-coloop result proves `removed + rank(residual)`; it does not by
  itself claim an original-coordinate row minor for the removed columns.
- Four-row blocks are the retained graph-attachment default: larger blocks did
  not remove additional order-17 columns and cost more preprocessing time.
- Parallelize independent primes, not the data-dependent pivot chain.
- Do not replace modular arithmetic with FMA, floating SIMD, or Metal. Those
  change the field and invalidate the certificate.

## Packed-family conventions

- Bit `s` of a packed family marks whether subset mask `s` is a member.
- One-word families support ground sizes zero through six.
- Closure means every pairwise union is present; no generator-only shortcut is
  allowed.
- The empty family and the singleton family containing only the empty set are
  union-closed.
- Batch before materializing Python tuples, and compare complete route outputs
  against the direct set verifier.

