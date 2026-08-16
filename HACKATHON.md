# Fast-Math Hackathon Guide

## Start here

```sh
uv sync --dev
make test
make portable-test
make graphs
make arb
make finufft
make finufft-cells
make finufft-canopy
make finufft-prime-shell
make metal
make union-closure
```

`make test` builds the native library and runs the complete native/reference
contract suite. When this repository is nested in the research laboratory,
`tools/run-compute.sh` automatically uses its governed compute scheduler; in a
standalone clone it executes the declared command directly. `make
portable-test` repeats the suite without native ISA tuning, ForkUnion,
CommonCrypto, or nauty. `make graphs` writes a local machine-readable benchmark
record under `benchmarks/results/`.

For a focused test:

```sh
tools/run-compute.sh --slots 5 --memory-mb 4096 --timeout-seconds 900 \
  --label fast-math-graph-tests -- \
  env PYTHONPATH=python \
  FAST_MATH_LIBRARY=build/libfast_math.dylib \
  .venv/bin/python -m pytest -q tests/test_graph64.py
```

Use `tools/run-compute.sh` around any sustained benchmark or exhaustive run.
Inside the research laboratory, declare the complete process tree's thread
count, peak memory, and timeout.

For the retained high-order interval-scout FINUFFT shape:

```sh
tools/run-compute.sh --target modal --slots 4 --memory-mb 12288 \
  --timeout-seconds 3600 \
  --workdir fast-math -- \
  env PYTHONPATH=python python3 benchmarks/benchmark_finufft_cells.py \
  --sources 690988 --targets 250000 --chunks 4 \
  --transforms 27 --threads 4 --eps 1e-12 --repeats 3
```

For focused sparse-rank work:

```sh
tools/run-compute.sh --slots 1 --memory-mb 2048 --timeout-seconds 900 \
  --label fast-math-sparse-tests -- \
  env PYTHONPATH=python \
  FAST_MATH_LIBRARY=build/libfast_math.dylib \
  .venv/bin/python -m pytest -q \
    tests/test_sparse_rank.py tests/test_sparse_coloops.py
```

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
tools/run-compute.sh --slots 3 --memory-mb 2048 --timeout-seconds 900 \
  --label fast-math-verify-build -- \
  cmake --build build-verify --parallel 3
```

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

## Good next targets

- Migrate retained profile certificates to stacked counts and fixed digests
  only when a current route attributes at least 5% of representative wall
  time or material peak memory to profile serialization and hashing, while
  preserving its legacy digest mapping. The measured triple-Gram scout made
  zero profile-digest calls; its current successors are also ineligible.
- Add induced filters over the large CSR graph representation only when a
  current classifier supplies a representative route benchmark. The former
  NetworkX factor-type merge target is stale; its source artifact is no longer
  retained.
- Extend packed Union primitives beyond the shipped closure check only when
  two current routes measure material cost in products, empty intersections,
  canonicalization, frequency vectors, or collision shadows.
- Add a native union-family enumerator only with exact shard, symmetry, and
  model-blocking contracts.
- Add proof-producing CNF export and CaDiCaL/Kissat certificate replay.
- Sparse exact incidence builders feeding fast modular rank and FLINT
  integer/Smith-form backends.
- Reopen black-box sparse rank only for a maintained deterministic
  implementation and a captured matrix where structured elimination exceeds
  memory or fails to finish. LinBox scalar Wiedemann loses on the retained
  order-8, order-17, and order-18 fixtures, and its deprecated block-rank
  example does not compile against the retained 1.7.0 headers.
- SIMD or Metal reduction for the experimental exact-prefix Chebyshev-Filon
  contraction. Beat the warmed full-cache NumPy baseline on a captured real
  autocorrelation; do not claim success from storage reduction alone.
- Migrate generated Arb source evaluators to the shipped ordered map/reduce and
  fixed-offset cache assembly, requiring byte-identical cache files and exact
  retained certificate equality.
- Migrate the remaining retained FINUFFT scans to `Type1Plan1D`,
  `Type3Plan1D`, `Type3FixedPlan1D`, or `Type3SignPairPlan1D`, with
  complete-output parity and an end-to-end route benchmark. Fixed-strength
  single-sign and paired-sign cell scans dispatch through retained type-3
  plans on portable Linux but retain simple calls on Darwin; the fixed-R
  product canopy and dyadic prime-shell type-1 scans are also migrated.
- Add precision-tiered Metal kernels only when a real route needs more than
  `complex64`; preserve the current NumPy parity benchmark and keep rigorous
  validation outside the ranking gate.

Choose a target only after recording a representative baseline. Keep new APIs
batch-first and make output allocation explicit.
