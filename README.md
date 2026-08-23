# fast-math

Shared native compute kernels for mathematical research. Python defines the
public APIs and executable reference models; C++20 performs high-volume
arithmetic behind a small C ABI. Repeated native calls use pinned ForkUnion
worker pools by default, parking persistent workers between batches, with a
build-time standard-thread fallback.

`import lambda_fast`, `LAMBDA_FAST_LIBRARY`, `liblambda_fast`, and the
`lambda_fast_*` C symbols remain supported compatibility interfaces.

## Is the kernel you need already here

Ask the library, not this file:

```sh
fast-math --find csr              # name, summary, and module all match
fast-math --find "common neighbor"
fast-math --index                 # every public name, one line each
```

and `fast_math.find("orbit", "subset")` from inside Python. The listing is
derived from the package's own exports and docstrings, so it is never a
separate list going stale. A miss is worth acting on: it means the loop you are
about to write by hand is missing, and [`CONTRIBUTING.md`](CONTRIBUTING.md) is
how it stops being missing for everyone after you.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the core/domain boundary, retained-plan
direction, backend contract, and evidence-based GPU priorities. See
[`EXPLORATION.md`](EXPLORATION.md) for the installed specialist-system survey and
the exact modular/CNF certification-batch contracts.

## Kernels

- `fast_math.actions`: retained exact permutation actions on packed subsets,
  including reference, native CPU, and HIP canonical-image and early-exit
  canonical-minimum backends.
- `fast_math.plans`: retained finite-group, Cayley-graph, and fixed-adjacency
  canonicalization structures for repeated workloads.
- `fast_math.runtime`: backend capability discovery without importing a domain
  package as the native runtime owner.
- `fast_math.modular`: retained exact uint32-prime polynomial value/derivative
  batches and dense determinant batches across reference, native CPU, and HIP.
- `fast_math.base_p`: batched index/digit codec over encoded `F_p^n` points
  (`p <= 251`, `n <= 16`) with digit-wise negation, projective normal forms,
  and dense negation/scalar class tables across reference and native backends.
- `fast_math.colex`: batched colexicographical rank/unrank for uint64 element
  masks (`element_count <= 64`) with fixed-weight visit marking against a
  caller-owned visited bitmap, across reference and native backends.
- `fast_math.fp_spans`: exact spans of little-endian base-p encoded points over
  small prime fields (`p <= 251`, `width <= 16`) — batched ragged ranks and
  canonical RREF span reduction with query membership and quotient
  coordinates, across reference and native backends.
- `fast_math.tuple_orbits`: orbit canonicalization, dense partitions, and
  whole-space Burnside-validated structure for mixed-radix digit tuples under
  positional permutation groups, across reference and native backends.
- `fast_math.modular_linear`: canonical RREF, rank, row transforms, right and
  left nullspaces, inverses, and retained fixed-matrix solve batches. Native and
  HIP backends return either an exact solution or a left-null inconsistency
  witness for every right-hand side.
- `fast_math.shift_gates`: exact shift-divisor gates for witness searches of
  Erdos-647-style divisor conditions on n = M*v. Derives per-prime alive
  tables and prime-forced linear forms from the modulus by exact integer
  arithmetic, re-verifies them against direct factorization, and scans
  v-intervals with a wheel-compressed segmented sieve plus deterministic
  Miller-Rabin across reference, native CPU, and HIP backends.
- `fast_math.cnf`: retained DIMACS-style clause plans with exact packed
  assignment verification and first-unsatisfied-clause witnesses on CPU and HIP.
- `fast_math.elliptic`: exact Mestre-locus family construction over Q, retained
  mod-p trace tables for one-parameter quartic families, batched Mestre-Nagao
  scoring of rational fibres, and bit-sieved rational point search on quartic
  models with exact confirmation.
- `fast_math.graphs`: graph6 encoding/decoding,
  degree/edge/triangle/wedge/induced-P3 invariants, induced-subgraph class
  censuses, pair profiles, and clique or independent-set witnesses for graph
  orders up to 64; loop-aware colored triangle enumeration on large symmetric
  CSR graphs; and optional batched nauty canonicalization for directed
  vertex-colored graphs.
- `fast_math.groups`: permutation point orbits, deterministic Schreier-Sims
  stabilizer chains, exact group order and membership, and explicit finite
  double-coset partitions for degrees through 4096.
- `fast_math.ci`: inverse-closed connection-set orbit enumeration, exact
  fixed-weight subset orbits without materializing the full weight slice, batched
  Cayley graph construction into the existing nauty API, generalized
  dihedral/Hol(A) helpers, R-0805 derivative-group orbits, exact
  coherent-configuration/2-WL refinement with intersection numbers, and
  bounded-degree `u64_mask_lut`/`compose_u64_mask_luts` microkernels for
  Python search loops over small packed connection masks.
- `fast_math.adaptive`: batched optimal adaptive-oracle areas over the ternary
  restriction lattice of `{-1,+1}^n`, with float and exact-integer Bellman
  backends, per-restriction conditional variances, optimal first queries, and
  the complete optimal-policy array.
- `fast_math.reductions`: deterministic power moments and segmented complex
  sums, L1 masses, and total variations.
- `fast_math.packing`: AVX-512/multicore and persistent-HIP oriented-square
  incidence and direct weighted-adversary scans. See [`PACKING.md`](PACKING.md)
  for the numerical contract, API, validation, and measured performance.
- `fast_math.taylor`: coefficient-stack preparation and Taylor value/log-moment
  evaluation around FINUFFT or another transform backend.
- `fast_math.filon`: experimental exact-prefix Chebyshev-Filon
  autocorrelation contractions with a portable SIMD integration-by-parts
  tail. The API is numerically tested and materially faster than its original
  scalar tail, but is not yet a production route replacement.
- `fast_math.finufft`: persistent one-dimensional type-1 and type-3 FINUFFT
  plans with retained coordinates, many-transform execution, reusable output
  buffers, setup/execute timing, fixed-strength cell plans, and paired-sign
  type-3 plans that fuse opposite signs through conjugate symmetry.
- `fast_math.affine`: one retained-array affine-plan contract with portable
  NumPy execution plus automatic NumPy, Metal, or CUDA backend selection.
- `fast_math.metal`: persistent MLX/Metal affine plans for large
  `base + steps @ basis` populations, including fused contour winding,
  phase-increment, and edge-floor metrics that avoid copying full value
  matrices back to the CPU.
- `fast_math.cuda`: the matching CuPy/CUDA affine-plan contract for Modal and
  other NVIDIA targets, with the same retained arrays, batching, values, and
  compact contour metrics.
- `fast_math.hip`: the ROCm/HIP affine-plan contract for the local AMD Strix
  Halo GPU. It uses native HIP kernels for affine populations and fused
  contour metrics, with the same NumPy parity contract and automatic Linux
  dispatch.
- `fast_math.arithmetic`: finite Dirichlet inverse construction.
- `fast_math.sparse`: deterministic exact rank and pivot witnesses for CSR
  matrices over 32-bit prime fields, including ForkUnion-parallel multi-prime
  certification.
- `fast_math.large_graph`: deterministic symmetric CSR construction, exact
  colored triangle enumeration, and batched common-neighbor counts or
  materialization for explicit sparse vertex pairs.
- `fast_math.union`: exact batch union-closure checks for families packed into
  one `uint64` word on ground sets through size six, with portable native and
  executable NumPy reference backends.
- `fast_math.lambda_kernels`: compatibility home for the streamed Lambda
  coefficient and two-level replay kernels.
- `fast_math_arb.h`: optional FLINT/Arb cache workspaces plus deterministic
  ordered parallel map/reduce. Expensive independent terms run through
  ForkUnion, while rigorous error streams are added in their original serial
  order. Independent two-sided endpoint weights use fixed output offsets and
  directed binary64 rounding.

The retained Lambda kernels are:

- `fused_two_level`: streams truncated Dirichlet products directly into the
  fine-piece complex sums, L1 envelopes, and outer two-level acceptance
  records used by the live L=0.1529 finite-bridge route.
- `accumulate_coefficients`: materializes common and low coefficient arrays
  when a caller actually needs every output coefficient.
- `truncated_inverse` and `dirichlet_inverse`: construct the finite inverse
  coefficients used by multiple Lambda mollifier routes.
- `power_moments`: computes powers 3 through 12 and their ordinary,
  phase-current, and radial derivative moments in one deterministic pass.

The streaming kernel aligns work with the retained rigorous weight intervals,
uses only thread-local tiles, and serially reduces fine pieces in mathematical
order. It never materializes the 100-million-element output arrays.

## Base-p point codec contract

Research routes hold `F_p^n` points as plain base-p integers. Digit `j` of a
code is its multiple of `p^j` (least-significant first), digits are uint8
rows, codes are uint64, primes run through 251, widths through sixteen, and
`p^width` must fit uint64; violations raise instead of wrapping.

`base_p_digits` and `base_p_codes` convert both ways. `base_p_negation_codes`
negates each digit modulo `p`, which is the additive inverse of the point,
not integer negation of the code. `base_p_scalar_normals` scales each point
so its least-significant nonzero digit becomes one: one representative per
projective point, fixed under unit multiples, with the zero vector mapping
to zero.

`base_p_class_table(prime, width, classes=...)` classifies the whole space.
`classes="negation"` pairs nonzero codes with their negations (singletons
when `p = 2`); `classes="scalar"` groups projective points. Class ids are
dense from zero in ascending representative order, so the zero vector forms
class zero; `representatives` and `counts` have exactly
`(p**width + 1) // 2` entries for odd-prime negation classes (`p**width`
when `p = 2`) and `(p**width - 1) // (p - 1) + 1` for scalar classes.
Negation-class representatives compose from shipped calls:
`np.minimum(codes, base_p_negation_codes(codes, prime, width))`.

## Colex ranking and orbit-marking contract

Orbit-enumeration scouts hold subsets of `{0, ..., element_count - 1}` as
uint64 element masks: bit `i` set means element `i`, low bit first. The
colexicographical rank of a mask with sorted elements `c_1 < ... < c_k` is
`C(c_1, 1) + ... + C(c_k, k)`. Ranks are unique only within a weight class,
so masks naming elements outside the declared range fail instead of wrapping,
and ranks at or above the weight-class size fail rather than wrapping.

`colex_rank` ranks arbitrary batches. `colex_unrank(ranks, element_count,
weight)` is its exact inverse for a declared weight. `colex_visit` is the
enumeration shape: every subset must carry exactly the declared weight, and
each rank test-and-sets one bit in a caller-owned uint64 visited bitmap of
`ceil(C(element_count, weight) / 64)` words, updated in place, returning a
per-subset flag that is true when its rank was not already marked. Reference
and native backends agree bitwise; the native route is single-threaded, so
results are stable across thread counts.

## Tuple orbit contract

`Z_base^width` tuples are mixed-radix codes, most significant digit first;
the code space is bounded at `2**24`. Generators are image arrays acting on
digit positions (`p[i]` moves digit `i` to `p[i]`, composed `left[right]`);
rows are validated as permutations, codes outside `base**width` fail rather
than wrap, and the internally closed group is capped at 200000 elements on
both backends so oversized closures raise identically off either route.
`tuple_orbit_canonicalize` returns the numeric-minimum code per batch element
with an is-canonical flag. `tuple_orbit_partition` formats that into dense
class ids from zero, sorted representatives, and per-orbit sizes, mirroring
`MaskOrbitPartition`. `tuple_orbit_space` canonicalizes every code and checks
the orbit count against the Burnside average of `base**cycles(g)` computed
independently from cycle counts, recording agreement in `burnside_valid`.
Reference and native backends agree bitwise; the native route is
single-threaded, so results are stable across thread counts.

## Encoded point span contract

Points are the same little-endian base-p codes as the base-p codec:
`p <= 251`, `width <= 16`, and `p**width` must fit in uint64; composite
fields, malformed ragged offsets, and out-of-space codes fail in both Python
and the C ABI. `fp_span_ranks(point_codes, span_offsets, prime, width)` takes
`span_count + 1` offsets (initial zero, final equal to the point count) and
returns one exact rank per contiguous batch; empty batches are valid.
`fp_point_span(point_codes, query_codes, prime, width)` returns the canonical
RREF basis (unique for the row space) with rows ordered by pivot column,
each row tagged with the input index that produced it, per-point independence
flags, per-query membership flags, query coordinates against that basis, and a
canonical quotient residual code: for every query,
`q = coordinates @ basis + quotient` over `F_p`, and membership is exactly
`quotient == 0`. Reference and native backends return complete identical
outputs; the native route is single-threaded, so results are stable across
thread counts.

## Build and test

On the canonical NixOS research host, FLINT/Arb, nauty, NumPy, NetworkX, and
pytest are declarative system dependencies. From this directory:

```sh
make test
```

This builds an x86-64 library tuned for the current Ryzen CPU, including the
FLINT/Arb and nauty kernels, creates the `liblambda_fast` compatibility link,
and runs the native CTest and Python parity suites. Do not create a local
wheel-based virtual environment on NixOS; the system Python owns the patched
native dependencies. Set `PYTHON=/path/to/python` explicitly when testing an
independent environment.

Research agents invoke the built library through the `fast-math` launcher,
which is on `PATH` and needs no environment setup:

```sh
fast-math -c 'import fast_math; print(fast_math.__file__)'
fast-math path/to/research_script.py
```

The launcher supplies `PYTHONPATH` and the native library path, resolving the
library from `build/` in this tree or `lib/` in a published one. `./deploy`
publishes the package, the CPU library, and the launcher to `/srv/pi/fast-math`
so the unprivileged orchestrator fleet user can run it too, then validates it
end to end with `smoke.py`. Each publish is a directory named by its commit with
an atomically flipped `current` symlink, so a running session keeps the tree it
started with. The published tree tracks `origin/main`: the PR lane republishes
after every merge, and a developer's own launcher stays on their checkout. The
HIP library is deliberately not published: it needs GPU device access the fleet
user does not have.

Test coverage includes
exhaustive behavior over every labeled graph through order five, invariant
parity over all 32,768 labeled order-six graphs, 64-vertex boundaries,
arithmetic identities, invalid inputs, input immutability, graph and
sparse-rank witness validity, hostile prime fields, Metal/NumPy parity,
CUDA/NumPy parity when CUDA is available, the complete 13-group Cayley-CI
atlas, and exact cross-thread determinism.

Run the retained Modal portability validation from this directory:

```sh
modal run modal_validation.py --target all --gpu L4
```

The CPU validation builds the complete native library on Linux x86_64 with
GCC, runs CTest, and runs the Python suite. The GPU validation runs the shared
affine-plan parity suite on an NVIDIA GPU. On the local AMD host, `make hip-test`
builds the `gfx1151` HIP backend and runs the same full-value and fused-metric
parity checks against NumPy. The explicit ARM NEON kernels in the
Taylor and moment paths retain scalar fallbacks, while Linux builds may use
the host x86 instruction set through `-march=native`.

Changes land through pull requests against `hara-seihun/fast-math`, which an
agent lane reviews, merges, and republishes. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for what a reviewable change carries and
for the mathematical conventions every kernel keeps.

## Benchmarks

Run individual benchmark families:

```sh
make benchmark
make real-checkpoint
make moments
make inverse
make tune
make general
make graphs
make groups-ci
make derivative-rank6
make ci-weight-orbits
make adaptive-area
make union-closure
make union-closure-routes
make digests
make arb
make finufft
make finufft-cells
make finufft-canopy
make finufft-prime-shell
make metal
make hip-test
make mask-lut
make subset-actions
make modular-batches
make modular-linear
make cnf-verification
```

Run output-size and thread-count sweeps:

```sh
make suite
```

Benchmarks emit JSON containing dimensions, exact update counts, wall and
kernel timings, throughput, peak RSS, and retained-result checks.
`benchmark_finufft_cells.py` accepts up to 128 transforms and constructs
Taylor-normalized coefficient rows with the same recurrence used by the
phase-faithful interval scouts.

## Measured production shapes

Measurements below are from the local M4 Pro on July 26-27, 2026, except for
the explicitly labeled Modal L4 row. They are reproducible baselines, not
portable performance promises.

| Kernel and real shape | Best wall time | Throughput | Peak RSS |
| --- | ---: | ---: | ---: |
| Modal x86_64 paired type-3, N=690,988, 27 transforms/sign, 4x250,000 targets | 2.041 s vs 2.388 s simple | 1.17x | benchmark-declared 12 GB |
| Fused two-level, N=690,988, 585,998,464 pairs | 0.113 s | 5.17B pairs/s | 65 MB |
| Fused two-level, N=1,448,739, 733,309,123 pairs | 0.119 s | 6.14B pairs/s | 76 MB |
| Fused two-level, N=2,429,999, 835,766,559 pairs | 0.142 s | 5.88B pairs/s | 93 MB |
| Power moments 3-12, 2,000,000 samples | 0.00345 s | 579M samples/s | benchmark record |
| Truncated inverse, P=690,988, 8,706,661 updates | 0.00852 s | 1.02B updates/s | benchmark record |
| Segmented complex stats, 2,000,000 samples | 0.00154 s | 56.7x reference | benchmark record |
| Taylor coefficients, 690,988 sources x 5 orders | 0.000688 s | 6.8x reference | benchmark record |
| Taylor evaluation, 500,000 targets x 5 orders | 0.00109 s | 5.4x reference | benchmark record |
| Graph pair profiles, 2,000 order-43 graphs | 0.00421 s | 441x reference | benchmark record |
| Graph invariants, 2,000 order-43 graphs | 0.00116 s | 539x reference | benchmark record |
| K5 detection, 2,000 random order-43 graphs | 0.000924 s | 325x reference | benchmark record |
| K5 exclusion, 2,000 K5-free order-43 graphs | 0.0111 s | 747x reference | benchmark record |
| graph6 decode, 20,000 order-nine graphs | 0.0150 s | 23.3x reference | benchmark record |
| Induced order-six census, 20,000 order-nine graphs | 0.00422 s | 889x reference | benchmark record |
| Fixed-width SHA-256, 50,000 x 208-field rows | 0.00741 s | 159x sparse repr path | benchmark record |
| Stacked induced orders 1-6, 50,000 order-nine graphs | 0.0741 s | 2.00x pre-plan stack | benchmark record |
| Union CSR triangles, 3,722 vertices / 35,292 edges / 371,915 triples | 0.00558 s | 3.78 logical GB/s | benchmark record |
| Union CSR common neighbors, 35,222 pairs / 1,057,077 outputs | 0.00492 s | 523x Python reference; 11.28 logical GB/s | benchmark record |
| Packed union closure, all families through ground size four | 0.0740 s CPU vs 0.247 s direct | 3.34x route CPU; identical output | benchmark record |
| Packed union closure, singleton-channel retained scout | 0.0402 s CPU vs 0.153 s direct | 3.80x route CPU; identical output | benchmark record |
| Native packed union closure, 65,535 ground-four families | 0.000968 s vs 0.004867 s NumPy | 5.03x kernel; route wall 1.067x / 1.031x | Modal portable x86_64 record |
| Nauty colored digraphs, 3,722 order-13 factor graphs | 0.0111 s | 335k graphs/s | benchmark record |
| Complete 13-group Cayley-CI atlas, 11,664 presentation orbits / 9,606 graph fibers | 0.0960 s vs 5.025 s Python/NetworkX/labelg | 52.33x end-to-end; exact fiber parity | local `make groups-ci` receipt |
| Degree-18 group order, point orbits, and 17 membership tests | 0.00111 s vs 1.467 s GAP process route | 1,324x; identical output | local `make groups-ci` receipt |
| Derivative orbits on `C_3^6`, order 729 | 0.00855 s vs 0.508 s reference | 59.5x; exact labels and deterministic replay | local `make derivative-rank6` receipt |
| Sparse rank, 162,864 x 12,346 with 1,311,046 nonzeros | 1.391 s | 89.8x route baseline | benchmark record |
| Two-prime sparse rank, same matrix | 1.230 s | 1.74x serial primes | benchmark record |
| Ordered Arb cache, 1,451,978 values at 256 bits, 3 workers | 0.397 s | 2.58x optimized serial; exact certificate | benchmark record |
| Arb weight intervals, 97,879 blocks at 256 bits, 3 workers | 0.081 s | 2.87x serial; byte-identical records | benchmark record |
| Persistent FINUFFT type-3, 16 x 5 transforms | 0.0430 s | 1.22x preallocated simple API | benchmark record |
| Fixed-strength type-3 cells, 690,988 sources / 4 x 500,000 targets | 0.307 s | 1.21x simple API | Modal CPU benchmark record |
| Paired-sign type-3 cells, 690,988 sources / 4 x 500,000 targets | 0.456 s | 1.42x simple API | Modal CPU benchmark record |
| Persistent FINUFFT type-1, 16 scans | 0.0163 s | 1.81x preallocated simple API | benchmark record |
| Fixed-R canopy type-1 stacks, 500,000 sources / 131,072 modes / 8 offsets | 0.530 s | 1.35x repeated simple calls; 2.73e-15 max difference | benchmark record |
| Dyadic prime-shell type-1 plans, 6 shells / 145,421 primes / 48 transforms | 0.428 s | 1.55x repeated simple calls; 2.02e-17 max difference | benchmark record |
| Metal affine contour metrics, 85 x 13,661 points | 0.00159 s | 4.97x NumPy complex64 | benchmark record |
| Metal affine contour metrics, 4,096 x 13,661 points | 0.0343 s | 10.88x NumPy complex64 | benchmark record |
| CUDA affine contour metrics, 1,024 x 4,097 synthetic points | 0.00271 s | 25.2x NumPy complex64 | Modal L4 validation |
| HIP affine contour metrics, 4,096 x 13,661 points on local gfx1151 | 0.0195 s | 28.27x current NumPy affine route; 1.76x raw wall-time advantage over the retained 0.0343 s Metal shape | local `make hip-benchmark` receipt; zero winding disagreements |
| Small packed-mask permutation, 64 permutations x 2,048 masks at degree 11 | 0.0018 s lookup apply | 49.1x Python bit-walk apply; exact output match | `make mask-lut` receipt |
| HIP packed-subset canonical images, 262,144 masks x 504 degree-39 permutations | 0.01173 s vs 0.18263 s native CPU | 15.57x end-to-end retained-plan call; exact image and flag parity | local `make subset-actions` receipt |
| HIP packed-subset canonical images, 1,000,000 masks x 41 degree-41 permutations | 0.00681 s vs 0.05902 s native CPU | 8.66x end-to-end retained-plan call; exact image and flag parity | local `make subset-actions` receipt |
| HIP modular polynomial values + derivatives, 128 x 65 coefficients / 1,024 points | 0.000426 s vs 0.3060 s SageMath | 718.9x end-to-end; native CPU is 59.6x SageMath; complete exact parity | local `make modular-batches` receipt |
| Native modular determinants, 1,000 x 8 x 8 | 0.000341 s vs 0.00919 s python-flint / 0.0657 s SageMath | 26.9x / 192.5x end-to-end; complete exact parity | local `make modular-batches` receipt |
| Retained modular solve, 8,192 changing RHS for one 64 x 64 matrix | 0.02363 s native vs 1.68547 s python-flint conversion + multiply | 71.3x NumPy-to-solution route; complete solution parity | local `make modular-linear` receipt |
| Retained rank-40 modular system, 8,192 RHS for a 48 x 64 matrix | 0.01248 s HIP vs 0.11295 s native CPU | 9.05x; 4,096 solutions and 4,096 left-null inconsistency witnesses replayed in SageMath | local `make modular-linear` receipt |
| HIP CNF valid-model checks, 100,000 assignments / 128 variables / 1,024 clauses | 0.00120 s vs 0.02857 s native CPU | 23.8x retained-plan call; exact acceptance and failure-index parity | local `make cnf-verification` receipt |
| Native CNF valid-model checks, 1,000 assignments / same formula | 0.000464 s vs 0.4949 s executable Python reference | 1,066x end-to-end; exact inspected-literal and witness parity | local `make cnf-verification` receipt |
| Fixed-weight subset orbits, degree 41 / weight 8 / 95,548,245 subsets | 2.262 s | 2,330,445 exact orbit representatives without domain materialization | `make ci-weight-orbits` receipt |
| Complete GL(3,3) action validation, 11,232 degree-27 rows | 0.725 s validation benchmark; 4.763 s first route stage vs 319.038 s prior | 67.0x end-to-end route-stage speedup; exact generated-order closure test | `make ci-weight-orbits` receipt |

All three fused runs reproduce the retained acceptance-driving
`two_level_upper` exactly at binary64 output precision. The original
materializing checkpoint baseline used 2.45 GB RSS; streaming cuts that to
65-93 MB while computing the final two-level result rather than merely the
intermediate coefficients.

The type-3 cell paths are platform-dispatched. On Modal x86_64, retaining one
three-transform coefficient stack cuts four single-sign chunks from 0.371 to
0.307 seconds median (1.21x). One retained six-transform positive-sign plan
evaluates the three positive and three negative transforms in 0.456 seconds
median versus 0.645 seconds for separate simple calls, with maximum
complete-array relative difference `9.02e-16`. The M4 Pro keeps simple calls
because persistent cell plans were slower in the local profiles.

Benchmark commands write local machine-readable records under
`benchmarks/results/`. Generated records are ignored because route-dependent
captures may contain machine-specific paths or fixtures outside this
repository; the reproducible public baselines are summarized above.

The Union representation-193 rows are historical benchmark records. Their
source report
`problems/union-closed/scratch/n13-r193-sharded-enumeration/n13-r193-merged.json`
is no longer present, so the former `make union-graphs` target was removed
rather than left as a broken command. `benchmark_union_graphs.py` remains
available for a future compatible report supplied explicitly.

## Experimental Filon compression

The live L=0.0999 Chebyshev-Filon scout materializes a
`129 x 17,865,985` complex128 kernel: `36,875,393,040` bytes, or 34.34 GiB.
The kernel head is genuinely high rank, but the oscillatory tail has a short
endpoint-derivative expansion. Retaining 8,192 exact columns plus ten endpoint
terms for every row uses about 16.15 MiB, a 2,178x representation reduction.

`filon_chebyshev_inner_product` fuses the positive/negative lag contraction,
generates tail weights in registers, uses FMA complex products, and reduces
fixed chunks deterministically through ForkUnion. Native/reference parity,
conjugated-row parity, endpoint-derivative reconstruction, invalid inputs,
and bitwise one-versus-two-worker determinism are tested.

Portable two-lag SIMD now evaluates adjacent tail weights together while
retaining the original lag-ordered compensated reduction and 256-lag phase
resets. On a captured real node-64 route autocorrelation, one-thread best fell
from `0.666 seconds` to `0.188 seconds` (3.54x), and eight workers reach
`0.0301 seconds`. The compact result is bit-identical to the pre-SIMD result
and differs from the full cached row by `1.42e-11` relatively.

This is still not a route replacement. On the same captured fixture the warmed
NumPy cached-row best is `0.0217 seconds`, 1.39x faster than the best compact
run. The compact path should replace the cache only after a same-fixture wall
time win or an explicit route decision to trade contraction speed for the
2,178x storage reduction.

See `FILON-HANDOFF.md` and the `filon-*` JSON records for the precise stopping
state.

## License

Fast Math is available under the MIT License.

## Metal, CUDA, and HIP affine plans

The portable NumPy backend is always available. Install a GPU backend with
`pip install 'fast-math[metal]'` on Apple silicon or
`pip install 'fast-math[cuda]'` on a CUDA host. All plans retain the shared
base and basis; each GPU call uploads only the new real step matrix and
downloads either the requested values or compact metrics:

```python
from fast_math import affine_plan

plan = affine_plan(base, basis)  # Metal, CUDA, or portable NumPy
metrics = plan.contour_metrics(
    candidate_steps,
    edge_slice=slice(edge_begin, edge_end),
    batch_size=None,  # one GPU batch when the retained value buffer fits
)
```

The retained full-strip packet has 20 directions and 13,661 contour points.
Across populations from 85 through 4,096, the Metal plan was 4.97x to 10.88x
faster than two-thread NumPy complex64 end to end, including step upload and
metric download, with no winding disagreements in the benchmark. On the local
Strix Halo, HIP reaches 0.0195 seconds for the 4,096 x 13,661 shape, 28.27x
faster than the current NumPy affine route and 1.76x faster in raw wall time
than the retained 0.0343-second Metal result. These are different GPUs, so the
cross-device comparison is descriptive rather than a portable promise. This is
a fast ranking gate, not a replacement for float-float, complex128, or rigorous
high-precision validation when the route requires them.

`AffineNumpyPlan`, `AffineMetalPlan`, `AffineCudaPlan`, and `AffineHipPlan`
share the same validation, batching, value, cache-management, and
`contour_metrics` contract. Pass `backend="numpy"`, `"metal"`, `"cuda"`, or
`"hip"` to force a route; `auto` prefers Metal on macOS, HIP on Linux AMD
hosts, CUDA on other GPU hosts, and otherwise uses NumPy. The CUDA
implementation has been parity-tested on a Modal NVIDIA L4; the HIP
implementation targets the local `gfx1151` Strix Halo device and is validated
by `make hip-test`.

## GPU suitability survey

The affine contour population is the current evidence-backed GPU path. Its
large complex64 matrix product and compact reductions amortize transfers. The
retained Metal benchmark is 4.97x-10.88x faster than two-thread NumPy. A
warmed best-of-three Modal L4 validation was 25.2x faster on a synthetic
1,024 x 4,097 shape. The local HIP path fuses the same reductions on `gfx1151`;
its end-to-end receipt is recorded only after the real host benchmark passes.
All backends must have zero winding disagreements.

The exact sparse-rank, graph64, digest, Dirichlet-inverse, and rigorous Arb
kernels remain CPU paths. Their finite-field or interval contracts, irregular
branching, or low arithmetic intensity do not map honestly to the current GPU
precision and transfer model.

For small search programs that repeatedly apply permutations to integer masks,
use `u64_mask_lut` once per permutation and compose lookup tables before the
candidate loop. For batched actions through degree 64, use
`PermutationActionPlan`: it retains the complete supplied permutation collection
and byte tables across calls, provides deterministic canonical images and an
early-exit canonical-minimum test, and dispatches sufficiently large measured
shapes to HIP. The degree-16 tuple lookup remains useful inside tiny Python loops;
the retained plan is the general batch interface.

The compact Chebyshev-Filon contraction is still a plausible GPU target, but
its accepted contract is complex128. MLX Metal does not execute float64, so a
direct port would weaken the result to a ranking scout. Promote it only after
a paired CUDA/Metal precision design, such as validated float-float Metal,
matches a captured real autocorrelation and beats the warmed CPU baseline.

## Parallel dispatch

The default build pins ForkUnion to the audited revision recorded in
`THIRD_PARTY.md`. Persistent caller-inclusive pools are cached per calling
thread and requested worker count, then placed in ForkUnion's sleeping state
after every batch so an intervening serial phase consumes no worker CPU. The
next dispatch wakes the retained pool. Balanced Taylor chunks use static
slices; irregular graph, segment, accumulation, moments, and two-level jobs use
dynamic work stealing. Disable the dependency with
`-DFAST_MATH_USE_FORKUNION=OFF` to use the equivalent standard-thread
fallback.

## Persistent FINUFFT plans

FINUFFT stays the transform backend; fast-math removes repeated orchestration
work around it:

```python
from fast_math import Type3Plan1D

plan = Type3Plan1D(
    log_n,
    targets,
    n_trans=5,
    eps=1e-12,
    nthreads=6,
)
basis = plan.execute(coefficient_stack, out=output_buffer)
```

`Type1Plan1D` retains scan nodes and FFT planning while coefficients change.
`Type3Plan1D` retains source and target coordinates, and `set_targets` reuses
the plan when only the target batch changes. FINUFFT is optional at import
time; install `fast-math[finufft]` to use these classes.

## Rigorous Arb cache workspace

When FLINT is discoverable through `pkg-config`, the shared library exports
`fast_math_arb.h`. A source loop initializes one workspace, computes
`fast_math_arb_weight_from_log` once per index, and passes the resulting Arb
weight to every related real or complex cache value. This preserves the
binary64 midpoint and rigorous directed error sum while removing per-value
Arb allocation and duplicate general powers.

For independent source terms, `fast_math_arb_ordered_map_reduce` maps chunks
through the persistent ForkUnion pool and retains one weighted Arb error term
per item and error stream. It then performs each stream's additions in original
item order. This is intentionally not a tree reduction: it preserves the exact
serial Arb certificate and binary64 cache centers across worker counts and
chunk boundaries. Temporary memory is linear in items times error streams.

## Induced-profile contract

`induced_subgraph_profiles` takes one dense class ID for every labeled
`k`-vertex edge mask and returns exact class counts for each input graph.
Edge-mask bits follow `itertools.combinations(range(k), 2)` order. This keeps
canonical labeling outside the hot loop: nauty, Traces, or an exhaustive small
lookup defines the classes once, while fast-math performs the repeated subset
extraction and counting. Orders through seven are supported; output counts are
64-bit.

`induced_subgraph_profile_stack` accepts several strictly increasing induced
orders and their class tables. It validates the graph batch once, enters the
scheduler once, and writes the per-order rows contiguously. `counts_for_order`
returns a view into that matrix without copying or reclassifying. For bounded
subset domains, native calls hoist the repeated combination traversal into
compact edge-probe plans shared by every graph; larger domains retain the
allocation-bounded direct traversal.

`encode_graph6` converts validated undirected adjacency-mask batches to exact
unheaded short-form graph6 records in input order. `decode_graph6` accepts the
same format with one shared order from 1 through 62. The remaining graph
kernels accept undirected adjacency-bitmask batches through order 64.

`undirected_csr` stores sorted symmetric nonloop adjacency plus an optional
loop mask and one optional 64-bit color-set mask per edge.
`enumerate_csr_triangles` emits exact nondecreasing triples, including
`(a,a,a)`, `(a,a,b)`, and `(a,b,b)` when looped type vertices are present.
For colored input, each output row carries the three edge color-option masks
in `(left,middle)`, `(left,right)`, `(middle,right)` order.

`csr_common_neighbors` accepts an explicit pair batch and returns cumulative
`pair_offsets`; `counts` is the zero-copy difference of those offsets.
`materialize=True` additionally returns each pair's sorted common-neighbor
row. The native wrapper uses a guarded degree-sum upper bound for one-pass
materialization when the temporary allocation is bounded, otherwise it falls
back to an exact count-then-fill path. On the retained N=13 representation-193
type graph, the one-pass route reproduced the reference's 1,057,077 entries
byte-for-byte in 0.00492 seconds, versus 2.576 seconds for Python.

`canonicalize_colored_digraphs` uses the nauty library directly when
`libnauty` is available. It accepts packed directed adjacency and vertex
colors, and returns canonical permutations, canonical adjacency/colors,
batch-local isomorphism class IDs, automorphism group sizes, orbit counts,
and optionally generator permutations. Native batches run independent nauty
calls in parallel when `threads` is nonzero. Set
`collect_automorphism_generators=False` for census routes that do not consume
the generators. The exhaustive Python backend is intentionally limited to
order nine; it is a contract reference, not a production isomorphism engine.

The group and Cayley-CI contracts, array conventions, formulas, and complete
13-group atlas replay are documented in `INTERFACE.md`. On the local Ryzen AI
Max+ 395, the August 17, 2026 native route takes `0.09601842903066427` seconds
while reproducing all 11,664 automorphism-orbit representatives and 9,606 graph
fibers exactly. This is `52.333918808805166x` faster end to end than the retained
Python/NetworkX/`labelg` route. The TLS-enabled nauty build parallelizes exact
independent canonical searches. A representative degree-18 order, point-orbit,
and membership route takes `0.0011079729697667062` seconds through a retained
`PermutationGroup`, `1323.7553063434973x` faster than the equivalent GAP process
route with identical output.

## Adaptive oracle area contract

`adaptive_areas` and `exact_adaptive_areas` solve the optimal legal adaptive
query policy for real targets on the Rademacher cube `{-1,+1}^n`. A target is a
table of length `2**n`; bit `i` of the point index is `0` for `X_i = -1` and `1`
for `X_i = +1`. A restriction is a code in `range(3**n)` whose base-three digit
`i` is `0` for free, `1` for `X_i = -1`, and `2` for `X_i = +1`. The recursion is

```text
A(rho) = 0                              when Var(g | rho) = 0
A(rho) = Var(g | rho) + min over fresh i of
         ( A(rho, X_i = -1) + A(rho, X_i = +1) ) / 2
```

Fixing a free coordinate strictly increases the restriction code, so one
descending scan of the lattice solves every state in `O(n * 3**n)` with two
`3**n` arrays per worker and no recursion, memo dictionary, or tuple key.

```python
import numpy as np
from fast_math import adaptive_areas, exact_adaptive_areas, quadratic_target_tables

batch = adaptive_areas(tables, threads=8, restrictions=True)
batch.areas                  # root area per target
batch.first_coordinates      # optimal first query, -1 for a constant target
batch.variances              # Var(g | rho) for every restriction
batch.areas_by_restriction   # A(rho) for every restriction
batch.policies               # optimal query per restriction, -1 at a leaf
```

`exact_adaptive_areas` runs the same recursion in integers for integer-valued
targets. Variance numerators use denominator `4**n` and area numerators use
denominator `2**(3*n)`; `ExactAdaptiveAreaBatch.areas()` returns exact
`Fraction` values. Overflow of the int64 numerators is an error, never a wrapped
result, and table entries must satisfy `2**n * max(entry**2) < 2**62`.

`quadratic_target_tables` expands degree-two Walsh data `(a, b)` into the full
tables, so a search that stores only linear and pair coefficients does not need
a second Bellman implementation. Ties choose the smallest optimal coordinate, so
both backends and every thread count return bitwise identical policies.

`zero_tolerance` prunes a subcube whose float variance falls to the given
threshold. Use `0.0` for the mathematically exact rule and a positive value only
as an explicit numerical policy.

## Fixed-width digest contract

`digest_u64_rows` hashes exact nonnegative integer rows without constructing
Python tuples, `repr` strings, or JSON. Each SHA-256 input is:

1. the ASCII/NUL format tag `fast-math/u64-rows/1\0`;
2. the namespace byte length as little-endian `uint64`;
3. the caller namespace bytes;
4. the field count as little-endian `uint64`;
5. every row value as little-endian `uint64`.

Use a namespace that identifies the mathematical coordinate system, including
the canonical-class table or its digest. The field count and namespace prevent
accidental equality across incompatible profile schemas. The native backend
uses ForkUnion across rows and the platform SHA-256 implementation on Apple;
the bundled C++ SHA-256 fallback and Python reference backend preserve the
same portable byte contract.

## Sparse modular rank contract

`sparse_rank_mod_u32` accepts zero-based CSR with strictly increasing column
indices within each row. Coefficients are reduced exactly over a caller-chosen
32-bit prime. A target of zero computes full rank; a positive target stops as
soon as that many independent rows have been certified. Returned pivot rows
and columns use the caller's original coordinates and form a nonsingular
witness minor.

The native kernel first peels degree-one rows and columns as exact pivots,
iterating structural cascades before allocating the numeric core. It then
compacts the residual matrix, applies deterministic low-degree column and
sparse-row ordering, and eliminates through one dense modular workspace with
an occupancy bitset and a contiguous immutable pivot arena. Peeling preserves
rank and the returned witness minor over every field; the result reports both
the peeled fringe and residual-core dimensions.

On the retained order-eight graph matrix, deterministic Markowitz row ordering
reduces exact elimination work from 69,038,394 to 10,350,016 steps while
reaching rank 12,341 with an exact witness. Three recorded runs take
1.391-1.400 seconds versus the 124.82-second route baseline, an 89.3x median
speedup. Generic degree-one peeling contributes two pivots on this instance;
fill-aware ordering supplies the dominant gain. Retained measurements are
recorded under `benchmarks/results/`.

`sparse_rank_mod_u32_batch` runs independent fields through the persistent
ForkUnion pool and preserves prime order. On the same matrix, two fields took
2.137 seconds serially versus 1.230 seconds in a cold two-worker batch and
1.246 seconds with the existing pool. The elimination within one field remains
serial because pivots are data-dependent. FMA, floating-point SIMD, and Metal
are intentionally excluded from this kernel because they do not preserve
exact finite-field arithmetic.

`sparse_block_coloops_mod_u32` exposes stronger preprocessing when rows have
meaningful bounded blocks, such as several invariant channels for one graph
card. A column is removed only when one block supplies a local linear
functional that evaluates to one on that column and zero on every other
currently active column. The returned removal order and fixed-width
functional coefficients form a triangular exact certificate: each functional
annihilates every later removed column and the residual core. Therefore
`rank(original) = removed_columns + rank(residual_columns)`.

This API deliberately uses dual certificates rather than claiming that the
removed columns already define an original-coordinate row minor. The ordinary
sparse-rank API continues to provide minor witnesses for the residual matrix.
On the retained attachment matrices, four-row blocks remove 13,115 of 19,320
order-16 columns in 0.072 seconds and 8,392 of 48,629 order-17 columns in
0.122 seconds. Larger eight- and sixteen-row blocks expose no additional
order-17 columns, so four rows remain the evidence-backed default.

## Elliptic curve rank search contract

The module exists to find elliptic curves of high Mordell-Weil rank relative to
their size, which is a search over families rather than over single curves.

`mestre_locus_sextuples` enumerates the integer sextuples that define those
families. With the first power sum zero, fixing `a1..a4` turns the locus
condition `12 p5 = 5 p2 p3` into a quadratic in `a5 a6`, so the enumeration
solves the locus instead of sampling it. Entries bounded by 40 give 10,356
primitive sextuples in 3 seconds, of which 201 are neither degenerate nor
closed under negation. That filter matters: a sextuple closed under negation
makes the quartic even in `x`, which pairs the twelve base points and halves
the rank the family can carry.

`mestre_quartic` builds one fibre exactly over Q. The returned quartic has
integer coefficients, cleared by a square so the twelve base points still
satisfy the returned equation, and `quartic_to_weierstrass` maps those points
onto the Jacobian through a rational base point. Both record curves that the
construction is known to produce are reproduced exactly by the tests, including
the rank-19 curve with `log|D_min| = 156.3436`.

`mestre_ap_tables` is the reason the search is cheap. The trace of Frobenius of
a fibre depends only on the parameter modulo `p`, so one `O(p^2)` table per
prime scores unboundedly many rational fibres afterwards by gather. Tabulating
every prime below 2000 for one family takes 0.089 seconds against 6.96 seconds
for the NumPy reference, and 1.7 seconds carries the bound to 6000. Scoring
then runs at 1.77 million fibres per second over 301 primes, which is 534
million gathers per second.

Table entries are `-sum chi(r(x)) - chi(lead)` over `F_p`. That equals `a_p` at
primes of good reduction and stays a bounded surrogate at the fibres that
degenerate modulo `p`, which is what a Mestre-Nagao sum wants and costs no
per-fibre discriminant test. Tests pin the values against an independent path,
reducing the exact integer fibre rather than building it modulo `p`.

`quartic_points` searches `y^2 = r(x)` for `x = u/w` in a box. Small primes
bit-sieve the box down to pairs where the homogenized quartic is a quadratic
residue at every sieve prime, and each survivor gets an exact integer square
test, so coefficients of thirty digits cost nothing extra. Acceptance patterns
are precomputed per prime and denominator class, which turns the per-denominator
work into one AND pass and reaches 350 million pairs per second.

The search never drops a point silently. The kernel reports the true candidate
count even when the output buffer is too small, and the Python layer responds by
halving the denominator span until every piece fits, so results are independent
of `capacity`. A single denominator that cannot fit raises rather than
truncating. This was a real defect, not a hypothetical one: silent truncation
made a wider search return fewer points than a narrower one, and it now has
regression tests for monotonicity in the box, in the sieve prime count, and
under a deliberately cramped buffer.

## Production survey

The live L=0.1529 route showed that coefficient generation was the wrong API
boundary: its real consumer immediately split 100 million outputs into about
97,881 weighted pieces and five outer blocks. Pulling that whole reduction
into `fused_two_level` removed the dominant allocation and Python traversal.

The live L=0.0999 radial-cubic route repeatedly computes power moments after
batched FINUFFT evaluation. The reusable postprocessing now lives in
`power_moments`, but its retained scans spend roughly 120-350 seconds in model
evaluation. Reimplementing FINUFFT inside this library would duplicate an
already specialized backend; the next useful shared layer there is reusable
Taylor/source preparation and phase-cache orchestration around FINUFFT.

The live L=0.1479 finite bridge repeatedly constructs rigorous Arb source
caches. Recent checkpoint runs around N=720,000-750,000 took 17-60 seconds.
At N=725,989, the shared paired-cache benchmark now evaluates 1,451,978 cache
values in 0.397 seconds with three workers, 2.58x faster than the optimized
serial path and 4.13x faster than the allocation-heavy baseline. Center hashes
and the complete printed Arb certificate are identical. The remaining route
work is generated-evaluator integration and fixed-offset cache assembly; the
library primitive itself is no longer the blocker.

The route's 97,879 independent two-sided output-weight intervals use
`fast_math_arb_weight_intervals_u64`. At N=756,989, three workers reduce the
native interval phase from 0.232 seconds to 0.081 seconds with byte-identical
binary64 records. One worker remains the route default for fleet throughput.

The live graph-reconstruction Kocay route formed a 162,864 by 12,346 sparse
moment matrix with 1,311,046 nonzeros. Its route-local modular elimination took
124.82 seconds at `p=1,000,000,007`. The shared sparse-rank kernel preserves
rank 12,341 and deterministic witnesses while fill-aware ordering cuts work to
10,350,016 eliminations. The repeatable benchmark records a 1.391-second best
and 1.398-second median. Independent prime checks use ForkUnion; their
data-dependent pivot chains remain separate.

The order-17 attachment matrix is a different regime: generic degree-one
peeling removes only 9 of 48,629 columns, while four-row local coloop
certificates expose an exact 8,392-column fringe. The resulting
40,237-column core reaches full residual rank in 1,140.15 seconds, proving full
rank 48,629 for the original matrix. Markowitz ordering cuts 28.7% of the
degree-ordered core's elimination steps and 15.8% from its maximum working
width. Reducer plus residual elimination takes 1,140.27 seconds of kernel time,
3.25x faster than the retained 3,704.68-second LinBox run. The complete
machine-readable records live under `benchmarks/results/`.

The retained order-18 proof quotient gives the same backend verdict:
deterministic Fast Math elimination takes 22.33 seconds median, while LinBox
1.7.0 scalar Wiedemann takes 49.36 seconds median. LinBox's deprecated
OpenMP block-rank example does not build against its retained installed
headers, so Fast Math does not expose a black-box rank backend.

## Optimization contract

An optimization is acceptable when:

1. Native output agrees with the executable reference model.
2. Repeated native runs are bitwise deterministic.
3. Pair counts and source indexing remain unchanged.
4. Throughput improves on both `smoke` and `medium`.
5. Peak RSS does not regress without a recorded reason.

Keep operations here only when they recur across Lambda routes and have a
stable mathematical contract. Keep rigorous interval semantics in Arb and
nonuniform Fourier transforms in FINUFFT; optimize their shared setup,
batching, and data movement rather than replacing them without evidence.

See `TARGETS.md` for the evidence-backed queue across Lambda, graph
reconstruction, Ramsey, and exact incidence workloads.
