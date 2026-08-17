# fast-math

Shared native compute kernels for mathematical research. Python defines the
public APIs and executable reference models; C++20 performs high-volume
arithmetic behind a small C ABI. Repeated native calls use pinned ForkUnion
worker pools by default, with a build-time standard-thread fallback.

`import lambda_fast`, `LAMBDA_FAST_LIBRARY`, `liblambda_fast`, and the
`lambda_fast_*` C symbols remain supported compatibility interfaces.

## Kernels

- `fast_math.graphs`: graph6 encoding/decoding,
  degree/edge/triangle/wedge/induced-P3 invariants, induced-subgraph class
  censuses, pair profiles, and clique or independent-set witnesses for graph
  orders up to 64; loop-aware colored triangle enumeration on large symmetric
  CSR graphs; and optional batched nauty canonicalization for directed
  vertex-colored graphs.
- `fast_math.groups`: permutation point orbits, deterministic Schreier-Sims
  stabilizer chains, exact group order and membership, and explicit finite
  double-coset partitions for degrees through 512.
- `fast_math.ci`: inverse-closed connection-set orbit enumeration, exact
  fixed-weight subset orbits without materializing the full weight slice, batched
  Cayley graph construction into the existing nauty API, generalized
  dihedral/Hol(A) helpers, R-0805 derivative-group orbits, exact
  coherent-configuration/2-WL refinement with intersection numbers, and
  bounded-degree `u64_mask_lut`/`compose_u64_mask_luts` microkernels for
  Python search loops over small packed connection masks.
- `fast_math.reductions`: deterministic power moments and segmented complex
  sums, L1 masses, and total variations.
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

Research agents can invoke the deployed build without environment setup:

```sh
cd /home/kenan/projects-research
./tools/fast-math -c 'import fast_math; print(fast_math.__file__)'
./tools/fast-math path/to/research_script.py
```

The launcher supplies `PYTHONPATH` and the native library paths. Coverage includes
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

See `HACKATHON.md` for the contributor workflow and optimization acceptance
rules.

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
make ci-weight-orbits
make union
make union-closure
make union-closure-routes
make digests
make sparse-rank
make sparse-rank-batch
make arb
make finufft
make finufft-cells
make finufft-canopy
make finufft-prime-shell
make metal
make hip-test
make mask-lut
make filon
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

For small CI search programs that repeatedly apply permutations to integer
connection masks, use `u64_mask_lut` once per permutation and compose lookup
tables before the candidate loop. This removes repeated Python set-bit walks;
the table is intentionally bounded at degree 16, while larger actions should
use the native packed orbit APIs instead.

The compact Chebyshev-Filon contraction is still a plausible GPU target, but
its accepted contract is complex128. MLX Metal does not execute float64, so a
direct port would weaken the result to a ranking scout. Promote it only after
a paired CUDA/Metal precision design, such as validated float-float Metal,
matches a captured real autocorrelation and beats the warmed CPU baseline.

## Parallel dispatch

The default build pins ForkUnion to the audited revision recorded in
`THIRD_PARTY.md`. Persistent caller-inclusive pools are cached per calling
thread and requested worker count. Balanced Taylor chunks use static slices;
irregular graph, segment, accumulation, moments, and two-level jobs use dynamic
work stealing. Disable the dependency with
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
