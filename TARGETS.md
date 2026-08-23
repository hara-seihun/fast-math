# Fast-Math Optimization Targets

This ledger records shared kernels only. A target belongs here when at least
two research routes need it, its mathematical contract is stable, and a
representative benchmark shows material time or memory cost.

## Shipped

| Target | Current users | Contract |
| --- | --- | --- |
| Streamed truncated Dirichlet reduction | Lambda L=0.1529 | Fuse coefficient generation, weighted fine pieces, and two-level outer records without materializing the output vector. |
| Finite Dirichlet inverse | Lambda mollifier routes and arithmetic scouts | Deterministic inverse coefficients for a finite source sequence. |
| Power/current moments | Lambda L=0.0999 and Sobolev scouts | One-pass value, ordinary derivative, phase-current, and radial moments. |
| Segmented complex statistics | Lambda weighted blocks and Abel/variation scans | Per-segment complex sum, L1 mass, and total variation. |
| Taylor source preparation | FINUFFT-based Lambda models | Batched coefficient stacks and value/log-moment evaluation around an external transform. |
| Graph64 pair profiles | Graph reconstruction and Ramsey R(5,5) | Batched exact adjacency/common/exclusive/nonneighbor pair counts for graphs of order at most 64. |
| Graph64 clique search | Ramsey R(5,5) and graph filters | Batched exact clique or independent-set witnesses for graphs of order at most 64. |
| graph6 batch decode | Graph reconstruction and Ramsey catalogue scans | Decode unheaded short-form graph6 directly into graph64 adjacency masks without NetworkX object construction. |
| graph6 batch encode | Tree attachment jets and bounded P3-gluing incidence routes | Encode validated graph64 adjacency-mask batches as exact unheaded short-form graph6 records in input order, with a portable reference backend. On the retained order-17 attachment route, cProfile wall time fell from 57.470 s to 19.340 s (2.972x); the 77,280-by-48,629 CSR matrix, 853,712-entry SMS file, card catalog, and tree catalog are exactly unchanged. |
| Graph invariants | Ramsey local identities and reconstruction filters | Batched degree, edge, triangle, wedge, and induced-P3 counts. |
| Large CSR colored triangles | Union-closed factor compatibility graphs | Enumerate exact nondecreasing type triples on sorted symmetric CSR, including loop-induced repeated vertices, and return one 64-bit color-option mask per triangle edge. |
| Batched CSR common neighbors | Union-closed classifiers and sparse graph filters | Count or deterministically materialize sorted common-neighbor rows for explicit vertex pairs without dense adjacency or quadratic all-pairs work. |
| Batched nauty canonical digraphs | Union-closed factor-type enumeration and shard merging | Canonical permutations, canonical adjacency/colors, batch isomorphism classes, automorphism group sizes, orbit counts, and generator permutations for directed vertex-colored graphs. |
| Permutation groups and Cayley-CI pipeline | Cayley-CI atlas, derivative-orbit, and coherent-configuration routes | Exact permutation orbits, reusable immutable stabilizer chains, deterministic Schreier-Sims order/membership, explicit double cosets, complete-action and generated-action inverse-closed subset quotients, native atom expansion, batched Cayley construction, TLS-parallel nauty, R-0805 derivative orbits, and 2-WL basic relations with separate refinement-only output or verified intersection numbers through degree/order 512. On the local Ryzen AI Max+ 395, the complete 13-group atlas reproduces 11,664 first-quotient orbits and 9,606 graph fibers exactly in 0.09601842903066427 s median, 52.333918808805166x faster end to end than Python/NetworkX/`labelg`. The representative GAP process route for order, point orbits, and membership is 1323.7553063434973x slower than `PermutationGroup`. |
| Packed union-closure batches | Three-minimal-generators and singleton-channel nonproduct scouts | Return exact closure flags for `uint64` family-membership masks on ground sets through size six, using a portable native backend with an executable NumPy reference. On Modal portable x86_64, 65,535 ground-four families fell from 0.004867 s to 0.000968 s median (5.03x), random 65,536-family ground-five and ground-six batches improved 11.75x and 50.59x, and complete outputs were identical. End-to-end retained route wall time improved 1.067x and 1.031x. |
| Batched adaptive oracle areas | Adaptive-oracle-area depth-two hull search, Plackett-Luce block engine, and exact supersolution probes | Solve the optimal legal adaptive policy on the ternary restriction lattice of `{-1,+1}^n` for a batch of targets in one descending scan, returning root area, optimal first query, and complete per-restriction variance, area, and policy arrays. The exact backend carries `A(rho) * 4^n * 2^(n-fixed)` so no division occurs, reports numerators over `2^(3n)`, and errors rather than wrapping. Both backends reproduce the campaign's independent `Fraction` and float engines exactly. Per target this is 300x to 900x the memo-dictionary route: `n=13` fell from 17.006 s to 0.040 s and `n=16` now solves in 1.407 s where the earlier route was unreachable. Batched search loops reach 446,120 targets/s at `n=6` and 65,453/s at `n=8` on eight threads. |
| Induced-subgraph profiles | Graph reconstruction attachment-rigidity shards | Count caller-defined canonical classes through order seven, hoisting bounded subset edge probes across the graph batch. |
| Multi-order induced-profile stacks | Reconstruction profile certificates and collision scans | Validate and dispatch once, then concatenate exact class counts for several induced orders. |
| Rigorous Arb cache accumulation | Lambda B-process source caches | Reuse caller-owned Arb/Acb scratch state and one rigorous weight per index while preserving binary64 centers and directed interval sums. |
| Deterministic ordered Arb map/reduce | Lambda B-process source caches and rigorous parameter sweeps | Evaluate independent interval terms through ForkUnion, retain per-item errors, then reduce each stream in original order for exact serial certificates. |
| Directed Arb endpoint intervals | Lambda B-process output-weight caches | Evaluate independent endpoint powers at fixed offsets with worker-local Arb state and outward binary64 rounding. |
| Persistent FINUFFT plans | Lambda Taylor models and canopy scans | Retain type-1/type-3 plans, coordinates, many-transform configuration, and output buffers across repeated executions. |
| Fixed-strength persistent FINUFFT type-3 plans | Primary-inverse, truncated-inverse, and phase-faithful interval scouts | Retain one coefficient stack and reusable output storage while target chunks change. On the portable Modal x86_64 fixture with 690,988 sources, four changing 500,000-target chunks, and three transforms, wall time fell from 0.371 s to 0.307 s median (1.21x) with complete output identical to the separate persistent-plan path. Darwin retains simple calls because persistent plans regress on the same route shape. |
| Paired-sign persistent FINUFFT type-3 plans | L=0.1479/L=0.1529 phase-faithful finite bridges and retained canopy/proof-system scans | Evaluate positive- and negative-sign coefficient stacks over shared source nodes through one retained positive-sign plan using conjugate symmetry. On the portable Modal x86_64 fixture with 690,988 sources, four changing 500,000-target chunks, and three transforms per sign, wall time fell from 0.645 s to 0.456 s median (1.42x), with maximum complete-array relative difference 9.02e-16. Darwin retains separate simple calls because the same fusion measured only 0.50x-0.77x there. |
| High-order phase-faithful interval plans | L=0.0999 retained replay, easiest-Lambda canopy floor, and L=0.157 canopy run | Reuse one 27-transform paired primary/dual type-3 plan while 250,000-target chunks change, with a fixed-plan branch for primary-only scans. On the retained portable x86_64 shape with 690,988 sources and four chunks, paired wall time fell from 2.388 s to 2.041 s median (1.17x), with maximum complete-array relative difference 8.65e-16. The live retained run spans 384 chunks, so the 0.227 s setup amortizes to a projected 1.17x transform speedup. Darwin retains simple calls. |
| Fixed-R FINUFFT canopy transform stacks | L=0.0999 product canopies and Sobolev derivative scans | Retain one type-1 plan per source factor, execute value and first/second log moments as one preallocated stack, and preserve complete complex outputs and witness locations. On the retained 500,000-source, 131,072-mode, eight-offset fixture, transform time fell from 0.716 s to 0.530 s median (1.35x) with maximum complete-array difference 2.73e-15; route scan time fell from 1.230 s to 0.861 s (1.43x) and total time from 1.368 s to 1.048 s (1.31x). |
| Dyadic prime-shell FINUFFT plans | L=0.0999 prime-shell canopies and carrier-band scouts | Retain one type-1 plan and preallocated input/output buffers per dyadic shell across mesh offsets. On the interleaved six-shell, 145,421-prime, 48-transform fixture, transform time fell from 0.665 s to 0.428 s median (1.55x); every full-mode maximizing witness and the final radius decision are unchanged, with maximum complete-array difference 2.02e-17. |
| Persistent GPU affine plans | Contour optimizers and population searches | Retain complex64 base/basis arrays on Metal or CUDA and fuse winding, phase-increment, and edge-floor reductions before downloading compact ranking outputs through one shared contract. |
| Portable affine plan and auto dispatch | Contour optimizers, Modal CPU fallbacks, and backend-neutral tests | Preserve the Metal/CUDA retained-array contract on NumPy and select Metal, CUDA, or NumPy without route-local backend branches. |
| Fixed-width exact row digests | Graph profile certificates and sparse incidence builders | Hash namespaced little-endian uint64 rows in parallel without Python tuple, repr, or JSON construction. |
| Exact sparse modular rank | Graph reconstruction incidence and moment matrices | Deterministic CSR rank and original-coordinate pivot witnesses over any 32-bit prime, with exact degree-one fringe peeling, compact dense-bitset elimination, and ForkUnion-parallel independent fields. The retained order-8 portable x86_64 route now hoists the row-constant factor-one branch and uses the reciprocal reduction's single sufficient correction. Two sequential interleaved A/Bs reduced median kernel wall time by 5.76% and 5.10%, a combined 1.118x speedup, with rank 12,341, all elimination statistics, and pivot digest `18014335152013357997` unchanged. |
| Retained dense modular systems | Finite-function profile and projective-function quotient scouts | Deterministic canonical RREF, pivot/free columns, row transform, both nullspaces, inverse, and fixed-matrix solution batches over uint32 prime fields. Native CPU and HIP return a particular solution for each consistent right-hand side or the first retained left-null inconsistency witness, with executable reference and SageMath certificate replay. |
| Bounded-block sparse coloop reduction | Multi-channel graph incidence matrices and Tanner-style observability systems | Iteratively remove columns exposed by exact local row-block duals, returning triangular certificates and a residual-column mask before numeric rank. |
| Portable SIMD Chebyshev-Filon tail contraction | L=0.0999 power-Gram and compact-Filon proof-system routes | Evaluate two adjacent endpoint-expansion lags through portable 128-bit compiler vectors while preserving lag-ordered compensated accumulation and 256-lag phase resets. On a captured real node-64 autocorrelation with 17,865,985 output lags, one-thread best fell from 0.666 s to 0.188 s (3.54x); eight workers reach 0.0301 s. The result is bit-identical to the pre-SIMD compact kernel and differs from the full cached row by 1.42e-11 relatively. |
| Shift-divisor gate scan | Erdos 647 witness campaign (ledger conjecture `f6f6409e`) | Derive per-prime alive tables and prime-forced linear forms from the modulus by exact integer arithmetic, verify them against direct factorization, then scan v-intervals with a wheel-compressed segmented sieve and deterministic Miller-Rabin, identically across reference, native CPU, and HIP. On the Erdos 647 gate (13 forms, extended wheel 2^7*17*19*23*29*31, density 0.0142, v ~ 2.7e14): reference 8.2e6 v/s, native CPU 5.67e10 v/s, HIP gfx1151 3.43e12 v/s, byte-identical survivors. The HIP kernel pairs adjacent sieve primes up to 251 into CRT bit-pattern tables stamped without atomics (the per-row shift is uniform across forms), leaves atomics only for (251, 2048], and hands survivors to a fast-path Miller-Rabin stage (split-word reduction, exact division by inverse, quadratic-residue prefilter before isqrt); sieve depth 2048 is the measured optimum against that stage. A decade of v (e.g. n up to 9.7e20) scans in about 12 GPU minutes. |
| Base-p point codec and class tables | Feistel key-schedule, Henon, depth2-coordinate, and projective-WL routes (the duplication scan counted this family at 88 scratch programs on 2026-08-23) | Batched little-endian index/digit conversion over encoded `F_p^n` points (`p <= 251`, `n <= 16`, uint64 codes), digit-wise negation codes, projective scalar normal forms, and whole-space dense negation/scalar class ids with exact sizes `(p**n + 1) // 2` (odd `p` negation) and `(p**n - 1) // (p - 1) + 1` (scalar). Reference and native agree bitwise; class tables match an independent unique-over-canonical-forms construction. Scratch-route workspace construction (digit table + negation codes + dense negation classes + projective representatives) fell 26x-55x on gmktec: F_3^6 0.00262 s -> 0.000094 s (27.95x), F_5^6 0.0533 s -> 0.00104 s (51.2x), F_7^6 0.407 s -> 0.00760 s (53.5x), F_5^8 1.490 s -> 0.0317 s (47.0x), with route outputs asserted equal before timing. |

## In progress

| Date | Target | Current users | Owner | Scope |
| --- | --- | --- | --- | --- |
| 2026-07-28 | Fast Math final validation | Shared native and portable library contract | `019faa50-230e-77f2-a93e-ee05c98707dc` | Build and test the exact current source on governed Liminal compute in native and portable configurations, retain the bounded receipt, and release before scheduler closeout. |

## Measured negative verdicts

### Sparse modular-rank floating-point reciprocal

Do not replace the exact reciprocal-high reduction with a corrected binary64
quotient estimate. On the retained order-8 portable x86_64 matrix, six
interleaved repetitions preserved rank 12,341 and pivot digest
`18014335152013357997`, but median kernel wall time regressed from `0.966197 s`
to `1.046745 s` (`0.923x`, 8.34% slower). Evidence:
`problems/graph-reconstruction/scratch/kocay-coverings--fast-math-sparse-rank-hot-path/float-reciprocal-ab-modal-2026-07-28.json`.

### L=0.1479 two-worker ordered-Arb source caches

Do not activate two-worker source-cache generation as the proof-route default.
After moving source, weight, and roundoff timings into non-proof sidecars, a
fresh paired N=863,989, Q=40 checkpoint produced byte-identical cache files and
byte-identical fresh proof certificates, but wall time changed only from
`197.722823 s` to `196.830230 s` (`1.004535x`). The earlier local run improved
from `96.408788 s` to `86.888187 s` (`1.109573x`), also far below the 2x
adoption bar. One worker remains the route default. Reopen only for a
materially different checkpoint profile or after a downstream verifier/replay
optimization changes the end-to-end ceiling. Evidence:
`fast-math/benchmarks/results/lambda-l01479-route-integration-after-sidecars-v2-2026-07-28.json`.

The route-local timing-sidecar repair itself is retained. Final Apple replay
produced identical baseline/integrated SHA-256 values for the replay,
transcendental, cache-input, and repaired certificates, so elapsed telemetry
no longer perturbs proof bytes.

### L=0.1479 downstream accepting verifier

Do not add a new Fast Math kernel for the residual verifier wall time. On the
fresh N=863,989 checkpoint, cache-input regeneration took about 19-21 seconds,
the transcendental derive-and-verify phase took about 29-35 seconds, and final
acceptance took 1-2 seconds. The transcendental phase already uses
`fused_two_level`; retained larger production shapes execute that kernel in
0.113-0.142 seconds, below one percent of the measured stage.

The remaining large cost is route orchestration. L=0.1479 derives the complete
replay twice in `--derive-and-verify-replay` mode, while the retained L=0.1529
checkpoint verifier derives it once. A future one-pass L=0.1479 transcript
generation/verification repair is route-local and does not meet the two-user
Fast Math admission bar. Evidence:
`problems/riemann-hypothesis/scratch/proof-system--fast-math-l01479-downstream-profile/profile.json`.

## Next

### Subgroup backtrack, centralizers, and normalizers

Evidenced 2026-08-18 from the Cayley-CI alignment corpus. `fast_math.groups`
ships orbits, immutable stabilizer chains, order, membership, and double
cosets, but every route needing a centralizer, normalizer, subgroup lattice,
semiregular or regular subgroup enumeration, or conjugacy of subgroups leaves
the library. Two artifacts state the gap directly: the colour-holonomy image
frontier note records "Fast Math has no centralizer or subgroup-backtrack API"
and falls back to GAP for automorphism groups, centralizers, and normalizers,
and the polycirculant counterexample note records the missing finite-field
module-submodule, group-cohomology, and subgroup-class APIs. A 665-line parity
script reimplements `perm_mul`, `perm_inv`, `perm_conj`, group closure, graph
automorphism backtracking, semiregular subgroup enumeration, subgroup
conjugacy labels, centers, and group isomorphism search in pure Python tuples.
Ship the group-side primitives before the specialized transporter contracts:
centralizer and normalizer of a subgroup, subgroup closure and conjugacy
classes, and automorphism generators for a colored graph reusing the existing
nauty path. GAP stays the independent reference model, not the production
route.

### Fused finite-field enumerate-and-reduce

Evidenced 2026-08-18 from the Cayley-CI Gale-circuit corpus. The modular and
modular-linear kernels take prepared batches, which is the wrong shape when the
batch is generated combinatorially: an exhaustive shear-profile search over
`GL`-gauge orbits builds a 2,496-by-500 `F_5` system per profile across millions
of profiles, so no caller can materialize the batch. Agents therefore hand-write
OpenMP C++ into `/tmp`, outside every reference-parity, determinism, and
certificate contract; one such run held an agent lane for three hours and
produced no output. The missing contract is a retained plan that owns the
enumeration: a declared finite-field domain, a per-element row generator, and
an incremental echelon with a rank threshold and early exit, returning
witnesses rather than a materialized batch. Require an executable reference
model over a bounded domain and exact parity with the existing
`modular_linear` RREF and nullspace semantics.

### FINUFFT plan-pool migration

The persistent plan API, fixed-strength and paired-sign type-3 cell
migrations, high-order interval-scout migration, fixed-R product canopy type-1
migration, and dyadic prime-shell type-1 migration are shipped. The
high-order interval scouts now retain their 27-row primary/dual moment stacks
across 250,000-target chunks on portable Linux; primary-only scans use the
fixed-strength plan, while Darwin retains simple calls. The fixed-R
second-derivative branch replaces six simple-interface calls per offset with
two retained three-transform plans and preallocated coefficient/output stacks.
The prime-shell branch retains one plan per shell across all offsets. All
end-to-end replays preserve maximizing offsets/indices and acceptance
decisions, with only last-bit binary64 variation far below the requested
FINUFFT tolerance.

Negative target verdict, 2026-07-28: do not pool the active-cell frontier or
VDC seam type-3 calls. Each uncached active-cell evaluation constructs a new
model, source set, and coefficient stack, then executes only one target; exact
repeats return the cached complete result. The VDC seam function likewise
builds each stack for one primary, dual, and mollifier call. No plan setup can
amortize. Also retain the interval scout's 32,768-source mollifier on the
simple interface: on the 27-transform, four-chunk portable fixture it improved
only from 0.475 s to 0.464 s median (1.02x), less than half a percent of the
combined route wall time.

Reopen type-3 plan-pool migration only when a current route performs at least
two target updates with unchanged source nodes, transform count, sign, and
coefficient stack. Preserve the route's `eps`, thread count, target ordering,
and complete complex output, and benchmark per platform before dispatch.

### Profile digest integration

Negative target verdict, 2026-07-28: the current triple-Gram small-host
certificate processed every unlabeled graph of orders seven and eight
(13,390 graphs) and called `update_collision_summary` 53,560 times, but made
zero calls to its tuple/repr `digest()` helper because no signature repeated.
A fail-fast replacement completed three times with byte-identical
certificates, fixing the perfect-removal speedup ceiling at exactly 1.00x.

The bounded successor survey found no eligible current route. The order-ten
triple-Gram verifier hashes its input artifact once, not profile rows.
First-shadow transversality uses BLAKE2b to define deterministic sketch
coefficients, so replacing it would change the mathematical map. The remaining
tuple/repr profile hashers are completed July 26 finite censuses whose equality
decisions use exact tuples.

Do not migrate these routes to fixed-width digests. Reopen only for a current
route where profile serialization and hashing accounts for at least 5% of
representative end-to-end wall time or material peak memory, preserving any
stored legacy digest mapping.

### Union graph integration

The direct nauty library boundary and loop-aware CSR triangle enumerator are
shipped. Historical representation-193 records report 3,722 order-13 colored
factor graphs canonicalized in 0.0111 seconds (335k graphs/s) and 371,915
allowed triples enumerated in 0.00558 seconds for the output pass.

Stale-target verdict, 2026-07-28: the source report named by those records,
`n13-r193-sharded-enumeration/n13-r193-merged.json`, is no longer present.
No current Union Python source imports NetworkX or calls `is_isomorphic`,
`GraphMatcher`, or `DiGraphMatcher`. The live `n13-eleven-exact-residue`
route consumes precomputed `colored_triangle_records` and does not construct
the historical factor graph. Therefore the former NetworkX bucket and Python
triangle-walk migration steps are not executable integration targets.

One current minimal-counterexample scout already uses shipped
`csr_common_neighbors` over `undirected_csr`. Reopen Union graph integration
only when a current route supplies a compatible source report and a measured
classifier hot path. At that point, preserve sparse CSR storage and add
induced filters only against that captured workload.

### Native union-family enumeration

The largest eventual Union-closed payoff is a specialized Boolean enumerator,
but it must preserve the current proof contract rather than merely imitate a
few Z3 runs.

Required contract:

1. Native union-closure, cardinality, empty-intersection, product-coverage,
   and coordinate-symmetry constraints over packed families.
2. Incremental model enumeration with exact blocking and deterministic shard
   partitions whose union is the full model space.
3. Replayable model records accepted by the existing direct family verifier.

The retained N=13 representation-193 census enumerates 48,099 orbit models;
the slowest shard takes 1,338.56 seconds. Use that shard as the first
end-to-end baseline.

### Union-family packed primitives

The one-word batch closure check is shipped for ground sizes through six. It
exhaustively matches the direct verifier through ground size four and matches
sampled hostile families at sizes five and six. The two retained route bodies
improve by 3.34x and 3.80x in median CPU time with complete output equality.

Remaining candidates are batched set-union products, empty-intersection
checks, coordinate-permutation canonicalization, frequency/support vectors,
and collision-shadow calculations. Accept only APIs with at least two live
route users and packed-bitset parity against the direct Python verifier.

Negative input-preparation verdict, 2026-07-28: do not specialize packed
family conversion for `range` or Python-list inputs under the current two
users. After native closure shipped, `_prepare_family_masks` costs 0.007763
seconds inside the 0.060456-second three-minimal-generators route (12.84%),
but only 0.000712 seconds inside the 0.049839-second singleton-channel route
(1.43%). Perfect removal caps the second route at 1.0145x. Reopen only when a
second current route attributes at least 5% of representative wall time or
material memory to the same stable input-preparation contract.

Negative statistics verdict, 2026-07-28: do not add packed frequency/support
or empty-intersection APIs for the two current closure users. On post-migration
Modal profiles, frequency vectors cost 0.052 seconds in the
three-minimal-generators census, but the singleton-channel scout spent only
0.015 seconds in its complete empty-intersection path and 0.009 seconds in
frequency iteration. Perfect removal of both singleton costs caps complete
process speedup at 1.085x, and only the census clears 5% individually. Reopen
when a second current route attributes at least 5% of representative wall time
or material memory to the same stable statistics contract.

Negative union-product verdict, 2026-07-28: the singleton-channel scout spent
0.026 seconds in 1,458 collision-preserving `product_profile` calls, but the
source-capacity scout's 682 exact trace-shape products cost only 0.001 seconds
inside a 1.870-second process. Perfectly deleting the latter caps speedup at
1.0005x. Do not add a packed product or collision-shadow API for these users;
reopen only when a second current route measures material product work and
requires the same output contract.

Negative coordinate-canonicalization verdict, 2026-07-28: the
three-generator equality census made eight `canonical_support` calls in a
0.007-second process, below profiler resolution individually. The retained
deficit-thirteen verifier made 259 `canonical_profile` calls costing 0.092
seconds inside a 15.303-second process; perfect removal caps speedup at
1.006x. Do not add packed family canonicalization for these records. Reopen
only when two current routes measure material coordinate-permutation work;
keep factor-graph canonicalization on the shipped nauty API.

### Proof-producing SAT

Add deterministic CNF generation for the specialized constraints, bounded
CaDiCaL/Kissat execution, complete model enumeration, and DRAT/LRAT artifact
replay. Solver speed without a checked exhaustive certificate is not a
theorem-grade exclusion.

### Exact sparse incidence construction

Graph reconstruction repeatedly builds deck, Kocay, shadow, and attachment
incidence matrices before exact rank or cokernel calculations.

The modular-rank half is shipped. On the retained order-eight Kocay moment
matrix, deterministic Markowitz ordering reaches rank 12,341 in 1.391 seconds
best versus the route-local 124.82-second baseline. It cuts elimination work
from 69,038,394 to 10,350,016 steps. Independent fields remain parallelized
through ForkUnion.

Next benchmark: stream sparse integer triplets directly from graph64 masks and
deduplicate rows by stable digest before CSR materialization. Fast-math should
own repeated modular rank lower bounds; FLINT should continue to own exact
integer rank, Smith form, and cokernel work when those stronger contracts are
required.

The rank kernel now removes every degree-one row/column cascade exactly before
numeric elimination. The grouped-incidence API is also shipped: order-17
scouting showed that generic peeling removes only 9 columns, while four-row
card blocks expose 8,392 columns through exact local duals in 0.122 seconds.
The 40,237-column residual reaches full rank in 1,140.15 seconds; reducer plus
rank is 3.25x faster than the retained 3,704.68-second LinBox baseline.
Markowitz ordering removes 28.7% of the prior residual elimination steps and
reduces maximum working width from 16,039 to 13,512.

Negative streaming verdict: on the retained order-17 profile, all measured
load/materialization/NPZ handoff costs total 0.457902 seconds inside the
1140.732496-second combined route, or 0.0401%. Even deleting that cost
perfectly caps speedup at 1.000402x. The in-memory residual CSR is 5.79 MiB,
only 0.82% of the minimum 708.43 MiB payload of the 92,855,069 retained rank
basis entries. Do not add a direct coloop-to-rank streaming API for this
profile. Reopen only when a captured route spends at least 5% of end-to-end
wall time in the handoff or the residual copy causes measured memory pressure.

Negative black-box verdict, 2026-07-28: do not add a LinBox-backed modular
rank backend. On the exact order-18 proof quotient, deterministic Fast Math
elimination returned rank 8,773 in 22.326 seconds median, while LinBox 1.7.0
scalar Wiedemann returned the same rank in 49.36 seconds median, 2.21x slower.
The retained order-17 comparison is 3.25x slower for LinBox, and order eight
is 235x slower. LinBox exposes block Wiedemann only through a deprecated,
untested solve path; its OpenMP block-rank example does not compile against
the retained 1.7.0 headers. Reopen only for a maintained deterministic
implementation with a stable rank-certificate contract and a captured matrix
where structured elimination exceeds memory or fails to complete.

Negative shared-construction verdict, 2026-07-28: do not add a generic
graph64-to-deduplicated-CSR incidence kernel for the current routes. The
order-eight low-overlap exporter already streams native induced-profile rows
directly to SMS and has no CSR or row-deduplication stage; its two native
profile kernels total only 0.286 seconds. On the exact order-17 attachment
build, cProfile measured 57.47 seconds total, of which 42.93 seconds generated
401,761 route-specific relabeled card graph6 records. Everything after
`build_sparse`, including CSR conversion, structural matching, NPZ
compression, and 853,712-line SMS output, totaled at most 2.89 seconds. The
P3-gluing builder also materializes COO/CSR, but its rows come from a distinct
oriented-gluing plus nauty-canonicalization contract. Reopen only when two
current routes expose the same row-generation contract before materialization
and that shared stage costs at least 5% end to end. The audit instead supports
a separate batched graph64-to-graph6 encoding target used by attachment and
gluing routes.

### Compact Chebyshev-Filon contraction

Status: experimental, correct, storage-effective, and SIMD-optimized; not yet
a production route replacement.

The retained L=0.0999 route currently stores 34.34 GiB of complex128 Filon
weights. Numerical rank scouting shows a high-rank head followed by a global
short endpoint expansion. The new exact-prefix/asymptotic-tail API reduces the
stored representation to about 16.15 MiB at cutoff 8,192 with ten endpoint
terms, and the full-shape synthetic contraction agreed with the cached-row
answer to `9.15e-14` relative error.

The scalar-tail implementation was 4.38x slower than the warmed synthetic
NumPy cached dots (`0.143 s` versus `0.0327 s`, two workers). Portable
two-lag SIMD is now shipped. On a captured real node-64 route autocorrelation,
one-thread best fell from `0.666 s` to `0.188 s` (3.54x), and eight workers
reach `0.0301 s`. The same-fixture warmed cached-row best is still faster at
`0.0217 s`, so the compact path remains parked rather than replacing the
34.34 GiB cache. Its value is unchanged from the pre-SIMD compact kernel and
differs from the full cached row by `1.42e-11` relatively.

A four-lag 256-bit vector batch regressed one-thread best to `0.302 s`, and a
four-lane compensated reduction regressed it to `0.339 s`; do not repeat
either experiment without a changed compiler or reduction model.

Negative retained-plan verdict, 2026-07-28: do not retain full float64 lag
geometry. A 136 MiB reciprocal array improved isolated ten-term weight
generation by only 1.059x, so even the impossible assumption that all
`0.0301 s` compact time is reciprocal work gives a `0.0284 s` floor, still
slower than the same-fixture `0.0217 s` cached-row best. Retaining reciprocals
plus phase costs 409 MiB and improves the isolated loop by 1.72x, but streams
1.000 GB per contraction including correlation, versus 857 MB for the cached
row. At the cached baseline's measured 39.6 GB/s, geometry traffic alone has
a `0.0253 s` floor before endpoint arithmetic and compensated reduction.
Reopen only with compressed precision that is proved sufficient and whose
complete route benchmark beats the cache.

Negative GPU precision verdict, 2026-07-28: a two-limb float-float complex
contraction is accurate on both Metal and CUDA, but it is not yet a viable
shared route. On the same 1,000,003-lag fixture, warm device-resident
contraction took `0.0509 s` on Metal and `0.00170 s` on CUDA, with relative
errors `1.02e-12` and `6.93e-14` against complex128. Host splitting plus
upload took `16.38 s` on Metal and `0.341 s` on CUDA, so a host-input API is
rejected. Ordinary complex64 vendor autocorrelation is also rejected: on the
same deterministic 131,073-source, 524,288-FFT fixture it changed the final
Filon contraction by `3.38e-7` on Metal and `2.83e-7` on CUDA. Reopen only
with a paired device-resident float-float autocorrelation and endpoint-tail
generator, and require its complete route wall time to beat the same-fixture
`0.0217 s` cached-row baseline. Merely splitting inputs around complex64 FFTs
does not meet this contract because each butterfly still rounds to float32.

Negative unfused radix-2 verdict, 2026-07-28: matching float-float butterfly
stages are numerically sound but globally streaming one kernel per stage is
not viable. On a 524,288-element representative stage, Metal and CUDA had
relative L2 errors `1.98e-15` and `1.97e-15`. Their warm bests were
`0.000507 s` and `0.0000192 s`, corresponding to 41.3 GB/s and 1.095 TB/s on
the reduced fixture. Scaling the measured 40 bytes per element per stage over
the route's two 26-stage FFT traversals projects to `3.38 s` on Metal and
`0.127 s` on CUDA, before spectral squaring, tail generation, or contraction.
Do not implement a shared global-memory radix-2 stage loop. Reopen only with
multiple stages fused per threadgroup/shared-memory residency and a measured
full-pass traffic model.

Negative fused-pass verdict, 2026-07-28: conventional threadgroup fusion also
cannot beat the cached row. Metal exposes 32 KiB of threadgroup memory, enough
for at most 2,048 in-place float-float complex values (11 stages); CUDA L4
exposes 101,376 opt-in bytes, enough for a 4,096-value block (12 stages).
Both therefore require at least three global passes for each `2^26` FFT.
Full-vector float-float copy benchmarks measured 167.3 GB/s on Metal and
242.4 GB/s on CUDA. Even fusing spectral square into the forward final pass
and contracting directly from the inverse final pass to avoid its full
output write leaves 5.5 copy-equivalent passes: traffic floors of `0.0706 s`
and `0.0487 s`, respectively, before butterfly arithmetic and tail-weight
generation. These are 3.26x and 2.25x over the `0.0217 s` cached-row budget.
Do not build a conventional full float-float FFT for this optimization.
Reopen only with a materially smaller proven representation, a non-FFT
correlation algorithm, or an explicit route decision to trade speed for
removing the 34.34 GiB artifact.

Negative representation-floor verdict, 2026-07-28: no conventional
full-vector GPU FFT representation can satisfy both the route's accuracy and
speed contracts. Under the optimistic 5.5-pass fusion floor and measured
full-vector bandwidth, Metal can afford at most 4.91 bytes per complex value
and CUDA at most 7.12 bytes before all arithmetic. Even ordinary 8-byte
complex64 has traffic floors of `0.0353 s` and `0.0244 s`, already above the
`0.0217 s` budget, while the paired precision scout showed complex64 changes
the final contraction by about `3e-7`. Twelve-byte encodings floor at
`0.0529 s` and `0.0365 s`; float-float's 16 bytes floor at `0.0706 s` and
`0.0487 s`. Do not retry compressed full-vector FFT formats on this target.
Only an algorithm that avoids the full `2^26` global transform, or an explicit
speed-for-storage route decision, survives.

Route decision packet, 2026-07-28: the retained primary power-Gram run performs
514 Filon contractions across 257 nodes and takes `1325.325 s`. Applying the
measured `0.008392 s` per-contraction compact penalty projects to `4.313 s`,
or a `0.325%` end-to-end runtime increase, while shrinking the kernel artifact
from 36,875,393,040 bytes to about 16.15 MiB (`2178x`). The QFL
signed-carrier route already uses compact Filon for 198 contractions; its
Filon time is `5.549 s` of a `22.543 s` run (`24.6%`) and it has no giant
cache dependency. PJM23's 1,473-node adjoint obligation is a separate
operation-graph problem and should not be counted as a consumer of this row
artifact. The route policy choice remains external to Fast Math; these are
the measured terms.

Next optimization:

1. Decide whether the primary route accepts a projected 4.31 s (`0.325%`)
   runtime increase to remove the 34.34 GiB cached row.
2. If not, seek a proven representation below two-float-per-component or a
   non-FFT correlation formulation before doing more GPU implementation.
3. Require complete accumulator parity against the existing cache and a
   measured wall-time win, unless the route explicitly accepts slower
   contraction to remove the 34.34 GiB artifact.
4. If the tail enters theorem-grade evidence, add a rigorous truncation
   remainder; the current binary64 comparison is scout evidence only.

Metal precision gate: MLX rejects float64 execution on the Apple GPU. Do not
route the existing complex128 contract through ordinary complex64 Metal. A
GPU promotion needs a validated precision-preserving Metal representation,
paired CUDA parity, and the same captured-real-input benchmark.

## Candidate targets

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

### Recurring hand-written kernels

Scanned: 2026-08-21.

`tools/duplication-scan` is what produced this table; rerun it over the current
scratch trees rather than trusting these counts a month from now. The
`fast-math-kernel` lane's demand probe reads the date above and the rows below,
so a rescan is what refills the queue and a merge is what drains it.

An alpha-normalized NCD scan of the fleet's scratch trees on 2026-08-21 found
190 hand-written C++ programs, none of which link this library, falling into ten
structural families. The five below are the ones whose contract is mathematical
rather than campaign-specific. Counts are files, against 1,394 scratch units in
total; 244 of the Python ones already import `fast_math`, while the native ones
import nothing.

| Candidate | Evidence | Contract |
| --- | --- | --- |
| Spans of encoded F_p points | 26 programs hand-roll mod-p elimination and 23 a modular inverse, over points held as encoded integers rather than as matrices | Rank, span membership, and quotient coordinates for batches of encoded `F_p^n` points, over the shipped RREF |
| Digit-tuple orbits under a permutation group | 24 programs transform base-m codes under generator arrays and deduplicate the images | Orbit representatives, orbit ids, and Burnside validation for `Z_m^k` tuples: the tuple analogue of the shipped packed-subset orbits |
| Higher-order WL | one six-file cluster implements 3-WL and 4-WL with hand-written signature hashing against the shipped 2-WL | Stable k-WL colorings with exact signature canonicalization for k = 3 and 4 |
| Colex subset ranking with orbit marking | 16 programs rank k-subsets combinadically to mark visited orbits during enumeration | Batched colex rank/unrank against a caller-owned visited bitmap |

A candidate leaves this table with a benchmark against the program it replaces,
not on its count alone.
