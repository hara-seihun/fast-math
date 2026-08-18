# Fast Math architecture

Fast Math is a reusable mathematical compute library. Campaign scripts choose
mathematical objects and search domains; Fast Math owns stable representations,
validated algorithms, execution plans, and accelerated backends.

A kernel belongs here when its contract is stated in terms of mathematical data
rather than a campaign handle, one group, one valency, or one retained census.
A specialized mathematical domain is acceptable when it has a stable interface
and more than one use. Representative census shapes belong in tests and
benchmarks, not production control flow.

## Current audit

The production implementation is substantially more general than its benchmark
names suggest. The group and Cayley code accepts multiplication tables,
permutations, packed subsets, atom partitions, and relation matrices; it does
not encode the current CI cells or particular census parameters. At the
2026-08-18 audit, 80 retained Projects Research Python artifacts imported Fast
Math. The most reused operations were:

- `canonicalize_cayley_graphs`: 59 artifacts;
- `group_order`: 53;
- `inverse_closed_atoms`: 52;
- `atom_subsets_to_element_words`: 49;
- `induced_atom_action`: 47;
- `cayley_graphs`: 35;
- `deduplicate_subset_orbits`: 32.

The group/CI pipeline is therefore a real shared library rather than one encoded
census. The retained atlas, rank-six, and fixed-weight cases are executable
fixtures for generic contracts.

The package boundary is less mature than the kernels:

1. The root package still eagerly exposes more than one hundred names from core
   primitives and domain modules; only the Lambda compatibility exports are lazy.
2. The common native bridge is owned by `fast_math._native`; Lambda imports it
   as a domain consumer and can still be imported independently.
3. CPU sources are explicitly grouped into core, domain, and Lambda components
   while retaining one ordinary shared-library installation. HIP remains a
   separate optional build because NixOS supplies a split ROCm toolchain.
4. Packed subset actions, finite-group tables, Cayley construction, and fixed
   graph canonicalization now have retained plans alongside permutation-group
   and transform plans.
5. General primitives and specialized fused operations remain installable
   together. Examples of the latter are Lambda two-level replay, oriented
   square coverage, Chebyshev-Filon contraction, rooted-leaf attachment
   features, and contour-specific affine reductions.

## Layers

### Runtime

The runtime owns native-library discovery, error conversion, backend
capabilities, array ownership, worker pools, timing, and CPU/GPU device memory.
Neither `lambda_fast` nor another domain module should own this layer.

### General primitives

These are broadly reusable independent of a research topic:

- packed permutations and finite permutation groups;
- packed subset actions and orbit partitions;
- dense and CSR graph batches, canonical labeling adapters, and graph profiles;
- exact sparse finite-field rank and retained dense finite-field systems;
- deterministic reductions and fixed-width digests;
- retained affine and transform plans;
- ordered rigorous map/reduce.

### Mathematical domain modules

These compose primitives while retaining a domain-level mathematical contract:

- Cayley graphs, inverse atoms, derivative actions, and coherent
  configurations;
- union-closed families;
- oriented-square incidence;
- Chebyshev-Filon quadrature;
- Lambda certificate computations.

Domain specificity is not census specificity. For example,
`canonicalize_cayley_graphs(table, connections)` is a library operation;
`scan C2^3 x C9 through valency 35` is a campaign program.

### Campaign programs

Campaign code owns group presentations, parameter ranges, candidate generation,
stopping rules, evidence schemas, and theorem-specific transfer. These should
compose library plans rather than adding a production kernel for each census.

## Retained plans

Repeated work should move toward four composable plans:

- `PermutationActionPlan`: validates and retains an action, packed-mask lookup
  data, stabilizer information, and subset-orbit workspaces;
- `FiniteGroupPlan`: validates and retains multiplication and inverse tables;
- `CayleyGraphPlan`: retains a finite group and optional atom expansion, then
  constructs graph batches from caller-supplied connection sets;
- `GraphCanonicalPlan`: retains fixed graph structure or color partitions when
  only batch labels change.

These plans do not choose a census domain. They remove repeated validation,
allocation, conversion, and host/device transfer from any campaign that uses
the same mathematical object repeatedly.

## Backend contract

Every accelerated operation has one result contract and an executable reference
implementation. Backends are capabilities of a plan, not separate mathematical
APIs:

- `reference`: simple executable specification;
- `cpu`: native scalar/SIMD and bounded multicore execution;
- `gpu`: a retained CUDA, HIP, or Metal plan when the exact same contract is
  supported.

Automatic dispatch must use measured size thresholds and an available,
validated backend. It must not hide a broken installed backend. GPU pipelines
should keep intermediate arrays device-resident and return compact terminal
results; copying a full intermediate back to NumPy after every primitive
usually destroys the benefit.

Exact integer kernels retain exact integer semantics on the GPU. Floating-point
ranking kernels state their precision explicitly and do not replace rigorous or
higher-precision validation.

## GPU priorities

| Primitive | Suitability | Reason |
| --- | --- | --- |
| Packed subset permutation and canonical-minimum tests | Shipped on CPU and HIP | Retained exact byte-lookup plans, compact outputs, deterministic parity, and many CI consumers |
| Prime-field polynomial values/derivatives and small dense determinants | Shipped on CPU and HIP | Regular exact uint32 arithmetic, retained coefficients/workspaces, compact outputs, and SageMath/FLINT parity |
| Fixed-matrix prime-field solve, RREF, and nullspaces | Shipped on CPU and HIP | One canonical elimination produces reusable solution and obstruction operators; every answer carries a solution or left-null inconsistency witness |
| Packed CNF assignment verification | Shipped on CPU and HIP | Retained clauses, independent candidate certificates, compact failure witnesses, and sampled dispatch for early rejection |
| Batched relation histograms and selected WL refinement stages | Promising | Dense repeated exact counts; requires deterministic color compaction and a real route benchmark |
| Batched graph invariants and dense incidence predicates | Promising at large batch sizes | Regular work and compact reductions; CPU remains preferable for small batches |
| Cayley adjacency construction | Conditional | Simple exact parallel work, but useful only when the next stage can consume device-resident adjacency |
| CSR common-neighbor and triangle batches | Conditional | GPU-friendly for sufficiently large regular batches; current retained CPU shapes already finish in milliseconds |
| Affine populations and oriented-square incidence | Shipped | They amortize retained inputs and return compact metrics or packed incidence |
| Nauty canonical labeling, Schreier-Sims, and transporter search | Poor | Branch-heavy, irregular state with little coherent SIMD work |
| Deterministic sparse modular elimination | Poor in its current form | Data-dependent pivot chains and exact witness ordering |
| Arb interval computation | Poor | Arbitrary precision and strict ordered certificate semantics |

The first general GPU implementation is therefore a retained packed-subset
action plan, not a GPU implementation of one census. `PermutationActionPlan`
shares reference, native CPU, and HIP semantics; the native and HIP plans retain
byte-permutation tables and workspaces, return deterministic canonical masks,
and provide an early-exit canonical-minimum predicate. The benchmark covers a
cyclic degree-41 action and a 504-permutation degree-39 product action with exact
full-batch parity.

## Completed refactor and next work

Completed:

1. native discovery and generic C structures moved into the Fast Math runtime;
2. CPU sources grouped into core, mathematical-domain, and Lambda components;
3. retained action, finite-group, Cayley, and fixed-graph plans introduced;
4. backend capability reporting added across reference, native, Metal, CUDA,
   and HIP;
5. exact native and HIP packed-subset action kernels shipped with deterministic
   parity tests and representative benchmarks;
6. HIP architecture selection changed from a host constant to runtime hardware
   discovery at build time;
7. retained dense modular systems now expose canonical RREF, row transforms,
   both nullspaces, repeated solves, and exact inconsistency certificates on
   native CPU and HIP.

Next work should implement existing one-shot finite-group and graph functions in
terms of the retained plans where profiling shows repeated validation or
allocation, then promote another GPU primitive only after a complete consumer
route shows a material wall-time or scope gain. This keeps CPU and GPU
implementations behind shared validation, types, reference semantics, and tests
instead of creating parallel domain-specific code paths.
