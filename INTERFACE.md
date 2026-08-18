# Group and Cayley-CI interfaces

`fast_math.groups` and `fast_math.ci` provide the portable group-theory layer
used by Projects Research Cayley-CI workloads. Python owns validation and executable reference
models. C++20 implements the native kernels behind the public C ABI. Every
function accepts `backend="auto"`, `"native"`, or `"reference"` unless noted.

## Permutation groups

Permutations are contiguous integer arrays of shape `(count, degree)` with
`p[x]` equal to the image of `x`. Degrees 1 through 4096 are supported.
Composition is `left after right`:

```python
from fast_math.groups import PermutationGroup, schreier_sims

chain = schreier_sims(generators, backend="native")
with PermutationGroup(generators, backend="native") as group:
    assert group.order == chain.order
    orbits = group.orbits
    present = group.contains(candidates, threads=4)
```

`SchreierSimsChain` returns the deterministic base, orbit sizes, flattened
strong generators, and offsets delimiting the generators at each stabilizer
level. Group order is the exact Python integer product of the orbit sizes.
`PermutationGroup` retains one immutable native stabilizer chain and point-orbit
partition across membership batches. It owns its native allocation, supports a
context manager, and rejects use after `close()`.

`permutation_double_cosets(candidates, left, right)` partitions an explicit
finite candidate set under left and right generated actions. The candidate set
must be invariant under both actions; missing images are errors.

## Connection sets and Cayley graphs

`inverse_closed_atoms` partitions nonidentity elements into sets `{g,g^-1}`.
Subset bit `i` selects atom `i`; packed arrays use little-endian `uint64`
words. `enumerate_subset_orbits` enumerates a complete powerset through 62
atoms, while `deduplicate_subset_orbits` accepts any explicit invariant packed
collection through 512 atoms.

```python
from fast_math.ci import (
    atom_subsets_to_element_words,
    canonicalize_cayley_graphs,
    enumerate_subset_orbits,
    induced_atom_action,
    inverse_closed_atoms,
)

atoms = inverse_closed_atoms(inverse_indices, identity=0)
action = induced_atom_action(atoms, automorphism_generators)
classes = enumerate_subset_orbits(action, backend="native")
connections = atom_subsets_to_element_words(
    classes.representative_words,
    atoms,
    group_order=len(multiplication_table),
    threads=8,
    backend="native",
)
batch = canonicalize_cayley_graphs(
    multiplication_table,
    connections,
    inverse_indices=inverse_indices,
    threads=8,
    collect_automorphism_generators=False,
    graph_backend="native",
    canonical_backend="native",
)
```

`action_is_group=None` auto-detects whether the supplied action rows are the
complete finite group. A complete action is validated for unique rows,
identity, and exact generated-group order, then each subset orbit is generated
from one seed instead of traversing the action graph. Equality between the
row count and generated order is equivalent to closure and avoids a quadratic
all-pairs composition scan. Set `action_is_group=True` to require that contract
or `False` for an ordinary generator list. One-word masks
(through 64 atoms) use a byte-lookup bit-permutation kernel; multiword masks
retain the portable degree-512 route.

`enumerate_fixed_weight_subset_orbits(complete_action, weight)` takes every
member of a finite permutation group, validates identity, uniqueness, and
closure, and returns one representative and exact orbit size for each orbit on
the requested weight slice. It scans the combinatorial domain through a compact
combinadic bitset rather than materializing all masks. The default
`max_subsets` bound is explicit; degrees through 64 are supported. Native output
is checked against the Burnside orbit count, while the executable reference
backend forms the same complete-action image orbits directly.

The Cayley convention is the arc `(g, s*g)`, so multiplication-table row `s`
acts on every vertex `g`. The native constructor writes the packed adjacency
batch consumed directly by `canonicalize_colored_digraphs`; no graph objects
or shellouts occur in the hot path. Atom expansion and inverse-closure
validation are batched and do not construct Python sets per connection.

`canonicalize_colored_digraphs(..., threads=N)` schedules independent nauty
calls across worker-local workspaces when the linked nauty build exposes TLS;
non-TLS builds are forced to one worker. The common generator-producing path
is single-pass with deterministic graph-order compaction and an exact legacy
fallback if its bounded buffer is exceeded. Census callers that only need
canonical forms, group sizes, and orbit counts should set
`collect_automorphism_generators=False`; offsets are then all zero and the
generator array has shape `(0, vertex_count)`.

`generalized_dihedral_group(moduli)` constructs
`Dih(C_moduli[0] x ... x C_moduli[k])`. Rotations precede reflections.
`generalized_dihedral_automorphisms` lifts explicit automorphisms of the
abelian kernel and translations to the `Hol(A)` action. Callers may instead
pass any explicit automorphism generators to `induced_atom_action`; this is
required when a larger action is intended, such as the full `GL(3,2)` action
on `Dih(C2^2) = C2^3`.

## Derivative orbits

`derivative_group_orbits` implements the normalized R-0805 construction for a
bijection `f` fixing the identity. For every `x`, it forms the permutation

```text
Delta_f(x)(s) = f^-1(f(s*x) * f(x)^-1)
```

and returns all `Delta_f(x)` generators plus their point-orbit partition.
`derivative_orbit_partitions` applies the same exact reference/native contract
to a batch of bijections. Derivative-orbit inputs support group orders through
4096; this larger bound is specific to the quadratic-storage derivative kernel,
while packed Cayley-graph construction retains its order-512 bound.

## Coherent configurations

`coherent_configuration(initial_relations)` runs exact 2-WL refinement on a
square relation-color matrix through order 512. It returns the stable basic
relation matrix, basic-set sizes, and the dense intersection tensor. Its
indexing is

```text
intersection_numbers[i, j, k]
  = |{z : relation(x,z)=i and relation(z,y)=j}|
```

for any `(x,y)` in basic relation `k`. The implementation verifies that this
value is independent of the chosen pair. `graph_coherent_configuration`
constructs the initial diagonal/edge/nonedge coloring from a Boolean or packed
loopless graph.

Routes that need only stable 2-WL relations should call
`wl2_refinement` or `graph_wl2_refinement`. These return `WL2Refinement`
without constructing or verifying the intersection tensor; they are
deliberately not named coherent configurations.

Use `max_tensor_entries` to bound the cubic output allocation. The refinement
kernel supports graphs of order a few hundred; the test suite exercises order
257.

## Atlas and benchmark contract

`tests/test_ci_atlas.py` replays the complete retained 13-group atlas. It
compares byte-exact serialized fiber partitions containing orbit
representatives, CI multiplicities, and raw connection-set counts: 11,664
first-quotient orbits and 9,606 graph fibers.

The in-process directed-nauty canonical labeling can encode an isomorphic
graph6 string different from the retained undirected `labelg` convention.
Atlas parity therefore compares the invariant fiber partition, not spelling
of the canonical graph6 representative.

Run the self-contained retained benchmarks with:

```sh
make groups-ci
PYTHONPATH=python FAST_MATH_LIBRARY="$PWD/build/libfast_math.so" \
  python benchmarks/benchmark_ci_pipeline_stages.py \
    --threads 8 --repeats 5 \
    --output benchmarks/results/ci-pipeline-atlas-local.json
```

On the local Ryzen AI Max+ 395, the August 17, 2026 complete 13-group route
reproduces all 11,664 automorphism-orbit representatives and 9,606 graph fibers
in `0.09601842903066427` seconds median versus `5.0250206690398045` seconds for
the retained Python/NetworkX/`labelg` route, a `52.333918808805166x` end-to-end
speedup. A representative degree-18 GAP process route performing group order,
point orbits, and 17 membership tests takes `1.4666850980138406` seconds versus
`0.0011079729697667062` seconds for `PermutationGroup`, a
`1323.7553063434973x` speedup with identical output.

The canonical graph batch uses the system's TLS-enabled nauty build, allowing
independent exact calls to run through the retained worker pool. Permutation
stabilizer chains and nauty search remain CPU kernels: their branch-heavy exact
state is a poor GPU fit. Dense affine populations and oriented-square incidence
already dispatch to the Strix Halo HIP backend, where unified memory and compact
result transfers are beneficial. All native group kernels remain portable;
nauty is optional, and tests requiring it skip when unavailable.
