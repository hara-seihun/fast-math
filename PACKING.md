# Oriented-square coverage kernels

Fast Math provides the high-throughput geometric predicates used by square-packing certificate searches. The kernel classifies fixed support points against batches of oriented square poses; it does **not** claim that a finite pose sample is a proof.

## Model

A pose consists of center `(cx, cy)` and a unit direction `(ux, uy)`. For point `(x, y)`, write

```text
p = abs((x-cx)*ux + (y-cy)*uy)
q = abs((y-cy)*ux - (x-cx)*uy)
```

For half-extent `h` and numerical uncertainty `e`:

- definitely inside: `p <= h-e && q <= h-e`;
- possible: `p <= h+e && q <= h+e`;
- uncertain: possible but not definitely inside.

Callers choose the uncertainty from their error analysis. Exploratory floating-point output is not an interval proof by itself.

## Python API

`oriented_square_cover_words(points, centers, *, angles=None, directions=None, side_length=1, uncertainty=0, threads=0, backend="auto")` returns `SquareCoverBatch`. Incidences are packed in `uint64` arrays of shape `(ceil(point_count/64), pose_count)`. This is the appropriate API when a search needs distinct coverage states or LP rows.

`oriented_square_weighted_scores(points, weights, centers, ...)` returns `SquareScoreBatch` with one definite and possible weighted sum per pose. This avoids constructing or unpacking incidence matrices and is the appropriate adversary primitive in cutting-plane searches.

The native CPU backend uses AVX-512 over eight poses and Fast Math's persistent multicore scheduler. The HIP backend uses `SquareCoverHipPlan`, which keeps support points and reusable output storage on the GPU:

```python
from fast_math import SquareCoverHipPlan

plan = SquareCoverHipPlan(points)
inside, uncertain, seconds = plan.evaluate(poses, uncertainty=1e-12)
definite, possible, seconds = plan.weighted_scores(
    poses, weights, uncertainty=1e-12
)
plan.close()
```

`poses` has columns `(cx, cy, ux, uy)`. The plan is a context manager and automatically releases device storage.

## Validation

The NumPy implementation is an executable reference. Tests compare native scalar/AVX-512/multicore and HIP results against it, including boundary uncertainty, tail words, direction normalization, weighted sums, and persistent-plan reuse.

Run:

```bash
make test
make packing
```

`make packing` builds CPU and HIP libraries and writes the local benchmark artifact to `benchmarks/results/square-packing-local.json`.

On the Ryzen AI Max+ PRO 395 / Radeon 8060S host, one million random poses gave these representative warm-run rates:

| Support points | CPU packed incidence | HIP packed incidence | CPU direct score | HIP direct score |
|---:|---:|---:|---:|---:|
| 64 | 4.13 billion tests/s | 17.43 billion tests/s | 4.52 billion tests/s wall | 15.54 billion tests/s wall |
| 289 | 11.06 billion tests/s | 12.01 billion tests/s | 15.87 billion tests/s wall | 20.61 billion tests/s wall |

Wall rates include Python/C-ABI dispatch and host/device copies. Kernel timings and complete metadata are retained in the JSON artifact.

## Research use

The two intended stages are:

1. construct active LP constraints from packed coverage states;
2. scan millions of adversarial poses directly with current LP weights, add newly violated states, and repeat.

A sampled cutting-plane optimum remains a scout. Promotion to a certificate requires a separate universal argument that controls every center and angle, plus exact or interval verification of all final constants and inequalities.

### Current `s(12)` scout results

The scalar point-measure cutting plane reached total weight `12.4756` on a `0.025` support mesh after random adversarial refinement, so it does not prove the required strict bound below 12. A specialized bitset exchange walk replaced minute-scale generic hitting-set MILPs: it generated 3,493 sampled 14-point covers in 20 seconds, and the HIP adversary checked 100 diverse covers against 3.58 million poses in 4.81 seconds.

Colored-cover interaction alone also remains insufficient. Thirty-nine broad-scan survivors and all eight dihedral images gave 312 candidate unavoidable families on 248 points. After reducing 84,735 sampled coverage states to 3,621 inclusion-minimal states, the exact support-sharing relaxation still had matching number 14. Adding sampled physical square non-overlap was much stronger, but is not yet a universal certificate.

The active direction is therefore an equality-case refinement of Friedman's 12-point *almost unavoidable* configuration for `s(14)=4`:

```text
(1,1) (2,1) (3,1)
      (1.7,1.7) (2.3,1.7)
(1,2)             (3,2)
      (1.7,2.3) (2.3,2.3)
(1,3) (2,3) (3,3)
```

A box avoiding these points has its center in one of eight unit boundary regions. Combining this set with the canonical 14-point unavoidable set and requiring avoiders to receive extra additive score did not improve enough: sampled LP tradeoffs gave avoider-count bounds converging to 8, exactly the number of boundary regions. The next certificate must therefore classify those regions and retain pairwise square non-overlap rather than collapsing them to additive score.
