# Chebyshev-Filon Compression Handoff

Status: parked, SIMD-optimized prototype

Date: 2026-07-28

## Source workload

The retained L=0.0999 RH scout stores one half of a degree-256
Chebyshev-Lobatto Filon kernel:

```text
rows                 129
columns              17,865,985
dtype                complex128
stored bytes         36,875,393,040
stored GiB           34.34
```

The consumer contracts each row against positive and negative lags of one
autocorrelation. The current route reads the cached row with two NumPy dots.

## Retained findings

The first 1,024 columns are high rank: a rank-48 approximation still has
relative residual about `1.03e-7`, while rank 64 reaches binary64 noise.
The tail is different. Across columns sampled from 4,096 through 17,865,984,
raw rank six has relative residual `3.05e-15`, and normalized rank eight has
relative residual `5.00e-16`.

The short tail is explained by repeated integration by parts. For one
Lobatto cardinal polynomial `l_j`,

```text
integral_-1^1 l_j(x) exp(i t x) dx
  = sum_k (-1)^k / (i t)^(k+1)
      [exp(i t) l_j^(k)(1) - exp(-i t) l_j^(k)(-1)].
```

The finite degree makes the full expansion exact. Truncating it is accurate
only after an exact low-frequency prefix. On retained checkpoint columns:

```text
exact cutoff   endpoint terms   relative column error at cutoff
4,096          12               5.74e-13
8,192          10               3.63e-13
24,576          8               9.01e-14
```

These are binary64 sampled comparisons, not rigorous uniform remainder
bounds.

## Prototype

`fast_math.filon.filon_chebyshev_inner_product`:

- retains an exact complex prefix for one kernel row;
- derives endpoint derivatives from the Chebyshev-Lobatto differentiation
  matrix;
- generates the tail by the endpoint expansion without materializing it;
- combines positive and negative correlation lags algebraically;
- uses FMA for complex products;
- resets phase recurrence every 256 lags;
- uses fixed chunks, ForkUnion workers, compensated partial sums, and an
  ordered final reduction.

The 8,192/10 all-row representation is about 16.15 MiB including endpoint
coefficients, 2,178x smaller than the current cache.

Focused tests pass for endpoint polynomial reconstruction, native/reference
parity, conjugated rows, invalid inputs, and bitwise one-versus-two-worker
determinism.

Validation receipts:

```text
native build                         PASS
native Filon/API Python tests        9 passed
portable CTest                       4 passed
portable Filon/API Python tests      9 passed
```

## Performance verdict

Full retained shape, node 64, synthetic Hermitian correlation, M4 Pro:

```text
warmed NumPy cached-row dots          0.032689 s
native exact cached-row scan          0.041646 s
hybrid 8,192/10, one worker           0.302230 s
hybrid 8,192/10, two workers          0.143330 s
hybrid relative difference            9.15e-14
```

The original prototype is a decisive storage reduction but a 4.38x
contraction slowdown.

The July 28 continuation captured the real node-64 primary autocorrelation in
the route's padded 67,108,864-element FFT layout and added portable two-lag
SIMD to the endpoint tail. On that fixture:

```text
pre-SIMD compact, one worker          0.665956 s
SIMD compact, one worker              0.188277 s
speedup                               3.54x
SIMD compact, eight workers           0.030065 s
warmed cached-row NumPy               0.021674 s
compact / cached-row                  1.39x slower
relative compact/full-row difference  1.42e-11
```

The SIMD result is bit-identical to the pre-SIMD compact result. A four-lag
vector batch (`0.301712 s`) and four-lane compensated reduction (`0.339164 s`)
both regressed one-worker performance and were removed. The shared kernel
speedup is shipped, but the route remains parked because the same-fixture
cached row is still faster.

## Closed optimization branches

Retaining full float64 lag geometry does not close the gap. A reciprocal-only
plan improved isolated weight generation by 1.059x, which gives an impossible
best-case whole-kernel floor of `0.0284 s`. Retaining reciprocal, cosine, and
sine arrays improved the isolated loop by 1.72x but raises complete
contraction traffic to 1.000 GB; at the cached path's measured bandwidth its
traffic floor is `0.0253 s` before endpoint arithmetic and reduction.

A two-limb float-float GPU contraction is accurate when its inputs are already
device-resident:

```text
Metal, 1,000,003 lags       0.0509 s   relative error 1.02e-12
CUDA L4, 1,000,003 lags     0.00170 s  relative error 6.93e-14
```

It is not a viable host-input API: splitting plus upload took `16.38 s` on
Metal and `0.341 s` on CUDA. Ordinary complex64 autocorrelation also fails,
changing the final Filon contraction by `3.38e-7` on Metal and `2.83e-7` on
CUDA.

Building the needed float-float autocorrelation cannot meet the speed target
through a conventional full-vector FFT. Exact shared-memory limits force at
least three global passes per `2^26` transform. Full-vector copy measurements
give an optimistic 5.5-pass traffic floor of `0.0706 s` on Metal and
`0.0487 s` on CUDA before butterfly arithmetic. More generally, the
`0.0217 s` budget allows at most 4.91 bytes per global complex value on Metal
and 7.12 bytes on CUDA. Even 8-byte complex64 is too large by traffic and is
already numerically inadequate. Do not retry retained float64 geometry,
host-expanded GPU contraction, ordinary complex64 FFTs, unfused float-float
radix-2 stages, or conventional threadgroup-fused full-vector FFTs.

## Resume when

Resume with one of:

1. an explicit route decision that removing the 34.34 GiB cache is worth a
   slower subsecond contraction.
2. a proof that materially truncates the required lag domain, together with a
   complete route benchmark on the reduced domain.
3. a non-FFT correlation formulation whose measured end-to-end traffic and
   arithmetic can beat the cached-row baseline.
4. a representation that avoids the full `2^26` global transform entirely;
   merely compressing each global complex value is insufficient.

Before theorem-grade use, derive and check a uniform truncation remainder or
use the complete finite endpoint expansion.

## Evidence

- `benchmarks/benchmark_filon_kernel_rank.py`
- `benchmarks/benchmark_filon_asymptotic.py`
- `benchmarks/benchmark_filon_inner_product.py`
- `benchmarks/benchmark_filon_real_correlation.py`
- `benchmarks/results/filon-kernel-rank-P32768-eta32-d16-p4-2026-07-27.json`
- `benchmarks/results/filon-asymptotic-P32768-eta32-d16-p4-2026-07-27.json`
- `benchmarks/results/filon-asymptotic-cutoff-P32768-eta32-d16-p4-2026-07-27.json`
- `benchmarks/results/filon-inner-product-P32768-eta32-d16-p4-N1m-2026-07-27.json`
- `benchmarks/results/filon-inner-product-P32768-eta32-d16-p4-full-2026-07-27.json`
- `benchmarks/results/filon-real-node64-vector-pair-before-2026-07-28.json`
- `benchmarks/results/filon-real-node64-vector-pair-after-2026-07-28.json`
- `../problems/riemann-hypothesis/scratch/proof-system--fast-math-filon-retained-lag-plan/RESULT.md`
- `../problems/riemann-hypothesis/scratch/proof-system--fast-math-filon-gpu-precision-audit/RESULT.md`
- `../problems/riemann-hypothesis/scratch/proof-system--fast-math-filon-ff-radix2/RESULT.md`
- `../problems/riemann-hypothesis/scratch/proof-system--fast-math-filon-ff-fused-pass/RESULT.md`
- `../problems/riemann-hypothesis/scratch/proof-system--fast-math-filon-gpu-representation-floor/RESULT.md`
- `tests/test_filon.py`

## Promotion disposition

No new global-canon packet is warranted. The Filon evidence is numerical and
the integration-by-parts identity is classical. The exact block-coloop
construction developed in the same optimization pass is already represented
in the canon by D-0084, D-0117, and D-0125; duplicating it under a new ID would
violate the one-corpus rule. Markowitz ordering and the measured speedups are
software-engineering results, not new mathematical statements.
