# Benchmark results

Benchmark commands write JSON records into this directory. Generated records
are intentionally ignored because many benchmarks capture machine-specific
paths or depend on research fixtures that are not part of this repository.

The reproducible public baselines are summarized in the main README. When
reporting a performance change, include the command, source revision, hardware,
dimensions, repeat count, and complete result check with the pull request.

## Cayley-CI, July 29, 2026

The optimized complete 13-group atlas replay used 8 threads and 7 repeats on
the local Apple Silicon host. It reproduced 11,664 automorphism orbits and
9,606 graph fibers exactly. The median was `0.12926016957499087` seconds with
automorphism generators retained, a `18.980477590430294x` improvement over
the pre-optimization native stage baseline and `40.14429013199191x` over the
retained Python/NetworkX/`labelg` path.

The exact baseline samples, source revisions, hardware, and oracle dimensions
are tracked in `benchmarks/baselines/ci-atlas-2026-07-29.json`. The optimized
benchmark loads that record and writes both unrounded speedup factors into its
JSON output.

The exact 131,072-subset Q60 shard used 7 threads and 5 repeats. Fiber parity
with `labelg` was exact; median wall time fell from `0.10309333307668567` to
`0.04627570789307356` seconds, a `2.227806721290944x` speedup.

The retained 256-graph Q60 2-WL residual used 3 threads and 3 alternating
repeats. Refinement-only mode preserved every stable relation matrix, relation
count, and iteration count; median wall time fell from
`8.055843458045274` to `5.954400500049815` seconds, a
`1.3529226759231057x` speedup.

Commands:

```sh
../bin/compute --slots 8 --memory-mb 8192 --timeout-seconds 600 -- \
  python benchmarks/benchmark_ci_pipeline_stages.py \
    --threads 8 --repeats 7 \
    --output benchmarks/results/ci-pipeline-atlas-final.json
../bin/compute --slots 7 --memory-mb 8192 --timeout-seconds 600 -- \
  python benchmarks/benchmark_ci_q60_shard.py \
    --threads 7 --repeats 5 \
    --output benchmarks/results/ci-q60-shard-final.json
../bin/compute --slots 3 --memory-mb 4096 --timeout-seconds 900 -- \
  python benchmarks/benchmark_ci_wl2_residual.py \
    --limit 256 --repeats 3 --threads 3 \
    --output benchmarks/results/ci-wl2-residual-final.json
```
