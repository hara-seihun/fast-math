PYTHON ?= python3
SYSTEM_PKG_CONFIG_PATH ?= /run/current-system/sw/lib/pkgconfig
export PKG_CONFIG_PATH := $(SYSTEM_PKG_CONFIG_PATH)$(if $(PKG_CONFIG_PATH),:$(PKG_CONFIG_PATH))
COMPUTE := tools/run-compute.sh
LIBRARY = $(shell find build -maxdepth 3 -type f \( -name 'libfast_math.dylib' -o -name 'libfast_math.so' -o -name 'fast_math.dll' \) 2>/dev/null | head -1)

.PHONY: configure build test portable-test benchmark suite real-checkpoint moments inverse tune general graphs groups-ci union union-closure union-closure-routes digests sparse-rank sparse-rank-batch sparse-coloops arb finufft finufft-cells finufft-canopy finufft-prime-shell metal filon clean

configure:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

build: configure
	$(COMPUTE) --slots 4 --memory-mb 2048 --timeout-seconds 900 --label fast-math-build -- cmake --build build --parallel 4

test: build
	$(COMPUTE) --slots 5 --memory-mb 4096 --timeout-seconds 1800 --label fast-math-tests -- sh -c 'ctest --test-dir build --output-on-failure && env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) -m pytest -q'

portable-test:
	cmake -S . -B build-portable -DCMAKE_BUILD_TYPE=Release -DFAST_MATH_NATIVE_ARCH=OFF -DFAST_MATH_USE_FORKUNION=OFF -DFAST_MATH_USE_COMMONCRYPTO=OFF -DFAST_MATH_USE_NAUTY=OFF
	$(COMPUTE) --slots 4 --memory-mb 2048 --timeout-seconds 900 --label fast-math-portable-build -- cmake --build build-portable --parallel 4
	$(COMPUTE) --slots 5 --memory-mb 4096 --timeout-seconds 1800 --label fast-math-portable-tests -- sh -c 'ctest --test-dir build-portable --output-on-failure && env PYTHONPATH=python FAST_MATH_LIBRARY="$$(find build-portable -maxdepth 3 -type f \( -name "libfast_math.dylib" -o -name "libfast_math.so" -o -name "fast_math.dll" \) | head -1)" $(PYTHON) -m pytest -q'

benchmark: build
	$(COMPUTE) --slots 10 --memory-mb 4096 --timeout-seconds 1800 --label fast-math-benchmark -- env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_accumulate.py --case medium --backend both --threads 10 --repeats 3

suite: build
	$(COMPUTE) --slots 10 --memory-mb 8192 --timeout-seconds 3600 --label fast-math-suite -- env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/run_suite.py --output benchmarks/results/baseline-2026-07-26.json

real-checkpoint: build
	$(COMPUTE) --slots 8 --memory-mb 2048 --timeout-seconds 1800 --label fast-math-real-checkpoint -- env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_real_checkpoint.py --threads 8 --tile-size 8192 --output benchmarks/results/real-checkpoint-fused-2026-07-26.json

moments: build
	$(COMPUTE) --slots 8 --memory-mb 2048 --timeout-seconds 1800 --label fast-math-moments -- env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_power_moments.py --case medium --backend both --threads 8 --output benchmarks/results/power-moments-2026-07-26.json

inverse: build
	$(COMPUTE) --slots 1 --memory-mb 512 --timeout-seconds 1800 --label fast-math-inverse -- env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_inverse.py --case adaptive --backend both --output benchmarks/results/truncated-inverse-2026-07-26.json

tune: build
	$(COMPUTE) --slots 8 --memory-mb 2048 --timeout-seconds 1800 --label fast-math-tune -- env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_two_level_tuning.py --output benchmarks/results/two-level-tuning-2026-07-26.json

general: build
	$(COMPUTE) --slots 8 --memory-mb 2048 --timeout-seconds 1800 --label fast-math-general -- env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_general_kernels.py --output benchmarks/results/general-kernels-2026-07-26.json

graphs: build
	$(COMPUTE) --slots 7 --memory-mb 2048 --timeout-seconds 1800 --label fast-math-graphs -- env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_graph64.py --threads 7 --output benchmarks/results/graph64-2026-07-26.json

groups-ci: build
	$(COMPUTE) --slots 4 --memory-mb 4096 --timeout-seconds 900 --label fast-math-groups-ci -- env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_groups_ci.py --repeats 3 --kernel-repeats 5 --native-query-iterations 100 --threads 4 --output benchmarks/results/groups-ci-atlas-2026-07-29.json

union: build
	$(COMPUTE) --slots 1 --memory-mb 2048 --timeout-seconds 1800 --label fast-math-union -- env FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) ../problems/union-closed/scratch/minimal-counterexample--fast-math-packed-union-closure-audit/benchmark.py --repeats 11 --output benchmarks/results/union-closure-routes-2026-07-28.json

union-closure: build
	$(COMPUTE) --slots 1 --memory-mb 2048 --timeout-seconds 1800 --label fast-math-union-closure -- env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_union_closure.py --repeats 11 --output benchmarks/results/union-closure-native-2026-07-28.json

union-closure-routes: build
	$(COMPUTE) --slots 1 --memory-mb 2048 --timeout-seconds 1800 --label fast-math-union-closure-routes -- env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_union_closure_routes.py --repeats 11 --output benchmarks/results/union-closure-native-routes-2026-07-28.json

digests: build
	$(COMPUTE) --slots 7 --memory-mb 2048 --timeout-seconds 1800 --label fast-math-digests -- env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_profile_digests.py --threads 7 --output benchmarks/results/profile-digests-2026-07-26.json

sparse-rank: build
	$(COMPUTE) --slots 1 --memory-mb 2048 --timeout-seconds 1800 --label fast-math-sparse-rank -- $(PYTHON) benchmarks/benchmark_sparse_rank.py ../problems/graph-reconstruction/scratch/kocay-coverings--quadratic-cokernel-moment-rigidity/build/order8-low-overlap-moment.sms --target-rank 12341 --baseline-seconds 124.82 --output benchmarks/results/sparse-rank-order8-markowitz-2026-07-27.json

sparse-rank-batch: build
	$(COMPUTE) --slots 2 --memory-mb 1024 --timeout-seconds 1800 --label fast-math-sparse-rank-batch -- build/fast_math_sparse_rank_batch_benchmark ../problems/graph-reconstruction/scratch/kocay-coverings--quadratic-cokernel-moment-rigidity/build/order8-low-overlap-moment.sms 12341 2 1000000007 1000003

sparse-coloops: build
	$(COMPUTE) --slots 1 --memory-mb 2048 --timeout-seconds 1800 --label fast-math-sparse-coloops -- env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_sparse_coloops.py ../problems/graph-reconstruction/scratch/kocay-coverings--uniform-bounded-redundancy-trees/order17-sparse/attachment17.npz --row-block-size 4 --output benchmarks/results/sparse-coloops-order17-2026-07-26.json

arb: build
	$(COMPUTE) --slots 3 --memory-mb 4096 --timeout-seconds 1800 --label fast-math-arb -- env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_arb_cache.py --threads 3 --chunk-size 256 --output benchmarks/results/arb-cache-2026-07-26.json

finufft: build
	$(COMPUTE) --slots 6 --memory-mb 4096 --timeout-seconds 1800 --label fast-math-finufft -- env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_finufft_plans.py --threads 6 --executions 16 --output benchmarks/results/finufft-plans-2026-07-26.json

finufft-cells:
	$(COMPUTE) --slots 4 --memory-mb 4096 --timeout-seconds 1800 --label fast-math-finufft-cells -- env PYTHONPATH=python $(PYTHON) benchmarks/benchmark_finufft_cells.py --output benchmarks/results/finufft-type3-cells-2026-07-28.json

finufft-canopy:
	$(COMPUTE) --slots 1 --memory-mb 4096 --timeout-seconds 1800 --label fast-math-finufft-canopy -- env PYTHONPATH=python $(PYTHON) benchmarks/benchmark_finufft_canopy.py --sources 500000 --inverse-sources 32768 --modes 131072 --mesh 8 --repeats 5 --output benchmarks/results/finufft-canopy-migration-2026-07-28.json

finufft-prime-shell:
	$(COMPUTE) --slots 1 --memory-mb 2048 --timeout-seconds 1800 --label fast-math-finufft-prime-shell -- env PYTHONPATH=python $(PYTHON) benchmarks/benchmark_finufft_prime_shell.py --N 32767 --P 32768 --R 2000000 --mesh 8 --nufft-eps 1e-10 --repeats 5 --output benchmarks/results/finufft-prime-shell-migration-2026-07-28.json

metal:
	$(COMPUTE) --slots 2 --memory-mb 8192 --timeout-seconds 1800 --label fast-math-metal -- env VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=python $(PYTHON) benchmarks/benchmark_metal_affine.py --output benchmarks/results/metal-affine-2026-07-26.json

filon: build
	$(COMPUTE) --slots 2 --memory-mb 2048 --timeout-seconds 1800 --label fast-math-filon -- env VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_filon_inner_product.py ../problems/riemann-hypothesis/scratch/adjacent-unconditional--debruijn-newman-l00999-direct-radius-one-radial-cubic/build/primary-cheb-filon-kernels-P32768-eta32-d16-p4.c128 --degree 256 --node-index 64 --eta 32 --length 11989041.291992188 --output-count 17865985 --cutoff-terms 4096 12 --cutoff-terms 8192 10 --cutoff-terms 24576 8 --threads 1 2 --repeat 2 --output benchmarks/results/filon-inner-product-P32768-eta32-d16-p4-full-2026-07-27.json

clean:
	cmake -E remove_directory build
