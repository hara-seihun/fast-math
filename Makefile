PYTHON ?= python3
SYSTEM_PKG_CONFIG_PATH ?= /run/current-system/sw/lib/pkgconfig
export PKG_CONFIG_PATH := $(SYSTEM_PKG_CONFIG_PATH)$(if $(PKG_CONFIG_PATH),:$(PKG_CONFIG_PATH))
LIBRARY = $(shell find build -maxdepth 3 -type f \( -name 'libfast_math.dylib' -o -name 'libfast_math.so' -o -name 'fast_math.dll' \) 2>/dev/null | head -1)

.PHONY: configure build hip hip-test hip-benchmark packing mask-lut subset-actions modular-batches modular-linear cnf-verification adaptive-area test portable-test benchmark suite real-checkpoint moments inverse tune general graphs groups-ci derivative-rank6 ci-weight-orbits union-closure union-closure-routes base-p digests arb finufft finufft-cells finufft-canopy finufft-prime-shell metal clean

configure:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

build: configure
	cmake --build build --parallel 4

hip:
	tools/build-hip.sh

hip-test: build hip
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" FAST_MATH_HIP_LIBRARY="$(CURDIR)/build/libfast_math_hip.so" $(PYTHON) -m pytest -q tests/test_hip.py tests/test_actions.py tests/test_modular.py tests/test_modular_linear.py tests/test_cnf.py

hip-benchmark: hip
	env PYTHONPATH=python FAST_MATH_HIP_LIBRARY="$(CURDIR)/build/libfast_math_hip.so" $(PYTHON) benchmarks/benchmark_hip_affine.py --output benchmarks/results/hip-affine-local.json

packing: build hip
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" FAST_MATH_HIP_LIBRARY="$(CURDIR)/build/libfast_math_hip.so" $(PYTHON) benchmarks/benchmark_square_packing.py --threads 8 --output benchmarks/results/square-packing-local.json

mask-lut:
	env PYTHONPATH=python $(PYTHON) benchmarks/benchmark_u64_mask_lut.py

subset-actions: build hip
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" FAST_MATH_HIP_LIBRARY="$(CURDIR)/build/libfast_math_hip.so" $(PYTHON) benchmarks/benchmark_subset_actions.py --threads 8 --output benchmarks/results/subset-actions-local.json

modular-batches: build hip
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" FAST_MATH_HIP_LIBRARY="$(CURDIR)/build/libfast_math_hip.so" $(PYTHON) benchmarks/benchmark_modular_batches.py --output benchmarks/results/modular-batches-local.json

modular-linear: build hip
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" FAST_MATH_HIP_LIBRARY="$(CURDIR)/build/libfast_math_hip.so" $(PYTHON) benchmarks/benchmark_modular_linear.py --output benchmarks/results/modular-linear-local.json

cnf-verification: build hip
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" FAST_MATH_HIP_LIBRARY="$(CURDIR)/build/libfast_math_hip.so" $(PYTHON) benchmarks/benchmark_cnf_verification.py --output benchmarks/results/cnf-verification-local.json

test: build
	sh -c 'ctest --test-dir build --output-on-failure && env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) -m pytest -q -n 10'

portable-test:
	cmake -S . -B build-portable -DCMAKE_BUILD_TYPE=Release -DFAST_MATH_NATIVE_ARCH=OFF -DFAST_MATH_USE_FORKUNION=OFF -DFAST_MATH_USE_COMMONCRYPTO=OFF -DFAST_MATH_USE_NAUTY=OFF
	cmake --build build-portable --parallel 4
	sh -c 'ctest --test-dir build-portable --output-on-failure && env PYTHONPATH=python FAST_MATH_LIBRARY="$$(find build-portable -maxdepth 3 -type f \( -name "libfast_math.dylib" -o -name "libfast_math.so" -o -name "fast_math.dll" \) | head -1)" $(PYTHON) -m pytest -q -n 10'

benchmark: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_accumulate.py --case medium --backend both --threads 10 --repeats 3

suite: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/run_suite.py --output benchmarks/results/baseline-2026-07-26.json

real-checkpoint: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_real_checkpoint.py --threads 8 --tile-size 8192 --output benchmarks/results/real-checkpoint-fused-2026-07-26.json

moments: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_power_moments.py --case medium --backend both --threads 8 --output benchmarks/results/power-moments-2026-07-26.json

inverse: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_inverse.py --case adaptive --backend both --output benchmarks/results/truncated-inverse-2026-07-26.json

tune: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_two_level_tuning.py --output benchmarks/results/two-level-tuning-2026-07-26.json

general: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_general_kernels.py --output benchmarks/results/general-kernels-2026-07-26.json

graphs: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_graph64.py --threads 7 --output benchmarks/results/graph64-2026-07-26.json

groups-ci: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_groups_ci.py --repeats 5 --kernel-repeats 3 --native-query-iterations 100 --threads 8 --output benchmarks/results/groups-gap-local.json

derivative-rank6: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_derivative_rank6.py --output benchmarks/results/derivative-rank6-local.json

ci-weight-orbits: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_ci_weight_orbits.py --repeats 3 --output benchmarks/results/ci-weight-orbits.json

adaptive-area: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_adaptive_area.py --threads 8 --output benchmarks/results/adaptive-area-local.json

union-closure: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_union_closure.py --repeats 11 --output benchmarks/results/union-closure-native-2026-07-28.json

union-closure-routes: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_union_closure_routes.py --repeats 11 --output benchmarks/results/union-closure-native-routes-2026-07-28.json

base-p: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_base_p.py --repeats 7 --output benchmarks/results/base-p-local.json

digests: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_profile_digests.py --threads 7 --output benchmarks/results/profile-digests-2026-07-26.json

arb: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_arb_cache.py --threads 3 --chunk-size 256 --output benchmarks/results/arb-cache-2026-07-26.json

finufft: build
	env PYTHONPATH=python FAST_MATH_LIBRARY="$(LIBRARY)" $(PYTHON) benchmarks/benchmark_finufft_plans.py --threads 6 --executions 16 --output benchmarks/results/finufft-plans-2026-07-26.json

finufft-cells:
	env PYTHONPATH=python $(PYTHON) benchmarks/benchmark_finufft_cells.py --output benchmarks/results/finufft-type3-cells-2026-07-28.json

finufft-canopy:
	env PYTHONPATH=python $(PYTHON) benchmarks/benchmark_finufft_canopy.py --sources 500000 --inverse-sources 32768 --modes 131072 --mesh 8 --repeats 5 --output benchmarks/results/finufft-canopy-migration-2026-07-28.json

finufft-prime-shell:
	env PYTHONPATH=python $(PYTHON) benchmarks/benchmark_finufft_prime_shell.py --N 32767 --P 32768 --R 2000000 --mesh 8 --nufft-eps 1e-10 --repeats 5 --output benchmarks/results/finufft-prime-shell-migration-2026-07-28.json

metal:
	env VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=python $(PYTHON) benchmarks/benchmark_metal_affine.py --output benchmarks/results/metal-affine-2026-07-26.json

clean:
	cmake -E remove_directory build
