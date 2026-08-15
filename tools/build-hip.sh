#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${HIP_PLATFORM:=amd}"
: "${HIP_CLANG_PATH:=/run/current-system/sw/bin}"
: "${HIP_PATH:=/run/current-system/sw}"
: "${ROCM_PATH:=/run/current-system/sw}"
: "${HIP_LIB_PATH:=/run/current-system/sw/lib}"
: "${FAST_MATH_HIP_ARCH:=gfx1151}"

# Nixpkgs exposes ROCm's linker wrapper under amdgcn-link only after the
# system package is activated. Keep a local fallback for older login shells.
tool_dir=$(mktemp -d)
trap 'rm -rf "$tool_dir"' EXIT
ln -s "$(command -v amdgcn-link)" "$tool_dir/amdgcn-link"
ln -s "$(command -v llvm-objcopy)" "$tool_dir/llvm-objcopy"
ln -s "$(command -v lld)" "$tool_dir/lld"
export PATH="$tool_dir:$PATH"
export HIP_PLATFORM HIP_CLANG_PATH HIP_PATH ROCM_PATH HIP_LIB_PATH

mkdir -p "$root/build"
hipcc --offload-arch="$FAST_MATH_HIP_ARCH" \
  -O3 -fPIC -shared \
  "$root/cpp/src/hip_affine.hip" \
  -o "$root/build/libfast_math_hip.so"
