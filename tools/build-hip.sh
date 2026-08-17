#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${HIP_PLATFORM:=amd}"
: "${FAST_MATH_HIP_ARCH:=gfx1151}"

# Resolve the split NixOS ROCm packages rather than pretending /run/current-
# system/sw is a monolithic ROCm tree. Respect an already configured shell.
clr_library=$(readlink -f /run/current-system/sw/lib/libamdhip64.so)
clr_root=$(dirname "$(dirname "$clr_library")")
linker_script=$(readlink -f "$(command -v amdgcn-link)")
linker_wrapper=$(awk '/^exec / { print $2; exit }' "$linker_script")
toolchain_root=$(dirname "$(dirname "$linker_wrapper")")
device_lib_root=$(nix-store -q --references "$clr_root" | grep -- '-rocm-device-libs-' | head -1)
: "${HIP_CLANG_PATH:=$toolchain_root/bin}"
: "${HIP_PATH:=$clr_root}"
: "${ROCM_PATH:=$device_lib_root}"
: "${HIP_LIB_PATH:=$clr_root/lib}"

# Nixpkgs exposes ROCm's linker wrapper under amdgcn-link only after the
# system package is activated. Keep local names for hipcc's subprocesses.
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
  "$root/cpp/src/hip_packing.hip" \
  -o "$root/build/libfast_math_hip.so"
