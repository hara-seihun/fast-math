#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${HIP_PLATFORM:=amd}"
if [[ -z "${FAST_MATH_HIP_ARCH:-}" ]]; then
  FAST_MATH_HIP_ARCH=$(rocminfo | awk '$1 == "Name:" && $2 ~ /^gfx[0-9]+$/ && found == "" { found = $2 } END { print found }')
  if [[ -z "$FAST_MATH_HIP_ARCH" ]]; then
    printf 'fast-math: no AMD GPU architecture found through rocminfo\n' >&2
    exit 1
  fi
fi

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

build_dir="$root/build/hip-objects"
mkdir -p "$build_dir"
sources=(
  hip_affine.hip
  hip_cnf.hip
  hip_modular.hip
  hip_packing.hip
  hip_subset_action.hip
)
fingerprint=$(printf '%s\n' \
  "$(readlink -f "$(command -v hipcc)")" \
  "$FAST_MATH_HIP_ARCH" \
  '-O3 -fPIC')
if [[ ! -f "$build_dir/config" ]] ||
   [[ "$(<"$build_dir/config")" != "$fingerprint" ]]; then
  rm -f "$build_dir"/*.o
  printf '%s' "$fingerprint" >"$build_dir/config"
fi

objects=()
for source_name in "${sources[@]}"; do
  source_path="$root/cpp/src/$source_name"
  object_path="$build_dir/${source_name%.hip}.o"
  objects+=("$object_path")
  if [[ ! -f "$object_path" || "$source_path" -nt "$object_path" ]]; then
    temporary="$object_path.tmp"
    rm -f "$temporary"
    hipcc --offload-arch="$FAST_MATH_HIP_ARCH" \
      -O3 -fPIC -c "$source_path" -o "$temporary"
    mv "$temporary" "$object_path"
  fi
done

library="$root/build/libfast_math_hip.so"
if [[ ! -f "$library" ]] ||
   find "${objects[@]}" -newer "$library" -print -quit | grep -q .; then
  temporary="$library.tmp"
  hipcc --offload-arch="$FAST_MATH_HIP_ARCH" \
    -shared "${objects[@]}" -o "$temporary"
  mv "$temporary" "$library"
fi
