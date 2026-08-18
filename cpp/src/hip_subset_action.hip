#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace {

struct HipSubsetActionPlan {
  std::uint64_t* byte_luts;
  std::uint64_t* masks;
  std::uint64_t* canonical;
  std::uint8_t* flags;
  std::size_t permutation_count;
  std::uint32_t degree;
  std::uint32_t byte_count;
  std::uint64_t valid_mask;
  std::size_t capacity;
};

int status(hipError_t error) {
  return error == hipSuccess ? 0 : static_cast<int>(error);
}

bool multiply_overflows(std::size_t left, std::size_t right) {
  return left != 0 && right > std::numeric_limits<std::size_t>::max() / left;
}

void release_batches(HipSubsetActionPlan* plan) {
  (void)hipFree(plan->masks);
  (void)hipFree(plan->canonical);
  (void)hipFree(plan->flags);
  plan->masks = nullptr;
  plan->canonical = nullptr;
  plan->flags = nullptr;
  plan->capacity = 0;
}

int reserve(HipSubsetActionPlan* plan, std::size_t count) {
  if (count <= plan->capacity) return 0;
  release_batches(plan);
  if (multiply_overflows(count, sizeof(std::uint64_t))) return -2;
  if (status(hipMalloc(&plan->masks, count * sizeof(std::uint64_t))) != 0 ||
      status(hipMalloc(&plan->canonical, count * sizeof(std::uint64_t))) != 0 ||
      status(hipMalloc(&plan->flags, count * sizeof(std::uint8_t))) != 0) {
    release_batches(plan);
    return -2;
  }
  plan->capacity = count;
  return 0;
}

__global__ void subset_masks_are_canonical_kernel(
    const std::uint64_t* __restrict__ masks,
    std::size_t mask_count,
    const std::uint64_t* __restrict__ byte_luts,
    std::size_t permutation_count,
    std::uint32_t byte_count,
    std::uint8_t* __restrict__ flags) {
  const auto index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= mask_count) return;
  const auto mask = masks[index];
  for (std::size_t row = 0; row < permutation_count; ++row) {
    const auto* tables = byte_luts + row * byte_count * 256;
    auto remaining = mask;
    std::uint64_t image = 0;
#pragma unroll
    for (std::uint32_t byte = 0; byte < 8; ++byte) {
      if (byte >= byte_count) break;
      image |= tables[byte * 256 + (remaining & 255)];
      remaining >>= 8;
    }
    if (image < mask) {
      flags[index] = 0;
      return;
    }
  }
  flags[index] = 1;
}

__global__ void canonicalize_subset_masks_kernel(
    const std::uint64_t* __restrict__ masks,
    std::size_t mask_count,
    const std::uint64_t* __restrict__ byte_luts,
    std::size_t permutation_count,
    std::uint32_t byte_count,
    std::uint64_t* __restrict__ canonical,
    std::uint8_t* __restrict__ flags) {
  const auto index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= mask_count) return;
  const auto mask = masks[index];
  auto best = mask;
  for (std::size_t row = 0; row < permutation_count; ++row) {
    const auto* tables = byte_luts + row * byte_count * 256;
    auto remaining = mask;
    std::uint64_t image = 0;
#pragma unroll
    for (std::uint32_t byte = 0; byte < 8; ++byte) {
      if (byte >= byte_count) break;
      image |= tables[byte * 256 + (remaining & 255)];
      remaining >>= 8;
    }
    best = min(best, image);
  }
  canonical[index] = best;
  flags[index] = best == mask ? 1 : 0;
}

}  // namespace

extern "C" int fast_math_hip_subset_action_create(
    const std::uint32_t* permutations,
    std::size_t permutation_count,
    std::uint32_t degree,
    void** result) {
  try {
    if (result != nullptr) *result = nullptr;
    if (permutations == nullptr || permutation_count == 0 || result == nullptr ||
        degree == 0 || degree > 64) return -1;
    const auto byte_count = (degree + 7) / 8;
    if (multiply_overflows(permutation_count, byte_count) ||
        multiply_overflows(permutation_count * byte_count, 256) ||
        multiply_overflows(permutation_count * byte_count * 256,
                           sizeof(std::uint64_t))) return -2;
    std::vector<std::uint8_t> seen(degree);
    std::vector<std::uint64_t> tables(
        permutation_count * byte_count * 256, 0);
    for (std::size_t row = 0; row < permutation_count; ++row) {
      std::fill(seen.begin(), seen.end(), 0);
      for (std::uint32_t point = 0; point < degree; ++point) {
        const auto image = permutations[row * degree + point];
        if (image >= degree || seen[image] != 0) return -1;
        seen[image] = 1;
      }
      for (std::uint32_t byte = 0; byte < byte_count; ++byte) {
        auto* table = tables.data() + (row * byte_count + byte) * 256;
        for (std::uint32_t value = 1; value < 256; ++value) {
          const auto low = value & (~value + 1);
          const auto bit = static_cast<std::uint32_t>(__builtin_ctz(low));
          const auto point = byte * 8 + bit;
          table[value] = table[value ^ low];
          if (point < degree) {
            table[value] |= std::uint64_t{1}
                << permutations[row * degree + point];
          }
        }
      }
    }
    auto* plan = new (std::nothrow) HipSubsetActionPlan{
        nullptr, nullptr, nullptr, nullptr,
        permutation_count, degree, byte_count,
        degree == 64 ? std::numeric_limits<std::uint64_t>::max()
                     : (std::uint64_t{1} << degree) - 1,
        0};
    if (plan == nullptr) return -2;
    const auto table_bytes = tables.size() * sizeof(std::uint64_t);
    if (status(hipMalloc(&plan->byte_luts, table_bytes)) != 0 ||
        status(hipMemcpy(plan->byte_luts, tables.data(), table_bytes,
                         hipMemcpyHostToDevice)) != 0) {
      (void)hipFree(plan->byte_luts);
      delete plan;
      return -2;
    }
    *result = plan;
    return 0;
  } catch (...) {
    return -2;
  }
}

extern "C" int fast_math_hip_subset_action_destroy(void* opaque) {
  auto* plan = static_cast<HipSubsetActionPlan*>(opaque);
  if (plan == nullptr) return 0;
  release_batches(plan);
  const auto result = status(hipFree(plan->byte_luts));
  delete plan;
  return result;
}

extern "C" int fast_math_hip_subset_action_is_canonical(
    void* opaque,
    const std::uint64_t* masks,
    std::size_t mask_count,
    std::uint8_t* flags) {
  auto* plan = static_cast<HipSubsetActionPlan*>(opaque);
  if (plan == nullptr || (mask_count != 0 &&
      (masks == nullptr || flags == nullptr))) return -1;
  if (mask_count == 0) return 0;
  for (std::size_t index = 0; index < mask_count; ++index) {
    if ((masks[index] & ~plan->valid_mask) != 0) return -1;
  }
  int result = reserve(plan, mask_count);
  if (result == 0) result = status(hipMemcpy(
      plan->masks, masks, mask_count * sizeof(std::uint64_t),
      hipMemcpyHostToDevice));
  if (result == 0) {
    constexpr std::size_t threads = 256;
    const auto blocks = (mask_count + threads - 1) / threads;
    hipLaunchKernelGGL(
        subset_masks_are_canonical_kernel,
        dim3(blocks), dim3(threads), 0, 0,
        plan->masks, mask_count, plan->byte_luts,
        plan->permutation_count, plan->byte_count, plan->flags);
    result = status(hipGetLastError());
  }
  if (result == 0) result = status(hipDeviceSynchronize());
  if (result == 0) result = status(hipMemcpy(
      flags, plan->flags, mask_count * sizeof(std::uint8_t),
      hipMemcpyDeviceToHost));
  return result;
}

extern "C" int fast_math_hip_subset_action_canonicalize(
    void* opaque,
    const std::uint64_t* masks,
    std::size_t mask_count,
    std::uint64_t* canonical,
    std::uint8_t* flags) {
  auto* plan = static_cast<HipSubsetActionPlan*>(opaque);
  if (plan == nullptr || (mask_count != 0 &&
      (masks == nullptr || canonical == nullptr || flags == nullptr))) return -1;
  if (mask_count == 0) return 0;
  for (std::size_t index = 0; index < mask_count; ++index) {
    if ((masks[index] & ~plan->valid_mask) != 0) return -1;
  }
  int result = reserve(plan, mask_count);
  if (result == 0) result = status(hipMemcpy(
      plan->masks, masks, mask_count * sizeof(std::uint64_t),
      hipMemcpyHostToDevice));
  if (result == 0) {
    constexpr std::size_t threads = 256;
    const auto blocks = (mask_count + threads - 1) / threads;
    hipLaunchKernelGGL(
        canonicalize_subset_masks_kernel,
        dim3(blocks), dim3(threads), 0, 0,
        plan->masks, mask_count, plan->byte_luts,
        plan->permutation_count, plan->byte_count,
        plan->canonical, plan->flags);
    result = status(hipGetLastError());
  }
  if (result == 0) result = status(hipDeviceSynchronize());
  if (result == 0) result = status(hipMemcpy(
      canonical, plan->canonical, mask_count * sizeof(std::uint64_t),
      hipMemcpyDeviceToHost));
  if (result == 0) result = status(hipMemcpy(
      flags, plan->flags, mask_count * sizeof(std::uint8_t),
      hipMemcpyDeviceToHost));
  return result;
}
