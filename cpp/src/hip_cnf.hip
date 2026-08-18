#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace {

struct HipCnfPlan {
  std::uint64_t* clause_offsets;
  std::int32_t* literals;
  std::uint64_t* assignments;
  std::int64_t* first_unsatisfied;
  std::uint32_t variable_count;
  std::uint32_t word_count;
  std::size_t clause_count;
  std::size_t literal_count;
  std::size_t assignment_capacity;
};

int status(hipError_t error) {
  return error == hipSuccess ? 0 : static_cast<int>(error);
}

bool multiply_overflows(std::size_t left, std::size_t right) {
  return left != 0 && right > std::numeric_limits<std::size_t>::max() / left;
}

__global__ void cnf_evaluate_kernel(
    const std::uint64_t* __restrict__ clause_offsets,
    const std::int32_t* __restrict__ literals,
    std::size_t clause_count,
    const std::uint64_t* __restrict__ assignments,
    std::size_t assignment_count,
    std::uint32_t word_count,
    std::int64_t* __restrict__ first_unsatisfied) {
  const auto assignment =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (assignment >= assignment_count) return;
  const auto* words = assignments + assignment * word_count;
  std::int64_t first = -1;
  for (std::size_t clause = 0; clause < clause_count; ++clause) {
    bool clause_satisfied = false;
    for (auto offset = clause_offsets[clause];
         offset < clause_offsets[clause + 1];
         ++offset) {
      const auto literal = literals[offset];
      const auto magnitude = literal < 0
          ? -static_cast<std::int64_t>(literal)
          : static_cast<std::int64_t>(literal);
      const auto variable = static_cast<std::uint32_t>(magnitude - 1);
      const bool value =
          ((words[variable / 64] >> (variable % 64)) & 1U) != 0;
      if ((literal > 0 && value) || (literal < 0 && !value)) {
        clause_satisfied = true;
        break;
      }
    }
    if (!clause_satisfied) {
      first = static_cast<std::int64_t>(clause);
      break;
    }
  }
  first_unsatisfied[assignment] = first;
}

void release_batches(HipCnfPlan* plan) {
  (void)hipFree(plan->assignments);
  (void)hipFree(plan->first_unsatisfied);
  plan->assignments = nullptr;
  plan->first_unsatisfied = nullptr;
  plan->assignment_capacity = 0;
}

int reserve(HipCnfPlan* plan, std::size_t assignment_count) {
  if (assignment_count <= plan->assignment_capacity) return 0;
  release_batches(plan);
  if (multiply_overflows(assignment_count, plan->word_count) ||
      multiply_overflows(assignment_count * plan->word_count,
                         sizeof(std::uint64_t)) ||
      multiply_overflows(assignment_count, sizeof(std::int64_t))) return -2;
  if (status(hipMalloc(&plan->assignments,
                       assignment_count * plan->word_count *
                       sizeof(std::uint64_t))) != 0 ||
      status(hipMalloc(&plan->first_unsatisfied,
                       assignment_count * sizeof(std::int64_t))) != 0) {
    release_batches(plan);
    return -2;
  }
  plan->assignment_capacity = assignment_count;
  return 0;
}

}  // namespace

extern "C" int fast_math_hip_cnf_create(
    const std::uint64_t* clause_offsets,
    std::size_t clause_count,
    const std::int32_t* literals,
    std::size_t literal_count,
    std::uint32_t variable_count,
    void** result) {
  try {
    if (result != nullptr) *result = nullptr;
    if (result == nullptr || clause_offsets == nullptr || variable_count == 0 ||
        (literal_count != 0 && literals == nullptr)) return -1;
    if (clause_offsets[0] != 0 ||
        clause_offsets[clause_count] != literal_count) return -1;
    for (std::size_t clause = 0; clause < clause_count; ++clause) {
      if (clause_offsets[clause] > clause_offsets[clause + 1]) return -1;
    }
    for (std::size_t index = 0; index < literal_count; ++index) {
      const auto literal = static_cast<std::int64_t>(literals[index]);
      const auto variable = literal < 0 ? -literal : literal;
      if (variable == 0 || variable > variable_count) return -1;
    }
    auto* plan = new (std::nothrow) HipCnfPlan{
        nullptr, nullptr, nullptr, nullptr,
        variable_count, (variable_count + 63) / 64,
        clause_count, literal_count, 0};
    if (plan == nullptr) return -2;
    const auto offset_bytes = (clause_count + 1) * sizeof(std::uint64_t);
    const auto literal_bytes = literal_count * sizeof(std::int32_t);
    if (status(hipMalloc(&plan->clause_offsets, offset_bytes)) != 0 ||
        (literal_count != 0 &&
         status(hipMalloc(&plan->literals, literal_bytes)) != 0) ||
        status(hipMemcpy(plan->clause_offsets, clause_offsets, offset_bytes,
                         hipMemcpyHostToDevice)) != 0 ||
        (literal_count != 0 &&
         status(hipMemcpy(plan->literals, literals, literal_bytes,
                          hipMemcpyHostToDevice)) != 0)) {
      (void)hipFree(plan->clause_offsets);
      (void)hipFree(plan->literals);
      delete plan;
      return -2;
    }
    *result = plan;
    return 0;
  } catch (...) {
    return -2;
  }
}

extern "C" int fast_math_hip_cnf_destroy(void* opaque) {
  auto* plan = static_cast<HipCnfPlan*>(opaque);
  if (plan == nullptr) return 0;
  release_batches(plan);
  const auto offset_result = status(hipFree(plan->clause_offsets));
  const auto literal_result = status(hipFree(plan->literals));
  delete plan;
  return offset_result != 0 ? offset_result : literal_result;
}

extern "C" int fast_math_hip_cnf_evaluate(
    void* opaque,
    const std::uint64_t* assignments,
    std::size_t assignment_count,
    std::uint32_t word_count,
    std::int64_t* first_unsatisfied) {
  auto* plan = static_cast<HipCnfPlan*>(opaque);
  if (plan == nullptr || word_count != plan->word_count ||
      (assignment_count != 0 &&
       (assignments == nullptr || first_unsatisfied == nullptr))) return -1;
  if (assignment_count == 0) return 0;
  if (multiply_overflows(assignment_count, word_count)) return -2;
  const auto final_bits = plan->variable_count % 64;
  if (final_bits != 0) {
    const auto valid_mask = (std::uint64_t{1} << final_bits) - 1;
    for (std::size_t assignment = 0;
         assignment < assignment_count;
         ++assignment) {
      if ((assignments[assignment * word_count + word_count - 1] &
           ~valid_mask) != 0) return -1;
    }
  }
  int result = reserve(plan, assignment_count);
  if (result == 0) result = status(hipMemcpy(
      plan->assignments, assignments,
      assignment_count * word_count * sizeof(std::uint64_t),
      hipMemcpyHostToDevice));
  if (result == 0) {
    constexpr std::size_t threads = 256;
    const auto blocks = (assignment_count + threads - 1) / threads;
    hipLaunchKernelGGL(
        cnf_evaluate_kernel,
        dim3(blocks), dim3(threads), 0, 0,
        plan->clause_offsets, plan->literals, plan->clause_count,
        plan->assignments, assignment_count, plan->word_count,
        plan->first_unsatisfied);
    result = status(hipGetLastError());
  }
  if (result == 0) result = status(hipDeviceSynchronize());
  if (result == 0) result = status(hipMemcpy(
      first_unsatisfied, plan->first_unsatisfied,
      assignment_count * sizeof(std::int64_t), hipMemcpyDeviceToHost));
  return result;
}
