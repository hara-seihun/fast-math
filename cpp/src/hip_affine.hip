#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

struct Complex32 {
  float real;
  float imag;
};

struct HipAffinePlan {
  Complex32* base;
  Complex32* basis;
  float* steps;
  Complex32* values;
  std::int64_t* windings;
  float* maximum_phases;
  float* edge_floors;
  std::size_t direction_count;
  std::size_t point_count;
  std::size_t population_capacity;
};

__device__ __forceinline__ Complex32 add_scaled(
    Complex32 value, Complex32 direction, float scale) {
  value.real += scale * direction.real;
  value.imag += scale * direction.imag;
  return value;
}

__global__ void evaluate_kernel(
    const Complex32* base, const Complex32* basis,
    const float* steps, std::size_t direction_count,
    std::size_t point_count, std::size_t population,
    Complex32* output) {
  const std::size_t linear =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = population * point_count;
  if (linear >= total) return;
  const std::size_t row = linear / point_count;
  const std::size_t point = linear % point_count;
  Complex32 value = base[point];
  for (std::size_t direction = 0; direction < direction_count; ++direction) {
    value = add_scaled(
        value, basis[direction * point_count + point],
        steps[row * direction_count + direction]);
  }
  output[linear] = value;
}

__global__ void metrics_kernel(
    const Complex32* values, std::size_t point_count,
    std::size_t population, std::size_t edge_start, std::size_t edge_stop,
    std::int64_t* windings, float* maximum_phases, float* edge_floors) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= population) return;
  const Complex32* current = values + row * point_count;
  float phase_sum = 0.0f;
  float phase_max = 0.0f;
  for (std::size_t point = 0; point + 1 < point_count; ++point) {
    const Complex32 left = current[point];
    const Complex32 right = current[point + 1];
    const float product_real =
        right.real * left.real + right.imag * left.imag;
    const float product_imag =
        right.imag * left.real - right.real * left.imag;
    const float phase = atan2f(product_imag, product_real);
    phase_sum += phase;
    phase_max = fmaxf(phase_max, fabsf(phase));
  }
  float floor = std::numeric_limits<float>::infinity();
  for (std::size_t point = edge_start; point < edge_stop; ++point) {
    const Complex32 value = current[point];
    floor = fminf(floor, hypotf(value.real, value.imag));
  }
  windings[row] = static_cast<std::int64_t>(nearbyintf(
      phase_sum / (2.0f * static_cast<float>(M_PI))));
  maximum_phases[row] = phase_max;
  edge_floors[row] = floor;
}

int status(hipError_t error) {
  return error == hipSuccess ? 0 : static_cast<int>(error);
}

void free_buffers(HipAffinePlan* plan) {
  hipFree(plan->steps);
  hipFree(plan->values);
  hipFree(plan->windings);
  hipFree(plan->maximum_phases);
  hipFree(plan->edge_floors);
  plan->steps = nullptr;
  plan->values = nullptr;
  plan->windings = nullptr;
  plan->maximum_phases = nullptr;
  plan->edge_floors = nullptr;
  plan->population_capacity = 0;
}

int reserve_buffers(HipAffinePlan* plan, std::size_t population) {
  if (population <= plan->population_capacity) return 0;
  free_buffers(plan);
  const std::size_t step_bytes =
      population * plan->direction_count * sizeof(float);
  const std::size_t value_bytes =
      population * plan->point_count * sizeof(Complex32);
  if (status(hipMalloc(&plan->steps, step_bytes)) != 0 ||
      status(hipMalloc(&plan->values, value_bytes)) != 0 ||
      status(hipMalloc(&plan->windings, population * sizeof(std::int64_t))) != 0 ||
      status(hipMalloc(&plan->maximum_phases, population * sizeof(float))) != 0 ||
      status(hipMalloc(&plan->edge_floors, population * sizeof(float))) != 0) {
    free_buffers(plan);
    return -2;
  }
  plan->population_capacity = population;
  return 0;
}

}  // namespace

extern "C" int fast_math_hip_create(
    const float* base, std::size_t point_count,
    const float* basis, std::size_t direction_count, void** result) {
  if (base == nullptr || basis == nullptr || result == nullptr ||
      point_count == 0 || direction_count == 0) {
    return -1;
  }
  auto* plan = new HipAffinePlan{
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      direction_count, point_count, 0};
  if (status(hipMalloc(&plan->base, point_count * sizeof(Complex32))) != 0 ||
      status(hipMalloc(&plan->basis,
                       direction_count * point_count * sizeof(Complex32))) != 0) {
    free_buffers(plan);
    hipFree(plan->base);
    hipFree(plan->basis);
    delete plan;
    return -2;
  }
  if (status(hipMemcpy(plan->base, base, point_count * sizeof(Complex32),
                       hipMemcpyHostToDevice)) != 0 ||
      status(hipMemcpy(plan->basis, basis,
                       direction_count * point_count * sizeof(Complex32),
                       hipMemcpyHostToDevice)) != 0) {
    free_buffers(plan);
    hipFree(plan->base);
    hipFree(plan->basis);
    delete plan;
    return -3;
  }
  *result = plan;
  return 0;
}

extern "C" int fast_math_hip_destroy(void* opaque) {
  auto* plan = static_cast<HipAffinePlan*>(opaque);
  if (plan == nullptr) return 0;
  free_buffers(plan);
  const int first = status(hipFree(plan->base));
  const int second = status(hipFree(plan->basis));
  delete plan;
  return first != 0 ? first : second;
}

extern "C" int fast_math_hip_evaluate(
    void* opaque, const float* steps, std::size_t population, float* output) {
  auto* plan = static_cast<HipAffinePlan*>(opaque);
  if (plan == nullptr || steps == nullptr || output == nullptr ||
      population == 0) return -1;
  const std::size_t step_bytes =
      population * plan->direction_count * sizeof(float);
  const std::size_t output_bytes =
      population * plan->point_count * sizeof(Complex32);
  int result = reserve_buffers(plan, population);
  if (result == 0) result = status(hipMemcpy(
      plan->steps, steps, step_bytes, hipMemcpyHostToDevice));
  if (result == 0) {
    const std::size_t threads = 256;
    const std::size_t blocks =
        (population * plan->point_count + threads - 1) / threads;
    hipLaunchKernelGGL(
        evaluate_kernel, dim3(blocks), dim3(threads), 0, 0,
        plan->base, plan->basis, plan->steps, plan->direction_count,
        plan->point_count, population, plan->values);
    result = status(hipGetLastError());
  }
  if (result == 0) result = status(hipDeviceSynchronize());
  if (result == 0) result = status(hipMemcpy(
      output, plan->values, output_bytes, hipMemcpyDeviceToHost));
  return result;
}

extern "C" int fast_math_hip_metrics(
    void* opaque, const float* steps, std::size_t population,
    std::size_t edge_start, std::size_t edge_stop,
    std::int64_t* windings, float* maximum_phases, float* edge_floors) {
  auto* plan = static_cast<HipAffinePlan*>(opaque);
  if (plan == nullptr || steps == nullptr || population == 0 ||
      windings == nullptr || maximum_phases == nullptr || edge_floors == nullptr ||
      edge_start >= edge_stop || edge_stop > plan->point_count ||
      plan->point_count < 2) return -1;
  const std::size_t step_bytes =
      population * plan->direction_count * sizeof(float);
  const std::size_t result_bytes = population * sizeof(float);
  int result = reserve_buffers(plan, population);
  if (result == 0) result = status(hipMemcpy(
      plan->steps, steps, step_bytes, hipMemcpyHostToDevice));
  if (result == 0) {
    const std::size_t threads = 256;
    const std::size_t blocks =
        (population * plan->point_count + threads - 1) / threads;
    hipLaunchKernelGGL(
        evaluate_kernel, dim3(blocks), dim3(threads), 0, 0,
        plan->base, plan->basis, plan->steps, plan->direction_count,
        plan->point_count, population, plan->values);
    result = status(hipGetLastError());
  }
  if (result == 0) {
    const std::size_t threads = 256;
    const std::size_t blocks = (population + threads - 1) / threads;
    hipLaunchKernelGGL(
        metrics_kernel, dim3(blocks), dim3(threads), 0, 0,
        plan->values, plan->point_count, population, edge_start, edge_stop,
        plan->windings, plan->maximum_phases, plan->edge_floors);
    result = status(hipGetLastError());
  }
  if (result == 0) result = status(hipDeviceSynchronize());
  if (result == 0) result = status(hipMemcpy(
      windings, plan->windings, population * sizeof(std::int64_t),
      hipMemcpyDeviceToHost));
  if (result == 0) result = status(hipMemcpy(
      maximum_phases, plan->maximum_phases, result_bytes,
      hipMemcpyDeviceToHost));
  if (result == 0) result = status(hipMemcpy(
      edge_floors, plan->edge_floors, result_bytes, hipMemcpyDeviceToHost));
  return result;
}
