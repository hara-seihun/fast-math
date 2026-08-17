#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

struct HipSquareCoverPlan {
  double* points;
  double* weights;
  double* poses;
  std::uint64_t* inside;
  std::uint64_t* uncertain;
  double* definite_scores;
  double* possible_scores;
  std::size_t point_count;
  std::size_t word_count;
  std::size_t pose_capacity;
};

int status(hipError_t error) {
  return error == hipSuccess ? 0 : static_cast<int>(error);
}

void free_buffers(HipSquareCoverPlan* plan) {
  (void)hipFree(plan->poses);
  (void)hipFree(plan->inside);
  (void)hipFree(plan->uncertain);
  (void)hipFree(plan->definite_scores);
  (void)hipFree(plan->possible_scores);
  plan->poses = nullptr;
  plan->inside = nullptr;
  plan->uncertain = nullptr;
  plan->definite_scores = nullptr;
  plan->possible_scores = nullptr;
  plan->pose_capacity = 0;
}

int reserve(HipSquareCoverPlan* plan, std::size_t pose_count) {
  if (pose_count <= plan->pose_capacity) return 0;
  free_buffers(plan);
  const auto pose_bytes = pose_count * 4 * sizeof(double);
  const auto output_bytes =
      pose_count * plan->word_count * sizeof(std::uint64_t);
  if (status(hipMalloc(&plan->poses, pose_bytes)) != 0 ||
      status(hipMalloc(&plan->inside, output_bytes)) != 0 ||
      status(hipMalloc(&plan->uncertain, output_bytes)) != 0 ||
      status(hipMalloc(
          &plan->definite_scores, pose_count * sizeof(double))) != 0 ||
      status(hipMalloc(
          &plan->possible_scores, pose_count * sizeof(double))) != 0) {
    free_buffers(plan);
    return -2;
  }
  plan->pose_capacity = pose_count;
  return 0;
}

__global__ void square_cover_kernel(
    const double* __restrict__ points,
    std::size_t point_count,
    const double* __restrict__ poses,
    std::size_t pose_count,
    double inner_extent,
    double outer_extent,
    std::uint64_t* __restrict__ inside,
    std::uint64_t* __restrict__ uncertain) {
  const auto pose =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto word = static_cast<std::size_t>(blockIdx.y);
  if (pose >= pose_count) return;
  const auto* values = poses + 4 * pose;
  const double center_x = values[0];
  const double center_y = values[1];
  const double direction_x = values[2];
  const double direction_y = values[3];
  const auto point_begin = word * 64;
  const auto point_end = min(point_count, point_begin + 64);
  std::uint64_t inside_word = 0;
  std::uint64_t uncertain_word = 0;
  for (auto point = point_begin; point < point_end; ++point) {
    const double dx = points[2 * point] - center_x;
    const double dy = points[2 * point + 1] - center_y;
    const double first = fabs(fma(dx, direction_x, dy * direction_y));
    const double second = fabs(fma(dy, direction_x, -dx * direction_y));
    const bool possible = first <= outer_extent && second <= outer_extent;
    const bool definite = first <= inner_extent && second <= inner_extent;
    const auto bit = std::uint64_t{1} << (point - point_begin);
    if (definite) inside_word |= bit;
    if (possible && !definite) uncertain_word |= bit;
  }
  const auto output = word * pose_count + pose;
  inside[output] = inside_word;
  uncertain[output] = uncertain_word;
}

__global__ void square_weighted_kernel(
    const double* __restrict__ points,
    const double* __restrict__ weights,
    std::size_t point_count,
    const double* __restrict__ poses,
    std::size_t pose_count,
    double inner_extent,
    double outer_extent,
    double* __restrict__ definite_scores,
    double* __restrict__ possible_scores) {
  const auto pose =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pose >= pose_count) return;
  const auto* values = poses + 4 * pose;
  const double center_x = values[0];
  const double center_y = values[1];
  const double direction_x = values[2];
  const double direction_y = values[3];
  double definite_score = 0.0;
  double possible_score = 0.0;
  for (std::size_t point = 0; point < point_count; ++point) {
    const double dx = points[2 * point] - center_x;
    const double dy = points[2 * point + 1] - center_y;
    const double first = fabs(fma(dx, direction_x, dy * direction_y));
    const double second = fabs(fma(dy, direction_x, -dx * direction_y));
    const bool possible = first <= outer_extent && second <= outer_extent;
    const bool definite = first <= inner_extent && second <= inner_extent;
    if (definite) definite_score += weights[point];
    if (possible) possible_score += weights[point];
  }
  definite_scores[pose] = definite_score;
  possible_scores[pose] = possible_score;
}

}  // namespace

extern "C" int fast_math_hip_square_cover_create(
    const double* points,
    std::size_t point_count,
    void** result) {
  if (points == nullptr || point_count == 0 || result == nullptr) return -1;
  auto* plan = new HipSquareCoverPlan{
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      point_count, (point_count + 63) / 64, 0};
  const auto point_bytes = point_count * 2 * sizeof(double);
  if (status(hipMalloc(&plan->points, point_bytes)) != 0 ||
      status(hipMalloc(&plan->weights, point_count * sizeof(double))) != 0) {
    (void)hipFree(plan->points);
    (void)hipFree(plan->weights);
    delete plan;
    return -2;
  }
  if (status(hipMemcpy(
          plan->points, points, point_bytes, hipMemcpyHostToDevice)) != 0) {
    (void)hipFree(plan->points);
    (void)hipFree(plan->weights);
    delete plan;
    return -3;
  }
  *result = plan;
  return 0;
}

extern "C" int fast_math_hip_square_cover_destroy(void* opaque) {
  auto* plan = static_cast<HipSquareCoverPlan*>(opaque);
  if (plan == nullptr) return 0;
  free_buffers(plan);
  const auto first = status(hipFree(plan->points));
  const auto second = status(hipFree(plan->weights));
  delete plan;
  return first != 0 ? first : second;
}

extern "C" int fast_math_hip_square_cover_evaluate(
    void* opaque,
    const double* poses,
    std::size_t pose_count,
    double half_extent,
    double uncertainty,
    std::uint64_t* inside,
    std::uint64_t* uncertain) {
  auto* plan = static_cast<HipSquareCoverPlan*>(opaque);
  if (plan == nullptr || poses == nullptr || pose_count == 0 ||
      half_extent <= 0.0 || uncertainty < 0.0 ||
      uncertainty >= half_extent || inside == nullptr || uncertain == nullptr) {
    return -1;
  }
  int result = reserve(plan, pose_count);
  const auto output_bytes =
      pose_count * plan->word_count * sizeof(std::uint64_t);
  if (result == 0) result = status(hipMemcpy(
      plan->poses, poses, pose_count * 4 * sizeof(double),
      hipMemcpyHostToDevice));
  if (result == 0) {
    constexpr std::size_t threads = 256;
    const auto blocks = (pose_count + threads - 1) / threads;
    hipLaunchKernelGGL(
        square_cover_kernel,
        dim3(blocks, plan->word_count), dim3(threads), 0, 0,
        plan->points, plan->point_count, plan->poses, pose_count,
        half_extent - uncertainty, half_extent + uncertainty,
        plan->inside, plan->uncertain);
    result = status(hipGetLastError());
  }
  if (result == 0) result = status(hipDeviceSynchronize());
  if (result == 0) result = status(hipMemcpy(
      inside, plan->inside, output_bytes, hipMemcpyDeviceToHost));
  if (result == 0) result = status(hipMemcpy(
      uncertain, plan->uncertain, output_bytes, hipMemcpyDeviceToHost));
  return result;
}

extern "C" int fast_math_hip_square_weighted_evaluate(
    void* opaque,
    const double* poses,
    std::size_t pose_count,
    const double* weights,
    double half_extent,
    double uncertainty,
    double* definite_scores,
    double* possible_scores) {
  auto* plan = static_cast<HipSquareCoverPlan*>(opaque);
  if (plan == nullptr || poses == nullptr || pose_count == 0 ||
      weights == nullptr || half_extent <= 0.0 || uncertainty < 0.0 ||
      uncertainty >= half_extent || definite_scores == nullptr ||
      possible_scores == nullptr) return -1;
  int result = reserve(plan, pose_count);
  if (result == 0) result = status(hipMemcpy(
      plan->poses, poses, pose_count * 4 * sizeof(double),
      hipMemcpyHostToDevice));
  if (result == 0) result = status(hipMemcpy(
      plan->weights, weights, plan->point_count * sizeof(double),
      hipMemcpyHostToDevice));
  if (result == 0) {
    constexpr std::size_t threads = 256;
    const auto blocks = (pose_count + threads - 1) / threads;
    hipLaunchKernelGGL(
        square_weighted_kernel, dim3(blocks), dim3(threads), 0, 0,
        plan->points, plan->weights, plan->point_count,
        plan->poses, pose_count,
        half_extent - uncertainty, half_extent + uncertainty,
        plan->definite_scores, plan->possible_scores);
    result = status(hipGetLastError());
  }
  if (result == 0) result = status(hipDeviceSynchronize());
  if (result == 0) result = status(hipMemcpy(
      definite_scores, plan->definite_scores, pose_count * sizeof(double),
      hipMemcpyDeviceToHost));
  if (result == 0) result = status(hipMemcpy(
      possible_scores, plan->possible_scores, pose_count * sizeof(double),
      hipMemcpyDeviceToHost));
  return result;
}
