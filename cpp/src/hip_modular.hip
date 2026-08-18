#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace {

struct HipPolynomialPlan {
  std::uint32_t* coefficients;
  std::uint32_t* points;
  std::uint32_t* values;
  std::uint32_t* derivatives;
  std::size_t polynomial_count;
  std::size_t coefficient_count;
  std::uint32_t prime;
  std::uint64_t reciprocal;
  std::size_t point_capacity;
};

struct HipDeterminantPlan {
  std::uint32_t* matrices;
  std::uint32_t* determinants;
  std::uint32_t order;
  std::uint32_t prime;
  std::uint64_t reciprocal;
  std::size_t matrix_capacity;
};

struct HipLinearSystemPlan {
  std::uint32_t* solution_operator;
  std::uint32_t* pivot_columns;
  std::uint32_t* left_nullspace;
  std::uint32_t* right_hand_sides;
  std::uint32_t* solutions;
  std::uint64_t* inconsistency_rows;
  std::size_t row_count;
  std::size_t column_count;
  std::size_t rank;
  std::size_t obstruction_count;
  std::uint32_t prime;
  std::uint64_t reciprocal;
  std::uint64_t reduction_terms;
  std::size_t batch_capacity;
};

int status(hipError_t error) {
  return error == hipSuccess ? 0 : static_cast<int>(error);
}

bool multiply_overflows(std::size_t left, std::size_t right) {
  return left != 0 && right > std::numeric_limits<std::size_t>::max() / left;
}

std::uint64_t reciprocal_for(std::uint32_t prime) {
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  return maximum / prime + (maximum % prime == prime - 1 ? 1 : 0);
}

bool is_prime(std::uint32_t value) {
  if (value < 2) return false;
  if ((value & 1U) == 0) return value == 2;
  for (std::uint32_t divisor = 3;
       divisor <= value / divisor;
       divisor += 2) {
    if (value % divisor == 0) return false;
  }
  return true;
}

__device__ std::uint32_t add_mod(
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t prime) {
  const auto sum = static_cast<std::uint64_t>(left) + right;
  return static_cast<std::uint32_t>(sum >= prime ? sum - prime : sum);
}

__device__ std::uint32_t subtract_mod(
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t prime) {
  return left >= right ? left - right : prime - (right - left);
}

__device__ std::uint32_t multiply_mod(
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t prime,
    std::uint64_t reciprocal) {
  const auto product = static_cast<std::uint64_t>(left) * right;
  const auto quotient = __umul64hi(product, reciprocal);
  auto remainder = product - quotient * prime;
  if (remainder >= prime) remainder -= prime;
  return static_cast<std::uint32_t>(remainder);
}

__device__ std::uint32_t power_mod(
    std::uint32_t base,
    std::uint32_t exponent,
    std::uint32_t prime,
    std::uint64_t reciprocal) {
  std::uint32_t result = 1 % prime;
  while (exponent != 0) {
    if ((exponent & 1U) != 0) {
      result = multiply_mod(result, base, prime, reciprocal);
    }
    exponent >>= 1;
    if (exponent != 0) base = multiply_mod(base, base, prime, reciprocal);
  }
  return result;
}

template <bool WithDerivative>
__global__ void polynomial_evaluate_kernel(
    const std::uint32_t* __restrict__ coefficients,
    std::size_t polynomial_count,
    std::size_t coefficient_count,
    const std::uint32_t* __restrict__ points,
    std::size_t point_count,
    std::uint32_t prime,
    std::uint64_t reciprocal,
    std::uint32_t* __restrict__ values,
    std::uint32_t* __restrict__ derivatives) {
  const auto index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto count = polynomial_count * point_count;
  if (index >= count) return;
  const auto polynomial = index / point_count;
  const auto point = points[index % point_count];
  const auto* row = coefficients + polynomial * coefficient_count;
  auto value = row[coefficient_count - 1];
  std::uint32_t derivative = 0;
  for (std::size_t coefficient = coefficient_count - 1;
       coefficient != 0;
       --coefficient) {
    if constexpr (WithDerivative) {
      derivative = add_mod(
          multiply_mod(derivative, point, prime, reciprocal), value, prime);
    }
    value = add_mod(
        multiply_mod(value, point, prime, reciprocal),
        row[coefficient - 1], prime);
  }
  values[index] = value;
  if constexpr (WithDerivative) derivatives[index] = derivative;
}

__device__ std::uint32_t dot_mod(
    const std::uint32_t* left,
    const std::uint32_t* right,
    std::size_t count,
    std::uint32_t prime,
    std::uint64_t reciprocal,
    std::uint64_t reduction_terms) {
  if (reduction_terms == 1) {
    std::uint32_t sum = 0;
    for (std::size_t index = 0; index < count; ++index) {
      sum = add_mod(
          sum,
          multiply_mod(left[index], right[index], prime, reciprocal),
          prime);
    }
    return sum;
  }
  std::uint64_t sum = 0;
  std::uint64_t terms = 0;
  for (std::size_t index = 0; index < count; ++index) {
    sum += static_cast<std::uint64_t>(left[index]) * right[index];
    ++terms;
    if (terms == reduction_terms) {
      sum %= prime;
      terms = 0;
    }
  }
  return static_cast<std::uint32_t>(sum % prime);
}

__global__ void linear_obstruction_kernel(
    const std::uint32_t* __restrict__ left_nullspace,
    const std::uint32_t* __restrict__ right_hand_sides,
    std::size_t batch_count,
    std::size_t row_count,
    std::size_t obstruction_count,
    std::uint32_t prime,
    std::uint64_t reciprocal,
    std::uint64_t reduction_terms,
    std::uint64_t* inconsistency_rows) {
  const auto index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto count = batch_count * obstruction_count;
  if (index >= count) return;
  const auto batch = index / obstruction_count;
  const auto row = index % obstruction_count;
  if (dot_mod(
          left_nullspace + row * row_count,
          right_hand_sides + batch * row_count,
          row_count,
          prime,
          reciprocal,
          reduction_terms) != 0) {
    atomicMin(
        reinterpret_cast<unsigned long long*>(inconsistency_rows + batch),
        static_cast<unsigned long long>(row));
  }
}

__global__ void linear_solution_kernel(
    const std::uint32_t* __restrict__ solution_operator,
    const std::uint32_t* __restrict__ pivot_columns,
    const std::uint32_t* __restrict__ right_hand_sides,
    const std::uint64_t* __restrict__ inconsistency_rows,
    std::size_t batch_count,
    std::size_t row_count,
    std::size_t column_count,
    std::size_t rank,
    std::uint32_t prime,
    std::uint64_t reciprocal,
    std::uint64_t reduction_terms,
    std::uint32_t* solutions) {
  const auto index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto count = batch_count * rank;
  if (index >= count) return;
  const auto batch = index / rank;
  const auto column = pivot_columns[index % rank];
  if (inconsistency_rows[batch] !=
      std::numeric_limits<std::uint64_t>::max()) {
    return;
  }
  solutions[batch * column_count + column] = dot_mod(
      solution_operator + column * row_count,
      right_hand_sides + batch * row_count,
      row_count,
      prime,
      reciprocal,
      reduction_terms);
}

__global__ void determinant_kernel(
    std::uint32_t* matrices,
    std::size_t matrix_count,
    std::uint32_t order,
    std::uint32_t prime,
    std::uint64_t reciprocal,
    std::uint32_t* determinants) {
  const auto matrix_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (matrix_index >= matrix_count) return;
  const auto matrix_size = static_cast<std::size_t>(order) * order;
  auto* matrix = matrices + matrix_index * matrix_size;
  std::uint32_t determinant = 1 % prime;
  bool negate = false;
  for (std::uint32_t column = 0; column < order; ++column) {
    std::uint32_t pivot = column;
    while (pivot < order && matrix[pivot * order + column] == 0) ++pivot;
    if (pivot == order) {
      determinant = 0;
      break;
    }
    if (pivot != column) {
      for (std::uint32_t entry = column; entry < order; ++entry) {
        const auto temporary = matrix[column * order + entry];
        matrix[column * order + entry] = matrix[pivot * order + entry];
        matrix[pivot * order + entry] = temporary;
      }
      negate = !negate;
    }
    const auto pivot_value = matrix[column * order + column];
    determinant = multiply_mod(
        determinant, pivot_value, prime, reciprocal);
    const auto inverse = power_mod(
        pivot_value, prime - 2, prime, reciprocal);
    for (std::uint32_t row = column + 1; row < order; ++row) {
      const auto leading = matrix[row * order + column];
      if (leading == 0) continue;
      const auto factor = multiply_mod(
          leading, inverse, prime, reciprocal);
      for (std::uint32_t entry = column + 1; entry < order; ++entry) {
        const auto product = multiply_mod(
            factor, matrix[column * order + entry], prime, reciprocal);
        matrix[row * order + entry] = subtract_mod(
            matrix[row * order + entry], product, prime);
      }
    }
  }
  if (negate && determinant != 0) determinant = prime - determinant;
  determinants[matrix_index] = determinant;
}

void release_polynomial_batches(HipPolynomialPlan* plan) {
  (void)hipFree(plan->points);
  (void)hipFree(plan->values);
  (void)hipFree(plan->derivatives);
  plan->points = nullptr;
  plan->values = nullptr;
  plan->derivatives = nullptr;
  plan->point_capacity = 0;
}

int reserve_polynomial(HipPolynomialPlan* plan, std::size_t point_count) {
  if (point_count <= plan->point_capacity) return 0;
  release_polynomial_batches(plan);
  if (multiply_overflows(plan->polynomial_count, point_count)) return -2;
  const auto output_count = plan->polynomial_count * point_count;
  if (multiply_overflows(output_count, sizeof(std::uint32_t))) return -2;
  if (status(hipMalloc(&plan->points,
                       point_count * sizeof(std::uint32_t))) != 0 ||
      status(hipMalloc(&plan->values,
                       output_count * sizeof(std::uint32_t))) != 0 ||
      status(hipMalloc(&plan->derivatives,
                       output_count * sizeof(std::uint32_t))) != 0) {
    release_polynomial_batches(plan);
    return -2;
  }
  plan->point_capacity = point_count;
  return 0;
}

void release_determinant_batches(HipDeterminantPlan* plan) {
  (void)hipFree(plan->matrices);
  (void)hipFree(plan->determinants);
  plan->matrices = nullptr;
  plan->determinants = nullptr;
  plan->matrix_capacity = 0;
}

void release_linear_batches(HipLinearSystemPlan* plan) {
  (void)hipFree(plan->right_hand_sides);
  (void)hipFree(plan->solutions);
  (void)hipFree(plan->inconsistency_rows);
  plan->right_hand_sides = nullptr;
  plan->solutions = nullptr;
  plan->inconsistency_rows = nullptr;
  plan->batch_capacity = 0;
}

int reserve_linear(HipLinearSystemPlan* plan, std::size_t batch_count) {
  if (batch_count <= plan->batch_capacity) return 0;
  release_linear_batches(plan);
  if (multiply_overflows(batch_count, plan->row_count) ||
      multiply_overflows(batch_count, plan->column_count) ||
      multiply_overflows(batch_count * plan->row_count, sizeof(std::uint32_t)) ||
      multiply_overflows(batch_count * plan->column_count, sizeof(std::uint32_t)) ||
      multiply_overflows(batch_count, sizeof(std::uint64_t))) {
    return -2;
  }
  if (status(hipMalloc(
          &plan->right_hand_sides,
          batch_count * plan->row_count * sizeof(std::uint32_t))) != 0 ||
      status(hipMalloc(
          &plan->solutions,
          batch_count * plan->column_count * sizeof(std::uint32_t))) != 0 ||
      status(hipMalloc(
          &plan->inconsistency_rows,
          batch_count * sizeof(std::uint64_t))) != 0) {
    release_linear_batches(plan);
    return -2;
  }
  plan->batch_capacity = batch_count;
  return 0;
}

int reserve_determinants(HipDeterminantPlan* plan, std::size_t matrix_count) {
  if (matrix_count <= plan->matrix_capacity) return 0;
  release_determinant_batches(plan);
  const auto matrix_size = static_cast<std::size_t>(plan->order) * plan->order;
  if (multiply_overflows(matrix_count, matrix_size) ||
      multiply_overflows(matrix_count * matrix_size, sizeof(std::uint32_t))) {
    return -2;
  }
  if (status(hipMalloc(&plan->matrices,
                       matrix_count * matrix_size * sizeof(std::uint32_t))) != 0 ||
      status(hipMalloc(&plan->determinants,
                       matrix_count * sizeof(std::uint32_t))) != 0) {
    release_determinant_batches(plan);
    return -2;
  }
  plan->matrix_capacity = matrix_count;
  return 0;
}

}  // namespace

extern "C" int fast_math_hip_polynomial_create(
    const std::uint32_t* coefficients,
    std::size_t polynomial_count,
    std::size_t coefficient_count,
    std::uint32_t prime,
    void** result) {
  try {
    if (result != nullptr) *result = nullptr;
    if (coefficients == nullptr || polynomial_count == 0 ||
        coefficient_count == 0 || !is_prime(prime) || result == nullptr ||
        multiply_overflows(polynomial_count, coefficient_count)) return -1;
    const auto entry_count = polynomial_count * coefficient_count;
    for (std::size_t index = 0; index < entry_count; ++index) {
      if (coefficients[index] >= prime) return -1;
    }
    auto* plan = new (std::nothrow) HipPolynomialPlan{
        nullptr, nullptr, nullptr, nullptr,
        polynomial_count, coefficient_count, prime,
        reciprocal_for(prime), 0};
    if (plan == nullptr) return -2;
    const auto bytes = entry_count * sizeof(std::uint32_t);
    if (status(hipMalloc(&plan->coefficients, bytes)) != 0 ||
        status(hipMemcpy(plan->coefficients, coefficients, bytes,
                         hipMemcpyHostToDevice)) != 0) {
      (void)hipFree(plan->coefficients);
      delete plan;
      return -2;
    }
    *result = plan;
    return 0;
  } catch (...) {
    return -2;
  }
}

extern "C" int fast_math_hip_polynomial_destroy(void* opaque) {
  auto* plan = static_cast<HipPolynomialPlan*>(opaque);
  if (plan == nullptr) return 0;
  release_polynomial_batches(plan);
  const auto result = status(hipFree(plan->coefficients));
  delete plan;
  return result;
}

extern "C" int fast_math_hip_polynomial_evaluate(
    void* opaque,
    const std::uint32_t* points,
    std::size_t point_count,
    std::uint8_t with_derivative,
    std::uint32_t* values,
    std::uint32_t* derivatives) {
  auto* plan = static_cast<HipPolynomialPlan*>(opaque);
  if (plan == nullptr || (point_count != 0 &&
      (points == nullptr || values == nullptr ||
       (with_derivative != 0 && derivatives == nullptr)))) return -1;
  if (point_count == 0) return 0;
  for (std::size_t index = 0; index < point_count; ++index) {
    if (points[index] >= plan->prime) return -1;
  }
  int result = reserve_polynomial(plan, point_count);
  if (result == 0) result = status(hipMemcpy(
      plan->points, points, point_count * sizeof(std::uint32_t),
      hipMemcpyHostToDevice));
  if (result == 0) {
    const auto count = plan->polynomial_count * point_count;
    constexpr std::size_t threads = 256;
    const auto blocks = (count + threads - 1) / threads;
    if (with_derivative != 0) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(polynomial_evaluate_kernel<true>),
          dim3(blocks), dim3(threads), 0, 0,
          plan->coefficients, plan->polynomial_count,
          plan->coefficient_count, plan->points, point_count,
          plan->prime, plan->reciprocal,
          plan->values, plan->derivatives);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(polynomial_evaluate_kernel<false>),
          dim3(blocks), dim3(threads), 0, 0,
          plan->coefficients, plan->polynomial_count,
          plan->coefficient_count, plan->points, point_count,
          plan->prime, plan->reciprocal,
          plan->values, plan->derivatives);
    }
    result = status(hipGetLastError());
  }
  if (result == 0) result = status(hipDeviceSynchronize());
  const auto bytes = plan->polynomial_count * point_count * sizeof(std::uint32_t);
  if (result == 0) result = status(hipMemcpy(
      values, plan->values, bytes, hipMemcpyDeviceToHost));
  if (result == 0 && with_derivative != 0) result = status(hipMemcpy(
      derivatives, plan->derivatives, bytes, hipMemcpyDeviceToHost));
  return result;
}

extern "C" int fast_math_hip_determinant_create(
    std::uint32_t order,
    std::uint32_t prime,
    void** result) {
  if (result != nullptr) *result = nullptr;
  if (result == nullptr || order == 0 || order > 32 || !is_prime(prime)) {
    return -1;
  }
  auto* plan = new (std::nothrow) HipDeterminantPlan{
      nullptr, nullptr, order, prime, reciprocal_for(prime), 0};
  if (plan == nullptr) return -2;
  *result = plan;
  return 0;
}

extern "C" int fast_math_hip_determinant_destroy(void* opaque) {
  auto* plan = static_cast<HipDeterminantPlan*>(opaque);
  if (plan == nullptr) return 0;
  release_determinant_batches(plan);
  delete plan;
  return 0;
}

extern "C" int fast_math_hip_determinants(
    void* opaque,
    const std::uint32_t* matrices,
    std::size_t matrix_count,
    std::uint32_t* determinants) {
  auto* plan = static_cast<HipDeterminantPlan*>(opaque);
  if (plan == nullptr || (matrix_count != 0 &&
      (matrices == nullptr || determinants == nullptr))) return -1;
  if (matrix_count == 0) return 0;
  const auto matrix_size = static_cast<std::size_t>(plan->order) * plan->order;
  if (multiply_overflows(matrix_count, matrix_size)) return -2;
  const auto entry_count = matrix_count * matrix_size;
  for (std::size_t index = 0; index < entry_count; ++index) {
    if (matrices[index] >= plan->prime) return -1;
  }
  int result = reserve_determinants(plan, matrix_count);
  if (result == 0) result = status(hipMemcpy(
      plan->matrices, matrices, entry_count * sizeof(std::uint32_t),
      hipMemcpyHostToDevice));
  if (result == 0) {
    constexpr std::size_t threads = 128;
    const auto blocks = (matrix_count + threads - 1) / threads;
    hipLaunchKernelGGL(
        determinant_kernel,
        dim3(blocks), dim3(threads), 0, 0,
        plan->matrices, matrix_count, plan->order,
        plan->prime, plan->reciprocal, plan->determinants);
    result = status(hipGetLastError());
  }
  if (result == 0) result = status(hipDeviceSynchronize());
  if (result == 0) result = status(hipMemcpy(
      determinants, plan->determinants,
      matrix_count * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
  return result;
}

extern "C" int fast_math_hip_linear_system_create(
    const std::uint32_t* solution_operator,
    const std::uint32_t* pivot_columns,
    const std::uint32_t* left_nullspace,
    std::size_t row_count,
    std::size_t column_count,
    std::size_t rank,
    std::size_t obstruction_count,
    std::uint32_t prime,
    void** result) {
  try {
    if (result != nullptr) *result = nullptr;
    if (result == nullptr || solution_operator == nullptr || row_count == 0 ||
        column_count == 0 || rank > row_count || rank > column_count ||
        obstruction_count != row_count - rank || !is_prime(prime) ||
        (rank != 0 && pivot_columns == nullptr) ||
        multiply_overflows(column_count, row_count) ||
        multiply_overflows(obstruction_count, row_count)) {
      return -1;
    }
    for (std::size_t index = 0; index < rank; ++index) {
      if (pivot_columns[index] >= column_count ||
          (index != 0 && pivot_columns[index - 1] >= pivot_columns[index])) {
        return -1;
      }
    }
    const auto operator_count = column_count * row_count;
    const auto obstruction_entries = obstruction_count * row_count;
    if (multiply_overflows(operator_count, sizeof(std::uint32_t)) ||
        multiply_overflows(rank, sizeof(std::uint32_t)) ||
        multiply_overflows(obstruction_entries, sizeof(std::uint32_t))) {
      return -2;
    }
    if (obstruction_entries != 0 && left_nullspace == nullptr) return -1;
    for (std::size_t index = 0; index < operator_count; ++index) {
      if (solution_operator[index] >= prime) return -1;
    }
    for (std::size_t index = 0; index < obstruction_entries; ++index) {
      if (left_nullspace[index] >= prime) return -1;
    }
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    const auto maximum_value = static_cast<std::uint64_t>(prime - 1);
    const auto maximum_product = maximum_value * maximum_value;
    const auto reduction_terms = std::max<std::uint64_t>(
        1, (maximum - maximum_value) / maximum_product);
    auto* plan = new (std::nothrow) HipLinearSystemPlan{
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        row_count,
        column_count,
        rank,
        obstruction_count,
        prime,
        reciprocal_for(prime),
        reduction_terms,
        0};
    if (plan == nullptr) return -2;
    int code = status(hipMalloc(
        &plan->solution_operator,
        operator_count * sizeof(std::uint32_t)));
    if (code == 0) code = status(hipMemcpy(
        plan->solution_operator,
        solution_operator,
        operator_count * sizeof(std::uint32_t),
        hipMemcpyHostToDevice));
    if (code == 0 && rank != 0) code = status(hipMalloc(
        &plan->pivot_columns, rank * sizeof(std::uint32_t)));
    if (code == 0 && rank != 0) code = status(hipMemcpy(
        plan->pivot_columns,
        pivot_columns,
        rank * sizeof(std::uint32_t),
        hipMemcpyHostToDevice));
    if (code == 0 && obstruction_entries != 0) code = status(hipMalloc(
        &plan->left_nullspace,
        obstruction_entries * sizeof(std::uint32_t)));
    if (code == 0 && obstruction_entries != 0) code = status(hipMemcpy(
        plan->left_nullspace,
        left_nullspace,
        obstruction_entries * sizeof(std::uint32_t),
        hipMemcpyHostToDevice));
    if (code != 0) {
      (void)hipFree(plan->solution_operator);
      (void)hipFree(plan->pivot_columns);
      (void)hipFree(plan->left_nullspace);
      delete plan;
      return code;
    }
    *result = plan;
    return 0;
  } catch (...) {
    return -2;
  }
}

extern "C" int fast_math_hip_linear_system_destroy(void* opaque) {
  auto* plan = static_cast<HipLinearSystemPlan*>(opaque);
  if (plan == nullptr) return 0;
  release_linear_batches(plan);
  int result = status(hipFree(plan->solution_operator));
  const auto pivot_result = status(hipFree(plan->pivot_columns));
  const auto left_result = status(hipFree(plan->left_nullspace));
  if (result == 0) result = pivot_result;
  if (result == 0) result = left_result;
  delete plan;
  return result;
}

extern "C" int fast_math_hip_linear_system_solve(
    void* opaque,
    const std::uint32_t* right_hand_sides,
    std::size_t batch_count,
    std::uint32_t* solutions,
    std::int64_t* inconsistency_rows) {
  auto* plan = static_cast<HipLinearSystemPlan*>(opaque);
  if (plan == nullptr || (batch_count != 0 &&
      (right_hand_sides == nullptr || solutions == nullptr ||
       inconsistency_rows == nullptr)) ||
      multiply_overflows(batch_count, plan->row_count) ||
      multiply_overflows(batch_count, plan->column_count)) {
    return -1;
  }
  if (batch_count == 0) return 0;
  const auto input_count = batch_count * plan->row_count;
  for (std::size_t index = 0; index < input_count; ++index) {
    if (right_hand_sides[index] >= plan->prime) return -1;
  }
  int result = reserve_linear(plan, batch_count);
  if (result == 0) result = status(hipMemcpy(
      plan->right_hand_sides,
      right_hand_sides,
      input_count * sizeof(std::uint32_t),
      hipMemcpyHostToDevice));
  if (result == 0) result = status(hipMemset(
      plan->inconsistency_rows,
      0xff,
      batch_count * sizeof(std::uint64_t)));
  if (result == 0) result = status(hipMemset(
      plan->solutions,
      0,
      batch_count * plan->column_count * sizeof(std::uint32_t)));
  constexpr std::size_t threads = 256;
  if (result == 0 && plan->obstruction_count != 0) {
    const auto count = batch_count * plan->obstruction_count;
    const auto blocks = (count + threads - 1) / threads;
    hipLaunchKernelGGL(
        linear_obstruction_kernel,
        dim3(blocks), dim3(threads), 0, 0,
        plan->left_nullspace,
        plan->right_hand_sides,
        batch_count,
        plan->row_count,
        plan->obstruction_count,
        plan->prime,
        plan->reciprocal,
        plan->reduction_terms,
        plan->inconsistency_rows);
    result = status(hipGetLastError());
  }
  if (result == 0 && plan->rank != 0) {
    const auto count = batch_count * plan->rank;
    const auto blocks = (count + threads - 1) / threads;
    hipLaunchKernelGGL(
        linear_solution_kernel,
        dim3(blocks), dim3(threads), 0, 0,
        plan->solution_operator,
        plan->pivot_columns,
        plan->right_hand_sides,
        plan->inconsistency_rows,
        batch_count,
        plan->row_count,
        plan->column_count,
        plan->rank,
        plan->prime,
        plan->reciprocal,
        plan->reduction_terms,
        plan->solutions);
    result = status(hipGetLastError());
  }
  if (result == 0) result = status(hipDeviceSynchronize());
  if (result == 0) result = status(hipMemcpy(
      solutions,
      plan->solutions,
      batch_count * plan->column_count * sizeof(std::uint32_t),
      hipMemcpyDeviceToHost));
  if (result == 0) result = status(hipMemcpy(
      inconsistency_rows,
      plan->inconsistency_rows,
      batch_count * sizeof(std::uint64_t),
      hipMemcpyDeviceToHost));
  return result;
}
