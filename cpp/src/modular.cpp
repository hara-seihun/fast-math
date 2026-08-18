#include "fast_math.h"
#include "modulus_u32.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(FAST_MATH_HAVE_FLINT)
#include <flint/nmod_mat.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

void set_error(char* output, std::size_t capacity, const std::string& message) {
  if (output == nullptr || capacity == 0) return;
  const auto count = std::min(capacity - 1, message.size());
  std::memcpy(output, message.data(), count);
  output[count] = '\0';
}

std::size_t checked_product(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("modular batch shape overflows");
  }
  return left * right;
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

using Modulus = fast_math_internal::PrimeModulusU32;

}  // namespace

extern "C" int fast_math_polynomial_evaluate_mod_u32(
    const std::uint32_t* coefficients,
    std::size_t polynomial_count,
    std::size_t coefficient_count,
    const std::uint32_t* points,
    std::size_t point_count,
    std::uint32_t prime,
    std::uint32_t thread_count,
    std::uint32_t* values,
    std::uint32_t* derivatives,
    fast_math_modular_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr || coefficient_count == 0 || !is_prime(prime)) {
      throw std::invalid_argument("modular polynomial input is invalid");
    }
    const auto coefficient_entries = checked_product(
        polynomial_count, coefficient_count);
    const auto output_entries = checked_product(polynomial_count, point_count);
    if ((coefficient_entries != 0 && coefficients == nullptr) ||
        (point_count != 0 && points == nullptr) ||
        (output_entries != 0 && values == nullptr)) {
      throw std::invalid_argument("modular polynomial pointer is null");
    }
    for (std::size_t index = 0; index < coefficient_entries; ++index) {
      if (coefficients[index] >= prime) {
        throw std::invalid_argument("polynomial coefficient is outside the field");
      }
    }
    for (std::size_t index = 0; index < point_count; ++index) {
      if (points[index] >= prime) {
        throw std::invalid_argument("evaluation point is outside the field");
      }
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const Modulus modulus(prime);
    fast_math_internal::parallel_for_static(
        output_entries,
        thread_count,
        [&](std::size_t index) {
          const auto polynomial = index / point_count;
          const auto point = points[index % point_count];
          const auto* row = coefficients + polynomial * coefficient_count;
          auto value = row[coefficient_count - 1];
          std::uint32_t derivative = 0;
          for (std::size_t coefficient = coefficient_count - 1;
               coefficient != 0;
               --coefficient) {
            if (derivatives != nullptr) {
              derivative = modulus.add(
                  modulus.multiply(derivative, point), value);
            }
            value = modulus.add(
                modulus.multiply(value, point), row[coefficient - 1]);
          }
          values[index] = value;
          if (derivatives != nullptr) derivatives[index] = derivative;
        });
    stats->batch_count = polynomial_count;
    stats->item_count = output_entries;
    stats->operation_count = checked_product(output_entries, coefficient_count);
    stats->prime = prime;
    stats->thread_count = fast_math_internal::parallel_worker_count(
        output_entries, thread_count);
    stats->elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(error_message, error_message_size, "unknown modular polynomial error");
    return 2;
  }
}

extern "C" int fast_math_determinants_mod_u32(
    const std::uint32_t* matrices,
    std::size_t matrix_count,
    std::uint32_t order,
    std::uint32_t prime,
    std::uint32_t thread_count,
    std::uint32_t* determinants,
    fast_math_modular_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr || order == 0 || order > 64 || !is_prime(prime)) {
      throw std::invalid_argument("modular determinant input is invalid");
    }
    const auto matrix_size = checked_product(order, order);
    const auto entry_count = checked_product(matrix_count, matrix_size);
    if ((entry_count != 0 && matrices == nullptr) ||
        (matrix_count != 0 && determinants == nullptr)) {
      throw std::invalid_argument("modular determinant pointer is null");
    }
    for (std::size_t index = 0; index < entry_count; ++index) {
      if (matrices[index] >= prime) {
        throw std::invalid_argument("matrix entry is outside the field");
      }
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const Modulus modulus(prime);
    fast_math_internal::parallel_for_static(
        matrix_count,
        thread_count,
        [&](std::size_t matrix_index) {
          std::vector<std::uint32_t> work(
              matrices + matrix_index * matrix_size,
              matrices + (matrix_index + 1) * matrix_size);
          std::uint32_t determinant = 1 % prime;
          bool negate = false;
          for (std::uint32_t column = 0; column < order; ++column) {
            std::uint32_t pivot = column;
            while (pivot < order &&
                   work[static_cast<std::size_t>(pivot) * order + column] == 0) {
              ++pivot;
            }
            if (pivot == order) {
              determinant = 0;
              break;
            }
            if (pivot != column) {
              for (std::uint32_t entry = column; entry < order; ++entry) {
                std::swap(
                    work[static_cast<std::size_t>(column) * order + entry],
                    work[static_cast<std::size_t>(pivot) * order + entry]);
              }
              negate = !negate;
            }
            const auto pivot_value =
                work[static_cast<std::size_t>(column) * order + column];
            determinant = modulus.multiply(determinant, pivot_value);
            const auto inverse = modulus.inverse(pivot_value);
            for (std::uint32_t row = column + 1; row < order; ++row) {
              const auto leading =
                  work[static_cast<std::size_t>(row) * order + column];
              if (leading == 0) continue;
              const auto factor = modulus.multiply(leading, inverse);
              for (std::uint32_t entry = column + 1; entry < order; ++entry) {
                auto& target =
                    work[static_cast<std::size_t>(row) * order + entry];
                const auto product = modulus.multiply(
                    factor,
                    work[static_cast<std::size_t>(column) * order + entry]);
                target = modulus.subtract(target, product);
              }
            }
          }
          if (negate && determinant != 0) determinant = prime - determinant;
          determinants[matrix_index] = determinant;
        });
    stats->batch_count = matrix_count;
    stats->item_count = matrix_count;
    stats->operation_count = checked_product(
        matrix_count,
        checked_product(order, checked_product(order, order)));
    stats->prime = prime;
    stats->thread_count = fast_math_internal::parallel_worker_count(
        matrix_count, thread_count);
    stats->elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(error_message, error_message_size, "unknown modular determinant error");
    return 2;
  }
}

struct fast_math_modular_linear_system {
  std::size_t row_count;
  std::size_t column_count;
  std::size_t rank;
  std::uint32_t prime;
  std::vector<std::uint32_t> reduced_row_echelon;
  std::vector<std::uint32_t> pivot_columns;
  std::vector<std::uint32_t> solution_operator;
  std::vector<std::uint32_t> right_nullspace;
  std::vector<std::uint32_t> left_nullspace;
#if defined(FAST_MATH_HAVE_FLINT)
  nmod_mat_struct flint_transform{};
  mutable nmod_mat_struct flint_right{};
  mutable nmod_mat_struct flint_transformed{};
  mutable std::mutex flint_mutex;
  bool flint_transform_initialized = false;
  mutable bool flint_workspace_initialized = false;
  mutable std::size_t flint_batch_capacity = 0;

  ~fast_math_modular_linear_system() {
    if (flint_workspace_initialized) {
      nmod_mat_clear(&flint_transformed);
      nmod_mat_clear(&flint_right);
    }
    if (flint_transform_initialized) {
      nmod_mat_clear(&flint_transform);
    }
  }
#endif
};

namespace {

std::uint32_t dot_mod_u32(
    const std::uint32_t* left,
    const std::uint32_t* right,
    std::size_t count,
    const Modulus& modulus) {
#if defined(__SIZEOF_INT128__)
  unsigned __int128 sum = 0;
  for (std::size_t index = 0; index < count; ++index) {
    sum += static_cast<unsigned __int128>(left[index]) * right[index];
  }
  return static_cast<std::uint32_t>(sum % modulus.prime());
#else
  std::uint32_t sum = 0;
  for (std::size_t index = 0; index < count; ++index) {
    sum = modulus.add(sum, modulus.multiply(left[index], right[index]));
  }
  return sum;
#endif
}

#if defined(FAST_MATH_HAVE_FLINT)
void solve_linear_system_flint(
    const fast_math_modular_linear_system& system,
    const std::uint32_t* right_hand_sides,
    std::size_t right_hand_side_count,
    std::uint32_t* solutions,
    std::int64_t* inconsistency_rows) {
  const std::lock_guard lock(system.flint_mutex);
  if (!system.flint_workspace_initialized ||
      system.flint_batch_capacity != right_hand_side_count) {
    if (system.flint_workspace_initialized) {
      nmod_mat_clear(&system.flint_transformed);
      nmod_mat_clear(&system.flint_right);
    }
    nmod_mat_init(
        &system.flint_right,
        static_cast<slong>(system.row_count),
        static_cast<slong>(right_hand_side_count),
        system.prime);
    nmod_mat_init(
        &system.flint_transformed,
        static_cast<slong>(system.row_count),
        static_cast<slong>(right_hand_side_count),
        system.prime);
    system.flint_workspace_initialized = true;
    system.flint_batch_capacity = right_hand_side_count;
  }
  constexpr std::size_t kTransposeTile = 128;
  for (std::size_t batch_start = 0;
       batch_start < right_hand_side_count;
       batch_start += kTransposeTile) {
    const auto batch_end = std::min(
        right_hand_side_count, batch_start + kTransposeTile);
    for (std::size_t row = 0; row < system.row_count; ++row) {
      for (std::size_t batch = batch_start; batch < batch_end; ++batch) {
        nmod_mat_entry(&system.flint_right, row, batch) =
            right_hand_sides[batch * system.row_count + row];
      }
    }
  }
  nmod_mat_mul(
      &system.flint_transformed,
      &system.flint_transform,
      &system.flint_right);
  for (std::size_t batch_start = 0;
       batch_start < right_hand_side_count;
       batch_start += kTransposeTile) {
    const auto batch_end = std::min(
        right_hand_side_count, batch_start + kTransposeTile);
    for (std::size_t batch = batch_start; batch < batch_end; ++batch) {
      std::fill_n(
          solutions + batch * system.column_count,
          system.column_count,
          0);
      std::int64_t obstruction = -1;
      for (std::size_t row = system.rank;
           row < system.row_count;
           ++row) {
        if (nmod_mat_entry(&system.flint_transformed, row, batch) != 0) {
          obstruction = static_cast<std::int64_t>(row - system.rank);
          break;
        }
      }
      inconsistency_rows[batch] = obstruction;
    }
    for (std::size_t row = 0; row < system.rank; ++row) {
      const auto column = system.pivot_columns[row];
      for (std::size_t batch = batch_start; batch < batch_end; ++batch) {
        if (inconsistency_rows[batch] < 0) {
          solutions[batch * system.column_count + column] =
              static_cast<std::uint32_t>(nmod_mat_entry(
                  &system.flint_transformed, row, batch));
        }
      }
    }
  }
}
#endif

void set_linear_stats(
    fast_math_modular_linear_stats* stats,
    const fast_math_modular_linear_system& system,
    std::size_t batch_count,
    std::size_t operation_count,
    std::uint32_t thread_count,
    double elapsed_seconds) {
  stats->row_count = system.row_count;
  stats->column_count = system.column_count;
  stats->rank = system.rank;
  stats->batch_count = batch_count;
  stats->operation_count = operation_count;
  stats->prime = system.prime;
  stats->thread_count = thread_count;
  stats->elapsed_seconds = elapsed_seconds;
}

}  // namespace

extern "C" int fast_math_modular_linear_system_create_u32(
    const std::uint32_t* matrix,
    std::size_t row_count,
    std::size_t column_count,
    std::uint32_t prime,
    fast_math_modular_linear_system** system,
    fast_math_modular_linear_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (system != nullptr) *system = nullptr;
    if (system == nullptr || stats == nullptr || row_count == 0 ||
        column_count == 0 ||
        row_count > std::numeric_limits<std::uint32_t>::max() ||
        column_count > std::numeric_limits<std::uint32_t>::max() ||
        !is_prime(prime)) {
      throw std::invalid_argument("modular linear system input is invalid");
    }
    const auto entry_count = checked_product(row_count, column_count);
    if (matrix == nullptr) {
      throw std::invalid_argument("modular linear system matrix is null");
    }
    for (std::size_t index = 0; index < entry_count; ++index) {
      if (matrix[index] >= prime) {
        throw std::invalid_argument("matrix entry is outside the field");
      }
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    auto candidate = std::make_unique<fast_math_modular_linear_system>();
    candidate->row_count = row_count;
    candidate->column_count = column_count;
    candidate->rank = 0;
    candidate->prime = prime;
    candidate->reduced_row_echelon.assign(matrix, matrix + entry_count);
    auto transformation_size = checked_product(row_count, row_count);
    std::vector<std::uint32_t> transformation(transformation_size, 0);
    for (std::size_t row = 0; row < row_count; ++row) {
      transformation[row * row_count + row] = 1 % prime;
    }
    const Modulus modulus(prime);
    for (std::size_t column = 0;
         column < column_count && candidate->rank < row_count;
         ++column) {
      auto pivot = candidate->rank;
      while (pivot < row_count &&
             candidate->reduced_row_echelon[pivot * column_count + column] == 0) {
        ++pivot;
      }
      if (pivot == row_count) continue;
      const auto pivot_row = candidate->rank;
      if (pivot != pivot_row) {
        for (std::size_t entry = 0; entry < column_count; ++entry) {
          std::swap(
              candidate->reduced_row_echelon[pivot_row * column_count + entry],
              candidate->reduced_row_echelon[pivot * column_count + entry]);
        }
        for (std::size_t entry = 0; entry < row_count; ++entry) {
          std::swap(
              transformation[pivot_row * row_count + entry],
              transformation[pivot * row_count + entry]);
        }
      }
      const auto inverse = modulus.inverse(
          candidate->reduced_row_echelon[pivot_row * column_count + column]);
      for (std::size_t entry = column; entry < column_count; ++entry) {
        auto& value = candidate->reduced_row_echelon[
            pivot_row * column_count + entry];
        value = modulus.multiply(value, inverse);
      }
      for (std::size_t entry = 0; entry < row_count; ++entry) {
        auto& value = transformation[pivot_row * row_count + entry];
        value = modulus.multiply(value, inverse);
      }
      for (std::size_t row = 0; row < row_count; ++row) {
        if (row == pivot_row) continue;
        const auto factor = candidate->reduced_row_echelon[
            row * column_count + column];
        if (factor == 0) continue;
        candidate->reduced_row_echelon[row * column_count + column] = 0;
        for (std::size_t entry = column + 1;
             entry < column_count;
             ++entry) {
          auto& target = candidate->reduced_row_echelon[
              row * column_count + entry];
          target = modulus.subtract(
              target,
              modulus.multiply(
                  factor,
                  candidate->reduced_row_echelon[
                      pivot_row * column_count + entry]));
        }
        for (std::size_t entry = 0; entry < row_count; ++entry) {
          auto& target = transformation[row * row_count + entry];
          target = modulus.subtract(
              target,
              modulus.multiply(
                  factor,
                  transformation[pivot_row * row_count + entry]));
        }
      }
      candidate->pivot_columns.push_back(static_cast<std::uint32_t>(column));
      ++candidate->rank;
    }
    candidate->solution_operator.assign(
        checked_product(column_count, row_count), 0);
    for (std::size_t row = 0; row < candidate->rank; ++row) {
      const auto pivot_column = candidate->pivot_columns[row];
      std::copy_n(
          transformation.data() + row * row_count,
          row_count,
          candidate->solution_operator.data() + pivot_column * row_count);
    }
    const auto free_count = column_count - candidate->rank;
    candidate->right_nullspace.assign(
        checked_product(free_count, column_count), 0);
    std::size_t free_row = 0;
    std::size_t pivot_index = 0;
    for (std::size_t column = 0; column < column_count; ++column) {
      if (pivot_index < candidate->rank &&
          candidate->pivot_columns[pivot_index] == column) {
        ++pivot_index;
        continue;
      }
      auto* basis = candidate->right_nullspace.data() +
          free_row * column_count;
      basis[column] = 1 % prime;
      for (std::size_t row = 0; row < candidate->rank; ++row) {
        const auto value = candidate->reduced_row_echelon[
            row * column_count + column];
        basis[candidate->pivot_columns[row]] =
            value == 0 ? 0 : prime - value;
      }
      ++free_row;
    }
#if defined(FAST_MATH_HAVE_FLINT)
    nmod_mat_init(
        &candidate->flint_transform,
        static_cast<slong>(row_count),
        static_cast<slong>(row_count),
        prime);
    candidate->flint_transform_initialized = true;
    for (std::size_t row = 0; row < row_count; ++row) {
      for (std::size_t column = 0; column < row_count; ++column) {
        nmod_mat_entry(&candidate->flint_transform, row, column) =
            transformation[row * row_count + column];
      }
    }
#endif
    const auto obstruction_count = row_count - candidate->rank;
    candidate->left_nullspace.assign(
        transformation.begin() + candidate->rank * row_count,
        transformation.begin() +
            (candidate->rank + obstruction_count) * row_count);
    const auto elapsed =
        std::chrono::duration<double>(Clock::now() - started).count();
    const auto width = checked_product(row_count, row_count + column_count);
    set_linear_stats(
        stats,
        *candidate,
        0,
        checked_product(candidate->rank, width),
        1,
        elapsed);
    *system = candidate.release();
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(
        error_message,
        error_message_size,
        "unknown modular linear system creation error");
    return 2;
  }
}

extern "C" void fast_math_modular_linear_system_destroy(
    fast_math_modular_linear_system* system) {
  delete system;
}

extern "C" int fast_math_modular_linear_system_export_u32(
    const fast_math_modular_linear_system* system,
    std::uint32_t* reduced_row_echelon,
    std::uint32_t* pivot_columns,
    std::uint32_t* solution_operator,
    std::uint32_t* right_nullspace,
    std::uint32_t* left_nullspace,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (system == nullptr || reduced_row_echelon == nullptr ||
        solution_operator == nullptr ||
        (system->rank != 0 && pivot_columns == nullptr) ||
        (!system->right_nullspace.empty() && right_nullspace == nullptr) ||
        (!system->left_nullspace.empty() && left_nullspace == nullptr)) {
      throw std::invalid_argument("modular linear system export is invalid");
    }
    set_error(error_message, error_message_size, "");
    std::copy(
        system->reduced_row_echelon.begin(),
        system->reduced_row_echelon.end(),
        reduced_row_echelon);
    if (!system->pivot_columns.empty()) {
      std::copy(
          system->pivot_columns.begin(),
          system->pivot_columns.end(),
          pivot_columns);
    }
    std::copy(
        system->solution_operator.begin(),
        system->solution_operator.end(),
        solution_operator);
    if (!system->right_nullspace.empty()) {
      std::copy(
          system->right_nullspace.begin(),
          system->right_nullspace.end(),
          right_nullspace);
    }
    if (!system->left_nullspace.empty()) {
      std::copy(
          system->left_nullspace.begin(),
          system->left_nullspace.end(),
          left_nullspace);
    }
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(
        error_message,
        error_message_size,
        "unknown modular linear system export error");
    return 2;
  }
}

extern "C" int fast_math_modular_linear_system_solve_u32(
    const fast_math_modular_linear_system* system,
    const std::uint32_t* right_hand_sides,
    std::size_t right_hand_side_count,
    std::uint32_t thread_count,
    std::uint32_t* solutions,
    std::int64_t* inconsistency_rows,
    fast_math_modular_linear_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (system == nullptr || stats == nullptr ||
        (right_hand_side_count != 0 &&
         (right_hand_sides == nullptr || solutions == nullptr ||
          inconsistency_rows == nullptr))) {
      throw std::invalid_argument("modular linear system solve is invalid");
    }
    const auto input_count = checked_product(
        right_hand_side_count, system->row_count);
    for (std::size_t index = 0; index < input_count; ++index) {
      if (right_hand_sides[index] >= system->prime) {
        throw std::invalid_argument("right-hand side is outside the field");
      }
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const auto obstruction_count = system->row_count - system->rank;
    bool used_flint = false;
#if defined(FAST_MATH_HAVE_FLINT)
    if (right_hand_side_count >= 256 && system->row_count >= 8 &&
        right_hand_side_count <= static_cast<std::size_t>(
            std::numeric_limits<slong>::max())) {
      solve_linear_system_flint(
          *system,
          right_hand_sides,
          right_hand_side_count,
          solutions,
          inconsistency_rows);
      used_flint = true;
    }
#endif
    if (!used_flint) {
      const Modulus modulus(system->prime);
      fast_math_internal::parallel_for_static(
          right_hand_side_count,
          thread_count,
          [&](std::size_t batch) {
            const auto* right = right_hand_sides + batch * system->row_count;
            auto* solution = solutions + batch * system->column_count;
            std::int64_t obstruction = -1;
            for (std::size_t row = 0; row < obstruction_count; ++row) {
              if (dot_mod_u32(
                      system->left_nullspace.data() +
                          row * system->row_count,
                      right,
                      system->row_count,
                      modulus) != 0) {
                obstruction = static_cast<std::int64_t>(row);
                break;
              }
            }
            inconsistency_rows[batch] = obstruction;
            std::fill_n(solution, system->column_count, 0);
            if (obstruction >= 0) return;
            for (const auto column : system->pivot_columns) {
              solution[column] = dot_mod_u32(
                  system->solution_operator.data() +
                      column * system->row_count,
                  right,
                  system->row_count,
                  modulus);
            }
          });
    }
    const auto elapsed =
        std::chrono::duration<double>(Clock::now() - started).count();
    const auto output_rows = checked_product(
        system->rank + obstruction_count, system->row_count);
    set_linear_stats(
        stats,
        *system,
        right_hand_side_count,
        checked_product(right_hand_side_count, output_rows),
        used_flint
            ? 1
            : fast_math_internal::parallel_worker_count(
                  right_hand_side_count, thread_count),
        elapsed);
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(
        error_message,
        error_message_size,
        "unknown modular linear system solve error");
    return 2;
  }
}
