#include "fast_math.h"
#include "modulus_u32.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
