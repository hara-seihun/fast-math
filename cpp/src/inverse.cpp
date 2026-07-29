#include "fast_math.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void set_error(
    char* destination,
    std::size_t destination_size,
    const char* message) {
  if (destination == nullptr || destination_size == 0) {
    return;
  }
  std::strncpy(destination, message, destination_size - 1);
  destination[destination_size - 1] = '\0';
}

}  // namespace

extern "C" {

int lambda_fast_dirichlet_inverse_f64(
    const double* source,
    std::size_t source_size,
    double* coefficients,
    lambda_fast_inverse_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (source == nullptr || coefficients == nullptr || stats == nullptr) {
      throw std::invalid_argument("source, coefficient, or stats pointer is null");
    }
    if (source_size == 0) {
      throw std::invalid_argument("source must be nonempty");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();

    std::fill_n(coefficients, source_size + 1, 0.0);
    std::vector<double> accumulated(source_size + 1, 0.0);
    coefficients[1] = 1.0;
    for (std::size_t divisor = 1; divisor <= source_size; ++divisor) {
      if (divisor > 1) {
        coefficients[divisor] = -accumulated[divisor];
      }
      const auto maximum = source_size / divisor;
      for (std::size_t index = 2; index <= maximum; ++index) {
        accumulated[divisor * index] +=
            coefficients[divisor] * source[index - 1];
        stats->update_count += 1;
      }
    }
    stats->elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(error_message, error_message_size, "unknown native error");
    return 2;
  }
}

}
