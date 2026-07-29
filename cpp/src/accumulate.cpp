#include "fast_math.h"
#include "parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t ceil_div(std::uint64_t numerator, std::uint64_t denominator) {
  return numerator / denominator + (numerator % denominator != 0);
}

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

std::uint64_t count_pairs(
    const double* inverse,
    std::size_t inverse_size,
    std::uint64_t source_first,
    std::size_t source_size,
    std::uint64_t output_limit) {
  if (source_size == 0) {
    return 0;
  }
  const auto source_last =
      source_first + static_cast<std::uint64_t>(source_size) - 1;
  std::uint64_t pairs = 0;
  for (std::uint64_t divisor = 1; divisor < inverse_size; ++divisor) {
    if (inverse[divisor] == 0.0) {
      continue;
    }
    const auto maximum = std::min(source_last, output_limit / divisor);
    if (maximum >= source_first) {
      pairs += maximum - source_first + 1;
    }
  }
  return pairs;
}

void validate_arguments(
    const double* inverse,
    std::size_t inverse_size,
    const double* primary,
    std::size_t primary_size,
    const double* transformed,
    std::size_t transformed_size,
    std::uint64_t transformed_first,
    const double* low,
    std::size_t low_size,
    std::uint64_t output_limit,
    std::uint64_t tile_size,
    double* common_output,
    double* low_output,
    lambda_fast_stats* stats) {
  if (inverse == nullptr || inverse_size == 0) {
    throw std::invalid_argument("inverse must contain index zero");
  }
  if (primary_size != 0 && primary == nullptr) {
    throw std::invalid_argument("primary pointer is null");
  }
  if (transformed_size != 0 && transformed == nullptr) {
    throw std::invalid_argument("transformed pointer is null");
  }
  if (low_size != 0 && low == nullptr) {
    throw std::invalid_argument("low pointer is null");
  }
  if (transformed_first == 0) {
    throw std::invalid_argument("transformed_first must be positive");
  }
  if (output_limit == 0) {
    throw std::invalid_argument("output_limit must be positive");
  }
  if (tile_size == 0) {
    throw std::invalid_argument("tile_size must be positive");
  }
  if (common_output == nullptr || low_output == nullptr || stats == nullptr) {
    throw std::invalid_argument("output or stats pointer is null");
  }
  if (output_limit >
      (std::numeric_limits<std::size_t>::max() / (2 * sizeof(double))) - 1) {
    throw std::overflow_error("output_limit exceeds addressable memory");
  }
  if (transformed_size != 0 &&
      transformed_first >
          std::numeric_limits<std::uint64_t>::max() - transformed_size + 1) {
    throw std::overflow_error("transformed source range overflows");
  }
}

}  // namespace

extern "C" {

const char* fast_math_version() {
  return "0.7.0";
}

const char* lambda_fast_version() {
  return fast_math_version();
}

int lambda_fast_accumulate_f64(
    const double* inverse,
    std::size_t inverse_size,
    const double* primary,
    std::size_t primary_size,
    const double* transformed_interleaved,
    std::size_t transformed_size,
    std::uint64_t transformed_first,
    const double* low,
    std::size_t low_size,
    std::uint64_t output_limit,
    std::uint64_t tile_size,
    std::uint32_t thread_count,
    double* common_interleaved,
    double* low_output,
    lambda_fast_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    validate_arguments(
        inverse,
        inverse_size,
        primary,
        primary_size,
        transformed_interleaved,
        transformed_size,
        transformed_first,
        low,
        low_size,
        output_limit,
        tile_size,
        common_interleaved,
        low_output,
        stats);
    set_error(error_message, error_message_size, "");

    const auto started = Clock::now();
    const auto output_size = static_cast<std::size_t>(output_limit + 1);
    std::fill_n(common_interleaved, output_size * 2, 0.0);
    std::fill_n(low_output, output_size, 0.0);

    stats->primary_pairs =
        count_pairs(inverse, inverse_size, 1, primary_size, output_limit);
    stats->transformed_pairs = count_pairs(
        inverse,
        inverse_size,
        transformed_first,
        transformed_size,
        output_limit);
    stats->low_pairs =
        count_pairs(inverse, inverse_size, 1, low_size, output_limit);

    const auto tile_count =
        static_cast<std::size_t>((output_limit + tile_size - 1) / tile_size);
    fast_math_internal::parallel_for_dynamic(
        tile_count, thread_count, [&](std::size_t tile) {
        const auto left =
            1 + static_cast<std::uint64_t>(tile) * tile_size;
        const auto right =
            std::min(output_limit, left + tile_size - 1);

        for (std::uint64_t divisor = 1; divisor < inverse_size; ++divisor) {
          const double inverse_coefficient = inverse[divisor];
          if (inverse_coefficient == 0.0) {
            continue;
          }

          const auto source_first =
              std::max<std::uint64_t>(1, ceil_div(left, divisor));
          const auto primary_last = std::min<std::uint64_t>(
              primary_size, right / divisor);
          const auto low_last =
              std::min<std::uint64_t>(low_size, right / divisor);
          const auto shared_last = std::min(primary_last, low_last);
          for (auto index = source_first; index <= shared_last; ++index) {
            const auto output = static_cast<std::size_t>(divisor * index);
            common_interleaved[2 * output] +=
                inverse_coefficient * primary[index - 1];
            low_output[output] +=
                inverse_coefficient * low[index - 1];
          }
          for (auto index = std::max(source_first, shared_last + 1);
               index <= primary_last;
               ++index) {
            const auto output = static_cast<std::size_t>(divisor * index);
            common_interleaved[2 * output] +=
                inverse_coefficient * primary[index - 1];
          }
          for (auto index = std::max(source_first, shared_last + 1);
               index <= low_last;
               ++index) {
            const auto output = static_cast<std::size_t>(divisor * index);
            low_output[output] +=
                inverse_coefficient * low[index - 1];
          }

          if (transformed_size != 0) {
            const auto transformed_last_source =
                transformed_first +
                static_cast<std::uint64_t>(transformed_size) - 1;
            const auto transformed_begin = std::max(
                transformed_first, ceil_div(left, divisor));
            const auto transformed_end = std::min(
                transformed_last_source, right / divisor);
            for (auto index = transformed_begin;
                 index <= transformed_end;
                 ++index) {
              const auto output = static_cast<std::size_t>(divisor * index);
              const auto source =
                  static_cast<std::size_t>(index - transformed_first);
              common_interleaved[2 * output] +=
                  inverse_coefficient * transformed_interleaved[2 * source];
              common_interleaved[2 * output + 1] +=
                  inverse_coefficient *
                  transformed_interleaved[2 * source + 1];
            }
          }

        }
        });

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
