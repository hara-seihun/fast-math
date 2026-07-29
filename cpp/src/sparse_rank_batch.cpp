#include "fast_math.h"
#include "parallel.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
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

int fast_math_sparse_rank_mod_u32_batch(
    const std::uint64_t* row_offsets,
    const std::uint32_t* column_indices,
    const std::uint32_t* values_by_prime,
    std::size_t row_count,
    std::size_t column_count,
    std::size_t nonzero_count,
    const std::uint32_t* primes,
    std::size_t prime_count,
    std::size_t target_rank,
    std::uint32_t thread_count,
    std::uint64_t* pivot_rows,
    std::uint32_t* pivot_columns,
    std::size_t pivot_capacity,
    fast_math_sparse_rank_stats* stats,
    fast_math_sparse_rank_batch_stats* batch_stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (row_offsets == nullptr || primes == nullptr ||
        stats == nullptr || batch_stats == nullptr) {
      throw std::invalid_argument("sparse rank batch pointer is null");
    }
    if (prime_count == 0) {
      throw std::invalid_argument(
          "sparse rank batch requires at least one prime");
    }
    if (nonzero_count != 0 &&
        (column_indices == nullptr || values_by_prime == nullptr)) {
      throw std::invalid_argument(
          "sparse rank batch entry pointer is null");
    }
    if ((pivot_rows == nullptr) != (pivot_columns == nullptr)) {
      throw std::invalid_argument(
          "sparse rank batch pivot outputs must both be null or nonnull");
    }
    if (nonzero_count >
        std::numeric_limits<std::size_t>::max() / prime_count) {
      throw std::overflow_error("sparse rank batch values are too large");
    }
    if (pivot_capacity != 0 &&
        prime_count >
            std::numeric_limits<std::size_t>::max() / pivot_capacity) {
      throw std::overflow_error("sparse rank batch pivots are too large");
    }

    set_error(error_message, error_message_size, "");
    *batch_stats = {};
    std::fill(stats, stats + prime_count, fast_math_sparse_rank_stats{});
    const auto started = Clock::now();
    std::vector<int> statuses(prime_count, 0);
    std::vector<std::array<char, 256>> errors(prime_count);
    fast_math_internal::parallel_for_static(
        prime_count,
        thread_count,
        [&](std::size_t task) {
          auto* task_pivot_rows = pivot_rows == nullptr
              ? nullptr
              : pivot_rows + task * pivot_capacity;
          auto* task_pivot_columns = pivot_columns == nullptr
              ? nullptr
              : pivot_columns + task * pivot_capacity;
          const auto* task_values = nonzero_count == 0
              ? nullptr
              : values_by_prime + task * nonzero_count;
          statuses[task] = fast_math_sparse_rank_mod_u32(
              row_offsets,
              column_indices,
              task_values,
              row_count,
              column_count,
              nonzero_count,
              primes[task],
              target_rank,
              task_pivot_rows,
              task_pivot_columns,
              pivot_capacity,
              &stats[task],
              errors[task].data(),
              errors[task].size());
        });
    for (std::size_t task = 0; task < prime_count; ++task) {
      if (statuses[task] != 0) {
        set_error(
            error_message,
            error_message_size,
            errors[task].data());
        return statuses[task];
      }
    }
    batch_stats->prime_count = prime_count;
    batch_stats->thread_count =
        fast_math_internal::parallel_worker_count(
            prime_count,
            thread_count);
    batch_stats->elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(
        error_message,
        error_message_size,
        "unknown sparse rank batch error");
    return 2;
  }
}

}  // extern "C"
