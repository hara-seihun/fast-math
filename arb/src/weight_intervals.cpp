#include "fast_math_arb.h"
#include "parallel.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {

struct Worker {
  arb_t base;
  arb_t exponent;
  arb_t weight;
  arb_t binary64_value;
  arb_t sigma_lower;
  arb_t sigma_upper;
  arf_t lower;
  arf_t upper;
};

void worker_init(
    Worker& worker,
    const arb_t sigma_lower,
    const arb_t sigma_upper) {
  arb_init(worker.base);
  arb_init(worker.exponent);
  arb_init(worker.weight);
  arb_init(worker.binary64_value);
  arb_init(worker.sigma_lower);
  arb_init(worker.sigma_upper);
  arf_init(worker.lower);
  arf_init(worker.upper);
  arb_set(worker.sigma_lower, sigma_lower);
  arb_set(worker.sigma_upper, sigma_upper);
}

void worker_clear(Worker& worker) {
  arb_clear(worker.base);
  arb_clear(worker.exponent);
  arb_clear(worker.weight);
  arb_clear(worker.binary64_value);
  arb_clear(worker.sigma_lower);
  arb_clear(worker.sigma_upper);
  arf_clear(worker.lower);
  arf_clear(worker.upper);
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

std::size_t checked_chunk_count(
    std::size_t item_count,
    std::uint64_t chunk_size) {
  if (chunk_size == 0) {
    throw std::invalid_argument("chunk_size must be positive");
  }
  return item_count / chunk_size +
      static_cast<std::size_t>(item_count % chunk_size != 0);
}

}  // namespace

extern "C" {

int fast_math_arb_weight_intervals_u64(
    const std::uint64_t* left,
    const std::uint64_t* right,
    std::size_t block_count,
    const arb_t sigma_lower,
    const arb_t sigma_upper,
    slong precision,
    std::uint32_t requested_workers,
    std::uint64_t chunk_size,
    double* weight_lower,
    double* weight_upper,
    fast_math_arb_weight_interval_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  std::unique_ptr<Worker[]> workers;
  std::uint32_t worker_count = 0;
  try {
    if (
        left == nullptr || right == nullptr
        || weight_lower == nullptr || weight_upper == nullptr) {
      throw std::invalid_argument("weight interval pointer is null");
    }
    if (precision <= 0) {
      throw std::invalid_argument("precision must be positive");
    }
    const auto chunk_count =
        checked_chunk_count(block_count, chunk_size);
    worker_count = fast_math_internal::parallel_worker_count(
        chunk_count, requested_workers);
    workers = std::make_unique<Worker[]>(worker_count);
    for (std::uint32_t worker = 0; worker < worker_count; ++worker) {
      worker_init(workers[worker], sigma_lower, sigma_upper);
    }
    set_error(error_message, error_message_size, "");
    const auto started = std::chrono::steady_clock::now();
    std::atomic<int> failure{0};
    fast_math_internal::parallel_for_dynamic_indexed(
        chunk_count,
        requested_workers,
        [&](std::size_t chunk, std::size_t worker_index) {
          if (failure.load(std::memory_order_relaxed) != 0) {
            return;
          }
          auto& worker = workers[worker_index];
          const auto begin =
              static_cast<std::size_t>(chunk * chunk_size);
          const auto end = std::min(
              block_count,
              begin + static_cast<std::size_t>(chunk_size));
          for (auto block = begin; block < end; ++block) {
            if (
                left[block] == 0 || right[block] < left[block]
                || left[block] > std::numeric_limits<ulong>::max()
                || right[block] > std::numeric_limits<ulong>::max()) {
              failure.store(1, std::memory_order_relaxed);
              return;
            }

            arb_set_ui(worker.base, static_cast<ulong>(left[block]));
            arb_neg(worker.exponent, worker.sigma_lower);
            arb_pow(
                worker.weight,
                worker.base,
                worker.exponent,
                precision);
            arb_get_ubound_arf(worker.upper, worker.weight, precision);
            weight_upper[block] =
                arf_get_d(worker.upper, ARF_RND_CEIL);
            arb_set_d(
                worker.binary64_value, weight_upper[block]);
            if (!arb_le(worker.weight, worker.binary64_value)) {
              failure.store(2, std::memory_order_relaxed);
              return;
            }

            arb_set_ui(worker.base, static_cast<ulong>(right[block]));
            arb_neg(worker.exponent, worker.sigma_upper);
            arb_pow(
                worker.weight,
                worker.base,
                worker.exponent,
                precision);
            arb_get_lbound_arf(worker.lower, worker.weight, precision);
            weight_lower[block] =
                arf_get_d(worker.lower, ARF_RND_FLOOR);
            arb_set_d(
                worker.binary64_value, weight_lower[block]);
            if (
                !arb_le(worker.binary64_value, worker.weight)
                || weight_lower[block] <= 0.0
                || weight_lower[block] > weight_upper[block]) {
              failure.store(3, std::memory_order_relaxed);
              return;
            }
          }
        });
    const auto finished = std::chrono::steady_clock::now();
    const int status = failure.load(std::memory_order_relaxed);
    for (std::uint32_t worker = 0; worker < worker_count; ++worker) {
      worker_clear(workers[worker]);
    }
    if (status != 0) {
      set_error(
          error_message,
          error_message_size,
          "weight interval evaluation failed");
      return 3;
    }
    if (stats != nullptr) {
      stats->block_count = block_count;
      stats->worker_count = worker_count;
      stats->elapsed_seconds =
          std::chrono::duration<double>(finished - started).count();
    }
    return 0;
  } catch (const std::exception& error) {
    if (workers != nullptr) {
      for (std::uint32_t worker = 0; worker < worker_count; ++worker) {
        worker_clear(workers[worker]);
      }
    }
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    if (workers != nullptr) {
      for (std::uint32_t worker = 0; worker < worker_count; ++worker) {
        worker_clear(workers[worker]);
      }
    }
    set_error(
        error_message,
        error_message_size,
        "unknown weight interval error");
    return 2;
  }
}

}  // extern "C"
