#include "fast_math.h"
#include "parallel.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>

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

std::uint64_t width_mask(std::uint32_t width) {
  return width == 64
      ? std::numeric_limits<std::uint64_t>::max()
      : (std::uint64_t{1} << width) - 1;
}

std::uint64_t rotate_left_width(
    std::uint64_t value,
    std::uint32_t lag,
    std::uint32_t width,
    std::uint64_t mask) {
  if (lag == 0) {
    return value;
  }
  return ((value << lag) | (value >> (width - lag))) & mask;
}

}  // namespace

extern "C" {

int fast_math_cyclic_correlation_profiles_u64(
    const std::uint64_t* masks,
    std::size_t mask_count,
    std::uint32_t bit_width,
    std::uint32_t thread_count,
    std::uint8_t* intersection_counts,
    std::int16_t* signed_correlations,
    fast_math_cyclic_profile_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("cyclic profile stats pointer is null");
    }
    *stats = {};
    stats->mask_count = mask_count;
    stats->bit_width = bit_width;
    stats->lag_count = bit_width;
    if (bit_width == 0 || bit_width > 64) {
      throw std::invalid_argument(
          "cyclic profile bit width must be between one and 64");
    }
    if (thread_count > 1024) {
      throw std::invalid_argument(
          "cyclic profile thread count must be at most 1024");
    }
    if (mask_count != 0 &&
        (masks == nullptr || intersection_counts == nullptr ||
         signed_correlations == nullptr)) {
      throw std::invalid_argument("cyclic profile input or output pointer is null");
    }
    if (mask_count >
        std::numeric_limits<std::size_t>::max() / bit_width) {
      throw std::invalid_argument("cyclic profile output size overflows size_t");
    }
    const auto valid_bits = width_mask(bit_width);
    for (std::size_t index = 0; index < mask_count; ++index) {
      if ((masks[index] & ~valid_bits) != 0) {
        throw std::invalid_argument(
            "cyclic profile mask has bits outside bit_width");
      }
    }
    set_error(error_message, error_message_size, "");
    if (mask_count == 0) {
      return 0;
    }
    const auto unique_lags = bit_width / 2 + 1;
    if (mask_count >
        std::numeric_limits<std::uint64_t>::max() / unique_lags) {
      throw std::invalid_argument("cyclic profile evaluation count overflows uint64");
    }
    stats->popcount_evaluations = mask_count * unique_lags;

    const auto requested_workers =
        fast_math_internal::parallel_worker_count(mask_count, thread_count);
    const auto use_parallel = mask_count >= 256 && requested_workers > 1;
    stats->worker_count = use_parallel ? requested_workers : 1;
    const auto started = Clock::now();
    auto process = [&](std::size_t index) {
      const auto value = masks[index];
      const auto weight = static_cast<std::int32_t>(std::popcount(value));
      auto* intersections =
          intersection_counts + index * static_cast<std::size_t>(bit_width);
      auto* correlations =
          signed_correlations + index * static_cast<std::size_t>(bit_width);
      for (std::uint32_t lag = 0; lag < unique_lags; ++lag) {
        const auto rotated =
            rotate_left_width(value, lag, bit_width, valid_bits);
        const auto overlap = static_cast<std::int32_t>(
            std::popcount(value & rotated));
        const auto correlation = static_cast<std::int32_t>(bit_width) -
            4 * (weight - overlap);
        intersections[lag] = static_cast<std::uint8_t>(overlap);
        correlations[lag] = static_cast<std::int16_t>(correlation);
        if (lag != 0 && 2 * lag != bit_width) {
          intersections[bit_width - lag] =
              static_cast<std::uint8_t>(overlap);
          correlations[bit_width - lag] =
              static_cast<std::int16_t>(correlation);
        }
      }
    };
    if (use_parallel) {
      fast_math_internal::parallel_for_static(
          mask_count, requested_workers, process);
    } else {
      for (std::size_t index = 0; index < mask_count; ++index) {
        process(index);
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

}  // extern "C"
