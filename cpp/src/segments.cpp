#include "fast_math.h"
#include "parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
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

}  // namespace

extern "C" {

int fast_math_segmented_complex_stats_f64(
    const double* values_interleaved,
    std::size_t sample_count,
    const std::uint64_t* offsets,
    std::size_t segment_count,
    std::uint32_t thread_count,
    double* sums_interleaved,
    double* l1,
    double* variation,
    fast_math_segment_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (values_interleaved == nullptr || offsets == nullptr ||
        sums_interleaved == nullptr || l1 == nullptr ||
        variation == nullptr || stats == nullptr) {
      throw std::invalid_argument("segmented stats pointer is null");
    }
    if (sample_count == 0 || segment_count == 0) {
      throw std::invalid_argument("samples and segments must be nonempty");
    }
    if (offsets[0] != 0 || offsets[segment_count] != sample_count) {
      throw std::invalid_argument("offsets must cover the sample vector");
    }
    for (std::size_t segment = 0; segment < segment_count; ++segment) {
      if (offsets[segment] >= offsets[segment + 1]) {
        throw std::invalid_argument(
            "offsets must define nonempty ordered segments");
      }
    }

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    fast_math_internal::parallel_for_dynamic(
        segment_count, thread_count, [&](std::size_t segment) {
        const auto begin = static_cast<std::size_t>(offsets[segment]);
        const auto end = static_cast<std::size_t>(offsets[segment + 1]);
        double sum_real = 0.0;
        double sum_imag = 0.0;
        double segment_l1 = 0.0;
        double segment_variation = 0.0;
        double previous_real = values_interleaved[2 * begin];
        double previous_imag = values_interleaved[2 * begin + 1];
        for (auto sample = begin; sample < end; ++sample) {
          const double real = values_interleaved[2 * sample];
          const double imag = values_interleaved[2 * sample + 1];
          sum_real += real;
          sum_imag += imag;
          segment_l1 += std::hypot(real, imag);
          if (sample != begin) {
            segment_variation +=
                std::hypot(real - previous_real, imag - previous_imag);
          }
          previous_real = real;
          previous_imag = imag;
        }
        sums_interleaved[2 * segment] = sum_real;
        sums_interleaved[2 * segment + 1] = sum_imag;
        l1[segment] = segment_l1;
        variation[segment] = segment_variation;
        });

    stats->sample_count = sample_count;
    stats->segment_count = segment_count;
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
