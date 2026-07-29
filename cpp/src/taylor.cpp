#include "fast_math.h"
#include "parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

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

template <typename Work>
void run_chunks(
    std::size_t sample_count,
    std::uint64_t chunk_size,
    std::uint32_t thread_count,
    Work work) {
  if (chunk_size == 0) {
    throw std::invalid_argument("chunk_size must be positive");
  }
  const auto chunk_count = static_cast<std::size_t>(
      (sample_count + chunk_size - 1) / chunk_size);
  fast_math_internal::parallel_for_static(
      chunk_count, thread_count, [&](std::size_t chunk) {
      const auto begin = static_cast<std::size_t>(chunk * chunk_size);
      const auto end = std::min(
          sample_count,
          static_cast<std::size_t>(begin + chunk_size));
      work(begin, end);
      });
}

void multiply_add(
    double first_real,
    double first_imag,
    double second_real,
    double second_imag,
    double add_real,
    double add_imag,
    double& result_real,
    double& result_imag) {
  result_real = std::fma(
      first_real,
      second_real,
      std::fma(-first_imag, second_imag, add_real));
  result_imag = std::fma(
      first_real,
      second_imag,
      std::fma(first_imag, second_real, add_imag));
}

}  // namespace

extern "C" {

int fast_math_taylor_coefficients_f64(
    const double* base_interleaved,
    const double* logarithms,
    std::size_t sample_count,
    std::uint32_t maximum_order,
    std::uint64_t chunk_size,
    std::uint32_t thread_count,
    double* coefficients_interleaved,
    fast_math_taylor_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (base_interleaved == nullptr || logarithms == nullptr ||
        coefficients_interleaved == nullptr || stats == nullptr) {
      throw std::invalid_argument("Taylor coefficient pointer is null");
    }
    if (sample_count == 0) {
      throw std::invalid_argument("Taylor source must be nonempty");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const auto order_count = static_cast<std::size_t>(maximum_order) + 1;
    run_chunks(
        sample_count,
        chunk_size,
        thread_count,
        [&](std::size_t begin, std::size_t end) {
          for (auto sample = begin; sample < end; ++sample) {
            const double base_real = base_interleaved[2 * sample];
            const double base_imag = base_interleaved[2 * sample + 1];
            double scale = 1.0;
            for (std::size_t order = 0; order < order_count; ++order) {
              const auto output = order * sample_count + sample;
              coefficients_interleaved[2 * output] = base_real * scale;
              coefficients_interleaved[2 * output + 1] = base_imag * scale;
              scale *= logarithms[sample] /
                  static_cast<double>(order + 1);
            }
          }
        });
    stats->sample_count = sample_count;
    stats->order_count = static_cast<std::uint32_t>(order_count);
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

int fast_math_taylor_evaluate_f64(
    const double* basis_interleaved,
    const double* delta_interleaved,
    std::size_t sample_count,
    std::uint32_t order_count,
    std::uint64_t chunk_size,
    std::uint32_t thread_count,
    double* values_interleaved,
    double* log_moments_interleaved,
    fast_math_taylor_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (basis_interleaved == nullptr || delta_interleaved == nullptr ||
        values_interleaved == nullptr ||
        log_moments_interleaved == nullptr || stats == nullptr) {
      throw std::invalid_argument("Taylor evaluation pointer is null");
    }
    if (sample_count == 0 || order_count == 0) {
      throw std::invalid_argument(
          "Taylor basis and sample vector must be nonempty");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const auto highest_order =
        static_cast<std::size_t>(order_count - 1);
    run_chunks(
        sample_count,
        chunk_size,
        thread_count,
        [&](std::size_t begin, std::size_t end) {
          auto sample = begin;
#if defined(__aarch64__)
          for (; sample + 1 < end; sample += 2) {
            const auto delta =
                vld2q_f64(delta_interleaved + 2 * sample);
            const auto minus_delta_real = vnegq_f64(delta.val[0]);
            const auto minus_delta_imag = vnegq_f64(delta.val[1]);
            const auto highest_input =
                highest_order * sample_count + sample;
            auto highest = vld2q_f64(
                basis_interleaved + 2 * highest_input);
            auto value_real = highest.val[0];
            auto value_imag = highest.val[1];
            auto moment_real = highest_order == 0
                ? vdupq_n_f64(0.0)
                : vmulq_n_f64(
                    value_real,
                    static_cast<double>(highest_order));
            auto moment_imag = highest_order == 0
                ? vdupq_n_f64(0.0)
                : vmulq_n_f64(
                    value_imag,
                    static_cast<double>(highest_order));

            for (auto order = highest_order; order > 1;) {
              --order;
              const auto input = order * sample_count + sample;
              const auto basis = vld2q_f64(
                  basis_interleaved + 2 * input);
              const auto next_value_real = vfmaq_f64(
                  vfmsq_f64(
                      basis.val[0],
                      minus_delta_imag,
                      value_imag),
                  minus_delta_real,
                  value_real);
              const auto next_value_imag = vfmaq_f64(
                  vfmaq_f64(
                      basis.val[1],
                      minus_delta_imag,
                      value_real),
                  minus_delta_real,
                  value_imag);
              value_real = next_value_real;
              value_imag = next_value_imag;

              const auto order_scale = static_cast<double>(order);
              const auto order_real =
                  vmulq_n_f64(basis.val[0], order_scale);
              const auto order_imag =
                  vmulq_n_f64(basis.val[1], order_scale);
              const auto next_moment_real = vfmaq_f64(
                  vfmsq_f64(
                      order_real,
                      minus_delta_imag,
                      moment_imag),
                  minus_delta_real,
                  moment_real);
              const auto next_moment_imag = vfmaq_f64(
                  vfmaq_f64(
                      order_imag,
                      minus_delta_imag,
                      moment_real),
                  minus_delta_real,
                  moment_imag);
              moment_real = next_moment_real;
              moment_imag = next_moment_imag;
            }

            if (highest_order != 0) {
              const auto basis = vld2q_f64(
                  basis_interleaved + 2 * sample);
              const auto next_value_real = vfmaq_f64(
                  vfmsq_f64(
                      basis.val[0],
                      minus_delta_imag,
                      value_imag),
                  minus_delta_real,
                  value_real);
              const auto next_value_imag = vfmaq_f64(
                  vfmaq_f64(
                      basis.val[1],
                      minus_delta_imag,
                      value_real),
                  minus_delta_real,
                  value_imag);
              value_real = next_value_real;
              value_imag = next_value_imag;
            }
            const float64x2x2_t values = {
                value_real, value_imag};
            const float64x2x2_t moments = {
                moment_real, moment_imag};
            vst2q_f64(values_interleaved + 2 * sample, values);
            vst2q_f64(
                log_moments_interleaved + 2 * sample, moments);
          }
#endif
          for (; sample < end; ++sample) {
            const double minus_delta_real =
                -delta_interleaved[2 * sample];
            const double minus_delta_imag =
                -delta_interleaved[2 * sample + 1];
            const auto highest_input =
                highest_order * sample_count + sample;
            double value_real =
                basis_interleaved[2 * highest_input];
            double value_imag =
                basis_interleaved[2 * highest_input + 1];
            double moment_real = highest_order == 0
                ? 0.0
                : static_cast<double>(highest_order) * value_real;
            double moment_imag = highest_order == 0
                ? 0.0
                : static_cast<double>(highest_order) * value_imag;

            for (auto order = highest_order; order > 1;) {
              --order;
              const auto input = order * sample_count + sample;
              const double basis_real = basis_interleaved[2 * input];
              const double basis_imag = basis_interleaved[2 * input + 1];
              double next_real = 0.0;
              double next_imag = 0.0;
              multiply_add(
                  minus_delta_real,
                  minus_delta_imag,
                  value_real,
                  value_imag,
                  basis_real,
                  basis_imag,
                  next_real,
                  next_imag);
              value_real = next_real;
              value_imag = next_imag;
              const double order_real =
                  static_cast<double>(order) * basis_real;
              const double order_imag =
                  static_cast<double>(order) * basis_imag;
              multiply_add(
                  minus_delta_real,
                  minus_delta_imag,
                  moment_real,
                  moment_imag,
                  order_real,
                  order_imag,
                  next_real,
                  next_imag);
              moment_real = next_real;
              moment_imag = next_imag;
            }

            if (highest_order != 0) {
              multiply_add(
                  minus_delta_real,
                  minus_delta_imag,
                  value_real,
                  value_imag,
                  basis_interleaved[2 * sample],
                  basis_interleaved[2 * sample + 1],
                  value_real,
                  value_imag);
            }
            values_interleaved[2 * sample] = value_real;
            values_interleaved[2 * sample + 1] = value_imag;
            log_moments_interleaved[2 * sample] = moment_real;
            log_moments_interleaved[2 * sample + 1] = moment_imag;
          }
        });
    stats->sample_count = sample_count;
    stats->order_count = order_count;
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
