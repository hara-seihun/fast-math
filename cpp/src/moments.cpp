#include "fast_math.h"
#include "lambda_fast.h"
#include "parallel.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct ChunkMoments {
  std::vector<double> value;
  std::vector<double> ordinary;
  std::vector<double> phase_current;
  std::vector<double> radial;
  double maximum_modulus_squared = 0.0;
  double maximum_derivative_squared = 0.0;
};

#if defined(__aarch64__)
constexpr std::size_t kMaximumSimdPowerCount = 64;

struct SimdMoments {
  float64x2_t value;
  float64x2_t ordinary;
  float64x2_t phase_current;
  float64x2_t radial;
};
#endif

double integer_power(double base, std::uint32_t exponent) {
  double result = 1.0;
  for (std::uint32_t index = 0; index < exponent; ++index) {
    result *= base;
  }
  return result;
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

}  // namespace

extern "C" {

int lambda_fast_power_moments_f64(
    const double* values_interleaved,
    const double* derivatives_interleaved,
    std::size_t sample_count,
    double mesh_step,
    std::uint32_t minimum_power,
    std::uint32_t maximum_power,
    std::uint64_t chunk_size,
    std::uint32_t thread_count,
    lambda_fast_power_moment* moments,
    std::size_t moment_capacity,
    lambda_fast_power_moment_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (sample_count != 0 &&
        (values_interleaved == nullptr || derivatives_interleaved == nullptr)) {
      throw std::invalid_argument("value or derivative pointer is null");
    }
    if (!(mesh_step > 0.0) || !std::isfinite(mesh_step)) {
      throw std::invalid_argument("mesh_step must be finite and positive");
    }
    if (minimum_power == 0 || maximum_power < minimum_power) {
      throw std::invalid_argument("invalid power range");
    }
    if (chunk_size == 0) {
      throw std::invalid_argument("chunk_size must be positive");
    }
    const auto power_count =
        static_cast<std::size_t>(maximum_power - minimum_power + 1);
    if (moments == nullptr || stats == nullptr ||
        moment_capacity < power_count) {
      throw std::invalid_argument("moment output capacity is too small");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();

    const auto chunk_count = static_cast<std::size_t>(
        (sample_count + chunk_size - 1) / chunk_size);
    std::vector<ChunkMoments> chunks(chunk_count);
    for (auto& chunk : chunks) {
      chunk.value.assign(power_count, 0.0);
      chunk.ordinary.assign(power_count, 0.0);
      chunk.phase_current.assign(power_count, 0.0);
      chunk.radial.assign(power_count, 0.0);
    }
    std::vector<double> powers(power_count);
    std::vector<double> power_squares(power_count);
    for (std::size_t index = 0; index < power_count; ++index) {
      powers[index] =
          static_cast<double>(minimum_power + index);
      power_squares[index] = powers[index] * powers[index];
    }

    fast_math_internal::parallel_for_dynamic(
        chunk_count, thread_count, [&](std::size_t chunk_index) {
        const auto begin =
            static_cast<std::size_t>(chunk_index * chunk_size);
        const auto end = std::min(
            sample_count,
            static_cast<std::size_t>(begin + chunk_size));
        auto& output = chunks[chunk_index];
        auto sample = begin;
#if defined(__aarch64__)
        if (power_count <= kMaximumSimdPowerCount) {
          std::array<SimdMoments, kMaximumSimdPowerCount>
              accumulators{};
          auto maximum_modulus_squared = vdupq_n_f64(0.0);
          auto maximum_derivative_squared = vdupq_n_f64(0.0);
          const auto one = vdupq_n_f64(1.0);
          for (; sample + 1 < end; sample += 2) {
            const auto values =
                vld2q_f64(values_interleaved + 2 * sample);
            const auto derivatives =
                vld2q_f64(derivatives_interleaved + 2 * sample);
            auto modulus_squared =
                vmulq_f64(values.val[1], values.val[1]);
            modulus_squared = vfmaq_f64(
                modulus_squared, values.val[0], values.val[0]);
            auto derivative_squared =
                vmulq_f64(derivatives.val[1], derivatives.val[1]);
            derivative_squared = vfmaq_f64(
                derivative_squared,
                derivatives.val[0],
                derivatives.val[0]);
            auto radial_real = vmulq_f64(
                derivatives.val[1], values.val[1]);
            radial_real = vfmaq_f64(
                radial_real, derivatives.val[0], values.val[0]);
            auto radial_imag = vmulq_f64(
                derivatives.val[1], values.val[0]);
            radial_imag = vfmsq_f64(
                radial_imag, derivatives.val[0], values.val[1]);
            const auto radial_real_squared =
                vmulq_f64(radial_real, radial_real);
            maximum_modulus_squared = vmaxq_f64(
                maximum_modulus_squared, modulus_squared);
            maximum_derivative_squared = vmaxq_f64(
                maximum_derivative_squared, derivative_squared);

            auto weight = one;
            auto radial_weight = one;
            if (minimum_power >= 2) {
              for (std::uint32_t exponent = 0;
                   exponent < minimum_power - 2;
                   ++exponent) {
                radial_weight =
                    vmulq_f64(radial_weight, modulus_squared);
              }
              weight = vmulq_f64(
                  radial_weight, modulus_squared);
            } else {
              radial_weight = vdivq_f64(one, modulus_squared);
            }
            for (std::size_t index = 0;
                 index < power_count;
                 ++index) {
              auto& accumulator = accumulators[index];
              accumulator.value = vfmaq_f64(
                  accumulator.value, weight, modulus_squared);
              accumulator.ordinary = vfmaq_f64(
                  accumulator.ordinary,
                  vmulq_n_f64(weight, power_squares[index]),
                  derivative_squared);
              accumulator.phase_current = vfmaq_f64(
                  accumulator.phase_current,
                  vmulq_n_f64(weight, powers[index]),
                  radial_imag);
              accumulator.radial = vfmaq_f64(
                  accumulator.radial,
                  vmulq_n_f64(
                      radial_weight, power_squares[index]),
                  radial_real_squared);
              weight = vmulq_f64(weight, modulus_squared);
              radial_weight =
                  vmulq_f64(radial_weight, modulus_squared);
            }
          }
          output.maximum_modulus_squared = std::max(
              vgetq_lane_f64(maximum_modulus_squared, 0),
              vgetq_lane_f64(maximum_modulus_squared, 1));
          output.maximum_derivative_squared = std::max(
              vgetq_lane_f64(maximum_derivative_squared, 0),
              vgetq_lane_f64(maximum_derivative_squared, 1));
          for (std::size_t index = 0;
               index < power_count;
               ++index) {
            output.value[index] =
                vaddvq_f64(accumulators[index].value);
            output.ordinary[index] =
                vaddvq_f64(accumulators[index].ordinary);
            output.phase_current[index] =
                vaddvq_f64(accumulators[index].phase_current);
            output.radial[index] =
                vaddvq_f64(accumulators[index].radial);
          }
        }
#endif
        for (; sample < end; ++sample) {
          const double value_real = values_interleaved[2 * sample];
          const double value_imag = values_interleaved[2 * sample + 1];
          const double derivative_real =
              derivatives_interleaved[2 * sample];
          const double derivative_imag =
              derivatives_interleaved[2 * sample + 1];
          const double modulus_squared =
              value_real * value_real + value_imag * value_imag;
          const double derivative_squared =
              derivative_real * derivative_real +
              derivative_imag * derivative_imag;
          const double radial_real =
              derivative_real * value_real +
              derivative_imag * value_imag;
          const double radial_imag =
              derivative_imag * value_real -
              derivative_real * value_imag;
          const double radial_real_squared =
              radial_real * radial_real;
          output.maximum_modulus_squared = std::max(
              output.maximum_modulus_squared, modulus_squared);
          output.maximum_derivative_squared = std::max(
              output.maximum_derivative_squared, derivative_squared);

          double weight = 1.0;
          double radial_weight = 1.0 / modulus_squared;
          if (minimum_power >= 2) {
            radial_weight = integer_power(
                modulus_squared, minimum_power - 2);
            weight = radial_weight * modulus_squared;
          }
          for (std::size_t index = 0; index < power_count; ++index) {
            output.value[index] = std::fma(
                weight, modulus_squared, output.value[index]);
            output.ordinary[index] = std::fma(
                power_squares[index] * weight,
                derivative_squared,
                output.ordinary[index]);
            output.phase_current[index] = std::fma(
                powers[index] * weight,
                radial_imag,
                output.phase_current[index]);
            output.radial[index] = std::fma(
                power_squares[index] * radial_weight,
                radial_real_squared,
                output.radial[index]);
            weight *= modulus_squared;
            radial_weight *= modulus_squared;
          }
        }
        });

    for (std::size_t power_index = 0;
         power_index < power_count;
         ++power_index) {
      moments[power_index] = {
          static_cast<std::uint32_t>(minimum_power + power_index),
          0.0,
          0.0,
          0.0,
          0.0,
      };
    }
    for (const auto& chunk : chunks) {
      stats->maximum_modulus = std::max(
          stats->maximum_modulus, chunk.maximum_modulus_squared);
      stats->maximum_derivative = std::max(
          stats->maximum_derivative,
          chunk.maximum_derivative_squared);
      for (std::size_t power_index = 0;
           power_index < power_count;
           ++power_index) {
        moments[power_index].value +=
            mesh_step * chunk.value[power_index];
        moments[power_index].ordinary +=
            mesh_step * chunk.ordinary[power_index];
        moments[power_index].phase_current +=
            mesh_step * chunk.phase_current[power_index];
        moments[power_index].radial +=
            mesh_step * chunk.radial[power_index];
      }
    }
    stats->maximum_modulus = std::sqrt(stats->maximum_modulus);
    stats->maximum_derivative = std::sqrt(stats->maximum_derivative);

    stats->sample_count = sample_count;
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
