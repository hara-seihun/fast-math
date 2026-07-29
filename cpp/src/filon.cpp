#include "fast_math.h"
#include "parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kPhaseResetInterval = 256;

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

struct CompensatedSum {
  double sum = 0.0;
  double correction = 0.0;

  void add(double value) {
    const double next = sum + value;
    if (std::abs(sum) >= std::abs(value)) {
      correction += (sum - next) + value;
    } else {
      correction += (value - next) + sum;
    }
    sum = next;
  }

  double value() const {
    return sum + correction;
  }
};

struct alignas(64) ComplexPartial {
  double real = 0.0;
  double imag = 0.0;
};

void add_weighted_pair(
    const double* correlation,
    std::size_t correlation_count,
    std::size_t lag,
    double weight_real,
    double weight_imag,
    bool conjugate_kernel,
    CompensatedSum& real,
    CompensatedSum& imag) {
  const double positive_real = correlation[2 * lag];
  const double positive_imag = correlation[2 * lag + 1];
  if (lag == 0) {
    if (conjugate_kernel) {
      weight_imag = -weight_imag;
    }
    real.add(std::fma(
        positive_real,
        weight_real,
        -positive_imag * weight_imag));
    imag.add(std::fma(
        positive_real,
        weight_imag,
        positive_imag * weight_real));
    return;
  }

  const auto negative_index = correlation_count - lag;
  const double negative_real = correlation[2 * negative_index];
  const double negative_imag = correlation[2 * negative_index + 1];
  if (conjugate_kernel) {
    weight_imag = -weight_imag;
  }

  // cp*w + cn*conj(w), with w conjugated first when requested.
  real.add(std::fma(
      positive_real + negative_real,
      weight_real,
      (negative_imag - positive_imag) * weight_imag));
  imag.add(std::fma(
      positive_imag + negative_imag,
      weight_real,
      (positive_real - negative_real) * weight_imag));
}

void asymptotic_weight(
    double phase_cosine,
    double phase_sine,
    double inverse_frequency,
    const std::vector<double>& endpoint_difference,
    const std::vector<double>& endpoint_sum,
    double length_scale,
    double& weight_real,
    double& weight_imag) {
  double inverse_power = inverse_frequency;
  weight_real = 0.0;
  weight_imag = 0.0;
  for (std::size_t order = 0;
       order < endpoint_difference.size();
       ++order) {
    const double boundary_real =
        endpoint_difference[order] * phase_cosine;
    const double boundary_imag =
        endpoint_sum[order] * phase_sine;
    switch (order & 3u) {
      case 0:
        weight_real =
            std::fma(boundary_imag, inverse_power, weight_real);
        weight_imag =
            std::fma(-boundary_real, inverse_power, weight_imag);
        break;
      case 1:
        weight_real =
            std::fma(boundary_real, inverse_power, weight_real);
        weight_imag =
            std::fma(boundary_imag, inverse_power, weight_imag);
        break;
      case 2:
        weight_real =
            std::fma(-boundary_imag, inverse_power, weight_real);
        weight_imag =
            std::fma(boundary_real, inverse_power, weight_imag);
        break;
      default:
        weight_real =
            std::fma(-boundary_real, inverse_power, weight_real);
        weight_imag =
            std::fma(-boundary_imag, inverse_power, weight_imag);
        break;
    }
    inverse_power *= inverse_frequency;
  }
  weight_real *= length_scale;
  weight_imag *= length_scale;
}

#if defined(__clang__) || defined(__GNUC__)
using Double2 = double __attribute__((vector_size(16)));

void asymptotic_weight_pair(
    double phase_cosine0,
    double phase_sine0,
    double phase_cosine1,
    double phase_sine1,
    double inverse_frequency0,
    double inverse_frequency1,
    const std::vector<double>& endpoint_difference,
    const std::vector<double>& endpoint_sum,
    double length_scale,
    double* weight_real,
    double* weight_imag) {
  const Double2 phase_cosine = {phase_cosine0, phase_cosine1};
  const Double2 phase_sine = {phase_sine0, phase_sine1};
  const Double2 inverse_frequency = {
      inverse_frequency0, inverse_frequency1};
  Double2 inverse_power = inverse_frequency;
  Double2 real = {0.0, 0.0};
  Double2 imag = {0.0, 0.0};
  for (std::size_t order = 0;
       order < endpoint_difference.size();
       ++order) {
    const Double2 boundary_real =
        endpoint_difference[order] * phase_cosine;
    const Double2 boundary_imag =
        endpoint_sum[order] * phase_sine;
    switch (order & 3u) {
      case 0:
        real += boundary_imag * inverse_power;
        imag -= boundary_real * inverse_power;
        break;
      case 1:
        real += boundary_real * inverse_power;
        imag += boundary_imag * inverse_power;
        break;
      case 2:
        real -= boundary_imag * inverse_power;
        imag += boundary_real * inverse_power;
        break;
      default:
        real -= boundary_real * inverse_power;
        imag -= boundary_imag * inverse_power;
        break;
    }
    inverse_power *= inverse_frequency;
  }
  const Double2 scale = {length_scale, length_scale};
  real *= scale;
  imag *= scale;
  weight_real[0] = real[0];
  weight_real[1] = real[1];
  weight_imag[0] = imag[0];
  weight_imag[1] = imag[1];
}
#endif

}  // namespace

extern "C" {

int fast_math_filon_chebyshev_inner_product_f64(
    const double* correlation_interleaved,
    std::size_t correlation_count,
    const double* exact_weights_interleaved,
    std::size_t exact_count,
    const double* positive_endpoint_derivatives,
    const double* negative_endpoint_derivatives,
    std::uint32_t term_count,
    std::size_t output_count,
    double eta,
    double length,
    bool conjugate_kernel,
    std::uint64_t chunk_size,
    std::uint32_t thread_count,
    double* result_interleaved,
    fast_math_filon_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (correlation_interleaved == nullptr ||
        result_interleaved == nullptr || stats == nullptr) {
      throw std::invalid_argument("Filon pointer is null");
    }
    if (output_count == 0 || exact_count == 0) {
      throw std::invalid_argument(
          "Filon output and exact prefix must be nonempty");
    }
    if (exact_count > output_count) {
      throw std::invalid_argument(
          "Filon exact prefix exceeds output count");
    }
    if (output_count >
        std::numeric_limits<std::size_t>::max() / 2 + 1) {
      throw std::overflow_error("Filon output count overflows");
    }
    const auto required_correlation_count = 2 * output_count - 1;
    if (correlation_count < required_correlation_count) {
      throw std::invalid_argument(
          "Filon correlation does not contain both lag directions");
    }
    if (exact_weights_interleaved == nullptr) {
      throw std::invalid_argument("Filon exact prefix pointer is null");
    }
    if (exact_count < output_count &&
        (term_count == 0 ||
         positive_endpoint_derivatives == nullptr ||
         negative_endpoint_derivatives == nullptr)) {
      throw std::invalid_argument(
          "Filon tail derivatives must be nonempty");
    }
    if (!std::isfinite(eta) || eta == 0.0 ||
        !std::isfinite(length)) {
      throw std::invalid_argument(
          "Filon eta and length must be finite with nonzero eta");
    }
    if (chunk_size == 0) {
      throw std::invalid_argument("Filon chunk size must be positive");
    }
    if (chunk_size > std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error("Filon chunk size exceeds addressable memory");
    }

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const auto native_chunk_size = static_cast<std::size_t>(chunk_size);
    const auto chunk_count =
        output_count / native_chunk_size +
        (output_count % native_chunk_size != 0);
    const auto workers = fast_math_internal::parallel_worker_count(
        chunk_count, thread_count);
    std::vector<double> endpoint_difference(term_count);
    std::vector<double> endpoint_sum(term_count);
    for (std::size_t order = 0; order < term_count; ++order) {
      endpoint_difference[order] =
          positive_endpoint_derivatives[order] -
          negative_endpoint_derivatives[order];
      endpoint_sum[order] =
          positive_endpoint_derivatives[order] +
          negative_endpoint_derivatives[order];
    }

    std::vector<ComplexPartial> partials(chunk_count);
    const double angular_step = eta / 2.0;
    const double phase_step_cosine = std::cos(angular_step);
    const double phase_step_sine = std::sin(angular_step);
    const double length_scale = length / 2.0;
    fast_math_internal::parallel_for_static(
        chunk_count, thread_count, [&](std::size_t chunk) {
          const auto begin = chunk * native_chunk_size;
          const auto end = begin + std::min(
              native_chunk_size, output_count - begin);
          const auto exact_end = std::min(end, exact_count);
          CompensatedSum real;
          CompensatedSum imag;
          for (auto lag = begin; lag < exact_end; ++lag) {
            add_weighted_pair(
                correlation_interleaved,
                correlation_count,
                lag,
                exact_weights_interleaved[2 * lag],
                exact_weights_interleaved[2 * lag + 1],
                conjugate_kernel,
                real,
                imag);
          }

          const auto tail_begin = std::max(begin, exact_count);
          if (tail_begin < end) {
            double phase_cosine =
                std::cos(angular_step * static_cast<double>(tail_begin));
            double phase_sine =
                std::sin(angular_step * static_cast<double>(tail_begin));
            auto lag = tail_begin;
#if defined(__clang__) || defined(__GNUC__)
            for (; lag + 1 < end; lag += 2) {
              const auto second_lag = lag + 1;
              double second_cosine;
              double second_sine;
              if (second_lag % kPhaseResetInterval == 0) {
                second_cosine = std::cos(
                    angular_step * static_cast<double>(second_lag));
                second_sine = std::sin(
                    angular_step * static_cast<double>(second_lag));
              } else {
                second_cosine = std::fma(
                    phase_cosine,
                    phase_step_cosine,
                    -phase_sine * phase_step_sine);
                second_sine = std::fma(
                    phase_sine,
                    phase_step_cosine,
                    phase_cosine * phase_step_sine);
              }
              const double inverse_frequency0 =
                  1.0 /
                  (angular_step * static_cast<double>(lag));
              const double inverse_frequency1 =
                  1.0 /
                  (angular_step * static_cast<double>(second_lag));
              double weight_real[2];
              double weight_imag[2];
              asymptotic_weight_pair(
                  phase_cosine,
                  phase_sine,
                  second_cosine,
                  second_sine,
                  inverse_frequency0,
                  inverse_frequency1,
                  endpoint_difference,
                  endpoint_sum,
                  length_scale,
                  weight_real,
                  weight_imag);
              add_weighted_pair(
                  correlation_interleaved,
                  correlation_count,
                  lag,
                  weight_real[0],
                  weight_imag[0],
                  conjugate_kernel,
                  real,
                  imag);
              add_weighted_pair(
                  correlation_interleaved,
                  correlation_count,
                  second_lag,
                  weight_real[1],
                  weight_imag[1],
                  conjugate_kernel,
                  real,
                  imag);

              const auto next_lag = lag + 2;
              if (next_lag < end) {
                if (next_lag % kPhaseResetInterval == 0) {
                  phase_cosine = std::cos(
                      angular_step * static_cast<double>(next_lag));
                  phase_sine = std::sin(
                      angular_step * static_cast<double>(next_lag));
                } else {
                  phase_cosine = std::fma(
                      second_cosine,
                      phase_step_cosine,
                      -second_sine * phase_step_sine);
                  phase_sine = std::fma(
                      second_sine,
                      phase_step_cosine,
                      second_cosine * phase_step_sine);
                }
              }
            }
#endif
            for (; lag < end; ++lag) {
              const double inverse_frequency =
                  1.0 / (angular_step * static_cast<double>(lag));
              double weight_real_scalar;
              double weight_imag_scalar;
              asymptotic_weight(
                  phase_cosine,
                  phase_sine,
                  inverse_frequency,
                  endpoint_difference,
                  endpoint_sum,
                  length_scale,
                  weight_real_scalar,
                  weight_imag_scalar);
              add_weighted_pair(
                  correlation_interleaved,
                  correlation_count,
                  lag,
                  weight_real_scalar,
                  weight_imag_scalar,
                  conjugate_kernel,
                  real,
                  imag);

              const auto next_lag = lag + 1;
              if (next_lag < end) {
                if (next_lag % kPhaseResetInterval == 0) {
                  phase_cosine = std::cos(
                      angular_step * static_cast<double>(next_lag));
                  phase_sine = std::sin(
                      angular_step * static_cast<double>(next_lag));
                } else {
                  const double next_cosine = std::fma(
                      phase_cosine,
                      phase_step_cosine,
                      -phase_sine * phase_step_sine);
                  phase_sine = std::fma(
                      phase_sine,
                      phase_step_cosine,
                      phase_cosine * phase_step_sine);
                  phase_cosine = next_cosine;
                }
              }
            }
          }
          partials[chunk] = {real.value(), imag.value()};
        });

    CompensatedSum result_real;
    CompensatedSum result_imag;
    for (const auto& partial : partials) {
      result_real.add(partial.real);
      result_imag.add(partial.imag);
    }
    result_interleaved[0] = result_real.value();
    result_interleaved[1] = result_imag.value();
    stats->correlation_count = correlation_count;
    stats->output_count = output_count;
    stats->exact_count = exact_count;
    stats->tail_count = output_count - exact_count;
    stats->chunk_count = chunk_count;
    stats->term_count = term_count;
    stats->thread_count = workers;
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
