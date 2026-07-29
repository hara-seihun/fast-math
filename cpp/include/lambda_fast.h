#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define LAMBDA_FAST_API __declspec(dllexport)
#else
#define LAMBDA_FAST_API
#endif

extern "C" {

struct lambda_fast_stats {
  std::uint64_t primary_pairs;
  std::uint64_t transformed_pairs;
  std::uint64_t low_pairs;
  double elapsed_seconds;
};

struct lambda_fast_two_level_record {
  std::uint64_t left;
  std::uint64_t right;
  std::uint64_t fine_piece_count;
  double first_real;
  double first_imag;
  double second_real;
  double second_imag;
  double center_cost;
  double weight_variation_upper;
  double fine_phase_drift_upper;
  double two_level_upper;
};

struct lambda_fast_two_level_stats {
  std::uint64_t primary_pairs;
  std::uint64_t transformed_pairs;
  std::uint64_t low_pairs;
  std::uint64_t fine_weight_block_count;
  std::uint64_t fine_piece_count;
  std::uint64_t outer_block_count;
  double constant_common_error;
  double constant_low_error;
  double center_cost;
  double weight_variation_upper;
  double fine_phase_drift_upper;
  double common_weighted_l1_upper;
  double low_weighted_l1_upper;
  double weighted_l1_upper;
  double two_level_upper;
  double elapsed_seconds;
};

struct lambda_fast_power_moment {
  std::uint32_t power;
  double value;
  double ordinary;
  double phase_current;
  double radial;
};

struct lambda_fast_power_moment_stats {
  std::uint64_t sample_count;
  double maximum_modulus;
  double maximum_derivative;
  double elapsed_seconds;
};

struct lambda_fast_inverse_stats {
  std::uint64_t update_count;
  double elapsed_seconds;
};

LAMBDA_FAST_API const char* lambda_fast_version();

LAMBDA_FAST_API int lambda_fast_accumulate_f64(
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
    std::size_t error_message_size);

LAMBDA_FAST_API int lambda_fast_fused_two_level_f64(
    const double* inverse,
    std::size_t inverse_size,
    const double* primary,
    std::size_t primary_size,
    const double* transformed_interleaved,
    std::size_t transformed_size,
    std::uint64_t transformed_first,
    const double* low,
    std::size_t low_size,
    const std::uint64_t* weight_left,
    const std::uint64_t* weight_right,
    const double* weight_lower,
    const double* weight_upper,
    std::size_t weight_count,
    std::uint64_t output_limit,
    double gamma_abs,
    double sigma,
    double q_primary_real,
    double q_primary_imag,
    double q_dual_real,
    double q_dual_imag,
    double outer_ratio,
    std::uint64_t target_tile_size,
    std::uint32_t thread_count,
    lambda_fast_two_level_record* records,
    std::size_t record_capacity,
    lambda_fast_two_level_stats* stats,
    char* error_message,
    std::size_t error_message_size);

LAMBDA_FAST_API int lambda_fast_power_moments_f64(
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
    std::size_t error_message_size);

LAMBDA_FAST_API int lambda_fast_dirichlet_inverse_f64(
    const double* source,
    std::size_t source_size,
    double* coefficients,
    lambda_fast_inverse_stats* stats,
    char* error_message,
    std::size_t error_message_size);

}
