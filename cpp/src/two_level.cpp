#include "fast_math.h"
#include "parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct OuterBlock {
  std::uint64_t left;
  std::uint64_t right;
};

struct FinePiece {
  std::uint64_t left;
  std::uint64_t right;
  double lower;
  double upper;
  std::size_t outer_index;
};

struct WorkChunk {
  std::size_t piece_begin;
  std::size_t piece_end;
  std::uint64_t left;
  std::uint64_t right;
};

struct PieceAggregate {
  double common_real = 0.0;
  double common_imag = 0.0;
  double low = 0.0;
  double common_l1 = 0.0;
  double low_l1 = 0.0;
};

struct OuterAggregate {
  std::complex<double> first = 0.0;
  std::complex<double> second = 0.0;
  std::uint64_t fine_piece_count = 0;
  double weight_variation = 0.0;
  double phase_drift = 0.0;
};

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

std::vector<OuterBlock> multiplicative_blocks(
    std::uint64_t first,
    std::uint64_t last,
    double ratio) {
  std::vector<OuterBlock> blocks;
  auto left = first;
  while (left <= last) {
    const auto scaled = static_cast<std::uint64_t>(
        std::floor(static_cast<double>(left) * ratio));
    const auto right = std::min(last, std::max(left, scaled));
    blocks.push_back({left, right});
    if (right == std::numeric_limits<std::uint64_t>::max()) {
      break;
    }
    left = right + 1;
  }
  return blocks;
}

void validate_source_pointer(
    const double* pointer,
    std::size_t size,
    const char* name) {
  if (size != 0 && pointer == nullptr) {
    throw std::invalid_argument(std::string(name) + " pointer is null");
  }
}

}  // namespace

extern "C" {

int lambda_fast_fused_two_level_f64(
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
    std::size_t error_message_size) {
  try {
    if (inverse == nullptr || inverse_size == 0) {
      throw std::invalid_argument("inverse must contain index zero");
    }
    validate_source_pointer(primary, primary_size, "primary");
    validate_source_pointer(
        transformed_interleaved, transformed_size, "transformed");
    validate_source_pointer(low, low_size, "low");
    if (transformed_first == 0) {
      throw std::invalid_argument("transformed_first must be positive");
    }
    if (output_limit < 2) {
      throw std::invalid_argument("output_limit must be at least two");
    }
    if (weight_count == 0 || weight_left == nullptr ||
        weight_right == nullptr || weight_lower == nullptr ||
        weight_upper == nullptr) {
      throw std::invalid_argument("weight intervals are required");
    }
    if (weight_left[0] != 2 ||
        weight_right[weight_count - 1] != output_limit) {
      throw std::invalid_argument(
          "weight intervals must cover outputs two through output_limit");
    }
    for (std::size_t index = 0; index < weight_count; ++index) {
      if (weight_right[index] < weight_left[index] ||
          weight_lower[index] <= 0.0 ||
          weight_upper[index] < weight_lower[index]) {
        throw std::invalid_argument("invalid weight interval");
      }
      if (index != 0 && weight_left[index] != weight_right[index - 1] + 1) {
        throw std::invalid_argument("weight intervals are not contiguous");
      }
    }
    if (!(gamma_abs >= 0.0 && gamma_abs < 1.0)) {
      throw std::invalid_argument("gamma_abs must lie in [0, 1)");
    }
    if (!(outer_ratio > 1.0) || !std::isfinite(outer_ratio)) {
      throw std::invalid_argument("outer_ratio must be finite and above one");
    }
    if (target_tile_size == 0) {
      throw std::invalid_argument("target_tile_size must be positive");
    }
    if (stats == nullptr || records == nullptr) {
      throw std::invalid_argument("output pointers are null");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};

    const auto started = Clock::now();
    const auto outer_blocks =
        multiplicative_blocks(2, output_limit, outer_ratio);
    if (record_capacity < outer_blocks.size()) {
      throw std::invalid_argument("two-level record capacity is too small");
    }

    std::vector<FinePiece> pieces;
    pieces.reserve(weight_count + outer_blocks.size());
    std::size_t outer_index = 0;
    for (std::size_t index = 0; index < weight_count; ++index) {
      auto left = weight_left[index];
      const auto interval_right = weight_right[index];
      while (left <= interval_right) {
        while (left > outer_blocks[outer_index].right) {
          ++outer_index;
        }
        const auto right =
            std::min(interval_right, outer_blocks[outer_index].right);
        pieces.push_back(
            {left,
             right,
             weight_lower[index],
             weight_upper[index],
             outer_index});
        left = right + 1;
      }
    }

    std::vector<WorkChunk> chunks;
    for (std::size_t begin = 0; begin < pieces.size();) {
      auto end = begin + 1;
      const auto left = pieces[begin].left;
      auto right = pieces[begin].right;
      while (end < pieces.size()) {
        const auto candidate_right = pieces[end].right;
        if (candidate_right - left + 1 > target_tile_size) {
          break;
        }
        right = candidate_right;
        ++end;
      }
      chunks.push_back({begin, end, left, right});
      begin = end;
    }

    std::vector<PieceAggregate> piece_aggregates(pieces.size());
    const auto workers = fast_math_internal::parallel_worker_count(
        chunks.size(), thread_count);
    std::vector<std::vector<double>> common_real_by_worker(workers);
    std::vector<std::vector<double>> common_imag_by_worker(workers);
    std::vector<std::vector<double>> low_output_by_worker(workers);

    fast_math_internal::parallel_for_dynamic_indexed(
        chunks.size(),
        thread_count,
        [&](std::size_t chunk_index, std::size_t worker_index) {
        auto& common_real = common_real_by_worker[worker_index];
        auto& common_imag = common_imag_by_worker[worker_index];
        auto& low_output = low_output_by_worker[worker_index];
        const auto& chunk = chunks[chunk_index];
        const auto width =
            static_cast<std::size_t>(chunk.right - chunk.left + 1);
        common_real.assign(width, 0.0);
        common_imag.assign(width, 0.0);
        low_output.assign(width, 0.0);

        for (std::uint64_t divisor = 1; divisor < inverse_size; ++divisor) {
          const double coefficient = inverse[divisor];
          if (coefficient == 0.0) {
            continue;
          }

          const auto source_first =
              std::max<std::uint64_t>(1, ceil_div(chunk.left, divisor));
          const auto primary_last = std::min<std::uint64_t>(
              primary_size, chunk.right / divisor);
          const auto low_last =
              std::min<std::uint64_t>(low_size, chunk.right / divisor);
          const auto shared_last = std::min(primary_last, low_last);
          for (auto index = source_first; index <= shared_last; ++index) {
            const auto output = divisor * index;
            const auto local =
                static_cast<std::size_t>(output - chunk.left);
            common_real[local] += coefficient * primary[index - 1];
            low_output[local] += coefficient * low[index - 1];
          }
          for (auto index = std::max(source_first, shared_last + 1);
               index <= primary_last;
               ++index) {
            const auto output = divisor * index;
            const auto local =
                static_cast<std::size_t>(output - chunk.left);
            common_real[local] += coefficient * primary[index - 1];
          }
          for (auto index = std::max(source_first, shared_last + 1);
               index <= low_last;
               ++index) {
            const auto output = divisor * index;
            const auto local =
                static_cast<std::size_t>(output - chunk.left);
            low_output[local] += coefficient * low[index - 1];
          }

          if (transformed_size != 0) {
            const auto transformed_last_source =
                transformed_first +
                static_cast<std::uint64_t>(transformed_size) - 1;
            const auto transformed_begin = std::max(
                transformed_first, ceil_div(chunk.left, divisor));
            const auto transformed_end = std::min(
                transformed_last_source, chunk.right / divisor);
            for (auto index = transformed_begin;
                 index <= transformed_end;
                 ++index) {
              const auto output = divisor * index;
              const auto local =
                  static_cast<std::size_t>(output - chunk.left);
              const auto source =
                  static_cast<std::size_t>(index - transformed_first);
              common_real[local] +=
                  coefficient * transformed_interleaved[2 * source];
              common_imag[local] +=
                  coefficient * transformed_interleaved[2 * source + 1];
            }
          }

        }

        for (auto piece_index = chunk.piece_begin;
             piece_index < chunk.piece_end;
             ++piece_index) {
          const auto& piece = pieces[piece_index];
          auto& aggregate = piece_aggregates[piece_index];
          const auto begin =
              static_cast<std::size_t>(piece.left - chunk.left);
          const auto end =
              static_cast<std::size_t>(piece.right - chunk.left + 1);
          for (auto local = begin; local < end; ++local) {
            const auto real = common_real[local];
            const auto imag = common_imag[local];
            const auto low_value = low_output[local];
            aggregate.common_real += real;
            aggregate.common_imag += imag;
            aggregate.low += low_value;
            aggregate.common_l1 += std::hypot(real, imag);
            aggregate.low_l1 += std::abs(low_value);
          }
        }
        });

    const auto primary_exponent =
        std::complex<double>(q_primary_real - sigma, q_primary_imag);
    const auto dual_exponent =
        std::complex<double>(q_dual_real - sigma, q_dual_imag);
    const double denominator = 1.0 + gamma_abs;
    std::vector<OuterAggregate> outer_aggregates(outer_blocks.size());

    for (std::size_t index = 0; index < pieces.size(); ++index) {
      const auto& piece = pieces[index];
      const auto& aggregate = piece_aggregates[index];
      auto& outer = outer_aggregates[piece.outer_index];
      const double midpoint = 0.5 * (piece.lower + piece.upper);
      const double radius = 0.5 * (piece.upper - piece.lower);
      const auto common_sum =
          midpoint *
          std::complex<double>(
              aggregate.common_real, aggregate.common_imag);
      const auto low_sum =
          std::complex<double>(midpoint * aggregate.low, 0.0);
      const double log_left =
          std::log(static_cast<double>(piece.left));
      const auto primary_anchor =
          std::exp(-primary_exponent * log_left);
      const auto dual_anchor = std::exp(-dual_exponent * log_left);
      outer.first +=
          primary_anchor * common_sum -
          gamma_abs * dual_anchor * low_sum;
      outer.second +=
          dual_anchor * low_sum -
          gamma_abs * primary_anchor * common_sum;
      outer.weight_variation +=
          radius * (aggregate.common_l1 + aggregate.low_l1);

      const double log_ratio = std::log(
          static_cast<double>(piece.right) /
          static_cast<double>(piece.left));
      const double primary_chord =
          std::abs(std::exp(-primary_exponent * log_ratio) - 1.0);
      const double dual_chord =
          std::abs(std::exp(-dual_exponent * log_ratio) - 1.0);
      outer.phase_drift +=
          primary_chord * piece.upper * aggregate.common_l1 +
          dual_chord * piece.upper * aggregate.low_l1;
      outer.fine_piece_count += 1;
      stats->common_weighted_l1_upper +=
          piece.upper * aggregate.common_l1;
      stats->low_weighted_l1_upper +=
          piece.upper * aggregate.low_l1;
    }

    for (std::size_t index = 0; index < outer_blocks.size(); ++index) {
      const auto& block = outer_blocks[index];
      const auto& aggregate = outer_aggregates[index];
      const double center_cost =
          (std::abs(aggregate.first) + std::abs(aggregate.second)) /
          denominator;
      records[index] = {
          block.left,
          block.right,
          aggregate.fine_piece_count,
          aggregate.first.real(),
          aggregate.first.imag(),
          aggregate.second.real(),
          aggregate.second.imag(),
          center_cost,
          aggregate.weight_variation,
          aggregate.phase_drift,
          center_cost + aggregate.weight_variation + aggregate.phase_drift,
      };
      stats->center_cost += center_cost;
      stats->weight_variation_upper += aggregate.weight_variation;
      stats->fine_phase_drift_upper += aggregate.phase_drift;
    }

    std::complex<double> constant_common = 0.0;
    double constant_low = 0.0;
    if (inverse_size > 1) {
      const double coefficient = inverse[1];
      if (primary_size > 0) {
        constant_common += coefficient * primary[0];
      }
      if (transformed_size > 0 && transformed_first == 1) {
        constant_common += coefficient * std::complex<double>(
            transformed_interleaved[0], transformed_interleaved[1]);
      }
      if (low_size > 0) {
        constant_low += coefficient * low[0];
      }
    }

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
    stats->fine_weight_block_count = weight_count;
    stats->fine_piece_count = pieces.size();
    stats->outer_block_count = outer_blocks.size();
    stats->constant_common_error = std::abs(constant_common - 1.0);
    stats->constant_low_error = std::abs(constant_low - gamma_abs);
    stats->weighted_l1_upper =
        stats->common_weighted_l1_upper + stats->low_weighted_l1_upper;
    stats->two_level_upper =
        stats->center_cost +
        stats->weight_variation_upper +
        stats->fine_phase_drift_upper;
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
