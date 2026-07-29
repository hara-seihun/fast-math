#include "fast_math.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Matrix {
  std::size_t row_count;
  std::size_t column_count;
  std::vector<std::uint64_t> row_offsets;
  std::vector<std::uint32_t> columns;
  std::vector<std::uint32_t> values;
};

struct Triplet {
  std::uint64_t row;
  std::uint32_t column;
  std::uint32_t value;
};

Matrix read_sms(
    const std::string& path,
    std::uint32_t prime) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open matrix: " + path);
  }
  Matrix matrix{};
  char format = '\0';
  if (!(input >> matrix.row_count >> matrix.column_count >> format) ||
      matrix.row_count == 0 || matrix.column_count == 0 || format != 'M') {
    throw std::runtime_error("invalid SMS header");
  }
  std::vector<Triplet> triplets;
  while (true) {
    std::uint64_t row = 0;
    std::uint64_t column = 0;
    std::int64_t value = 0;
    if (!(input >> row >> column >> value)) {
      throw std::runtime_error("truncated SMS matrix");
    }
    if (row == 0 && column == 0 && value == 0) {
      break;
    }
    if (row == 0 || row > matrix.row_count ||
        column == 0 || column > matrix.column_count) {
      throw std::runtime_error("invalid SMS entry");
    }
    auto reduced = value % static_cast<std::int64_t>(prime);
    if (reduced < 0) {
      reduced += prime;
    }
    if (reduced != 0) {
      triplets.push_back({
          row - 1,
          static_cast<std::uint32_t>(column - 1),
          static_cast<std::uint32_t>(reduced),
      });
    }
  }
  std::sort(
      triplets.begin(),
      triplets.end(),
      [](const Triplet& left, const Triplet& right) {
        return left.row < right.row ||
            (left.row == right.row && left.column < right.column);
      });

  std::vector<Triplet> canonical;
  canonical.reserve(triplets.size());
  for (const auto& triplet : triplets) {
    if (!canonical.empty() &&
        canonical.back().row == triplet.row &&
        canonical.back().column == triplet.column) {
      canonical.back().value = static_cast<std::uint32_t>(
          (static_cast<std::uint64_t>(canonical.back().value) +
           triplet.value) %
          prime);
    } else {
      canonical.push_back(triplet);
    }
  }

  matrix.row_offsets.assign(matrix.row_count + 1, 0);
  std::size_t current_row = 0;
  for (const auto& triplet : canonical) {
    while (current_row < triplet.row) {
      matrix.row_offsets[++current_row] = matrix.columns.size();
    }
    if (triplet.value != 0) {
      matrix.columns.push_back(triplet.column);
      matrix.values.push_back(triplet.value);
    }
  }
  while (current_row < matrix.row_count) {
    matrix.row_offsets[++current_row] = matrix.columns.size();
  }
  return matrix;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr
        << "usage: " << argv[0] << " MATRIX.sms PRIME TARGET_RANK\n";
    return 2;
  }
  try {
    const auto parse_started = Clock::now();
    const auto prime = static_cast<std::uint32_t>(std::stoul(argv[2]));
    const auto target_rank =
        static_cast<std::size_t>(std::stoull(argv[3]));
    const auto matrix = read_sms(argv[1], prime);
    const auto parse_seconds =
        std::chrono::duration<double>(Clock::now() - parse_started).count();
    const auto pivot_capacity = target_rank == 0
        ? std::min(matrix.row_count, matrix.column_count)
        : target_rank;
    std::vector<std::uint64_t> pivot_rows(pivot_capacity);
    std::vector<std::uint32_t> pivot_columns(pivot_capacity);
    fast_math_sparse_rank_stats stats{};
    char error[512]{};
    const int status = fast_math_sparse_rank_mod_u32(
        matrix.row_offsets.data(),
        matrix.columns.data(),
        matrix.values.data(),
        matrix.row_count,
        matrix.column_count,
        matrix.columns.size(),
        prime,
        target_rank,
        pivot_rows.data(),
        pivot_columns.data(),
        pivot_capacity,
        &stats,
        error,
        sizeof(error));
    if (status != 0) {
      throw std::runtime_error(
          "fast-math sparse rank failed: " + std::string(error));
    }
    std::uint64_t pivot_digest = 1469598103934665603ULL;
    for (std::size_t index = 0; index < stats.rank; ++index) {
      for (unsigned shift = 0; shift < 64; shift += 8) {
        pivot_digest ^=
            static_cast<std::uint8_t>(pivot_rows[index] >> shift);
        pivot_digest *= 1099511628211ULL;
      }
      for (unsigned shift = 0; shift < 32; shift += 8) {
        pivot_digest ^=
            static_cast<std::uint8_t>(pivot_columns[index] >> shift);
        pivot_digest *= 1099511628211ULL;
      }
    }
    std::cout
        << "{\"benchmark\":\"sparse_modular_rank\","
        << "\"rows\":" << stats.row_count << ','
        << "\"columns\":" << stats.column_count << ','
        << "\"input_nonzeros\":" << stats.input_nonzeros << ','
        << "\"prime\":" << stats.prime << ','
        << "\"target_rank\":" << target_rank << ','
        << "\"rank\":" << stats.rank << ','
        << "\"pivot_digest\":" << pivot_digest << ','
        << "\"active_rows\":" << stats.active_rows << ','
        << "\"processed_rows\":" << stats.processed_rows << ','
        << "\"dependent_rows\":" << stats.dependent_rows << ','
        << "\"peeled_pivots\":" << stats.peeled_pivots << ','
        << "\"residual_rows\":" << stats.residual_rows << ','
        << "\"residual_columns\":" << stats.residual_columns << ','
        << "\"residual_nonzeros\":" << stats.residual_nonzeros << ','
        << "\"elimination_steps\":" << stats.elimination_steps << ','
        << "\"basis_nonzeros\":" << stats.basis_nonzeros << ','
        << "\"maximum_basis_size\":" << stats.maximum_basis_size << ','
        << "\"maximum_working_size\":" << stats.maximum_working_size << ','
        << "\"target_reached\":"
        << (stats.target_reached ? "true" : "false") << ','
        << "\"parse_seconds\":" << parse_seconds << ','
        << "\"preprocessing_seconds\":"
        << stats.preprocessing_seconds << ','
        << "\"wall_seconds\":" << stats.elapsed_seconds
        << "}\n";
    return stats.target_reached ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "sparse rank benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
