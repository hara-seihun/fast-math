#include "fast_math.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
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
    triplets.push_back({
        row - 1,
        static_cast<std::uint32_t>(column - 1),
        static_cast<std::uint32_t>(reduced),
    });
  }
  std::sort(
      triplets.begin(),
      triplets.end(),
      [](const Triplet& left, const Triplet& right) {
        return left.row < right.row ||
            (left.row == right.row && left.column < right.column);
      });

  matrix.row_offsets.assign(matrix.row_count + 1, 0);
  std::size_t current_row = 0;
  for (const auto& triplet : triplets) {
    while (current_row < triplet.row) {
      matrix.row_offsets[++current_row] = matrix.columns.size();
    }
    if (!matrix.columns.empty() &&
        matrix.row_offsets[current_row] < matrix.columns.size() &&
        matrix.columns.back() == triplet.column) {
      throw std::runtime_error("duplicate SMS coordinate");
    }
    matrix.columns.push_back(triplet.column);
    matrix.values.push_back(triplet.value);
  }
  while (current_row < matrix.row_count) {
    matrix.row_offsets[++current_row] = matrix.columns.size();
  }
  return matrix;
}

void require_same_structure(
    const Matrix& left,
    const Matrix& right) {
  if (left.row_count != right.row_count ||
      left.column_count != right.column_count ||
      left.row_offsets != right.row_offsets ||
      left.columns != right.columns) {
    throw std::runtime_error(
        "prime-specific SMS reductions changed the CSR structure");
  }
}

void require_success(int status, const char* error) {
  if (status != 0) {
    throw std::runtime_error(
        "fast-math sparse rank failed: " + std::string(error));
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr
        << "usage: " << argv[0]
        << " MATRIX.sms TARGET_RANK THREADS PRIME PRIME [PRIME...]\n";
    return 2;
  }
  try {
    const auto target_rank =
        static_cast<std::size_t>(std::stoull(argv[2]));
    const auto threads =
        static_cast<std::uint32_t>(std::stoul(argv[3]));
    std::vector<std::uint32_t> primes;
    for (int index = 4; index < argc; ++index) {
      primes.push_back(
          static_cast<std::uint32_t>(std::stoul(argv[index])));
    }

    const auto parse_started = Clock::now();
    std::vector<Matrix> matrices;
    matrices.reserve(primes.size());
    for (const auto prime : primes) {
      matrices.push_back(read_sms(argv[1], prime));
      require_same_structure(matrices.front(), matrices.back());
    }
    const auto parse_seconds =
        std::chrono::duration<double>(Clock::now() - parse_started).count();
    const auto& matrix = matrices.front();
    const auto nonzero_count = matrix.columns.size();
    std::vector<std::uint32_t> values_by_prime;
    values_by_prime.reserve(primes.size() * nonzero_count);
    for (const auto& prime_matrix : matrices) {
      values_by_prime.insert(
          values_by_prime.end(),
          prime_matrix.values.begin(),
          prime_matrix.values.end());
    }

    const auto pivot_capacity =
        target_rank == 0
        ? std::min(matrix.row_count, matrix.column_count)
        : target_rank;
    std::vector<std::uint64_t> serial_pivot_rows(
        primes.size() * pivot_capacity);
    std::vector<std::uint32_t> serial_pivot_columns(
        primes.size() * pivot_capacity);
    std::vector<fast_math_sparse_rank_stats> serial_stats(primes.size());
    char error[512]{};
    double serial_seconds = 0;
    for (std::size_t task = 0; task < primes.size(); ++task) {
      require_success(
          fast_math_sparse_rank_mod_u32(
              matrix.row_offsets.data(),
              matrix.columns.data(),
              values_by_prime.data() + task * nonzero_count,
              matrix.row_count,
              matrix.column_count,
              nonzero_count,
              primes[task],
              target_rank,
              serial_pivot_rows.data() + task * pivot_capacity,
              serial_pivot_columns.data() + task * pivot_capacity,
              pivot_capacity,
              &serial_stats[task],
              error,
              sizeof(error)),
          error);
      serial_seconds += serial_stats[task].elapsed_seconds;
    }

    std::vector<std::uint64_t> batch_pivot_rows(
        primes.size() * pivot_capacity);
    std::vector<std::uint32_t> batch_pivot_columns(
        primes.size() * pivot_capacity);
    std::vector<fast_math_sparse_rank_stats> batch_stats(primes.size());
    fast_math_sparse_rank_batch_stats cold_stats{};
    fast_math_sparse_rank_batch_stats warm_stats{};
    require_success(
        fast_math_sparse_rank_mod_u32_batch(
            matrix.row_offsets.data(),
            matrix.columns.data(),
            values_by_prime.data(),
            matrix.row_count,
            matrix.column_count,
            nonzero_count,
            primes.data(),
            primes.size(),
            target_rank,
            threads,
            batch_pivot_rows.data(),
            batch_pivot_columns.data(),
            pivot_capacity,
            batch_stats.data(),
            &cold_stats,
            error,
            sizeof(error)),
        error);
    require_success(
        fast_math_sparse_rank_mod_u32_batch(
            matrix.row_offsets.data(),
            matrix.columns.data(),
            values_by_prime.data(),
            matrix.row_count,
            matrix.column_count,
            nonzero_count,
            primes.data(),
            primes.size(),
            target_rank,
            threads,
            batch_pivot_rows.data(),
            batch_pivot_columns.data(),
            pivot_capacity,
            batch_stats.data(),
            &warm_stats,
            error,
            sizeof(error)),
        error);

    for (std::size_t task = 0; task < primes.size(); ++task) {
      if (batch_stats[task].rank != serial_stats[task].rank ||
          batch_stats[task].elimination_steps !=
              serial_stats[task].elimination_steps ||
          batch_stats[task].peeled_pivots !=
              serial_stats[task].peeled_pivots ||
          batch_stats[task].residual_rows !=
              serial_stats[task].residual_rows ||
          batch_stats[task].residual_columns !=
              serial_stats[task].residual_columns ||
          batch_stats[task].residual_nonzeros !=
              serial_stats[task].residual_nonzeros ||
          !std::equal(
              serial_pivot_rows.begin() + task * pivot_capacity,
              serial_pivot_rows.begin() + task * pivot_capacity +
                  serial_stats[task].rank,
              batch_pivot_rows.begin() + task * pivot_capacity) ||
          !std::equal(
              serial_pivot_columns.begin() + task * pivot_capacity,
              serial_pivot_columns.begin() + task * pivot_capacity +
                  serial_stats[task].rank,
              batch_pivot_columns.begin() + task * pivot_capacity)) {
        throw std::runtime_error("serial/batch sparse-rank mismatch");
      }
    }

    std::cout
        << "{\"benchmark\":\"sparse_modular_rank_batch\","
        << "\"rows\":" << matrix.row_count << ','
        << "\"columns\":" << matrix.column_count << ','
        << "\"input_nonzeros\":" << nonzero_count << ','
        << "\"prime_count\":" << primes.size() << ','
        << "\"threads\":" << warm_stats.thread_count << ','
        << "\"target_rank\":" << target_rank << ','
        << "\"rank\":" << batch_stats.front().rank << ','
        << "\"peeled_pivots\":"
        << batch_stats.front().peeled_pivots << ','
        << "\"residual_rows\":"
        << batch_stats.front().residual_rows << ','
        << "\"residual_columns\":"
        << batch_stats.front().residual_columns << ','
        << "\"residual_nonzeros\":"
        << batch_stats.front().residual_nonzeros << ','
        << "\"serial_seconds\":" << serial_seconds << ','
        << "\"batch_cold_seconds\":" << cold_stats.elapsed_seconds << ','
        << "\"batch_warm_seconds\":" << warm_stats.elapsed_seconds << ','
        << "\"cold_speedup\":"
        << serial_seconds / cold_stats.elapsed_seconds << ','
        << "\"warm_speedup\":"
        << serial_seconds / warm_stats.elapsed_seconds << ','
        << "\"parse_seconds\":" << parse_seconds
        << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr
        << "sparse rank batch benchmark failed: "
        << error.what() << '\n';
    return 1;
  }
}
