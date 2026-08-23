#include "fast_math.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint32_t kPrime = 5;
constexpr std::uint32_t kWidth = 6;
using Row = std::array<std::uint32_t, kWidth>;

std::uint32_t hand_written_rank(
    const std::uint64_t* codes,
    std::size_t count) {
  std::vector<Row> basis;
  for (std::size_t index = 0; index < count; ++index) {
    auto code = codes[index];
    Row row{};
    for (auto& value : row) {
      value = static_cast<std::uint32_t>(code % kPrime);
      code /= kPrime;
    }
    for (const auto existing : basis) {
      std::size_t pivot = 0;
      while (pivot < kWidth && existing[pivot] == 0) {
        ++pivot;
      }
      if (pivot == kWidth || row[pivot] == 0) {
        continue;
      }
      std::uint32_t inverse = 1;
      while (existing[pivot] * inverse % kPrime != 1) {
        ++inverse;
      }
      const auto scale = row[pivot] * inverse % kPrime;
      for (std::size_t column = 0; column < kWidth; ++column) {
        row[column] =
            (row[column] + kPrime * kPrime - scale * existing[column]) %
            kPrime;
      }
    }
    std::size_t pivot = 0;
    while (pivot < kWidth && row[pivot] == 0) {
      ++pivot;
    }
    if (pivot == kWidth) {
      continue;
    }
    std::uint32_t inverse = 1;
    while (row[pivot] * inverse % kPrime != 1) {
      ++inverse;
    }
    for (auto& value : row) {
      value = value * inverse % kPrime;
    }
    for (auto& existing : basis) {
      if (existing[pivot] == 0) {
        continue;
      }
      const auto scale = existing[pivot];
      for (std::size_t column = 0; column < kWidth; ++column) {
        existing[column] =
            (existing[column] + kPrime * kPrime - scale * row[column]) %
            kPrime;
      }
    }
    basis.push_back(row);
    std::sort(
        basis.begin(), basis.end(), [](const Row& left, const Row& right) {
          const auto left_pivot = std::distance(
              left.begin(),
              std::find_if(left.begin(), left.end(), [](auto value) {
                return value != 0;
              }));
          const auto right_pivot = std::distance(
              right.begin(),
              std::find_if(right.begin(), right.end(), [](auto value) {
                return value != 0;
              }));
          return left_pivot < right_pivot;
        });
  }
  return static_cast<std::uint32_t>(basis.size());
}

void reference_ranks(
    const std::vector<std::uint64_t>& points,
    std::size_t points_per_span,
    std::vector<std::uint32_t>& ranks) {
  for (std::size_t span = 0; span < ranks.size(); ++span) {
    ranks[span] = hand_written_rank(
        points.data() + span * points_per_span, points_per_span);
  }
}

void native_ranks(
    const std::vector<std::uint64_t>& points,
    const std::vector<std::uint64_t>& offsets,
    std::vector<std::uint32_t>& ranks) {
  fast_math_fp_span_stats stats{};
  char error[512]{};
  const auto status = fast_math_fp_span_ranks_u64(
      points.data(),
      points.size(),
      offsets.data(),
      ranks.size(),
      kPrime,
      kWidth,
      ranks.data(),
      &stats,
      error,
      sizeof(error));
  if (status != 0) {
    throw std::runtime_error(error);
  }
}

template <typename Function>
double measure(Function&& function) {
  const auto started = Clock::now();
  function();
  return std::chrono::duration<double>(Clock::now() - started).count();
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void print_times(const std::vector<double>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << values[index];
  }
  std::cout << ']';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::size_t span_count =
        argc > 1 ? std::stoull(argv[1]) : 200000;
    const std::size_t points_per_span =
        argc > 2 ? std::stoull(argv[2]) : 8;
    const std::size_t repeats = argc > 3 ? std::stoull(argv[3]) : 5;
    if (span_count == 0 || points_per_span == 0 || repeats == 0) {
      throw std::invalid_argument("benchmark sizes must be positive");
    }

    std::mt19937_64 random(20260823);
    std::vector<std::uint64_t> points(span_count * points_per_span);
    for (auto& code : points) {
      code = random() % 15625;
    }
    std::vector<std::uint64_t> offsets(span_count + 1);
    for (std::size_t span = 0; span <= span_count; ++span) {
      offsets[span] = span * points_per_span;
    }
    std::vector<std::uint32_t> reference(span_count);
    std::vector<std::uint32_t> native(span_count);
    reference_ranks(points, points_per_span, reference);
    native_ranks(points, offsets, native);
    if (reference != native) {
      throw std::runtime_error("native ranks differ from the hand-written loop");
    }

    std::vector<double> reference_seconds;
    std::vector<double> native_seconds;
    for (std::size_t repetition = 0; repetition < repeats; ++repetition) {
      reference_seconds.push_back(measure([&] {
        reference_ranks(points, points_per_span, reference);
      }));
      native_seconds.push_back(measure([&] {
        native_ranks(points, offsets, native);
      }));
      if (reference != native) {
        throw std::runtime_error("rank parity changed during measurement");
      }
    }
    const auto reference_median = median(reference_seconds);
    const auto native_median = median(native_seconds);
    const auto rank_sum = std::accumulate(
        native.begin(), native.end(), std::uint64_t{0});

    std::cout
        << "{\"kernel\":\"encoded_fp_span_ranks\","
        << "\"source_pattern\":\"f5_components.cpp::rankbasis\","
        << "\"prime\":" << kPrime << ','
        << "\"width\":" << kWidth << ','
        << "\"span_count\":" << span_count << ','
        << "\"points_per_span\":" << points_per_span << ','
        << "\"repeats\":" << repeats << ','
        << "\"rank_sum\":" << rank_sum << ','
        << "\"reference_wall_seconds\":";
    print_times(reference_seconds);
    std::cout << ",\"native_wall_seconds\":";
    print_times(native_seconds);
    std::cout
        << ",\"median_reference_wall_seconds\":" << reference_median
        << ",\"median_native_wall_seconds\":" << native_median
        << ",\"wall_speedup\":" << reference_median / native_median
        << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "encoded span rank benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
