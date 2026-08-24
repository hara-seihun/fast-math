#include "fast_math.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Output {
  std::vector<std::uint8_t> intersections;
  std::vector<std::int16_t> correlations;
};

struct PackedKey {
  std::uint64_t lo = 0;
  std::uint16_t hi = 0;

  bool operator==(const PackedKey&) const = default;
};

void put5(PackedKey& key, int position, unsigned value) {
  const auto shift = 5 * position;
  if (shift <= 59) {
    key.lo |= static_cast<std::uint64_t>(value) << shift;
  } else if (shift < 64) {
    key.lo |= static_cast<std::uint64_t>(value) << shift;
    key.hi |= static_cast<std::uint16_t>(value >> (64 - shift));
  } else {
    key.hi |= static_cast<std::uint16_t>(value << (shift - 64));
  }
}

unsigned get5(const PackedKey& key, int position) {
  const auto shift = 5 * position;
  std::uint64_t value = 0;
  if (shift < 64) {
    value = key.lo >> shift;
    if (shift > 59) {
      value |= static_cast<std::uint64_t>(key.hi) << (64 - shift);
    }
  } else {
    value = key.hi >> (shift - 64);
  }
  return static_cast<unsigned>(value & 31);
}

// Retained verbatim in structure from
// qlp64_profile0_binary_census.cpp::pack_counts: the candidates have weight
// 16, and the 16 unique nonzero lags are packed into 80 bits.
void retained_qlp64_pack_counts(
    const std::vector<std::uint64_t>& masks,
    std::vector<PackedKey>& keys) {
  for (std::size_t index = 0; index < masks.size(); ++index) {
    const auto value = static_cast<std::uint32_t>(masks[index]);
    PackedKey key;
    for (int lag = 1; lag <= 16; ++lag) {
      put5(
          key,
          lag - 1,
          std::popcount(value & std::rotl(value, lag)));
    }
    keys[index] = key;
  }
}

void verify_complete_output(
    const std::vector<std::uint64_t>& masks,
    const std::vector<PackedKey>& keys,
    const Output& output) {
  constexpr std::uint32_t width = 32;
  for (std::size_t index = 0; index < masks.size(); ++index) {
    const auto value = static_cast<std::uint32_t>(masks[index]);
    const auto weight = static_cast<int>(std::popcount(value));
    for (std::uint32_t lag = 0; lag < width; ++lag) {
      const auto overlap = static_cast<int>(
          std::popcount(value & std::rotl(value, static_cast<int>(lag))));
      const auto flat_index = index * width + lag;
      if (output.intersections[flat_index] != overlap ||
          output.correlations[flat_index] !=
              static_cast<int>(width) - 4 * (weight - overlap)) {
        throw std::runtime_error("complete output mismatch");
      }
      if (lag >= 1 && lag <= 16 &&
          get5(keys[index], static_cast<int>(lag - 1)) != overlap) {
        throw std::runtime_error("retained packed-key mismatch");
      }
    }
  }
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2;
  if (values.size() % 2 != 0) {
    return values[middle];
  }
  return (values[middle - 1] + values[middle]) / 2;
}

std::vector<std::uint64_t> make_weight_16_masks(std::size_t count) {
  // Gosper's fixed-weight enumeration is the retained census's input loop.
  std::vector<std::uint64_t> masks(count);
  std::uint64_t value = (std::uint64_t{1} << 16) - 1;
  for (auto& mask : masks) {
    if (value >= (std::uint64_t{1} << 32)) {
      throw std::invalid_argument("mask count exceeds the width-32 weight-16 census");
    }
    mask = value;
    const auto low = value & -value;
    const auto next = value + low;
    value = (((next ^ value) >> 2) / low) | next;
  }
  return masks;
}

}  // namespace

int main(int argc, char** argv) {
  constexpr std::uint32_t width = 32;
  const auto mask_count = argc > 1
      ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10))
      : 500000;
  const auto repetitions = argc > 2 ? std::atoi(argv[2]) : 7;
  const auto threads = argc > 3
      ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10))
      : 0;
  if (mask_count == 0 || repetitions < 3) {
    std::cerr << "usage: benchmark_cyclic_profiles [masks>0] [repetitions>=3] "
                 "[threads]\n";
    return 2;
  }

  const auto masks = make_weight_16_masks(mask_count);
  const auto output_count = mask_count * width;
  std::vector<PackedKey> baseline(mask_count);
  Output native{
      std::vector<std::uint8_t>(output_count),
      std::vector<std::int16_t>(output_count)};
  fast_math_cyclic_profile_stats stats{};
  char error[1024]{};
  auto run_native = [&]() {
    const auto status = fast_math_cyclic_correlation_profiles_u64(
        masks.data(), masks.size(), width, threads,
        native.intersections.data(), native.correlations.data(), &stats,
        error, sizeof(error));
    if (status != 0) {
      throw std::runtime_error(error);
    }
  };

  retained_qlp64_pack_counts(masks, baseline);
  run_native();
  verify_complete_output(masks, baseline, native);

  std::vector<double> baseline_seconds;
  std::vector<double> native_seconds;
  baseline_seconds.reserve(repetitions);
  native_seconds.reserve(repetitions);
  auto time_call = [](auto&& callable) {
    const auto started = Clock::now();
    callable();
    return std::chrono::duration<double>(Clock::now() - started).count();
  };
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    if (repetition % 2 == 0) {
      baseline_seconds.push_back(time_call(
          [&]() { retained_qlp64_pack_counts(masks, baseline); }));
      native_seconds.push_back(time_call(run_native));
    } else {
      native_seconds.push_back(time_call(run_native));
      baseline_seconds.push_back(time_call(
          [&]() { retained_qlp64_pack_counts(masks, baseline); }));
    }
  }
  verify_complete_output(masks, baseline, native);

  const auto baseline_median = median(baseline_seconds);
  const auto native_median = median(native_seconds);
  const auto checksum = std::accumulate(
      native.intersections.begin(), native.intersections.end(),
      std::uint64_t{0});
  std::cout << "masks=" << mask_count << " width=" << width
            << " weight=16 repetitions=" << repetitions
            << " workers=" << stats.worker_count << '\n';
  std::cout << "retained_qlp64_seconds=" << baseline_median << '\n';
  std::cout << "native_seconds=" << native_median << '\n';
  std::cout << "speedup=" << baseline_median / native_median << '\n';
  std::cout << "intersection_checksum=" << checksum
            << " complete_output_parity=true\n";
}
