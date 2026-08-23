#include "fast_math.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint32_t kVertexCount = 72;
constexpr std::size_t kTupleCount =
    static_cast<std::size_t>(kVertexCount) * kVertexCount * kVertexCount;
constexpr std::size_t kSignatureSize = 1 + 3 * kVertexCount;
using Signature = std::array<std::uint32_t, kSignatureSize>;

std::uint64_t mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

struct HashPair {
  std::uint64_t first;
  std::uint64_t second;

  bool operator==(const HashPair&) const = default;
};

struct HashPairFunction {
  std::size_t operator()(const HashPair& value) const noexcept {
    return static_cast<std::size_t>(
        mix(value.first ^ mix(value.second)));
  }
};

struct Candidate {
  Signature signature;
  std::uint32_t color;
};

HashPair signature_hash(const Signature& signature) {
  std::uint64_t first = 0x123456789abcdef0ULL;
  std::uint64_t second = 0xfedcba9876543210ULL;
  for (const auto value : signature) {
    first = mix(first ^ value);
    second = mix(second + value + 0x9e3779b9ULL);
  }
  return {first, second};
}

std::uint32_t quaternion_product(std::uint32_t left, std::uint32_t right) {
  const auto left_sign = left < 4 ? 1 : -1;
  const auto right_sign = right < 4 ? 1 : -1;
  const auto left_unit = left % 4;
  const auto right_unit = right % 4;
  int product_sign = 1;
  std::uint32_t product_unit = 0;
  if (left_unit == 0) {
    product_unit = right_unit;
  } else if (right_unit == 0) {
    product_unit = left_unit;
  } else if (left_unit == right_unit) {
    product_sign = -1;
  } else if (
      (left_unit == 1 && right_unit == 2) ||
      (left_unit == 2 && right_unit == 3) ||
      (left_unit == 3 && right_unit == 1)) {
    product_unit = left_unit == 1 ? 3 : (left_unit == 2 ? 1 : 2);
  } else {
    product_sign = -1;
    product_unit = left_unit == 2 ? 3 : (left_unit == 3 ? 1 : 2);
  }
  return (left_sign * right_sign * product_sign == 1 ? 0 : 4) +
      product_unit;
}

std::uint32_t quaternion_inverse(std::uint32_t value) {
  for (std::uint32_t candidate = 0; candidate < 8; ++candidate) {
    if (quaternion_product(value, candidate) == 0 &&
        quaternion_product(candidate, value) == 0) {
      return candidate;
    }
  }
  throw std::runtime_error("quaternion inverse is missing");
}

std::uint32_t group_product(std::uint32_t left, std::uint32_t right) {
  return quaternion_product(left / 9, right / 9) * 9 +
      (left % 9 + right % 9) % 9;
}

std::uint32_t group_inverse(std::uint32_t value) {
  return quaternion_inverse(value / 9) * 9 + (9 - value % 9) % 9;
}

std::vector<std::vector<std::uint32_t>> inverse_atoms() {
  std::array<std::int32_t, kVertexCount> atom_of{};
  atom_of.fill(-1);
  std::vector<std::vector<std::uint32_t>> atoms;
  for (std::uint32_t element = 1; element < kVertexCount; ++element) {
    if (atom_of[element] >= 0) {
      continue;
    }
    const auto inverse = group_inverse(element);
    std::vector<std::uint32_t> atom{element};
    if (inverse != element) {
      atom.push_back(inverse);
      std::sort(atom.begin(), atom.end());
    }
    const auto atom_index = static_cast<std::int32_t>(atoms.size());
    for (const auto member : atom) {
      atom_of[member] = atom_index;
    }
    atoms.push_back(std::move(atom));
  }
  return atoms;
}

std::vector<std::uint8_t> q8_c9_adjacency(
    const std::array<std::uint32_t, 3>& atom_ids) {
  const auto atoms = inverse_atoms();
  std::vector<std::uint32_t> connection_set;
  for (const auto atom_id : atom_ids) {
    if (atom_id >= atoms.size()) {
      throw std::invalid_argument("atom id is outside the inverse atoms");
    }
    connection_set.insert(
        connection_set.end(), atoms[atom_id].begin(), atoms[atom_id].end());
  }
  std::vector<std::uint8_t> adjacency(
      static_cast<std::size_t>(kVertexCount) * kVertexCount);
  for (std::uint32_t left = 0; left < kVertexCount; ++left) {
    for (const auto connection : connection_set) {
      const auto right = group_product(left, connection);
      adjacency[static_cast<std::size_t>(left) * kVertexCount + right] = 1;
    }
  }
  return adjacency;
}

std::size_t tuple_index(
    std::uint32_t first,
    std::uint32_t second,
    std::uint32_t third) {
  return (static_cast<std::size_t>(first) * kVertexCount + second) *
      kVertexCount + third;
}

struct BaselineResult {
  std::vector<std::uint32_t> colors;
  std::uint32_t color_count;
  std::uint64_t iterations;
};

BaselineResult hand_written_wl3(
    const std::vector<std::uint8_t>& adjacency) {
  std::vector<std::uint32_t> colors(kTupleCount);
  std::vector<std::uint32_t> refined(kTupleCount);
  for (std::uint32_t first = 0; first < kVertexCount; ++first) {
    for (std::uint32_t second = 0; second < kVertexCount; ++second) {
      for (std::uint32_t third = 0; third < kVertexCount; ++third) {
        auto code = static_cast<std::uint32_t>(first == second) |
            (static_cast<std::uint32_t>(first == third) << 1) |
            (static_cast<std::uint32_t>(second == third) << 2) |
            (static_cast<std::uint32_t>(adjacency[
                 static_cast<std::size_t>(first) * kVertexCount + second])
             << 3) |
            (static_cast<std::uint32_t>(adjacency[
                 static_cast<std::size_t>(first) * kVertexCount + third])
             << 4) |
            (static_cast<std::uint32_t>(adjacency[
                 static_cast<std::size_t>(second) * kVertexCount + third])
             << 5);
        colors[tuple_index(first, second, third)] = code;
      }
    }
  }

  std::uint32_t final_count = 0;
  std::uint64_t iterations = 0;
  std::array<std::uint32_t, kVertexCount> replacements{};
  while (true) {
    std::unordered_map<
        HashPair,
        std::vector<Candidate>,
        HashPairFunction> classes;
    classes.reserve(1024);
    std::uint32_t next_color = 0;
    for (std::uint32_t first = 0; first < kVertexCount; ++first) {
      for (std::uint32_t second = 0; second < kVertexCount; ++second) {
        for (std::uint32_t third = 0; third < kVertexCount; ++third) {
          Signature signature{};
          signature[0] = colors[tuple_index(first, second, third)];
          std::size_t output = 1;
          for (std::uint32_t vertex = 0;
               vertex < kVertexCount;
               ++vertex) {
            replacements[vertex] = colors[
                tuple_index(vertex, second, third)];
          }
          std::sort(replacements.begin(), replacements.end());
          for (const auto color : replacements) {
            signature[output++] = color;
          }
          for (std::uint32_t vertex = 0;
               vertex < kVertexCount;
               ++vertex) {
            replacements[vertex] = colors[
                tuple_index(first, vertex, third)];
          }
          std::sort(replacements.begin(), replacements.end());
          for (const auto color : replacements) {
            signature[output++] = color;
          }
          for (std::uint32_t vertex = 0;
               vertex < kVertexCount;
               ++vertex) {
            replacements[vertex] = colors[
                tuple_index(first, second, vertex)];
          }
          std::sort(replacements.begin(), replacements.end());
          for (const auto color : replacements) {
            signature[output++] = color;
          }

          auto& bucket = classes[signature_hash(signature)];
          auto color = std::numeric_limits<std::uint32_t>::max();
          for (const auto& candidate : bucket) {
            if (candidate.signature == signature) {
              color = candidate.color;
              break;
            }
          }
          if (color == std::numeric_limits<std::uint32_t>::max()) {
            color = next_color++;
            bucket.push_back({signature, color});
          }
          refined[tuple_index(first, second, third)] = color;
        }
      }
    }

    const auto old_count =
        *std::max_element(colors.begin(), colors.end()) + 1;
    std::vector<std::int32_t> old_to_new(old_count, -1);
    std::vector<std::int32_t> new_to_old(next_color, -1);
    bool stable = true;
    for (std::size_t tuple = 0; tuple < kTupleCount; ++tuple) {
      const auto old_color = colors[tuple];
      const auto new_color = refined[tuple];
      auto& forward = old_to_new[old_color];
      auto& reverse = new_to_old[new_color];
      if (forward < 0) {
        forward = static_cast<std::int32_t>(new_color);
      } else if (forward != static_cast<std::int32_t>(new_color)) {
        stable = false;
      }
      if (reverse < 0) {
        reverse = static_cast<std::int32_t>(old_color);
      } else if (reverse != static_cast<std::int32_t>(old_color)) {
        stable = false;
      }
    }
    colors.swap(refined);
    final_count = next_color;
    ++iterations;
    if (stable) {
      return {std::move(colors), final_count, iterations};
    }
    if (iterations == 20) {
      throw std::runtime_error("hand-written 3-WL did not stabilize");
    }
  }
}

struct NativeResult {
  std::vector<std::uint32_t> colors;
  std::vector<std::uint64_t> color_sizes;
  fast_math_ci_stats stats;
};

NativeResult native_wl3(const std::vector<std::uint8_t>& adjacency) {
  constexpr std::uint32_t word_count = 2;
  std::vector<std::uint64_t> words(
      static_cast<std::size_t>(kVertexCount) * word_count);
  for (std::uint32_t left = 0; left < kVertexCount; ++left) {
    for (std::uint32_t right = 0; right < kVertexCount; ++right) {
      if (adjacency[static_cast<std::size_t>(left) * kVertexCount + right]) {
        words[static_cast<std::size_t>(left) * word_count + right / 64] |=
            std::uint64_t{1} << (right % 64);
      }
    }
  }
  NativeResult result{
      std::vector<std::uint32_t>(kTupleCount),
      std::vector<std::uint64_t>(kTupleCount),
      {},
  };
  char error[512]{};
  const auto status = fast_math_graph_wlk_refine_u64(
      words.data(),
      kVertexCount,
      word_count,
      3,
      kTupleCount,
      result.colors.data(),
      result.color_sizes.data(),
      &result.stats,
      error,
      sizeof(error));
  if (status != 0) {
    throw std::runtime_error(error);
  }
  result.color_sizes.resize(result.stats.class_count);
  return result;
}

void verify_same_partition(
    const BaselineResult& baseline,
    const NativeResult& native) {
  if (baseline.color_count != native.stats.class_count ||
      baseline.colors.size() != native.colors.size()) {
    throw std::runtime_error("3-WL color counts differ");
  }
  std::vector<std::int32_t> baseline_to_native(
      baseline.color_count, -1);
  std::vector<std::int32_t> native_to_baseline(
      native.stats.class_count, -1);
  for (std::size_t tuple = 0; tuple < baseline.colors.size(); ++tuple) {
    const auto left = baseline.colors[tuple];
    const auto right = native.colors[tuple];
    auto& forward = baseline_to_native[left];
    auto& reverse = native_to_baseline[right];
    if (forward < 0) {
      forward = static_cast<std::int32_t>(right);
    } else if (forward != static_cast<std::int32_t>(right)) {
      throw std::runtime_error("native 3-WL splits a baseline color");
    }
    if (reverse < 0) {
      reverse = static_cast<std::int32_t>(left);
    } else if (reverse != static_cast<std::int32_t>(left)) {
      throw std::runtime_error("baseline 3-WL splits a native color");
    }
  }
  std::vector<std::uint64_t> baseline_sizes(baseline.color_count);
  for (const auto color : baseline.colors) {
    ++baseline_sizes[color];
  }
  auto native_sizes = native.color_sizes;
  std::sort(baseline_sizes.begin(), baseline_sizes.end());
  std::sort(native_sizes.begin(), native_sizes.end());
  if (baseline_sizes != native_sizes) {
    throw std::runtime_error("3-WL color histograms differ");
  }
}

template <typename Function>
auto measure(Function&& function) {
  const auto started = Clock::now();
  auto result = function();
  const auto seconds =
      std::chrono::duration<double>(Clock::now() - started).count();
  return std::pair{std::move(result), seconds};
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
    const std::array<std::uint32_t, 3> atom_ids{
        argc > 1 ? static_cast<std::uint32_t>(std::stoul(argv[1])) : 1,
        argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 4,
        argc > 3 ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 13,
    };
    const auto repeats =
        argc > 4 ? static_cast<std::size_t>(std::stoull(argv[4])) : 3;
    if (repeats == 0) {
      throw std::invalid_argument("repeat count must be positive");
    }
    const auto adjacency = q8_c9_adjacency(atom_ids);
    auto baseline = hand_written_wl3(adjacency);
    auto native = native_wl3(adjacency);
    verify_same_partition(baseline, native);

    std::vector<double> baseline_seconds;
    std::vector<double> native_seconds;
    for (std::size_t repetition = 0; repetition < repeats; ++repetition) {
      auto measured_baseline = measure([&] {
        return hand_written_wl3(adjacency);
      });
      baseline = std::move(measured_baseline.first);
      baseline_seconds.push_back(measured_baseline.second);
      auto measured_native = measure([&] {
        return native_wl3(adjacency);
      });
      native = std::move(measured_native.first);
      native_seconds.push_back(measured_native.second);
      verify_same_partition(baseline, native);
    }
    const auto baseline_median = median(baseline_seconds);
    const auto native_median = median(native_seconds);

    std::cout
        << "{\"kernel\":\"higher_order_wl\","
        << "\"source_pattern\":\"wl3_q8_exact.cpp\","
        << "\"group\":\"Q8xC9\","
        << "\"atom_ids\":[" << atom_ids[0] << ',' << atom_ids[1]
        << ',' << atom_ids[2] << "],"
        << "\"vertex_count\":" << kVertexCount << ','
        << "\"dimension\":3,"
        << "\"tuple_count\":" << kTupleCount << ','
        << "\"color_count\":" << native.stats.class_count << ','
        << "\"iterations\":" << native.stats.iteration_count << ','
        << "\"repeats\":" << repeats << ','
        << "\"reference_wall_seconds\":";
    print_times(baseline_seconds);
    std::cout << ",\"native_wall_seconds\":";
    print_times(native_seconds);
    std::cout
        << ",\"median_reference_wall_seconds\":" << baseline_median
        << ",\"median_native_wall_seconds\":" << native_median
        << ",\"wall_speedup\":" << baseline_median / native_median
        << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "higher-order WL benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
