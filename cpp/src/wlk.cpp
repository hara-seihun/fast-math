#include "fast_math.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kMaximumTupleCount = 10'000'000;
constexpr std::size_t kMaximumSignatureEntries = 64'000'000;
constexpr auto kNoClass = std::numeric_limits<std::uint32_t>::max();

void set_error(
    char* destination,
    std::size_t destination_size,
    const char* message) {
  if (destination == nullptr || destination_size == 0) {
    return;
  }
  const auto length = std::min(
      destination_size - 1,
      std::strlen(message));
  std::memcpy(destination, message, length);
  destination[length] = '\0';
}

std::uint64_t mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

struct SignatureHash {
  std::uint64_t first;
  std::uint64_t second;

  bool operator==(const SignatureHash&) const = default;
};

struct SignatureHashFunction {
  std::size_t operator()(const SignatureHash& value) const noexcept {
    return static_cast<std::size_t>(mix(value.first ^ mix(value.second)));
  }
};

SignatureHash hash_signature(const std::vector<std::uint32_t>& signature) {
  std::uint64_t first = 0x123456789abcdef0ULL;
  std::uint64_t second = 0xfedcba9876543210ULL;
  for (const auto value : signature) {
    first = mix(first ^ value);
    second = mix(second + value + 0x9e3779b9ULL);
  }
  return {first, second};
}

class SignatureClasses {
 public:
  explicit SignatureClasses(std::size_t tuple_count) {
    const auto initial_classes = std::min<std::size_t>(tuple_count, 4096);
    offsets_.reserve(initial_classes + 1);
    offsets_.push_back(0);
    collision_next_.reserve(initial_classes);
    first_by_hash_.reserve(initial_classes);
  }

  std::uint32_t intern(const std::vector<std::uint32_t>& signature) {
    const auto hash = hash_signature(signature);
    const auto found = first_by_hash_.find(hash);
    if (found != first_by_hash_.end()) {
      auto candidate = found->second;
      while (candidate != kNoClass) {
        const auto begin = representatives_.begin() + offsets_[candidate];
        const auto end = representatives_.begin() + offsets_[candidate + 1];
        if (static_cast<std::size_t>(end - begin) == signature.size() &&
            std::equal(signature.begin(), signature.end(), begin)) {
          return candidate;
        }
        candidate = collision_next_[candidate];
      }
    }
    const auto class_index = class_count();
    if (class_index == kNoClass) {
      throw std::overflow_error("k-WL color count exceeds uint32");
    }
    if (signature.size() >
        kMaximumSignatureEntries - representatives_.size()) {
      throw std::invalid_argument(
          "k-WL exact signature storage exceeds 256 MB");
    }
    representatives_.insert(
        representatives_.end(), signature.begin(), signature.end());
    offsets_.push_back(representatives_.size());
    collision_next_.push_back(
        found == first_by_hash_.end() ? kNoClass : found->second);
    if (found == first_by_hash_.end()) {
      first_by_hash_.emplace(hash, class_index);
    } else {
      found->second = class_index;
    }
    return class_index;
  }

  std::uint32_t class_count() const {
    return static_cast<std::uint32_t>(collision_next_.size());
  }

  std::vector<std::uint32_t> canonical_mapping() const {
    std::vector<std::uint32_t> order(class_count());
    std::iota(order.begin(), order.end(), 0);
    std::sort(
        order.begin(),
        order.end(),
        [&](std::uint32_t left, std::uint32_t right) {
          return std::lexicographical_compare(
              representatives_.begin() + offsets_[left],
              representatives_.begin() + offsets_[left + 1],
              representatives_.begin() + offsets_[right],
              representatives_.begin() + offsets_[right + 1]);
        });
    std::vector<std::uint32_t> mapping(class_count());
    for (std::uint32_t canonical = 0;
         canonical < order.size();
         ++canonical) {
      mapping[order[canonical]] = canonical;
    }
    return mapping;
  }

 private:
  std::vector<std::uint32_t> representatives_;
  std::vector<std::size_t> offsets_;
  std::vector<std::uint32_t> collision_next_;
  std::unordered_map<
      SignatureHash,
      std::uint32_t,
      SignatureHashFunction> first_by_hash_;
};

std::size_t checked_tuple_count(
    std::uint32_t vertex_count,
    std::uint32_t dimension) {
  std::size_t count = 1;
  for (std::uint32_t coordinate = 0;
       coordinate < dimension;
       ++coordinate) {
    if (count > kMaximumTupleCount / vertex_count) {
      throw std::invalid_argument(
          "k-WL tuple count exceeds the 10,000,000 limit");
    }
    count *= vertex_count;
  }
  return count;
}

void validate_graph(
    const std::uint64_t* adjacency_words,
    std::uint32_t vertex_count,
    std::uint32_t word_count) {
  if (adjacency_words == nullptr) {
    throw std::invalid_argument("k-WL adjacency pointer is null");
  }
  if (vertex_count == 0 || vertex_count > 512) {
    throw std::invalid_argument(
        "k-WL vertex count must be between one and 512");
  }
  const auto expected_words = (vertex_count + 63) / 64;
  if (word_count != expected_words) {
    throw std::invalid_argument("k-WL packed adjacency shape is invalid");
  }
  const auto final_bits = vertex_count % 64;
  const auto final_mask = final_bits == 0
      ? std::numeric_limits<std::uint64_t>::max()
      : (std::uint64_t{1} << final_bits) - 1;
  for (std::uint32_t vertex = 0; vertex < vertex_count; ++vertex) {
    const auto* row = adjacency_words +
        static_cast<std::size_t>(vertex) * word_count;
    if ((row[word_count - 1] & ~final_mask) != 0) {
      throw std::invalid_argument(
          "k-WL adjacency contains an out-of-range vertex");
    }
    if ((row[vertex / 64] &
         (std::uint64_t{1} << (vertex % 64))) != 0) {
      throw std::invalid_argument("k-WL graph must be loopless");
    }
  }
}

bool adjacent(
    const std::uint64_t* adjacency_words,
    std::uint32_t word_count,
    std::uint32_t left,
    std::uint32_t right) {
  return (adjacency_words[
              static_cast<std::size_t>(left) * word_count + right / 64] &
          (std::uint64_t{1} << (right % 64))) != 0;
}

void decode_tuple(
    std::size_t index,
    std::uint32_t vertex_count,
    std::uint32_t dimension,
    std::array<std::uint32_t, 4>& vertices) {
  for (std::uint32_t coordinate = dimension;
       coordinate-- > 0;) {
    vertices[coordinate] = static_cast<std::uint32_t>(
        index % vertex_count);
    index /= vertex_count;
  }
}

std::vector<std::uint32_t> initial_colors(
    const std::uint64_t* adjacency_words,
    std::uint32_t vertex_count,
    std::uint32_t word_count,
    std::uint32_t dimension,
    std::size_t tuple_count) {
  std::uint32_t atomic_code_count = 1;
  for (std::uint32_t pair = 0;
       pair < dimension * (dimension - 1);
       ++pair) {
    atomic_code_count *= 3;
  }
  std::vector<std::uint32_t> colors(tuple_count);
  std::vector<std::uint8_t> present(atomic_code_count);
  std::array<std::uint32_t, 4> vertices{};
  for (std::size_t tuple = 0; tuple < tuple_count; ++tuple) {
    decode_tuple(tuple, vertex_count, dimension, vertices);
    std::uint32_t code = 0;
    for (std::uint32_t left = 0; left < dimension; ++left) {
      for (std::uint32_t right = 0; right < dimension; ++right) {
        if (left == right) {
          continue;
        }
        const auto relation = vertices[left] == vertices[right]
            ? 0U
            : (adjacent(
                   adjacency_words,
                   word_count,
                   vertices[left],
                   vertices[right])
                   ? 1U
                   : 2U);
        code = code * 3 + relation;
      }
    }
    colors[tuple] = code;
    present[code] = 1;
  }
  std::vector<std::uint32_t> mapping(atomic_code_count);
  std::uint32_t color_count = 0;
  for (std::uint32_t code = 0; code < atomic_code_count; ++code) {
    if (present[code] != 0) {
      mapping[code] = color_count++;
    }
  }
  for (auto& color : colors) {
    color = mapping[color];
  }
  return colors;
}

std::uint32_t refine(
    std::vector<std::uint32_t>& colors,
    std::uint32_t vertex_count,
    std::uint32_t dimension,
    std::uint64_t* iteration_count) {
  const auto tuple_count = colors.size();
  std::array<std::size_t, 4> strides{};
  strides[dimension - 1] = 1;
  for (std::uint32_t coordinate = dimension - 1;
       coordinate > 0;
       --coordinate) {
    strides[coordinate - 1] = strides[coordinate] * vertex_count;
  }
  std::vector<std::uint32_t> signature;
  signature.reserve(
      1 + static_cast<std::size_t>(dimension) * (1 + 2 * vertex_count));
  std::vector<std::uint32_t> refined(tuple_count);
  std::array<std::uint32_t, 4> vertices{};
  auto color_count =
      *std::max_element(colors.begin(), colors.end()) + 1;
  std::vector<std::uint32_t> histogram(color_count);
  std::vector<std::uint32_t> active_colors;
  active_colors.reserve(vertex_count);
  *iteration_count = 0;

  while (true) {
    SignatureClasses classes(tuple_count);
    for (std::size_t tuple = 0; tuple < tuple_count; ++tuple) {
      decode_tuple(tuple, vertex_count, dimension, vertices);
      signature.clear();
      signature.push_back(colors[tuple]);
      for (std::uint32_t coordinate = 0;
           coordinate < dimension;
           ++coordinate) {
        const auto fixed_index = tuple -
            static_cast<std::size_t>(vertices[coordinate]) *
                strides[coordinate];
        for (std::uint32_t replacement = 0;
             replacement < vertex_count;
             ++replacement) {
          const auto color = colors[
              fixed_index +
              static_cast<std::size_t>(replacement) *
                  strides[coordinate]];
          if (histogram[color]++ == 0) {
            active_colors.push_back(color);
          }
        }
        std::sort(active_colors.begin(), active_colors.end());
        signature.push_back(
            static_cast<std::uint32_t>(active_colors.size()));
        for (const auto color : active_colors) {
          signature.push_back(color);
          signature.push_back(histogram[color]);
          histogram[color] = 0;
        }
        active_colors.clear();
      }
      refined[tuple] = classes.intern(signature);
    }

    const auto mapping = classes.canonical_mapping();
    for (auto& color : refined) {
      color = mapping[color];
    }
    ++*iteration_count;
    const auto refined_count = classes.class_count();
    if (refined == colors) {
      return refined_count;
    }
    colors.swap(refined);
    color_count = refined_count;
    histogram.assign(color_count, 0);
  }
}

}  // namespace

extern "C" {

int fast_math_graph_wlk_refine_u64(
    const std::uint64_t* adjacency_words,
    std::uint32_t vertex_count,
    std::uint32_t word_count,
    std::uint32_t dimension,
    std::size_t tuple_capacity,
    std::uint32_t* stable_colors,
    std::uint64_t* color_sizes,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (dimension != 3 && dimension != 4) {
      throw std::invalid_argument("k-WL dimension must be three or four");
    }
    validate_graph(adjacency_words, vertex_count, word_count);
    const auto tuple_count = checked_tuple_count(vertex_count, dimension);
    if (tuple_capacity < tuple_count) {
      throw std::invalid_argument("k-WL tuple output capacity is too small");
    }
    if (stable_colors == nullptr || color_sizes == nullptr || stats == nullptr) {
      throw std::invalid_argument("k-WL output pointer is null");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    auto colors = initial_colors(
        adjacency_words,
        vertex_count,
        word_count,
        dimension,
        tuple_count);
    std::uint64_t iterations = 0;
    const auto color_count = refine(
        colors, vertex_count, dimension, &iterations);
    std::copy(colors.begin(), colors.end(), stable_colors);
    std::fill_n(color_sizes, color_count, 0);
    for (const auto color : colors) {
      ++color_sizes[color];
    }
    stats->degree = vertex_count;
    stats->item_count = tuple_count;
    stats->class_count = color_count;
    stats->relation_count = color_count;
    stats->iteration_count = iterations;
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

}  // extern "C"
