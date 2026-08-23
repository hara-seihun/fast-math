#include "fast_math.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

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

constexpr std::uint32_t kMaxDegree = 64;
constexpr std::uint32_t kMaxBase = 64;
constexpr std::uint64_t kMaxGroupSize = 200000;

void validate_shape(std::uint32_t base, std::uint32_t degree) {
  if (base < 2 || base > kMaxBase) {
    throw std::invalid_argument("base must be between 2 and 64");
  }
  if (degree == 0 || degree > kMaxDegree) {
    throw std::invalid_argument("width must be between 1 and 64");
  }
}

void validate_generators(
    const std::uint32_t* generators,
    std::size_t generator_count,
    std::uint32_t degree) {
  if (generator_count == 0) {
    throw std::invalid_argument("at least one generator is required");
  }
  if (generators == nullptr) {
    throw std::invalid_argument("generator pointer is null");
  }
  for (std::size_t g = 0; g < generator_count; ++g) {
    const std::uint32_t* row = generators + g * degree;
    std::vector<bool> seen(degree, false);
    for (std::uint32_t i = 0; i < degree; ++i) {
      if (row[i] >= degree || seen[row[i]]) {
        throw std::invalid_argument(
            "generator rows must be permutations of 0..width-1");
      }
      seen[row[i]] = true;
    }
  }
}

// Exact base^width, erroring on overflow of uint64.
std::uint64_t space_size(std::uint32_t base, std::uint32_t degree) {
  std::uint64_t size = 1;
  for (std::uint32_t i = 0; i < degree; ++i) {
    if (size > ~std::uint64_t{0} / base) {
      throw std::invalid_argument(
          "base^width exceeds the uint64 code range");
    }
    size *= base;
  }
  return size;
}

// Permutations are image arrays: p[x] is the image of x; composition is
// left[right]. A group element p moves the digit at position i to position
// p[i] in the image tuple.
struct PermutationGroup {
  std::uint32_t degree;
  std::vector<std::uint32_t> elements;  // group_size * degree, row-major

  static std::string key(
      const std::uint32_t* permutation,
      std::uint32_t degree) {
    return std::string(
        reinterpret_cast<const char*>(permutation),
        static_cast<std::size_t>(degree) * sizeof(std::uint32_t));
  }

  void build(
      const std::uint32_t* generators,
      std::size_t generator_count) {
    std::unordered_set<std::string> known;
    std::vector<std::uint32_t> queue;
    auto push = [&](const std::uint32_t* permutation) {
      auto label = key(permutation, degree);
      if (known.insert(label).second) {
        queue.insert(
            queue.end(), permutation, permutation + degree);
        if (queue.size() / degree > kMaxGroupSize) {
          throw std::invalid_argument(
              "generated group exceeds the 200000-element limit");
        }
      }
    };
    std::vector<std::uint32_t> identity(degree);
    for (std::uint32_t i = 0; i < degree; ++i) {
      identity[i] = i;
    }
    push(identity.data());
    for (std::size_t g = 0; g < generator_count; ++g) {
      push(generators + g * degree);
    }
    for (std::size_t cursor = 0; cursor < queue.size(); cursor += degree) {
      const std::uint32_t* current = queue.data() + cursor;
      for (std::size_t g = 0; g < generator_count; ++g) {
        const std::uint32_t* generator = generators + g * degree;
        std::vector<std::uint32_t> product(degree);
        for (std::uint32_t i = 0; i < degree; ++i) {
          product[i] = current[generator[i]];  // current composed after generator
        }
        push(product.data());
      }
    }
    elements = std::move(queue);
  }

  std::size_t size() const { return elements.size() / degree; }
};

std::uint64_t apply_permutation(
    const std::uint32_t* permutation,
    std::uint32_t base,
    std::uint32_t degree,
    std::uint64_t code) {
  std::array<std::uint32_t, kMaxDegree> digits{};
  std::uint64_t rest = code;
  for (std::uint32_t i = degree; i-- > 0;) {
    digits[i] = rest % base;
    rest /= base;
  }
  // e[p[i]] = d[i] over positional digits 0..degree-1, position 0 most
  // significant in the mixed-radix code.
  std::array<std::uint32_t, kMaxDegree> out{};
  for (std::uint32_t i = 0; i < degree; ++i) {
    out[permutation[i]] = digits[i];
  }
  std::uint64_t image = 0;
  for (std::uint32_t i = 0; i < degree; ++i) {
    image = image * base + out[i];
  }
  return image;
}

std::uint64_t cycle_count(
    const std::uint32_t* permutation,
    std::uint32_t degree) {
  std::vector<bool> seen(degree, false);
  std::uint64_t cycles = 0;
  for (std::uint32_t i = 0; i < degree; ++i) {
    if (seen[i]) {
      continue;
    }
    cycles += 1;
    std::uint32_t node = i;
    while (!seen[node]) {
      seen[node] = true;
      node = permutation[node];
    }
  }
  return cycles;
}

}  // namespace

extern "C" {

int fast_math_tuple_orbit_canonicalize_u64(
    const std::uint32_t* generators,
    std::size_t generator_count,
    std::uint32_t degree,
    std::uint32_t base,
    const std::uint64_t* codes,
    std::size_t code_count,
    std::uint64_t* canonical_codes,
    std::uint8_t* is_canonical,
    fast_math_tuple_orbit_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("stats pointer is null");
    }
    validate_shape(base, degree);
    validate_generators(generators, generator_count, degree);
    if (code_count != 0 &&
        (codes == nullptr || canonical_codes == nullptr ||
         is_canonical == nullptr)) {
      throw std::invalid_argument("code or output pointer is null");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    stats->code_count = code_count;
    const auto started = Clock::now();

    const auto limit = space_size(base, degree);
    PermutationGroup group{degree, {}};
    group.build(generators, generator_count);
    stats->group_size = group.size();

    for (std::uint64_t index = 0; index < code_count; ++index) {
      const auto code = codes[index];
      if (code >= limit) {
        throw std::invalid_argument(
            "code is outside the base^width code range");
      }
      std::uint64_t best = code;
      for (std::size_t g = 0; g < group.size(); ++g) {
        const auto image = apply_permutation(
            group.elements.data() + g * degree, base, degree, code);
        stats->canonical_evaluations += 1;
        best = std::min(best, image);
      }
      canonical_codes[index] = best;
      is_canonical[index] = best == code ? 1 : 0;
    }

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

int fast_math_tuple_orbit_space_u64(
    const std::uint32_t* generators,
    std::size_t generator_count,
    std::uint32_t degree,
    std::uint32_t base,
    std::uint64_t* canonical_codes,
    std::size_t space_capacity,
    fast_math_tuple_orbit_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("stats pointer is null");
    }
    validate_shape(base, degree);
    validate_generators(generators, generator_count, degree);
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();

    const auto limit = space_size(base, degree);
    if (limit > space_capacity) {
      throw std::invalid_argument(
          "output buffer is smaller than the base^width code space");
    }
    if (canonical_codes == nullptr && limit != 0) {
      throw std::invalid_argument("canonical output pointer is null");
    }
    stats->code_count = limit;

    PermutationGroup group{degree, {}};
    group.build(generators, generator_count);
    stats->group_size = group.size();

    for (std::uint64_t code = 0; code < limit; ++code) {
      std::uint64_t best = code;
      for (std::size_t g = 0; g < group.size(); ++g) {
        const auto image = apply_permutation(
            group.elements.data() + g * degree, base, degree, code);
        stats->canonical_evaluations += 1;
        best = std::min(best, image);
      }
      canonical_codes[code] = best;
    }

    // Burnside check: a position permutation with c cycles fixes
    // base^c tuples, so the orbit count must equal the average fix.
    std::uint64_t fixed_total = 0;
    for (std::size_t g = 0; g < group.size(); ++g) {
      std::uint64_t fixed = 1;
      for (std::uint64_t c = cycle_count(
               group.elements.data() + g * degree, degree);
           c-- > 0;) {
        fixed *= base;
      }
      fixed_total += fixed;
    }
    if (fixed_total % group.size() != 0) {
      throw std::runtime_error(
          "Burnside fixed-point total is not divisible by the group size");
    }
    stats->burnside_orbit_count = fixed_total / group.size();

    // Distinct canonical codes count the full-space orbits.
    std::vector<std::uint64_t> reps(
        canonical_codes, canonical_codes + limit);
    std::sort(reps.begin(), reps.end());
    const auto distinct =
        std::unique(reps.begin(), reps.end()) - reps.begin();
    stats->orbit_count = static_cast<std::uint64_t>(distinct);
    stats->burnside_valid =
        stats->orbit_count == stats->burnside_orbit_count ? 1 : 0;

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
