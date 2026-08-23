#include "fast_math.h"

#include "parallel.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kMaximumBasePPrime = 251;
constexpr std::uint32_t kMaximumBasePWidth = 16;

// Bound on the direct-labeling table used by scalar-class batches. When the
// index space is small relative to the batch, classes are labeled by direct
// ascending scan instead of sorting the batch.
constexpr std::uint64_t kDirectClassTableCap = std::uint64_t{1} << 24;

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

bool is_small_prime(std::uint32_t value) {
  if (value < 2) {
    return false;
  }
  for (std::uint32_t divisor = 2; divisor * divisor <= value; ++divisor) {
    if (value % divisor == 0) {
      return false;
    }
  }
  return true;
}

// Returns p**width, throwing when the value does not fit in uint64.
std::uint64_t index_space_size(std::uint32_t prime, std::uint32_t width) {
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t size = 1;
  for (std::uint32_t dimension = 0; dimension < width; ++dimension) {
    if (size > maximum / prime) {
      throw std::invalid_argument(
          "base-p index space p^width exceeds uint64 range");
    }
    size *= prime;
  }
  return size;
}

std::uint32_t modular_inverse(std::uint32_t value, std::uint32_t prime) {
  std::int64_t left = value % prime;
  std::int64_t middle = prime;
  std::int64_t left_coefficient = 1;
  std::int64_t middle_coefficient = 0;
  while (middle != 0) {
    const std::int64_t quotient = left / middle;
    const std::int64_t remainder = left - quotient * middle;
    left = middle;
    middle = remainder;
    const std::int64_t next = left_coefficient - quotient * middle_coefficient;
    left_coefficient = middle_coefficient;
    middle_coefficient = next;
  }
  std::int64_t inverse = left_coefficient % prime;
  if (inverse < 0) {
    inverse += prime;
  }
  return static_cast<std::uint32_t>(inverse);
}

struct BasePFields {
  std::uint32_t digits[kMaximumBasePWidth];
};

void decode_index(
    std::uint64_t index,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint32_t (&digits)[kMaximumBasePWidth]) {
  for (std::uint32_t dimension = 0; dimension < width; ++dimension) {
    digits[dimension] = static_cast<std::uint32_t>(index % prime);
    index /= prime;
  }
}

std::uint64_t encode_digits(
    const std::uint32_t (&digits)[kMaximumBasePWidth],
    std::uint32_t prime,
    std::uint32_t width) {
  std::uint64_t index = 0;
  for (std::uint32_t dimension = width; dimension-- > 0;) {
    index = index * prime + digits[dimension];
  }
  return index;
}

// Canonical representative of {v, -v}: the smaller mixed-radix index.
std::uint64_t negation_representative(
    std::uint64_t index,
    std::uint32_t prime,
    std::uint32_t width) {
  BasePFields fields;
  decode_index(index, prime, width, fields.digits);
  for (std::uint32_t dimension = 0; dimension < width; ++dimension) {
    const std::uint32_t digit = fields.digits[dimension];
    fields.digits[dimension] = digit == 0 ? 0 : prime - digit;
  }
  const std::uint64_t negated =
      encode_digits(fields.digits, prime, width);
  return negated < index ? negated : index;
}

// Canonical representative of the orbit {c*v mod p : c in F_p*} under scalar
// multiplication: the minimum mixed-radix index over all nonzero multiples.
// The zero vector is its own class with representative zero.
//
// Closed form: scaling by c never changes which digits are zero, so every
// multiple shares the same highest nonzero digit position t. As c ranges over
// F_p* the digit at position t takes each nonzero value exactly once, so
// exactly one multiple has digit one at position t, and comparing encodings
// from the most significant end shows that multiple is the minimum. Therefore
// the representative is v scaled by the modular inverse of its highest
// nonzero digit -- O(width) work and no factor scan.
std::uint64_t scalar_representative(
    std::uint64_t index,
    std::uint32_t prime,
    std::uint32_t width) {
  BasePFields fields;
  decode_index(index, prime, width, fields.digits);

  std::uint32_t top = width;
  while (top > 0 && fields.digits[top - 1] == 0) {
    --top;
  }
  if (top == 0) {
    return 0;
  }

  const std::uint32_t scale =
      modular_inverse(fields.digits[top - 1], prime);
  std::uint64_t representative = 0;
  for (std::uint32_t dimension = width; dimension-- > 0;) {
    representative = representative * prime +
        static_cast<std::uint64_t>(fields.digits[dimension] * scale % prime);
  }
  return representative;
}

}  // namespace

extern "C" {

FAST_MATH_API int fast_math_base_p_decode_u64(
    const std::uint64_t* indices,
    std::size_t element_count,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint32_t thread_count,
    std::uint32_t* digits,
    fast_math_base_p_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("stats pointer is null");
    }
    if (element_count != 0 &&
        (indices == nullptr || digits == nullptr)) {
      throw std::invalid_argument("input or output pointer is null");
    }
    if (!is_small_prime(prime)) {
      throw std::invalid_argument("prime must be a prime at most 251");
    }
    if (width < 1 || width > kMaximumBasePWidth) {
      throw std::invalid_argument("width must be between one and sixteen");
    }
    const std::uint64_t space = index_space_size(prime, width);

    set_error(error_message, error_message_size, "");
    *stats = {};
    stats->element_count = element_count;
    const auto started = Clock::now();

    std::atomic<bool> invalid{false};
    fast_math_internal::parallel_for_static_indexed(
        element_count,
        thread_count,
        [&](std::size_t element, std::size_t) noexcept {
          const std::uint64_t index = indices[element];
          if (index >= space) {
            invalid.store(true, std::memory_order_relaxed);
            return;
          }
          std::uint64_t remaining = index;
          std::uint32_t* row = digits + element * width;
          for (std::uint32_t dimension = 0; dimension < width; ++dimension) {
            row[dimension] =
                static_cast<std::uint32_t>(remaining % prime);
            remaining /= prime;
          }
        });
    if (invalid.load()) {
      throw std::invalid_argument(
          "index is outside the base-p index space p^width");
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

FAST_MATH_API int fast_math_base_p_encode_u64(
    const std::uint32_t* digit_rows,
    std::size_t element_count,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint32_t thread_count,
    std::uint64_t* indices,
    fast_math_base_p_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("stats pointer is null");
    }
    if (element_count != 0 &&
        (digit_rows == nullptr || indices == nullptr)) {
      throw std::invalid_argument("input or output pointer is null");
    }
    if (!is_small_prime(prime)) {
      throw std::invalid_argument("prime must be a prime at most 251");
    }
    if (width < 1 || width > kMaximumBasePWidth) {
      throw std::invalid_argument("width must be between one and sixteen");
    }
    // Also validates that p^width fits in uint64.
    (void)index_space_size(prime, width);

    set_error(error_message, error_message_size, "");
    *stats = {};
    stats->element_count = element_count;
    const auto started = Clock::now();

    std::atomic<bool> invalid{false};
    fast_math_internal::parallel_for_static_indexed(
        element_count,
        thread_count,
        [&](std::size_t element, std::size_t) noexcept {
          const std::uint32_t* row = digit_rows + element * width;
          std::uint64_t index = 0;
          for (std::uint32_t dimension = width; dimension-- > 0;) {
            const std::uint32_t digit = row[dimension];
            if (digit >= prime) {
              invalid.store(true, std::memory_order_relaxed);
              return;
            }
            index = index * prime + digit;
          }
          indices[element] = index;
        });
    if (invalid.load()) {
      throw std::invalid_argument("digit is outside the field F_p");
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

FAST_MATH_API int fast_math_base_p_negation_representatives_u64(
    const std::uint64_t* indices,
    std::size_t element_count,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint32_t thread_count,
    std::uint64_t* representatives,
    fast_math_base_p_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("stats pointer is null");
    }
    if (element_count != 0 &&
        (indices == nullptr || representatives == nullptr)) {
      throw std::invalid_argument("input or output pointer is null");
    }
    if (!is_small_prime(prime)) {
      throw std::invalid_argument("prime must be a prime at most 251");
    }
    if (width < 1 || width > kMaximumBasePWidth) {
      throw std::invalid_argument("width must be between one and sixteen");
    }
    const std::uint64_t space = index_space_size(prime, width);

    set_error(error_message, error_message_size, "");
    *stats = {};
    stats->element_count = element_count;
    const auto started = Clock::now();

    std::atomic<bool> invalid{false};
    fast_math_internal::parallel_for_static_indexed(
        element_count,
        thread_count,
        [&](std::size_t element, std::size_t) noexcept {
          const std::uint64_t index = indices[element];
          if (index >= space) {
            invalid.store(true, std::memory_order_relaxed);
            return;
          }
          representatives[element] =
              negation_representative(index, prime, width);
        });
    if (invalid.load()) {
      throw std::invalid_argument(
          "index is outside the base-p index space p^width");
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

FAST_MATH_API int fast_math_base_p_scalar_class_ids_u64(
    const std::uint64_t* indices,
    std::size_t element_count,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint32_t thread_count,
    std::uint64_t* representatives,
    std::uint32_t* class_ids,
    fast_math_base_p_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("stats pointer is null");
    }
    if (element_count != 0 &&
        (indices == nullptr || representatives == nullptr ||
         class_ids == nullptr)) {
      throw std::invalid_argument("input or output pointer is null");
    }
    if (!is_small_prime(prime)) {
      throw std::invalid_argument("prime must be a prime at most 251");
    }
    if (width < 1 || width > kMaximumBasePWidth) {
      throw std::invalid_argument("width must be between one and sixteen");
    }
    const std::uint64_t space = index_space_size(prime, width);

    set_error(error_message, error_message_size, "");
    *stats = {};
    stats->element_count = element_count;
    const auto started = Clock::now();

    // Pass one: canonical representative per element.
    std::atomic<bool> invalid{false};
    fast_math_internal::parallel_for_static_indexed(
        element_count,
        thread_count,
        [&](std::size_t element, std::size_t) noexcept {
          const std::uint64_t index = indices[element];
          if (index >= space) {
            invalid.store(true, std::memory_order_relaxed);
            return;
          }
          representatives[element] =
              scalar_representative(index, prime, width);
        });
    if (invalid.load()) {
      throw std::invalid_argument(
          "index is outside the base-p index space p^width");
    }

    // Pass two: dense ids are ranks of representatives in ascending order,
    // which makes the labeling deterministic and independent of input order
    // and thread count.
    //
    // When the index space is bounded relative to the batch, label directly:
    // scanning values in ascending order meets every orbit first at its
    // minimal member, so the discovery order is exactly ascending
    // representative order and no ranking pass is required. Multiples are
    // written by direct table indexing instead of binary search.
    const bool direct_labeling =
        space <= kDirectClassTableCap &&
        space / 8 <= element_count;
    if (direct_labeling) {
      std::vector<std::int32_t> id_by_index(
          static_cast<std::size_t>(space),
          -1);
      std::vector<std::uint64_t> representative_table;
      representative_table.reserve(
          static_cast<std::size_t>(space / (prime > 1 ? prime - 1 : 1)) + 1);
      representative_table.push_back(0);
      id_by_index[0] = 0;

      BasePFields fields;
      for (std::uint64_t value = 1; value < space; ++value) {
        if (id_by_index[static_cast<std::size_t>(value)] >= 0) {
          continue;
        }
        decode_index(value, prime, width, fields.digits);
        const auto class_id =
            static_cast<std::int32_t>(representative_table.size());
        representative_table.push_back(value);
        id_by_index[static_cast<std::size_t>(value)] = class_id;
        for (std::uint32_t factor = 2; factor < prime; ++factor) {
          std::uint64_t multiple = 0;
          for (std::uint32_t dimension = width; dimension-- > 0;) {
            multiple = multiple * prime +
                static_cast<std::uint64_t>(
                    fields.digits[dimension] * factor % prime);
          }
          id_by_index[static_cast<std::size_t>(multiple)] = class_id;
        }
      }

      fast_math_internal::parallel_for_static_indexed(
          element_count,
          thread_count,
          [&](std::size_t element, std::size_t) noexcept {
            const auto class_id = static_cast<std::uint32_t>(
                id_by_index[static_cast<std::size_t>(indices[element])]);
            class_ids[element] = class_id;
            representatives[element] = representative_table[class_id];
          });

      stats->elapsed_seconds =
          std::chrono::duration<double>(Clock::now() - started).count();
      return 0;
    }

    std::vector<std::uint64_t> distinct(
        representatives,
        representatives + element_count);
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(
        std::unique(distinct.begin(), distinct.end()),
        distinct.end());

    // Pass three: rank each representative by binary search.
    fast_math_internal::parallel_for_static_indexed(
        element_count,
        thread_count,
        [&](std::size_t element, std::size_t) noexcept {
          const auto position = std::lower_bound(
              distinct.begin(),
              distinct.end(),
              representatives[element]);
          class_ids[element] = static_cast<std::uint32_t>(
              position - distinct.begin());
        });

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
