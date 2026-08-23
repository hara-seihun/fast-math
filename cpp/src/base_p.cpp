#include "fast_math.h"

#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kMaxPrime = 251;
constexpr std::uint32_t kMaxWidth = 16;
constexpr std::uint32_t kUnmarked = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t kNegationKind = 0;
constexpr std::uint32_t kScalarKind = 1;

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

// Validates the shared (prime, width) domain and returns p^width, failing
// instead of wrapping when the encoded space does not fit uint64 codes.
std::uint64_t validated_space_size(
    std::uint32_t prime,
    std::uint32_t width) {
  if (!is_small_prime(prime)) {
    throw std::invalid_argument("prime must be a prime between two and 251");
  }
  if (width < 1 || width > kMaxWidth) {
    throw std::invalid_argument("width must be between one and sixteen");
  }
  std::uint64_t size = 1;
  for (std::uint32_t index = 0; index < width; ++index) {
    if (size > std::numeric_limits<std::uint64_t>::max() / prime) {
      throw std::invalid_argument(
          "p^width does not fit in unsigned 64-bit codes");
    }
    size *= prime;
  }
  return size;
}

// Multiplicative inverse modulo a small prime; zero maps to zero.
std::uint32_t inverse_mod(std::uint32_t value, std::uint32_t prime) {
  if (value % prime == 0) {
    return 0;
  }
  std::int64_t high = value % prime;
  std::int64_t low = prime;
  std::int64_t previous = 1;
  std::int64_t current = 0;
  while (low != 0) {
    const std::int64_t quotient = high / low;
    std::int64_t next = high - quotient * low;
    high = low;
    low = next;
    next = previous - quotient * current;
    previous = current;
    current = next;
  }
  const std::int64_t reduced =
      previous % static_cast<std::int64_t>(prime);
  return static_cast<std::uint32_t>(
      reduced < 0 ? reduced + static_cast<std::int64_t>(prime) : reduced);
}

// Digit-wise negation: each nonzero digit d becomes prime - d. Digits are
// little-endian, so digit zero is the coefficient of p^0.
std::uint64_t negate_code(
    std::uint64_t code,
    std::uint32_t prime,
    std::uint32_t width) {
  std::uint64_t result = 0;
  std::uint64_t weight = 1;
  for (std::uint32_t position = 0; position < width; ++position) {
    const std::uint32_t digit =
        static_cast<std::uint32_t>(code % prime);
    code /= prime;
    if (digit != 0) {
      result += static_cast<std::uint64_t>(prime - digit) * weight;
    }
    weight *= prime;
  }
  return result;
}

void to_digits(
    std::uint64_t code,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint32_t* digits) {
  for (std::uint32_t position = 0; position < width; ++position) {
    digits[position] = static_cast<std::uint32_t>(code % prime);
    code /= prime;
  }
}

std::uint64_t from_digits(
    const std::uint32_t* digits,
    std::uint32_t prime,
    std::uint32_t width) {
  std::uint64_t code = 0;
  for (std::uint32_t position = width; position > 0; --position) {
    code = code * prime + digits[position - 1];
  }
  return code;
}

// Canonical projective representative: scale so the least-significant nonzero
// digit becomes one. The zero vector is its own representative.
std::uint64_t scalar_normal_code(
    std::uint64_t code,
    std::uint32_t prime,
    std::uint32_t width) {
  std::uint32_t digits[kMaxWidth];
  to_digits(code, prime, width, digits);
  std::uint32_t lead = 0;
  while (lead < width && digits[lead] == 0) {
    ++lead;
  }
  if (lead == width) {
    return 0;
  }
  const std::uint32_t scale = inverse_mod(digits[lead], prime);
  if (scale == 1) {
    return code;
  }
  for (std::uint32_t position = 0; position < width; ++position) {
    digits[position] = digits[position] * scale % prime;
  }
  return from_digits(digits, prime, width);
}

}  // namespace

extern "C" {

int fast_math_base_p_digits_u64(
    const std::uint64_t* codes,
    std::size_t code_count,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint8_t* digits,
    fast_math_base_p_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    const std::uint64_t space_size = validated_space_size(prime, width);
    if (code_count != 0 && (codes == nullptr || digits == nullptr)) {
      throw std::invalid_argument("code or digit pointer is null");
    }

    const auto started = Clock::now();
    for (std::size_t index = 0; index < code_count; ++index) {
      std::uint64_t remainder = codes[index];
      if (remainder >= space_size) {
        throw std::invalid_argument(
            "encoded point is outside the p^width space");
      }
      std::uint8_t* row = digits + index * static_cast<std::size_t>(width);
      for (std::uint32_t position = 0; position < width; ++position) {
        row[position] = static_cast<std::uint8_t>(remainder % prime);
        remainder /= prime;
      }
    }

    stats->element_count = code_count;
    stats->class_count = 0;
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

int fast_math_base_p_codes_u64(
    const std::uint8_t* digits,
    std::size_t code_count,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint64_t* codes,
    fast_math_base_p_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    validated_space_size(prime, width);
    if (code_count != 0 && (digits == nullptr || codes == nullptr)) {
      throw std::invalid_argument("digit or code pointer is null");
    }

    const auto started = Clock::now();
    for (std::size_t index = 0; index < code_count; ++index) {
      const std::uint8_t* row =
          digits + index * static_cast<std::size_t>(width);
      std::uint64_t code = 0;
      for (std::uint32_t position = width; position > 0; --position) {
        const std::uint8_t digit = row[position - 1];
        if (digit >= prime) {
          throw std::invalid_argument("digit is outside the prime field");
        }
        code = code * prime + digit;
      }
      codes[index] = code;
    }

    stats->element_count = code_count;
    stats->class_count = 0;
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

int fast_math_base_p_negation_codes_u64(
    const std::uint64_t* codes,
    std::size_t code_count,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint64_t* negated,
    fast_math_base_p_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    const std::uint64_t space_size = validated_space_size(prime, width);
    if (code_count != 0 && (codes == nullptr || negated == nullptr)) {
      throw std::invalid_argument("input or output pointer is null");
    }

    const auto started = Clock::now();
    for (std::size_t index = 0; index < code_count; ++index) {
      if (codes[index] >= space_size) {
        throw std::invalid_argument(
            "encoded point is outside the p^width space");
      }
      negated[index] = negate_code(codes[index], prime, width);
    }

    stats->element_count = code_count;
    stats->class_count = 0;
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

int fast_math_base_p_scalar_normals_u64(
    const std::uint64_t* codes,
    std::size_t code_count,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint64_t* normals,
    fast_math_base_p_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    const std::uint64_t space_size = validated_space_size(prime, width);
    if (code_count != 0 && (codes == nullptr || normals == nullptr)) {
      throw std::invalid_argument("input or output pointer is null");
    }

    const auto started = Clock::now();
    for (std::size_t index = 0; index < code_count; ++index) {
      if (codes[index] >= space_size) {
        throw std::invalid_argument(
            "encoded point is outside the p^width space");
      }
      normals[index] = scalar_normal_code(codes[index], prime, width);
    }

    stats->element_count = code_count;
    stats->class_count = 0;
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

int fast_math_base_p_class_table_u64(
    std::uint32_t prime,
    std::uint32_t width,
    std::uint32_t class_kind,
    std::uint32_t* class_ids,
    std::uint64_t* representatives,
    std::uint32_t* class_counts,
    std::size_t representative_capacity,
    fast_math_base_p_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    const std::uint64_t space_size = validated_space_size(prime, width);
    if (class_kind != kNegationKind && class_kind != kScalarKind) {
      throw std::invalid_argument(
          "class_kind must be zero for negation classes or one for scalar "
          "classes");
    }
    if (space_size > static_cast<std::uint64_t>(kUnmarked) + 1ULL) {
      throw std::invalid_argument(
          "p^width exceeds the uint32 class-id range");
    }
    if (class_ids == nullptr ||
        (representative_capacity != 0 &&
         (representatives == nullptr || class_counts == nullptr))) {
      throw std::invalid_argument(
          "class-id, representative, or count pointer is null");
    }
    std::uint64_t expected_classes;
    if (class_kind == kNegationKind) {
      expected_classes = prime == 2 ? space_size : (space_size + 1) / 2;
    } else {
      expected_classes = (space_size - 1) / (prime - 1) + 1;
    }
    if (expected_classes > representative_capacity) {
      throw std::invalid_argument(
          "representative buffer is smaller than the class count");
    }

    const auto started = Clock::now();
    std::memset(
        class_ids,
        0xFF,
        static_cast<std::size_t>(space_size) * sizeof(std::uint32_t));
    std::uint32_t next_id = 0;

    if (class_kind == kNegationKind) {
      // Ascending order makes an unmarked code the minimum of its pair.
      for (std::uint64_t code = 0; code < space_size; ++code) {
        if (class_ids[code] != kUnmarked) {
          continue;
        }
        const std::uint32_t id = next_id++;
        representatives[id] = code;
        class_ids[code] = id;
        class_counts[id] = 1;
        const std::uint64_t partner = negate_code(code, prime, width);
        if (partner != code) {
          class_ids[partner] = id;
          class_counts[id] = 2;
        }
      }
    } else {
      // An unmarked code whose canonical form is itself starts one class;
      // its multiples are marked digit-wise. A code whose canonical form is
      // larger is marked when that canonical form is reached.
      std::uint32_t digits[kMaxWidth];
      for (std::uint64_t code = 0; code < space_size; ++code) {
        if (class_ids[code] != kUnmarked) {
          continue;
        }
        if (scalar_normal_code(code, prime, width) != code) {
          continue;
        }
        const std::uint32_t id = next_id++;
        representatives[id] = code;
        std::uint32_t members = 0;
        to_digits(code, prime, width, digits);
        for (std::uint32_t unit = 1; unit < prime; ++unit) {
          std::uint32_t scaled[kMaxWidth];
          for (std::uint32_t position = 0; position < width; ++position) {
            scaled[position] = digits[position] * unit % prime;
          }
          const std::uint64_t multiple =
              from_digits(scaled, prime, width);
          if (class_ids[multiple] == kUnmarked) {
            class_ids[multiple] = id;
            ++members;
          }
        }
        class_counts[id] = members;
      }
    }

    if (static_cast<std::uint64_t>(next_id) != expected_classes) {
      throw std::logic_error("class enumeration disagrees with the formula");
    }

    stats->element_count = space_size;
    stats->class_count = next_id;
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
