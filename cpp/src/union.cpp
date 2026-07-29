#include "fast_math.h"

#include <bit>
#include <chrono>
#include <cstring>
#include <exception>
#include <stdexcept>

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

bool is_union_closed(
    std::uint64_t family,
    fast_math_union_stats* stats) {
  auto left_members = family;
  while (left_members != 0) {
    const auto left = std::countr_zero(left_members);
    left_members &= left_members - 1;

    auto right_members = left_members;
    while (right_members != 0) {
      const auto right = std::countr_zero(right_members);
      right_members &= right_members - 1;
      const auto united = left | right;
      if (united == right) {
        continue;
      }
      stats->pair_checks += 1;
      if ((family & (std::uint64_t{1} << united)) == 0) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

extern "C" {

int fast_math_union_closed_family_masks_u64(
    const std::uint64_t* family_masks,
    std::size_t family_count,
    std::uint32_t ground_size,
    std::uint8_t* closed,
    fast_math_union_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("stats pointer is null");
    }
    if (family_count != 0 &&
        (family_masks == nullptr || closed == nullptr)) {
      throw std::invalid_argument("family or output pointer is null");
    }
    if (ground_size > 6) {
      throw std::invalid_argument("ground_size must be at most six");
    }

    set_error(error_message, error_message_size, "");
    *stats = {};
    stats->family_count = family_count;
    const auto started = Clock::now();
    const auto set_count = std::uint32_t{1} << ground_size;
    const auto allowed_mask =
        set_count == 64
            ? ~std::uint64_t{0}
            : (std::uint64_t{1} << set_count) - 1;

    for (std::size_t index = 0; index < family_count; ++index) {
      if ((family_masks[index] & ~allowed_mask) != 0) {
        throw std::invalid_argument(
            "family mask contains a set outside the ground set");
      }
      closed[index] = is_union_closed(family_masks[index], stats);
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

}
