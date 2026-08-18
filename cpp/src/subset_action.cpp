#include "fast_math.h"
#include "parallel.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void set_error(char* output, std::size_t capacity, const std::string& message) {
  if (output == nullptr || capacity == 0) return;
  const auto count = std::min(capacity - 1, message.size());
  std::memcpy(output, message.data(), count);
  output[count] = '\0';
}

std::size_t checked_product(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("packed subset action allocation overflows");
  }
  return left * right;
}

}  // namespace

struct fast_math_subset_action {
  std::uint32_t degree;
  std::uint32_t byte_count;
  std::size_t permutation_count;
  std::uint64_t valid_mask;
  std::vector<std::uint64_t> byte_luts;
};

extern "C" int fast_math_subset_action_create_u32(
    const std::uint32_t* permutations,
    std::size_t permutation_count,
    std::uint32_t degree,
    fast_math_subset_action** plan,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (permutations == nullptr || plan == nullptr || stats == nullptr) {
      throw std::invalid_argument("packed subset action pointer is null");
    }
    *plan = nullptr;
    *stats = {};
    set_error(error_message, error_message_size, "");
    if (permutation_count == 0) {
      throw std::invalid_argument("packed subset action requires a permutation");
    }
    if (degree == 0 || degree > 64) {
      throw std::invalid_argument("packed subset action degree must be in 1..64");
    }
    const auto started = Clock::now();
    const auto row_size = static_cast<std::size_t>(degree);
    std::vector<std::uint8_t> seen(degree);
    for (std::size_t row = 0; row < permutation_count; ++row) {
      std::fill(seen.begin(), seen.end(), 0);
      for (std::uint32_t point = 0; point < degree; ++point) {
        const auto image = permutations[row * row_size + point];
        if (image >= degree || seen[image] != 0) {
          throw std::invalid_argument(
              "packed subset action row is not a permutation");
        }
        seen[image] = 1;
      }
    }

    const auto byte_count = (degree + 7) / 8;
    const auto table_count = checked_product(
        checked_product(permutation_count, byte_count), 256);
    auto candidate = std::make_unique<fast_math_subset_action>();
    candidate->degree = degree;
    candidate->byte_count = byte_count;
    candidate->permutation_count = permutation_count;
    candidate->valid_mask = degree == 64
        ? std::numeric_limits<std::uint64_t>::max()
        : (std::uint64_t{1} << degree) - 1;
    candidate->byte_luts.assign(table_count, 0);
    for (std::size_t row = 0; row < permutation_count; ++row) {
      for (std::uint32_t byte = 0; byte < candidate->byte_count; ++byte) {
        auto* table = candidate->byte_luts.data() +
            (row * candidate->byte_count + byte) * 256;
        for (std::uint32_t value = 1; value < 256; ++value) {
          const auto low = value & (~value + 1);
          const auto bit = static_cast<std::uint32_t>(std::countr_zero(low));
          const auto point = byte * 8 + bit;
          table[value] = table[value ^ low];
          if (point < degree) {
            const auto image = permutations[row * row_size + point];
            table[value] |= std::uint64_t{1} << image;
          }
        }
      }
    }
    *plan = candidate.release();
    stats->degree = degree;
    stats->generator_count = permutation_count;
    stats->item_count = table_count;
    stats->elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(error_message, error_message_size, "unknown packed subset action error");
    return 2;
  }
}

extern "C" void fast_math_subset_action_destroy(
    fast_math_subset_action* plan) {
  delete plan;
}

extern "C" int fast_math_subset_action_canonicalize_u64(
    const fast_math_subset_action* plan,
    const std::uint64_t* masks,
    std::size_t mask_count,
    std::uint32_t thread_count,
    std::uint64_t* canonical_masks,
    std::uint8_t* is_canonical,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (plan == nullptr || stats == nullptr) {
      throw std::invalid_argument("packed subset action plan or stats is null");
    }
    if (mask_count != 0 &&
        (masks == nullptr || (canonical_masks == nullptr && is_canonical == nullptr))) {
      throw std::invalid_argument("packed subset action input or output is null");
    }
    for (std::size_t index = 0; index < mask_count; ++index) {
      if ((masks[index] & ~plan->valid_mask) != 0) {
        throw std::invalid_argument("packed subset mask has an out-of-range bit");
      }
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const bool flags_only = canonical_masks == nullptr;
    fast_math_internal::parallel_for_static(
        mask_count,
        thread_count,
        [&](std::size_t index) {
          const auto mask = masks[index];
          std::uint64_t best = mask;
          bool minimal = true;
          for (std::size_t row = 0; row < plan->permutation_count; ++row) {
            const auto* tables = plan->byte_luts.data() +
                row * plan->byte_count * 256;
            auto remaining = mask;
            std::uint64_t image = 0;
            for (std::uint32_t byte = 0; byte < plan->byte_count; ++byte) {
              image |= tables[byte * 256 + (remaining & 255)];
              remaining >>= 8;
            }
            if (image < best) best = image;
            if (image < mask) {
              minimal = false;
              if (flags_only) break;
            }
          }
          if (canonical_masks != nullptr) canonical_masks[index] = best;
          if (is_canonical != nullptr) is_canonical[index] = minimal ? 1 : 0;
        });
    stats->degree = plan->degree;
    stats->generator_count = plan->permutation_count;
    stats->item_count = mask_count;
    stats->thread_count = fast_math_internal::parallel_worker_count(
        mask_count, thread_count);
    stats->elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(error_message, error_message_size, "unknown packed subset action error");
    return 2;
  }
}
