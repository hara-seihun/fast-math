#include "fast_math.h"

#include <bit>
#include <chrono>
#include <cstring>
#include <exception>
#include <stdexcept>
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

// Exact binomial table for n <= 64 built once per call. C(64, 32) is the
// largest entry at 1832624140942590534, which fits in uint64.
struct BinomialTable {
  static constexpr std::uint32_t kMaxN = 64;
  std::uint64_t values[kMaxN + 1][kMaxN + 1];

  BinomialTable() {
    std::memset(values, 0, sizeof(values));
    for (std::uint32_t n = 0; n <= kMaxN; ++n) {
      values[n][0] = 1;
      for (std::uint32_t k = 1; k <= n; ++k) {
        values[n][k] = values[n - 1][k - 1] + values[n - 1][k];
      }
    }
  }
};

std::uint64_t rank_mask(const BinomialTable& table, std::uint64_t mask) {
  std::uint64_t rank = 0;
  std::uint32_t position = 0;
  while (mask != 0) {
    const auto bit = static_cast<std::uint32_t>(std::countr_zero(mask));
    mask &= mask - 1;
    position += 1;
    rank += table.values[bit][position];
  }
  return rank;
}

std::uint64_t unrank_mask(
    const BinomialTable& table,
    std::uint32_t element_count,
    std::uint32_t weight,
    std::uint64_t rank,
    std::uint32_t* popcount) {
  std::uint64_t mask = 0;
  std::uint32_t remaining = weight;
  for (std::uint32_t probe = element_count;
       probe != 0 && remaining != 0;
       --probe) {
    const std::uint32_t bit = probe - 1;
    const auto below = table.values[bit][remaining];
    if (rank >= below) {
      rank -= below;
      mask |= std::uint64_t{1} << bit;
      remaining -= 1;
    }
  }
  if (popcount != nullptr) {
    *popcount = weight - remaining;
  }
  return mask;
}

std::uint64_t binomial(
    const BinomialTable& table,
    std::uint32_t n,
    std::uint32_t k) {
  if (k > n) {
    return 0;
  }
  return table.values[n][k];
}

void validate_element_count(std::uint32_t element_count) {
  if (element_count == 0 || element_count > BinomialTable::kMaxN) {
    throw std::invalid_argument("element_count must be between 1 and 64");
  }
}

void validate_mask_bits(std::uint64_t mask, std::uint32_t element_count) {
  const auto used_bits =
      64 - std::countl_zero(mask);
  if (mask != 0 && used_bits > element_count) {
    throw std::invalid_argument(
        "subset mask contains an element outside the element range");
  }
}

}  // namespace

extern "C" {

int fast_math_colex_rank_u64(
    const std::uint64_t* subset_masks,
    std::size_t subset_count,
    std::uint32_t element_count,
    std::uint64_t* ranks,
    fast_math_colex_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("stats pointer is null");
    }
    if (subset_count != 0 &&
        (subset_masks == nullptr || ranks == nullptr)) {
      throw std::invalid_argument("subset or output pointer is null");
    }
    validate_element_count(element_count);
    set_error(error_message, error_message_size, "");
    *stats = {};
    stats->subset_count = subset_count;
    const auto started = Clock::now();

    BinomialTable table;
    for (std::size_t index = 0; index < subset_count; ++index) {
      const auto mask = subset_masks[index];
      validate_mask_bits(mask, element_count);
      ranks[index] = rank_mask(table, mask);
    }
    stats->binomial_evaluations = subset_count;

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

int fast_math_colex_unrank_u64(
    const std::uint64_t* ranks,
    std::size_t rank_count,
    std::uint32_t element_count,
    std::uint32_t weight,
    std::uint64_t* subset_masks,
    fast_math_colex_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("stats pointer is null");
    }
    if (rank_count != 0 &&
        (ranks == nullptr || subset_masks == nullptr)) {
      throw std::invalid_argument("rank or output pointer is null");
    }
    validate_element_count(element_count);
    if (weight > element_count) {
      throw std::invalid_argument("weight must not exceed element_count");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    stats->subset_count = rank_count;
    const auto started = Clock::now();

    BinomialTable table;
    const auto total = binomial(table, element_count, weight);
    for (std::size_t index = 0; index < rank_count; ++index) {
      const auto rank = ranks[index];
      if (rank >= total) {
        throw std::invalid_argument(
            "colex rank is outside the valid range for the weight");
      }
      subset_masks[index] =
          unrank_mask(table, element_count, weight, rank, nullptr);
    }
    stats->binomial_evaluations = rank_count;

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

int fast_math_colex_visit_u64(
    const std::uint64_t* subset_masks,
    std::size_t subset_count,
    std::uint32_t element_count,
    std::uint32_t weight,
    std::uint64_t* visited_words,
    std::size_t visited_word_count,
    std::uint8_t* newly_visited,
    fast_math_colex_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("stats pointer is null");
    }
    if (subset_count != 0 &&
        (subset_masks == nullptr || newly_visited == nullptr)) {
      throw std::invalid_argument("subset or output pointer is null");
    }
    validate_element_count(element_count);
    if (weight > element_count) {
      throw std::invalid_argument("weight must not exceed element_count");
    }
    if (visited_words == nullptr) {
      throw std::invalid_argument("visited bitmap pointer is null");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    stats->subset_count = subset_count;
    const auto started = Clock::now();

    BinomialTable table;
    const auto total = binomial(table, element_count, weight);
    if (visited_word_count * 64 < total) {
      throw std::invalid_argument(
          "visited bitmap is too small for the weight class");
    }

    for (std::size_t index = 0; index < subset_count; ++index) {
      const auto mask = subset_masks[index];
      validate_mask_bits(mask, element_count);
      if (static_cast<std::uint32_t>(std::popcount(mask)) != weight) {
        throw std::invalid_argument(
            "subset mask does not have the declared weight");
      }
      const auto rank = rank_mask(table, mask);
      const auto word = rank >> 6;
      const auto bit = std::uint64_t{1} << (rank & 63);
      const auto was_set = (visited_words[word] & bit) != 0;
      visited_words[word] |= bit;
      newly_visited[index] = was_set ? 0 : 1;
      stats->newly_visited += was_set ? 0 : 1;
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
