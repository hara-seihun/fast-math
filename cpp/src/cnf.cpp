#include "fast_math.h"
#include "parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
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
    throw std::overflow_error("CNF batch shape overflows");
  }
  return left * right;
}

}  // namespace

struct fast_math_cnf_plan {
  std::uint32_t variable_count;
  std::uint32_t word_count;
  std::vector<std::uint64_t> clause_offsets;
  std::vector<std::int32_t> literals;
};

extern "C" int fast_math_cnf_create_i32(
    const std::uint64_t* clause_offsets,
    std::size_t clause_count,
    const std::int32_t* literals,
    std::size_t literal_count,
    std::uint32_t variable_count,
    fast_math_cnf_plan** plan,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (plan == nullptr || clause_offsets == nullptr || variable_count == 0 ||
        (literal_count != 0 && literals == nullptr)) {
      throw std::invalid_argument("CNF plan pointer or shape is invalid");
    }
    *plan = nullptr;
    if (clause_offsets[0] != 0 || clause_offsets[clause_count] != literal_count) {
      throw std::invalid_argument("CNF clause offsets are inconsistent");
    }
    for (std::size_t clause = 0; clause < clause_count; ++clause) {
      if (clause_offsets[clause] > clause_offsets[clause + 1]) {
        throw std::invalid_argument("CNF clause offsets are not monotone");
      }
    }
    for (std::size_t index = 0; index < literal_count; ++index) {
      const auto literal = static_cast<std::int64_t>(literals[index]);
      const auto variable = literal < 0 ? -literal : literal;
      if (variable == 0 || variable > variable_count) {
        throw std::invalid_argument("CNF literal is outside the variable range");
      }
    }
    auto candidate = new fast_math_cnf_plan{};
    try {
      candidate->variable_count = variable_count;
      candidate->word_count = (variable_count + 63) / 64;
      candidate->clause_offsets.assign(
          clause_offsets, clause_offsets + clause_count + 1);
      if (literal_count != 0) {
        candidate->literals.assign(literals, literals + literal_count);
      }
    } catch (...) {
      delete candidate;
      throw;
    }
    *plan = candidate;
    set_error(error_message, error_message_size, "");
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(error_message, error_message_size, "unknown CNF plan error");
    return 2;
  }
}

extern "C" void fast_math_cnf_destroy(fast_math_cnf_plan* plan) {
  delete plan;
}

extern "C" int fast_math_cnf_evaluate_u64(
    const fast_math_cnf_plan* plan,
    const std::uint64_t* assignment_words,
    std::size_t assignment_count,
    std::uint32_t word_count,
    std::uint32_t thread_count,
    std::uint8_t* satisfied,
    std::int64_t* first_unsatisfied_clause,
    fast_math_cnf_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (plan == nullptr || stats == nullptr || word_count != plan->word_count ||
        (assignment_count != 0 &&
         (assignment_words == nullptr || satisfied == nullptr ||
          first_unsatisfied_clause == nullptr))) {
      throw std::invalid_argument("CNF evaluation pointer or shape is invalid");
    }
    checked_product(assignment_count, word_count);
    const auto final_bits = plan->variable_count % 64;
    if (final_bits != 0) {
      const auto valid_mask = (std::uint64_t{1} << final_bits) - 1;
      for (std::size_t assignment = 0;
           assignment < assignment_count;
           ++assignment) {
        if ((assignment_words[assignment * word_count + word_count - 1] &
             ~valid_mask) != 0) {
          throw std::invalid_argument("assignment contains an out-of-range bit");
        }
      }
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    std::vector<std::uint64_t> inspected_literals(assignment_count);
    fast_math_internal::parallel_for_static(
        assignment_count,
        thread_count,
        [&](std::size_t assignment) {
          const auto* words = assignment_words + assignment * word_count;
          std::int64_t first = -1;
          std::uint64_t inspected = 0;
          const auto clause_count = plan->clause_offsets.size() - 1;
          for (std::size_t clause = 0; clause < clause_count; ++clause) {
            bool clause_satisfied = false;
            for (auto offset = plan->clause_offsets[clause];
                 offset < plan->clause_offsets[clause + 1];
                 ++offset) {
              ++inspected;
              const auto literal = plan->literals[offset];
              const auto magnitude = literal < 0
                  ? -static_cast<std::int64_t>(literal)
                  : static_cast<std::int64_t>(literal);
              const auto variable = static_cast<std::uint32_t>(magnitude - 1);
              const bool value =
                  ((words[variable / 64] >> (variable % 64)) & 1U) != 0;
              if ((literal > 0 && value) || (literal < 0 && !value)) {
                clause_satisfied = true;
                break;
              }
            }
            if (!clause_satisfied) {
              first = static_cast<std::int64_t>(clause);
              break;
            }
          }
          first_unsatisfied_clause[assignment] = first;
          satisfied[assignment] = first < 0 ? 1 : 0;
          inspected_literals[assignment] = inspected;
        });
    stats->variable_count = plan->variable_count;
    stats->clause_count = plan->clause_offsets.size() - 1;
    stats->literal_count = plan->literals.size();
    stats->assignment_count = assignment_count;
    for (const auto count : inspected_literals) {
      stats->inspected_literal_count += count;
    }
    stats->thread_count = fast_math_internal::parallel_worker_count(
        assignment_count, thread_count);
    stats->elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(error_message, error_message_size, "unknown CNF evaluation error");
    return 2;
  }
}
