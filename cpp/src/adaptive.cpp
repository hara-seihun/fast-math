#include "fast_math.h"

#include "parallel.hpp"

#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using fast_math_internal::parallel_for_dynamic_indexed;
using fast_math_internal::parallel_worker_count;

constexpr std::uint32_t kMaxCoordinates = 20;

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

std::size_t ternary_total(std::uint32_t coordinate_count) {
  std::size_t total = 1;
  for (std::uint32_t index = 0; index < coordinate_count; ++index) {
    total *= 3;
  }
  return total;
}

// Restriction code r encodes coordinate i in base-three digit i:
//   0 -> free, 1 -> X_i = -1, 2 -> X_i = +1.
// Point index x encodes coordinate i in bit i: 0 -> X_i = -1, 1 -> X_i = +1.
std::size_t point_to_restriction(
    std::size_t point,
    std::uint32_t coordinate_count) {
  std::size_t code = 0;
  std::size_t power = 1;
  for (std::uint32_t index = 0; index < coordinate_count; ++index) {
    code += power * (((point >> index) & 1) != 0 ? 2 : 1);
    power *= 3;
  }
  return code;
}

// Aggregates fully fixed leaf values up the restriction lattice: after the pass
// for coordinate i every entry whose free coordinates lie in {0..i} is exact.
template <typename Value>
void free_coordinate_transform(
    Value* values,
    std::uint32_t coordinate_count,
    std::size_t total) {
  std::size_t power = 1;
  for (std::uint32_t index = 0; index < coordinate_count; ++index) {
    const std::size_t block = power * 3;
    for (std::size_t high = 0; high < total; high += block) {
      Value* base = values + high;
      for (std::size_t low = 0; low < power; ++low) {
        base[low] = base[low + power] + base[low + 2 * power];
      }
    }
    power = block;
  }
}

struct Odometer {
  explicit Odometer(std::uint32_t coordinate_count)
      : digits(coordinate_count, 2), free_count(0) {}

  void step_down() {
    std::size_t index = 0;
    while (digits[index] == 0) {
      digits[index] = 2;
      --free_count;
      ++index;
    }
    if (--digits[index] == 0) {
      ++free_count;
    }
  }

  std::vector<std::uint8_t> digits;
  std::uint32_t free_count;
};

struct FloatWorkspace {
  std::vector<double> sums;
  std::vector<double> squares;
};

struct ExactWorkspace {
  std::vector<std::int64_t> sums;
  std::vector<std::int64_t> squares;
};

void solve_float_target(
    const double* table,
    std::uint32_t coordinate_count,
    std::size_t total,
    double zero_tolerance,
    FloatWorkspace& workspace,
    double* area,
    std::int32_t* first_coordinate,
    double* variances,
    double* areas_by_restriction,
    std::int8_t* policies) {
  const std::size_t point_count = std::size_t{1} << coordinate_count;
  auto& sums = workspace.sums;
  auto& squares = workspace.squares;
  sums.assign(total, 0.0);
  squares.assign(total, 0.0);
  for (std::size_t point = 0; point < point_count; ++point) {
    const std::size_t code = point_to_restriction(point, coordinate_count);
    const double value = table[point];
    sums[code] = value;
    squares[code] = value * value;
  }
  free_coordinate_transform(sums.data(), coordinate_count, total);
  free_coordinate_transform(squares.data(), coordinate_count, total);

  std::vector<double> powers(coordinate_count + 1);
  std::size_t stride = 1;
  std::vector<std::size_t> strides(coordinate_count);
  for (std::uint32_t index = 0; index < coordinate_count; ++index) {
    strides[index] = stride;
    stride *= 3;
  }
  for (std::uint32_t index = 0; index <= coordinate_count; ++index) {
    powers[index] = static_cast<double>(std::size_t{1} << index);
  }

  // squares[] is reused as the Bellman value array; descending restriction
  // order visits every child before its parent because fixing a free digit
  // strictly increases the code.
  Odometer odometer(coordinate_count);
  for (std::size_t code = total; code-- > 0;) {
    if (code + 1 != total) {
      odometer.step_down();
    }
    const std::uint32_t free_count = odometer.free_count;
    const double cell = powers[free_count];
    const double mean = sums[code] / cell;
    const double variance = squares[code] / cell - mean * mean;
    if (variances != nullptr) {
      variances[code] = variance;
    }
    if (variance <= zero_tolerance || free_count == 0) {
      squares[code] = 0.0;
      if (areas_by_restriction != nullptr) {
        areas_by_restriction[code] = 0.0;
      }
      if (policies != nullptr) {
        policies[code] = -1;
      }
      if (code == 0) {
        *first_coordinate = -1;
      }
      continue;
    }
    double best = std::numeric_limits<double>::infinity();
    std::int32_t choice = -1;
    for (std::uint32_t index = 0; index < coordinate_count; ++index) {
      if (odometer.digits[index] != 0) {
        continue;
      }
      const std::size_t offset = strides[index];
      const double child =
          squares[code + offset] + squares[code + 2 * offset];
      if (child < best) {
        best = child;
        choice = static_cast<std::int32_t>(index);
      }
    }
    const double value = variance + 0.5 * best;
    squares[code] = value;
    if (areas_by_restriction != nullptr) {
      areas_by_restriction[code] = value;
    }
    if (policies != nullptr) {
      policies[code] = static_cast<std::int8_t>(choice);
    }
    if (code == 0) {
      *first_coordinate = choice;
    }
  }
  *area = squares[0];
}

bool checked_int64(__int128 value, std::int64_t* out) {
  constexpr __int128 limit = static_cast<__int128>(
      std::numeric_limits<std::int64_t>::max());
  if (value > limit || value < -limit) {
    return false;
  }
  *out = static_cast<std::int64_t>(value);
  return true;
}

// Exact scale: variance numerators use denominator 4^n and area numerators use
// denominator 2^(3n). The recursion stores B(r) = A(r) * 4^n * 2^(n - fixed),
// which removes every division.
bool solve_exact_target(
    const std::int64_t* table,
    std::uint32_t coordinate_count,
    std::size_t total,
    ExactWorkspace& workspace,
    std::int64_t* area,
    std::int32_t* first_coordinate,
    std::int64_t* variances,
    std::int64_t* areas_by_restriction,
    std::int8_t* policies) {
  const std::size_t point_count = std::size_t{1} << coordinate_count;
  auto& sums = workspace.sums;
  auto& squares = workspace.squares;
  sums.assign(total, 0);
  squares.assign(total, 0);
  for (std::size_t point = 0; point < point_count; ++point) {
    const std::size_t code = point_to_restriction(point, coordinate_count);
    const std::int64_t value = table[point];
    sums[code] = value;
    const __int128 square = static_cast<__int128>(value) * value;
    if (!checked_int64(square, &squares[code])) {
      return false;
    }
  }
  free_coordinate_transform(sums.data(), coordinate_count, total);
  free_coordinate_transform(squares.data(), coordinate_count, total);

  std::vector<std::size_t> strides(coordinate_count);
  std::size_t stride = 1;
  for (std::uint32_t index = 0; index < coordinate_count; ++index) {
    strides[index] = stride;
    stride *= 3;
  }

  Odometer odometer(coordinate_count);
  for (std::size_t code = total; code-- > 0;) {
    if (code + 1 != total) {
      odometer.step_down();
    }
    const std::uint32_t free_count = odometer.free_count;
    const std::uint32_t fixed_count = coordinate_count - free_count;
    const __int128 cell = static_cast<__int128>(std::size_t{1} << free_count);
    const __int128 sum = sums[code];
    const __int128 raw = static_cast<__int128>(squares[code]) * cell - sum * sum;
    const __int128 variance = raw << (2 * fixed_count);
    if (variances != nullptr && !checked_int64(variance, &variances[code])) {
      return false;
    }
    if (variance == 0 || free_count == 0) {
      squares[code] = 0;
      if (areas_by_restriction != nullptr) {
        areas_by_restriction[code] = 0;
      }
      if (policies != nullptr) {
        policies[code] = -1;
      }
      if (code == 0) {
        *first_coordinate = -1;
      }
      continue;
    }
    __int128 best = 0;
    bool have_best = false;
    std::int32_t choice = -1;
    for (std::uint32_t index = 0; index < coordinate_count; ++index) {
      if (odometer.digits[index] != 0) {
        continue;
      }
      const std::size_t offset = strides[index];
      const __int128 child = static_cast<__int128>(squares[code + offset]) +
          static_cast<__int128>(squares[code + 2 * offset]);
      if (!have_best || child < best) {
        best = child;
        have_best = true;
        choice = static_cast<std::int32_t>(index);
      }
    }
    const __int128 value = (variance << free_count) + best;
    if (!checked_int64(value, &squares[code])) {
      return false;
    }
    if (areas_by_restriction != nullptr) {
      const __int128 scaled = value << fixed_count;
      if (!checked_int64(scaled, &areas_by_restriction[code])) {
        return false;
      }
    }
    if (policies != nullptr) {
      policies[code] = static_cast<std::int8_t>(choice);
    }
    if (code == 0) {
      *first_coordinate = choice;
    }
  }
  *area = squares[0];
  return true;
}

}  // namespace

extern "C" {

int fast_math_adaptive_area_f64(
    const double* tables,
    std::size_t target_count,
    std::uint32_t coordinate_count,
    double zero_tolerance,
    std::uint32_t thread_count,
    double* areas,
    std::int32_t* first_coordinates,
    double* variances,
    double* areas_by_restriction,
    std::int8_t* policies,
    fast_math_adaptive_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (coordinate_count == 0 || coordinate_count > kMaxCoordinates) {
      set_error(
          error_message,
          error_message_size,
          "coordinate_count must be between one and twenty");
      return 1;
    }
    if (target_count != 0 &&
        (tables == nullptr || areas == nullptr ||
         first_coordinates == nullptr)) {
      set_error(error_message, error_message_size, "null required buffer");
      return 2;
    }
    const auto start = Clock::now();
    const std::size_t total = ternary_total(coordinate_count);
    const std::size_t points = std::size_t{1} << coordinate_count;
    const auto workers = parallel_worker_count(target_count, thread_count);
    std::vector<FloatWorkspace> workspaces(workers);
    parallel_for_dynamic_indexed(
        target_count,
        workers,
        [&](std::size_t target, std::size_t worker) {
          solve_float_target(
              tables + target * points,
              coordinate_count,
              total,
              zero_tolerance,
              workspaces[worker],
              areas + target,
              first_coordinates + target,
              variances == nullptr ? nullptr : variances + target * total,
              areas_by_restriction == nullptr
                  ? nullptr
                  : areas_by_restriction + target * total,
              policies == nullptr ? nullptr : policies + target * total);
        });
    if (stats != nullptr) {
      stats->target_count = target_count;
      stats->restriction_count = total;
      stats->coordinate_count = coordinate_count;
      stats->worker_count = workers;
      stats->elapsed_seconds =
          std::chrono::duration<double>(Clock::now() - start).count();
    }
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 3;
  } catch (...) {
    set_error(error_message, error_message_size, "unknown failure");
    return 3;
  }
}

int fast_math_adaptive_area_exact_i64(
    const std::int64_t* tables,
    std::size_t target_count,
    std::uint32_t coordinate_count,
    std::uint32_t thread_count,
    std::int64_t* areas,
    std::int32_t* first_coordinates,
    std::int64_t* variances,
    std::int64_t* areas_by_restriction,
    std::int8_t* policies,
    fast_math_adaptive_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (coordinate_count == 0 || coordinate_count > kMaxCoordinates) {
      set_error(
          error_message,
          error_message_size,
          "coordinate_count must be between one and twenty");
      return 1;
    }
    if (target_count != 0 &&
        (tables == nullptr || areas == nullptr ||
         first_coordinates == nullptr)) {
      set_error(error_message, error_message_size, "null required buffer");
      return 2;
    }
    const auto start = Clock::now();
    const std::size_t total = ternary_total(coordinate_count);
    const std::size_t points = std::size_t{1} << coordinate_count;
    const auto workers = parallel_worker_count(target_count, thread_count);
    std::vector<ExactWorkspace> workspaces(workers);
    std::vector<std::uint8_t> overflow(target_count, 0);
    parallel_for_dynamic_indexed(
        target_count,
        workers,
        [&](std::size_t target, std::size_t worker) {
          const bool ok = solve_exact_target(
              tables + target * points,
              coordinate_count,
              total,
              workspaces[worker],
              areas + target,
              first_coordinates + target,
              variances == nullptr ? nullptr : variances + target * total,
              areas_by_restriction == nullptr
                  ? nullptr
                  : areas_by_restriction + target * total,
              policies == nullptr ? nullptr : policies + target * total);
          overflow[target] = ok ? 0 : 1;
        });
    for (std::size_t target = 0; target < target_count; ++target) {
      if (overflow[target] != 0) {
        const std::string message =
            "exact adaptive area overflowed int64 for target " +
            std::to_string(target);
        set_error(error_message, error_message_size, message.c_str());
        return 4;
      }
    }
    if (stats != nullptr) {
      stats->target_count = target_count;
      stats->restriction_count = total;
      stats->coordinate_count = coordinate_count;
      stats->worker_count = workers;
      stats->elapsed_seconds =
          std::chrono::duration<double>(Clock::now() - start).count();
    }
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 3;
  } catch (...) {
    set_error(error_message, error_message_size, "unknown failure");
    return 3;
  }
}

}  // extern "C"
