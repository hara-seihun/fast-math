#include "fast_math.h"

#include "parallel.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Permutation = std::vector<std::uint32_t>;

struct PermutationHash {
  std::size_t operator()(const Permutation& permutation) const noexcept {
    std::size_t value = 1469598103934665603ULL;
    for (const auto image : permutation) {
      value ^= image;
      value *= 1099511628211ULL;
    }
    return value;
  }
};

struct ChainLevel {
  std::uint32_t base = 0;
  std::vector<Permutation> generators;
  std::vector<std::uint32_t> orbit;
  std::vector<std::int32_t> orbit_index;
  std::vector<Permutation> transversals;
  std::vector<Permutation> inverse_transversals;
};

struct StabilizerChain {
  std::uint32_t degree = 0;
  std::vector<ChainLevel> levels;
};

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

Permutation identity_permutation(std::uint32_t degree) {
  Permutation identity(degree);
  std::iota(identity.begin(), identity.end(), 0);
  return identity;
}

bool is_identity(const Permutation& permutation) {
  for (std::size_t point = 0; point < permutation.size(); ++point) {
    if (permutation[point] != point) {
      return false;
    }
  }
  return true;
}

Permutation compose(
    const Permutation& left,
    const Permutation& right) {
  Permutation result(left.size());
  for (std::size_t point = 0; point < left.size(); ++point) {
    result[point] = left[right[point]];
  }
  return result;
}

Permutation invert(const Permutation& permutation) {
  Permutation result(permutation.size());
  for (std::size_t point = 0; point < permutation.size(); ++point) {
    result[permutation[point]] = static_cast<std::uint32_t>(point);
  }
  return result;
}

std::vector<Permutation> prepare_permutations(
    const std::uint32_t* values,
    std::size_t permutation_count,
    std::uint32_t degree,
    bool remove_identity,
    bool deduplicate) {
  if (degree == 0 || degree > 4096) {
    throw std::invalid_argument(
        "permutation degree must be between one and 4096");
  }
  if (permutation_count != 0 && values == nullptr) {
    throw std::invalid_argument("permutation pointer is null");
  }
  std::unordered_set<Permutation, PermutationHash> seen;
  std::vector<Permutation> result;
  result.reserve(permutation_count);
  std::vector<std::uint8_t> image_seen(degree);
  for (std::size_t index = 0; index < permutation_count; ++index) {
    std::fill(image_seen.begin(), image_seen.end(), 0);
    Permutation permutation(degree);
    for (std::uint32_t point = 0; point < degree; ++point) {
      const auto image =
          values[index * static_cast<std::size_t>(degree) + point];
      if (image >= degree || image_seen[image] != 0) {
        throw std::invalid_argument(
            "input row is not a permutation");
      }
      image_seen[image] = 1;
      permutation[point] = image;
    }
    if (remove_identity && is_identity(permutation)) {
      continue;
    }
    if (!deduplicate || seen.insert(permutation).second) {
      result.push_back(std::move(permutation));
    }
  }
  if (deduplicate) {
    std::sort(result.begin(), result.end());
  }
  return result;
}

std::vector<Permutation> symmetric_generators(
    const std::vector<Permutation>& generators) {
  std::unordered_set<Permutation, PermutationHash> seen;
  std::vector<Permutation> result;
  result.reserve(generators.size() * 2);
  for (const auto& generator : generators) {
    if (seen.insert(generator).second) {
      result.push_back(generator);
    }
    auto inverse = invert(generator);
    if (seen.insert(inverse).second) {
      result.push_back(std::move(inverse));
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

std::uint32_t first_moved_point(
    const std::vector<Permutation>& generators,
    std::uint32_t degree) {
  for (std::uint32_t point = 0; point < degree; ++point) {
    for (const auto& generator : generators) {
      if (generator[point] != point) {
        return point;
      }
    }
  }
  return degree;
}

std::vector<Permutation> deduplicate_nonidentity(
    std::vector<Permutation> permutations) {
  std::unordered_set<Permutation, PermutationHash> seen;
  std::vector<Permutation> result;
  result.reserve(permutations.size());
  for (auto& permutation : permutations) {
    if (!is_identity(permutation) &&
        seen.insert(permutation).second) {
      result.push_back(std::move(permutation));
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

ChainLevel build_level(
    const std::vector<Permutation>& generators,
    std::uint32_t degree,
    std::uint32_t base) {
  ChainLevel level;
  level.base = base;
  level.generators = generators;
  level.orbit_index.assign(degree, -1);

  const auto symmetric = symmetric_generators(generators);
  level.orbit.push_back(base);
  level.orbit_index[base] = 0;
  level.transversals.push_back(identity_permutation(degree));
  std::size_t head = 0;
  while (head < level.orbit.size()) {
    const auto point = level.orbit[head];
    const auto transversal = level.transversals[head];
    ++head;
    for (const auto& generator : symmetric) {
      const auto image = generator[point];
      if (level.orbit_index[image] >= 0) {
        continue;
      }
      level.orbit_index[image] =
          static_cast<std::int32_t>(level.orbit.size());
      level.orbit.push_back(image);
      level.transversals.push_back(
          compose(generator, transversal));
    }
  }
  level.inverse_transversals.reserve(level.transversals.size());
  for (const auto& transversal : level.transversals) {
    level.inverse_transversals.push_back(invert(transversal));
  }
  return level;
}

std::vector<Permutation> schreier_generators(
    const ChainLevel& level) {
  const auto symmetric = symmetric_generators(level.generators);
  std::vector<Permutation> result;
  result.reserve(level.orbit.size() * symmetric.size());
  for (std::size_t orbit_index = 0;
       orbit_index < level.orbit.size();
       ++orbit_index) {
    const auto point = level.orbit[orbit_index];
    const auto& transversal = level.transversals[orbit_index];
    for (const auto& generator : symmetric) {
      const auto image = generator[point];
      const auto image_index = level.orbit_index[image];
      if (image_index < 0) {
        throw std::runtime_error(
            "internal Schreier orbit lookup failed");
      }
      result.push_back(
          compose(
              level.inverse_transversals[
                  static_cast<std::size_t>(image_index)],
              compose(generator, transversal)));
    }
  }
  return deduplicate_nonidentity(std::move(result));
}

StabilizerChain build_chain(
    std::vector<Permutation> generators,
    std::uint32_t degree) {
  StabilizerChain chain;
  chain.degree = degree;
  generators = deduplicate_nonidentity(std::move(generators));
  while (!generators.empty()) {
    const auto base = first_moved_point(generators, degree);
    if (base == degree) {
      break;
    }
    auto level = build_level(generators, degree, base);
    generators = schreier_generators(level);
    chain.levels.push_back(std::move(level));
  }
  return chain;
}

bool chain_contains(
    const StabilizerChain& chain,
    const Permutation& element) {
  auto residual = element;
  for (const auto& level : chain.levels) {
    const auto image = residual[level.base];
    const auto orbit_index = level.orbit_index[image];
    if (orbit_index < 0) {
      return false;
    }
    residual = compose(
        level.inverse_transversals[
            static_cast<std::size_t>(orbit_index)],
        residual);
  }
  return is_identity(residual);
}

std::size_t strong_generator_count(const StabilizerChain& chain) {
  std::size_t count = 0;
  for (const auto& level : chain.levels) {
    count += level.generators.size();
  }
  return count;
}

void write_chain(
    const StabilizerChain& chain,
    std::size_t base_capacity,
    std::uint32_t* base_points,
    std::uint32_t* orbit_sizes,
    std::uint64_t* level_generator_offsets,
    std::size_t strong_generator_capacity,
    std::uint32_t* strong_generators) {
  if (base_capacity < chain.levels.size()) {
    throw std::invalid_argument(
        "Schreier-Sims base capacity is too small");
  }
  const auto strong_count = strong_generator_count(chain);
  if (strong_generator_capacity < strong_count) {
    throw std::invalid_argument(
        "strong generator capacity is too small");
  }
  if (!chain.levels.empty() &&
      (base_points == nullptr || orbit_sizes == nullptr ||
       level_generator_offsets == nullptr)) {
    throw std::invalid_argument(
        "Schreier-Sims output pointer is null");
  }
  if (strong_count != 0 && strong_generators == nullptr) {
    throw std::invalid_argument(
        "strong generator output pointer is null");
  }

  std::size_t strong_offset = 0;
  if (level_generator_offsets != nullptr) {
    level_generator_offsets[0] = 0;
  }
  for (std::size_t level_index = 0;
       level_index < chain.levels.size();
       ++level_index) {
    const auto& level = chain.levels[level_index];
    base_points[level_index] = level.base;
    orbit_sizes[level_index] =
        static_cast<std::uint32_t>(level.orbit.size());
    for (const auto& generator : level.generators) {
      std::copy(
          generator.begin(),
          generator.end(),
          strong_generators +
              strong_offset * chain.degree);
      ++strong_offset;
    }
    level_generator_offsets[level_index + 1] = strong_offset;
  }
}

}  // namespace

struct fast_math_permutation_group {
  std::uint32_t degree = 0;
  std::uint64_t generator_count = 0;
  StabilizerChain chain;
  std::vector<std::uint32_t> point_orbit_labels;
  std::uint32_t point_orbit_count = 0;
};

extern "C" {

int fast_math_permutation_group_create_u32(
    const std::uint32_t* generators,
    std::size_t generator_count,
    std::uint32_t degree,
    fast_math_permutation_group** group,
    fast_math_group_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (group == nullptr || stats == nullptr) {
      throw std::invalid_argument(
          "permutation group output or stats pointer is null");
    }
    *group = nullptr;
    auto prepared = prepare_permutations(
        generators, generator_count, degree, true, true);
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    auto plan = std::make_unique<fast_math_permutation_group>();
    plan->degree = degree;
    plan->generator_count = generator_count;
    plan->chain = build_chain(prepared, degree);
    plan->point_orbit_labels.assign(
        degree, std::numeric_limits<std::uint32_t>::max());
    const auto symmetric = symmetric_generators(prepared);
    std::deque<std::uint32_t> queue;
    for (std::uint32_t seed = 0; seed < degree; ++seed) {
      if (plan->point_orbit_labels[seed] !=
          std::numeric_limits<std::uint32_t>::max()) {
        continue;
      }
      plan->point_orbit_labels[seed] = plan->point_orbit_count;
      queue.push_back(seed);
      while (!queue.empty()) {
        const auto point = queue.front();
        queue.pop_front();
        for (const auto& generator : symmetric) {
          const auto image = generator[point];
          if (plan->point_orbit_labels[image] ==
              std::numeric_limits<std::uint32_t>::max()) {
            plan->point_orbit_labels[image] = plan->point_orbit_count;
            queue.push_back(image);
          }
        }
      }
      ++plan->point_orbit_count;
    }
    stats->degree = degree;
    stats->generator_count = generator_count;
    stats->item_count = degree;
    stats->orbit_count = plan->point_orbit_count;
    stats->chain_level_count = plan->chain.levels.size();
    stats->strong_generator_count = strong_generator_count(plan->chain);
    stats->elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    *group = plan.release();
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(error_message, error_message_size, "unknown native error");
    return 2;
  }
}

void fast_math_permutation_group_destroy(
    fast_math_permutation_group* group) {
  delete group;
}

int fast_math_permutation_group_summary_u32(
    const fast_math_permutation_group* group,
    std::size_t base_capacity,
    std::uint32_t* base_points,
    std::uint32_t* orbit_sizes,
    std::uint32_t* point_orbit_labels,
    std::uint32_t* point_orbit_count,
    fast_math_group_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (group == nullptr || point_orbit_count == nullptr || stats == nullptr) {
      throw std::invalid_argument(
          "permutation group plan or output pointer is null");
    }
    if (base_capacity < group->chain.levels.size()) {
      throw std::invalid_argument(
          "permutation group base capacity is too small");
    }
    if (!group->chain.levels.empty() &&
        (base_points == nullptr || orbit_sizes == nullptr)) {
      throw std::invalid_argument(
          "permutation group chain output pointer is null");
    }
    if (point_orbit_labels == nullptr) {
      throw std::invalid_argument(
          "permutation group orbit output pointer is null");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    for (std::size_t index = 0; index < group->chain.levels.size(); ++index) {
      base_points[index] = group->chain.levels[index].base;
      orbit_sizes[index] = static_cast<std::uint32_t>(
          group->chain.levels[index].orbit.size());
    }
    std::copy(
        group->point_orbit_labels.begin(),
        group->point_orbit_labels.end(),
        point_orbit_labels);
    *point_orbit_count = group->point_orbit_count;
    stats->degree = group->degree;
    stats->generator_count = group->generator_count;
    stats->item_count = group->degree;
    stats->orbit_count = group->point_orbit_count;
    stats->chain_level_count = group->chain.levels.size();
    stats->strong_generator_count = strong_generator_count(group->chain);
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

int fast_math_permutation_group_plan_contains_u32(
    const fast_math_permutation_group* group,
    const std::uint32_t* elements,
    std::size_t element_count,
    std::uint32_t thread_count,
    std::uint8_t* contains,
    fast_math_group_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (group == nullptr || stats == nullptr) {
      throw std::invalid_argument(
          "permutation group plan or stats pointer is null");
    }
    if (element_count != 0 &&
        (elements == nullptr || contains == nullptr)) {
      throw std::invalid_argument(
          "membership input or output pointer is null");
    }
    const auto candidates = prepare_permutations(
        elements, element_count, group->degree, false, false);
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    fast_math_internal::parallel_for_static(
        element_count,
        thread_count,
        [&](std::size_t index) {
          contains[index] = chain_contains(group->chain, candidates[index]);
        });
    stats->degree = group->degree;
    stats->generator_count = group->generator_count;
    stats->item_count = element_count;
    stats->orbit_count = group->point_orbit_count;
    stats->chain_level_count = group->chain.levels.size();
    stats->strong_generator_count = strong_generator_count(group->chain);
    stats->thread_count =
        fast_math_internal::parallel_worker_count(
            element_count, thread_count);
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

int fast_math_permutation_orbits_u32(
    const std::uint32_t* generators,
    std::size_t generator_count,
    std::uint32_t degree,
    std::uint32_t* orbit_labels,
    std::uint32_t* orbit_count,
    fast_math_group_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (orbit_labels == nullptr || orbit_count == nullptr ||
        stats == nullptr) {
      throw std::invalid_argument(
          "permutation orbit output pointer is null");
    }
    const auto prepared = prepare_permutations(
        generators, generator_count, degree, true, true);
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const auto symmetric = symmetric_generators(prepared);
    std::fill_n(
        orbit_labels,
        static_cast<std::size_t>(degree),
        std::numeric_limits<std::uint32_t>::max());
    std::uint32_t count = 0;
    std::deque<std::uint32_t> queue;
    for (std::uint32_t seed = 0; seed < degree; ++seed) {
      if (orbit_labels[seed] !=
          std::numeric_limits<std::uint32_t>::max()) {
        continue;
      }
      orbit_labels[seed] = count;
      queue.push_back(seed);
      while (!queue.empty()) {
        const auto point = queue.front();
        queue.pop_front();
        for (const auto& generator : symmetric) {
          const auto image = generator[point];
          if (orbit_labels[image] ==
              std::numeric_limits<std::uint32_t>::max()) {
            orbit_labels[image] = count;
            queue.push_back(image);
          }
        }
      }
      ++count;
    }
    *orbit_count = count;
    stats->degree = degree;
    stats->generator_count = generator_count;
    stats->item_count = degree;
    stats->orbit_count = count;
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

int fast_math_schreier_sims_u32(
    const std::uint32_t* generators,
    std::size_t generator_count,
    std::uint32_t degree,
    std::size_t base_capacity,
    std::uint32_t* base_points,
    std::uint32_t* orbit_sizes,
    std::uint64_t* level_generator_offsets,
    std::size_t strong_generator_capacity,
    std::uint32_t* strong_generators,
    std::uint64_t* base_count,
    std::uint64_t* strong_generator_count_output,
    fast_math_group_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (base_count == nullptr ||
        strong_generator_count_output == nullptr ||
        stats == nullptr) {
      throw std::invalid_argument(
          "Schreier-Sims count or stats pointer is null");
    }
    auto prepared = prepare_permutations(
        generators, generator_count, degree, true, true);
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const auto chain = build_chain(std::move(prepared), degree);
    const auto strong_count = strong_generator_count(chain);
    *base_count = chain.levels.size();
    *strong_generator_count_output = strong_count;
    if (base_points != nullptr || orbit_sizes != nullptr ||
        level_generator_offsets != nullptr ||
        strong_generators != nullptr) {
      write_chain(
          chain,
          base_capacity,
          base_points,
          orbit_sizes,
          level_generator_offsets,
          strong_generator_capacity,
          strong_generators);
    }
    stats->degree = degree;
    stats->generator_count = generator_count;
    stats->chain_level_count = chain.levels.size();
    stats->strong_generator_count = strong_count;
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

int fast_math_permutation_group_contains_u32(
    const std::uint32_t* generators,
    std::size_t generator_count,
    std::uint32_t degree,
    const std::uint32_t* elements,
    std::size_t element_count,
    std::uint32_t thread_count,
    std::uint8_t* contains,
    fast_math_group_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("group stats pointer is null");
    }
    if (element_count != 0 &&
        (elements == nullptr || contains == nullptr)) {
      throw std::invalid_argument(
          "membership input or output pointer is null");
    }
    auto prepared = prepare_permutations(
        generators, generator_count, degree, true, true);
    const auto candidates = prepare_permutations(
        elements, element_count, degree, false, false);
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const auto chain = build_chain(std::move(prepared), degree);
    fast_math_internal::parallel_for_static(
        element_count,
        thread_count,
        [&](std::size_t index) {
          contains[index] = chain_contains(chain, candidates[index]);
        });
    stats->degree = degree;
    stats->generator_count = generator_count;
    stats->item_count = element_count;
    stats->chain_level_count = chain.levels.size();
    stats->strong_generator_count = strong_generator_count(chain);
    stats->thread_count =
        fast_math_internal::parallel_worker_count(
            element_count, thread_count);
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
