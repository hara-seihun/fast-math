#include "fast_math.h"

#include "parallel.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Permutation = std::vector<std::uint32_t>;
using PackedSubset = std::vector<std::uint64_t>;

struct VectorHash {
  template <typename T>
  std::size_t operator()(const std::vector<T>& values) const noexcept {
    std::size_t state = 1469598103934665603ULL;
    for (const auto value : values) {
      state ^= static_cast<std::size_t>(value);
      state *= 1099511628211ULL;
    }
    return state;
  }
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
  for (std::uint32_t point = 0; point < permutation.size(); ++point) {
    result[permutation[point]] = point;
  }
  return result;
}

std::vector<Permutation> read_permutations(
    const std::uint32_t* values,
    std::size_t count,
    std::uint32_t degree,
    const char* error,
    std::uint32_t maximum_degree = 512) {
  if (degree == 0 || degree > maximum_degree) {
    throw std::invalid_argument(
        "permutation degree exceeds the operation limit");
  }
  if (count != 0 && values == nullptr) {
    throw std::invalid_argument("permutation pointer is null");
  }
  std::vector<Permutation> result;
  result.reserve(count);
  std::vector<std::uint8_t> seen(degree);
  for (std::size_t index = 0; index < count; ++index) {
    std::fill(seen.begin(), seen.end(), 0);
    Permutation permutation(degree);
    for (std::uint32_t point = 0; point < degree; ++point) {
      const auto image =
          values[index * static_cast<std::size_t>(degree) + point];
      if (image >= degree || seen[image] != 0) {
        throw std::invalid_argument(error);
      }
      seen[image] = 1;
      permutation[point] = image;
    }
    result.push_back(std::move(permutation));
  }
  return result;
}

std::vector<Permutation> symmetric_generators(
    const std::vector<Permutation>& generators) {
  std::unordered_map<Permutation, std::uint8_t, VectorHash> seen;
  std::vector<Permutation> result;
  result.reserve(generators.size() * 2);
  for (const auto& generator : generators) {
    if (seen.emplace(generator, 1).second) {
      result.push_back(generator);
    }
    auto inverse = invert(generator);
    if (seen.emplace(inverse, 1).second) {
      result.push_back(std::move(inverse));
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

std::uint32_t point_orbits(
    const std::vector<Permutation>& generators,
    std::uint32_t degree,
    std::uint32_t* labels) {
  const auto symmetric = symmetric_generators(generators);
  std::fill_n(
      labels,
      degree,
      std::numeric_limits<std::uint32_t>::max());
  std::deque<std::uint32_t> queue;
  std::uint32_t count = 0;
  for (std::uint32_t seed = 0; seed < degree; ++seed) {
    if (labels[seed] != std::numeric_limits<std::uint32_t>::max()) {
      continue;
    }
    labels[seed] = count;
    queue.push_back(seed);
    while (!queue.empty()) {
      const auto point = queue.front();
      queue.pop_front();
      for (const auto& generator : symmetric) {
        const auto image = generator[point];
        if (labels[image] !=
            std::numeric_limits<std::uint32_t>::max()) {
          continue;
        }
        labels[image] = count;
        queue.push_back(image);
      }
    }
    ++count;
  }
  return count;
}

bool packed_subset_less(
    const PackedSubset& left,
    const PackedSubset& right) {
  for (std::size_t word = left.size(); word-- > 0;) {
    if (left[word] != right[word]) {
      return left[word] < right[word];
    }
  }
  return false;
}

PackedSubset permute_subset(
    const PackedSubset& subset,
    const Permutation& permutation,
    std::uint32_t atom_count,
    std::uint32_t word_count) {
  PackedSubset result(word_count, 0);
  for (std::uint32_t word = 0; word < word_count; ++word) {
    auto active = subset[word];
    while (active != 0) {
      const auto bit =
          static_cast<std::uint32_t>(std::countr_zero(active));
      const auto atom = word * 64 + bit;
      if (atom >= atom_count) {
        break;
      }
      const auto image = permutation[atom];
      result[image / 64] |= std::uint64_t{1} << (image % 64);
      active &= active - 1;
    }
  }
  return result;
}

bool is_complete_action(
    const std::vector<Permutation>& action,
    std::uint32_t degree,
    bool required) {
  std::unordered_set<Permutation, VectorHash> elements;
  elements.reserve(action.size() * 2 + 1);
  for (const auto& permutation : action) {
    if (!elements.insert(permutation).second) {
      if (required) {
        throw std::invalid_argument(
            "complete action rows must be unique");
      }
      return false;
    }
  }
  Permutation identity(degree);
  std::iota(identity.begin(), identity.end(), 0);
  if (!elements.contains(identity)) {
    if (required) {
      throw std::invalid_argument(
          "complete action must contain the identity");
    }
    return false;
  }
  // Generate the subgroup row by row.  Because every supplied row belongs to
  // the generated subgroup, the rows are the complete group exactly when the
  // generated order equals their unique row count.  Stop as soon as the order
  // exceeds that count.  This replaces the quadratic all-pairs closure scan
  // by a bounded Schreier-style traversal using only the few rows that enlarge
  // the generated subgroup.
  std::unordered_set<Permutation, VectorHash> generated;
  generated.reserve(action.size() * 2 + 1);
  generated.insert(identity);
  std::vector<Permutation> generators;
  for (const auto& permutation : action) {
    if (generated.contains(permutation)) {
      continue;
    }
    generators.push_back(permutation);
    std::deque<Permutation> queue;
    for (const auto& member : generated) {
      queue.push_back(member);
    }
    while (!queue.empty()) {
      const auto current = std::move(queue.front());
      queue.pop_front();
      for (const auto& generator : generators) {
        auto product = compose(current, generator);
        if (!generated.insert(product).second) {
          continue;
        }
        if (generated.size() > action.size()) {
          if (required) {
            throw std::invalid_argument(
                "complete action is not closed under composition");
          }
          return false;
        }
        queue.push_back(std::move(product));
      }
    }
  }
  if (generated.size() != action.size()) {
    if (required) {
      throw std::invalid_argument(
          "complete action is not closed under composition");
    }
    return false;
  }
  return true;
}

struct MaskPermutation64 {
  std::array<std::uint64_t, 8 * 256> lookup{};
};

std::vector<MaskPermutation64> build_mask_permutations(
    const std::vector<Permutation>& permutations,
    std::uint32_t degree) {
  std::vector<MaskPermutation64> result(permutations.size());
  for (std::size_t index = 0; index < permutations.size(); ++index) {
    for (std::uint32_t byte = 0; byte < 8; ++byte) {
      const auto base = byte * 256;
      for (std::uint32_t value = 1; value < 256; ++value) {
        const auto bit =
            static_cast<std::uint32_t>(std::countr_zero(value));
        const auto point = byte * 8 + bit;
        auto image = result[index].lookup[
            base + (value & (value - 1))];
        if (point < degree) {
          image |= std::uint64_t{1}
              << permutations[index][point];
        }
        result[index].lookup[base + value] = image;
      }
    }
  }
  return result;
}

std::uint64_t permute_mask(
    std::uint64_t mask,
    const MaskPermutation64& permutation) {
  std::uint64_t result = 0;
  std::uint32_t byte = 0;
  while (mask != 0) {
    result |= permutation.lookup[
        byte * 256 + (mask & 0xff)];
    mask >>= 8;
    ++byte;
  }
  return result;
}

using BinomialTable = std::array<std::array<std::uint64_t, 65>, 65>;

BinomialTable binomial_table(std::uint32_t degree) {
  BinomialTable table{};
  for (std::uint32_t value = 0; value <= degree; ++value) {
    table[value][0] = 1;
    table[value][value] = 1;
    for (std::uint32_t weight = 1; weight < value; ++weight) {
      table[value][weight] =
          table[value - 1][weight - 1] +
          table[value - 1][weight];
    }
  }
  return table;
}

std::uint64_t combinadic_rank(
    std::uint64_t mask,
    std::uint32_t expected_weight,
    const BinomialTable& choose) {
  std::uint64_t rank = 0;
  std::uint32_t index = 1;
  while (mask != 0) {
    const auto point =
        static_cast<std::uint32_t>(std::countr_zero(mask));
    rank += choose[point][index];
    ++index;
    mask &= mask - 1;
  }
  if (index != expected_weight + 1) {
    throw std::runtime_error("subset image has the wrong weight");
  }
  return rank;
}

void validate_multiplication_table(
    const std::uint32_t* multiplication_table,
    std::uint32_t order) {
  if (order == 0 || order > 512 || multiplication_table == nullptr) {
    throw std::invalid_argument(
        "group order must be between one and 512");
  }
  const auto entry_count = static_cast<std::size_t>(order) * order;
  for (std::size_t index = 0; index < entry_count; ++index) {
    if (multiplication_table[index] >= order) {
      throw std::invalid_argument(
          "multiplication table entry is out of range");
    }
  }
}

std::uint32_t normalize_relations(
    std::vector<std::uint32_t>& relations) {
  auto values = relations;
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  for (auto& relation : relations) {
    relation = static_cast<std::uint32_t>(
        std::lower_bound(values.begin(), values.end(), relation) -
        values.begin());
  }
  return static_cast<std::uint32_t>(values.size());
}

std::uint32_t refine_wl2(
    std::vector<std::uint32_t>& relations,
    std::uint32_t vertex_count,
    std::uint64_t* iteration_count) {
  auto relation_count = normalize_relations(relations);
  const auto pair_count =
      static_cast<std::size_t>(vertex_count) * vertex_count;
  std::vector<std::vector<std::uint64_t>> signatures(pair_count);
  std::vector<std::uint64_t> keys(vertex_count);
  std::vector<std::size_t> order(pair_count);
  std::iota(order.begin(), order.end(), 0);
  *iteration_count = 0;

  while (true) {
    for (std::uint32_t left = 0; left < vertex_count; ++left) {
      for (std::uint32_t right = 0; right < vertex_count; ++right) {
        const auto pair =
            static_cast<std::size_t>(left) * vertex_count + right;
        for (std::uint32_t middle = 0;
             middle < vertex_count;
             ++middle) {
          const auto first = relations[
              static_cast<std::size_t>(left) * vertex_count + middle];
          const auto second = relations[
              static_cast<std::size_t>(middle) * vertex_count + right];
          keys[middle] =
              static_cast<std::uint64_t>(first) * relation_count +
              second;
        }
        std::sort(keys.begin(), keys.end());
        auto& signature = signatures[pair];
        signature.clear();
        signature.reserve(1 + vertex_count * 2);
        signature.push_back(relations[pair]);
        for (auto begin = keys.begin(); begin != keys.end();) {
          auto end = begin + 1;
          while (end != keys.end() && *end == *begin) {
            ++end;
          }
          signature.push_back(*begin);
          signature.push_back(
              static_cast<std::uint64_t>(end - begin));
          begin = end;
        }
      }
    }

    std::sort(
        order.begin(),
        order.end(),
        [&](std::size_t left, std::size_t right) {
          if (signatures[left] != signatures[right]) {
            return signatures[left] < signatures[right];
          }
          return left < right;
        });
    std::vector<std::uint32_t> refined(pair_count);
    std::uint32_t refined_count = 1;
    refined[order[0]] = 0;
    for (std::size_t position = 1; position < order.size(); ++position) {
      if (signatures[order[position]] !=
          signatures[order[position - 1]]) {
        ++refined_count;
      }
      refined[order[position]] = refined_count - 1;
    }
    ++*iteration_count;
    if (refined == relations) {
      return refined_count;
    }
    relations.swap(refined);
    relation_count = refined_count;
  }
}

}  // namespace

extern "C" {

int fast_math_permutation_double_cosets_u32(
    const std::uint32_t* candidates,
    std::size_t candidate_count,
    const std::uint32_t* left_generators,
    std::size_t left_generator_count,
    const std::uint32_t* right_generators,
    std::size_t right_generator_count,
    std::uint32_t degree,
    std::uint64_t* class_ids,
    std::uint64_t* representative_indices,
    std::uint64_t* class_sizes,
    std::uint64_t* class_count,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (class_count == nullptr || stats == nullptr ||
        (candidate_count != 0 &&
         (candidates == nullptr || class_ids == nullptr ||
          representative_indices == nullptr || class_sizes == nullptr))) {
      throw std::invalid_argument("double-coset pointer is null");
    }
    const auto candidate_values = read_permutations(
        candidates,
        candidate_count,
        degree,
        "double-coset candidate is not a permutation");
    const auto left = symmetric_generators(read_permutations(
        left_generators,
        left_generator_count,
        degree,
        "left generator is not a permutation"));
    const auto right = symmetric_generators(read_permutations(
        right_generators,
        right_generator_count,
        degree,
        "right generator is not a permutation"));
    std::vector<Permutation> inverse_right;
    inverse_right.reserve(right.size());
    for (const auto& generator : right) {
      inverse_right.push_back(invert(generator));
    }
    std::unordered_map<Permutation, std::size_t, VectorHash> index_of;
    index_of.reserve(candidate_count * 2 + 1);
    for (std::size_t index = 0; index < candidate_count; ++index) {
      if (!index_of.emplace(candidate_values[index], index).second) {
        throw std::invalid_argument(
            "double-coset candidates must be unique");
      }
      class_ids[index] = std::numeric_limits<std::uint64_t>::max();
    }

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    std::deque<std::size_t> queue;
    std::uint64_t classes = 0;
    for (std::size_t seed = 0; seed < candidate_count; ++seed) {
      if (class_ids[seed] != std::numeric_limits<std::uint64_t>::max()) {
        continue;
      }
      representative_indices[classes] = seed;
      class_ids[seed] = classes;
      std::uint64_t size = 0;
      queue.push_back(seed);
      while (!queue.empty()) {
        const auto current = queue.front();
        queue.pop_front();
        ++size;
        auto visit = [&](const Permutation& image) {
          const auto found = index_of.find(image);
          if (found == index_of.end()) {
            throw std::invalid_argument(
                "candidate collection is not double-coset invariant");
          }
          if (class_ids[found->second] !=
              std::numeric_limits<std::uint64_t>::max()) {
            return;
          }
          class_ids[found->second] = classes;
          queue.push_back(found->second);
        };
        for (const auto& generator : left) {
          visit(compose(generator, candidate_values[current]));
        }
        for (const auto& generator : inverse_right) {
          visit(compose(candidate_values[current], generator));
        }
      }
      class_sizes[classes] = size;
      ++classes;
    }
    *class_count = classes;
    stats->degree = degree;
    stats->generator_count =
        left_generator_count + right_generator_count;
    stats->item_count = candidate_count;
    stats->class_count = classes;
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

int fast_math_subset_orbits_v2_u64(
    const std::uint64_t* subset_words,
    std::size_t subset_count,
    std::uint32_t word_count,
    std::uint32_t atom_count,
    const std::uint32_t* action_generators,
    std::size_t generator_count,
    std::uint32_t action_mode,
    std::uint64_t* class_ids,
    std::uint64_t* representative_indices,
    std::uint64_t* class_sizes,
    std::uint64_t* class_count,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (atom_count == 0 || atom_count > 512 ||
        word_count != (atom_count + 63) / 64) {
      throw std::invalid_argument("subset packed shape is invalid");
    }
    if (action_mode > 2) {
      throw std::invalid_argument("subset action mode is invalid");
    }
    if (class_count == nullptr || stats == nullptr ||
        (subset_count != 0 &&
         (subset_words == nullptr || class_ids == nullptr ||
          representative_indices == nullptr || class_sizes == nullptr))) {
      throw std::invalid_argument("subset orbit pointer is null");
    }
    const auto generators = read_permutations(
        action_generators,
        generator_count,
        atom_count,
        "subset action generator is not a permutation");
    const auto final_bits = atom_count % 64;
    const auto final_mask = final_bits == 0
        ? ~std::uint64_t{0}
        : (std::uint64_t{1} << final_bits) - 1;

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const auto complete_action =
        action_mode == 2 ||
        (action_mode == 0 &&
         is_complete_action(generators, atom_count, false));
    if (action_mode == 2) {
      is_complete_action(generators, atom_count, true);
    }
    if (word_count == 1) {
      std::vector<std::uint64_t> masks(subset_count);
      std::unordered_map<std::uint64_t, std::size_t> mask_index;
      mask_index.reserve(subset_count * 2 + 1);
      for (std::size_t index = 0; index < subset_count; ++index) {
        const auto mask = subset_words[index];
        if ((mask & ~final_mask) != 0) {
          throw std::invalid_argument(
              "subset contains an out-of-range atom");
        }
        if (!mask_index.emplace(mask, index).second) {
          throw std::invalid_argument("subset rows must be unique");
        }
        masks[index] = mask;
        class_ids[index] = std::numeric_limits<std::uint64_t>::max();
      }
      const auto mask_permutations =
          build_mask_permutations(generators, atom_count);
      struct MaskOrbit {
        std::size_t representative = 0;
        std::vector<std::size_t> members;
      };
      std::vector<MaskOrbit> mask_orbits;
      std::deque<std::size_t> mask_queue;
      for (std::size_t seed = 0; seed < subset_count; ++seed) {
        if (class_ids[seed] !=
            std::numeric_limits<std::uint64_t>::max()) {
          continue;
        }
        const auto provisional_class = mask_orbits.size();
        MaskOrbit orbit;
        orbit.representative = seed;
        class_ids[seed] = provisional_class;
        if (complete_action) {
          orbit.members.push_back(seed);
          for (const auto& permutation : mask_permutations) {
            const auto image =
                permute_mask(masks[seed], permutation);
            const auto found = mask_index.find(image);
            if (found == mask_index.end()) {
              throw std::invalid_argument(
                  "subset collection is not invariant under the action");
            }
            const auto image_class = class_ids[found->second];
            if (image_class == provisional_class) {
              continue;
            }
            if (image_class !=
                std::numeric_limits<std::uint64_t>::max()) {
              throw std::runtime_error(
                  "complete subset action crossed an earlier orbit");
            }
            class_ids[found->second] = provisional_class;
            orbit.members.push_back(found->second);
            if (masks[found->second] <
                masks[orbit.representative]) {
              orbit.representative = found->second;
            }
          }
        } else {
          mask_queue.push_back(seed);
          while (!mask_queue.empty()) {
            const auto current = mask_queue.front();
            mask_queue.pop_front();
            orbit.members.push_back(current);
            if (masks[current] < masks[orbit.representative]) {
              orbit.representative = current;
            }
            for (const auto& permutation : mask_permutations) {
              const auto image =
                  permute_mask(masks[current], permutation);
              const auto found = mask_index.find(image);
              if (found == mask_index.end()) {
                throw std::invalid_argument(
                    "subset collection is not invariant under the action");
              }
              if (class_ids[found->second] !=
                  std::numeric_limits<std::uint64_t>::max()) {
                continue;
              }
              class_ids[found->second] = provisional_class;
              mask_queue.push_back(found->second);
            }
          }
        }
        mask_orbits.push_back(std::move(orbit));
      }
      std::vector<std::size_t> mask_order(mask_orbits.size());
      std::iota(mask_order.begin(), mask_order.end(), 0);
      std::sort(
          mask_order.begin(),
          mask_order.end(),
          [&](std::size_t left, std::size_t right) {
            return masks[mask_orbits[left].representative] <
                masks[mask_orbits[right].representative];
          });
      for (std::size_t output_class = 0;
           output_class < mask_order.size();
           ++output_class) {
        const auto& orbit =
            mask_orbits[mask_order[output_class]];
        representative_indices[output_class] =
            orbit.representative;
        class_sizes[output_class] = orbit.members.size();
        for (const auto member : orbit.members) {
          class_ids[member] = output_class;
        }
      }
      *class_count = mask_orbits.size();
      stats->degree = atom_count;
      stats->generator_count = generator_count;
      stats->item_count = subset_count;
      stats->class_count = mask_orbits.size();
      stats->elapsed_seconds =
          std::chrono::duration<double>(
              Clock::now() - started).count();
      return 0;
    }
    std::vector<PackedSubset> subsets;
    subsets.reserve(subset_count);
    std::unordered_map<PackedSubset, std::size_t, VectorHash> index_of;
    index_of.reserve(subset_count * 2 + 1);
    for (std::size_t index = 0; index < subset_count; ++index) {
      PackedSubset subset(
          subset_words + index * word_count,
          subset_words + (index + 1) * word_count);
      if ((subset.back() & ~final_mask) != 0) {
        throw std::invalid_argument(
            "subset contains an out-of-range atom");
      }
      if (!index_of.emplace(subset, index).second) {
        throw std::invalid_argument("subset rows must be unique");
      }
      subsets.push_back(std::move(subset));
      class_ids[index] = std::numeric_limits<std::uint64_t>::max();
    }
    struct Orbit {
      std::size_t representative = 0;
      std::vector<std::size_t> members;
    };
    std::vector<Orbit> orbits;
    std::deque<std::size_t> queue;
    for (std::size_t seed = 0; seed < subset_count; ++seed) {
      if (class_ids[seed] != std::numeric_limits<std::uint64_t>::max()) {
        continue;
      }
      const auto provisional_class = orbits.size();
      Orbit orbit;
      orbit.representative = seed;
      class_ids[seed] = provisional_class;
      if (complete_action) {
        orbit.members.push_back(seed);
        for (const auto& generator : generators) {
          const auto image = permute_subset(
              subsets[seed],
              generator,
              atom_count,
              word_count);
          const auto found = index_of.find(image);
          if (found == index_of.end()) {
            throw std::invalid_argument(
                "subset collection is not invariant under the action");
          }
          const auto image_class = class_ids[found->second];
          if (image_class == provisional_class) {
            continue;
          }
          if (image_class != std::numeric_limits<std::uint64_t>::max()) {
            throw std::runtime_error(
                "complete subset action crossed an earlier orbit");
          }
          class_ids[found->second] = provisional_class;
          orbit.members.push_back(found->second);
          if (packed_subset_less(
                  subsets[found->second],
                  subsets[orbit.representative])) {
            orbit.representative = found->second;
          }
        }
      } else {
        queue.push_back(seed);
        while (!queue.empty()) {
          const auto current = queue.front();
          queue.pop_front();
          orbit.members.push_back(current);
          if (packed_subset_less(
                  subsets[current],
                  subsets[orbit.representative])) {
            orbit.representative = current;
          }
          for (const auto& generator : generators) {
            const auto image = permute_subset(
                subsets[current],
                generator,
                atom_count,
                word_count);
            const auto found = index_of.find(image);
            if (found == index_of.end()) {
              throw std::invalid_argument(
                  "subset collection is not invariant under the action");
            }
            if (class_ids[found->second] !=
                std::numeric_limits<std::uint64_t>::max()) {
              continue;
            }
            class_ids[found->second] = provisional_class;
            queue.push_back(found->second);
          }
        }
      }
      orbits.push_back(std::move(orbit));
    }
    std::vector<std::size_t> order(orbits.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(
        order.begin(),
        order.end(),
        [&](std::size_t left, std::size_t right) {
          return packed_subset_less(
              subsets[orbits[left].representative],
              subsets[orbits[right].representative]);
        });
    for (std::size_t output_class = 0;
         output_class < order.size();
         ++output_class) {
      const auto& orbit = orbits[order[output_class]];
      representative_indices[output_class] = orbit.representative;
      class_sizes[output_class] = orbit.members.size();
      for (const auto member : orbit.members) {
        class_ids[member] = output_class;
      }
    }
    *class_count = orbits.size();
    stats->degree = atom_count;
    stats->generator_count = generator_count;
    stats->item_count = subset_count;
    stats->class_count = orbits.size();
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

int fast_math_fixed_weight_subset_orbits_u64(
    const std::uint32_t* complete_action,
    std::size_t action_count,
    std::uint32_t atom_count,
    std::uint32_t subset_weight,
    std::uint64_t max_subset_count,
    std::uint64_t* representative_masks,
    std::uint64_t representative_capacity,
    std::uint64_t* orbit_sizes,
    std::uint64_t* representative_count,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (atom_count == 0 || atom_count > 64) {
      throw std::invalid_argument(
          "fixed-weight atom count must be between one and 64");
    }
    if (subset_weight > atom_count) {
      throw std::invalid_argument(
          "fixed subset weight exceeds the atom count");
    }
    if (action_count == 0 || representative_count == nullptr ||
        stats == nullptr ||
        (representative_capacity != 0 &&
         (representative_masks == nullptr || orbit_sizes == nullptr))) {
      throw std::invalid_argument(
          "fixed-weight subset orbit pointer is null");
    }
    const auto action = read_permutations(
        complete_action,
        action_count,
        atom_count,
        "fixed-weight action row is not a permutation");
    is_complete_action(action, atom_count, true);
    const auto choose = binomial_table(atom_count);
    const auto subset_count = choose[atom_count][subset_weight];
    if (subset_count > max_subset_count) {
      throw std::invalid_argument(
          "fixed-weight subset domain exceeds max_subset_count");
    }
    if (subset_count >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max() - 63)) {
      throw std::invalid_argument(
          "fixed-weight subset domain is too large for this platform");
    }

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const auto mask_permutations =
        build_mask_permutations(action, atom_count);
    std::vector<std::uint64_t> seen(
        static_cast<std::size_t>((subset_count + 63) / 64),
        0);
    std::vector<std::uint32_t> combination(subset_weight);
    std::iota(combination.begin(), combination.end(), 0);
    std::vector<std::uint64_t> image_ranks;
    image_ranks.reserve(action_count);
    std::uint64_t output_count = 0;
    std::uint64_t visited_inputs = 0;

    while (true) {
      std::uint64_t rank = 0;
      std::uint64_t mask = 0;
      for (std::uint32_t index = 0; index < subset_weight; ++index) {
        rank += choose[combination[index]][index + 1];
        mask |= std::uint64_t{1} << combination[index];
      }
      if (((seen[rank / 64] >> (rank % 64)) & 1U) == 0) {
        if (output_count >= representative_capacity) {
          throw std::invalid_argument(
              "fixed-weight representative capacity is too small");
        }
        representative_masks[output_count] = mask;
        image_ranks.clear();
        for (const auto& permutation : mask_permutations) {
          const auto image = permute_mask(mask, permutation);
          const auto image_rank =
              combinadic_rank(image, subset_weight, choose);
          if (image_rank >= subset_count) {
            throw std::runtime_error(
                "fixed-weight image rank is out of range");
          }
          image_ranks.push_back(image_rank);
        }
        std::sort(image_ranks.begin(), image_ranks.end());
        image_ranks.erase(
            std::unique(image_ranks.begin(), image_ranks.end()),
            image_ranks.end());
        orbit_sizes[output_count] = image_ranks.size();
        for (const auto image_rank : image_ranks) {
          if (((seen[image_rank / 64] >> (image_rank % 64)) & 1U) != 0) {
            throw std::runtime_error(
                "fixed-weight action crossed an earlier orbit");
          }
          seen[image_rank / 64] |=
              std::uint64_t{1} << (image_rank % 64);
        }
        ++output_count;
      }
      ++visited_inputs;
      if (subset_weight == 0) {
        break;
      }
      std::int32_t pivot =
          static_cast<std::int32_t>(subset_weight) - 1;
      while (pivot >= 0 &&
             combination[static_cast<std::size_t>(pivot)] ==
                 atom_count - subset_weight +
                     static_cast<std::uint32_t>(pivot)) {
        --pivot;
      }
      if (pivot < 0) {
        break;
      }
      ++combination[static_cast<std::size_t>(pivot)];
      for (std::uint32_t index =
               static_cast<std::uint32_t>(pivot) + 1;
           index < subset_weight;
           ++index) {
        combination[index] = combination[index - 1] + 1;
      }
    }

    std::uint64_t seen_count = 0;
    for (const auto word : seen) {
      seen_count += std::popcount(word);
    }
    if (visited_inputs != subset_count || seen_count != subset_count) {
      throw std::runtime_error(
          "fixed-weight orbit enumeration did not cover its domain");
    }
    *representative_count = output_count;
    stats->degree = atom_count;
    stats->generator_count = action_count;
    stats->item_count = subset_count;
    stats->class_count = output_count;
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

int fast_math_subset_orbits_u64(
    const std::uint64_t* subset_words,
    std::size_t subset_count,
    std::uint32_t word_count,
    std::uint32_t atom_count,
    const std::uint32_t* action_generators,
    std::size_t generator_count,
    std::uint64_t* class_ids,
    std::uint64_t* representative_indices,
    std::uint64_t* class_sizes,
    std::uint64_t* class_count,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  return fast_math_subset_orbits_v2_u64(
      subset_words,
      subset_count,
      word_count,
      atom_count,
      action_generators,
      generator_count,
      1,
      class_ids,
      representative_indices,
      class_sizes,
      class_count,
      stats,
      error_message,
      error_message_size);
}

int fast_math_expand_atom_subsets_u64(
    const std::uint64_t* subset_words,
    std::size_t subset_count,
    std::uint32_t subset_word_count,
    std::uint32_t atom_count,
    const std::uint64_t* atom_offsets,
    const std::uint32_t* atom_elements,
    std::size_t atom_element_count,
    std::uint32_t group_order,
    std::uint32_t thread_count,
    std::uint64_t* element_words,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (atom_count > 512 ||
        subset_word_count != (atom_count + 63) / 64) {
      throw std::invalid_argument("atom subset packed shape is invalid");
    }
    if (group_order == 0 || group_order > 512) {
      throw std::invalid_argument(
          "group order must be between one and 512");
    }
    if (atom_offsets == nullptr || stats == nullptr ||
        (subset_count != 0 &&
         (subset_words == nullptr || element_words == nullptr)) ||
        (atom_element_count != 0 && atom_elements == nullptr)) {
      throw std::invalid_argument("atom expansion pointer is null");
    }
    if (atom_offsets[0] != 0 ||
        atom_offsets[atom_count] != atom_element_count) {
      throw std::invalid_argument("atom offsets are invalid");
    }
    for (std::uint32_t atom = 0; atom < atom_count; ++atom) {
      if (atom_offsets[atom] > atom_offsets[atom + 1]) {
        throw std::invalid_argument("atom offsets are not monotone");
      }
    }
    for (std::size_t index = 0; index < atom_element_count; ++index) {
      if (atom_elements[index] >= group_order) {
        throw std::invalid_argument(
            "atom contains an out-of-range element");
      }
    }
    if (atom_count != 0 && subset_count != 0) {
      const auto final_bits = atom_count % 64;
      const auto final_mask = final_bits == 0
          ? ~std::uint64_t{0}
          : (std::uint64_t{1} << final_bits) - 1;
      for (std::size_t subset = 0; subset < subset_count; ++subset) {
        if ((subset_words[
                 subset * subset_word_count + subset_word_count - 1] &
             ~final_mask) != 0) {
          throw std::invalid_argument(
              "subset contains an out-of-range atom");
        }
      }
    }

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const auto element_word_count = (group_order + 63) / 64;
    fast_math_internal::parallel_for_static(
        subset_count,
        thread_count,
        [&](std::size_t subset) {
          auto* output = element_words + subset * element_word_count;
          std::fill_n(output, element_word_count, 0);
          for (std::uint32_t word = 0;
               word < subset_word_count;
               ++word) {
            auto active =
                subset_words[subset * subset_word_count + word];
            while (active != 0) {
              const auto bit =
                  static_cast<std::uint32_t>(std::countr_zero(active));
              const auto atom = word * 64 + bit;
              if (atom >= atom_count) {
                break;
              }
              for (auto offset = atom_offsets[atom];
                   offset < atom_offsets[atom + 1];
                   ++offset) {
                const auto element = atom_elements[offset];
                output[element / 64] |=
                    std::uint64_t{1} << (element % 64);
              }
              active &= active - 1;
            }
          }
        });
    stats->degree = group_order;
    stats->generator_count = atom_count;
    stats->item_count = subset_count;
    stats->thread_count =
        fast_math_internal::parallel_worker_count(
            subset_count, thread_count);
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

int fast_math_cayley_graphs_u32(
    const std::uint32_t* multiplication_table,
    std::uint32_t group_order,
    const std::uint64_t* connection_words,
    std::size_t connection_count,
    std::uint32_t word_count,
    std::uint32_t thread_count,
    std::uint64_t* adjacency_words,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    validate_multiplication_table(multiplication_table, group_order);
    if (word_count != (group_order + 63) / 64) {
      throw std::invalid_argument("connection packed shape is invalid");
    }
    if (stats == nullptr ||
        (connection_count != 0 &&
         (connection_words == nullptr || adjacency_words == nullptr))) {
      throw std::invalid_argument("Cayley graph pointer is null");
    }
    const auto final_bits = group_order % 64;
    const auto final_mask = final_bits == 0
        ? ~std::uint64_t{0}
        : (std::uint64_t{1} << final_bits) - 1;
    for (std::size_t connection = 0;
         connection < connection_count;
         ++connection) {
      const auto* words =
          connection_words + connection * word_count;
      if ((words[word_count - 1] & ~final_mask) != 0) {
        throw std::invalid_argument(
            "connection set contains an out-of-range element");
      }
    }

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    fast_math_internal::parallel_for_static(
        connection_count,
        thread_count,
        [&](std::size_t connection) {
          const auto* words =
              connection_words + connection * word_count;
          auto* output =
              adjacency_words +
              connection * group_order * word_count;
          std::fill_n(
              output,
              static_cast<std::size_t>(group_order) * word_count,
              0);
          for (std::uint32_t step = 0; step < group_order; ++step) {
            if ((words[step / 64] &
                 (std::uint64_t{1} << (step % 64))) == 0) {
              continue;
            }
            for (std::uint32_t vertex = 0;
                 vertex < group_order;
                 ++vertex) {
              const auto neighbor = multiplication_table[
                  static_cast<std::size_t>(step) * group_order + vertex];
              output[
                  static_cast<std::size_t>(vertex) * word_count +
                  neighbor / 64] |=
                  std::uint64_t{1} << (neighbor % 64);
            }
          }
        });
    stats->degree = group_order;
    stats->item_count = connection_count;
    stats->thread_count =
        fast_math_internal::parallel_worker_count(
            connection_count,
            thread_count);
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

int fast_math_derivative_orbits_u32(
    const std::uint32_t* multiplication_table,
    const std::uint32_t* inverse_indices,
    const std::uint32_t* bijection,
    std::uint32_t group_order,
    std::uint32_t* derivative_generators,
    std::uint32_t* orbit_labels,
    std::uint32_t* orbit_count,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (group_order == 0 || group_order > 4096 ||
        multiplication_table == nullptr) {
      throw std::invalid_argument(
          "derivative group order must be between one and 4096");
    }
    const auto table_entry_count =
        static_cast<std::size_t>(group_order) * group_order;
    for (std::size_t index = 0; index < table_entry_count; ++index) {
      if (multiplication_table[index] >= group_order) {
        throw std::invalid_argument(
            "multiplication table entry is out of range");
      }
    }
    if (inverse_indices == nullptr || bijection == nullptr ||
        orbit_labels == nullptr || orbit_count == nullptr ||
        stats == nullptr) {
      throw std::invalid_argument("derivative pointer is null");
    }
    const auto mapping = read_permutations(
        bijection,
        1,
        group_order,
        "bijection is not a permutation",
        4096)[0];
    for (std::uint32_t element = 0;
         element < group_order;
         ++element) {
      if (inverse_indices[element] >= group_order) {
        throw std::invalid_argument("inverse index is out of range");
      }
    }
    const auto mapping_inverse = invert(mapping);
    std::vector<Permutation> derivatives(
        group_order,
        Permutation(group_order));

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    for (std::uint32_t vertex = 0;
         vertex < group_order;
         ++vertex) {
      const auto mapped_vertex_inverse =
          inverse_indices[mapping[vertex]];
      for (std::uint32_t connection = 0;
           connection < group_order;
           ++connection) {
        const auto product = multiplication_table[
            static_cast<std::size_t>(connection) * group_order +
            vertex];
        const auto difference = multiplication_table[
            static_cast<std::size_t>(mapping[product]) * group_order +
            mapped_vertex_inverse];
        derivatives[vertex][connection] =
            mapping_inverse[difference];
      }
      if (derivative_generators != nullptr) {
        std::copy(
            derivatives[vertex].begin(),
            derivatives[vertex].end(),
            derivative_generators +
                static_cast<std::size_t>(vertex) * group_order);
      }
    }
    *orbit_count = point_orbits(
        derivatives,
        group_order,
        orbit_labels);
    stats->degree = group_order;
    stats->generator_count = group_order;
    stats->class_count = *orbit_count;
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

int fast_math_wl2_refine_u32(
    const std::uint32_t* initial_relations,
    std::uint32_t vertex_count,
    std::uint32_t* stable_relations,
    std::uint32_t* relation_count,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (vertex_count == 0 || vertex_count > 512 ||
        initial_relations == nullptr || stable_relations == nullptr ||
        relation_count == nullptr || stats == nullptr) {
      throw std::invalid_argument("2-WL input or output pointer is null");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const auto pair_count =
        static_cast<std::size_t>(vertex_count) * vertex_count;
    std::vector<std::uint32_t> relations(
        initial_relations,
        initial_relations + pair_count);
    std::uint64_t iterations = 0;
    *relation_count = refine_wl2(
        relations,
        vertex_count,
        &iterations);
    std::copy(
        relations.begin(),
        relations.end(),
        stable_relations);
    stats->degree = vertex_count;
    stats->relation_count = *relation_count;
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

int fast_math_intersection_numbers_u64(
    const std::uint32_t* relations,
    std::uint32_t vertex_count,
    std::uint32_t relation_count,
    std::uint64_t* intersection_numbers,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (vertex_count == 0 || relation_count == 0 ||
        relations == nullptr || intersection_numbers == nullptr ||
        stats == nullptr) {
      throw std::invalid_argument(
          "intersection-number input or output pointer is null");
    }
    const auto pair_count =
        static_cast<std::size_t>(vertex_count) * vertex_count;
    std::vector<std::size_t> representatives(
        relation_count,
        pair_count);
    for (std::size_t pair = 0; pair < pair_count; ++pair) {
      const auto relation = relations[pair];
      if (relation >= relation_count) {
        throw std::invalid_argument("relation index is out of range");
      }
      if (representatives[relation] == pair_count) {
        representatives[relation] = pair;
      }
    }
    if (std::find(
            representatives.begin(),
            representatives.end(),
            pair_count) != representatives.end()) {
      throw std::invalid_argument("relation indices must be dense");
    }

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const auto tensor_size =
        static_cast<std::size_t>(relation_count) *
        relation_count * relation_count;
    std::fill_n(intersection_numbers, tensor_size, 0);
    auto tensor_index = [&](std::uint32_t first,
                            std::uint32_t second,
                            std::uint32_t target) {
      return (
          static_cast<std::size_t>(first) * relation_count + second
      ) * relation_count + target;
    };
    for (std::uint32_t target = 0;
         target < relation_count;
         ++target) {
      const auto pair = representatives[target];
      const auto left =
          static_cast<std::uint32_t>(pair / vertex_count);
      const auto right =
          static_cast<std::uint32_t>(pair % vertex_count);
      for (std::uint32_t middle = 0;
           middle < vertex_count;
           ++middle) {
        const auto first = relations[
            static_cast<std::size_t>(left) * vertex_count + middle];
        const auto second = relations[
            static_cast<std::size_t>(middle) * vertex_count + right];
        ++intersection_numbers[tensor_index(first, second, target)];
      }
    }
    std::vector<std::uint64_t> counts(
        static_cast<std::size_t>(relation_count) * relation_count);
    for (std::uint32_t left = 0; left < vertex_count; ++left) {
      for (std::uint32_t right = 0; right < vertex_count; ++right) {
        std::fill(counts.begin(), counts.end(), 0);
        const auto target = relations[
            static_cast<std::size_t>(left) * vertex_count + right];
        for (std::uint32_t middle = 0;
             middle < vertex_count;
             ++middle) {
          const auto first = relations[
              static_cast<std::size_t>(left) * vertex_count + middle];
          const auto second = relations[
              static_cast<std::size_t>(middle) * vertex_count + right];
          ++counts[
              static_cast<std::size_t>(first) * relation_count + second];
        }
        for (std::uint32_t first = 0;
             first < relation_count;
             ++first) {
          for (std::uint32_t second = 0;
               second < relation_count;
               ++second) {
            if (counts[
                    static_cast<std::size_t>(first) * relation_count +
                    second] !=
                intersection_numbers[
                    tensor_index(first, second, target)]) {
              throw std::invalid_argument(
                  "relation partition is not coherent");
            }
          }
        }
      }
    }
    stats->degree = vertex_count;
    stats->relation_count = relation_count;
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
