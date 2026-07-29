#include "fast_math.h"
#include "parallel.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
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

std::uint64_t vertex_mask(std::uint32_t vertex_count) {
  return vertex_count == 64
      ? std::numeric_limits<std::uint64_t>::max()
      : (std::uint64_t{1} << vertex_count) - 1;
}

std::uint64_t pair_count(std::uint32_t vertex_count) {
  return static_cast<std::uint64_t>(vertex_count) *
      static_cast<std::uint64_t>(vertex_count - 1) / 2;
}

std::uint64_t binomial(
    std::uint32_t total,
    std::uint32_t chosen) {
  if (chosen > total) {
    return 0;
  }
  chosen = std::min(chosen, total - chosen);
  std::uint64_t result = 1;
  for (std::uint32_t index = 1; index <= chosen; ++index) {
    result = result * (total - chosen + index) / index;
  }
  return result;
}

std::uint64_t vertices_after(
    std::uint32_t vertex,
    std::uint64_t all_vertices) {
  if (vertex == 63) {
    return 0;
  }
  return all_vertices &
      ~((std::uint64_t{1} << (vertex + 1)) - 1);
}

std::array<std::uint64_t, 64> suffix_masks(
    std::uint32_t vertex_count) {
  std::array<std::uint64_t, 64> masks{};
  const auto all_vertices = vertex_mask(vertex_count);
  for (std::uint32_t vertex = 0; vertex < vertex_count; ++vertex) {
    masks[vertex] = vertices_after(vertex, all_vertices);
  }
  return masks;
}

enum class GraphValidationError : std::uint8_t {
  none,
  invalid_vertex,
  asymmetric,
};

GraphValidationError validate_graph_pairwise(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count,
    std::uint64_t all_vertices) noexcept {
  for (std::uint32_t left = 0; left < vertex_count; ++left) {
    if ((adjacency[left] & ~all_vertices) != 0 ||
        (adjacency[left] & (std::uint64_t{1} << left)) != 0) {
      return GraphValidationError::invalid_vertex;
    }
    for (std::uint32_t right = left + 1;
         right < vertex_count;
         ++right) {
      const bool forward =
          (adjacency[left] & (std::uint64_t{1} << right)) != 0;
      const bool reverse =
          (adjacency[right] & (std::uint64_t{1} << left)) != 0;
      if (forward != reverse) {
        return GraphValidationError::asymmetric;
      }
    }
  }
  return GraphValidationError::none;
}

void transpose64(std::array<std::uint64_t, 64>& rows) {
  auto shift = 32u;
  auto mask = std::uint64_t{0x00000000FFFFFFFF};
  while (shift != 0) {
    for (std::uint32_t left = 0; left < 64;) {
      const auto right = left + shift;
      const auto changed =
          (rows[left] ^ (rows[right] >> shift)) & mask;
      rows[left] ^= changed;
      rows[right] ^= changed << shift;
      left = (left + shift + 1) & ~shift;
    }
    shift >>= 1;
    if (shift != 0) {
      mask ^= mask << shift;
    }
  }
}

std::uint64_t reverse_bits64(std::uint64_t value) {
#if defined(__has_builtin)
#if __has_builtin(__builtin_bitreverse64)
  return __builtin_bitreverse64(value);
#endif
#endif
  value = ((value >> 1) & 0x5555555555555555ULL) |
      ((value & 0x5555555555555555ULL) << 1);
  value = ((value >> 2) & 0x3333333333333333ULL) |
      ((value & 0x3333333333333333ULL) << 2);
  value = ((value >> 4) & 0x0F0F0F0F0F0F0F0FULL) |
      ((value & 0x0F0F0F0F0F0F0F0FULL) << 4);
  value = ((value >> 8) & 0x00FF00FF00FF00FFULL) |
      ((value & 0x00FF00FF00FF00FFULL) << 8);
  value = ((value >> 16) & 0x0000FFFF0000FFFFULL) |
      ((value & 0x0000FFFF0000FFFFULL) << 16);
  return (value >> 32) | (value << 32);
}

GraphValidationError validate_graph_swar(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count,
    std::uint64_t all_vertices) noexcept {
  std::array<std::uint64_t, 64> rows{};
  for (std::uint32_t vertex = 0; vertex < vertex_count; ++vertex) {
    const auto row = adjacency[vertex];
    if ((row & ~all_vertices) != 0 ||
        (row & (std::uint64_t{1} << vertex)) != 0) {
      return GraphValidationError::invalid_vertex;
    }
    rows[vertex] = row;
  }
  transpose64(rows);
  for (std::uint32_t vertex = 0; vertex < 64; ++vertex) {
    const auto source = 63 - vertex;
    const auto expected = source < vertex_count
        ? reverse_bits64(adjacency[source])
        : 0;
    if (rows[vertex] != expected) {
      return GraphValidationError::asymmetric;
    }
  }
  return GraphValidationError::none;
}

GraphValidationError validate_graph(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count,
    std::uint64_t all_vertices) noexcept {
  return vertex_count < 32
      ? validate_graph_pairwise(
            adjacency, vertex_count, all_vertices)
      : validate_graph_swar(
            adjacency, vertex_count, all_vertices);
}

[[noreturn]] void throw_validation_error(
    GraphValidationError error) {
  if (error == GraphValidationError::invalid_vertex) {
    throw std::invalid_argument(
        "adjacency masks contain an invalid vertex or self-loop");
  }
  throw std::invalid_argument(
      "adjacency masks must describe undirected graphs");
}

constexpr std::size_t kParallelValidationMinimumGraphs = 256;

void validate_graphs(
    const std::uint64_t* adjacency_masks,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    std::uint32_t thread_count) {
  if (adjacency_masks == nullptr) {
    throw std::invalid_argument("adjacency pointer is null");
  }
  if (graph_count == 0 || vertex_count == 0 || vertex_count > 64) {
    throw std::invalid_argument(
        "graph batch must have 1-64 vertices and be nonempty");
  }
  const auto all_vertices = vertex_mask(vertex_count);
  const auto workers = fast_math_internal::parallel_worker_count(
      graph_count, thread_count);
  if (workers == 1 ||
      graph_count < kParallelValidationMinimumGraphs) {
    for (std::size_t graph = 0; graph < graph_count; ++graph) {
      const auto error = validate_graph(
          adjacency_masks + graph * vertex_count,
          vertex_count,
          all_vertices);
      if (error != GraphValidationError::none) {
        throw_validation_error(error);
      }
    }
    return;
  }

  struct ValidationFailure {
    std::size_t graph;
    GraphValidationError error;
  };
  std::vector<ValidationFailure> failures(
      workers,
      ValidationFailure{
          graph_count, GraphValidationError::none});
  fast_math_internal::parallel_for_static_indexed(
      graph_count,
      workers,
      [&](std::size_t graph, std::size_t worker) noexcept {
        auto& failure = failures[worker];
        if (graph >= failure.graph) {
          return;
        }
        const auto error = validate_graph(
            adjacency_masks + graph * vertex_count,
            vertex_count,
            all_vertices);
        if (error != GraphValidationError::none) {
          failure = {graph, error};
        }
      });

  auto first_failure = ValidationFailure{
      graph_count, GraphValidationError::none};
  for (const auto& failure : failures) {
    if (failure.graph < first_failure.graph) {
      first_failure = failure;
    }
  }
  if (first_failure.error != GraphValidationError::none) {
    throw_validation_error(first_failure.error);
  }
}

std::uint64_t choose_three(std::uint32_t value) {
  return value < 3
      ? 0
      : static_cast<std::uint64_t>(value) *
          static_cast<std::uint64_t>(value - 1) *
          static_cast<std::uint64_t>(value - 2) / 6;
}

std::uint64_t count_triangles_walk(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count,
    const std::array<std::uint64_t, 64>& suffixes) {
  std::uint64_t triangles = 0;
  for (std::uint32_t left = 0; left < vertex_count; ++left) {
    auto later = adjacency[left] & suffixes[left];
    while (later != 0) {
      const auto right = static_cast<std::uint32_t>(
          std::countr_zero(later));
      later &= later - 1;
      triangles += std::popcount(
          adjacency[left] & adjacency[right] & suffixes[right]);
    }
  }
  return triangles;
}

struct CliqueSearch {
  const std::uint64_t* adjacency;
  std::uint64_t nodes_visited = 0;
  std::uint64_t witness = 0;

  bool search(
      std::uint64_t candidates,
      std::uint32_t remaining,
      std::uint64_t chosen) {
    nodes_visited += 1;
    if (remaining == 0) {
      witness = chosen;
      return true;
    }
    while (static_cast<std::uint32_t>(
               std::popcount(candidates)) >= remaining) {
      const auto vertex = static_cast<std::uint32_t>(
          std::countr_zero(candidates));
      const auto bit = std::uint64_t{1} << vertex;
      candidates &= ~bit;
      if (search(
              candidates & adjacency[vertex],
              remaining - 1,
              chosen | bit)) {
        return true;
      }
    }
    return false;
  }
};

template <typename Work>
void run_graphs(
    std::size_t graph_count,
    std::uint32_t thread_count,
    Work work) {
  fast_math_internal::parallel_for_dynamic(
      graph_count, thread_count, work);
}

struct InducedSubsetPlan {
  std::array<std::uint8_t, 21> left_vertices{};
  std::array<std::uint8_t, 21> right_vertices{};
  std::uint8_t edge_count = 0;
};

constexpr std::uint64_t kMaximumPlannedSubsets = 200'000;

std::vector<InducedSubsetPlan> build_induced_subset_plans(
    std::uint32_t vertex_count,
    std::uint32_t induced_order) {
  const auto subset_count = binomial(vertex_count, induced_order);
  if (subset_count > kMaximumPlannedSubsets) {
    return {};
  }
  std::vector<InducedSubsetPlan> plans;
  plans.reserve(static_cast<std::size_t>(subset_count));
  std::array<std::uint32_t, 7> vertices{};
  for (std::uint32_t position = 0;
       position < induced_order;
       ++position) {
    vertices[position] = position;
  }
  while (true) {
    InducedSubsetPlan plan;
    for (std::uint32_t left = 0;
         left < induced_order;
         ++left) {
      for (std::uint32_t right = left + 1;
           right < induced_order;
           ++right) {
        const auto edge = plan.edge_count++;
        plan.left_vertices[edge] =
            static_cast<std::uint8_t>(vertices[left]);
        plan.right_vertices[edge] =
            static_cast<std::uint8_t>(vertices[right]);
      }
    }
    plans.push_back(plan);

    auto position = static_cast<std::int32_t>(induced_order) - 1;
    while (position >= 0 &&
           vertices[static_cast<std::size_t>(position)] ==
               vertex_count - induced_order +
               static_cast<std::uint32_t>(position)) {
      position -= 1;
    }
    if (position < 0) {
      break;
    }
    const auto changed = static_cast<std::size_t>(position);
    vertices[changed] += 1;
    for (auto next = changed + 1;
         next < induced_order;
         ++next) {
      vertices[next] = vertices[next - 1] + 1;
    }
  }
  return plans;
}

void count_induced_profile_planned(
    const std::uint64_t* adjacency,
    const std::vector<InducedSubsetPlan>& plans,
    const std::uint32_t* class_lookup,
    std::uint64_t* graph_counts) {
  for (const auto& plan : plans) {
    std::uint32_t raw_mask = 0;
    for (std::uint32_t edge = 0; edge < plan.edge_count; ++edge) {
      raw_mask |= static_cast<std::uint32_t>(
          (adjacency[plan.left_vertices[edge]] >>
           plan.right_vertices[edge]) &
          1) << edge;
    }
    graph_counts[class_lookup[raw_mask]] += 1;
  }
}

void count_induced_profile_unplanned(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count,
    std::uint32_t induced_order,
    const std::uint32_t* class_lookup,
    std::uint64_t* graph_counts) {
  std::array<std::uint32_t, 7> vertices{};
  for (std::uint32_t position = 0;
       position < induced_order;
       ++position) {
    vertices[position] = position;
  }
  while (true) {
    std::uint32_t raw_mask = 0;
    std::uint32_t edge = 0;
    for (std::uint32_t left = 0;
         left < induced_order;
         ++left) {
      for (std::uint32_t right = left + 1;
           right < induced_order;
           ++right) {
        if ((adjacency[vertices[left]] &
             (std::uint64_t{1} << vertices[right])) != 0) {
          raw_mask |= std::uint32_t{1} << edge;
        }
        edge += 1;
      }
    }
    graph_counts[class_lookup[raw_mask]] += 1;

    auto position = static_cast<std::int32_t>(induced_order) - 1;
    while (position >= 0 &&
           vertices[static_cast<std::size_t>(position)] ==
               vertex_count - induced_order +
               static_cast<std::uint32_t>(position)) {
      position -= 1;
    }
    if (position < 0) {
      break;
    }
    const auto changed = static_cast<std::size_t>(position);
    vertices[changed] += 1;
    for (auto next = changed + 1;
         next < induced_order;
         ++next) {
      vertices[next] = vertices[next - 1] + 1;
    }
  }
}

}  // namespace

extern "C" {

int fast_math_graph_pair_profiles_u64(
    const std::uint64_t* adjacency_masks,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    std::uint32_t thread_count,
    std::uint8_t* adjacent,
    std::uint32_t* common_neighbors,
    std::uint32_t* common_nonneighbors,
    std::uint32_t* only_left,
    std::uint32_t* only_right,
    fast_math_graph_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    validate_graphs(
        adjacency_masks, graph_count, vertex_count, thread_count);
    const auto pairs = pair_count(vertex_count);
    if (stats == nullptr ||
        (pairs != 0 &&
         (adjacent == nullptr || common_neighbors == nullptr ||
          common_nonneighbors == nullptr || only_left == nullptr ||
          only_right == nullptr))) {
      throw std::invalid_argument("pair-profile output pointer is null");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    run_graphs(
        graph_count,
        thread_count,
        [&](std::size_t graph) {
          const auto* adjacency_masks_for_graph =
              adjacency_masks + graph * vertex_count;
          std::array<std::uint32_t, 64> degrees{};
          for (std::uint32_t vertex = 0;
               vertex < vertex_count;
               ++vertex) {
            degrees[vertex] = static_cast<std::uint32_t>(
                std::popcount(adjacency_masks_for_graph[vertex]));
          }
          auto output = graph * pairs;
          for (std::uint32_t left = 0;
               left < vertex_count;
               ++left) {
            for (std::uint32_t right = left + 1;
                 right < vertex_count;
                 ++right) {
              const auto is_adjacent = static_cast<std::uint32_t>(
                  (adjacency_masks_for_graph[left] >> right) & 1);
              const auto common = static_cast<std::uint32_t>(
                  std::popcount(
                      adjacency_masks_for_graph[left] &
                      adjacency_masks_for_graph[right]));
              const auto left_only =
                  degrees[left] - is_adjacent - common;
              const auto right_only =
                  degrees[right] - is_adjacent - common;
              adjacent[output] =
                  static_cast<std::uint8_t>(is_adjacent);
              common_neighbors[output] = common;
              only_left[output] = left_only;
              only_right[output] = right_only;
              common_nonneighbors[output] =
                  vertex_count - 2 -
                  common - left_only - right_only;
              output += 1;
            }
          }
        });
    stats->graph_count = graph_count;
    stats->vertex_count = vertex_count;
    stats->pair_count = pairs;
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

int fast_math_graph_find_clique_u64(
    const std::uint64_t* adjacency_masks,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    std::uint32_t order,
    bool complement,
    std::uint32_t thread_count,
    std::uint64_t* witnesses,
    std::uint64_t* nodes_visited,
    fast_math_graph_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    validate_graphs(
        adjacency_masks, graph_count, vertex_count, thread_count);
    if (witnesses == nullptr || nodes_visited == nullptr ||
        stats == nullptr) {
      throw std::invalid_argument("clique output pointer is null");
    }
    if (order == 0 || order > vertex_count) {
      throw std::invalid_argument("clique order is outside the graph");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const auto all_vertices = vertex_mask(vertex_count);
    run_graphs(
        graph_count,
        thread_count,
        [&](std::size_t graph) {
          const auto* source_adjacency =
              adjacency_masks + graph * vertex_count;
          std::array<std::uint64_t, 64> effective_adjacency{};
          for (std::uint32_t vertex = 0;
               vertex < vertex_count;
               ++vertex) {
            if (complement) {
              effective_adjacency[vertex] =
                  all_vertices &
                  ~source_adjacency[vertex] &
                  ~(std::uint64_t{1} << vertex);
            } else {
              effective_adjacency[vertex] =
                  source_adjacency[vertex];
            }
          }
          CliqueSearch search{effective_adjacency.data()};
          search.search(all_vertices, order, 0);
          witnesses[graph] = search.witness;
          nodes_visited[graph] = search.nodes_visited;
        });
    std::uint64_t total_nodes = 0;
    for (std::size_t graph = 0; graph < graph_count; ++graph) {
      total_nodes += nodes_visited[graph];
    }
    stats->graph_count = graph_count;
    stats->vertex_count = vertex_count;
    stats->pair_count = pair_count(vertex_count);
    stats->nodes_visited = total_nodes;
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

int fast_math_graph6_decode_u64(
    const std::uint8_t* data,
    std::size_t data_size,
    const std::uint64_t* offsets,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    std::uint32_t thread_count,
    std::uint64_t* adjacency_masks,
    fast_math_graph_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (data == nullptr || offsets == nullptr ||
        adjacency_masks == nullptr || stats == nullptr) {
      throw std::invalid_argument("graph6 pointer is null");
    }
    if (graph_count == 0 || vertex_count == 0 || vertex_count > 62) {
      throw std::invalid_argument(
          "short-form graph6 batch must have 1-62 vertices");
    }
    if (offsets[0] != 0 || offsets[graph_count] != data_size) {
      throw std::invalid_argument("graph6 offsets must cover the input");
    }
    const auto edge_bits =
        static_cast<std::uint64_t>(vertex_count) *
        static_cast<std::uint64_t>(vertex_count - 1) / 2;
    const auto encoded_size =
        static_cast<std::uint64_t>(1 + (edge_bits + 5) / 6);
    for (std::size_t graph = 0; graph < graph_count; ++graph) {
      if (offsets[graph + 1] - offsets[graph] != encoded_size) {
        throw std::invalid_argument(
            "graph6 records must be unheaded short-form encodings");
      }
      const auto begin =
          static_cast<std::size_t>(offsets[graph]);
      const auto end =
          static_cast<std::size_t>(offsets[graph + 1]);
      if (data[begin] != vertex_count + 63) {
        throw std::invalid_argument(
            "graph6 records must have one shared vertex count");
      }
      for (auto index = begin; index < end; ++index) {
        if (data[index] < 63 || data[index] > 126) {
          throw std::invalid_argument(
              "graph6 record contains an invalid byte");
        }
      }
    }

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    run_graphs(
        graph_count,
        thread_count,
        [&](std::size_t graph) {
          const auto begin =
              static_cast<std::size_t>(offsets[graph]);
          auto* adjacency =
              adjacency_masks + graph * vertex_count;
          std::fill_n(adjacency, vertex_count, 0);
          std::uint64_t bit_index = 0;
          for (std::uint32_t right = 1;
               right < vertex_count;
               ++right) {
            for (std::uint32_t left = 0;
                 left < right;
                 ++left) {
              const auto encoded =
                  static_cast<std::uint8_t>(
                      data[begin + 1 + bit_index / 6] - 63);
              const auto present =
                  (encoded >> (5 - bit_index % 6)) & 1;
              bit_index += 1;
              if (present == 0) {
                continue;
              }
              adjacency[left] |= std::uint64_t{1} << right;
              adjacency[right] |= std::uint64_t{1} << left;
            }
          }
        });
    stats->graph_count = graph_count;
    stats->vertex_count = vertex_count;
    stats->pair_count = pair_count(vertex_count);
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

int fast_math_graph6_encode_u64(
    const std::uint64_t* adjacency_masks,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    std::uint32_t thread_count,
    std::uint8_t* data,
    std::size_t data_size,
    fast_math_graph_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (data == nullptr || stats == nullptr) {
      throw std::invalid_argument("graph6 output pointer is null");
    }
    if (graph_count == 0 || vertex_count == 0 || vertex_count > 62) {
      throw std::invalid_argument(
          "short-form graph6 batch must have 1-62 vertices");
    }
    const auto edge_bits =
        static_cast<std::uint64_t>(vertex_count) *
        static_cast<std::uint64_t>(vertex_count - 1) / 2;
    const auto encoded_size =
        static_cast<std::size_t>(1 + (edge_bits + 5) / 6);
    if (graph_count >
        std::numeric_limits<std::size_t>::max() / encoded_size ||
        data_size != graph_count * encoded_size) {
      throw std::invalid_argument(
          "graph6 output size does not match the graph batch");
    }
    validate_graphs(
        adjacency_masks, graph_count, vertex_count, thread_count);

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    run_graphs(
        graph_count,
        thread_count,
        [&](std::size_t graph) {
          const auto* adjacency =
              adjacency_masks + graph * vertex_count;
          auto* record = data + graph * encoded_size;
          record[0] = static_cast<std::uint8_t>(vertex_count + 63);
          std::fill_n(
              record + 1,
              encoded_size - 1,
              static_cast<std::uint8_t>(63));
          std::uint64_t bit_index = 0;
          for (std::uint32_t right = 1;
               right < vertex_count;
               ++right) {
            for (std::uint32_t left = 0;
                 left < right;
                 ++left) {
              if ((adjacency[left] &
                   (std::uint64_t{1} << right)) != 0) {
                record[1 + bit_index / 6] +=
                    static_cast<std::uint8_t>(
                        1U << (5 - bit_index % 6));
              }
              ++bit_index;
            }
          }
        });
    stats->graph_count = graph_count;
    stats->vertex_count = vertex_count;
    stats->pair_count = pair_count(vertex_count);
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

int fast_math_graph_invariants_u64(
    const std::uint64_t* adjacency_masks,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    std::uint32_t thread_count,
    std::uint32_t* degrees,
    std::uint64_t* edge_counts,
    std::uint64_t* triangle_counts,
    std::uint64_t* wedge_counts,
    std::uint64_t* induced_path3_counts,
    fast_math_graph_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    validate_graphs(
        adjacency_masks, graph_count, vertex_count, thread_count);
    if (degrees == nullptr || edge_counts == nullptr ||
        triangle_counts == nullptr || wedge_counts == nullptr ||
        induced_path3_counts == nullptr || stats == nullptr) {
      throw std::invalid_argument("graph-invariant output pointer is null");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const auto all_vertices = vertex_mask(vertex_count);
    const auto suffixes = suffix_masks(vertex_count);
    const auto possible_edges = pair_count(vertex_count);
    run_graphs(
        graph_count,
        thread_count,
        [&](std::size_t graph) {
          const auto* adjacency =
              adjacency_masks + graph * vertex_count;
          auto* graph_degrees = degrees + graph * vertex_count;
          std::uint64_t degree_sum = 0;
          std::uint64_t wedges = 0;
          std::uint64_t mixed_twice = 0;
          for (std::uint32_t vertex = 0;
               vertex < vertex_count;
               ++vertex) {
            const auto degree = static_cast<std::uint32_t>(
                std::popcount(adjacency[vertex]));
            graph_degrees[vertex] = degree;
            degree_sum += degree;
            if (degree >= 2) {
              wedges += static_cast<std::uint64_t>(degree) *
                  static_cast<std::uint64_t>(degree - 1) / 2;
            }
            mixed_twice += static_cast<std::uint64_t>(degree) *
                static_cast<std::uint64_t>(
                    vertex_count - 1 - degree);
          }
          const auto edges = degree_sum / 2;
          std::uint64_t triangles;
          if (4 * edges < possible_edges) {
            triangles = count_triangles_walk(
                adjacency, vertex_count, suffixes);
          } else if (
              4 * (possible_edges - edges) < possible_edges) {
            std::array<std::uint64_t, 64> complement{};
            for (std::uint32_t vertex = 0;
                 vertex < vertex_count;
                 ++vertex) {
              complement[vertex] =
                  (~adjacency[vertex]) &
                  all_vertices &
                  ~(std::uint64_t{1} << vertex);
            }
            const auto complement_triangles = count_triangles_walk(
                complement.data(), vertex_count, suffixes);
            triangles =
                choose_three(vertex_count) -
                mixed_twice / 2 -
                complement_triangles;
          } else {
            std::uint64_t triangle_edge_incidences = 0;
            for (std::uint32_t left = 0;
                 left < vertex_count;
                 ++left) {
              const auto left_neighbors = adjacency[left];
              for (std::uint32_t right = left + 1;
                   right < vertex_count;
                   ++right) {
                const auto present =
                    (left_neighbors >> right) & std::uint64_t{1};
                triangle_edge_incidences += present * std::popcount(
                    left_neighbors & adjacency[right]);
              }
            }
            triangles = triangle_edge_incidences / 3;
          }
          edge_counts[graph] = edges;
          triangle_counts[graph] = triangles;
          wedge_counts[graph] = wedges;
          induced_path3_counts[graph] = wedges - 3 * triangles;
        });
    stats->graph_count = graph_count;
    stats->vertex_count = vertex_count;
    stats->pair_count = pair_count(vertex_count);
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

int fast_math_graph_induced_profiles_u64(
    const std::uint64_t* adjacency_masks,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    std::uint32_t induced_order,
    const std::uint32_t* class_lookup,
    std::size_t lookup_size,
    std::uint32_t class_count,
    std::uint32_t thread_count,
    std::uint64_t* counts,
    fast_math_graph_profile_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    validate_graphs(
        adjacency_masks, graph_count, vertex_count, thread_count);
    if (class_lookup == nullptr || counts == nullptr || stats == nullptr) {
      throw std::invalid_argument("induced-profile pointer is null");
    }
    if (induced_order == 0 || induced_order > vertex_count ||
        induced_order > 7) {
      throw std::invalid_argument(
          "induced order must be between 1 and min(vertices, 7)");
    }
    const auto induced_edge_count =
        induced_order * (induced_order - 1) / 2;
    const auto expected_lookup_size =
        std::size_t{1} << induced_edge_count;
    if (lookup_size != expected_lookup_size || class_count == 0) {
      throw std::invalid_argument(
          "class lookup has the wrong size or no classes");
    }
    for (std::size_t mask = 0; mask < lookup_size; ++mask) {
      if (class_lookup[mask] >= class_count) {
        throw std::invalid_argument(
            "class lookup contains an out-of-range class");
      }
    }
    if (graph_count >
        std::numeric_limits<std::size_t>::max() / class_count) {
      throw std::overflow_error("induced-profile output is too large");
    }

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    std::fill_n(counts, graph_count * class_count, 0);
    const auto subsets_per_graph =
        binomial(vertex_count, induced_order);
    const auto plans =
        build_induced_subset_plans(vertex_count, induced_order);
    run_graphs(
        graph_count,
        thread_count,
        [&](std::size_t graph) {
          const auto* adjacency =
              adjacency_masks + graph * vertex_count;
          auto* graph_counts = counts + graph * class_count;
          if (plans.empty()) {
            count_induced_profile_unplanned(
                adjacency,
                vertex_count,
                induced_order,
                class_lookup,
                graph_counts);
          } else {
            count_induced_profile_planned(
                adjacency,
                plans,
                class_lookup,
                graph_counts);
          }
        });
    stats->graph_count = graph_count;
    stats->vertex_count = vertex_count;
    stats->induced_order = induced_order;
    stats->class_count = class_count;
    stats->subsets_per_graph = subsets_per_graph;
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

int fast_math_graph_induced_profile_stack_u64(
    const std::uint64_t* adjacency_masks,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    const std::uint32_t* induced_orders,
    std::size_t order_count,
    const std::uint32_t* class_lookups,
    const std::uint64_t* lookup_offsets,
    const std::uint32_t* class_counts,
    std::uint32_t thread_count,
    std::uint64_t* counts,
    fast_math_graph_profile_stack_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    validate_graphs(
        adjacency_masks, graph_count, vertex_count, thread_count);
    if (induced_orders == nullptr || class_lookups == nullptr ||
        lookup_offsets == nullptr || class_counts == nullptr ||
        counts == nullptr || stats == nullptr) {
      throw std::invalid_argument("induced-profile stack pointer is null");
    }
    if (order_count == 0 || order_count > 7) {
      throw std::invalid_argument(
          "induced-profile stack must contain 1-7 orders");
    }
    if (lookup_offsets[0] != 0) {
      throw std::invalid_argument(
          "induced-profile lookup offsets must start at zero");
    }

    std::vector<std::size_t> output_offsets(order_count + 1, 0);
    std::vector<std::vector<InducedSubsetPlan>> plans(order_count);
    std::uint64_t subsets_per_graph = 0;
    std::uint32_t previous_order = 0;
    for (std::size_t order_index = 0;
         order_index < order_count;
         ++order_index) {
      const auto induced_order = induced_orders[order_index];
      if (induced_order <= previous_order ||
          induced_order > vertex_count || induced_order > 7) {
        throw std::invalid_argument(
            "induced-profile orders must be strictly increasing and valid");
      }
      previous_order = induced_order;
      const auto induced_edge_count =
          induced_order * (induced_order - 1) / 2;
      const auto expected_lookup_size =
          std::uint64_t{1} << induced_edge_count;
      if (lookup_offsets[order_index + 1] <
              lookup_offsets[order_index] ||
          lookup_offsets[order_index + 1] -
                  lookup_offsets[order_index] !=
              expected_lookup_size ||
          class_counts[order_index] == 0) {
        throw std::invalid_argument(
            "induced-profile stack lookup has the wrong size");
      }
      const auto lookup_begin =
          static_cast<std::size_t>(lookup_offsets[order_index]);
      const auto lookup_end =
          static_cast<std::size_t>(lookup_offsets[order_index + 1]);
      for (auto mask = lookup_begin; mask < lookup_end; ++mask) {
        if (class_lookups[mask] >= class_counts[order_index]) {
          throw std::invalid_argument(
              "induced-profile stack lookup class is out of range");
        }
      }
      if (output_offsets[order_index] >
          std::numeric_limits<std::size_t>::max() -
              class_counts[order_index]) {
        throw std::overflow_error(
            "induced-profile stack field count is too large");
      }
      output_offsets[order_index + 1] =
          output_offsets[order_index] + class_counts[order_index];
      subsets_per_graph += binomial(vertex_count, induced_order);
      plans[order_index] =
          build_induced_subset_plans(vertex_count, induced_order);
    }
    const auto field_count = output_offsets.back();
    if (graph_count >
        std::numeric_limits<std::size_t>::max() / field_count) {
      throw std::overflow_error(
          "induced-profile stack output is too large");
    }

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    std::fill_n(counts, graph_count * field_count, 0);
    run_graphs(
        graph_count,
        thread_count,
        [&](std::size_t graph) {
          const auto* adjacency =
              adjacency_masks + graph * vertex_count;
          auto* graph_counts = counts + graph * field_count;
          for (std::size_t order_index = 0;
               order_index < order_count;
               ++order_index) {
            const auto* lookup =
                class_lookups + lookup_offsets[order_index];
            auto* output =
                graph_counts + output_offsets[order_index];
            if (plans[order_index].empty()) {
              count_induced_profile_unplanned(
                  adjacency,
                  vertex_count,
                  induced_orders[order_index],
                  lookup,
                  output);
            } else {
              count_induced_profile_planned(
                  adjacency,
                  plans[order_index],
                  lookup,
                  output);
            }
          }
        });

    stats->graph_count = graph_count;
    stats->vertex_count = vertex_count;
    stats->order_count = static_cast<std::uint32_t>(order_count);
    stats->field_count = field_count;
    stats->subsets_per_graph = subsets_per_graph;
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
