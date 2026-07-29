#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Totals {
  std::uint64_t degrees = 0;
  std::uint64_t edges = 0;
  std::uint64_t triangles = 0;
  std::uint64_t wedges = 0;
  std::uint64_t mixed_twice = 0;

  Totals& operator+=(const Totals& other) {
    degrees += other.degrees;
    edges += other.edges;
    triangles += other.triangles;
    wedges += other.wedges;
    mixed_twice += other.mixed_twice;
    return *this;
  }
};

std::uint64_t vertex_mask(std::uint32_t vertex_count) {
  return vertex_count == 64
      ? std::numeric_limits<std::uint64_t>::max()
      : (std::uint64_t{1} << vertex_count) - 1;
}

std::array<std::uint64_t, 64> suffix_masks(
    std::uint32_t vertex_count) {
  std::array<std::uint64_t, 64> masks{};
  const auto all_vertices = vertex_mask(vertex_count);
  for (std::uint32_t vertex = 0; vertex < vertex_count; ++vertex) {
    masks[vertex] = vertex == 63
        ? 0
        : all_vertices &
            ~((std::uint64_t{1} << (vertex + 1)) - 1);
  }
  return masks;
}

std::uint64_t vertices_after(
    std::uint32_t vertex,
    std::uint64_t all_vertices) {
  return vertex == 63
      ? 0
      : all_vertices &
          ~((std::uint64_t{1} << (vertex + 1)) - 1);
}

std::vector<std::uint64_t> random_graphs(
    std::size_t graph_count,
    std::uint32_t vertex_count,
    double density,
    std::uint64_t seed) {
  std::vector<std::uint64_t> graphs(graph_count * vertex_count, 0);
  std::mt19937_64 generator(seed);
  std::bernoulli_distribution present(density);
  for (std::size_t graph = 0; graph < graph_count; ++graph) {
    auto* adjacency = graphs.data() + graph * vertex_count;
    for (std::uint32_t left = 0; left < vertex_count; ++left) {
      for (std::uint32_t right = left + 1;
           right < vertex_count;
           ++right) {
        if (!present(generator)) {
          continue;
        }
        adjacency[left] |= std::uint64_t{1} << right;
        adjacency[right] |= std::uint64_t{1} << left;
      }
    }
  }
  return graphs;
}

bool validate_pairwise(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count) {
  const auto all_vertices = vertex_mask(vertex_count);
  for (std::uint32_t left = 0; left < vertex_count; ++left) {
    if ((adjacency[left] & ~all_vertices) != 0 ||
        (adjacency[left] & (std::uint64_t{1} << left)) != 0) {
      return false;
    }
    for (std::uint32_t right = left + 1;
         right < vertex_count;
         ++right) {
      const auto forward =
          (adjacency[left] >> right) & std::uint64_t{1};
      const auto reverse =
          (adjacency[right] >> left) & std::uint64_t{1};
      if (forward != reverse) {
        return false;
      }
    }
  }
  return true;
}

bool validate_edges(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count) {
  const auto all_vertices = vertex_mask(vertex_count);
  for (std::uint32_t left = 0; left < vertex_count; ++left) {
    auto neighbors = adjacency[left];
    if ((neighbors & ~all_vertices) != 0 ||
        (neighbors & (std::uint64_t{1} << left)) != 0) {
      return false;
    }
    while (neighbors != 0) {
      const auto right = static_cast<std::uint32_t>(
          std::countr_zero(neighbors));
      neighbors &= neighbors - 1;
      if ((adjacency[right] &
           (std::uint64_t{1} << left)) == 0) {
        return false;
      }
    }
  }
  return true;
}

bool validate_transpose(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count) {
  const auto all_vertices = vertex_mask(vertex_count);
  std::array<std::uint64_t, 64> transpose{};
  for (std::uint32_t left = 0; left < vertex_count; ++left) {
    auto neighbors = adjacency[left];
    if ((neighbors & ~all_vertices) != 0 ||
        (neighbors & (std::uint64_t{1} << left)) != 0) {
      return false;
    }
    while (neighbors != 0) {
      const auto right = static_cast<std::uint32_t>(
          std::countr_zero(neighbors));
      neighbors &= neighbors - 1;
      transpose[right] |= std::uint64_t{1} << left;
    }
  }
  for (std::uint32_t vertex = 0;
       vertex < vertex_count;
       ++vertex) {
    if (transpose[vertex] != adjacency[vertex]) {
      return false;
    }
  }
  return true;
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
#if defined(__clang__) && __has_builtin(__builtin_bitreverse64)
  return __builtin_bitreverse64(value);
#else
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
#endif
}

bool validate_swar_transpose(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count) {
  const auto all_vertices = vertex_mask(vertex_count);
  std::array<std::uint64_t, 64> rows{};
  for (std::uint32_t vertex = 0; vertex < vertex_count; ++vertex) {
    const auto row = adjacency[vertex];
    if ((row & ~all_vertices) != 0 ||
        (row & (std::uint64_t{1} << vertex)) != 0) {
      return false;
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
      return false;
    }
  }
  return true;
}

Totals degree_totals(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count) {
  Totals totals;
  for (std::uint32_t vertex = 0; vertex < vertex_count; ++vertex) {
    const auto degree = static_cast<std::uint32_t>(
        std::popcount(adjacency[vertex]));
    totals.degrees += degree;
    totals.wedges += static_cast<std::uint64_t>(degree) *
        static_cast<std::uint64_t>(degree - (degree != 0)) / 2;
    totals.mixed_twice += static_cast<std::uint64_t>(degree) *
        static_cast<std::uint64_t>(vertex_count - 1 - degree);
  }
  totals.edges = totals.degrees / 2;
  return totals;
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

Totals invariants_suffix_walk(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count,
    const std::array<std::uint64_t, 64>& suffixes) {
  auto totals = degree_totals(adjacency, vertex_count);
  for (std::uint32_t left = 0; left < vertex_count; ++left) {
    auto later = adjacency[left] & suffixes[left];
    while (later != 0) {
      const auto right = static_cast<std::uint32_t>(
          std::countr_zero(later));
      later &= later - 1;
      totals.triangles += std::popcount(
          adjacency[left] & adjacency[right] & suffixes[right]);
    }
  }
  return totals;
}

Totals invariants_production(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count,
    const std::array<std::uint64_t, 64>&) {
  auto totals = degree_totals(adjacency, vertex_count);
  const auto all_vertices = vertex_mask(vertex_count);
  for (std::uint32_t left = 0; left < vertex_count; ++left) {
    auto later =
        adjacency[left] & vertices_after(left, all_vertices);
    while (later != 0) {
      const auto right = static_cast<std::uint32_t>(
          std::countr_zero(later));
      const auto bit = std::uint64_t{1} << right;
      later &= ~bit;
      totals.triangles += std::popcount(
          adjacency[left] &
          adjacency[right] &
          vertices_after(right, all_vertices));
    }
  }
  return totals;
}

Totals invariants_dynamic_clear_lowest(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count,
    const std::array<std::uint64_t, 64>&) {
  auto totals = degree_totals(adjacency, vertex_count);
  const auto all_vertices = vertex_mask(vertex_count);
  for (std::uint32_t left = 0; left < vertex_count; ++left) {
    auto later =
        adjacency[left] & vertices_after(left, all_vertices);
    while (later != 0) {
      const auto right = static_cast<std::uint32_t>(
          std::countr_zero(later));
      later &= later - 1;
      totals.triangles += std::popcount(
          adjacency[left] &
          adjacency[right] &
          vertices_after(right, all_vertices));
    }
  }
  return totals;
}

Totals invariants_precomputed_bit_clear(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count,
    const std::array<std::uint64_t, 64>& suffixes) {
  auto totals = degree_totals(adjacency, vertex_count);
  for (std::uint32_t left = 0; left < vertex_count; ++left) {
    auto later = adjacency[left] & suffixes[left];
    while (later != 0) {
      const auto right = static_cast<std::uint32_t>(
          std::countr_zero(later));
      const auto bit = std::uint64_t{1} << right;
      later &= ~bit;
      totals.triangles += std::popcount(
          adjacency[left] & adjacency[right] & suffixes[right]);
    }
  }
  return totals;
}

Totals invariants_common_walk(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count,
    const std::array<std::uint64_t, 64>& suffixes) {
  auto totals = degree_totals(adjacency, vertex_count);
  std::uint64_t triangle_edge_incidences = 0;
  for (std::uint32_t left = 0; left < vertex_count; ++left) {
    auto later = adjacency[left] & suffixes[left];
    while (later != 0) {
      const auto right = static_cast<std::uint32_t>(
          std::countr_zero(later));
      later &= later - 1;
      triangle_edge_incidences += std::popcount(
          adjacency[left] & adjacency[right]);
    }
  }
  totals.triangles = triangle_edge_incidences / 3;
  return totals;
}

Totals invariants_suffix_scan(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count,
    const std::array<std::uint64_t, 64>& suffixes) {
  auto totals = degree_totals(adjacency, vertex_count);
  for (std::uint32_t left = 0; left < vertex_count; ++left) {
    const auto left_neighbors = adjacency[left];
    for (std::uint32_t right = left + 1;
         right < vertex_count;
         ++right) {
      const auto present =
          (left_neighbors >> right) & std::uint64_t{1};
      totals.triangles += present * std::popcount(
          left_neighbors & adjacency[right] & suffixes[right]);
    }
  }
  return totals;
}

Totals invariants_common_scan(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count,
    const std::array<std::uint64_t, 64>&) {
  auto totals = degree_totals(adjacency, vertex_count);
  std::uint64_t triangle_edge_incidences = 0;
  for (std::uint32_t left = 0; left < vertex_count; ++left) {
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
  totals.triangles = triangle_edge_incidences / 3;
  return totals;
}

Totals invariants_adaptive(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count,
    const std::array<std::uint64_t, 64>& suffixes) {
  auto totals = degree_totals(adjacency, vertex_count);
  const auto possible_edges =
      static_cast<std::uint64_t>(vertex_count) *
      static_cast<std::uint64_t>(vertex_count - 1) / 2;
  if (4 * totals.edges < possible_edges) {
    for (std::uint32_t left = 0; left < vertex_count; ++left) {
      auto later = adjacency[left] & suffixes[left];
      while (later != 0) {
        const auto right = static_cast<std::uint32_t>(
            std::countr_zero(later));
        later &= later - 1;
        totals.triangles += std::popcount(
            adjacency[left] &
            adjacency[right] &
            suffixes[right]);
      }
    }
    return totals;
  }

  std::uint64_t triangle_edge_incidences = 0;
  for (std::uint32_t left = 0; left < vertex_count; ++left) {
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
  totals.triangles = triangle_edge_incidences / 3;
  return totals;
}

Totals invariants_min_edge_walk(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count,
    const std::array<std::uint64_t, 64>& suffixes) {
  auto totals = degree_totals(adjacency, vertex_count);
  const auto possible_edges =
      static_cast<std::uint64_t>(vertex_count) *
      static_cast<std::uint64_t>(vertex_count - 1) / 2;
  if (2 * totals.edges <= possible_edges) {
    totals.triangles =
        count_triangles_walk(adjacency, vertex_count, suffixes);
    return totals;
  }

  const auto all_vertices = vertex_mask(vertex_count);
  std::array<std::uint64_t, 64> complement{};
  for (std::uint32_t vertex = 0; vertex < vertex_count; ++vertex) {
    complement[vertex] =
        (~adjacency[vertex]) &
        all_vertices &
        ~(std::uint64_t{1} << vertex);
  }
  const auto complement_triangles = count_triangles_walk(
      complement.data(), vertex_count, suffixes);
  totals.triangles =
      choose_three(vertex_count) -
      totals.mixed_twice / 2 -
      complement_triangles;
  return totals;
}

Totals invariants_three_way(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count,
    const std::array<std::uint64_t, 64>& suffixes) {
  auto totals = degree_totals(adjacency, vertex_count);
  const auto possible_edges =
      static_cast<std::uint64_t>(vertex_count) *
      static_cast<std::uint64_t>(vertex_count - 1) / 2;
  if (4 * totals.edges < possible_edges) {
    totals.triangles =
        count_triangles_walk(adjacency, vertex_count, suffixes);
    return totals;
  }
  if (4 * (possible_edges - totals.edges) < possible_edges) {
    const auto all_vertices = vertex_mask(vertex_count);
    std::array<std::uint64_t, 64> complement{};
    for (std::uint32_t vertex = 0; vertex < vertex_count; ++vertex) {
      complement[vertex] =
          (~adjacency[vertex]) &
          all_vertices &
          ~(std::uint64_t{1} << vertex);
    }
    const auto complement_triangles = count_triangles_walk(
        complement.data(), vertex_count, suffixes);
    totals.triangles =
        choose_three(vertex_count) -
        totals.mixed_twice / 2 -
        complement_triangles;
    return totals;
  }

  std::uint64_t triangle_edge_incidences = 0;
  for (std::uint32_t left = 0; left < vertex_count; ++left) {
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
  totals.triangles = triangle_edge_incidences / 3;
  return totals;
}

template <typename Function>
double best_seconds(std::size_t repeats, Function function) {
  auto best = std::numeric_limits<double>::infinity();
  for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
    const auto started = Clock::now();
    function();
    const auto seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    best = std::min(best, seconds);
  }
  return best;
}

volatile std::uint64_t checksum_sink = 0;

bool same_totals(const Totals& left, const Totals& right) {
  return left.degrees == right.degrees &&
      left.edges == right.edges &&
      left.triangles == right.triangles &&
      left.wedges == right.wedges &&
      left.mixed_twice == right.mixed_twice;
}

bool verify_variants() {
  for (std::uint32_t vertex_count = 1;
       vertex_count <= 6;
       ++vertex_count) {
    const auto suffixes = suffix_masks(vertex_count);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> edges;
    for (std::uint32_t left = 0; left < vertex_count; ++left) {
      for (std::uint32_t right = left + 1;
           right < vertex_count;
           ++right) {
        edges.emplace_back(left, right);
      }
    }
    const auto graph_count = std::uint64_t{1} << edges.size();
    for (std::uint64_t graph = 0; graph < graph_count; ++graph) {
      std::array<std::uint64_t, 64> adjacency{};
      for (std::size_t edge = 0; edge < edges.size(); ++edge) {
        if (((graph >> edge) & 1) == 0) {
          continue;
        }
        const auto [left, right] = edges[edge];
        adjacency[left] |= std::uint64_t{1} << right;
        adjacency[right] |= std::uint64_t{1} << left;
      }
      const auto expected = invariants_production(
          adjacency.data(), vertex_count, suffixes);
      const std::array<Totals, 9> alternatives = {
          invariants_suffix_walk(
              adjacency.data(), vertex_count, suffixes),
          invariants_dynamic_clear_lowest(
              adjacency.data(), vertex_count, suffixes),
          invariants_precomputed_bit_clear(
              adjacency.data(), vertex_count, suffixes),
          invariants_common_walk(
              adjacency.data(), vertex_count, suffixes),
          invariants_suffix_scan(
              adjacency.data(), vertex_count, suffixes),
          invariants_common_scan(
              adjacency.data(), vertex_count, suffixes),
          invariants_adaptive(
              adjacency.data(), vertex_count, suffixes),
          invariants_min_edge_walk(
              adjacency.data(), vertex_count, suffixes),
          invariants_three_way(
              adjacency.data(), vertex_count, suffixes),
      };
      if (!std::all_of(
              alternatives.begin(),
              alternatives.end(),
              [&](const Totals& actual) {
                return same_totals(actual, expected);
              })) {
        return false;
      }
      if (!validate_pairwise(adjacency.data(), vertex_count) ||
          !validate_edges(adjacency.data(), vertex_count) ||
          !validate_transpose(adjacency.data(), vertex_count) ||
          !validate_swar_transpose(
              adjacency.data(), vertex_count)) {
        return false;
      }
    }
  }

  std::array<std::uint64_t, 64> invalid{};
  invalid[0] = std::uint64_t{1} << 1;
  if (validate_pairwise(invalid.data(), 8) ||
      validate_edges(invalid.data(), 8) ||
      validate_transpose(invalid.data(), 8) ||
      validate_swar_transpose(invalid.data(), 8)) {
    return false;
  }
  invalid = {};
  invalid[3] = std::uint64_t{1} << 3;
  if (validate_pairwise(invalid.data(), 8) ||
      validate_edges(invalid.data(), 8) ||
      validate_transpose(invalid.data(), 8) ||
      validate_swar_transpose(invalid.data(), 8)) {
    return false;
  }
  invalid = {};
  invalid[0] = std::uint64_t{1} << 8;
  return !validate_pairwise(invalid.data(), 8) &&
      !validate_edges(invalid.data(), 8) &&
      !validate_transpose(invalid.data(), 8) &&
      !validate_swar_transpose(invalid.data(), 8);
}

template <typename Function>
void report(
    std::string_view benchmark,
    const std::vector<std::uint64_t>& graphs,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    double density,
    std::size_t repeats,
    Function function) {
  std::uint64_t checksum = 0;
  const auto seconds = best_seconds(repeats, [&] {
    Totals total;
    for (std::size_t graph = 0; graph < graph_count; ++graph) {
      total += function(graphs.data() + graph * vertex_count);
    }
    checksum = total.degrees ^ total.edges ^
        total.triangles ^ total.wedges ^ total.mixed_twice;
    checksum_sink = checksum;
  });
  std::cout << std::setprecision(12)
            << "{\"benchmark\":\"" << benchmark
            << "\",\"vertex_count\":" << vertex_count
            << ",\"density\":" << density
            << ",\"graph_count\":" << graph_count
            << ",\"seconds\":" << seconds
            << ",\"graphs_per_second\":"
            << graph_count / seconds
            << ",\"checksum\":" << checksum
            << "}\n";
}

template <typename Function>
void report_validation(
    std::string_view benchmark,
    const std::vector<std::uint64_t>& graphs,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    double density,
    std::size_t repeats,
    Function function) {
  std::uint64_t valid_count = 0;
  const auto seconds = best_seconds(repeats, [&] {
    std::uint64_t local = 0;
    for (std::size_t graph = 0; graph < graph_count; ++graph) {
      local += function(graphs.data() + graph * vertex_count);
    }
    valid_count = local;
    checksum_sink = local;
  });
  std::cout << std::setprecision(12)
            << "{\"benchmark\":\"" << benchmark
            << "\",\"vertex_count\":" << vertex_count
            << ",\"density\":" << density
            << ",\"graph_count\":" << graph_count
            << ",\"seconds\":" << seconds
            << ",\"graphs_per_second\":"
            << graph_count / seconds
            << ",\"valid_count\":" << valid_count
            << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (!verify_variants()) {
    std::cerr << "graph invariant variant verification failed\n";
    return 2;
  }
  const auto graph_count = argc > 1
      ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10))
      : 20'000;
  const auto repeats = argc > 2
      ? static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10))
      : 5;
  for (const auto vertex_count :
       {8u, 16u, 24u, 32u, 43u, 48u, 64u}) {
    const auto suffixes = suffix_masks(vertex_count);
    for (const auto density :
         {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0}) {
      const auto graphs = random_graphs(
          graph_count,
          vertex_count,
          density,
          7601 + vertex_count * 101 +
              static_cast<std::uint64_t>(density * 1000));
      report_validation(
          "validate_pairwise",
          graphs,
          graph_count,
          vertex_count,
          density,
          repeats,
          [&](const std::uint64_t* adjacency) {
            return validate_pairwise(adjacency, vertex_count);
          });
      report_validation(
          "validate_edges",
          graphs,
          graph_count,
          vertex_count,
          density,
          repeats,
          [&](const std::uint64_t* adjacency) {
            return validate_edges(adjacency, vertex_count);
          });
      report_validation(
          "validate_transpose",
          graphs,
          graph_count,
          vertex_count,
          density,
          repeats,
          [&](const std::uint64_t* adjacency) {
            return validate_transpose(adjacency, vertex_count);
          });
      report_validation(
          "validate_swar_transpose",
          graphs,
          graph_count,
          vertex_count,
          density,
          repeats,
          [&](const std::uint64_t* adjacency) {
            return validate_swar_transpose(adjacency, vertex_count);
          });
      report(
          "invariants_production",
          graphs,
          graph_count,
          vertex_count,
          density,
          repeats,
          [&](const std::uint64_t* adjacency) {
            return invariants_production(
                adjacency, vertex_count, suffixes);
          });
      report(
          "invariants_suffix_walk",
          graphs,
          graph_count,
          vertex_count,
          density,
          repeats,
          [&](const std::uint64_t* adjacency) {
            return invariants_suffix_walk(
                adjacency, vertex_count, suffixes);
          });
      report(
          "invariants_dynamic_clear_lowest",
          graphs,
          graph_count,
          vertex_count,
          density,
          repeats,
          [&](const std::uint64_t* adjacency) {
            return invariants_dynamic_clear_lowest(
                adjacency, vertex_count, suffixes);
          });
      report(
          "invariants_precomputed_bit_clear",
          graphs,
          graph_count,
          vertex_count,
          density,
          repeats,
          [&](const std::uint64_t* adjacency) {
            return invariants_precomputed_bit_clear(
                adjacency, vertex_count, suffixes);
          });
      report(
          "invariants_common_walk",
          graphs,
          graph_count,
          vertex_count,
          density,
          repeats,
          [&](const std::uint64_t* adjacency) {
            return invariants_common_walk(
                adjacency, vertex_count, suffixes);
          });
      report(
          "invariants_suffix_scan",
          graphs,
          graph_count,
          vertex_count,
          density,
          repeats,
          [&](const std::uint64_t* adjacency) {
            return invariants_suffix_scan(
                adjacency, vertex_count, suffixes);
          });
      report(
          "invariants_common_scan",
          graphs,
          graph_count,
          vertex_count,
          density,
          repeats,
          [&](const std::uint64_t* adjacency) {
            return invariants_common_scan(
                adjacency, vertex_count, suffixes);
          });
      report(
          "invariants_adaptive",
          graphs,
          graph_count,
          vertex_count,
          density,
          repeats,
          [&](const std::uint64_t* adjacency) {
            return invariants_adaptive(
                adjacency, vertex_count, suffixes);
          });
      report(
          "invariants_min_edge_walk",
          graphs,
          graph_count,
          vertex_count,
          density,
          repeats,
          [&](const std::uint64_t* adjacency) {
            return invariants_min_edge_walk(
                adjacency, vertex_count, suffixes);
          });
      report(
          "invariants_three_way",
          graphs,
          graph_count,
          vertex_count,
          density,
          repeats,
          [&](const std::uint64_t* adjacency) {
            return invariants_three_way(
                adjacency, vertex_count, suffixes);
          });
    }
  }
  return checksum_sink == std::numeric_limits<std::uint64_t>::max();
}
