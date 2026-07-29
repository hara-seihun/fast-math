#include "fast_math.h"

#include "parallel.hpp"

#include <nauty.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
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
  const auto length = std::min(
      destination_size - 1,
      std::strlen(message));
  std::memcpy(destination, message, length);
  destination[length] = '\0';
}

void validate_graph(
    const std::uint64_t* adjacency,
    std::uint32_t vertex_count,
    std::uint32_t word_count) {
  const auto expected_words =
      static_cast<std::uint32_t>((vertex_count + 63) / 64);
  if (word_count != expected_words) {
    throw std::invalid_argument(
        "canonical digraph word_count is inconsistent with vertex_count");
  }
  const auto final_bits = vertex_count % 64;
  const auto final_mask = final_bits == 0
      ? ~std::uint64_t{0}
      : (std::uint64_t{1} << final_bits) - 1;
  for (std::uint32_t vertex = 0; vertex < vertex_count; ++vertex) {
    const auto* row =
        adjacency + static_cast<std::size_t>(vertex) * word_count;
    if ((row[word_count - 1] & ~final_mask) != 0) {
      throw std::invalid_argument(
          "canonical digraph contains an out-of-range vertex");
    }
    if ((row[vertex / 64] & (std::uint64_t{1} << (vertex % 64))) != 0) {
      throw std::invalid_argument(
          "canonical digraph contains a self-loop");
    }
  }
}

thread_local std::vector<std::uint32_t>* active_generators = nullptr;

void collect_generator(
    int,
    int* permutation,
    int*,
    int,
    int,
    int vertex_count) {
  if (active_generators == nullptr) {
    return;
  }
  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    active_generators->push_back(
        static_cast<std::uint32_t>(permutation[vertex]));
  }
}

struct FixedGeneratorCollector {
  std::uint32_t* output = nullptr;
  std::size_t size = 0;
  std::size_t capacity = 0;
  bool overflow = false;
};

thread_local FixedGeneratorCollector* active_fixed_generators = nullptr;

void collect_generator_fixed(
    int,
    int* permutation,
    int*,
    int,
    int,
    int vertex_count) {
  if (active_fixed_generators == nullptr) {
    return;
  }
  const auto count = static_cast<std::size_t>(vertex_count);
  if (active_fixed_generators->size >
      active_fixed_generators->capacity - count) {
    active_fixed_generators->overflow = true;
    return;
  }
  for (std::size_t vertex = 0; vertex < count; ++vertex) {
    active_fixed_generators->output[
        active_fixed_generators->size + vertex] =
        static_cast<std::uint32_t>(permutation[vertex]);
  }
  active_fixed_generators->size += count;
}

struct NautyWorkspace {
  std::vector<graph> source;
  std::vector<graph> canonical;
  std::vector<int> lab;
  std::vector<int> partition;
  std::vector<int> orbits;
  std::vector<std::uint32_t> color_order;

  NautyWorkspace(std::uint32_t vertex_count, std::uint32_t nauty_words)
      : source(static_cast<std::size_t>(nauty_words) * vertex_count),
        canonical(source.size()),
        lab(vertex_count),
        partition(vertex_count),
        orbits(vertex_count),
        color_order(vertex_count) {}
};

}  // namespace

extern "C" {

int fast_math_canonical_digraphs_nauty_u64(
    const std::uint64_t* adjacency_words,
    const std::uint32_t* vertex_colors,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    std::uint32_t word_count,
    std::uint32_t* canonical_permutations,
    std::uint64_t* canonical_adjacency_words,
    std::uint32_t* canonical_vertex_colors,
    double* automorphism_group_mantissas,
    std::int32_t* automorphism_group_exponents,
    std::uint32_t* orbit_counts,
    std::uint64_t* generator_offsets,
    std::size_t generator_capacity,
    std::uint32_t* generator_permutations,
    std::uint64_t* generator_count,
    fast_math_canonical_graph_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (vertex_count == 0) {
      throw std::invalid_argument(
          "canonical digraphs require at least one vertex");
    }
    if (adjacency_words == nullptr || vertex_colors == nullptr ||
        canonical_permutations == nullptr ||
        canonical_adjacency_words == nullptr ||
        canonical_vertex_colors == nullptr ||
        automorphism_group_mantissas == nullptr ||
        automorphism_group_exponents == nullptr ||
        orbit_counts == nullptr || generator_offsets == nullptr ||
        generator_count == nullptr || stats == nullptr) {
      throw std::invalid_argument(
          "canonical digraph input or output pointer is null");
    }
    if (generator_capacity != 0 && generator_permutations == nullptr) {
      throw std::invalid_argument(
          "automorphism generator buffer is null");
    }
    static_assert(sizeof(setword) == sizeof(std::uint64_t));
    const auto nauty_words = SETWORDSNEEDED(vertex_count);
    nauty_check(WORDSIZE, nauty_words, vertex_count, NAUTYVERSIONID);
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    std::uint64_t search_nodes = 0;
    std::vector<std::uint32_t> generators;
    generator_offsets[0] = 0;

    std::vector<graph> source(
        static_cast<std::size_t>(nauty_words) * vertex_count);
    std::vector<graph> canonical(source.size());
    std::vector<int> lab(vertex_count);
    std::vector<int> partition(vertex_count);
    std::vector<int> orbits(vertex_count);
    std::vector<std::uint32_t> color_order(vertex_count);

    for (std::size_t graph_index = 0;
         graph_index < graph_count;
         ++graph_index) {
      const auto* input =
          adjacency_words +
          graph_index * vertex_count * word_count;
      const auto* colors =
          vertex_colors + graph_index * vertex_count;
      validate_graph(input, vertex_count, word_count);
      EMPTYGRAPH(source.data(), nauty_words, vertex_count);
      for (std::uint32_t vertex = 0;
           vertex < vertex_count;
           ++vertex) {
        const auto* row =
            input + static_cast<std::size_t>(vertex) * word_count;
        for (std::uint32_t word = 0; word < word_count; ++word) {
          auto active = row[word];
          while (active != 0) {
            const auto bit_index = static_cast<std::uint32_t>(
                std::countr_zero(active));
            const auto neighbor = word * 64 + bit_index;
            ADDONEARC(
                source.data(),
                static_cast<int>(vertex),
                static_cast<int>(neighbor),
                nauty_words);
            active &= active - 1;
          }
        }
      }

      std::iota(color_order.begin(), color_order.end(), 0);
      std::stable_sort(
          color_order.begin(),
          color_order.end(),
          [&](std::uint32_t left, std::uint32_t right) {
            return colors[left] < colors[right];
          });
      for (std::uint32_t position = 0;
           position < vertex_count;
           ++position) {
        lab[position] = static_cast<int>(color_order[position]);
        partition[position] =
            position + 1 < vertex_count &&
                    colors[color_order[position]] ==
                        colors[color_order[position + 1]]
                ? 1
                : 0;
      }

      DEFAULTOPTIONS_GRAPH(options);
      options.getcanon = TRUE;
      options.digraph = TRUE;
      options.defaultptn = FALSE;
      options.writeautoms = FALSE;
      options.outfile = nullptr;
      options.userautomproc = collect_generator;
      statsblk nauty_stats{};
      active_generators = &generators;
      densenauty(
          source.data(),
          lab.data(),
          partition.data(),
          orbits.data(),
          &options,
          &nauty_stats,
          nauty_words,
          vertex_count,
          canonical.data());
      active_generators = nullptr;
      if (nauty_stats.errstatus != 0) {
        throw std::runtime_error("nauty canonicalization failed");
      }
      search_nodes += nauty_stats.numnodes;

      auto* output_permutation =
          canonical_permutations + graph_index * vertex_count;
      auto* output_colors =
          canonical_vertex_colors + graph_index * vertex_count;
      auto* output_adjacency =
          canonical_adjacency_words +
          graph_index * vertex_count * word_count;
      std::fill_n(
          output_adjacency,
          static_cast<std::size_t>(vertex_count) * word_count,
          0);
      for (std::uint32_t position = 0;
           position < vertex_count;
           ++position) {
        output_permutation[position] =
            static_cast<std::uint32_t>(lab[position]);
        output_colors[position] = colors[lab[position]];
        const auto* row = GRAPHROW(
            canonical.data(),
            static_cast<int>(position),
            nauty_words);
        for (std::uint32_t neighbor = 0;
             neighbor < vertex_count;
             ++neighbor) {
          if (ISELEMENT(row, neighbor)) {
            output_adjacency[
                static_cast<std::size_t>(position) * word_count +
                neighbor / 64] |=
                std::uint64_t{1} << (neighbor % 64);
          }
        }
      }
      automorphism_group_mantissas[graph_index] =
          nauty_stats.grpsize1;
      automorphism_group_exponents[graph_index] =
          nauty_stats.grpsize2;
      orbit_counts[graph_index] =
          static_cast<std::uint32_t>(nauty_stats.numorbits);
      generator_offsets[graph_index + 1] =
          generators.size() / vertex_count;
    }

    const auto total_generators = generators.size() / vertex_count;
    *generator_count = total_generators;
    if (generator_permutations != nullptr) {
      if (generator_capacity < total_generators) {
        throw std::invalid_argument(
            "automorphism generator capacity is too small");
      }
      std::copy(
          generators.begin(),
          generators.end(),
          generator_permutations);
    }
    stats->graph_count = graph_count;
    stats->vertex_count = vertex_count;
    stats->word_count = word_count;
    stats->search_nodes = search_nodes;
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

int fast_math_canonical_digraphs_nauty_v2_u64(
    const std::uint64_t* adjacency_words,
    const std::uint32_t* vertex_colors,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    std::uint32_t word_count,
    std::uint32_t thread_count,
    std::uint8_t collect_automorphism_generators,
    std::uint32_t* canonical_permutations,
    std::uint64_t* canonical_adjacency_words,
    std::uint32_t* canonical_vertex_colors,
    double* automorphism_group_mantissas,
    std::int32_t* automorphism_group_exponents,
    std::uint32_t* orbit_counts,
    std::uint64_t* generator_offsets,
    std::size_t generator_capacity,
    std::uint32_t* generator_permutations,
    std::uint64_t* generator_count,
    fast_math_canonical_graph_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (vertex_count == 0) {
      throw std::invalid_argument(
          "canonical digraphs require at least one vertex");
    }
    if (collect_automorphism_generators > 1) {
      throw std::invalid_argument(
          "collect_automorphism_generators must be zero or one");
    }
    if (adjacency_words == nullptr || vertex_colors == nullptr ||
        canonical_permutations == nullptr ||
        canonical_adjacency_words == nullptr ||
        canonical_vertex_colors == nullptr ||
        automorphism_group_mantissas == nullptr ||
        automorphism_group_exponents == nullptr ||
        orbit_counts == nullptr || generator_offsets == nullptr ||
        generator_count == nullptr || stats == nullptr) {
      throw std::invalid_argument(
          "canonical digraph input or output pointer is null");
    }
    const auto collect = collect_automorphism_generators != 0;
    if (collect) {
      if (graph_count >
          std::numeric_limits<std::size_t>::max() / vertex_count) {
        throw std::invalid_argument(
            "automorphism generator capacity overflows size_t");
      }
      const auto required_capacity = graph_count * vertex_count;
      if (generator_capacity < required_capacity ||
          (required_capacity != 0 &&
           generator_permutations == nullptr)) {
        throw std::invalid_argument(
            "one-pass automorphism generator capacity is too small");
      }
    }
    static_assert(sizeof(setword) == sizeof(std::uint64_t));
    const auto nauty_words = SETWORDSNEEDED(vertex_count);
    nauty_check(WORDSIZE, nauty_words, vertex_count, NAUTYVERSIONID);
    for (std::size_t graph_index = 0;
         graph_index < graph_count;
         ++graph_index) {
      validate_graph(
          adjacency_words +
              graph_index * vertex_count * word_count,
          vertex_count,
          word_count);
    }

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
#if defined(HAVE_TLS) && HAVE_TLS
    const auto nauty_thread_count = thread_count;
#else
    const std::uint32_t nauty_thread_count = 1;
#endif
    const auto workers =
        fast_math_internal::parallel_worker_count(
            graph_count, nauty_thread_count);
    std::vector<NautyWorkspace> workspaces;
    workspaces.reserve(workers);
    for (std::uint32_t worker = 0; worker < workers; ++worker) {
      workspaces.emplace_back(vertex_count, nauty_words);
    }
    std::vector<std::uint64_t> graph_search_nodes(graph_count, 0);
    std::vector<std::uint32_t> graph_generator_counts(graph_count, 0);
    std::vector<std::uint8_t> graph_errors(graph_count, 0);

    fast_math_internal::parallel_for_dynamic_indexed(
        graph_count,
        nauty_thread_count,
        [&](std::size_t graph_index, std::size_t worker_index) {
          auto& workspace = workspaces[worker_index];
          const auto* input =
              adjacency_words +
              graph_index * vertex_count * word_count;
          const auto* colors =
              vertex_colors + graph_index * vertex_count;
          EMPTYGRAPH(
              workspace.source.data(),
              nauty_words,
              vertex_count);
          for (std::uint32_t vertex = 0;
               vertex < vertex_count;
               ++vertex) {
            const auto* row =
                input + static_cast<std::size_t>(vertex) * word_count;
            for (std::uint32_t word = 0; word < word_count; ++word) {
              auto active = row[word];
              while (active != 0) {
                const auto bit_index = static_cast<std::uint32_t>(
                    std::countr_zero(active));
                const auto neighbor = word * 64 + bit_index;
                ADDONEARC(
                    workspace.source.data(),
                    static_cast<int>(vertex),
                    static_cast<int>(neighbor),
                    nauty_words);
                active &= active - 1;
              }
            }
          }

          std::iota(
              workspace.color_order.begin(),
              workspace.color_order.end(),
              0);
          std::sort(
              workspace.color_order.begin(),
              workspace.color_order.end(),
              [&](std::uint32_t left, std::uint32_t right) {
                if (colors[left] != colors[right]) {
                  return colors[left] < colors[right];
                }
                return left < right;
              });
          for (std::uint32_t position = 0;
               position < vertex_count;
               ++position) {
            workspace.lab[position] = static_cast<int>(
                workspace.color_order[position]);
            workspace.partition[position] =
                position + 1 < vertex_count &&
                        colors[workspace.color_order[position]] ==
                            colors[workspace.color_order[position + 1]]
                    ? 1
                    : 0;
          }

          DEFAULTOPTIONS_GRAPH(options);
          options.getcanon = TRUE;
          options.digraph = TRUE;
          options.defaultptn = FALSE;
          options.writeautoms = FALSE;
          options.outfile = nullptr;
          FixedGeneratorCollector collector;
          if (collect) {
            const auto block_elements =
                static_cast<std::size_t>(vertex_count) * vertex_count;
            collector.output =
                generator_permutations +
                graph_index * block_elements;
            collector.capacity = block_elements;
            options.userautomproc = collect_generator_fixed;
            active_fixed_generators = &collector;
          } else {
            options.userautomproc = nullptr;
            active_fixed_generators = nullptr;
          }
          statsblk nauty_stats{};
          densenauty(
              workspace.source.data(),
              workspace.lab.data(),
              workspace.partition.data(),
              workspace.orbits.data(),
              &options,
              &nauty_stats,
              nauty_words,
              vertex_count,
              workspace.canonical.data());
          active_fixed_generators = nullptr;
          if (nauty_stats.errstatus != 0 || collector.overflow) {
            graph_errors[graph_index] =
                collector.overflow ? 2 : 1;
            return;
          }
          graph_search_nodes[graph_index] = nauty_stats.numnodes;
          graph_generator_counts[graph_index] =
              static_cast<std::uint32_t>(
                  collector.size / vertex_count);

          auto* output_permutation =
              canonical_permutations + graph_index * vertex_count;
          auto* output_colors =
              canonical_vertex_colors + graph_index * vertex_count;
          auto* output_adjacency =
              canonical_adjacency_words +
              graph_index * vertex_count * word_count;
          std::fill_n(
              output_adjacency,
              static_cast<std::size_t>(vertex_count) * word_count,
              0);
          for (std::uint32_t position = 0;
               position < vertex_count;
               ++position) {
            output_permutation[position] =
                static_cast<std::uint32_t>(workspace.lab[position]);
            output_colors[position] =
                colors[workspace.lab[position]];
            const auto* row = GRAPHROW(
                workspace.canonical.data(),
                static_cast<int>(position),
                nauty_words);
            for (std::uint32_t neighbor = 0;
                 neighbor < vertex_count;
                 ++neighbor) {
              if (ISELEMENT(row, neighbor)) {
                output_adjacency[
                    static_cast<std::size_t>(position) * word_count +
                    neighbor / 64] |=
                    std::uint64_t{1} << (neighbor % 64);
              }
            }
          }
          automorphism_group_mantissas[graph_index] =
              nauty_stats.grpsize1;
          automorphism_group_exponents[graph_index] =
              nauty_stats.grpsize2;
          orbit_counts[graph_index] =
              static_cast<std::uint32_t>(nauty_stats.numorbits);
        });

    for (const auto error : graph_errors) {
      if (error == 1) {
        throw std::runtime_error("nauty canonicalization failed");
      }
      if (error == 2) {
        throw std::runtime_error(
            "nauty exceeded the one-pass per-graph generator bound");
      }
    }
    std::uint64_t total_generators = 0;
    generator_offsets[0] = 0;
    if (collect) {
      const auto block_elements =
          static_cast<std::size_t>(vertex_count) * vertex_count;
      for (std::size_t graph_index = 0;
           graph_index < graph_count;
           ++graph_index) {
        const auto count = graph_generator_counts[graph_index];
        std::memmove(
            generator_permutations +
                total_generators * vertex_count,
            generator_permutations +
                graph_index * block_elements,
            static_cast<std::size_t>(count) *
                vertex_count * sizeof(std::uint32_t));
        total_generators += count;
        generator_offsets[graph_index + 1] = total_generators;
      }
    } else {
      std::fill_n(generator_offsets, graph_count + 1, 0);
    }
    *generator_count = total_generators;
    stats->graph_count = graph_count;
    stats->vertex_count = vertex_count;
    stats->word_count = word_count;
    stats->search_nodes = std::accumulate(
        graph_search_nodes.begin(),
        graph_search_nodes.end(),
        std::uint64_t{0});
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
