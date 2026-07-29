#include "fast_math.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
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
  const auto length = std::min(
      destination_size - 1,
      std::strlen(message));
  std::memcpy(destination, message, length);
  destination[length] = '\0';
}

std::size_t find_neighbor(
    const std::uint32_t* columns,
    std::size_t begin,
    std::size_t end,
    std::uint32_t target) {
  const auto* first = columns + begin;
  const auto* last = columns + end;
  const auto* found = std::lower_bound(first, last, target);
  if (found == last || *found != target) {
    return std::numeric_limits<std::size_t>::max();
  }
  return static_cast<std::size_t>(found - columns);
}

void validate_csr(
    const std::uint64_t* row_offsets,
    const std::uint32_t* column_indices,
    const std::uint64_t* edge_color_masks,
    std::size_t vertex_count,
    std::size_t directed_edge_count) {
  if (row_offsets == nullptr) {
    throw std::invalid_argument("large-graph row_offsets pointer is null");
  }
  if (directed_edge_count != 0 && column_indices == nullptr) {
    throw std::invalid_argument(
        "large-graph column_indices pointer is null");
  }
  if (vertex_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(
        "large-graph vertex count exceeds uint32 coordinates");
  }
  if (row_offsets[0] != 0 ||
      row_offsets[vertex_count] != directed_edge_count) {
    throw std::invalid_argument(
        "large-graph CSR offsets do not span the edge array");
  }
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    const auto begin = static_cast<std::size_t>(row_offsets[vertex]);
    const auto end = static_cast<std::size_t>(row_offsets[vertex + 1]);
    if (begin > end || end > directed_edge_count) {
      throw std::invalid_argument(
          "large-graph CSR offsets are not monotone");
    }
    std::uint32_t previous = 0;
    bool have_previous = false;
    for (std::size_t offset = begin; offset < end; ++offset) {
      const auto neighbor = column_indices[offset];
      if (neighbor >= vertex_count || neighbor == vertex) {
        throw std::invalid_argument(
            "large graph contains an invalid vertex or self-loop");
      }
      if (have_previous && neighbor <= previous) {
        throw std::invalid_argument(
            "large-graph CSR rows must be strictly increasing");
      }
      previous = neighbor;
      have_previous = true;
      const auto reverse = find_neighbor(
          column_indices,
          static_cast<std::size_t>(row_offsets[neighbor]),
          static_cast<std::size_t>(row_offsets[neighbor + 1]),
          static_cast<std::uint32_t>(vertex));
      if (reverse == std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(
            "large-graph CSR must describe an undirected graph");
      }
      if (edge_color_masks != nullptr &&
          edge_color_masks[reverse] != edge_color_masks[offset]) {
        throw std::invalid_argument(
            "large-graph edge colors must be symmetric");
      }
    }
  }
}

}  // namespace

extern "C" {

int fast_math_graph_common_neighbors_csr_u32(
    const std::uint64_t* row_offsets,
    const std::uint32_t* column_indices,
    std::size_t vertex_count,
    std::size_t directed_edge_count,
    const std::uint32_t* pairs,
    std::size_t pair_count,
    std::size_t common_neighbor_capacity,
    std::uint64_t* pair_offsets,
    std::uint32_t* common_neighbors,
    std::uint64_t* common_neighbor_count,
    fast_math_common_neighbor_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    validate_csr(
        row_offsets,
        column_indices,
        nullptr,
        vertex_count,
        directed_edge_count);
    if (pair_count != 0 && pairs == nullptr) {
      throw std::invalid_argument(
          "large-graph pair pointer is null");
    }
    if (pair_offsets == nullptr ||
        common_neighbor_count == nullptr ||
        stats == nullptr) {
      throw std::invalid_argument(
          "large-graph common-neighbor output pointer is null");
    }
    if (common_neighbor_capacity != 0 && common_neighbors == nullptr) {
      throw std::invalid_argument(
          "large-graph common-neighbor buffer is null");
    }

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    std::uint64_t count = 0;
    std::uint64_t intersection_steps = 0;
    pair_offsets[0] = 0;
    for (std::size_t pair_index = 0; pair_index < pair_count; ++pair_index) {
      const auto left = pairs[pair_index * 2];
      const auto right = pairs[pair_index * 2 + 1];
      if (left >= vertex_count || right >= vertex_count) {
        throw std::invalid_argument(
            "large-graph pair endpoint is outside the vertex range");
      }
      auto left_offset =
          static_cast<std::size_t>(row_offsets[left]);
      const auto left_end =
          static_cast<std::size_t>(row_offsets[left + 1]);
      auto right_offset =
          static_cast<std::size_t>(row_offsets[right]);
      const auto right_end =
          static_cast<std::size_t>(row_offsets[right + 1]);
      while (left_offset < left_end && right_offset < right_end) {
        ++intersection_steps;
        const auto left_neighbor = column_indices[left_offset];
        const auto right_neighbor = column_indices[right_offset];
        if (left_neighbor < right_neighbor) {
          ++left_offset;
          continue;
        }
        if (right_neighbor < left_neighbor) {
          ++right_offset;
          continue;
        }
        if (common_neighbors != nullptr &&
            count < common_neighbor_capacity) {
          common_neighbors[count] = left_neighbor;
        }
        ++count;
        ++left_offset;
        ++right_offset;
      }
      pair_offsets[pair_index + 1] = count;
    }

    *common_neighbor_count = count;
    stats->vertex_count = vertex_count;
    stats->directed_edge_count = directed_edge_count;
    stats->pair_count = pair_count;
    stats->intersection_steps = intersection_steps;
    stats->common_neighbor_count = count;
    stats->elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    if (common_neighbors != nullptr &&
        count > common_neighbor_capacity) {
      throw std::invalid_argument(
          "large-graph common-neighbor capacity is too small");
    }
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(error_message, error_message_size, "unknown native error");
    return 2;
  }
}

int fast_math_graph_triangles_csr_u32(
    const std::uint64_t* row_offsets,
    const std::uint32_t* column_indices,
    const std::uint64_t* edge_color_masks,
    const std::uint64_t* vertex_loop_color_masks,
    std::size_t vertex_count,
    std::size_t directed_edge_count,
    std::size_t triangle_capacity,
    std::uint32_t* triangles,
    std::uint64_t* triangle_edge_color_masks,
    std::uint64_t* triangle_count,
    fast_math_large_graph_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    validate_csr(
        row_offsets,
        column_indices,
        edge_color_masks,
        vertex_count,
        directed_edge_count);
    if (triangle_count == nullptr || stats == nullptr) {
      throw std::invalid_argument(
          "large-graph triangle output pointer is null");
    }
    if (triangle_capacity != 0 && triangles == nullptr) {
      throw std::invalid_argument(
          "large-graph triangle buffer is null");
    }
    if (triangle_edge_color_masks != nullptr &&
        edge_color_masks == nullptr) {
      throw std::invalid_argument(
          "triangle colors requested without edge colors");
    }

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    std::uint64_t count = 0;
    std::uint64_t intersection_steps = 0;
    const auto emit =
        [&](std::uint32_t left,
            std::uint32_t middle,
            std::uint32_t right,
            std::uint64_t left_middle_color,
            std::uint64_t left_right_color,
            std::uint64_t middle_right_color) {
      if (count < triangle_capacity) {
        const auto output = static_cast<std::size_t>(count) * 3;
        triangles[output] = left;
        triangles[output + 1] = middle;
        triangles[output + 2] = right;
        if (triangle_edge_color_masks != nullptr) {
          triangle_edge_color_masks[output] = left_middle_color;
          triangle_edge_color_masks[output + 1] = left_right_color;
          triangle_edge_color_masks[output + 2] = middle_right_color;
        }
      }
      ++count;
    };

    for (std::size_t left = 0; left < vertex_count; ++left) {
      const auto left_loop = vertex_loop_color_masks == nullptr
          ? 0
          : vertex_loop_color_masks[left];
      if (left_loop != 0) {
        emit(
            static_cast<std::uint32_t>(left),
            static_cast<std::uint32_t>(left),
            static_cast<std::uint32_t>(left),
            left_loop,
            left_loop,
            left_loop);
      }
      const auto left_begin =
          static_cast<std::size_t>(row_offsets[left]);
      const auto left_end =
          static_cast<std::size_t>(row_offsets[left + 1]);
      auto left_middle = std::lower_bound(
          column_indices + left_begin,
          column_indices + left_end,
          static_cast<std::uint32_t>(left + 1));
      if (left_loop != 0) {
        for (auto repeated_right = left_middle;
             repeated_right != column_indices + left_end;
             ++repeated_right) {
          const auto edge_offset = static_cast<std::size_t>(
              repeated_right - column_indices);
          const auto edge_color = edge_color_masks == nullptr
              ? 0
              : edge_color_masks[edge_offset];
          emit(
              static_cast<std::uint32_t>(left),
              static_cast<std::uint32_t>(left),
              *repeated_right,
              left_loop,
              edge_color,
              edge_color);
        }
      }
      for (; left_middle != column_indices + left_end; ++left_middle) {
        const auto middle = *left_middle;
        const auto left_middle_offset =
            static_cast<std::size_t>(left_middle - column_indices);
        const auto edge_color = edge_color_masks == nullptr
            ? 0
            : edge_color_masks[left_middle_offset];
        const auto middle_loop = vertex_loop_color_masks == nullptr
            ? 0
            : vertex_loop_color_masks[middle];
        if (middle_loop != 0) {
          emit(
              static_cast<std::uint32_t>(left),
              middle,
              middle,
              edge_color,
              edge_color,
              middle_loop);
        }
        auto left_right = left_middle + 1;
        const auto middle_begin =
            static_cast<std::size_t>(row_offsets[middle]);
        const auto middle_end =
            static_cast<std::size_t>(row_offsets[middle + 1]);
        auto middle_right = std::lower_bound(
            column_indices + middle_begin,
            column_indices + middle_end,
            middle + 1);

        while (left_right != column_indices + left_end &&
               middle_right != column_indices + middle_end) {
          ++intersection_steps;
          if (*left_right < *middle_right) {
            ++left_right;
            continue;
          }
          if (*middle_right < *left_right) {
            ++middle_right;
            continue;
          }
          emit(
              static_cast<std::uint32_t>(left),
              middle,
              *left_right,
              edge_color_masks == nullptr
                  ? 0
                  : edge_color_masks[left_middle_offset],
              edge_color_masks == nullptr
                  ? 0
                  : edge_color_masks[
                        static_cast<std::size_t>(
                            left_right - column_indices)],
              edge_color_masks == nullptr
                  ? 0
                  : edge_color_masks[
                        static_cast<std::size_t>(
                            middle_right - column_indices)]);
          ++left_right;
          ++middle_right;
        }
      }
    }

    *triangle_count = count;
    stats->vertex_count = vertex_count;
    stats->directed_edge_count = directed_edge_count;
    stats->intersection_steps = intersection_steps;
    stats->triangle_count = count;
    stats->elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    if (count > triangle_capacity && triangles != nullptr) {
      throw std::invalid_argument(
          "large-graph triangle capacity is too small");
    }
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
