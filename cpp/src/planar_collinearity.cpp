#include "fast_math.h"
#include "parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Wide = __int128_t;

struct Point {
  std::int32_t x;
  std::int32_t y;
};

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

bool collinear(const Point& first, const Point& second, const Point& third) {
  const auto dx1 = static_cast<Wide>(second.x) - first.x;
  const auto dy1 = static_cast<Wide>(second.y) - first.y;
  const auto dx2 = static_cast<Wide>(third.x) - first.x;
  const auto dy2 = static_cast<Wide>(third.y) - first.y;
  return dx1 * dy2 == dy1 * dx2;
}

std::uint64_t point_key(const Point& point) {
  return (static_cast<std::uint64_t>(
              static_cast<std::uint32_t>(point.x))
          << 32) |
      static_cast<std::uint32_t>(point.y);
}

std::uint64_t choose3(std::uint64_t value) {
  return value < 3 ? 0 : value * (value - 1) * (value - 2) / 6;
}

void validate_offsets(
    const std::uint64_t* offsets,
    std::size_t edit_count,
    std::size_t item_count,
    const char* name) {
  if (offsets == nullptr) {
    throw std::invalid_argument(std::string(name) + " pointer is null");
  }
  if (offsets[0] != 0 || offsets[edit_count] != item_count) {
    throw std::invalid_argument(
        std::string(name) + " must start at zero and end at the item count");
  }
  for (std::size_t index = 0; index < edit_count; ++index) {
    if (offsets[index] > offsets[index + 1]) {
      throw std::invalid_argument(
          std::string(name) + " must be nondecreasing");
    }
  }
}

}  // namespace

extern "C" {

int fast_math_planar_collinearity_edits_i32(
    const std::int32_t* base_points_interleaved,
    std::size_t base_point_count,
    const std::uint32_t* delete_indices,
    std::size_t delete_index_count,
    const std::uint64_t* delete_offsets,
    const std::int32_t* added_points_interleaved,
    std::size_t added_point_count,
    const std::uint64_t* add_offsets,
    std::size_t edit_count,
    std::uint64_t score_cutoff,
    std::uint32_t thread_count,
    std::uint64_t* base_score,
    std::uint64_t* point_degrees,
    std::uint64_t* edit_scores,
    std::int64_t* edit_deltas,
    std::uint8_t* cutoff_reached,
    fast_math_planar_collinearity_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("planar collinearity stats pointer is null");
    }
    *stats = {};
    stats->base_point_count = base_point_count;
    stats->edit_count = edit_count;
    if (base_point_count > 512) {
      throw std::invalid_argument(
          "planar collinearity supports at most 512 base points");
    }
    if (thread_count > 1024) {
      throw std::invalid_argument(
          "planar collinearity thread count must be at most 1024");
    }
    if (added_point_count > std::numeric_limits<std::size_t>::max() / 2) {
      throw std::invalid_argument(
          "planar collinearity added-point storage overflows size_t");
    }
    if (base_score == nullptr) {
      throw std::invalid_argument("planar collinearity base score pointer is null");
    }
    if (base_point_count != 0 &&
        (base_points_interleaved == nullptr || point_degrees == nullptr)) {
      throw std::invalid_argument(
          "planar collinearity base input or degree output pointer is null");
    }
    if (delete_index_count != 0 && delete_indices == nullptr) {
      throw std::invalid_argument("planar collinearity delete pointer is null");
    }
    if (added_point_count != 0 && added_points_interleaved == nullptr) {
      throw std::invalid_argument("planar collinearity added-point pointer is null");
    }
    if (edit_count != 0 &&
        (edit_scores == nullptr || edit_deltas == nullptr ||
         cutoff_reached == nullptr)) {
      throw std::invalid_argument("planar collinearity edit output pointer is null");
    }
    validate_offsets(
        delete_offsets, edit_count, delete_index_count, "delete_offsets");
    validate_offsets(add_offsets, edit_count, added_point_count, "add_offsets");

    std::vector<Point> base_points(base_point_count);
    std::unordered_map<std::uint64_t, std::uint32_t> base_index;
    base_index.reserve(base_point_count * 2 + 1);
    for (std::size_t index = 0; index < base_point_count; ++index) {
      const Point point{
          base_points_interleaved[2 * index],
          base_points_interleaved[2 * index + 1]};
      if (!base_index.emplace(point_key(point), index).second) {
        throw std::invalid_argument("planar collinearity base points are not unique");
      }
      base_points[index] = point;
    }
    std::vector<Point> added_points(added_point_count);
    for (std::size_t index = 0; index < added_point_count; ++index) {
      added_points[index] = Point{
          added_points_interleaved[2 * index],
          added_points_interleaved[2 * index + 1]};
    }

    std::vector<std::uint8_t> validation_deleted(base_point_count);
    for (std::size_t edit = 0; edit < edit_count; ++edit) {
      std::fill(validation_deleted.begin(), validation_deleted.end(), 0);
      for (auto at = delete_offsets[edit]; at < delete_offsets[edit + 1]; ++at) {
        const auto index = delete_indices[at];
        if (index >= base_point_count) {
          throw std::invalid_argument(
              "planar collinearity delete index is out of range");
        }
        if (validation_deleted[index] != 0) {
          throw std::invalid_argument(
              "planar collinearity delete indices repeat within an edit");
        }
        validation_deleted[index] = 1;
      }
      const auto added_begin = add_offsets[edit];
      const auto added_end = add_offsets[edit + 1];
      if (added_end - added_begin > 64) {
        throw std::invalid_argument(
            "planar collinearity supports at most 64 additions per edit");
      }
      std::unordered_set<std::uint64_t> seen_added;
      seen_added.reserve((added_end - added_begin) * 2 + 1);
      for (auto at = added_begin; at < added_end; ++at) {
        const auto key = point_key(added_points[at]);
        if (!seen_added.insert(key).second) {
          throw std::invalid_argument(
              "planar collinearity added points repeat within an edit");
        }
        const auto found = base_index.find(key);
        if (found != base_index.end() && validation_deleted[found->second] == 0) {
          throw std::invalid_argument(
              "planar collinearity edit produces duplicate points");
        }
      }
    }

    const auto started = Clock::now();
    if (base_point_count != 0) {
      std::fill(
          point_degrees,
          point_degrees + base_point_count,
          std::uint64_t{0});
    }
    // Three nine-bit indices keep the all-collinear 512-point boundary below
    // 90 MB instead of storing a padded triple object for every conflict.
    std::vector<std::uint32_t> base_conflicts;
    std::uint64_t score = 0;
    for (std::uint32_t first = 0; first < base_point_count; ++first) {
      for (std::uint32_t second = first + 1; second < base_point_count; ++second) {
        for (std::uint32_t third = second + 1; third < base_point_count; ++third) {
          if (!collinear(
                  base_points[first], base_points[second], base_points[third])) {
            continue;
          }
          ++score;
          ++point_degrees[first];
          ++point_degrees[second];
          ++point_degrees[third];
          base_conflicts.push_back(first | (second << 9) | (third << 18));
        }
      }
    }
    *base_score = score;
    stats->base_score = score;
    stats->base_determinant_evaluations = choose3(base_point_count);

    if (edit_count != 0) {
      const auto requested_workers =
          fast_math_internal::parallel_worker_count(edit_count, thread_count);
      const auto use_parallel = edit_count >= 4 && requested_workers > 1;
      const auto workers = use_parallel ? requested_workers : 1;
      stats->worker_count = workers;
      std::vector<std::vector<std::uint8_t>> deleted_by_worker(
          workers, std::vector<std::uint8_t>(base_point_count));
      std::vector<std::uint64_t> determinant_evaluations(edit_count);
      auto process_edit = [&](std::size_t edit, std::size_t worker) {
        auto& deleted = deleted_by_worker[worker];
        std::fill(deleted.begin(), deleted.end(), 0);
        for (auto at = delete_offsets[edit]; at < delete_offsets[edit + 1]; ++at) {
          deleted[delete_indices[at]] = 1;
        }
        std::uint64_t edited_score = 0;
        bool reached = false;
        auto record_conflict = [&]() {
          ++edited_score;
          if (score_cutoff != 0 && edited_score >= score_cutoff) {
            edited_score = score_cutoff;
            reached = true;
          }
        };
        for (const auto conflict : base_conflicts) {
          const auto first = conflict & 511;
          const auto second = (conflict >> 9) & 511;
          const auto third = conflict >> 18;
          if (deleted[first] == 0 && deleted[second] == 0 &&
              deleted[third] == 0) {
            record_conflict();
            if (reached) {
              break;
            }
          }
        }
        const auto added_begin = add_offsets[edit];
        const auto added_end = add_offsets[edit + 1];
        for (auto added = added_begin; added < added_end && !reached; ++added) {
          for (std::uint32_t first = 0;
               first < base_point_count && !reached;
               ++first) {
            if (deleted[first] != 0) {
              continue;
            }
            for (std::uint32_t second = first + 1;
                 second < base_point_count && !reached;
                 ++second) {
              if (deleted[second] != 0) {
                continue;
              }
              ++determinant_evaluations[edit];
              if (collinear(
                      added_points[added],
                      base_points[first],
                      base_points[second])) {
                record_conflict();
              }
            }
          }
        }
        for (auto first = added_begin; first < added_end && !reached; ++first) {
          for (auto second = first + 1;
               second < added_end && !reached;
               ++second) {
            for (std::uint32_t base = 0;
                 base < base_point_count && !reached;
                 ++base) {
              if (deleted[base] != 0) {
                continue;
              }
              ++determinant_evaluations[edit];
              if (collinear(
                      added_points[first],
                      added_points[second],
                      base_points[base])) {
                record_conflict();
              }
            }
          }
        }
        for (auto first = added_begin; first < added_end && !reached; ++first) {
          for (auto second = first + 1;
               second < added_end && !reached;
               ++second) {
            for (auto third = second + 1;
                 third < added_end && !reached;
                 ++third) {
              ++determinant_evaluations[edit];
              if (collinear(
                      added_points[first],
                      added_points[second],
                      added_points[third])) {
                record_conflict();
              }
            }
          }
        }
        edit_scores[edit] = edited_score;
        edit_deltas[edit] = static_cast<std::int64_t>(edited_score) -
            static_cast<std::int64_t>(score);
        cutoff_reached[edit] = reached ? 1 : 0;
      };
      if (use_parallel) {
        fast_math_internal::parallel_for_static_indexed(
            edit_count, requested_workers, process_edit);
      } else {
        for (std::size_t edit = 0; edit < edit_count; ++edit) {
          process_edit(edit, 0);
        }
      }
      for (const auto evaluations : determinant_evaluations) {
        if (stats->edit_determinant_evaluations >
            std::numeric_limits<std::uint64_t>::max() - evaluations) {
          throw std::overflow_error(
              "planar collinearity determinant count overflows uint64");
        }
        stats->edit_determinant_evaluations += evaluations;
      }
    }
    stats->elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    set_error(error_message, error_message_size, "");
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
