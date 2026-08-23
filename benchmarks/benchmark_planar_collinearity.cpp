#include "fast_math.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Wide = __int128_t;

struct Point {
  std::int32_t x;
  std::int32_t y;
};

bool collinear(const Point& first, const Point& second, const Point& third) {
  const auto dx1 = static_cast<Wide>(second.x) - first.x;
  const auto dy1 = static_cast<Wide>(second.y) - first.y;
  const auto dx2 = static_cast<Wide>(third.x) - first.x;
  const auto dy2 = static_cast<Wide>(third.y) - first.y;
  return dx1 * dy2 == dy1 * dx2;
}

std::uint64_t key(const Point& point) {
  return (static_cast<std::uint64_t>(
              static_cast<std::uint32_t>(point.x))
          << 32) |
      static_cast<std::uint32_t>(point.y);
}

// The complete triple audit repeated after each y-swap in
// repair_no_three_fast.cpp::audit.
void retained_full_rescore(
    const std::vector<Point>& base,
    const std::vector<std::uint32_t>& deleted,
    const std::vector<Point>& added,
    std::vector<std::uint64_t>& scores) {
  auto points = base;
  for (std::size_t edit = 0; edit < scores.size(); ++edit) {
    const auto first = deleted[2 * edit];
    const auto second = deleted[2 * edit + 1];
    points[first] = added[2 * edit];
    points[second] = added[2 * edit + 1];
    std::uint64_t score = 0;
    for (std::size_t i = 0; i < points.size(); ++i) {
      for (std::size_t j = i + 1; j < points.size(); ++j) {
        for (std::size_t k = j + 1; k < points.size(); ++k) {
          score += collinear(points[i], points[j], points[k]);
        }
      }
    }
    scores[edit] = score;
    points[first] = base[first];
    points[second] = base[second];
  }
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2;
  if (values.size() % 2 != 0) {
    return values[middle];
  }
  return (values[middle - 1] + values[middle]) / 2;
}

struct Fixture {
  std::vector<Point> base;
  std::vector<std::uint32_t> deleted;
  std::vector<std::uint64_t> delete_offsets;
  std::vector<Point> added;
  std::vector<std::uint64_t> add_offsets;
};

Fixture make_fixture(std::size_t edit_count) {
  Fixture fixture;
  fixture.base.reserve(122);
  for (std::int32_t x = 0; x < 61; ++x) {
    fixture.base.push_back({x, x});
    fixture.base.push_back({x, (x + 1) % 61});
  }
  std::unordered_set<std::uint64_t> occupied;
  for (const auto& point : fixture.base) {
    occupied.insert(key(point));
  }
  fixture.delete_offsets.push_back(0);
  fixture.add_offsets.push_back(0);
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  while (fixture.delete_offsets.size() <= edit_count) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    const auto first = static_cast<std::uint32_t>(state % fixture.base.size());
    state *= 0x2545f4914f6cdd1dULL;
    const auto second = static_cast<std::uint32_t>(state % fixture.base.size());
    if (first == second || fixture.base[first].x == fixture.base[second].x) {
      continue;
    }
    const Point replacement_first{
        fixture.base[first].x, fixture.base[second].y};
    const Point replacement_second{
        fixture.base[second].x, fixture.base[first].y};
    if (replacement_first.x == replacement_second.x &&
        replacement_first.y == replacement_second.y) {
      continue;
    }
    const auto first_key = key(replacement_first);
    const auto second_key = key(replacement_second);
    if ((occupied.count(first_key) != 0 &&
         first_key != key(fixture.base[first]) &&
         first_key != key(fixture.base[second])) ||
        (occupied.count(second_key) != 0 &&
         second_key != key(fixture.base[first]) &&
         second_key != key(fixture.base[second]))) {
      continue;
    }
    fixture.deleted.push_back(first);
    fixture.deleted.push_back(second);
    fixture.added.push_back(replacement_first);
    fixture.added.push_back(replacement_second);
    fixture.delete_offsets.push_back(fixture.deleted.size());
    fixture.add_offsets.push_back(fixture.added.size());
  }
  return fixture;
}

}  // namespace

int main(int argc, char** argv) {
  const auto edit_count = argc > 1
      ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10))
      : 2000;
  const auto repetitions = argc > 2 ? std::atoi(argv[2]) : 5;
  const auto threads = argc > 3
      ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10))
      : 0;
  if (edit_count == 0 || repetitions < 3) {
    std::cerr << "usage: benchmark_planar_collinearity [edits>0] "
                 "[repetitions>=3] [threads]\n";
    return 2;
  }
  const auto fixture = make_fixture(edit_count);
  std::vector<std::int32_t> base_interleaved;
  std::vector<std::int32_t> added_interleaved;
  base_interleaved.reserve(2 * fixture.base.size());
  added_interleaved.reserve(2 * fixture.added.size());
  for (const auto& point : fixture.base) {
    base_interleaved.push_back(point.x);
    base_interleaved.push_back(point.y);
  }
  for (const auto& point : fixture.added) {
    added_interleaved.push_back(point.x);
    added_interleaved.push_back(point.y);
  }
  std::vector<std::uint64_t> baseline_scores(edit_count);
  std::vector<std::uint64_t> native_scores(edit_count);
  std::vector<std::int64_t> native_deltas(edit_count);
  std::vector<std::uint8_t> cutoff_reached(edit_count);
  std::vector<std::uint64_t> degrees(fixture.base.size());
  std::uint64_t base_score = 0;
  fast_math_planar_collinearity_stats stats{};
  char error[1024]{};
  auto run_native = [&]() {
    const auto status = fast_math_planar_collinearity_edits_i32(
        base_interleaved.data(), fixture.base.size(), fixture.deleted.data(),
        fixture.deleted.size(), fixture.delete_offsets.data(),
        added_interleaved.data(), fixture.added.size(),
        fixture.add_offsets.data(), edit_count, 0, threads,
        &base_score, degrees.data(), native_scores.data(), native_deltas.data(),
        cutoff_reached.data(), &stats, error, sizeof(error));
    if (status != 0) {
      throw std::runtime_error(error);
    }
  };

  retained_full_rescore(
      fixture.base, fixture.deleted, fixture.added, baseline_scores);
  run_native();
  if (baseline_scores != native_scores) {
    std::cerr << "complete score mismatch\n";
    return 3;
  }

  std::vector<double> baseline_seconds;
  std::vector<double> native_seconds;
  auto time_call = [](auto&& callable) {
    const auto started = Clock::now();
    callable();
    return std::chrono::duration<double>(Clock::now() - started).count();
  };
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    if (repetition % 2 == 0) {
      baseline_seconds.push_back(time_call([&]() {
        retained_full_rescore(
            fixture.base, fixture.deleted, fixture.added, baseline_scores);
      }));
      native_seconds.push_back(time_call(run_native));
    } else {
      native_seconds.push_back(time_call(run_native));
      baseline_seconds.push_back(time_call([&]() {
        retained_full_rescore(
            fixture.base, fixture.deleted, fixture.added, baseline_scores);
      }));
    }
  }
  if (baseline_scores != native_scores) {
    std::cerr << "timed score mismatch\n";
    return 4;
  }
  const auto baseline_median = median(baseline_seconds);
  const auto native_median = median(native_seconds);
  std::uint64_t checksum = 0;
  for (const auto score : native_scores) {
    checksum += score;
  }
  std::cout << "base_points=" << fixture.base.size()
            << " edits=" << edit_count << " repetitions=" << repetitions
            << " workers=" << stats.worker_count << '\n';
  std::cout << "retained_full_rescore_seconds=" << baseline_median << '\n';
  std::cout << "native_seconds=" << native_median << '\n';
  std::cout << "speedup=" << baseline_median / native_median << '\n';
  std::cout << "score_checksum=" << checksum
            << " complete_output_parity=true\n";
}
