#include "parallel.hpp"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kRequestedWorkers = 8;

double process_cpu_seconds() {
  return static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
}

bool verify_dispatch(std::size_t count) {
  std::vector<std::uint64_t> values(count, 0);
  fast_math_internal::parallel_for_static(
      count, kRequestedWorkers, [&](std::size_t index) {
        values[index] = static_cast<std::uint64_t>(index + 1);
      });
  for (std::size_t index = 0; index < count; ++index) {
    if (values[index] != index + 1) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  constexpr std::size_t task_count = 8192;
  constexpr auto idle_window = std::chrono::milliseconds(250);
  constexpr double maximum_idle_utilization = 0.10;
  if (!verify_dispatch(task_count)) {
    std::cerr << "initial ForkUnion dispatch failed\n";
    return 1;
  }

  const auto wall_started = std::chrono::steady_clock::now();
  const auto cpu_started = process_cpu_seconds();
  std::this_thread::sleep_for(idle_window);
  const auto idle_cpu_seconds = process_cpu_seconds() - cpu_started;
  const auto idle_wall_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wall_started).count();
  // std::clock is process-wide, so budget CPU in per-worker terms. The old
  // fixed 0.1-second threshold became stricter as worker count grew and
  // failed when periodic wake-up overhead crossed that absolute value.
  const auto idle_utilization = idle_cpu_seconds /
      (idle_wall_seconds * kRequestedWorkers);
  if (idle_utilization > maximum_idle_utilization) {
    std::cerr << "idle ForkUnion workers used "
              << 100.0 * idle_utilization
              << "% of requested worker capacity (" << idle_cpu_seconds
              << " process CPU seconds over " << idle_wall_seconds
              << " wall seconds)\n";
    return 1;
  }

  if (!verify_dispatch(task_count)) {
    std::cerr << "ForkUnion wake dispatch failed\n";
    return 1;
  }
  return 0;
}
