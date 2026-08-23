#include "parallel.hpp"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <thread>
#include <vector>

namespace {

double process_cpu_seconds() {
  return static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
}

bool verify_dispatch(std::size_t count) {
  std::vector<std::uint64_t> values(count, 0);
  fast_math_internal::parallel_for_static(
      count, 8, [&](std::size_t index) {
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
  // First dispatch spawns the pool; give the newborn workers time to finish
  // their initial grind and park before any accounting starts. On loaded
  // continuous-integration runners their settle-out otherwise lands inside
  // the measured window and masquerades as idle spinning.
  if (!verify_dispatch(task_count)) {
    std::cerr << "initial ForkUnion dispatch failed\n";
    return 1;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  const auto cpu_started = process_cpu_seconds();
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  const auto idle_cpu_seconds = process_cpu_seconds() - cpu_started;
  if (idle_cpu_seconds > 0.1) {
    std::cerr << "idle ForkUnion workers consumed " << idle_cpu_seconds
              << " process CPU seconds\n";
    return 1;
  }

  if (!verify_dispatch(task_count)) {
    std::cerr << "ForkUnion wake dispatch failed\n";
    return 1;
  }
  return 0;
}
