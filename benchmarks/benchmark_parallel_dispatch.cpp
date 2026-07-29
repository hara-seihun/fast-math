#include <forkunion.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace fu = ashvardanian::forkunion;

namespace {

using Clock = std::chrono::steady_clock;

std::size_t parse_size(char const* text, char const* name) {
  char* end = nullptr;
  auto const value = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0' || value == 0) {
    std::fprintf(stderr, "invalid %s\n", name);
    std::exit(2);
  }
  return static_cast<std::size_t>(value);
}

void execute_task(
    std::size_t task,
    std::size_t work,
    std::vector<double>& output) {
  double value = static_cast<double>(task + 1) * 0.000001;
  for (std::size_t index = 0; index < work; ++index) {
    value = std::fma(value, 1.0000001192092896, 0.0000009536743164);
  }
  output[task] = value;
}

double run_std_threads(
    std::size_t threads,
    std::size_t iterations,
    std::size_t tasks,
    std::size_t work,
    std::vector<double>& output) {
  auto const started = Clock::now();
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    std::atomic<std::size_t> next_task{0};
    auto worker = [&]() {
      while (true) {
        auto const task =
            next_task.fetch_add(1, std::memory_order_relaxed);
        if (task >= tasks) {
          return;
        }
        execute_task(task, work, output);
      }
    };
    std::vector<std::thread> workers;
    workers.reserve(threads - 1);
    for (std::size_t index = 1; index < threads; ++index) {
      workers.emplace_back(worker);
    }
    worker();
    for (auto& thread : workers) {
      thread.join();
    }
  }
  return std::chrono::duration<double>(Clock::now() - started).count();
}

double run_forkunion(
    std::size_t threads,
    std::size_t iterations,
    std::size_t tasks,
    std::size_t work,
    bool dynamic,
    std::vector<double>& output) {
  fu::flat_pool_t pool;
  if (!pool.try_spawn(threads, fu::caller_inclusive_k)) {
    std::fprintf(stderr, "ForkUnion pool creation failed\n");
    std::exit(1);
  }
  auto const started = Clock::now();
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    if (dynamic) {
      pool.for_n_dynamic(tasks, [&](std::size_t task) noexcept {
        execute_task(task, work, output);
      });
    } else {
      pool.for_n(tasks, [&](std::size_t task) noexcept {
        execute_task(task, work, output);
      });
    }
  }
  return std::chrono::duration<double>(Clock::now() - started).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::fprintf(
        stderr,
        "usage: %s std_threads|forkunion_static|forkunion_dynamic "
        "THREADS ITERATIONS TASKS WORK\n",
        argv[0]);
    return 2;
  }
  auto const threads = parse_size(argv[2], "threads");
  auto const iterations = parse_size(argv[3], "iterations");
  auto const tasks = parse_size(argv[4], "tasks");
  auto const work = parse_size(argv[5], "work");
  std::vector<double> output(tasks, 0.0);
  double elapsed = 0.0;
  if (std::strcmp(argv[1], "std_threads") == 0) {
    elapsed = run_std_threads(
        threads, iterations, tasks, work, output);
  } else if (std::strcmp(argv[1], "forkunion_static") == 0) {
    elapsed = run_forkunion(
        threads, iterations, tasks, work, false, output);
  } else if (std::strcmp(argv[1], "forkunion_dynamic") == 0) {
    elapsed = run_forkunion(
        threads, iterations, tasks, work, true, output);
  } else {
    std::fprintf(stderr, "invalid backend\n");
    return 2;
  }
  double checksum = 0.0;
  for (double value : output) {
    checksum += value;
  }
  std::printf(
      "{\"benchmark\":\"parallel_dispatch\","
      "\"backend\":\"%s\",\"threads\":%zu,\"iterations\":%zu,"
      "\"tasks\":%zu,\"work\":%zu,\"wall_seconds\":%.9f,"
      "\"dispatches_per_second\":%.9f,\"checksum\":%.17g}\n",
      argv[1],
      threads,
      iterations,
      tasks,
      work,
      elapsed,
      static_cast<double>(iterations) / elapsed,
      checksum);
  return 0;
}
