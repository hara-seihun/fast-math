#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#if FAST_MATH_USE_FORKUNION
#include <forkunion.hpp>
#endif

namespace fast_math_internal {

inline std::uint32_t parallel_worker_count(
    std::size_t task_count,
    std::uint32_t requested_workers) {
  auto workers = requested_workers == 0
      ? std::max(1u, std::thread::hardware_concurrency())
      : requested_workers;
  return static_cast<std::uint32_t>(
      std::min<std::size_t>(
          workers, std::max<std::size_t>(1, task_count)));
}

#if FAST_MATH_USE_FORKUNION

namespace fu = ashvardanian::forkunion;

class ForkUnionCache {
 public:
  fu::flat_pool_t& get(std::uint32_t workers) {
    if (pools_.size() <= workers) {
      pools_.resize(static_cast<std::size_t>(workers) + 1);
    }
    auto& pool = pools_[workers];
    if (!pool) {
      pool = std::make_unique<fu::flat_pool_t>();
      if (!pool->try_spawn(workers, fu::caller_inclusive_k)) {
        pool.reset();
        throw std::runtime_error("ForkUnion pool creation failed");
      }
    }
    return *pool;
  }

 private:
  std::vector<std::unique_ptr<fu::flat_pool_t>> pools_;
};

inline ForkUnionCache& fork_union_cache() {
  thread_local ForkUnionCache cache;
  return cache;
}

template <typename Work>
void parallel_for_dynamic_indexed(
    std::size_t task_count,
    std::uint32_t requested_workers,
    Work work) {
  if (task_count == 0) {
    return;
  }
  const auto workers =
      parallel_worker_count(task_count, requested_workers);
  if (workers == 1) {
    for (std::size_t task = 0; task < task_count; ++task) {
      work(task, 0);
    }
    return;
  }
  auto& pool = fork_union_cache().get(workers);
  pool.for_n_dynamic(
      task_count,
      [&](fu::flat_pool_t::prong_t prong) noexcept {
        work(
            static_cast<std::size_t>(prong.task),
            static_cast<std::size_t>(prong.thread));
      });
}

template <typename Work>
void parallel_for_static_indexed(
    std::size_t task_count,
    std::uint32_t requested_workers,
    Work work) {
  if (task_count == 0) {
    return;
  }
  const auto workers =
      parallel_worker_count(task_count, requested_workers);
  if (workers == 1) {
    for (std::size_t task = 0; task < task_count; ++task) {
      work(task, 0);
    }
    return;
  }
  auto& pool = fork_union_cache().get(workers);
  pool.for_n(
      task_count,
      [&](fu::flat_pool_t::prong_t prong) noexcept {
        work(
            static_cast<std::size_t>(prong.task),
            static_cast<std::size_t>(prong.thread));
      });
}

#else

template <typename Work>
void parallel_for_dynamic_indexed(
    std::size_t task_count,
    std::uint32_t requested_workers,
    Work work) {
  if (task_count == 0) {
    return;
  }
  const auto workers =
      parallel_worker_count(task_count, requested_workers);
  std::atomic<std::size_t> next_task{0};
  auto worker = [&](std::size_t worker_index) {
    while (true) {
      const auto task =
          next_task.fetch_add(1, std::memory_order_relaxed);
      if (task >= task_count) {
        return;
      }
      work(task, worker_index);
    }
  };
  std::vector<std::thread> threads;
  threads.reserve(workers - 1);
  for (std::uint32_t index = 1; index < workers; ++index) {
    threads.emplace_back(worker, index);
  }
  worker(0);
  for (auto& thread : threads) {
    thread.join();
  }
}

template <typename Work>
void parallel_for_static_indexed(
    std::size_t task_count,
    std::uint32_t requested_workers,
    Work work) {
  if (task_count == 0) {
    return;
  }
  const auto workers =
      parallel_worker_count(task_count, requested_workers);
  auto worker = [&](std::size_t worker_index) {
    const auto begin = task_count * worker_index / workers;
    const auto end = task_count * (worker_index + 1) / workers;
    for (auto task = begin; task < end; ++task) {
      work(task, worker_index);
    }
  };
  std::vector<std::thread> threads;
  threads.reserve(workers - 1);
  for (std::uint32_t index = 1; index < workers; ++index) {
    threads.emplace_back(worker, index);
  }
  worker(0);
  for (auto& thread : threads) {
    thread.join();
  }
}

#endif

template <typename Work>
void parallel_for_dynamic(
    std::size_t task_count,
    std::uint32_t requested_workers,
    Work work) {
  parallel_for_dynamic_indexed(
      task_count,
      requested_workers,
      [&](std::size_t task, std::size_t) { work(task); });
}

template <typename Work>
void parallel_for_static(
    std::size_t task_count,
    std::uint32_t requested_workers,
    Work work) {
  parallel_for_static_indexed(
      task_count,
      requested_workers,
      [&](std::size_t task, std::size_t) { work(task); });
}

}  // namespace fast_math_internal
