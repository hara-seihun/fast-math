#include "fast_math_arb.h"
#include "parallel.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>

namespace {

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

std::size_t checked_chunk_count(
    std::uint64_t item_count,
    std::uint64_t chunk_size) {
  if (chunk_size == 0) {
    throw std::invalid_argument("chunk_size must be positive");
  }
  const auto chunks = item_count / chunk_size +
      static_cast<std::uint64_t>(item_count % chunk_size != 0);
  if (chunks > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("chunk count exceeds addressable memory");
  }
  return static_cast<std::size_t>(chunks);
}

}  // namespace

extern "C" {

std::uint32_t fast_math_arb_ordered_worker_count(
    std::uint64_t item_count,
    std::uint64_t chunk_size,
    std::uint32_t requested_workers) {
  try {
    const auto chunk_count = checked_chunk_count(item_count, chunk_size);
    return fast_math_internal::parallel_worker_count(
        chunk_count, requested_workers);
  } catch (...) {
    return 0;
  }
}

int fast_math_arb_ordered_map_reduce(
    std::uint64_t item_count,
    std::uint64_t chunk_size,
    std::uint32_t requested_workers,
    std::uint32_t error_stream_count,
    slong precision,
    fast_math_arb_ordered_map_callback callback,
    void* context,
    arb_ptr error_sums,
    char* error_message,
    std::size_t error_message_size) {
  arb_ptr terms = nullptr;
  slong term_count_slong = 0;
  try {
    if (error_stream_count == 0) {
      throw std::invalid_argument("error_stream_count must be positive");
    }
    if (precision <= 0) {
      throw std::invalid_argument("precision must be positive");
    }
    if (callback == nullptr || error_sums == nullptr) {
      throw std::invalid_argument("ordered map/reduce pointer is null");
    }
    const auto chunk_count = checked_chunk_count(item_count, chunk_size);
    if (item_count >
        std::numeric_limits<std::size_t>::max() / error_stream_count) {
      throw std::overflow_error("temporary Arb term count exceeds memory");
    }
    const auto term_count = static_cast<std::size_t>(item_count) *
        static_cast<std::size_t>(error_stream_count);
    if (term_count >
        static_cast<std::size_t>(std::numeric_limits<slong>::max())) {
      throw std::overflow_error("temporary Arb term count exceeds FLINT limits");
    }
    term_count_slong = static_cast<slong>(term_count);
    set_error(error_message, error_message_size, "");
    for (std::uint32_t stream = 0; stream < error_stream_count; ++stream) {
      arb_zero(error_sums + stream);
    }
    if (item_count == 0) {
      return 0;
    }

    terms = _arb_vec_init(term_count_slong);
    std::atomic<int> callback_failure{0};
    fast_math_internal::parallel_for_dynamic_indexed(
        chunk_count,
        requested_workers,
        [&](std::size_t chunk, std::size_t worker) {
          if (callback_failure.load(std::memory_order_relaxed) != 0) {
            return;
          }
          const auto begin = static_cast<std::uint64_t>(chunk) * chunk_size;
          const auto remaining = item_count - begin;
          const auto end = remaining < chunk_size
              ? item_count
              : begin + chunk_size;
          const auto offset = static_cast<std::size_t>(begin) *
              static_cast<std::size_t>(error_stream_count);
          const int status = callback(
              begin,
              end,
              static_cast<std::uint32_t>(worker),
              terms + offset,
              error_stream_count,
              context);
          if (status != 0) {
            int expected = 0;
            callback_failure.compare_exchange_strong(
                expected, status, std::memory_order_relaxed);
          }
        });
    if (callback_failure.load(std::memory_order_relaxed) != 0) {
      set_error(error_message, error_message_size, "ordered map callback failed");
      _arb_vec_clear(terms, term_count_slong);
      return 3;
    }

    for (std::size_t item = 0; item < item_count; ++item) {
      const auto offset =
          item * static_cast<std::size_t>(error_stream_count);
      for (std::uint32_t stream = 0; stream < error_stream_count; ++stream) {
        arb_add(
            error_sums + stream,
            error_sums + stream,
            terms + offset + stream,
            precision);
      }
    }
    _arb_vec_clear(terms, term_count_slong);
    return 0;
  } catch (const std::exception& error) {
    if (terms != nullptr) {
      _arb_vec_clear(terms, term_count_slong);
    }
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    if (terms != nullptr) {
      _arb_vec_clear(terms, term_count_slong);
    }
    set_error(error_message, error_message_size, "unknown ordered map/reduce error");
    return 2;
  }
}

}  // extern "C"
