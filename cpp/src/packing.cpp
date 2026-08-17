#include "fast_math.h"
#include "parallel.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kPoseChunk = 4096;

void set_error(char* destination, std::size_t size, const char* message) {
  if (destination == nullptr || size == 0) return;
  std::strncpy(destination, message, size - 1);
  destination[size - 1] = '\0';
}

inline void classify_scalar(
    double point_x, double point_y,
    double center_x, double center_y,
    double direction_x, double direction_y,
    double inner_extent, double outer_extent,
    bool& definitely_inside, bool& uncertain) noexcept {
  const double dx = point_x - center_x;
  const double dy = point_y - center_y;
  const double first = std::abs(std::fma(dx, direction_x, dy * direction_y));
  const double second = std::abs(std::fma(dy, direction_x, -dx * direction_y));
  const bool possible = first <= outer_extent && second <= outer_extent;
  definitely_inside = first <= inner_extent && second <= inner_extent;
  uncertain = possible && !definitely_inside;
}

void classify_block_scalar(
    const double* points,
    std::size_t point_begin,
    std::size_t point_end,
    const double* center_x,
    const double* center_y,
    const double* direction_x,
    const double* direction_y,
    std::size_t pose_begin,
    std::size_t pose_end,
    double inner_extent,
    double outer_extent,
    std::uint64_t* inside,
    std::uint64_t* uncertain) noexcept {
  for (std::size_t pose = pose_begin; pose < pose_end; ++pose) {
    std::uint64_t inside_word = 0;
    std::uint64_t uncertain_word = 0;
    for (std::size_t point = point_begin; point < point_end; ++point) {
      bool point_inside;
      bool point_uncertain;
      classify_scalar(
          points[2 * point], points[2 * point + 1],
          center_x[pose], center_y[pose],
          direction_x[pose], direction_y[pose],
          inner_extent, outer_extent,
          point_inside, point_uncertain);
      const auto bit = std::uint64_t{1} << (point - point_begin);
      if (point_inside) inside_word |= bit;
      if (point_uncertain) uncertain_word |= bit;
    }
    inside[pose] = inside_word;
    uncertain[pose] = uncertain_word;
  }
}

#if defined(__AVX512F__)
void classify_block_avx512(
    const double* points,
    std::size_t point_begin,
    std::size_t point_end,
    const double* center_x,
    const double* center_y,
    const double* direction_x,
    const double* direction_y,
    std::size_t pose_begin,
    std::size_t pose_end,
    double inner_extent,
    double outer_extent,
    std::uint64_t* inside,
    std::uint64_t* uncertain) noexcept {
  const auto inner = _mm512_set1_pd(inner_extent);
  const auto outer = _mm512_set1_pd(outer_extent);
  const auto sign = _mm512_set1_epi64(
      static_cast<long long>(std::uint64_t{1} << 63));
  auto pose = pose_begin;
  for (; pose + 8 <= pose_end; pose += 8) {
    const auto cx = _mm512_loadu_pd(center_x + pose);
    const auto cy = _mm512_loadu_pd(center_y + pose);
    const auto c = _mm512_loadu_pd(direction_x + pose);
    const auto s = _mm512_loadu_pd(direction_y + pose);
    auto inside_lanes = _mm512_setzero_si512();
    auto uncertain_lanes = _mm512_setzero_si512();
    for (std::size_t point = point_begin; point < point_end; ++point) {
      const auto dx = _mm512_sub_pd(
          _mm512_set1_pd(points[2 * point]), cx);
      const auto dy = _mm512_sub_pd(
          _mm512_set1_pd(points[2 * point + 1]), cy);
      const auto first = _mm512_castsi512_pd(_mm512_andnot_si512(
          sign,
          _mm512_castpd_si512(_mm512_fmadd_pd(dx, c, _mm512_mul_pd(dy, s)))));
      const auto second = _mm512_castsi512_pd(_mm512_andnot_si512(
          sign,
          _mm512_castpd_si512(_mm512_fmsub_pd(dy, c, _mm512_mul_pd(dx, s)))));
      const auto possible = static_cast<__mmask8>(
          _mm512_cmp_pd_mask(first, outer, _CMP_LE_OQ) &
          _mm512_cmp_pd_mask(second, outer, _CMP_LE_OQ));
      const auto definite = static_cast<__mmask8>(
          _mm512_cmp_pd_mask(first, inner, _CMP_LE_OQ) &
          _mm512_cmp_pd_mask(second, inner, _CMP_LE_OQ));
      const auto bit = _mm512_set1_epi64(static_cast<long long>(
          std::uint64_t{1} << (point - point_begin)));
      inside_lanes = _mm512_mask_or_epi64(
          inside_lanes, definite, inside_lanes, bit);
      uncertain_lanes = _mm512_mask_or_epi64(
          uncertain_lanes,
          static_cast<__mmask8>(possible & ~definite),
          uncertain_lanes,
          bit);
    }
    _mm512_storeu_si512(inside + pose, inside_lanes);
    _mm512_storeu_si512(uncertain + pose, uncertain_lanes);
  }
  classify_block_scalar(
      points, point_begin, point_end,
      center_x, center_y, direction_x, direction_y,
      pose, pose_end, inner_extent, outer_extent,
      inside, uncertain);
}
#endif

void weighted_block_scalar(
    const double* points,
    const double* weights,
    std::size_t point_count,
    const double* center_x,
    const double* center_y,
    const double* direction_x,
    const double* direction_y,
    std::size_t pose_begin,
    std::size_t pose_end,
    double inner_extent,
    double outer_extent,
    double* definite_scores,
    double* possible_scores) noexcept {
  for (std::size_t pose = pose_begin; pose < pose_end; ++pose) {
    double definite_score = 0.0;
    double possible_score = 0.0;
    for (std::size_t point = 0; point < point_count; ++point) {
      bool inside;
      bool uncertain;
      classify_scalar(
          points[2 * point], points[2 * point + 1],
          center_x[pose], center_y[pose],
          direction_x[pose], direction_y[pose],
          inner_extent, outer_extent, inside, uncertain);
      if (inside) definite_score += weights[point];
      if (inside || uncertain) possible_score += weights[point];
    }
    definite_scores[pose] = definite_score;
    possible_scores[pose] = possible_score;
  }
}

#if defined(__AVX512F__)
void weighted_block_avx512(
    const double* points,
    const double* weights,
    std::size_t point_count,
    const double* center_x,
    const double* center_y,
    const double* direction_x,
    const double* direction_y,
    std::size_t pose_begin,
    std::size_t pose_end,
    double inner_extent,
    double outer_extent,
    double* definite_scores,
    double* possible_scores) noexcept {
  const auto inner = _mm512_set1_pd(inner_extent);
  const auto outer = _mm512_set1_pd(outer_extent);
  const auto sign = _mm512_set1_epi64(
      static_cast<long long>(std::uint64_t{1} << 63));
  auto pose = pose_begin;
  for (; pose + 8 <= pose_end; pose += 8) {
    const auto cx = _mm512_loadu_pd(center_x + pose);
    const auto cy = _mm512_loadu_pd(center_y + pose);
    const auto c = _mm512_loadu_pd(direction_x + pose);
    const auto s = _mm512_loadu_pd(direction_y + pose);
    auto definite_score = _mm512_setzero_pd();
    auto possible_score = _mm512_setzero_pd();
    for (std::size_t point = 0; point < point_count; ++point) {
      const auto dx = _mm512_sub_pd(
          _mm512_set1_pd(points[2 * point]), cx);
      const auto dy = _mm512_sub_pd(
          _mm512_set1_pd(points[2 * point + 1]), cy);
      const auto first = _mm512_castsi512_pd(_mm512_andnot_si512(
          sign,
          _mm512_castpd_si512(_mm512_fmadd_pd(dx, c, _mm512_mul_pd(dy, s)))));
      const auto second = _mm512_castsi512_pd(_mm512_andnot_si512(
          sign,
          _mm512_castpd_si512(_mm512_fmsub_pd(dy, c, _mm512_mul_pd(dx, s)))));
      const auto possible = static_cast<__mmask8>(
          _mm512_cmp_pd_mask(first, outer, _CMP_LE_OQ) &
          _mm512_cmp_pd_mask(second, outer, _CMP_LE_OQ));
      const auto definite = static_cast<__mmask8>(
          _mm512_cmp_pd_mask(first, inner, _CMP_LE_OQ) &
          _mm512_cmp_pd_mask(second, inner, _CMP_LE_OQ));
      const auto weight = _mm512_set1_pd(weights[point]);
      definite_score = _mm512_mask_add_pd(
          definite_score, definite, definite_score, weight);
      possible_score = _mm512_mask_add_pd(
          possible_score, possible, possible_score, weight);
    }
    _mm512_storeu_pd(definite_scores + pose, definite_score);
    _mm512_storeu_pd(possible_scores + pose, possible_score);
  }
  weighted_block_scalar(
      points, weights, point_count,
      center_x, center_y, direction_x, direction_y,
      pose, pose_end, inner_extent, outer_extent,
      definite_scores, possible_scores);
}
#endif

}  // namespace

extern "C" int fast_math_square_cover_words_f64(
    const double* points,
    std::size_t point_count,
    const double* center_x,
    const double* center_y,
    const double* direction_x,
    const double* direction_y,
    std::size_t pose_count,
    double half_extent,
    double uncertainty,
    std::uint32_t thread_count,
    std::uint64_t* inside_words,
    std::uint64_t* uncertain_words,
    fast_math_square_cover_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (points == nullptr || center_x == nullptr || center_y == nullptr ||
        direction_x == nullptr || direction_y == nullptr ||
        inside_words == nullptr || uncertain_words == nullptr || stats == nullptr) {
      throw std::invalid_argument("square-cover pointer is null");
    }
    if (point_count == 0 || pose_count == 0) {
      throw std::invalid_argument("square-cover points and poses must be nonempty");
    }
    if (!std::isfinite(half_extent) || half_extent <= 0.0 ||
        !std::isfinite(uncertainty) || uncertainty < 0.0 ||
        uncertainty >= half_extent) {
      throw std::invalid_argument("invalid square half-extent or uncertainty");
    }
    if (point_count > std::numeric_limits<std::size_t>::max() / pose_count) {
      throw std::overflow_error("square-cover incidence count overflows");
    }
    const std::size_t word_count = (point_count + 63) / 64;
    if (word_count > std::numeric_limits<std::size_t>::max() / pose_count) {
      throw std::overflow_error("square-cover output size overflows");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const auto chunks_per_word = (pose_count + kPoseChunk - 1) / kPoseChunk;
    const auto task_count = word_count * chunks_per_word;
    fast_math_internal::parallel_for_static(
        task_count,
        thread_count,
        [&](std::size_t task) noexcept {
          const auto word = task / chunks_per_word;
          const auto chunk = task % chunks_per_word;
          const auto pose_begin = chunk * kPoseChunk;
          const auto pose_end = std::min(pose_count, pose_begin + kPoseChunk);
          const auto point_begin = word * 64;
          const auto point_end = std::min(point_count, point_begin + 64);
          auto* word_inside = inside_words + word * pose_count;
          auto* word_uncertain = uncertain_words + word * pose_count;
#if defined(__AVX512F__)
          classify_block_avx512(
              points, point_begin, point_end,
              center_x, center_y, direction_x, direction_y,
              pose_begin, pose_end,
              half_extent - uncertainty,
              half_extent + uncertainty,
              word_inside, word_uncertain);
#else
          classify_block_scalar(
              points, point_begin, point_end,
              center_x, center_y, direction_x, direction_y,
              pose_begin, pose_end,
              half_extent - uncertainty,
              half_extent + uncertainty,
              word_inside, word_uncertain);
#endif
        });
    stats->point_count = point_count;
    stats->pose_count = pose_count;
    stats->word_count = word_count;
    stats->incidence_tests = point_count * pose_count;
    stats->thread_count = fast_math_internal::parallel_worker_count(
        task_count, thread_count);
#if defined(__AVX512F__)
    stats->simd_lanes = 8;
#else
    stats->simd_lanes = 1;
#endif
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

extern "C" int fast_math_square_weighted_scores_f64(
    const double* points,
    const double* weights,
    std::size_t point_count,
    const double* center_x,
    const double* center_y,
    const double* direction_x,
    const double* direction_y,
    std::size_t pose_count,
    double half_extent,
    double uncertainty,
    std::uint32_t thread_count,
    double* definite_scores,
    double* possible_scores,
    fast_math_square_cover_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (points == nullptr || weights == nullptr || center_x == nullptr ||
        center_y == nullptr || direction_x == nullptr || direction_y == nullptr ||
        definite_scores == nullptr || possible_scores == nullptr || stats == nullptr) {
      throw std::invalid_argument("square-score pointer is null");
    }
    if (point_count == 0 || pose_count == 0) {
      throw std::invalid_argument("square-score points and poses must be nonempty");
    }
    if (!std::isfinite(half_extent) || half_extent <= 0.0 ||
        !std::isfinite(uncertainty) || uncertainty < 0.0 ||
        uncertainty >= half_extent) {
      throw std::invalid_argument("invalid square half-extent or uncertainty");
    }
    if (point_count > std::numeric_limits<std::size_t>::max() / pose_count) {
      throw std::overflow_error("square-score incidence count overflows");
    }
    for (std::size_t point = 0; point < point_count; ++point) {
      if (!std::isfinite(weights[point]) || weights[point] < 0.0) {
        throw std::invalid_argument("square-score weights must be finite and nonnegative");
      }
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const auto task_count = (pose_count + kPoseChunk - 1) / kPoseChunk;
    fast_math_internal::parallel_for_static(
        task_count,
        thread_count,
        [&](std::size_t task) noexcept {
          const auto pose_begin = task * kPoseChunk;
          const auto pose_end = std::min(pose_count, pose_begin + kPoseChunk);
#if defined(__AVX512F__)
          weighted_block_avx512(
              points, weights, point_count,
              center_x, center_y, direction_x, direction_y,
              pose_begin, pose_end,
              half_extent - uncertainty, half_extent + uncertainty,
              definite_scores, possible_scores);
#else
          weighted_block_scalar(
              points, weights, point_count,
              center_x, center_y, direction_x, direction_y,
              pose_begin, pose_end,
              half_extent - uncertainty, half_extent + uncertainty,
              definite_scores, possible_scores);
#endif
        });
    stats->point_count = point_count;
    stats->pose_count = pose_count;
    stats->word_count = 0;
    stats->incidence_tests = point_count * pose_count;
    stats->thread_count = fast_math_internal::parallel_worker_count(
        task_count, thread_count);
#if defined(__AVX512F__)
    stats->simd_lanes = 8;
#else
    stats->simd_lanes = 1;
#endif
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
