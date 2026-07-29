#pragma once

#include <flint/acb.h>
#include <flint/arb.h>

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define FAST_MATH_ARB_API __declspec(dllexport)
#else
#define FAST_MATH_ARB_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    arb_t center;
    arb_t difference;
    arb_t weight;
    acb_t complex_center;
    acb_t complex_difference;
} fast_math_arb_cache_workspace;

typedef int (*fast_math_arb_ordered_map_callback)(
    uint64_t begin,
    uint64_t end,
    uint32_t worker_index,
    arb_ptr weighted_error_terms,
    uint32_t error_stream_count,
    void *context);

typedef struct {
    uint64_t block_count;
    uint32_t worker_count;
    double elapsed_seconds;
} fast_math_arb_weight_interval_stats;

FAST_MATH_ARB_API void fast_math_arb_cache_workspace_init(
    fast_math_arb_cache_workspace *workspace);

FAST_MATH_ARB_API void fast_math_arb_cache_workspace_clear(
    fast_math_arb_cache_workspace *workspace);

FAST_MATH_ARB_API void fast_math_arb_weight_from_log(
    arb_t result,
    const arb_t log_index,
    const arb_t sigma,
    slong precision);

FAST_MATH_ARB_API double fast_math_arb_cache_real_term(
    arb_t weighted_error,
    const arb_t exact,
    const arb_t weight,
    fast_math_arb_cache_workspace *workspace,
    slong precision);

FAST_MATH_ARB_API double fast_math_arb_cache_real(
    arb_t error_sum,
    const arb_t exact,
    const arb_t weight,
    fast_math_arb_cache_workspace *workspace,
    slong precision);

FAST_MATH_ARB_API double fast_math_arb_cache_real_from_log(
    arb_t error_sum,
    const arb_t exact,
    const arb_t log_index,
    const arb_t sigma,
    fast_math_arb_cache_workspace *workspace,
    slong precision);

FAST_MATH_ARB_API void fast_math_arb_cache_complex_term(
    double *real_center,
    double *imaginary_center,
    arb_t weighted_error,
    const acb_t exact,
    const arb_t weight,
    fast_math_arb_cache_workspace *workspace,
    slong precision);

FAST_MATH_ARB_API void fast_math_arb_cache_complex(
    double *real_center,
    double *imaginary_center,
    arb_t error_sum,
    const acb_t exact,
    const arb_t weight,
    fast_math_arb_cache_workspace *workspace,
    slong precision);

FAST_MATH_ARB_API void fast_math_arb_cache_complex_from_log(
    double *real_center,
    double *imaginary_center,
    arb_t error_sum,
    const acb_t exact,
    const arb_t log_index,
    const arb_t sigma,
    fast_math_arb_cache_workspace *workspace,
    slong precision);

/*
 * Map independent rigorous terms in parallel, then add each error stream in
 * original item order. The callback owns disjoint output rows and may use
 * worker_index to select thread-local scratch state. Temporary storage is
 * item_count * error_stream_count Arb values.
 */
FAST_MATH_ARB_API uint32_t fast_math_arb_ordered_worker_count(
    uint64_t item_count,
    uint64_t chunk_size,
    uint32_t requested_workers);

FAST_MATH_ARB_API int fast_math_arb_ordered_map_reduce(
    uint64_t item_count,
    uint64_t chunk_size,
    uint32_t requested_workers,
    uint32_t error_stream_count,
    slong precision,
    fast_math_arb_ordered_map_callback callback,
    void *context,
    arb_ptr error_sums,
    char *error_message,
    size_t error_message_size);

/*
 * Evaluate independent two-sided endpoint weights at fixed output offsets.
 * Each block stores an upper bound for left^(-sigma_lower) and a lower bound
 * for right^(-sigma_upper), rounded outward to binary64.
 */
FAST_MATH_ARB_API int fast_math_arb_weight_intervals_u64(
    const uint64_t *left,
    const uint64_t *right,
    size_t block_count,
    const arb_t sigma_lower,
    const arb_t sigma_upper,
    slong precision,
    uint32_t requested_workers,
    uint64_t chunk_size,
    double *weight_lower,
    double *weight_upper,
    fast_math_arb_weight_interval_stats *stats,
    char *error_message,
    size_t error_message_size);

#ifdef __cplusplus
}
#endif
