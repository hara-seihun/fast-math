#include "fast_math_arb.h"

#include <flint/arb.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double
wall_seconds(void)
{
    struct timespec value;

    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double) value.tv_sec + 1e-9 * (double) value.tv_nsec;
}

static uint64_t
mix_double(uint64_t hash, double value)
{
    uint64_t bits;

    memcpy(&bits, &value, sizeof(bits));
    hash ^= bits;
    hash *= UINT64_C(1099511628211);
    return hash;
}

static double
baseline_cache_real(
    arb_t error_sum,
    const arb_t exact,
    ulong index,
    const arb_t sigma,
    slong precision)
{
    double center_value;
    arb_t base;
    arb_t center;
    arb_t difference;
    arb_t exponent;
    arb_t weight;

    arb_init(base);
    arb_init(center);
    arb_init(difference);
    arb_init(exponent);
    arb_init(weight);
    center_value = arf_get_d(arb_midref(exact), ARF_RND_NEAR);
    arb_set_d(center, center_value);
    arb_sub(difference, exact, center, precision);
    arb_abs(difference, difference);
    arb_set_ui(base, index);
    arb_neg(exponent, sigma);
    arb_pow(weight, base, exponent, precision);
    arb_mul(difference, difference, weight, precision);
    arb_add(error_sum, error_sum, difference, precision);
    arb_clear(base);
    arb_clear(center);
    arb_clear(difference);
    arb_clear(exponent);
    arb_clear(weight);
    return center_value;
}

typedef struct {
    fast_math_arb_cache_workspace cache;
    arb_t heat;
    arb_t y;
    arb_t sigma;
    arb_t gamma_bound;
    arb_t log_index;
    arb_t primary;
    arb_t low;
} ordered_worker;

typedef struct {
    ordered_worker *workers;
    double *centers;
    slong precision;
} ordered_context;

static void
ordered_worker_init(ordered_worker *worker)
{
    fast_math_arb_cache_workspace_init(&worker->cache);
    arb_init(worker->heat);
    arb_init(worker->y);
    arb_init(worker->sigma);
    arb_init(worker->gamma_bound);
    arb_init(worker->log_index);
    arb_init(worker->primary);
    arb_init(worker->low);
    arb_set_d(worker->heat, 0.1429086752);
    arb_set_d(worker->y, 0.19166);
    arb_set_d(worker->sigma, 1.5555962870835469);
    arb_set_d(worker->gamma_bound, 0.076290732657071203);
}

static void
ordered_worker_clear(ordered_worker *worker)
{
    fast_math_arb_cache_workspace_clear(&worker->cache);
    arb_clear(worker->heat);
    arb_clear(worker->y);
    arb_clear(worker->sigma);
    arb_clear(worker->gamma_bound);
    arb_clear(worker->log_index);
    arb_clear(worker->primary);
    arb_clear(worker->low);
}

static int
ordered_map(
    uint64_t begin,
    uint64_t end,
    uint32_t worker_index,
    arb_ptr terms,
    uint32_t stream_count,
    void *opaque)
{
    uint64_t item;
    ordered_context *context = opaque;
    ordered_worker *worker = context->workers + worker_index;

    if (stream_count != 2)
        return 1;
    for (item = begin; item < end; item++)
    {
        ulong index = (ulong) item + 1;
        size_t offset = (size_t) (item - begin) * stream_count;

        arb_log_ui(worker->log_index, index, context->precision);
        arb_mul(
            worker->primary,
            worker->log_index,
            worker->log_index,
            context->precision);
        arb_mul(
            worker->primary,
            worker->primary,
            worker->heat,
            context->precision);
        arb_mul_2exp_si(worker->primary, worker->primary, -2);
        arb_exp(worker->primary, worker->primary, context->precision);
        fast_math_arb_weight_from_log(
            worker->cache.weight,
            worker->log_index,
            worker->sigma,
            context->precision);
        context->centers[2 * item] = fast_math_arb_cache_real_term(
            terms + offset,
            worker->primary,
            worker->cache.weight,
            &worker->cache,
            context->precision);

        arb_mul(
            worker->low,
            worker->y,
            worker->log_index,
            context->precision);
        arb_exp(worker->low, worker->low, context->precision);
        arb_mul(
            worker->low,
            worker->low,
            worker->primary,
            context->precision);
        arb_mul(
            worker->low,
            worker->low,
            worker->gamma_bound,
            context->precision);
        context->centers[2 * item + 1] =
            fast_math_arb_cache_real_term(
                terms + offset + 1,
                worker->low,
                worker->cache.weight,
                &worker->cache,
                context->precision);
    }
    return 0;
}

static int
run_backend(
    const char *backend,
    ulong count,
    slong precision,
    uint32_t requested_workers,
    uint64_t chunk_size)
{
    ulong index;
    uint32_t worker_index;
    uint32_t actual_workers = 1;
    uint64_t hash = UINT64_C(1469598103934665603);
    double center;
    double started;
    double *centers = NULL;
    char *error_text;
    char native_error[128];
    double error_upper;
    ordered_worker *workers = NULL;
    ordered_context context;
    arb_ptr ordered_errors = NULL;
    fast_math_arb_cache_workspace workspace;
    arb_t heat;
    arb_t y;
    arb_t sigma;
    arb_t gamma_bound;
    arb_t log_index;
    arb_t primary;
    arb_t low;
    arb_t primary_error;
    arb_t low_error;
    arb_t error_sum;
    arf_t upper;

    fast_math_arb_cache_workspace_init(&workspace);
    arb_init(heat);
    arb_init(y);
    arb_init(sigma);
    arb_init(gamma_bound);
    arb_init(log_index);
    arb_init(primary);
    arb_init(low);
    arb_init(primary_error);
    arb_init(low_error);
    arb_init(error_sum);
    arf_init(upper);
    arb_set_d(heat, 0.1429086752);
    arb_set_d(y, 0.19166);
    arb_set_d(sigma, 1.5555962870835469);
    arb_set_d(gamma_bound, 0.076290732657071203);
    arb_zero(primary_error);
    arb_zero(low_error);

    if (strcmp(backend, "ordered") == 0)
    {
        actual_workers = fast_math_arb_ordered_worker_count(
            count, chunk_size, requested_workers);
        workers = calloc(actual_workers, sizeof(*workers));
        centers = malloc(2 * count * sizeof(*centers));
        ordered_errors = _arb_vec_init(2);
        if (workers == NULL || centers == NULL || ordered_errors == NULL)
        {
            fprintf(stderr, "cannot allocate ordered benchmark buffers\n");
            return 1;
        }
        for (worker_index = 0; worker_index < actual_workers; worker_index++)
            ordered_worker_init(workers + worker_index);
        context.workers = workers;
        context.centers = centers;
        context.precision = precision;
        started = wall_seconds();
        if (fast_math_arb_ordered_map_reduce(
                count,
                chunk_size,
                requested_workers,
                2,
                precision,
                ordered_map,
                &context,
                ordered_errors,
                native_error,
                sizeof(native_error)) != 0)
        {
            fprintf(stderr, "ordered benchmark failed: %s\n", native_error);
            return 1;
        }
        arb_set(primary_error, ordered_errors);
        arb_set(low_error, ordered_errors + 1);
        started = wall_seconds() - started;
        for (index = 0; index < count; index++)
        {
            hash = mix_double(hash, centers[2 * index]);
            hash = mix_double(hash, centers[2 * index + 1]);
        }
    }
    else
    {
        started = wall_seconds();
        for (index = 1; index <= count; index++)
        {
            arb_log_ui(log_index, index, precision);
            arb_mul(primary, log_index, log_index, precision);
            arb_mul(primary, primary, heat, precision);
            arb_mul_2exp_si(primary, primary, -2);
            arb_exp(primary, primary, precision);
            if (strcmp(backend, "baseline") == 0)
            {
                center = baseline_cache_real(
                    primary_error, primary, index, sigma, precision);
            }
            else
            {
                fast_math_arb_weight_from_log(
                    workspace.weight, log_index, sigma, precision);
                center = fast_math_arb_cache_real(
                    primary_error,
                    primary,
                    workspace.weight,
                    &workspace,
                    precision);
            }
            hash = mix_double(hash, center);

            arb_mul(low, y, log_index, precision);
            arb_exp(low, low, precision);
            arb_mul(low, low, primary, precision);
            arb_mul(low, low, gamma_bound, precision);
            if (strcmp(backend, "baseline") == 0)
            {
                center = baseline_cache_real(
                    low_error, low, index, sigma, precision);
            }
            else
            {
                center = fast_math_arb_cache_real(
                    low_error,
                    low,
                    workspace.weight,
                    &workspace,
                    precision);
            }
            hash = mix_double(hash, center);
        }
        started = wall_seconds() - started;
    }
    arb_add(error_sum, primary_error, low_error, precision);
    error_text = arb_get_str(error_sum, 50, 0);
    arb_get_ubound_arf(upper, error_sum, precision);
    error_upper = arf_get_d(upper, ARF_RND_CEIL);
    printf(
        "{\"benchmark\":\"arb_source_cache_hot_loop\","
        "\"backend\":\"%s\",\"count\":%lu,\"cache_values\":%lu,"
        "\"precision_bits\":%ld,\"thread_count\":%u,"
        "\"chunk_size\":%llu,\"center_hash\":\"%016llx\","
        "\"weighted_error\":\"%s\",\"weighted_error_upper\":%.17g,"
        "\"wall_seconds\":%.9f}\n",
        backend,
        count,
        2UL * count,
        precision,
        actual_workers,
        (unsigned long long) chunk_size,
        (unsigned long long) hash,
        error_text,
        error_upper,
        started);
    flint_free(error_text);

    fast_math_arb_cache_workspace_clear(&workspace);
    arb_clear(heat);
    arb_clear(y);
    arb_clear(sigma);
    arb_clear(gamma_bound);
    arb_clear(log_index);
    arb_clear(primary);
    arb_clear(low);
    arb_clear(primary_error);
    arb_clear(low_error);
    arb_clear(error_sum);
    arf_clear(upper);
    if (workers != NULL)
    {
        for (worker_index = 0; worker_index < actual_workers; worker_index++)
            ordered_worker_clear(workers + worker_index);
    }
    free(workers);
    free(centers);
    if (ordered_errors != NULL)
        _arb_vec_clear(ordered_errors, 2);
    return 0;
}

int
main(int argc, char **argv)
{
    char *end = NULL;
    ulong count;
    long precision;
    unsigned long workers = 1;
    unsigned long long chunk_size = 2048;

    if (argc != 4 && argc != 6)
    {
        fprintf(
            stderr,
            "usage: %s baseline|optimized|ordered COUNT PRECISION "
            "[THREADS CHUNK_SIZE]\n",
            argv[0]);
        return 2;
    }
    if (
        strcmp(argv[1], "baseline") != 0
        && strcmp(argv[1], "optimized") != 0
        && strcmp(argv[1], "ordered") != 0)
    {
        fprintf(
            stderr,
            "backend must be baseline, optimized, or ordered\n");
        return 2;
    }
    count = strtoul(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || count == 0)
    {
        fprintf(stderr, "invalid count\n");
        return 2;
    }
    precision = strtol(argv[3], &end, 10);
    if (end == argv[3] || *end != '\0' || precision < 128)
    {
        fprintf(stderr, "invalid precision\n");
        return 2;
    }
    if (argc == 6)
    {
        workers = strtoul(argv[4], &end, 10);
        if (end == argv[4] || *end != '\0' || workers > UINT32_MAX)
        {
            fprintf(stderr, "invalid thread count\n");
            return 2;
        }
        chunk_size = strtoull(argv[5], &end, 10);
        if (end == argv[5] || *end != '\0' || chunk_size == 0)
        {
            fprintf(stderr, "invalid chunk size\n");
            return 2;
        }
    }
    return run_backend(
        argv[1],
        count,
        (slong) precision,
        (uint32_t) workers,
        (uint64_t) chunk_size);
}
