#include "fast_math_arb.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    fast_math_arb_cache_workspace cache;
    arb_t heat;
    arb_t y;
    arb_t sigma;
    arb_t gamma_bound;
    arb_t log_index;
    arb_t primary;
    arb_t low;
} worker_state;

typedef struct {
    worker_state *workers;
    double *centers;
    slong precision;
    uint64_t fail_at;
} map_context;

static int
same_double(double left, double right)
{
    uint64_t left_bits;
    uint64_t right_bits;

    memcpy(&left_bits, &left, sizeof(left_bits));
    memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

static void
worker_state_init(worker_state *state)
{
    fast_math_arb_cache_workspace_init(&state->cache);
    arb_init(state->heat);
    arb_init(state->y);
    arb_init(state->sigma);
    arb_init(state->gamma_bound);
    arb_init(state->log_index);
    arb_init(state->primary);
    arb_init(state->low);
    arb_set_d(state->heat, 0.1429086752);
    arb_set_d(state->y, 0.19166);
    arb_set_d(state->sigma, 1.5555962870835469);
    arb_set_d(state->gamma_bound, 0.076290732657071203);
}

static void
worker_state_clear(worker_state *state)
{
    fast_math_arb_cache_workspace_clear(&state->cache);
    arb_clear(state->heat);
    arb_clear(state->y);
    arb_clear(state->sigma);
    arb_clear(state->gamma_bound);
    arb_clear(state->log_index);
    arb_clear(state->primary);
    arb_clear(state->low);
}

static int
map_terms(
    uint64_t begin,
    uint64_t end,
    uint32_t worker_index,
    arb_ptr terms,
    uint32_t stream_count,
    void *opaque)
{
    uint64_t item;
    map_context *context = opaque;
    worker_state *state = context->workers + worker_index;

    if (stream_count != 2)
        return 11;
    if (context->fail_at >= begin && context->fail_at < end)
        return 12;
    for (item = begin; item < end; item++)
    {
        ulong index = (ulong) item + 1;
        size_t offset = (size_t) (item - begin) * stream_count;

        arb_log_ui(state->log_index, index, context->precision);
        arb_mul(
            state->primary,
            state->log_index,
            state->log_index,
            context->precision);
        arb_mul(
            state->primary,
            state->primary,
            state->heat,
            context->precision);
        arb_mul_2exp_si(state->primary, state->primary, -2);
        arb_exp(state->primary, state->primary, context->precision);
        fast_math_arb_weight_from_log(
            state->cache.weight,
            state->log_index,
            state->sigma,
            context->precision);
        context->centers[2 * item] = fast_math_arb_cache_real_term(
            terms + offset,
            state->primary,
            state->cache.weight,
            &state->cache,
            context->precision);

        arb_mul(
            state->low,
            state->y,
            state->log_index,
            context->precision);
        arb_exp(state->low, state->low, context->precision);
        arb_mul(
            state->low,
            state->low,
            state->primary,
            context->precision);
        arb_mul(
            state->low,
            state->low,
            state->gamma_bound,
            context->precision);
        context->centers[2 * item + 1] =
            fast_math_arb_cache_real_term(
                terms + offset + 1,
                state->low,
                state->cache.weight,
                &state->cache,
                context->precision);
    }
    return 0;
}

static int
run_configuration(
    uint64_t count,
    uint64_t chunk_size,
    uint32_t requested_workers,
    slong precision,
    const double *expected_centers,
    arb_srcptr expected_errors)
{
    uint64_t item;
    uint32_t worker;
    uint32_t worker_count = fast_math_arb_ordered_worker_count(
        count, chunk_size, requested_workers);
    worker_state *workers = calloc(worker_count, sizeof(*workers));
    double *centers = calloc(2 * count, sizeof(*centers));
    arb_ptr errors = _arb_vec_init(2);
    char message[128];
    map_context context;
    int status;

    if (workers == NULL || centers == NULL)
        return 1;
    for (worker = 0; worker < worker_count; worker++)
        worker_state_init(workers + worker);
    context.workers = workers;
    context.centers = centers;
    context.precision = precision;
    context.fail_at = UINT64_MAX;
    status = fast_math_arb_ordered_map_reduce(
        count,
        chunk_size,
        requested_workers,
        2,
        precision,
        map_terms,
        &context,
        errors,
        message,
        sizeof(message));
    if (status != 0)
    {
        fprintf(stderr, "ordered map/reduce failed: %s\n", message);
        return 1;
    }
    for (item = 0; item < 2 * count; item++)
    {
        if (!same_double(centers[item], expected_centers[item]))
        {
            fprintf(stderr, "center changed at item %llu\n",
                (unsigned long long) item);
            return 1;
        }
    }
    if (!arb_equal(errors, expected_errors)
        || !arb_equal(errors + 1, expected_errors + 1))
    {
        fprintf(stderr, "ordered error certificate changed\n");
        return 1;
    }

    for (worker = 0; worker < worker_count; worker++)
        worker_state_clear(workers + worker);
    free(workers);
    free(centers);
    _arb_vec_clear(errors, 2);
    return 0;
}

int
main(void)
{
    const uint64_t count = 1003;
    const slong precision = 256;
    const uint64_t chunk_sizes[] = {1, 7, 64, 1003};
    const uint32_t worker_counts[] = {1, 2, 3};
    size_t chunk_index;
    size_t worker_index;
    uint64_t item;
    worker_state serial;
    worker_state *failure_workers;
    double *expected_centers = calloc(2 * count, sizeof(*expected_centers));
    double *failure_centers;
    arb_ptr expected_errors = _arb_vec_init(2);
    arb_ptr failure_errors;
    map_context failure_context;
    char message[128];
    uint32_t failure_worker_count;
    uint32_t failure_worker;
    int failure_status;

    if (expected_centers == NULL)
        return 1;
    worker_state_init(&serial);
    arb_zero(expected_errors);
    arb_zero(expected_errors + 1);
    for (item = 0; item < count; item++)
    {
        ulong index = (ulong) item + 1;

        arb_log_ui(serial.log_index, index, precision);
        arb_mul(
            serial.primary,
            serial.log_index,
            serial.log_index,
            precision);
        arb_mul(serial.primary, serial.primary, serial.heat, precision);
        arb_mul_2exp_si(serial.primary, serial.primary, -2);
        arb_exp(serial.primary, serial.primary, precision);
        fast_math_arb_weight_from_log(
            serial.cache.weight,
            serial.log_index,
            serial.sigma,
            precision);
        expected_centers[2 * item] = fast_math_arb_cache_real(
            expected_errors,
            serial.primary,
            serial.cache.weight,
            &serial.cache,
            precision);

        arb_mul(serial.low, serial.y, serial.log_index, precision);
        arb_exp(serial.low, serial.low, precision);
        arb_mul(serial.low, serial.low, serial.primary, precision);
        arb_mul(serial.low, serial.low, serial.gamma_bound, precision);
        expected_centers[2 * item + 1] = fast_math_arb_cache_real(
            expected_errors + 1,
            serial.low,
            serial.cache.weight,
            &serial.cache,
            precision);
    }
    worker_state_clear(&serial);

    for (chunk_index = 0;
         chunk_index < sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);
         chunk_index++)
    {
        for (worker_index = 0;
             worker_index < sizeof(worker_counts) / sizeof(worker_counts[0]);
             worker_index++)
        {
            if (run_configuration(
                    count,
                    chunk_sizes[chunk_index],
                    worker_counts[worker_index],
                    precision,
                    expected_centers,
                    expected_errors) != 0)
            {
                return 1;
            }
        }
    }

    if (fast_math_arb_ordered_map_reduce(
            count,
            0,
            2,
            2,
            precision,
            map_terms,
            NULL,
            expected_errors,
            message,
            sizeof(message)) == 0)
    {
        fprintf(stderr, "zero chunk size was accepted\n");
        return 1;
    }

    failure_worker_count = fast_math_arb_ordered_worker_count(
        count, 64, 2);
    failure_workers = calloc(
        failure_worker_count, sizeof(*failure_workers));
    failure_centers = calloc(2 * count, sizeof(*failure_centers));
    failure_errors = _arb_vec_init(2);
    if (failure_workers == NULL || failure_centers == NULL)
        return 1;
    for (failure_worker = 0;
         failure_worker < failure_worker_count;
         failure_worker++)
    {
        worker_state_init(failure_workers + failure_worker);
    }
    failure_context.workers = failure_workers;
    failure_context.centers = failure_centers;
    failure_context.precision = precision;
    failure_context.fail_at = 13;
    failure_status = fast_math_arb_ordered_map_reduce(
        count,
        64,
        2,
        2,
        precision,
        map_terms,
        &failure_context,
        failure_errors,
        message,
        sizeof(message));
    if (failure_status != 3 || strstr(message, "callback failed") == NULL)
    {
        fprintf(stderr, "callback failure was not propagated\n");
        return 1;
    }
    for (failure_worker = 0;
         failure_worker < failure_worker_count;
         failure_worker++)
    {
        worker_state_clear(failure_workers + failure_worker);
    }
    free(failure_workers);
    free(failure_centers);
    _arb_vec_clear(failure_errors, 2);

    arb_one(expected_errors);
    arb_one(expected_errors + 1);
    if (fast_math_arb_ordered_map_reduce(
            0,
            64,
            2,
            2,
            precision,
            map_terms,
            NULL,
            expected_errors,
            message,
            sizeof(message)) != 0
        || !arb_is_zero(expected_errors)
        || !arb_is_zero(expected_errors + 1))
    {
        fprintf(stderr, "empty ordered map/reduce failed\n");
        return 1;
    }

    free(expected_centers);
    _arb_vec_clear(expected_errors, 2);
    flint_cleanup();
    return 0;
}
