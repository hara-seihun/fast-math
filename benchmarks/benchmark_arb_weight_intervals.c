#include "fast_math_arb.h"

#include <flint/fmpq.h>

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

static void
die(const char *message)
{
    fprintf(stderr, "error: %s\n", message);
    exit(EXIT_FAILURE);
}

static void
set_fraction(arb_t result, const char *text, slong precision)
{
    fmpq_t value;

    fmpq_init(value);
    if (fmpq_set_str(value, text, 10) != 0)
        die("invalid rational parameter");
    fmpq_canonicalise(value);
    arb_set_fmpq(result, value, precision);
    fmpq_clear(value);
}

static int
same_double(double left, double right)
{
    uint64_t left_bits, right_bits;

    memcpy(&left_bits, &left, sizeof(left_bits));
    memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

static size_t
build_blocks(
    uint64_t output_limit,
    uint64_t ratio_numerator,
    uint64_t ratio_denominator,
    uint64_t **left_out,
    uint64_t **right_out)
{
    size_t count = 0;
    uint64_t left = 2;
    uint64_t *left_values;
    uint64_t *right_values;

    while (left <= output_limit)
    {
        unsigned __int128 product =
            (unsigned __int128) left * ratio_numerator;
        uint64_t next_left = (uint64_t) (
            (product + ratio_denominator - 1) / ratio_denominator);
        uint64_t right;

        if (next_left <= left)
            next_left = left + 1;
        right = next_left - 1;
        if (right > output_limit)
            right = output_limit;
        count++;
        if (right == output_limit)
            break;
        left = right + 1;
    }
    left_values = malloc(count * sizeof(*left_values));
    right_values = malloc(count * sizeof(*right_values));
    if (left_values == NULL || right_values == NULL)
        die("cannot allocate block endpoints");

    count = 0;
    left = 2;
    while (left <= output_limit)
    {
        unsigned __int128 product =
            (unsigned __int128) left * ratio_numerator;
        uint64_t next_left = (uint64_t) (
            (product + ratio_denominator - 1) / ratio_denominator);
        uint64_t right;

        if (next_left <= left)
            next_left = left + 1;
        right = next_left - 1;
        if (right > output_limit)
            right = output_limit;
        left_values[count] = left;
        right_values[count] = right;
        count++;
        if (right == output_limit)
            break;
        left = right + 1;
    }
    *left_out = left_values;
    *right_out = right_values;
    return count;
}

static double
serial_weights(
    const uint64_t *left,
    const uint64_t *right,
    size_t count,
    const arb_t sigma_lower,
    const arb_t sigma_upper,
    slong precision,
    double *weight_lower,
    double *weight_upper)
{
    size_t block;
    double started = wall_seconds();
    arb_t base, exponent, weight;
    arf_t lower, upper;

    arb_init(base);
    arb_init(exponent);
    arb_init(weight);
    arf_init(lower);
    arf_init(upper);
    for (block = 0; block < count; block++)
    {
        arb_set_ui(base, (ulong) left[block]);
        arb_neg(exponent, sigma_lower);
        arb_pow(weight, base, exponent, precision);
        arb_get_ubound_arf(upper, weight, precision);
        weight_upper[block] = arf_get_d(upper, ARF_RND_CEIL);

        arb_set_ui(base, (ulong) right[block]);
        arb_neg(exponent, sigma_upper);
        arb_pow(weight, base, exponent, precision);
        arb_get_lbound_arf(lower, weight, precision);
        weight_lower[block] = arf_get_d(lower, ARF_RND_FLOOR);
    }
    arb_clear(base);
    arb_clear(exponent);
    arb_clear(weight);
    arf_clear(lower);
    arf_clear(upper);
    return wall_seconds() - started;
}

int
main(int argc, char **argv)
{
    uint64_t cutoff, output_limit = 100000000;
    slong precision = 256;
    size_t block_count, block;
    uint32_t workers;
    uint64_t *left, *right;
    double *expected_lower, *expected_upper;
    double *actual_lower, *actual_upper;
    double serial_seconds, wall_started, wall_seconds_value;
    char error[256];
    fast_math_arb_weight_interval_stats stats;
    arb_t heat, height, sigma_exact, sigma_lower, sigma_upper, temporary;
    arf_t lower_bound, upper_bound;

    if (argc != 3)
        die("usage: benchmark CUTOFF WORKERS");
    cutoff = strtoull(argv[1], NULL, 10);
    workers = (uint32_t) strtoul(argv[2], NULL, 10);
    if (cutoff == 0 || workers == 0)
        die("invalid benchmark parameters");

    block_count = build_blocks(
        output_limit, 10001, 10000, &left, &right);
    expected_lower = malloc(block_count * sizeof(*expected_lower));
    expected_upper = malloc(block_count * sizeof(*expected_upper));
    actual_lower = malloc(block_count * sizeof(*actual_lower));
    actual_upper = malloc(block_count * sizeof(*actual_upper));
    if (
        expected_lower == NULL || expected_upper == NULL
        || actual_lower == NULL || actual_upper == NULL)
    {
        die("cannot allocate weight outputs");
    }

    arb_init(heat);
    arb_init(height);
    arb_init(sigma_exact);
    arb_init(sigma_lower);
    arb_init(sigma_upper);
    arb_init(temporary);
    arf_init(lower_bound);
    arf_init(upper_bound);
    set_fraction(heat, "43096461/312500000", precision);
    set_fraction(height, "1767/12500", precision);
    arb_log_ui(temporary, (ulong) cutoff, precision);
    arb_add_ui(sigma_exact, height, 1, precision);
    arb_mul_2exp_si(sigma_exact, sigma_exact, -1);
    arb_mul(temporary, temporary, heat, precision);
    arb_mul_2exp_si(temporary, temporary, -1);
    arb_add(sigma_exact, sigma_exact, temporary, precision);
    set_fraction(temporary, "1/1000", precision);
    arb_sub(sigma_exact, sigma_exact, temporary, precision);
    arb_get_lbound_arf(lower_bound, sigma_exact, precision);
    arb_get_ubound_arf(upper_bound, sigma_exact, precision);
    arb_set_d(
        sigma_lower, arf_get_d(lower_bound, ARF_RND_FLOOR));
    arb_set_d(
        sigma_upper, arf_get_d(upper_bound, ARF_RND_CEIL));

    serial_seconds = serial_weights(
        left,
        right,
        block_count,
        sigma_lower,
        sigma_upper,
        precision,
        expected_lower,
        expected_upper);
    wall_started = wall_seconds();
    if (fast_math_arb_weight_intervals_u64(
            left,
            right,
            block_count,
            sigma_lower,
            sigma_upper,
            precision,
            workers,
            256,
            actual_lower,
            actual_upper,
            &stats,
            error,
            sizeof(error)) != 0)
    {
        fprintf(stderr, "error: %s\n", error);
        return 1;
    }
    wall_seconds_value = wall_seconds() - wall_started;
    for (block = 0; block < block_count; block++)
    {
        if (
            !same_double(expected_lower[block], actual_lower[block])
            || !same_double(expected_upper[block], actual_upper[block]))
        {
            die("parallel weight interval changed");
        }
    }

    printf(
        "{\"benchmark\":\"arb_weight_intervals\","
        "\"cutoff\":%llu,\"block_count\":%zu,\"workers\":%u,"
        "\"serial_seconds\":%.6f,\"native_seconds\":%.6f,"
        "\"wall_seconds\":%.6f,\"speedup\":%.6f,"
        "\"byte_identical\":true}\n",
        (unsigned long long) cutoff,
        block_count,
        workers,
        serial_seconds,
        stats.elapsed_seconds,
        wall_seconds_value,
        serial_seconds / wall_seconds_value);

    free(left);
    free(right);
    free(expected_lower);
    free(expected_upper);
    free(actual_lower);
    free(actual_upper);
    arb_clear(heat);
    arb_clear(height);
    arb_clear(sigma_exact);
    arb_clear(sigma_lower);
    arb_clear(sigma_upper);
    arb_clear(temporary);
    arf_clear(lower_bound);
    arf_clear(upper_bound);
    flint_cleanup();
    return 0;
}
