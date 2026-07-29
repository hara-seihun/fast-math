#include "fast_math_arb.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int
same_double(double left, double right)
{
    uint64_t left_bits, right_bits;

    memcpy(&left_bits, &left, sizeof(left_bits));
    memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

int
main(void)
{
    const uint64_t left[] = {2, 11, 101, 1001, 10001};
    const uint64_t right[] = {10, 100, 1000, 10000, 100000};
    const size_t count = sizeof(left) / sizeof(left[0]);
    const slong precision = 256;
    uint32_t worker_count;
    size_t block;
    double expected_lower[count], expected_upper[count];
    double actual_lower[count], actual_upper[count];
    char error[128];
    fast_math_arb_weight_interval_stats stats;
    arb_t sigma_lower, sigma_upper, base, exponent, weight;
    arf_t lower_bound, upper_bound;

    arb_init(sigma_lower);
    arb_init(sigma_upper);
    arb_init(base);
    arb_init(exponent);
    arb_init(weight);
    arf_init(lower_bound);
    arf_init(upper_bound);
    arb_set_d(sigma_lower, 1.5029396177866938);
    arb_set_d(sigma_upper, 1.5029396177866941);

    for (block = 0; block < count; block++)
    {
        arb_set_ui(base, (ulong) left[block]);
        arb_neg(exponent, sigma_lower);
        arb_pow(weight, base, exponent, precision);
        arb_get_ubound_arf(upper_bound, weight, precision);
        expected_upper[block] =
            arf_get_d(upper_bound, ARF_RND_CEIL);

        arb_set_ui(base, (ulong) right[block]);
        arb_neg(exponent, sigma_upper);
        arb_pow(weight, base, exponent, precision);
        arb_get_lbound_arf(lower_bound, weight, precision);
        expected_lower[block] =
            arf_get_d(lower_bound, ARF_RND_FLOOR);
    }

    for (worker_count = 1; worker_count <= 3; worker_count++)
    {
        if (fast_math_arb_weight_intervals_u64(
                left,
                right,
                count,
                sigma_lower,
                sigma_upper,
                precision,
                worker_count,
                2,
                actual_lower,
                actual_upper,
                &stats,
                error,
                sizeof(error)) != 0)
        {
            fprintf(stderr, "weight interval call failed: %s\n", error);
            return 1;
        }
        if (stats.block_count != count || stats.worker_count == 0)
            return 1;
        for (block = 0; block < count; block++)
        {
            if (
                !same_double(expected_lower[block], actual_lower[block])
                || !same_double(
                    expected_upper[block], actual_upper[block]))
            {
                fprintf(stderr, "weight interval changed at %zu\n", block);
                return 1;
            }
        }
    }

    if (fast_math_arb_weight_intervals_u64(
            left,
            right,
            count,
            sigma_lower,
            sigma_upper,
            precision,
            1,
            0,
            actual_lower,
            actual_upper,
            &stats,
            error,
            sizeof(error)) == 0)
    {
        return 1;
    }

    arb_clear(sigma_lower);
    arb_clear(sigma_upper);
    arb_clear(base);
    arb_clear(exponent);
    arb_clear(weight);
    arf_clear(lower_bound);
    arf_clear(upper_bound);
    flint_cleanup();
    return 0;
}
