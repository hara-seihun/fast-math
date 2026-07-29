#include "fast_math_arb.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int
same_double(double left, double right)
{
    uint64_t left_bits;
    uint64_t right_bits;

    memcpy(&left_bits, &left, sizeof(left_bits));
    memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

int
main(void)
{
    const slong precision = 256;
    double real_center;
    double imaginary_center;
    double expected_real;
    double expected_imaginary;
    fast_math_arb_cache_workspace workspace;
    arb_t sigma;
    arb_t log_index;
    arb_t exact_real;
    arb_t scalar;
    arb_t expected_weight;
    arb_t exact_error;
    arb_t term_error;
    arb_t real_error_sum;
    arb_t complex_error_sum;
    acb_t exact_complex;

    fast_math_arb_cache_workspace_init(&workspace);
    arb_init(sigma);
    arb_init(log_index);
    arb_init(exact_real);
    arb_init(scalar);
    arb_init(expected_weight);
    arb_init(exact_error);
    arb_init(term_error);
    arb_init(real_error_sum);
    arb_init(complex_error_sum);
    acb_init(exact_complex);

    arb_set_d(sigma, 1.5555962870835469);
    arb_log_ui(log_index, 690988UL, precision);
    arb_mul(exact_real, log_index, log_index, precision);
    arb_set_d(scalar, 0.0357431688);
    arb_mul(exact_real, exact_real, scalar, precision);
    arb_exp(exact_real, exact_real, precision);

    arb_zero(real_error_sum);
    real_center = fast_math_arb_cache_real_from_log(
        real_error_sum,
        exact_real,
        log_index,
        sigma,
        &workspace,
        precision);
    expected_real = arf_get_d(arb_midref(exact_real), ARF_RND_NEAR);
    if (!same_double(real_center, expected_real))
    {
        fprintf(stderr, "real cache center changed\n");
        return 1;
    }

    fast_math_arb_weight_from_log(
        expected_weight, log_index, sigma, precision);
    arb_zero(real_error_sum);
    real_center = fast_math_arb_cache_real_term(
        term_error,
        exact_real,
        expected_weight,
        &workspace,
        precision);
    if (!same_double(real_center, expected_real))
    {
        fprintf(stderr, "real term cache center changed\n");
        return 1;
    }
    arb_zero(real_error_sum);
    real_center = fast_math_arb_cache_real(
        real_error_sum,
        exact_real,
        expected_weight,
        &workspace,
        precision);
    if (!same_double(real_center, expected_real))
    {
        fprintf(stderr, "preweighted real cache center changed\n");
        return 1;
    }
    if (!arb_equal(real_error_sum, term_error))
    {
        fprintf(stderr, "real term cache error changed\n");
        return 1;
    }
    arb_set_d(exact_error, expected_real);
    arb_sub(exact_error, exact_real, exact_error, precision);
    arb_abs(exact_error, exact_error);
    arb_mul(exact_error, exact_error, expected_weight, precision);
    if (!arb_contains(real_error_sum, exact_error))
    {
        fprintf(stderr, "real cache error does not enclose exact expression\n");
        return 1;
    }

    arb_set(acb_realref(exact_complex), exact_real);
    arb_set_si(scalar, -1);
    arb_mul_2exp_si(scalar, scalar, -5);
    arb_mul(acb_imagref(exact_complex), log_index, scalar, precision);
    expected_real = arf_get_d(
        arb_midref(acb_realref(exact_complex)), ARF_RND_NEAR);
    expected_imaginary = arf_get_d(
        arb_midref(acb_imagref(exact_complex)), ARF_RND_NEAR);
    arb_zero(complex_error_sum);
    fast_math_arb_cache_complex_from_log(
        &real_center,
        &imaginary_center,
        complex_error_sum,
        exact_complex,
        log_index,
        sigma,
        &workspace,
        precision);
    if (
        !same_double(real_center, expected_real)
        || !same_double(imaginary_center, expected_imaginary))
    {
        fprintf(stderr, "complex cache center changed\n");
        return 1;
    }
    arb_zero(complex_error_sum);
    fast_math_arb_cache_complex_term(
        &real_center,
        &imaginary_center,
        term_error,
        exact_complex,
        expected_weight,
        &workspace,
        precision);
    if (
        !same_double(real_center, expected_real)
        || !same_double(imaginary_center, expected_imaginary))
    {
        fprintf(stderr, "complex term cache center changed\n");
        return 1;
    }
    arb_zero(complex_error_sum);
    fast_math_arb_cache_complex(
        &real_center,
        &imaginary_center,
        complex_error_sum,
        exact_complex,
        expected_weight,
        &workspace,
        precision);
    if (
        !same_double(real_center, expected_real)
        || !same_double(imaginary_center, expected_imaginary))
    {
        fprintf(stderr, "preweighted complex cache center changed\n");
        return 1;
    }
    if (!arb_equal(complex_error_sum, term_error))
    {
        fprintf(stderr, "complex term cache error changed\n");
        return 1;
    }
    if (!arb_is_finite(complex_error_sum) || !arb_is_nonnegative(complex_error_sum))
    {
        fprintf(stderr, "complex cache error is invalid\n");
        return 1;
    }

    fast_math_arb_cache_workspace_clear(&workspace);
    arb_clear(sigma);
    arb_clear(log_index);
    arb_clear(exact_real);
    arb_clear(scalar);
    arb_clear(expected_weight);
    arb_clear(exact_error);
    arb_clear(term_error);
    arb_clear(real_error_sum);
    arb_clear(complex_error_sum);
    acb_clear(exact_complex);
    return 0;
}
