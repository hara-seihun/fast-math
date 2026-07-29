#include "fast_math_arb.h"

void
fast_math_arb_cache_workspace_init(
    fast_math_arb_cache_workspace *workspace)
{
    arb_init(workspace->center);
    arb_init(workspace->difference);
    arb_init(workspace->weight);
    acb_init(workspace->complex_center);
    acb_init(workspace->complex_difference);
}

void
fast_math_arb_cache_workspace_clear(
    fast_math_arb_cache_workspace *workspace)
{
    arb_clear(workspace->center);
    arb_clear(workspace->difference);
    arb_clear(workspace->weight);
    acb_clear(workspace->complex_center);
    acb_clear(workspace->complex_difference);
}

void
fast_math_arb_weight_from_log(
    arb_t result,
    const arb_t log_index,
    const arb_t sigma,
    slong precision)
{
    arb_mul(result, sigma, log_index, precision);
    arb_neg(result, result);
    arb_exp(result, result, precision);
}

double
fast_math_arb_cache_real_term(
    arb_t weighted_error,
    const arb_t exact,
    const arb_t weight,
    fast_math_arb_cache_workspace *workspace,
    slong precision)
{
    double center_value = arf_get_d(arb_midref(exact), ARF_RND_NEAR);

    arb_set_d(workspace->center, center_value);
    arb_sub(weighted_error, exact, workspace->center, precision);
    arb_abs(weighted_error, weighted_error);
    arb_mul(
        weighted_error,
        weighted_error,
        weight,
        precision);
    return center_value;
}

double
fast_math_arb_cache_real(
    arb_t error_sum,
    const arb_t exact,
    const arb_t weight,
    fast_math_arb_cache_workspace *workspace,
    slong precision)
{
    double center_value = fast_math_arb_cache_real_term(
        workspace->difference,
        exact,
        weight,
        workspace,
        precision);

    arb_add(error_sum, error_sum, workspace->difference, precision);
    return center_value;
}

double
fast_math_arb_cache_real_from_log(
    arb_t error_sum,
    const arb_t exact,
    const arb_t log_index,
    const arb_t sigma,
    fast_math_arb_cache_workspace *workspace,
    slong precision)
{
    fast_math_arb_weight_from_log(
        workspace->weight, log_index, sigma, precision);
    return fast_math_arb_cache_real(
        error_sum,
        exact,
        workspace->weight,
        workspace,
        precision);
}

void
fast_math_arb_cache_complex_term(
    double *real_center,
    double *imaginary_center,
    arb_t weighted_error,
    const acb_t exact,
    const arb_t weight,
    fast_math_arb_cache_workspace *workspace,
    slong precision)
{
    *real_center = arf_get_d(
        arb_midref(acb_realref(exact)), ARF_RND_NEAR);
    *imaginary_center = arf_get_d(
        arb_midref(acb_imagref(exact)), ARF_RND_NEAR);
    acb_zero(workspace->complex_center);
    arb_set_d(acb_realref(workspace->complex_center), *real_center);
    arb_set_d(acb_imagref(workspace->complex_center), *imaginary_center);
    acb_sub(
        workspace->complex_difference,
        exact,
        workspace->complex_center,
        precision);
    acb_abs(
        weighted_error,
        workspace->complex_difference,
        precision);
    arb_mul(
        weighted_error,
        weighted_error,
        weight,
        precision);
}

void
fast_math_arb_cache_complex(
    double *real_center,
    double *imaginary_center,
    arb_t error_sum,
    const acb_t exact,
    const arb_t weight,
    fast_math_arb_cache_workspace *workspace,
    slong precision)
{
    fast_math_arb_cache_complex_term(
        real_center,
        imaginary_center,
        workspace->difference,
        exact,
        weight,
        workspace,
        precision);
    arb_add(error_sum, error_sum, workspace->difference, precision);
}

void
fast_math_arb_cache_complex_from_log(
    double *real_center,
    double *imaginary_center,
    arb_t error_sum,
    const acb_t exact,
    const arb_t log_index,
    const arb_t sigma,
    fast_math_arb_cache_workspace *workspace,
    slong precision)
{
    fast_math_arb_weight_from_log(
        workspace->weight, log_index, sigma, precision);
    fast_math_arb_cache_complex(
        real_center,
        imaginary_center,
        error_sum,
        exact,
        workspace->weight,
        workspace,
        precision);
}
