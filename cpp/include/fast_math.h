#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define FAST_MATH_API __declspec(dllexport)
#else
#define FAST_MATH_API
#endif

extern "C" {

struct fast_math_segment_stats {
  std::uint64_t sample_count;
  std::uint64_t segment_count;
  double elapsed_seconds;
};

struct fast_math_taylor_stats {
  std::uint64_t sample_count;
  std::uint32_t order_count;
  double elapsed_seconds;
};

struct fast_math_filon_stats {
  std::uint64_t correlation_count;
  std::uint64_t output_count;
  std::uint64_t exact_count;
  std::uint64_t tail_count;
  std::uint64_t chunk_count;
  std::uint32_t term_count;
  std::uint32_t thread_count;
  double elapsed_seconds;
};

struct fast_math_graph_stats {
  std::uint64_t graph_count;
  std::uint32_t vertex_count;
  std::uint64_t pair_count;
  std::uint64_t nodes_visited;
  double elapsed_seconds;
};

struct fast_math_graph_profile_stats {
  std::uint64_t graph_count;
  std::uint32_t vertex_count;
  std::uint32_t induced_order;
  std::uint32_t class_count;
  std::uint64_t subsets_per_graph;
  double elapsed_seconds;
};

struct fast_math_graph_profile_stack_stats {
  std::uint64_t graph_count;
  std::uint32_t vertex_count;
  std::uint32_t order_count;
  std::uint64_t field_count;
  std::uint64_t subsets_per_graph;
  double elapsed_seconds;
};

struct fast_math_large_graph_stats {
  std::uint64_t vertex_count;
  std::uint64_t directed_edge_count;
  std::uint64_t intersection_steps;
  std::uint64_t triangle_count;
  double elapsed_seconds;
};

struct fast_math_common_neighbor_stats {
  std::uint64_t vertex_count;
  std::uint64_t directed_edge_count;
  std::uint64_t pair_count;
  std::uint64_t intersection_steps;
  std::uint64_t common_neighbor_count;
  double elapsed_seconds;
};

struct fast_math_canonical_graph_stats {
  std::uint64_t graph_count;
  std::uint32_t vertex_count;
  std::uint32_t word_count;
  std::uint64_t search_nodes;
  double elapsed_seconds;
};

struct fast_math_digest_stats {
  std::uint64_t row_count;
  std::uint64_t field_count;
  double elapsed_seconds;
};

struct fast_math_base_p_stats {
  std::uint64_t element_count;
  std::uint64_t class_count;
  double elapsed_seconds;
};

struct fast_math_fp_span_stats {
  std::uint64_t span_count;
  std::uint64_t point_count;
  std::uint64_t query_count;
  std::uint64_t rank_sum;
  double elapsed_seconds;
};

struct fast_math_union_stats {
  std::uint64_t family_count;
  std::uint64_t pair_checks;
  double elapsed_seconds;
};

struct fast_math_colex_stats {
  std::uint64_t subset_count;
  std::uint64_t binomial_evaluations;
  std::uint64_t newly_visited;
  double elapsed_seconds;
};

struct fast_math_elliptic_stats {
  std::uint64_t prime_count;
  std::uint64_t parameter_count;
  std::uint64_t candidate_count;
  std::uint32_t truncated;
  double elapsed_seconds;
};

struct fast_math_adaptive_stats {
  std::uint64_t target_count;
  std::uint64_t restriction_count;
  std::uint64_t coordinate_count;
  std::uint64_t worker_count;
  double elapsed_seconds;
};

struct fast_math_modular_stats {
  std::uint64_t batch_count;
  std::uint64_t item_count;
  std::uint64_t operation_count;
  std::uint32_t prime;
  std::uint32_t thread_count;
  double elapsed_seconds;
};

struct fast_math_modular_linear_stats {
  std::uint64_t row_count;
  std::uint64_t column_count;
  std::uint64_t rank;
  std::uint64_t batch_count;
  std::uint64_t operation_count;
  std::uint32_t prime;
  std::uint32_t thread_count;
  double elapsed_seconds;
};

struct fast_math_cnf_stats {
  std::uint32_t variable_count;
  std::uint64_t clause_count;
  std::uint64_t literal_count;
  std::uint64_t assignment_count;
  std::uint64_t inspected_literal_count;
  std::uint32_t thread_count;
  double elapsed_seconds;
};

struct fast_math_sparse_rank_stats {
  std::uint64_t row_count;
  std::uint64_t column_count;
  std::uint64_t input_nonzeros;
  std::uint64_t active_rows;
  std::uint64_t processed_rows;
  std::uint64_t dependent_rows;
  std::uint64_t rank;
  std::uint64_t elimination_steps;
  std::uint64_t basis_nonzeros;
  std::uint64_t maximum_basis_size;
  std::uint64_t maximum_working_size;
  std::uint64_t peeled_pivots;
  std::uint64_t residual_rows;
  std::uint64_t residual_columns;
  std::uint64_t residual_nonzeros;
  std::uint32_t prime;
  std::uint8_t target_reached;
  double preprocessing_seconds;
  double elapsed_seconds;
};

struct fast_math_sparse_rank_batch_stats {
  std::uint64_t prime_count;
  std::uint32_t thread_count;
  double elapsed_seconds;
};

struct fast_math_sparse_block_coloop_stats {
  std::uint64_t row_count;
  std::uint64_t column_count;
  std::uint64_t input_nonzeros;
  std::uint64_t block_count;
  std::uint64_t block_incidences;
  std::uint64_t active_columns;
  std::uint64_t removed_columns;
  std::uint64_t residual_columns;
  std::uint64_t blocks_processed;
  std::uint64_t maximum_block_columns;
  std::uint32_t row_block_size;
  std::uint32_t prime;
  double elapsed_seconds;
};

struct fast_math_group_stats {
  std::uint32_t degree;
  std::uint64_t generator_count;
  std::uint64_t item_count;
  std::uint64_t orbit_count;
  std::uint64_t chain_level_count;
  std::uint64_t strong_generator_count;
  std::uint32_t thread_count;
  double elapsed_seconds;
};

struct fast_math_cnf_plan;
struct fast_math_modular_linear_system;
struct fast_math_permutation_group;
struct fast_math_subset_action;

struct fast_math_ci_stats {
  std::uint32_t degree;
  std::uint64_t generator_count;
  std::uint64_t item_count;
  std::uint64_t class_count;
  std::uint64_t relation_count;
  std::uint64_t iteration_count;
  std::uint32_t thread_count;
  double elapsed_seconds;
};

struct fast_math_square_cover_stats {
  std::uint64_t point_count;
  std::uint64_t pose_count;
  std::uint64_t word_count;
  std::uint64_t incidence_tests;
  std::uint32_t thread_count;
  std::uint32_t simd_lanes;
  double elapsed_seconds;
};

FAST_MATH_API const char* fast_math_version();

FAST_MATH_API int fast_math_square_cover_words_f64(
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
    std::size_t error_message_size);

FAST_MATH_API int fast_math_square_weighted_scores_f64(
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
    std::size_t error_message_size);

FAST_MATH_API int fast_math_permutation_group_create_u32(
    const std::uint32_t* generators,
    std::size_t generator_count,
    std::uint32_t degree,
    fast_math_permutation_group** group,
    fast_math_group_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API void fast_math_permutation_group_destroy(
    fast_math_permutation_group* group);

FAST_MATH_API int fast_math_permutation_group_summary_u32(
    const fast_math_permutation_group* group,
    std::size_t base_capacity,
    std::uint32_t* base_points,
    std::uint32_t* orbit_sizes,
    std::uint32_t* point_orbit_labels,
    std::uint32_t* point_orbit_count,
    fast_math_group_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_permutation_group_plan_contains_u32(
    const fast_math_permutation_group* group,
    const std::uint32_t* elements,
    std::size_t element_count,
    std::uint32_t thread_count,
    std::uint8_t* contains,
    fast_math_group_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_permutation_orbits_u32(
    const std::uint32_t* generators,
    std::size_t generator_count,
    std::uint32_t degree,
    std::uint32_t* orbit_labels,
    std::uint32_t* orbit_count,
    fast_math_group_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_schreier_sims_u32(
    const std::uint32_t* generators,
    std::size_t generator_count,
    std::uint32_t degree,
    std::size_t base_capacity,
    std::uint32_t* base_points,
    std::uint32_t* orbit_sizes,
    std::uint64_t* level_generator_offsets,
    std::size_t strong_generator_capacity,
    std::uint32_t* strong_generators,
    std::uint64_t* base_count,
    std::uint64_t* strong_generator_count,
    fast_math_group_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_permutation_group_contains_u32(
    const std::uint32_t* generators,
    std::size_t generator_count,
    std::uint32_t degree,
    const std::uint32_t* elements,
    std::size_t element_count,
    std::uint32_t thread_count,
    std::uint8_t* contains,
    fast_math_group_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_subset_action_create_u32(
    const std::uint32_t* permutations,
    std::size_t permutation_count,
    std::uint32_t degree,
    fast_math_subset_action** plan,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API void fast_math_subset_action_destroy(
    fast_math_subset_action* plan);

FAST_MATH_API int fast_math_subset_action_canonicalize_u64(
    const fast_math_subset_action* plan,
    const std::uint64_t* masks,
    std::size_t mask_count,
    std::uint32_t thread_count,
    std::uint64_t* canonical_masks,
    std::uint8_t* is_canonical,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_permutation_double_cosets_u32(
    const std::uint32_t* candidates,
    std::size_t candidate_count,
    const std::uint32_t* left_generators,
    std::size_t left_generator_count,
    const std::uint32_t* right_generators,
    std::size_t right_generator_count,
    std::uint32_t degree,
    std::uint64_t* class_ids,
    std::uint64_t* representative_indices,
    std::uint64_t* class_sizes,
    std::uint64_t* class_count,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_subset_orbits_u64(
    const std::uint64_t* subset_words,
    std::size_t subset_count,
    std::uint32_t word_count,
    std::uint32_t atom_count,
    const std::uint32_t* action_generators,
    std::size_t generator_count,
    std::uint64_t* class_ids,
    std::uint64_t* representative_indices,
    std::uint64_t* class_sizes,
    std::uint64_t* class_count,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_subset_orbits_v2_u64(
    const std::uint64_t* subset_words,
    std::size_t subset_count,
    std::uint32_t word_count,
    std::uint32_t atom_count,
    const std::uint32_t* action_generators,
    std::size_t generator_count,
    std::uint32_t action_mode,
    std::uint64_t* class_ids,
    std::uint64_t* representative_indices,
    std::uint64_t* class_sizes,
    std::uint64_t* class_count,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_fixed_weight_subset_orbits_u64(
    const std::uint32_t* complete_action,
    std::size_t action_count,
    std::uint32_t atom_count,
    std::uint32_t subset_weight,
    std::uint64_t max_subset_count,
    std::uint64_t* representative_masks,
    std::uint64_t representative_capacity,
    std::uint64_t* orbit_sizes,
    std::uint64_t* representative_count,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_expand_atom_subsets_u64(
    const std::uint64_t* subset_words,
    std::size_t subset_count,
    std::uint32_t subset_word_count,
    std::uint32_t atom_count,
    const std::uint64_t* atom_offsets,
    const std::uint32_t* atom_elements,
    std::size_t atom_element_count,
    std::uint32_t group_order,
    std::uint32_t thread_count,
    std::uint64_t* element_words,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_cayley_graphs_u32(
    const std::uint32_t* multiplication_table,
    std::uint32_t group_order,
    const std::uint64_t* connection_words,
    std::size_t connection_count,
    std::uint32_t word_count,
    std::uint32_t thread_count,
    std::uint64_t* adjacency_words,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_derivative_orbits_u32(
    const std::uint32_t* multiplication_table,
    const std::uint32_t* inverse_indices,
    const std::uint32_t* bijection,
    std::uint32_t group_order,
    std::uint32_t* derivative_generators,
    std::uint32_t* orbit_labels,
    std::uint32_t* orbit_count,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_wl2_refine_u32(
    const std::uint32_t* initial_relations,
    std::uint32_t vertex_count,
    std::uint32_t* stable_relations,
    std::uint32_t* relation_count,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_intersection_numbers_u64(
    const std::uint32_t* relations,
    std::uint32_t vertex_count,
    std::uint32_t relation_count,
    std::uint64_t* intersection_numbers,
    fast_math_ci_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_segmented_complex_stats_f64(
    const double* values_interleaved,
    std::size_t sample_count,
    const std::uint64_t* offsets,
    std::size_t segment_count,
    std::uint32_t thread_count,
    double* sums_interleaved,
    double* l1,
    double* variation,
    fast_math_segment_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_taylor_coefficients_f64(
    const double* base_interleaved,
    const double* logarithms,
    std::size_t sample_count,
    std::uint32_t maximum_order,
    std::uint64_t chunk_size,
    std::uint32_t thread_count,
    double* coefficients_interleaved,
    fast_math_taylor_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_taylor_evaluate_f64(
    const double* basis_interleaved,
    const double* delta_interleaved,
    std::size_t sample_count,
    std::uint32_t order_count,
    std::uint64_t chunk_size,
    std::uint32_t thread_count,
    double* values_interleaved,
    double* log_moments_interleaved,
    fast_math_taylor_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_filon_chebyshev_inner_product_f64(
    const double* correlation_interleaved,
    std::size_t correlation_count,
    const double* exact_weights_interleaved,
    std::size_t exact_count,
    const double* positive_endpoint_derivatives,
    const double* negative_endpoint_derivatives,
    std::uint32_t term_count,
    std::size_t output_count,
    double eta,
    double length,
    bool conjugate_kernel,
    std::uint64_t chunk_size,
    std::uint32_t thread_count,
    double* result_interleaved,
    fast_math_filon_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_graph_pair_profiles_u64(
    const std::uint64_t* adjacency_masks,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    std::uint32_t thread_count,
    std::uint8_t* adjacent,
    std::uint32_t* common_neighbors,
    std::uint32_t* common_nonneighbors,
    std::uint32_t* only_left,
    std::uint32_t* only_right,
    fast_math_graph_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_graph_find_clique_u64(
    const std::uint64_t* adjacency_masks,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    std::uint32_t order,
    bool complement,
    std::uint32_t thread_count,
    std::uint64_t* witnesses,
    std::uint64_t* nodes_visited,
    fast_math_graph_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_graph6_decode_u64(
    const std::uint8_t* data,
    std::size_t data_size,
    const std::uint64_t* offsets,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    std::uint32_t thread_count,
    std::uint64_t* adjacency_masks,
    fast_math_graph_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_graph6_encode_u64(
    const std::uint64_t* adjacency_masks,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    std::uint32_t thread_count,
    std::uint8_t* data,
    std::size_t data_size,
    fast_math_graph_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_graph_delete_vertices_u64(
    const std::uint64_t* adjacency_masks,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    const std::uint64_t* source_graphs,
    const std::uint32_t* deleted_vertices,
    std::size_t request_count,
    std::uint32_t thread_count,
    std::uint64_t* output_adjacency_masks,
    fast_math_graph_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_graph_invariants_u64(
    const std::uint64_t* adjacency_masks,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    std::uint32_t thread_count,
    std::uint32_t* degrees,
    std::uint64_t* edge_counts,
    std::uint64_t* triangle_counts,
    std::uint64_t* wedge_counts,
    std::uint64_t* induced_path3_counts,
    fast_math_graph_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_graph_induced_profiles_u64(
    const std::uint64_t* adjacency_masks,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    std::uint32_t induced_order,
    const std::uint32_t* class_lookup,
    std::size_t lookup_size,
    std::uint32_t class_count,
    std::uint32_t thread_count,
    std::uint64_t* counts,
    fast_math_graph_profile_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_graph_induced_profile_stack_u64(
    const std::uint64_t* adjacency_masks,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    const std::uint32_t* induced_orders,
    std::size_t order_count,
    const std::uint32_t* class_lookups,
    const std::uint64_t* lookup_offsets,
    const std::uint32_t* class_counts,
    std::uint32_t thread_count,
    std::uint64_t* counts,
    fast_math_graph_profile_stack_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_graph_triangles_csr_u32(
    const std::uint64_t* row_offsets,
    const std::uint32_t* column_indices,
    const std::uint64_t* edge_color_masks,
    const std::uint64_t* vertex_loop_color_masks,
    std::size_t vertex_count,
    std::size_t directed_edge_count,
    std::size_t triangle_capacity,
    std::uint32_t* triangles,
    std::uint64_t* triangle_edge_color_masks,
    std::uint64_t* triangle_count,
    fast_math_large_graph_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_graph_common_neighbors_csr_u32(
    const std::uint64_t* row_offsets,
    const std::uint32_t* column_indices,
    std::size_t vertex_count,
    std::size_t directed_edge_count,
    const std::uint32_t* pairs,
    std::size_t pair_count,
    std::size_t common_neighbor_capacity,
    std::uint64_t* pair_offsets,
    std::uint32_t* common_neighbors,
    std::uint64_t* common_neighbor_count,
    fast_math_common_neighbor_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_canonical_digraphs_nauty_u64(
    const std::uint64_t* adjacency_words,
    const std::uint32_t* vertex_colors,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    std::uint32_t word_count,
    std::uint32_t* canonical_permutations,
    std::uint64_t* canonical_adjacency_words,
    std::uint32_t* canonical_vertex_colors,
    double* automorphism_group_mantissas,
    std::int32_t* automorphism_group_exponents,
    std::uint32_t* orbit_counts,
    std::uint64_t* generator_offsets,
    std::size_t generator_capacity,
    std::uint32_t* generator_permutations,
    std::uint64_t* generator_count,
    fast_math_canonical_graph_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_canonical_digraphs_nauty_v2_u64(
    const std::uint64_t* adjacency_words,
    const std::uint32_t* vertex_colors,
    std::size_t graph_count,
    std::uint32_t vertex_count,
    std::uint32_t word_count,
    std::uint32_t thread_count,
    std::uint8_t collect_automorphism_generators,
    std::uint32_t* canonical_permutations,
    std::uint64_t* canonical_adjacency_words,
    std::uint32_t* canonical_vertex_colors,
    double* automorphism_group_mantissas,
    std::int32_t* automorphism_group_exponents,
    std::uint32_t* orbit_counts,
    std::uint64_t* generator_offsets,
    std::size_t generator_capacity,
    std::uint32_t* generator_permutations,
    std::uint64_t* generator_count,
    fast_math_canonical_graph_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_digest_u64_rows_sha256(
    const std::uint64_t* rows,
    std::size_t row_count,
    std::size_t field_count,
    const std::uint8_t* namespace_data,
    std::size_t namespace_size,
    std::uint32_t thread_count,
    std::uint8_t* digests,
    fast_math_digest_stats* stats,
    char* error_message,
    std::size_t error_message_size);

// Batched index/digit codec and scalar/negation class representatives over
// encoded F_p^n points. Digits are little-endian uint8 rows (digit zero is
// the p^0 coefficient); codes are little-endian base-p integers. The scalar
// canonical form scales the least-significant nonzero digit to one; the
// negation canonical form is the smaller of a code and its digit-wise
// negation. Class ids are dense from zero in ascending representative order,
// with the zero vector as class zero.
FAST_MATH_API int fast_math_base_p_digits_u64(
    const std::uint64_t* codes,
    std::size_t code_count,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint8_t* digits,
    fast_math_base_p_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_base_p_codes_u64(
    const std::uint8_t* digits,
    std::size_t code_count,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint64_t* codes,
    fast_math_base_p_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_base_p_negation_codes_u64(
    const std::uint64_t* codes,
    std::size_t code_count,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint64_t* negated,
    fast_math_base_p_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_base_p_scalar_normals_u64(
    const std::uint64_t* codes,
    std::size_t code_count,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint64_t* normals,
    fast_math_base_p_stats* stats,
    char* error_message,
    std::size_t error_message_size);

// Fills one class id per point of the whole p^width space plus the dense
// representative and member-count tables. class_kind selects negation
// classes (zero) or scalar classes (one). representative_capacity bounds the
// representative/count buffers and must cover the exact class count:
// (p^width + 1) / 2 for negation when p is odd (p^width when p is two), and
// (p^width - 1) / (p - 1) + 1 for scalar classes.
FAST_MATH_API int fast_math_base_p_class_table_u64(
    std::uint32_t prime,
    std::uint32_t width,
    std::uint32_t class_kind,
    std::uint32_t* class_ids,
    std::uint64_t* representatives,
    std::uint32_t* class_counts,
    std::size_t representative_capacity,
    fast_math_base_p_stats* stats,
    char* error_message,
    std::size_t error_message_size);

// Exact spans of encoded F_p^width points, using the base-p codec convention.
// The ragged rank API takes span_count + 1 offsets. The richer one-span API
// returns canonical RREF rows, the input points that increased rank, query
// coefficients against the RREF, and canonical quotient residual codes.
FAST_MATH_API int fast_math_fp_span_ranks_u64(
    const std::uint64_t* point_codes,
    std::size_t point_count,
    const std::uint64_t* span_offsets,
    std::size_t span_count,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint32_t* ranks,
    fast_math_fp_span_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_fp_point_span_u64(
    const std::uint64_t* point_codes,
    std::size_t point_count,
    const std::uint64_t* query_codes,
    std::size_t query_count,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint64_t* pivot_indices,
    std::uint32_t* pivot_columns,
    std::uint64_t* reduced_basis_codes,
    std::uint8_t* independent_points,
    std::uint8_t* query_members,
    std::uint32_t* query_coordinates,
    std::uint64_t* query_quotient_codes,
    fast_math_fp_span_stats* stats,
    char* error_message,
    std::size_t error_message_size);

// Colexicographical ranking for subsets packed as uint64 element masks.
// The rank of the subset {c_1 < c_2 < ... < c_k} is C(c_1, 1) + ... +
// C(c_k, k), a bijection between the weight-k subsets of
// {0, ..., element_count - 1} and [0, C(element_count, k)].
// element_count must be between 1 and 64.
FAST_MATH_API int fast_math_colex_rank_u64(
    const std::uint64_t* subset_masks,
    std::size_t subset_count,
    std::uint32_t element_count,
    std::uint64_t* ranks,
    fast_math_colex_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_colex_unrank_u64(
    const std::uint64_t* ranks,
    std::size_t rank_count,
    std::uint32_t element_count,
    std::uint32_t weight,
    std::uint64_t* subset_masks,
    fast_math_colex_stats* stats,
    char* error_message,
    std::size_t error_message_size);

// Marks the ranks of fixed-weight subsets in a caller-owned visited bitmap
// of ceil(C(element_count, weight) / 64) words, returning one flag per
// subset that is 1 when its rank was not already set. The bitmap is updated
// in place; all subsets must have exactly `weight` elements.
FAST_MATH_API int fast_math_colex_visit_u64(
    const std::uint64_t* subset_masks,
    std::size_t subset_count,
    std::uint32_t element_count,
    std::uint32_t weight,
    std::uint64_t* visited_words,
    std::size_t visited_word_count,
    std::uint8_t* newly_visited,
    fast_math_colex_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_union_closed_family_masks_u64(
    const std::uint64_t* family_masks,
    std::size_t family_count,
    std::uint32_t ground_size,
    std::uint8_t* closed,
    fast_math_union_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_elliptic_mestre_ap_tables_i32(
    const std::int64_t* sextuple,
    const std::uint32_t* primes,
    std::size_t prime_count,
    const std::uint64_t* table_offsets,
    std::uint32_t thread_count,
    std::int32_t* tables,
    fast_math_elliptic_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_elliptic_nagao_scores_f64(
    const std::int32_t* tables,
    const std::uint32_t* primes,
    const double* weights,
    std::size_t prime_count,
    const std::uint64_t* table_offsets,
    const std::int64_t* numerators,
    const std::int64_t* denominators,
    std::size_t parameter_count,
    std::uint32_t thread_count,
    double* scores,
    fast_math_elliptic_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_elliptic_quartic_sieve_i64(
    const std::uint32_t* coefficient_residues,
    const std::uint32_t* primes,
    std::size_t prime_count,
    std::int64_t numerator_low,
    std::int64_t numerator_high,
    std::int64_t denominator_low,
    std::int64_t denominator_high,
    std::uint32_t thread_count,
    std::int64_t* candidates,
    std::size_t candidate_capacity,
    fast_math_elliptic_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_adaptive_area_f64(
    const double* tables,
    std::size_t target_count,
    std::uint32_t coordinate_count,
    double zero_tolerance,
    std::uint32_t thread_count,
    double* areas,
    std::int32_t* first_coordinates,
    double* variances,
    double* areas_by_restriction,
    std::int8_t* policies,
    fast_math_adaptive_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_adaptive_area_exact_i64(
    const std::int64_t* tables,
    std::size_t target_count,
    std::uint32_t coordinate_count,
    std::uint32_t thread_count,
    std::int64_t* areas,
    std::int32_t* first_coordinates,
    std::int64_t* variances,
    std::int64_t* areas_by_restriction,
    std::int8_t* policies,
    fast_math_adaptive_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_cnf_create_i32(
    const std::uint64_t* clause_offsets,
    std::size_t clause_count,
    const std::int32_t* literals,
    std::size_t literal_count,
    std::uint32_t variable_count,
    fast_math_cnf_plan** plan,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API void fast_math_cnf_destroy(fast_math_cnf_plan* plan);

FAST_MATH_API int fast_math_cnf_evaluate_u64(
    const fast_math_cnf_plan* plan,
    const std::uint64_t* assignment_words,
    std::size_t assignment_count,
    std::uint32_t word_count,
    std::uint32_t thread_count,
    std::uint8_t* satisfied,
    std::int64_t* first_unsatisfied_clause,
    fast_math_cnf_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_polynomial_evaluate_mod_u32(
    const std::uint32_t* coefficients,
    std::size_t polynomial_count,
    std::size_t coefficient_count,
    const std::uint32_t* points,
    std::size_t point_count,
    std::uint32_t prime,
    std::uint32_t thread_count,
    std::uint32_t* values,
    std::uint32_t* derivatives,
    fast_math_modular_stats* stats,
    char* error_message,
    std::size_t error_message_size);

/* Shift-divisor gate scan (see python/fast_math/shift_gates.py).  Survivors
 * are the v in [v_start, v_start + v_count) lying in a wheel class, alive in
 * every packed lookup table, whose every linear form a*v - b has a
 * smooth-prime-free part that is prime or the square of a prime.  stats (may
 * be NULL) receives {scanned, sieve_candidates, lut_alive, survivors}. */
FAST_MATH_API int fast_math_shift_gate_scan_u64(
    const uint64_t* form_a,
    const uint64_t* form_b,
    size_t form_count,
    const uint32_t* smooth_primes,
    size_t smooth_prime_count,
    const uint64_t* lut_moduli,
    const uint64_t* lut_offsets,
    size_t lut_count,
    const uint64_t* lut_bits,
    uint64_t wheel,
    const uint64_t* wheel_classes,
    size_t wheel_class_count,
    uint32_t sieve_low,
    uint32_t sieve_bound,
    uint64_t v_start,
    uint64_t v_count,
    uint32_t thread_count,
    uint64_t* survivors,
    size_t survivor_capacity,
    size_t* survivor_count,
    uint64_t* stats,
    char* error_message,
    size_t error_capacity);

FAST_MATH_API int fast_math_determinants_mod_u32(
    const std::uint32_t* matrices,
    std::size_t matrix_count,
    std::uint32_t order,
    std::uint32_t prime,
    std::uint32_t thread_count,
    std::uint32_t* determinants,
    fast_math_modular_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_modular_linear_system_create_u32(
    const std::uint32_t* matrix,
    std::size_t row_count,
    std::size_t column_count,
    std::uint32_t prime,
    fast_math_modular_linear_system** system,
    fast_math_modular_linear_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API void fast_math_modular_linear_system_destroy(
    fast_math_modular_linear_system* system);

FAST_MATH_API int fast_math_modular_linear_system_export_u32(
    const fast_math_modular_linear_system* system,
    std::uint32_t* reduced_row_echelon,
    std::uint32_t* pivot_columns,
    std::uint32_t* solution_operator,
    std::uint32_t* right_nullspace,
    std::uint32_t* left_nullspace,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_modular_linear_system_solve_u32(
    const fast_math_modular_linear_system* system,
    const std::uint32_t* right_hand_sides,
    std::size_t right_hand_side_count,
    std::uint32_t thread_count,
    std::uint32_t* solutions,
    std::int64_t* inconsistency_rows,
    fast_math_modular_linear_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_sparse_rank_mod_u32(
    const std::uint64_t* row_offsets,
    const std::uint32_t* column_indices,
    const std::uint32_t* values,
    std::size_t row_count,
    std::size_t column_count,
    std::size_t nonzero_count,
    std::uint32_t prime,
    std::size_t target_rank,
    std::uint64_t* pivot_rows,
    std::uint32_t* pivot_columns,
    std::size_t pivot_capacity,
    fast_math_sparse_rank_stats* stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_sparse_rank_mod_u32_batch(
    const std::uint64_t* row_offsets,
    const std::uint32_t* column_indices,
    const std::uint32_t* values_by_prime,
    std::size_t row_count,
    std::size_t column_count,
    std::size_t nonzero_count,
    const std::uint32_t* primes,
    std::size_t prime_count,
    std::size_t target_rank,
    std::uint32_t thread_count,
    std::uint64_t* pivot_rows,
    std::uint32_t* pivot_columns,
    std::size_t pivot_capacity,
    fast_math_sparse_rank_stats* stats,
    fast_math_sparse_rank_batch_stats* batch_stats,
    char* error_message,
    std::size_t error_message_size);

FAST_MATH_API int fast_math_sparse_block_coloops_mod_u32(
    const std::uint64_t* row_offsets,
    const std::uint32_t* column_indices,
    const std::uint32_t* values,
    std::size_t row_count,
    std::size_t column_count,
    std::size_t nonzero_count,
    std::uint32_t prime,
    std::uint32_t row_block_size,
    std::uint8_t* residual_columns,
    std::size_t residual_capacity,
    std::uint32_t* removed_columns,
    std::uint64_t* certificate_row_starts,
    std::uint32_t* certificate_coefficients,
    std::size_t removed_capacity,
    fast_math_sparse_block_coloop_stats* stats,
    char* error_message,
    std::size_t error_message_size);

}
