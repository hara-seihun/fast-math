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

struct fast_math_union_stats {
  std::uint64_t family_count;
  std::uint64_t pair_checks;
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

FAST_MATH_API int fast_math_union_closed_family_masks_u64(
    const std::uint64_t* family_masks,
    std::size_t family_count,
    std::uint32_t ground_size,
    std::uint8_t* closed,
    fast_math_union_stats* stats,
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
