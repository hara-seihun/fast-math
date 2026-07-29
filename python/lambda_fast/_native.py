from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import sys

import numpy as np
from numpy.typing import NDArray

from ._inputs import PreparedInputs


class NativeUnavailable(RuntimeError):
    pass


class NativeStats(ctypes.Structure):
    _fields_ = [
        ("primary_pairs", ctypes.c_uint64),
        ("transformed_pairs", ctypes.c_uint64),
        ("low_pairs", ctypes.c_uint64),
        ("elapsed_seconds", ctypes.c_double),
    ]


class NativeTwoLevelRecord(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_uint64),
        ("right", ctypes.c_uint64),
        ("fine_piece_count", ctypes.c_uint64),
        ("first_real", ctypes.c_double),
        ("first_imag", ctypes.c_double),
        ("second_real", ctypes.c_double),
        ("second_imag", ctypes.c_double),
        ("center_cost", ctypes.c_double),
        ("weight_variation_upper", ctypes.c_double),
        ("fine_phase_drift_upper", ctypes.c_double),
        ("two_level_upper", ctypes.c_double),
    ]


class NativeTwoLevelStats(ctypes.Structure):
    _fields_ = [
        ("primary_pairs", ctypes.c_uint64),
        ("transformed_pairs", ctypes.c_uint64),
        ("low_pairs", ctypes.c_uint64),
        ("fine_weight_block_count", ctypes.c_uint64),
        ("fine_piece_count", ctypes.c_uint64),
        ("outer_block_count", ctypes.c_uint64),
        ("constant_common_error", ctypes.c_double),
        ("constant_low_error", ctypes.c_double),
        ("center_cost", ctypes.c_double),
        ("weight_variation_upper", ctypes.c_double),
        ("fine_phase_drift_upper", ctypes.c_double),
        ("common_weighted_l1_upper", ctypes.c_double),
        ("low_weighted_l1_upper", ctypes.c_double),
        ("weighted_l1_upper", ctypes.c_double),
        ("two_level_upper", ctypes.c_double),
        ("elapsed_seconds", ctypes.c_double),
    ]


class NativePowerMoment(ctypes.Structure):
    _fields_ = [
        ("power", ctypes.c_uint32),
        ("value", ctypes.c_double),
        ("ordinary", ctypes.c_double),
        ("phase_current", ctypes.c_double),
        ("radial", ctypes.c_double),
    ]


class NativePowerMomentStats(ctypes.Structure):
    _fields_ = [
        ("sample_count", ctypes.c_uint64),
        ("maximum_modulus", ctypes.c_double),
        ("maximum_derivative", ctypes.c_double),
        ("elapsed_seconds", ctypes.c_double),
    ]


class NativeInverseStats(ctypes.Structure):
    _fields_ = [
        ("update_count", ctypes.c_uint64),
        ("elapsed_seconds", ctypes.c_double),
    ]


class NativeSegmentStats(ctypes.Structure):
    _fields_ = [
        ("sample_count", ctypes.c_uint64),
        ("segment_count", ctypes.c_uint64),
        ("elapsed_seconds", ctypes.c_double),
    ]


class NativeTaylorStats(ctypes.Structure):
    _fields_ = [
        ("sample_count", ctypes.c_uint64),
        ("order_count", ctypes.c_uint32),
        ("elapsed_seconds", ctypes.c_double),
    ]


class NativeFilonStats(ctypes.Structure):
    _fields_ = [
        ("correlation_count", ctypes.c_uint64),
        ("output_count", ctypes.c_uint64),
        ("exact_count", ctypes.c_uint64),
        ("tail_count", ctypes.c_uint64),
        ("chunk_count", ctypes.c_uint64),
        ("term_count", ctypes.c_uint32),
        ("thread_count", ctypes.c_uint32),
        ("elapsed_seconds", ctypes.c_double),
    ]


class NativeGraphStats(ctypes.Structure):
    _fields_ = [
        ("graph_count", ctypes.c_uint64),
        ("vertex_count", ctypes.c_uint32),
        ("pair_count", ctypes.c_uint64),
        ("nodes_visited", ctypes.c_uint64),
        ("elapsed_seconds", ctypes.c_double),
    ]


class NativeGraphProfileStats(ctypes.Structure):
    _fields_ = [
        ("graph_count", ctypes.c_uint64),
        ("vertex_count", ctypes.c_uint32),
        ("induced_order", ctypes.c_uint32),
        ("class_count", ctypes.c_uint32),
        ("subsets_per_graph", ctypes.c_uint64),
        ("elapsed_seconds", ctypes.c_double),
    ]


class NativeGraphProfileStackStats(ctypes.Structure):
    _fields_ = [
        ("graph_count", ctypes.c_uint64),
        ("vertex_count", ctypes.c_uint32),
        ("order_count", ctypes.c_uint32),
        ("field_count", ctypes.c_uint64),
        ("subsets_per_graph", ctypes.c_uint64),
        ("elapsed_seconds", ctypes.c_double),
    ]


class NativeLargeGraphStats(ctypes.Structure):
    _fields_ = [
        ("vertex_count", ctypes.c_uint64),
        ("directed_edge_count", ctypes.c_uint64),
        ("intersection_steps", ctypes.c_uint64),
        ("triangle_count", ctypes.c_uint64),
        ("elapsed_seconds", ctypes.c_double),
    ]


class NativeCommonNeighborStats(ctypes.Structure):
    _fields_ = [
        ("vertex_count", ctypes.c_uint64),
        ("directed_edge_count", ctypes.c_uint64),
        ("pair_count", ctypes.c_uint64),
        ("intersection_steps", ctypes.c_uint64),
        ("common_neighbor_count", ctypes.c_uint64),
        ("elapsed_seconds", ctypes.c_double),
    ]


class NativeCanonicalGraphStats(ctypes.Structure):
    _fields_ = [
        ("graph_count", ctypes.c_uint64),
        ("vertex_count", ctypes.c_uint32),
        ("word_count", ctypes.c_uint32),
        ("search_nodes", ctypes.c_uint64),
        ("elapsed_seconds", ctypes.c_double),
    ]


class NativeDigestStats(ctypes.Structure):
    _fields_ = [
        ("row_count", ctypes.c_uint64),
        ("field_count", ctypes.c_uint64),
        ("elapsed_seconds", ctypes.c_double),
    ]


class NativeUnionStats(ctypes.Structure):
    _fields_ = [
        ("family_count", ctypes.c_uint64),
        ("pair_checks", ctypes.c_uint64),
        ("elapsed_seconds", ctypes.c_double),
    ]


class NativeSparseRankStats(ctypes.Structure):
    _fields_ = [
        ("row_count", ctypes.c_uint64),
        ("column_count", ctypes.c_uint64),
        ("input_nonzeros", ctypes.c_uint64),
        ("active_rows", ctypes.c_uint64),
        ("processed_rows", ctypes.c_uint64),
        ("dependent_rows", ctypes.c_uint64),
        ("rank", ctypes.c_uint64),
        ("elimination_steps", ctypes.c_uint64),
        ("basis_nonzeros", ctypes.c_uint64),
        ("maximum_basis_size", ctypes.c_uint64),
        ("maximum_working_size", ctypes.c_uint64),
        ("peeled_pivots", ctypes.c_uint64),
        ("residual_rows", ctypes.c_uint64),
        ("residual_columns", ctypes.c_uint64),
        ("residual_nonzeros", ctypes.c_uint64),
        ("prime", ctypes.c_uint32),
        ("target_reached", ctypes.c_uint8),
        ("preprocessing_seconds", ctypes.c_double),
        ("elapsed_seconds", ctypes.c_double),
    ]


class NativeSparseRankBatchStats(ctypes.Structure):
    _fields_ = [
        ("prime_count", ctypes.c_uint64),
        ("thread_count", ctypes.c_uint32),
        ("elapsed_seconds", ctypes.c_double),
    ]


class NativeSparseBlockColoopStats(ctypes.Structure):
    _fields_ = [
        ("row_count", ctypes.c_uint64),
        ("column_count", ctypes.c_uint64),
        ("input_nonzeros", ctypes.c_uint64),
        ("block_count", ctypes.c_uint64),
        ("block_incidences", ctypes.c_uint64),
        ("active_columns", ctypes.c_uint64),
        ("removed_columns", ctypes.c_uint64),
        ("residual_columns", ctypes.c_uint64),
        ("blocks_processed", ctypes.c_uint64),
        ("maximum_block_columns", ctypes.c_uint64),
        ("row_block_size", ctypes.c_uint32),
        ("prime", ctypes.c_uint32),
        ("elapsed_seconds", ctypes.c_double),
    ]


def _library_names() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return ("libfast_math.dylib", "liblambda_fast.dylib")
    if sys.platform == "win32":
        return (
            "fast_math.dll",
            "libfast_math.dll",
            "lambda_fast.dll",
            "liblambda_fast.dll",
        )
    return ("libfast_math.so", "liblambda_fast.so")


def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    for variable in ("FAST_MATH_LIBRARY", "LAMBDA_FAST_LIBRARY"):
        configured = os.environ.get(variable)
        if configured:
            candidates.append(Path(configured).expanduser())

    project_root = Path(__file__).resolve().parents[2]
    for name in _library_names():
        candidates.extend(
            (
                project_root / "build" / name,
                project_root / "build" / "Release" / name,
                project_root / "build" / "lib" / name,
            )
        )
    return candidates


@lru_cache(maxsize=1)
def load_library() -> ctypes.CDLL:
    attempted: list[str] = []
    for candidate in _candidate_paths():
        attempted.append(str(candidate))
        if not candidate.is_file():
            continue
        library = ctypes.CDLL(str(candidate))
        library.fast_math_version.argtypes = []
        library.fast_math_version.restype = ctypes.c_char_p
        library.lambda_fast_version.argtypes = []
        library.lambda_fast_version.restype = ctypes.c_char_p
        library.lambda_fast_accumulate_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(NativeStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.lambda_fast_accumulate_f64.restype = ctypes.c_int
        library.lambda_fast_fused_two_level_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(NativeTwoLevelRecord),
            ctypes.c_size_t,
            ctypes.POINTER(NativeTwoLevelStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.lambda_fast_fused_two_level_f64.restype = ctypes.c_int
        library.lambda_fast_power_moments_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(NativePowerMoment),
            ctypes.c_size_t,
            ctypes.POINTER(NativePowerMomentStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.lambda_fast_power_moments_f64.restype = ctypes.c_int
        library.lambda_fast_dirichlet_inverse_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(NativeInverseStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.lambda_fast_dirichlet_inverse_f64.restype = ctypes.c_int
        library.fast_math_segmented_complex_stats_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(NativeSegmentStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.fast_math_segmented_complex_stats_f64.restype = ctypes.c_int
        library.fast_math_taylor_coefficients_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(NativeTaylorStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.fast_math_taylor_coefficients_f64.restype = ctypes.c_int
        library.fast_math_taylor_evaluate_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(NativeTaylorStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.fast_math_taylor_evaluate_f64.restype = ctypes.c_int
        if hasattr(
            library,
            "fast_math_filon_chebyshev_inner_product_f64",
        ):
            library.fast_math_filon_chebyshev_inner_product_f64.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_uint32,
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_bool,
                ctypes.c_uint64,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(NativeFilonStats),
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            library.fast_math_filon_chebyshev_inner_product_f64.restype = (
                ctypes.c_int
            )
        library.fast_math_graph_pair_profiles_u64.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(NativeGraphStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.fast_math_graph_pair_profiles_u64.restype = ctypes.c_int
        library.fast_math_graph_find_clique_u64.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_bool,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(NativeGraphStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.fast_math_graph_find_clique_u64.restype = ctypes.c_int
        library.fast_math_graph6_decode_u64.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(NativeGraphStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.fast_math_graph6_decode_u64.restype = ctypes.c_int
        if hasattr(library, "fast_math_graph6_encode_u64"):
            library.fast_math_graph6_encode_u64.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
                ctypes.POINTER(NativeGraphStats),
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            library.fast_math_graph6_encode_u64.restype = ctypes.c_int
        if hasattr(library, "fast_math_graph_delete_vertices_u64"):
            library.fast_math_graph_delete_vertices_u64.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(NativeGraphStats),
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            library.fast_math_graph_delete_vertices_u64.restype = ctypes.c_int
        if hasattr(library, "fast_math_graph_rooted_leaf_features_u64"):
            library.fast_math_graph_rooted_leaf_features_u64.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(NativeGraphStats),
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            library.fast_math_graph_rooted_leaf_features_u64.restype = (
                ctypes.c_int
            )
        library.fast_math_graph_invariants_u64.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(NativeGraphStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.fast_math_graph_invariants_u64.restype = ctypes.c_int
        library.fast_math_graph_induced_profiles_u64.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(NativeGraphProfileStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.fast_math_graph_induced_profiles_u64.restype = ctypes.c_int
        library.fast_math_graph_induced_profile_stack_u64.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(NativeGraphProfileStackStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.fast_math_graph_induced_profile_stack_u64.restype = ctypes.c_int
        if hasattr(library, "fast_math_graph_triangles_csr_u32"):
            library.fast_math_graph_triangles_csr_u32.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(NativeLargeGraphStats),
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            library.fast_math_graph_triangles_csr_u32.restype = ctypes.c_int
        if hasattr(library, "fast_math_graph_common_neighbors_csr_u32"):
            library.fast_math_graph_common_neighbors_csr_u32.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(NativeCommonNeighborStats),
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            library.fast_math_graph_common_neighbors_csr_u32.restype = (
                ctypes.c_int
            )
        if hasattr(library, "fast_math_canonical_digraphs_nauty_u64"):
            library.fast_math_canonical_digraphs_nauty_u64.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(NativeCanonicalGraphStats),
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            library.fast_math_canonical_digraphs_nauty_u64.restype = (
                ctypes.c_int
            )
        if hasattr(library, "fast_math_canonical_digraphs_nauty_v2_u64"):
            library.fast_math_canonical_digraphs_nauty_v2_u64.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint8,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(NativeCanonicalGraphStats),
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            library.fast_math_canonical_digraphs_nauty_v2_u64.restype = (
                ctypes.c_int
            )
        library.fast_math_digest_u64_rows_sha256.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(NativeDigestStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.fast_math_digest_u64_rows_sha256.restype = ctypes.c_int
        if hasattr(library, "fast_math_union_closed_family_masks_u64"):
            library.fast_math_union_closed_family_masks_u64.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(NativeUnionStats),
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            ]
            library.fast_math_union_closed_family_masks_u64.restype = (
                ctypes.c_int
            )
        library.fast_math_sparse_rank_mod_u32.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.POINTER(NativeSparseRankStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.fast_math_sparse_rank_mod_u32.restype = ctypes.c_int
        library.fast_math_sparse_rank_mod_u32_batch.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.POINTER(NativeSparseRankStats),
            ctypes.POINTER(NativeSparseRankBatchStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.fast_math_sparse_rank_mod_u32_batch.restype = ctypes.c_int
        library.fast_math_sparse_block_coloops_mod_u32.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.POINTER(NativeSparseBlockColoopStats),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        library.fast_math_sparse_block_coloops_mod_u32.restype = ctypes.c_int
        return library
    raise NativeUnavailable(
        "fast-math native library is not built; attempted: "
        + ", ".join(attempted)
    )


def native_available() -> bool:
    try:
        load_library()
    except (NativeUnavailable, OSError):
        return False
    return True


def native_version() -> str | None:
    try:
        encoded = load_library().fast_math_version()
    except (NativeUnavailable, OSError):
        return None
    return encoded.decode("ascii")


def _double_pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _uint64_pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_uint64):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


def _uint32_pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_uint32):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))


def sparse_rank_mod_u32_native(
    row_offsets: NDArray[np.uint64],
    column_indices: NDArray[np.uint32],
    values: NDArray[np.uint32],
    *,
    column_count: int,
    prime: int,
    target_rank: int,
) -> tuple[
    NDArray[np.uint64],
    NDArray[np.uint32],
    NativeSparseRankStats,
]:
    row_count = len(row_offsets) - 1
    witness_capacity = (
        target_rank
        if target_rank != 0
        else min(row_count, column_count)
    )
    pivot_rows = np.empty(witness_capacity, dtype=np.uint64)
    pivot_columns = np.empty(witness_capacity, dtype=np.uint32)
    stats = NativeSparseRankStats()
    error = ctypes.create_string_buffer(1024)
    status = load_library().fast_math_sparse_rank_mod_u32(
        _uint64_pointer(row_offsets),
        _uint32_pointer(column_indices),
        _uint32_pointer(values),
        row_count,
        column_count,
        len(values),
        prime,
        target_rank,
        _uint64_pointer(pivot_rows),
        _uint32_pointer(pivot_columns),
        witness_capacity,
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return (
        pivot_rows[: stats.rank].copy(),
        pivot_columns[: stats.rank].copy(),
        stats,
    )


def sparse_rank_mod_u32_batch_native(
    row_offsets: NDArray[np.uint64],
    column_indices: NDArray[np.uint32],
    values_by_prime: NDArray[np.uint32],
    primes: NDArray[np.uint32],
    *,
    column_count: int,
    target_rank: int,
    threads: int,
) -> tuple[
    list[tuple[NDArray[np.uint64], NDArray[np.uint32]]],
    list[NativeSparseRankStats],
    NativeSparseRankBatchStats,
]:
    prime_count = len(primes)
    row_count = len(row_offsets) - 1
    witness_capacity = (
        target_rank
        if target_rank != 0
        else min(row_count, column_count)
    )
    pivot_rows = np.empty(
        (prime_count, witness_capacity),
        dtype=np.uint64,
    )
    pivot_columns = np.empty(
        (prime_count, witness_capacity),
        dtype=np.uint32,
    )
    stats_type = NativeSparseRankStats * prime_count
    stats = stats_type()
    batch_stats = NativeSparseRankBatchStats()
    error = ctypes.create_string_buffer(1024)
    status = load_library().fast_math_sparse_rank_mod_u32_batch(
        _uint64_pointer(row_offsets),
        _uint32_pointer(column_indices),
        _uint32_pointer(values_by_prime),
        row_count,
        column_count,
        values_by_prime.shape[1],
        _uint32_pointer(primes),
        prime_count,
        target_rank,
        threads,
        _uint64_pointer(pivot_rows),
        _uint32_pointer(pivot_columns),
        witness_capacity,
        stats,
        ctypes.byref(batch_stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    witnesses = [
        (
            pivot_rows[index, : stats[index].rank].copy(),
            pivot_columns[index, : stats[index].rank].copy(),
        )
        for index in range(prime_count)
    ]
    return witnesses, list(stats), batch_stats


def sparse_block_coloops_mod_u32_native(
    row_offsets: NDArray[np.uint64],
    column_indices: NDArray[np.uint32],
    values: NDArray[np.uint32],
    *,
    column_count: int,
    prime: int,
    row_block_size: int,
) -> tuple[
    NDArray[np.bool_],
    NDArray[np.uint32],
    NDArray[np.uint64],
    NDArray[np.uint32],
    NativeSparseBlockColoopStats,
]:
    row_count = len(row_offsets) - 1
    residual_columns = np.empty(column_count, dtype=np.uint8)
    removed_columns = np.empty(column_count, dtype=np.uint32)
    certificate_row_starts = np.empty(column_count, dtype=np.uint64)
    certificate_coefficients = np.empty(
        (column_count, row_block_size),
        dtype=np.uint32,
    )
    stats = NativeSparseBlockColoopStats()
    error = ctypes.create_string_buffer(1024)
    status = load_library().fast_math_sparse_block_coloops_mod_u32(
        _uint64_pointer(row_offsets),
        _uint32_pointer(column_indices),
        _uint32_pointer(values),
        row_count,
        column_count,
        len(values),
        prime,
        row_block_size,
        residual_columns.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint8)
        ),
        column_count,
        _uint32_pointer(removed_columns),
        _uint64_pointer(certificate_row_starts),
        _uint32_pointer(certificate_coefficients),
        column_count,
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    removed_count = int(stats.removed_columns)
    return (
        residual_columns.astype(np.bool_),
        removed_columns[:removed_count].copy(),
        certificate_row_starts[:removed_count].copy(),
        certificate_coefficients[:removed_count].copy(),
        stats,
    )


def digest_u64_rows_native(
    rows: NDArray[np.uint64],
    namespace: NDArray[np.uint8],
    *,
    threads: int,
) -> tuple[NDArray[np.uint8], NativeDigestStats]:
    row_count, field_count = rows.shape
    digests = np.empty((row_count, 32), dtype=np.uint8)
    stats = NativeDigestStats()
    error = ctypes.create_string_buffer(1024)
    namespace_pointer = (
        namespace.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        if len(namespace)
        else ctypes.POINTER(ctypes.c_uint8)()
    )
    status = load_library().fast_math_digest_u64_rows_sha256(
        _uint64_pointer(rows),
        row_count,
        field_count,
        namespace_pointer,
        len(namespace),
        threads,
        digests.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return digests, stats


def union_closed_family_masks_native(
    family_masks: NDArray[np.uint64],
    ground_size: int,
) -> tuple[NDArray[np.bool_], NativeUnionStats]:
    library = load_library()
    if not hasattr(library, "fast_math_union_closed_family_masks_u64"):
        raise NativeUnavailable(
            "fast-math was built without packed union-closure"
        )
    closed = np.empty(len(family_masks), dtype=np.uint8)
    stats = NativeUnionStats()
    error = ctypes.create_string_buffer(1024)
    status = library.fast_math_union_closed_family_masks_u64(
        _uint64_pointer(family_masks),
        len(family_masks),
        ground_size,
        closed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return closed.view(np.bool_), stats


def graph_induced_profile_stack_native(
    adjacency_masks: NDArray[np.uint64],
    induced_orders: NDArray[np.uint32],
    class_lookups: NDArray[np.uint32],
    lookup_offsets: NDArray[np.uint64],
    class_counts: NDArray[np.uint32],
    *,
    threads: int,
) -> tuple[NDArray[np.uint64], NativeGraphProfileStackStats]:
    graph_count, vertex_count = adjacency_masks.shape
    field_count = int(np.sum(class_counts, dtype=np.uint64))
    counts = np.empty((graph_count, field_count), dtype=np.uint64)
    stats = NativeGraphProfileStackStats()
    error = ctypes.create_string_buffer(1024)
    status = load_library().fast_math_graph_induced_profile_stack_u64(
        _uint64_pointer(adjacency_masks),
        graph_count,
        vertex_count,
        induced_orders.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        len(induced_orders),
        class_lookups.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        _uint64_pointer(lookup_offsets),
        class_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        threads,
        _uint64_pointer(counts),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return counts, stats


def graph_triangles_csr_native(
    row_offsets: NDArray[np.uint64],
    column_indices: NDArray[np.uint32],
    edge_color_masks: NDArray[np.uint64] | None,
    vertex_loop_color_masks: NDArray[np.uint64] | None,
) -> tuple[
    NDArray[np.uint32],
    NDArray[np.uint64] | None,
    NativeLargeGraphStats,
]:
    vertex_count = len(row_offsets) - 1
    color_pointer = (
        _uint64_pointer(edge_color_masks)
        if edge_color_masks is not None
        else ctypes.POINTER(ctypes.c_uint64)()
    )
    loop_pointer = (
        _uint64_pointer(vertex_loop_color_masks)
        if vertex_loop_color_masks is not None
        else ctypes.POINTER(ctypes.c_uint64)()
    )
    triangle_count = ctypes.c_uint64()
    stats = NativeLargeGraphStats()
    error = ctypes.create_string_buffer(1024)
    library = load_library()
    if not hasattr(library, "fast_math_graph_triangles_csr_u32"):
        raise NativeUnavailable(
            "fast-math was built without large-graph kernels"
        )
    status = library.fast_math_graph_triangles_csr_u32(
        _uint64_pointer(row_offsets),
        column_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        color_pointer,
        loop_pointer,
        vertex_count,
        len(column_indices),
        0,
        ctypes.POINTER(ctypes.c_uint32)(),
        ctypes.POINTER(ctypes.c_uint64)(),
        ctypes.byref(triangle_count),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")

    count = int(triangle_count.value)
    triangles = np.empty((count, 3), dtype=np.uint32)
    triangle_colors = (
        np.empty((count, 3), dtype=np.uint64)
        if edge_color_masks is not None
        else None
    )
    triangle_pointer = (
        triangles.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        if count
        else ctypes.POINTER(ctypes.c_uint32)()
    )
    triangle_color_pointer = (
        _uint64_pointer(triangle_colors)
        if triangle_colors is not None and count
        else ctypes.POINTER(ctypes.c_uint64)()
    )
    status = library.fast_math_graph_triangles_csr_u32(
        _uint64_pointer(row_offsets),
        column_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        color_pointer,
        loop_pointer,
        vertex_count,
        len(column_indices),
        count,
        triangle_pointer,
        triangle_color_pointer,
        ctypes.byref(triangle_count),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return triangles, triangle_colors, stats


def graph_common_neighbors_csr_native(
    row_offsets: NDArray[np.uint64],
    column_indices: NDArray[np.uint32],
    pairs: NDArray[np.uint32],
    *,
    materialize: bool,
) -> tuple[
    NDArray[np.uint64],
    NDArray[np.uint32] | None,
    NativeCommonNeighborStats,
]:
    library = load_library()
    if not hasattr(library, "fast_math_graph_common_neighbors_csr_u32"):
        raise NativeUnavailable(
            "fast-math was built without common-neighbor kernels"
        )
    pair_offsets = np.empty(len(pairs) + 1, dtype=np.uint64)
    common_neighbor_count = ctypes.c_uint64()
    stats = NativeCommonNeighborStats()
    error = ctypes.create_string_buffer(1024)
    null_neighbors = ctypes.POINTER(ctypes.c_uint32)()

    def execute(
        capacity: int,
        common_neighbors: NDArray[np.uint32] | None,
    ) -> None:
        neighbor_pointer = (
            _uint32_pointer(common_neighbors)
            if common_neighbors is not None and len(common_neighbors)
            else null_neighbors
        )
        status = library.fast_math_graph_common_neighbors_csr_u32(
            _uint64_pointer(row_offsets),
            _uint32_pointer(column_indices),
            len(row_offsets) - 1,
            len(column_indices),
            _uint32_pointer(pairs),
            len(pairs),
            capacity,
            _uint64_pointer(pair_offsets),
            neighbor_pointer,
            ctypes.byref(common_neighbor_count),
            ctypes.byref(stats),
            error,
            len(error),
        )
        if status != 0:
            message = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(f"fast-math native error {status}: {message}")

    if not materialize:
        execute(0, None)
        return pair_offsets, None, stats

    degrees = np.diff(row_offsets)
    pair_bounds = np.minimum(
        degrees[pairs[:, 0]],
        degrees[pairs[:, 1]],
    )
    maximum_bound = int(pair_bounds.max(initial=0))
    bound_cannot_overflow = (
        maximum_bound == 0
        or len(pair_bounds)
        <= np.iinfo(np.uint64).max // maximum_bound
    )
    capacity = (
        int(pair_bounds.sum(dtype=np.uint64))
        if bound_cannot_overflow
        else np.iinfo(np.intp).max
    )
    one_pass_limit = max(1_000_000, len(column_indices) * 32)
    if (
        capacity <= one_pass_limit
        and capacity <= 256 * 1024 * 1024 // np.dtype(np.uint32).itemsize
    ):
        common_neighbors = np.empty(capacity, dtype=np.uint32)
        execute(capacity, common_neighbors)
        common_neighbors.resize(
            int(common_neighbor_count.value),
            refcheck=False,
        )
        return pair_offsets, common_neighbors, stats

    execute(0, None)
    common_neighbors = np.empty(
        int(common_neighbor_count.value),
        dtype=np.uint32,
    )
    execute(len(common_neighbors), common_neighbors)
    return pair_offsets, common_neighbors, stats


def canonical_digraphs_nauty_native(
    adjacency_words: NDArray[np.uint64],
    vertex_colors: NDArray[np.uint32],
    *,
    threads: int = 0,
    collect_automorphism_generators: bool = True,
) -> tuple[
    NDArray[np.uint32],
    NDArray[np.uint64],
    NDArray[np.uint32],
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.uint32],
    NDArray[np.uint64],
    NDArray[np.uint32],
    NativeCanonicalGraphStats,
]:
    library = load_library()
    if not hasattr(library, "fast_math_canonical_digraphs_nauty_u64"):
        raise NativeUnavailable("fast-math was built without nauty")
    graph_count, vertex_count, word_count = adjacency_words.shape
    permutations = np.empty(
        (graph_count, vertex_count),
        dtype=np.uint32,
    )
    canonical_adjacency = np.empty_like(adjacency_words)
    canonical_colors = np.empty_like(vertex_colors)
    group_mantissas = np.empty(graph_count, dtype=np.float64)
    group_exponents = np.empty(graph_count, dtype=np.int32)
    orbit_counts = np.empty(graph_count, dtype=np.uint32)
    generator_offsets = np.empty(graph_count + 1, dtype=np.uint64)
    generator_count = ctypes.c_uint64()
    stats = NativeCanonicalGraphStats()
    error = ctypes.create_string_buffer(1024)
    maximum_generator_bytes = 512 << 20
    generator_capacity = (
        graph_count * vertex_count
        if collect_automorphism_generators
        else 0
    )
    use_v2 = (
        hasattr(library, "fast_math_canonical_digraphs_nauty_v2_u64")
        and generator_capacity * vertex_count * np.dtype(np.uint32).itemsize
        <= maximum_generator_bytes
    )
    if use_v2:
        generators = np.empty(
            (generator_capacity, vertex_count),
            dtype=np.uint32,
        )
        status = library.fast_math_canonical_digraphs_nauty_v2_u64(
            _uint64_pointer(adjacency_words),
            vertex_colors.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            graph_count,
            vertex_count,
            word_count,
            threads,
            int(collect_automorphism_generators),
            permutations.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            _uint64_pointer(canonical_adjacency),
            canonical_colors.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            group_mantissas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            group_exponents.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            orbit_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            _uint64_pointer(generator_offsets),
            generator_capacity,
            (
                generators.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_uint32)
                )
                if len(generators)
                else ctypes.POINTER(ctypes.c_uint32)()
            ),
            ctypes.byref(generator_count),
            ctypes.byref(stats),
            error,
            len(error),
        )
        if status == 0:
            generators.resize(
                (int(generator_count.value), vertex_count),
                refcheck=False,
            )
            return (
                permutations,
                canonical_adjacency,
                canonical_colors,
                group_mantissas,
                group_exponents,
                orbit_counts,
                generator_offsets,
                generators,
                stats,
            )
        message = error.value.decode("utf-8", errors="replace")
        if "one-pass per-graph generator bound" not in message:
            raise RuntimeError(
                f"fast-math native error {status}: {message}"
            )

    status = library.fast_math_canonical_digraphs_nauty_u64(
        _uint64_pointer(adjacency_words),
        vertex_colors.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        graph_count,
        vertex_count,
        word_count,
        permutations.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        _uint64_pointer(canonical_adjacency),
        canonical_colors.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        group_mantissas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        group_exponents.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        orbit_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        _uint64_pointer(generator_offsets),
        0,
        ctypes.POINTER(ctypes.c_uint32)(),
        ctypes.byref(generator_count),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    generators = np.empty(
        (int(generator_count.value), vertex_count),
        dtype=np.uint32,
    )
    if len(generators):
        status = library.fast_math_canonical_digraphs_nauty_u64(
            _uint64_pointer(adjacency_words),
            vertex_colors.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            graph_count,
            vertex_count,
            word_count,
            permutations.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            _uint64_pointer(canonical_adjacency),
            canonical_colors.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            group_mantissas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            group_exponents.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            orbit_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            _uint64_pointer(generator_offsets),
            len(generators),
            generators.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.byref(generator_count),
            ctypes.byref(stats),
            error,
            len(error),
        )
        if status != 0:
            message = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"fast-math native error {status}: {message}"
            )
    if not collect_automorphism_generators:
        generator_offsets.fill(0)
        generators = np.empty((0, vertex_count), dtype=np.uint32)
    return (
        permutations,
        canonical_adjacency,
        canonical_colors,
        group_mantissas,
        group_exponents,
        orbit_counts,
        generator_offsets,
        generators,
        stats,
    )


def accumulate_native(
    inputs: PreparedInputs,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.float64],
    NativeStats,
]:
    common = np.empty(inputs.output_limit + 1, dtype=np.complex128)
    low_output = np.empty(inputs.output_limit + 1, dtype=np.float64)
    stats = NativeStats()
    error = ctypes.create_string_buffer(1024)

    status = load_library().lambda_fast_accumulate_f64(
        _double_pointer(inputs.inverse),
        len(inputs.inverse),
        _double_pointer(inputs.primary),
        len(inputs.primary),
        _double_pointer(inputs.transformed.view(np.float64)),
        len(inputs.transformed),
        inputs.transformed_first,
        _double_pointer(inputs.low),
        len(inputs.low),
        inputs.output_limit,
        inputs.tile_size,
        inputs.threads,
        _double_pointer(common.view(np.float64)),
        _double_pointer(low_output),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return common, low_output, stats


def fused_two_level_native(
    inputs: PreparedInputs,
    *,
    weight_left: NDArray[np.uint64],
    weight_right: NDArray[np.uint64],
    weight_lower: NDArray[np.float64],
    weight_upper: NDArray[np.float64],
    gamma_abs: float,
    sigma: float,
    q_primary: complex,
    q_dual: complex,
    outer_ratio: float,
    record_count: int,
) -> tuple[list[NativeTwoLevelRecord], NativeTwoLevelStats]:
    record_array_type = NativeTwoLevelRecord * record_count
    records = record_array_type()
    stats = NativeTwoLevelStats()
    error = ctypes.create_string_buffer(1024)

    status = load_library().lambda_fast_fused_two_level_f64(
        _double_pointer(inputs.inverse),
        len(inputs.inverse),
        _double_pointer(inputs.primary),
        len(inputs.primary),
        _double_pointer(inputs.transformed.view(np.float64)),
        len(inputs.transformed),
        inputs.transformed_first,
        _double_pointer(inputs.low),
        len(inputs.low),
        _uint64_pointer(weight_left),
        _uint64_pointer(weight_right),
        _double_pointer(weight_lower),
        _double_pointer(weight_upper),
        len(weight_left),
        inputs.output_limit,
        gamma_abs,
        sigma,
        q_primary.real,
        q_primary.imag,
        q_dual.real,
        q_dual.imag,
        outer_ratio,
        inputs.tile_size,
        inputs.threads,
        records,
        record_count,
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return list(records), stats


def power_moments_native(
    values: NDArray[np.complex128],
    derivatives: NDArray[np.complex128],
    *,
    mesh_step: float,
    minimum_power: int,
    maximum_power: int,
    chunk_size: int,
    threads: int,
) -> tuple[list[NativePowerMoment], NativePowerMomentStats]:
    count = maximum_power - minimum_power + 1
    moment_array_type = NativePowerMoment * count
    moments = moment_array_type()
    stats = NativePowerMomentStats()
    error = ctypes.create_string_buffer(1024)
    status = load_library().lambda_fast_power_moments_f64(
        _double_pointer(values.view(np.float64)),
        _double_pointer(derivatives.view(np.float64)),
        len(values),
        mesh_step,
        minimum_power,
        maximum_power,
        chunk_size,
        threads,
        moments,
        count,
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return list(moments), stats


def dirichlet_inverse_native(
    source: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NativeInverseStats]:
    coefficients = np.empty(len(source) + 1, dtype=np.float64)
    stats = NativeInverseStats()
    error = ctypes.create_string_buffer(1024)
    status = load_library().lambda_fast_dirichlet_inverse_f64(
        _double_pointer(source),
        len(source),
        _double_pointer(coefficients),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return coefficients, stats


def segmented_complex_stats_native(
    values: NDArray[np.complex128],
    offsets: NDArray[np.uint64],
    *,
    threads: int,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.float64],
    NDArray[np.float64],
    NativeSegmentStats,
]:
    segment_count = len(offsets) - 1
    sums = np.empty(segment_count, dtype=np.complex128)
    l1 = np.empty(segment_count, dtype=np.float64)
    variation = np.empty(segment_count, dtype=np.float64)
    stats = NativeSegmentStats()
    error = ctypes.create_string_buffer(1024)
    status = load_library().fast_math_segmented_complex_stats_f64(
        _double_pointer(values.view(np.float64)),
        len(values),
        _uint64_pointer(offsets),
        segment_count,
        threads,
        _double_pointer(sums.view(np.float64)),
        _double_pointer(l1),
        _double_pointer(variation),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return sums, l1, variation, stats


def taylor_coefficients_native(
    base: NDArray[np.complex128],
    logarithms: NDArray[np.float64],
    *,
    maximum_order: int,
    chunk_size: int,
    threads: int,
) -> tuple[NDArray[np.complex128], NativeTaylorStats]:
    coefficients = np.empty(
        (maximum_order + 1, len(base)),
        dtype=np.complex128,
    )
    stats = NativeTaylorStats()
    error = ctypes.create_string_buffer(1024)
    status = load_library().fast_math_taylor_coefficients_f64(
        _double_pointer(base.view(np.float64)),
        _double_pointer(logarithms),
        len(base),
        maximum_order,
        chunk_size,
        threads,
        _double_pointer(coefficients.view(np.float64)),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return coefficients, stats


def taylor_evaluate_native(
    basis: NDArray[np.complex128],
    delta: NDArray[np.complex128],
    *,
    chunk_size: int,
    threads: int,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NativeTaylorStats,
]:
    values = np.empty(len(delta), dtype=np.complex128)
    log_moments = np.empty(len(delta), dtype=np.complex128)
    stats = NativeTaylorStats()
    error = ctypes.create_string_buffer(1024)
    status = load_library().fast_math_taylor_evaluate_f64(
        _double_pointer(basis.view(np.float64)),
        _double_pointer(delta.view(np.float64)),
        len(delta),
        basis.shape[0],
        chunk_size,
        threads,
        _double_pointer(values.view(np.float64)),
        _double_pointer(log_moments.view(np.float64)),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return values, log_moments, stats


def filon_chebyshev_inner_product_native(
    correlation: NDArray[np.complex128],
    exact_weights: NDArray[np.complex128],
    positive_endpoint_derivatives: NDArray[np.float64],
    negative_endpoint_derivatives: NDArray[np.float64],
    *,
    output_count: int,
    eta: float,
    length: float,
    conjugate_kernel: bool,
    chunk_size: int,
    threads: int,
) -> tuple[complex, NativeFilonStats]:
    library = load_library()
    if not hasattr(
        library,
        "fast_math_filon_chebyshev_inner_product_f64",
    ):
        raise NativeUnavailable(
            "fast-math was built without the Filon kernel"
        )
    result = np.empty(1, dtype=np.complex128)
    stats = NativeFilonStats()
    error = ctypes.create_string_buffer(1024)
    status = library.fast_math_filon_chebyshev_inner_product_f64(
        _double_pointer(correlation.view(np.float64)),
        len(correlation),
        _double_pointer(exact_weights.view(np.float64)),
        len(exact_weights),
        _double_pointer(positive_endpoint_derivatives),
        _double_pointer(negative_endpoint_derivatives),
        len(positive_endpoint_derivatives),
        output_count,
        eta,
        length,
        conjugate_kernel,
        chunk_size,
        threads,
        _double_pointer(result.view(np.float64)),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return complex(result[0]), stats


def graph_pair_profiles_native(
    adjacency_masks: NDArray[np.uint64],
    *,
    threads: int,
) -> tuple[
    NDArray[np.bool_],
    NDArray[np.uint32],
    NDArray[np.uint32],
    NDArray[np.uint32],
    NDArray[np.uint32],
    NativeGraphStats,
]:
    graph_count, vertex_count = adjacency_masks.shape
    pair_count = vertex_count * (vertex_count - 1) // 2
    shape = (graph_count, pair_count)
    adjacent_u8 = np.empty(shape, dtype=np.uint8)
    common_neighbors = np.empty(shape, dtype=np.uint32)
    common_nonneighbors = np.empty(shape, dtype=np.uint32)
    only_left = np.empty(shape, dtype=np.uint32)
    only_right = np.empty(shape, dtype=np.uint32)
    stats = NativeGraphStats()
    error = ctypes.create_string_buffer(1024)
    status = load_library().fast_math_graph_pair_profiles_u64(
        _uint64_pointer(adjacency_masks),
        graph_count,
        vertex_count,
        threads,
        adjacent_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        common_neighbors.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        common_nonneighbors.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        only_left.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        only_right.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return (
        adjacent_u8.view(np.bool_),
        common_neighbors,
        common_nonneighbors,
        only_left,
        only_right,
        stats,
    )


def graph_find_clique_native(
    adjacency_masks: NDArray[np.uint64],
    *,
    order: int,
    complement: bool,
    threads: int,
) -> tuple[NDArray[np.uint64], NDArray[np.uint64], NativeGraphStats]:
    graph_count, vertex_count = adjacency_masks.shape
    witnesses = np.empty(graph_count, dtype=np.uint64)
    nodes_visited = np.empty(graph_count, dtype=np.uint64)
    stats = NativeGraphStats()
    error = ctypes.create_string_buffer(1024)
    status = load_library().fast_math_graph_find_clique_u64(
        _uint64_pointer(adjacency_masks),
        graph_count,
        vertex_count,
        order,
        complement,
        threads,
        _uint64_pointer(witnesses),
        _uint64_pointer(nodes_visited),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return witnesses, nodes_visited, stats


def graph6_decode_native(
    data: NDArray[np.uint8],
    offsets: NDArray[np.uint64],
    *,
    vertex_count: int,
    threads: int,
) -> tuple[NDArray[np.uint64], NativeGraphStats]:
    graph_count = len(offsets) - 1
    adjacency_masks = np.empty(
        (graph_count, vertex_count),
        dtype=np.uint64,
    )
    stats = NativeGraphStats()
    error = ctypes.create_string_buffer(1024)
    status = load_library().fast_math_graph6_decode_u64(
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        len(data),
        _uint64_pointer(offsets),
        graph_count,
        vertex_count,
        threads,
        _uint64_pointer(adjacency_masks),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return adjacency_masks, stats


def graph6_encode_native(
    adjacency_masks: NDArray[np.uint64],
    *,
    threads: int,
) -> tuple[NDArray[np.uint8], NativeGraphStats]:
    library = load_library()
    if not hasattr(library, "fast_math_graph6_encode_u64"):
        raise NativeUnavailable(
            "native graph6 encoding is unavailable in this library"
        )
    graph_count, vertex_count = adjacency_masks.shape
    edge_bits = vertex_count * (vertex_count - 1) // 2
    record_size = 1 + (edge_bits + 5) // 6
    data = np.empty(graph_count * record_size, dtype=np.uint8)
    stats = NativeGraphStats()
    error = ctypes.create_string_buffer(1024)
    status = library.fast_math_graph6_encode_u64(
        _uint64_pointer(adjacency_masks),
        graph_count,
        vertex_count,
        threads,
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        len(data),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return data, stats


def graph_delete_vertices_native(
    adjacency_masks: NDArray[np.uint64],
    source_graphs: NDArray[np.uint64],
    deleted_vertices: NDArray[np.uint32],
    *,
    threads: int,
) -> tuple[NDArray[np.uint64], NativeGraphStats]:
    library = load_library()
    if not hasattr(library, "fast_math_graph_delete_vertices_u64"):
        raise NativeUnavailable(
            "native graph vertex deletion is unavailable in this library"
        )
    graph_count, vertex_count = adjacency_masks.shape
    output = np.empty(
        (len(source_graphs), vertex_count - 1),
        dtype=np.uint64,
    )
    stats = NativeGraphStats()
    error = ctypes.create_string_buffer(1024)
    status = library.fast_math_graph_delete_vertices_u64(
        _uint64_pointer(adjacency_masks),
        graph_count,
        vertex_count,
        _uint64_pointer(source_graphs),
        deleted_vertices.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint32)
        ),
        len(source_graphs),
        threads,
        _uint64_pointer(output),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return output, stats


def graph_rooted_leaf_features_native(
    adjacency_masks: NDArray[np.uint64],
    source_graphs: NDArray[np.uint64],
    leaf_vertices: NDArray[np.uint32],
    *,
    threads: int,
) -> tuple[
    NDArray[np.uint32],
    NDArray[np.uint32],
    NDArray[np.uint32],
    NativeGraphStats,
]:
    library = load_library()
    if not hasattr(library, "fast_math_graph_rooted_leaf_features_u64"):
        raise NativeUnavailable(
            "native rooted-leaf features are unavailable in this library"
        )
    graph_count, vertex_count = adjacency_masks.shape
    root_degrees = np.empty(len(source_graphs), dtype=np.uint32)
    two_star_counts = np.empty(len(source_graphs), dtype=np.uint32)
    two_step_counts = np.empty(len(source_graphs), dtype=np.uint32)
    stats = NativeGraphStats()
    error = ctypes.create_string_buffer(1024)
    status = library.fast_math_graph_rooted_leaf_features_u64(
        _uint64_pointer(adjacency_masks),
        graph_count,
        vertex_count,
        _uint64_pointer(source_graphs),
        leaf_vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        len(source_graphs),
        threads,
        root_degrees.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        two_star_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        two_step_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return root_degrees, two_star_counts, two_step_counts, stats


def graph_invariants_native(
    adjacency_masks: NDArray[np.uint64],
    *,
    threads: int,
) -> tuple[
    NDArray[np.uint32],
    NDArray[np.uint64],
    NDArray[np.uint64],
    NDArray[np.uint64],
    NDArray[np.uint64],
    NativeGraphStats,
]:
    graph_count, vertex_count = adjacency_masks.shape
    degrees = np.empty((graph_count, vertex_count), dtype=np.uint32)
    edge_counts = np.empty(graph_count, dtype=np.uint64)
    triangle_counts = np.empty(graph_count, dtype=np.uint64)
    wedge_counts = np.empty(graph_count, dtype=np.uint64)
    induced_path3_counts = np.empty(graph_count, dtype=np.uint64)
    stats = NativeGraphStats()
    error = ctypes.create_string_buffer(1024)
    status = load_library().fast_math_graph_invariants_u64(
        _uint64_pointer(adjacency_masks),
        graph_count,
        vertex_count,
        threads,
        degrees.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        _uint64_pointer(edge_counts),
        _uint64_pointer(triangle_counts),
        _uint64_pointer(wedge_counts),
        _uint64_pointer(induced_path3_counts),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return (
        degrees,
        edge_counts,
        triangle_counts,
        wedge_counts,
        induced_path3_counts,
        stats,
    )


def graph_induced_profiles_native(
    adjacency_masks: NDArray[np.uint64],
    *,
    induced_order: int,
    class_lookup: NDArray[np.uint32],
    class_count: int,
    threads: int,
) -> tuple[NDArray[np.uint64], NativeGraphProfileStats]:
    graph_count, vertex_count = adjacency_masks.shape
    counts = np.empty((graph_count, class_count), dtype=np.uint64)
    stats = NativeGraphProfileStats()
    error = ctypes.create_string_buffer(1024)
    status = load_library().fast_math_graph_induced_profiles_u64(
        _uint64_pointer(adjacency_masks),
        graph_count,
        vertex_count,
        induced_order,
        class_lookup.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        len(class_lookup),
        class_count,
        threads,
        _uint64_pointer(counts),
        ctypes.byref(stats),
        error,
        len(error),
    )
    if status != 0:
        message = error.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"fast-math native error {status}: {message}")
    return counts, stats
