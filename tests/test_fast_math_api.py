from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import fast_math
import fast_math.actions as fast_actions
import fast_math.graphs as fast_graphs
import fast_math.plans as fast_plans
import fast_math.runtime as fast_runtime
import lambda_fast
from fast_math._native import load_library


def test_public_exports_are_bound() -> None:
    for module in (
        fast_math,
        fast_actions,
        fast_graphs,
        fast_plans,
        fast_runtime,
    ):
        assert len(module.__all__) == len(set(module.__all__))
        assert not [
            name for name in module.__all__ if not hasattr(module, name)
        ]


def test_lambda_package_can_load_before_fast_math() -> None:
    subprocess.run(
        [sys.executable, "-c", "import lambda_fast; import fast_math"],
        check=True,
    )


def test_general_api_preserves_lambda_exports() -> None:
    assert fast_math.accumulate_coefficients is lambda_fast.accumulate_coefficients
    assert fast_math.dirichlet_inverse is lambda_fast.dirichlet_inverse
    assert fast_math.fused_two_level is lambda_fast.fused_two_level
    assert fast_math.power_moments is lambda_fast.power_moments
    assert fast_math.native_version() == "0.7.0"


def test_old_and_new_native_abi_symbols_are_available() -> None:
    library = load_library()
    assert library.fast_math_version().decode("ascii") == "0.7.0"
    assert library.lambda_fast_version().decode("ascii") == "0.7.0"
    for symbol in (
        "lambda_fast_accumulate_f64",
        "lambda_fast_fused_two_level_f64",
        "fast_math_graph6_decode_u64",
        "fast_math_graph6_encode_u64",
        "fast_math_graph_invariants_u64",
        "fast_math_graph_induced_profiles_u64",
        "fast_math_graph_induced_profile_stack_u64",
        "fast_math_digest_u64_rows_sha256",
        "fast_math_sparse_rank_mod_u32",
        "fast_math_sparse_rank_mod_u32_batch",
        "fast_math_sparse_block_coloops_mod_u32",
        "fast_math_filon_chebyshev_inner_product_f64",
        "fast_math_subset_action_create_u32",
        "fast_math_subset_action_canonicalize_u64",
        "fast_math_subset_action_destroy",
    ):
        assert getattr(library, symbol) is not None


def test_lambda_fast_compatibility_links() -> None:
    project = Path(__file__).resolve().parents[1]
    compatibility_project = project.parent / "lambda-fast"
    if compatibility_project.is_symlink():
        assert compatibility_project.resolve() == project.resolve()
    for suffix in (".dylib", ".so"):
        compatibility_library = project / "build" / f"liblambda_fast{suffix}"
        if compatibility_library.exists():
            assert compatibility_library.is_symlink()
            assert compatibility_library.resolve().name == f"libfast_math{suffix}"
