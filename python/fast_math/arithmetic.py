"""Arithmetic kernels reusable outside Lambda computations."""

from lambda_fast import InverseResult, dirichlet_inverse, truncated_inverse

__all__ = [
    "InverseResult",
    "dirichlet_inverse",
    "truncated_inverse",
]
