"""Backend discovery and capability reporting for Fast Math plans."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import platform

from ._native import native_available


@dataclass(frozen=True)
class BackendCapability:
    name: str
    available: bool
    device: str
    exact_integer: bool
    precisions: tuple[str, ...]
    operations: tuple[str, ...]


def backend_capabilities() -> dict[str, BackendCapability]:
    from .hip import hip_available

    system = platform.system()
    return {
        "reference": BackendCapability(
            name="reference",
            available=True,
            device="cpu",
            exact_integer=True,
            precisions=("float64", "complex128", "uint32", "uint64"),
            operations=("all executable specifications",),
        ),
        "native": BackendCapability(
            name="native",
            available=native_available(),
            device="cpu",
            exact_integer=True,
            precisions=("float64", "complex128", "uint32", "uint64"),
            operations=(
                "CNF assignment verification",
                "graphs",
                "groups",
                "modular polynomials, determinants, and linear systems",
                "packed subset actions",
                "sparse finite-field rank",
                "reductions",
            ),
        ),
        "hip": BackendCapability(
            name="hip",
            available=hip_available(),
            device="gpu",
            exact_integer=True,
            precisions=("float32", "float64", "uint32", "uint64"),
            operations=(
                "affine populations",
                "CNF assignment verification",
                "modular polynomials, determinants, and linear systems",
                "oriented-square incidence",
                "packed subset actions",
            ),
        ),
        "cuda": BackendCapability(
            name="cuda",
            available=(
                system != "Darwin" and importlib.util.find_spec("cupy") is not None
            ),
            device="gpu",
            exact_integer=False,
            precisions=("float32", "complex64"),
            operations=("affine populations",),
        ),
        "metal": BackendCapability(
            name="metal",
            available=(
                system == "Darwin" and importlib.util.find_spec("mlx") is not None
            ),
            device="gpu",
            exact_integer=False,
            precisions=("float32", "complex64"),
            operations=("affine populations",),
        ),
    }


__all__ = ["BackendCapability", "backend_capabilities"]
