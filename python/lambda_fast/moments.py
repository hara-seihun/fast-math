from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from fast_math._native import NativeUnavailable, power_moments_native


Backend = Literal["auto", "native", "reference"]


@dataclass(frozen=True)
class PowerMoment:
    power: int
    value: float
    ordinary: float
    phase_current: float
    radial: float


@dataclass(frozen=True)
class PowerMomentResult:
    moments: tuple[PowerMoment, ...]
    sample_count: int
    maximum_modulus: float
    maximum_derivative: float
    elapsed_seconds: float
    backend: str


def _complex_vector(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return np.ascontiguousarray(array)


def _reference(
    values: np.ndarray,
    derivatives: np.ndarray,
    *,
    mesh_step: float,
    minimum_power: int,
    maximum_power: int,
) -> PowerMomentResult:
    started = time.perf_counter()
    modulus_squared = np.abs(values) ** 2
    derivative_squared = np.abs(derivatives) ** 2
    radial_product = derivatives * np.conjugate(values)
    moments = []
    for power in range(minimum_power, maximum_power + 1):
        weight = modulus_squared ** (power - 1)
        radial_weight = (
            modulus_squared ** (power - 2)
            if power >= 2
            else 1.0 / modulus_squared
        )
        moments.append(
            PowerMoment(
                power=power,
                value=float(
                    mesh_step * np.sum(weight * modulus_squared)
                ),
                ordinary=float(
                    mesh_step
                    * power**2
                    * np.sum(weight * derivative_squared)
                ),
                phase_current=float(
                    mesh_step
                    * power
                    * np.sum(weight * np.imag(radial_product))
                ),
                radial=float(
                    mesh_step
                    * power**2
                    * np.sum(
                        radial_weight
                        * np.real(radial_product) ** 2
                    )
                ),
            )
        )
    return PowerMomentResult(
        moments=tuple(moments),
        sample_count=len(values),
        maximum_modulus=float(np.max(np.sqrt(modulus_squared))),
        maximum_derivative=float(np.max(np.sqrt(derivative_squared))),
        elapsed_seconds=time.perf_counter() - started,
        backend="reference",
    )


def power_moments(
    values: ArrayLike,
    derivatives: ArrayLike,
    *,
    mesh_step: float,
    minimum_power: int = 3,
    maximum_power: int = 12,
    chunk_size: int = 1 << 16,
    threads: int = 1,
    backend: Backend = "auto",
) -> PowerMomentResult:
    """Powers of one value stream with their ordinary, phase-current, and radial moments, in one pass."""
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    value_array = _complex_vector(values, "values")
    derivative_array = _complex_vector(derivatives, "derivatives")
    if len(value_array) == 0 or len(value_array) != len(derivative_array):
        raise ValueError("values and derivatives must have equal nonzero length")
    if not math.isfinite(mesh_step) or mesh_step <= 0.0:
        raise ValueError("mesh_step must be finite and positive")
    if minimum_power < 1 or maximum_power < minimum_power:
        raise ValueError("invalid power range")
    if chunk_size < 1 or threads < 0:
        raise ValueError("chunk_size must be positive and threads nonnegative")

    if backend in {"auto", "native"}:
        try:
            raw_moments, stats = power_moments_native(
                value_array,
                derivative_array,
                mesh_step=mesh_step,
                minimum_power=minimum_power,
                maximum_power=maximum_power,
                chunk_size=chunk_size,
                threads=threads,
            )
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return PowerMomentResult(
                moments=tuple(
                    PowerMoment(
                        int(moment.power),
                        float(moment.value),
                        float(moment.ordinary),
                        float(moment.phase_current),
                        float(moment.radial),
                    )
                    for moment in raw_moments
                ),
                sample_count=int(stats.sample_count),
                maximum_modulus=float(stats.maximum_modulus),
                maximum_derivative=float(stats.maximum_derivative),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )

    return _reference(
        value_array,
        derivative_array,
        mesh_step=mesh_step,
        minimum_power=minimum_power,
        maximum_power=maximum_power,
    )
