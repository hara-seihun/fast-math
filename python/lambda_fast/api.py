from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._inputs import prepare_inputs
from ._native import (
    NativeUnavailable,
    accumulate_native,
    native_available,
    native_version as _native_version,
)
from .reference import accumulate_reference


Backend = Literal["auto", "native", "reference"]


@dataclass(frozen=True)
class PairCounts:
    primary: int
    transformed: int
    low: int

    @property
    def total(self) -> int:
        return self.primary + self.transformed + self.low


@dataclass(frozen=True)
class AccumulationResult:
    common: NDArray[np.complex128]
    low: NDArray[np.float64]
    pairs: PairCounts
    elapsed_seconds: float
    backend: str


def available_backends() -> tuple[str, ...]:
    if native_available():
        return ("native", "reference")
    return ("reference",)


def native_version() -> str | None:
    return _native_version()


def accumulate_coefficients(
    inverse: ArrayLike,
    primary: ArrayLike,
    transformed: ArrayLike,
    low: ArrayLike,
    *,
    transformed_first: int,
    output_limit: int,
    tile_size: int = 1 << 14,
    threads: int = 1,
    backend: Backend = "auto",
) -> AccumulationResult:
    """Accumulate the three current lambda replay source channels.

    Arrays use the same indexing convention as the existing replay:
    ``inverse[0]`` and output index zero are unused, while source element zero
    represents mathematical index one. The transformed source begins at
    ``transformed_first``.
    """
    if backend not in {"auto", "native", "reference"}:
        raise ValueError(f"unknown backend: {backend}")
    inputs = prepare_inputs(
        inverse,
        primary,
        transformed,
        low,
        transformed_first=transformed_first,
        output_limit=output_limit,
        tile_size=tile_size,
        threads=threads,
    )

    if backend in {"auto", "native"}:
        try:
            common, low_output, stats = accumulate_native(inputs)
        except (NativeUnavailable, OSError):
            if backend == "native":
                raise
        else:
            return AccumulationResult(
                common=common,
                low=low_output,
                pairs=PairCounts(
                    primary=int(stats.primary_pairs),
                    transformed=int(stats.transformed_pairs),
                    low=int(stats.low_pairs),
                ),
                elapsed_seconds=float(stats.elapsed_seconds),
                backend="native",
            )

    output = accumulate_reference(inputs)
    return AccumulationResult(
        common=output.common,
        low=output.low,
        pairs=PairCounts(
            primary=output.primary_pairs,
            transformed=output.transformed_pairs,
            low=output.low_pairs,
        ),
        elapsed_seconds=output.elapsed_seconds,
        backend="reference",
    )
