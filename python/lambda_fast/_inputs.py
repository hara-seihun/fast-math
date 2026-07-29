from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PreparedInputs:
    inverse: NDArray[np.float64]
    primary: NDArray[np.float64]
    transformed: NDArray[np.complex128]
    low: NDArray[np.float64]
    transformed_first: int
    output_limit: int
    tile_size: int
    threads: int


def _one_dimensional(values: ArrayLike, dtype: np.dtype, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return np.ascontiguousarray(array)


def prepare_inputs(
    inverse: ArrayLike,
    primary: ArrayLike,
    transformed: ArrayLike,
    low: ArrayLike,
    *,
    transformed_first: int,
    output_limit: int,
    tile_size: int,
    threads: int,
) -> PreparedInputs:
    inverse_array = _one_dimensional(inverse, np.dtype(np.float64), "inverse")
    primary_array = _one_dimensional(primary, np.dtype(np.float64), "primary")
    transformed_array = _one_dimensional(
        transformed, np.dtype(np.complex128), "transformed"
    )
    low_array = _one_dimensional(low, np.dtype(np.float64), "low")

    if len(inverse_array) == 0:
        raise ValueError("inverse must contain index zero")
    if transformed_first < 1:
        raise ValueError("transformed_first must be positive")
    if output_limit < 1:
        raise ValueError("output_limit must be positive")
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    if threads < 0:
        raise ValueError("threads must be nonnegative")

    return PreparedInputs(
        inverse=inverse_array,
        primary=primary_array,
        transformed=transformed_array,
        low=low_array,
        transformed_first=int(transformed_first),
        output_limit=int(output_limit),
        tile_size=int(tile_size),
        threads=int(threads),
    )
