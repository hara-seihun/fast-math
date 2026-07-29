"""Executable NumPy reference model matching the existing certificate code."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from numpy.typing import NDArray

from ._inputs import PreparedInputs


@dataclass(frozen=True)
class ReferenceOutput:
    common: NDArray[np.complex128]
    low: NDArray[np.float64]
    primary_pairs: int
    transformed_pairs: int
    low_pairs: int
    elapsed_seconds: float


def accumulate_reference(inputs: PreparedInputs) -> ReferenceOutput:
    started = time.perf_counter()
    common = np.zeros(inputs.output_limit + 1, dtype=np.complex128)
    low_output = np.zeros(inputs.output_limit + 1, dtype=np.float64)
    primary_pairs = 0
    transformed_pairs = 0
    low_pairs = 0
    transformed_last = (
        inputs.transformed_first + len(inputs.transformed) - 1
    )

    for divisor in range(1, len(inputs.inverse)):
        inverse_coefficient = inputs.inverse[divisor]
        if inverse_coefficient == 0.0:
            continue

        primary_maximum = min(
            len(inputs.primary), inputs.output_limit // divisor
        )
        if primary_maximum >= 1:
            indices = divisor * np.arange(
                1, primary_maximum + 1, dtype=np.int64
            )
            common[indices] += (
                inverse_coefficient * inputs.primary[:primary_maximum]
            )
            primary_pairs += primary_maximum

        transformed_maximum = min(
            transformed_last, inputs.output_limit // divisor
        )
        if (
            len(inputs.transformed) != 0
            and transformed_maximum >= inputs.transformed_first
        ):
            count = transformed_maximum - inputs.transformed_first + 1
            indices = divisor * np.arange(
                inputs.transformed_first,
                transformed_maximum + 1,
                dtype=np.int64,
            )
            common[indices] += (
                inverse_coefficient * inputs.transformed[:count]
            )
            transformed_pairs += count

        low_maximum = min(len(inputs.low), inputs.output_limit // divisor)
        if low_maximum >= 1:
            indices = divisor * np.arange(
                1, low_maximum + 1, dtype=np.int64
            )
            low_output[indices] += (
                inverse_coefficient * inputs.low[:low_maximum]
            )
            low_pairs += low_maximum

    return ReferenceOutput(
        common=common,
        low=low_output,
        primary_pairs=primary_pairs,
        transformed_pairs=transformed_pairs,
        low_pairs=low_pairs,
        elapsed_seconds=time.perf_counter() - started,
    )
