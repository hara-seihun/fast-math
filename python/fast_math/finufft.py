from __future__ import annotations

from dataclasses import dataclass
import operator
import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexArray = NDArray[np.complex64] | NDArray[np.complex128]
RealArray = NDArray[np.float32] | NDArray[np.float64]


class FinufftUnavailable(RuntimeError):
    """Raised when the optional FINUFFT backend is not installed."""


@dataclass(frozen=True)
class PlanTimings:
    plan_seconds: float
    point_set_seconds: float
    execute_seconds: float
    execute_count: int
    point_set_count: int


def _finufft_module():
    try:
        import finufft
    except ImportError as error:
        raise FinufftUnavailable(
            "FINUFFT support requires the optional 'finufft' dependency"
        ) from error
    return finufft


def _normalize_dtype(dtype: Any) -> tuple[np.dtype, np.dtype]:
    complex_dtype = np.dtype(dtype)
    if complex_dtype == np.dtype(np.complex64):
        return complex_dtype, np.dtype(np.float32)
    if complex_dtype == np.dtype(np.complex128):
        return complex_dtype, np.dtype(np.float64)
    raise TypeError("dtype must be complex64 or complex128")


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _sign(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("isign must be -1 or 1")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError("isign must be -1 or 1") from error
    if result not in (-1, 1):
        raise ValueError("isign must be -1 or 1")
    return result


def _coordinates(
    values: ArrayLike,
    *,
    name: str,
    dtype: np.dtype,
) -> RealArray:
    result = np.array(values, dtype=dtype, order="C", copy=True)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.flags.writeable = False
    return result


def _strengths(
    values: ArrayLike,
    *,
    source_count: int,
    n_trans: int,
    dtype: np.dtype,
) -> ComplexArray:
    result = np.asarray(values, dtype=dtype, order="C")
    expected = (source_count,) if n_trans == 1 else (n_trans, source_count)
    if result.shape != expected:
        raise ValueError(
            f"strengths must have shape {expected}, got {result.shape}"
        )
    return result


def _retained_strengths(
    values: ArrayLike,
    *,
    source_count: int,
    n_trans: int,
    dtype: np.dtype,
) -> ComplexArray:
    result = np.array(
        _strengths(
            values,
            source_count=source_count,
            n_trans=n_trans,
            dtype=dtype,
        ),
        dtype=dtype,
        order="C",
        copy=True,
    )
    result.flags.writeable = False
    return result


class _Plan1D:
    def __init__(
        self,
        *,
        nufft_type: int,
        n_modes_or_dim: int | tuple[int, ...],
        n_trans: int,
        eps: float,
        isign: int,
        dtype: Any,
        options: dict[str, Any],
    ) -> None:
        normalized_n_trans = _positive_integer(
            n_trans, name="n_trans"
        )
        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be finite and positive")
        normalized_sign = _sign(isign)
        self._complex_dtype, self._real_dtype = _normalize_dtype(dtype)
        self._n_trans = normalized_n_trans
        self._plan_seconds = 0.0
        self._point_set_seconds = 0.0
        self._execute_seconds = 0.0
        self._execute_count = 0
        self._point_set_count = 0
        started = time.perf_counter()
        self._plan = _finufft_module().Plan(
            nufft_type,
            n_modes_or_dim,
            n_trans=self._n_trans,
            eps=eps,
            isign=normalized_sign,
            dtype=self._complex_dtype,
            **options,
        )
        self._plan_seconds = time.perf_counter() - started

    @property
    def n_trans(self) -> int:
        return self._n_trans

    @property
    def dtype(self) -> np.dtype:
        return self._complex_dtype

    @property
    def timings(self) -> PlanTimings:
        return PlanTimings(
            plan_seconds=self._plan_seconds,
            point_set_seconds=self._point_set_seconds,
            execute_seconds=self._execute_seconds,
            execute_count=self._execute_count,
            point_set_count=self._point_set_count,
        )

    def _record_setpts(self, function) -> None:
        started = time.perf_counter()
        function()
        self._point_set_seconds += time.perf_counter() - started
        self._point_set_count += 1

    def _execute(
        self,
        strengths: ArrayLike,
        *,
        source_count: int,
        out: ComplexArray | None,
    ) -> ComplexArray:
        prepared = _strengths(
            strengths,
            source_count=source_count,
            n_trans=self._n_trans,
            dtype=self._complex_dtype,
        )
        started = time.perf_counter()
        result = self._plan.execute(prepared, out=out)
        self._execute_seconds += time.perf_counter() - started
        self._execute_count += 1
        return result


class Type1Plan1D(_Plan1D):
    """Persistent one-dimensional type-1 FINUFFT plan."""

    def __init__(
        self,
        nodes: ArrayLike,
        n_modes: int,
        *,
        n_trans: int = 1,
        eps: float = 1e-6,
        isign: int = 1,
        dtype: Any = np.complex128,
        **options: Any,
    ) -> None:
        normalized_modes = _positive_integer(
            n_modes, name="n_modes"
        )
        super().__init__(
            nufft_type=1,
            n_modes_or_dim=(normalized_modes,),
            n_trans=n_trans,
            eps=eps,
            isign=isign,
            dtype=dtype,
            options=options,
        )
        self._n_modes = normalized_modes
        self.set_nodes(nodes)

    @property
    def nodes(self) -> RealArray:
        return self._nodes

    @property
    def n_modes(self) -> int:
        return self._n_modes

    def set_nodes(self, nodes: ArrayLike) -> None:
        prepared = _coordinates(
            nodes, name="nodes", dtype=self._real_dtype
        )
        self._record_setpts(lambda: self._plan.setpts(prepared))
        self._nodes = prepared

    def execute(
        self,
        strengths: ArrayLike,
        *,
        out: ComplexArray | None = None,
    ) -> ComplexArray:
        return self._execute(
            strengths, source_count=self._nodes.size, out=out
        )


class Type3Plan1D(_Plan1D):
    """Persistent one-dimensional type-3 FINUFFT plan."""

    def __init__(
        self,
        sources: ArrayLike,
        targets: ArrayLike,
        *,
        n_trans: int = 1,
        eps: float = 1e-6,
        isign: int = 1,
        dtype: Any = np.complex128,
        **options: Any,
    ) -> None:
        super().__init__(
            nufft_type=3,
            n_modes_or_dim=1,
            n_trans=n_trans,
            eps=eps,
            isign=isign,
            dtype=dtype,
            options=options,
        )
        self.set_points(sources, targets)

    @property
    def sources(self) -> RealArray:
        return self._sources

    @property
    def targets(self) -> RealArray:
        return self._targets

    def set_points(
        self, sources: ArrayLike, targets: ArrayLike
    ) -> None:
        prepared_sources = _coordinates(
            sources, name="sources", dtype=self._real_dtype
        )
        prepared_targets = _coordinates(
            targets, name="targets", dtype=self._real_dtype
        )
        self._record_setpts(
            lambda: self._plan.setpts(
                prepared_sources, s=prepared_targets
            )
        )
        self._sources = prepared_sources
        self._targets = prepared_targets

    def set_targets(self, targets: ArrayLike) -> None:
        prepared_targets = _coordinates(
            targets, name="targets", dtype=self._real_dtype
        )
        self._record_setpts(
            lambda: self._plan.setpts(
                self._sources, s=prepared_targets
            )
        )
        self._targets = prepared_targets

    def execute(
        self,
        strengths: ArrayLike,
        *,
        out: ComplexArray | None = None,
    ) -> ComplexArray:
        return self._execute(
            strengths, source_count=self._sources.size, out=out
        )


class Type3FixedPlan1D:
    """Persistent type-3 plan with retained strengths and output storage.

    Returned arrays are views of retained output storage and are overwritten
    by the next execution.
    """

    def __init__(
        self,
        sources: ArrayLike,
        strengths: ArrayLike,
        targets: ArrayLike,
        *,
        n_trans: int = 1,
        eps: float = 1e-6,
        isign: int = 1,
        dtype: Any = np.complex128,
        **options: Any,
    ) -> None:
        normalized_n_trans = _positive_integer(
            n_trans, name="n_trans"
        )
        self._n_trans = normalized_n_trans
        self._plan = Type3Plan1D(
            sources,
            targets,
            n_trans=normalized_n_trans,
            eps=eps,
            isign=isign,
            dtype=dtype,
            **options,
        )
        self.set_strengths(strengths)
        self._allocate_output()

    @property
    def sources(self) -> RealArray:
        return self._plan.sources

    @property
    def targets(self) -> RealArray:
        return self._plan.targets

    @property
    def strengths(self) -> ComplexArray:
        return self._strengths

    @property
    def n_trans(self) -> int:
        return self._n_trans

    @property
    def dtype(self) -> np.dtype:
        return self._plan.dtype

    @property
    def timings(self) -> PlanTimings:
        return self._plan.timings

    def _output_shape(self) -> tuple[int, ...]:
        if self._n_trans == 1:
            return (self.targets.size,)
        return (self._n_trans, self.targets.size)

    def _allocate_output(self) -> None:
        shape = self._output_shape()
        if (
            not hasattr(self, "_output")
            or self._output.shape != shape
        ):
            self._output = np.empty(shape, dtype=self.dtype)

    def set_targets(self, targets: ArrayLike) -> None:
        self._plan.set_targets(targets)
        self._allocate_output()

    def set_strengths(self, strengths: ArrayLike) -> None:
        self._strengths = _retained_strengths(
            strengths,
            source_count=self.sources.size,
            n_trans=self._n_trans,
            dtype=self.dtype,
        )

    def execute(
        self,
        *,
        targets: ArrayLike | None = None,
    ) -> ComplexArray:
        if targets is not None:
            self.set_targets(targets)
        return self._plan.execute(
            self._strengths,
            out=self._output,
        )


class Type3SignPairPlan1D:
    """Persistent paired-sign type-3 transforms on shared source nodes.

    The negative-sign stack is evaluated through
    ``conj(F_+(conj(c)))`` so both stacks share one positive-sign plan and
    target set. Returned arrays are views of retained output storage and are
    overwritten by the next execution.
    """

    def __init__(
        self,
        sources: ArrayLike,
        positive_strengths: ArrayLike,
        negative_strengths: ArrayLike,
        targets: ArrayLike,
        *,
        n_trans: int = 1,
        eps: float = 1e-6,
        dtype: Any = np.complex128,
        **options: Any,
    ) -> None:
        normalized_n_trans = _positive_integer(
            n_trans, name="n_trans"
        )
        complex_dtype, _ = _normalize_dtype(dtype)
        self._n_trans = normalized_n_trans
        self._plan = Type3Plan1D(
            sources,
            targets,
            n_trans=2 * normalized_n_trans,
            eps=eps,
            isign=1,
            dtype=complex_dtype,
            **options,
        )
        self.set_strengths(positive_strengths, negative_strengths)
        self._allocate_output()

    @property
    def sources(self) -> RealArray:
        return self._plan.sources

    @property
    def targets(self) -> RealArray:
        return self._plan.targets

    @property
    def positive_strengths(self) -> ComplexArray:
        return self._positive_strengths

    @property
    def negative_strengths(self) -> ComplexArray:
        return self._negative_strengths

    @property
    def n_trans(self) -> int:
        return self._n_trans

    @property
    def dtype(self) -> np.dtype:
        return self._plan.dtype

    @property
    def timings(self) -> PlanTimings:
        return self._plan.timings

    def _allocate_output(self) -> None:
        shape = (2 * self._n_trans, self.targets.size)
        if (
            not hasattr(self, "_output")
            or self._output.shape != shape
        ):
            self._output = np.empty(shape, dtype=self.dtype)

    def set_targets(self, targets: ArrayLike) -> None:
        self._plan.set_targets(targets)
        self._allocate_output()

    def set_strengths(
        self,
        positive_strengths: ArrayLike,
        negative_strengths: ArrayLike,
    ) -> None:
        source_count = self.sources.size
        self._positive_strengths = _retained_strengths(
            positive_strengths,
            source_count=source_count,
            n_trans=self._n_trans,
            dtype=self.dtype,
        )
        self._negative_strengths = _retained_strengths(
            negative_strengths,
            source_count=source_count,
            n_trans=self._n_trans,
            dtype=self.dtype,
        )
        positive = self._positive_strengths.reshape(
            self._n_trans, source_count
        )
        negative = self._negative_strengths.reshape(
            self._n_trans, source_count
        )
        self._packed_strengths = np.concatenate(
            (positive, np.conjugate(negative)),
            axis=0,
        )

    def execute(
        self,
        *,
        targets: ArrayLike | None = None,
    ) -> tuple[ComplexArray, ComplexArray]:
        if targets is not None:
            self.set_targets(targets)
        self._plan.execute(
            self._packed_strengths,
            out=self._output,
        )
        np.conjugate(
            self._output[self._n_trans :],
            out=self._output[self._n_trans :],
        )
        if self._n_trans == 1:
            return self._output[0], self._output[1]
        return (
            self._output[: self._n_trans],
            self._output[self._n_trans :],
        )
