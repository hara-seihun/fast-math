"""Retained exact CNF assignment verification with failure witnesses."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import Iterable, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cnf_native import NativeCnfPlan, cnf_native_available


CnfBackend = Literal["auto", "reference", "native", "hip"]


@dataclass(frozen=True)
class CnfEvaluation:
    satisfied: NDArray[np.bool_]
    first_unsatisfied_clause: NDArray[np.int64]
    inspected_literal_count: int | None
    elapsed_seconds: float
    backend: str


def _prepare_clauses(
    clauses: Iterable[Iterable[int]],
    variable_count: int | None,
) -> tuple[NDArray[np.uint64], NDArray[np.int32], int]:
    offsets = [0]
    flat: list[int] = []
    maximum = 0
    for clause in clauses:
        for literal in clause:
            if not isinstance(literal, Integral) or int(literal) == 0:
                raise ValueError("CNF literals must be nonzero integers")
            value = int(literal)
            if not np.iinfo(np.int32).min <= value <= np.iinfo(np.int32).max:
                raise ValueError("CNF literal must fit int32")
            maximum = max(maximum, abs(value))
            flat.append(value)
        offsets.append(len(flat))
    if variable_count is None:
        variable_count = maximum
    if (
        not isinstance(variable_count, Integral)
        or not 1 <= int(variable_count) <= np.iinfo(np.uint32).max
        or maximum > int(variable_count)
    ):
        raise ValueError("variable_count does not cover the CNF literals")
    return (
        np.asarray(offsets, dtype=np.uint64),
        np.asarray(flat, dtype=np.int32),
        int(variable_count),
    )


def pack_boolean_assignments(values: ArrayLike) -> NDArray[np.uint64]:
    raw = np.asarray(values)
    if raw.ndim != 2:
        raise ValueError("assignments must have shape (count, variables)")
    if raw.dtype != np.bool_:
        if not np.issubdtype(raw.dtype, np.integer) or np.any((raw != 0) & (raw != 1)):
            raise ValueError("assignments must contain Boolean values")
    count, variable_count = raw.shape
    if variable_count == 0:
        raise ValueError("assignments require at least one variable")
    result = np.zeros((count, (variable_count + 63) // 64), dtype=np.uint64)
    booleans = raw.astype(np.bool_, copy=False)
    for variable in range(variable_count):
        result[:, variable // 64] |= (
            booleans[:, variable].astype(np.uint64)
            << np.uint64(variable % 64)
        )
    return result


def _assignment_words(
    values: ArrayLike,
    variable_count: int,
) -> NDArray[np.uint64]:
    raw = np.asarray(values)
    word_count = (variable_count + 63) // 64
    if (
        raw.ndim != 2
        or raw.shape[1] != word_count
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise ValueError("assignments have an invalid packed shape")
    if np.issubdtype(raw.dtype, np.signedinteger) and np.any(raw < 0):
        raise ValueError("assignment words must be nonnegative")
    result = np.ascontiguousarray(raw, dtype=np.uint64)
    final_bits = variable_count % 64
    if final_bits and np.any(result[:, -1] >> np.uint64(final_bits)):
        raise ValueError("assignment contains an out-of-range bit")
    return result


def _reference(
    offsets: NDArray[np.uint64],
    literals: NDArray[np.int32],
    assignments: NDArray[np.uint64],
) -> tuple[NDArray[np.int64], int]:
    first = np.full(len(assignments), -1, dtype=np.int64)
    inspected = 0
    for assignment_index, words in enumerate(assignments):
        for clause in range(len(offsets) - 1):
            clause_satisfied = False
            for offset in range(int(offsets[clause]), int(offsets[clause + 1])):
                inspected += 1
                literal = int(literals[offset])
                variable = abs(literal) - 1
                value = bool(
                    (int(words[variable // 64]) >> (variable % 64)) & 1
                )
                if value == (literal > 0):
                    clause_satisfied = True
                    break
            if not clause_satisfied:
                first[assignment_index] = clause
                break
    return first, inspected


class CnfPlan:
    """Retain a DIMACS-style CNF and verify packed assignment batches."""

    def __init__(
        self,
        clauses: Sequence[Sequence[int]],
        *,
        variable_count: int | None = None,
    ) -> None:
        offsets, literals, variables = _prepare_clauses(clauses, variable_count)
        offsets.flags.writeable = False
        literals.flags.writeable = False
        self._offsets = offsets
        self._literals = literals
        self.variable_count = variables
        self._native = None
        self._hip = None
        self._closed = False

    @classmethod
    def from_csr(
        cls,
        clause_offsets: ArrayLike,
        literals: ArrayLike,
        *,
        variable_count: int,
    ) -> "CnfPlan":
        offsets_raw = np.asarray(clause_offsets)
        literals_raw = np.asarray(literals)
        if (
            offsets_raw.ndim != 1
            or not np.issubdtype(offsets_raw.dtype, np.integer)
            or literals_raw.ndim != 1
            or not np.issubdtype(literals_raw.dtype, np.integer)
        ):
            raise ValueError("CNF CSR arrays must be one-dimensional integers")
        if np.issubdtype(offsets_raw.dtype, np.signedinteger) and np.any(offsets_raw < 0):
            raise ValueError("clause offsets must be nonnegative")
        offsets = np.array(offsets_raw, dtype=np.uint64, order="C", copy=True)
        literal_values = [int(value) for value in literals_raw]
        if (
            len(offsets) == 0
            or offsets[0] != 0
            or offsets[-1] != len(literal_values)
            or np.any(offsets[1:] < offsets[:-1])
        ):
            raise ValueError("CNF clause offsets are inconsistent")
        clauses = [
            literal_values[int(offsets[index]) : int(offsets[index + 1])]
            for index in range(len(offsets) - 1)
        ]
        return cls(clauses, variable_count=variable_count)

    @property
    def clause_offsets(self) -> NDArray[np.uint64]:
        return self._offsets

    @property
    def literals(self) -> NDArray[np.int32]:
        return self._literals

    @property
    def clause_count(self) -> int:
        return len(self._offsets) - 1

    @property
    def literal_count(self) -> int:
        return len(self._literals)

    @property
    def word_count(self) -> int:
        return (self.variable_count + 63) // 64

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("CNF plan is closed")

    def _native_plan(self) -> NativeCnfPlan:
        if self._native is None:
            self._native = NativeCnfPlan(
                self._offsets, self._literals, self.variable_count
            )
        return self._native

    def _hip_plan(self):
        if self._hip is None:
            from .hip import CnfHipPlan

            self._hip = CnfHipPlan(
                self._offsets, self._literals, self.variable_count
            )
        return self._hip

    def evaluate(
        self,
        assignments: ArrayLike,
        *,
        threads: int = 1,
        backend: CnfBackend = "auto",
    ) -> CnfEvaluation:
        self._check_open()
        if backend not in {"auto", "reference", "native", "hip"}:
            raise ValueError(f"unknown CNF backend: {backend}")
        if not isinstance(threads, Integral) or int(threads) < 1:
            raise ValueError("threads must be positive")
        prepared = _assignment_words(assignments, self.variable_count)
        selected = backend
        if selected == "auto":
            nominal_work = len(prepared) * self.literal_count
            have_native = cnf_native_available()
            if nominal_work >= 20_000_000:
                from .hip import hip_cnf_available

                if self._hip is not None and len(prepared) >= 10_000:
                    selected = "hip"
                elif hip_cnf_available():
                    if have_native and len(prepared):
                        sample_count = min(256, len(prepared))
                        _, _, sample_stats = self._native_plan().evaluate(
                            prepared[:sample_count], threads=int(threads)
                        )
                        projected_work = (
                            int(sample_stats.inspected_literal_count)
                            * len(prepared)
                            // sample_count
                        )
                        selected = (
                            "hip" if projected_work >= 300_000_000 else "native"
                        )
                    else:
                        selected = "hip"
                else:
                    selected = "native" if have_native else "reference"
            else:
                selected = "native" if have_native else "reference"
        if selected == "reference":
            started = perf_counter()
            first, inspected = _reference(
                self._offsets, self._literals, prepared
            )
            elapsed = perf_counter() - started
        elif selected == "native":
            satisfied, first, stats = self._native_plan().evaluate(
                prepared, threads=int(threads)
            )
            elapsed = float(stats.elapsed_seconds)
            inspected = int(stats.inspected_literal_count)
            return CnfEvaluation(
                satisfied=satisfied,
                first_unsatisfied_clause=first,
                inspected_literal_count=inspected,
                elapsed_seconds=elapsed,
                backend="native",
            )
        else:
            first, elapsed = self._hip_plan().evaluate(prepared)
            inspected = None
        return CnfEvaluation(
            satisfied=first < 0,
            first_unsatisfied_clause=first,
            inspected_literal_count=inspected,
            elapsed_seconds=elapsed,
            backend=selected,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._native is not None:
            self._native.close()
        if self._hip is not None:
            self._hip.close()

    def __enter__(self) -> "CnfPlan":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "CnfBackend",
    "CnfEvaluation",
    "CnfPlan",
    "pack_boolean_assignments",
]
