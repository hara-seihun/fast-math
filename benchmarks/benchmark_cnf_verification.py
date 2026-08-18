#!/usr/bin/env python3
"""Benchmark retained exact CNF certificate verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import subprocess
import tempfile
from time import perf_counter

import numpy as np

from fast_math import CnfPlan, pack_boolean_assignments
from fast_math.hip import hip_cnf_available


def fixture() -> tuple[list[list[int]], np.ndarray]:
    rng = np.random.default_rng(0xC3F7)
    variable_count = 128
    planted = rng.integers(
        0, 2, size=variable_count, dtype=np.uint8
    ).astype(np.bool_)
    clauses: list[list[int]] = []
    for _ in range(1024):
        variables = rng.choice(variable_count, size=3, replace=False)
        signs = rng.integers(0, 2, size=3, dtype=np.uint8).astype(np.bool_)
        literals = [
            int(variable + 1) if sign else -int(variable + 1)
            for variable, sign in zip(variables, signs)
        ]
        if not any(planted[variable] == sign for variable, sign in zip(variables, signs)):
            literals[0] = (
                int(variables[0] + 1)
                if planted[variables[0]]
                else -int(variables[0] + 1)
            )
        clauses.append(literals)
    return clauses, planted


def cadical_single_assignment(
    clauses: list[list[int]],
    assignment: np.ndarray,
) -> float:
    with tempfile.TemporaryDirectory(prefix="fast-math-cnf-") as directory:
        path = Path(directory) / "certificate.cnf"
        lines = [
            f"p cnf {len(assignment)} {len(clauses) + len(assignment)}"
        ]
        lines.extend(" ".join(map(str, clause)) + " 0" for clause in clauses)
        lines.extend(
            f"{index + 1 if value else -(index + 1)} 0"
            for index, value in enumerate(assignment)
        )
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        started = perf_counter()
        completed = subprocess.run(
            ["cadical", "-q", str(path)],
            capture_output=True,
            text=True,
        )
        elapsed = perf_counter() - started
        if completed.returncode != 10:
            raise RuntimeError("CaDiCaL rejected the planted assignment")
    return elapsed


def measure(
    plan: CnfPlan,
    assignments: np.ndarray,
    backend: str,
    repeats: int,
) -> dict:
    setup_started = perf_counter()
    result = plan.evaluate(assignments, threads=8, backend=backend)
    setup_and_first = perf_counter() - setup_started
    walls = []
    kernels = []
    for _ in range(repeats):
        started = perf_counter()
        result = plan.evaluate(assignments, threads=8, backend=backend)
        walls.append(perf_counter() - started)
        kernels.append(result.elapsed_seconds)
    return {
        "setup_and_first_seconds": setup_and_first,
        "seconds": walls,
        "median_seconds": median(walls),
        "kernel_median_seconds": median(kernels),
        "satisfied_count": int(result.satisfied.sum()),
        "first_unsatisfied_checksum": int(
            result.first_unsatisfied_clause.sum(dtype=np.int64)
        ),
        "inspected_literal_count": result.inspected_literal_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    clauses, planted = fixture()
    small = pack_boolean_assignments(
        np.broadcast_to(planted, (1000, len(planted)))
    )
    large = pack_boolean_assignments(
        np.broadcast_to(planted, (100000, len(planted)))
    )
    rng = np.random.default_rng(0xBAD5)
    rejected = pack_boolean_assignments(
        rng.integers(
            0, 2, size=(100000, len(planted)), dtype=np.uint8
        )
    )
    with CnfPlan(clauses, variable_count=len(planted)) as plan:
        reference_started = perf_counter()
        reference = plan.evaluate(small, backend="reference")
        reference_wall = perf_counter() - reference_started
        small_records = {"native": measure(plan, small, "native", args.repeats)}
        large_records = {"native": measure(plan, large, "native", args.repeats)}
        rejected_native = measure(plan, rejected, "native", args.repeats)
        if hip_cnf_available():
            small_records["hip"] = measure(plan, small, "hip", args.repeats)
            large_records["hip"] = measure(plan, large, "hip", args.repeats)
            rejected_hip = measure(plan, rejected, "hip", args.repeats)
            np.testing.assert_array_equal(
                plan.evaluate(rejected, backend="native").first_unsatisfied_clause,
                plan.evaluate(rejected, backend="hip").first_unsatisfied_clause,
            )
        else:
            rejected_hip = None
        native_small = plan.evaluate(small, backend="native")
        np.testing.assert_array_equal(
            native_small.first_unsatisfied_clause,
            reference.first_unsatisfied_clause,
        )
    native_small_record = small_records["native"]
    native_large_record = large_records["native"]
    payload: dict[str, object] = {
        "schema": 1,
        "operation": "exact-cnf-assignment-verification",
        "variables": len(planted),
        "clauses": len(clauses),
        "literals": sum(map(len, clauses)),
        "python_reference_1000_seconds": reference_wall,
        "cadical_single_assignment_seconds": cadical_single_assignment(
            clauses, planted
        ),
        "satisfying_1000": small_records,
        "satisfying_100000": large_records,
        "rejected_100000_native": rejected_native,
        "rejected_100000_hip": rejected_hip,
        "python_reference_over_native_1000_best_wall": (
            reference_wall / min(native_small_record["seconds"])
        ),
    }
    if "hip" in large_records:
        hip_small_record = small_records["hip"]
        hip_large_record = large_records["hip"]
        payload["python_reference_over_hip_1000_best_wall"] = (
            reference_wall / min(hip_small_record["seconds"])
        )
        payload["native_over_hip_100000_best_wall"] = (
            min(native_large_record["seconds"])
            / min(hip_large_record["seconds"])
        )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="ascii")


if __name__ == "__main__":
    main()
