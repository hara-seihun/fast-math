#!/usr/bin/env python3
"""Benchmark retained exact solves and inconsistency certificates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import subprocess
import tempfile
from time import perf_counter

import numpy as np

from fast_math import ModularLinearSystemPlan
from fast_math.hip import hip_modular_linear_available


def timed_calls(call, repeats: int) -> tuple[object, list[float]]:
    result = None
    seconds = []
    for _ in range(repeats):
        started = perf_counter()
        result = call()
        seconds.append(perf_counter() - started)
    assert result is not None
    return result, seconds


def checksum(array: np.ndarray, prime: int) -> int:
    return int(array.sum(dtype=np.uint64) % prime)


def flint_retained_solve(
    matrix: np.ndarray,
    right_hand_sides: np.ndarray,
    prime: int,
    repeats: int,
) -> tuple[np.ndarray, dict[str, object]]:
    from flint import nmod_mat

    started = perf_counter()
    flint_matrix = nmod_mat(
        len(matrix), len(matrix), matrix.reshape(-1).tolist(), prime
    )
    inverse = flint_matrix.inv()
    setup_seconds = perf_counter() - started
    started = perf_counter()
    flint_right = nmod_mat(
        len(matrix),
        len(right_hand_sides),
        right_hand_sides.T.reshape(-1).tolist(),
        prime,
    )
    input_seconds = perf_counter() - started
    result, seconds = timed_calls(lambda: inverse * flint_right, repeats)
    values = np.asarray(result.entries(), dtype=np.uint32).reshape(
        len(matrix), len(right_hand_sides)
    ).T
    return values, {
        "setup_seconds": setup_seconds,
        "input_conversion_seconds": input_seconds,
        "seconds": seconds,
        "median_seconds": median(seconds),
        "checksum": checksum(values, prime),
    }


def sage_certificate_audit(
    matrix: np.ndarray,
    right_hand_sides: np.ndarray,
    solutions: np.ndarray,
    consistent: np.ndarray,
    inconsistency_rows: np.ndarray,
    left_nullspace: np.ndarray,
    prime: int,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(
        prefix="fast-math-linear-sage-"
    ) as directory:
        fixture = Path(directory) / "fixture.npz"
        np.savez(
            fixture,
            matrix=matrix,
            right_hand_sides=right_hand_sides,
            solutions=solutions,
            consistent=consistent,
            inconsistency_rows=inconsistency_rows,
            left_nullspace=left_nullspace,
        )
        program = f"""
import json
import numpy as np
from sage.all import GF, Matrix, vector
from time import perf_counter
f=np.load({str(fixture)!r})
field=GF({prime})
raw_a=f['matrix']
a=Matrix(field, raw_a.shape[0], raw_a.shape[1], raw_a.reshape(-1).tolist())
left=f['left_nullspace']
started=perf_counter()
sage_consistent=[]
for raw_b in f['right_hand_sides']:
    b=vector(field, raw_b.tolist())
    try:
        a.solve_right(b)
        sage_consistent.append(True)
    except ValueError:
        sage_consistent.append(False)
solve_seconds=perf_counter()-started
for index,raw_b in enumerate(f['right_hand_sides']):
    b=vector(field, raw_b.tolist())
    expected=bool(f['consistent'][index])
    assert sage_consistent[index] == expected
    if expected:
        x=vector(field, f['solutions'][index].tolist())
        assert a*x == b
    else:
        witness_index=int(f['inconsistency_rows'][index])
        y=vector(field, left[witness_index].tolist())
        assert y*a == 0
        assert y*b != 0
print(json.dumps({{'rank': int(a.rank()), 'solve_seconds': solve_seconds, 'audited': len(sage_consistent), 'consistent_count': sum(sage_consistent)}}))
"""
        completed = subprocess.run(
            ["sage", "-python", "-c", program],
            check=True,
            capture_output=True,
            text=True,
        )
    return json.loads(completed.stdout)


def backend_records(
    plan: ModularLinearSystemPlan,
    right_hand_sides: np.ndarray,
    repeats: int,
) -> tuple[dict[str, object], object]:
    records: dict[str, object] = {}
    final = None
    baseline = None
    for backend in ("native", "hip"):
        if backend == "hip" and not hip_modular_linear_available():
            continue
        started = perf_counter()
        first = plan.solve(
            right_hand_sides, backend=backend, threads=8
        )
        setup_and_first = perf_counter() - started
        if backend == "hip":
            for _ in range(2):
                plan.solve(right_hand_sides, backend=backend, threads=8)
        seconds = []
        kernels = []
        result = first
        for _ in range(repeats):
            started = perf_counter()
            result = plan.solve(
                right_hand_sides, backend=backend, threads=8
            )
            seconds.append(perf_counter() - started)
            kernels.append(result.elapsed_seconds)
        if baseline is None:
            baseline = result
        else:
            np.testing.assert_array_equal(
                result.solutions, baseline.solutions
            )
            np.testing.assert_array_equal(
                result.consistent, baseline.consistent
            )
            np.testing.assert_array_equal(
                result.inconsistency_rows, baseline.inconsistency_rows
            )
        final = result
        records[backend] = {
            "setup_and_first_seconds": setup_and_first,
            "seconds": seconds,
            "median_seconds": median(seconds),
            "kernel_median_seconds": median(kernels),
            "consistent_count": int(result.consistent.sum()),
            "solution_checksum": checksum(result.solutions, plan.prime),
        }
        np.testing.assert_array_equal(
            result.consistent, first.consistent
        )
        np.testing.assert_array_equal(
            result.inconsistency_rows, first.inconsistency_rows
        )
    assert final is not None
    return records, final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prime = 65521
    rng = np.random.default_rng(0x1A2B3C)

    square_matrix = rng.integers(
        0, prime, size=(64, 64), dtype=np.uint32
    )
    square_right = rng.integers(
        0, prime, size=(8192, 64), dtype=np.uint32
    )
    square_started = perf_counter()
    square_plan = ModularLinearSystemPlan(square_matrix, prime=prime)
    square_setup_wall = perf_counter() - square_started
    if square_plan.rank != 64:
        raise RuntimeError("deterministic square fixture is unexpectedly singular")
    square_records, square_result = backend_records(
        square_plan, square_right, args.repeats
    )
    np.testing.assert_array_equal(
        square_plan.verify(square_right[:16], square_plan.solve(
            square_right[:16], backend="native", threads=8
        )),
        True,
    )
    flint_values, flint = flint_retained_solve(
        square_matrix, square_right, prime, args.repeats
    )
    np.testing.assert_array_equal(flint_values, square_result.solutions)

    left_factor = rng.integers(
        0, prime, size=(48, 40), dtype=np.uint64
    )
    right_factor = rng.integers(
        0, prime, size=(40, 64), dtype=np.uint64
    )
    rectangular_matrix = (left_factor @ right_factor % prime).astype(np.uint32)
    rectangular_started = perf_counter()
    rectangular_plan = ModularLinearSystemPlan(
        rectangular_matrix, prime=prime
    )
    rectangular_setup_wall = perf_counter() - rectangular_started
    if rectangular_plan.rank != 40:
        raise RuntimeError("deterministic rectangular fixture has the wrong rank")
    planted = rng.integers(
        0, prime, size=(4096, 64), dtype=np.uint64
    )
    valid_right = (
        planted @ rectangular_matrix.astype(np.uint64).T % prime
    ).astype(np.uint32)
    invalid_right = rng.integers(
        0, prime, size=(4096, 48), dtype=np.uint32
    )
    rectangular_right = np.concatenate((valid_right, invalid_right))
    rectangular_records, rectangular_result = backend_records(
        rectangular_plan, rectangular_right, args.repeats
    )
    np.testing.assert_array_equal(
        rectangular_plan.verify(
            rectangular_right[:16],
            rectangular_plan.solve(
                rectangular_right[:16], backend="native", threads=8
            ),
        ),
        True,
    )
    np.testing.assert_array_equal(
        rectangular_plan.verify(
            rectangular_right[-16:],
            rectangular_plan.solve(
                rectangular_right[-16:], backend="native", threads=8
            ),
        ),
        True,
    )
    sample_indices = np.concatenate(
        (np.arange(32), np.arange(len(rectangular_right) - 32, len(rectangular_right)))
    )
    sage = sage_certificate_audit(
        rectangular_matrix,
        rectangular_right[sample_indices],
        rectangular_result.solutions[sample_indices],
        rectangular_result.consistent[sample_indices],
        rectangular_result.inconsistency_rows[sample_indices],
        rectangular_plan.left_nullspace,
        prime,
    )
    rectangular_plan.close()
    square_plan.close()

    payload: dict[str, object] = {
        "schema": 1,
        "operation": "retained-exact-modular-linear-systems",
        "prime": prime,
        "square": {
            "shape": [64, 64],
            "right_hand_sides": len(square_right),
            "setup_wall_seconds": square_setup_wall,
            "backends": square_records,
            "python_flint": flint,
        },
        "rank_deficient": {
            "shape": [48, 64],
            "rank": 40,
            "right_hand_sides": len(rectangular_right),
            "setup_wall_seconds": rectangular_setup_wall,
            "backends": rectangular_records,
            "sage_certificate_audit": sage,
        },
    }
    square_medians = {
        backend: record["median_seconds"]
        for backend, record in square_records.items()
        if isinstance(record, dict)
    }
    best_square_backend = min(square_medians, key=square_medians.get)
    best_square_seconds = square_medians[best_square_backend]
    payload["best_fast_math_square_backend"] = best_square_backend
    payload["flint_right_hand_side_route_over_best_fast_math_square_median"] = (
        (flint["input_conversion_seconds"] + flint["median_seconds"])
        / best_square_seconds
    )
    if "hip" in square_records:
        hip = square_records["hip"]
        assert isinstance(hip, dict)
        payload["flint_kernel_over_hip_square_median"] = (
            flint["median_seconds"] / hip["median_seconds"]
        )
    if "hip" in rectangular_records:
        native = rectangular_records["native"]
        hip = rectangular_records["hip"]
        assert isinstance(native, dict) and isinstance(hip, dict)
        payload["native_over_hip_rank_deficient_median"] = (
            native["median_seconds"] / hip["median_seconds"]
        )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="ascii")


if __name__ == "__main__":
    main()
