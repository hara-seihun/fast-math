#!/usr/bin/env python3
"""Benchmark exact retained modular batches against SageMath and FLINT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import subprocess
import tempfile
from time import perf_counter

import numpy as np

from fast_math import ModularDeterminantPlan, ModularPolynomialPlan
from fast_math.hip import hip_modular_available


def checksum(*arrays: np.ndarray, prime: int) -> int:
    return int(
        sum(int(array.sum(dtype=np.uint64)) for array in arrays) % prime
    )


def sage_baseline(
    coefficients: np.ndarray,
    points: np.ndarray,
    matrices: np.ndarray,
    prime: int,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="fast-math-sage-") as directory:
        path = Path(directory) / "fixture.npz"
        np.savez(path, coefficients=coefficients, points=points, matrices=matrices)
        program = f"""
import json
import numpy as np
from sage.all import GF, Matrix, PolynomialRing
from time import perf_counter
fixture=np.load({str(path)!r})
prime={prime}
field=GF(prime)
matrices=fixture['matrices']
started=perf_counter()
determinants=[Matrix(field, len(matrix), len(matrix), matrix.reshape(-1).tolist()).det() for matrix in matrices]
determinant_seconds=perf_counter()-started
ring=PolynomialRing(field, 'x')
polynomials=[ring(row.tolist()) for row in fixture['coefficients']]
derivatives=[polynomial.derivative() for polynomial in polynomials]
points=[field(int(point)) for point in fixture['points']]
started=perf_counter()
evaluated=[[(polynomial(point), derivative(point)) for point in points] for polynomial,derivative in zip(polynomials,derivatives)]
polynomial_seconds=perf_counter()-started
print(json.dumps({{
  'determinant_seconds': determinant_seconds,
  'determinant_checksum': int(sum(map(int, determinants)) % prime),
  'polynomial_seconds': polynomial_seconds,
  'polynomial_checksum': int(sum(int(value)+int(derivative) for row in evaluated for value,derivative in row) % prime),
}}))
"""
        completed = subprocess.run(
            ["sage", "-python", "-c", program],
            check=True,
            capture_output=True,
            text=True,
        )
    return json.loads(completed.stdout)


def flint_determinants(matrices: np.ndarray, prime: int) -> tuple[float, int]:
    from flint import nmod_mat

    started = perf_counter()
    values = [
        int(
            nmod_mat(
                len(matrix),
                len(matrix),
                matrix.reshape(-1).tolist(),
                prime,
            ).det()
        )
        for matrix in matrices
    ]
    return perf_counter() - started, sum(values) % prime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prime = 1_000_000_007
    rng = np.random.default_rng(0xF13D)
    coefficients = rng.integers(
        0, prime, size=(128, 65), dtype=np.uint32
    )
    points = rng.integers(0, prime, size=1024, dtype=np.uint32)
    matrices = rng.integers(
        0, prime, size=(1000, 8, 8), dtype=np.uint32
    )
    sage = sage_baseline(coefficients, points, matrices, prime)
    flint_seconds, flint_checksum = flint_determinants(matrices, prime)

    polynomial_records: dict[str, object] = {}
    with ModularPolynomialPlan(coefficients, prime=prime) as plan:
        reference = plan.evaluate(
            points[:31], derivative=True, backend="reference"
        )
        native_sample = plan.evaluate(
            points[:31], derivative=True, threads=8, backend="native"
        )
        np.testing.assert_array_equal(native_sample.values, reference.values)
        np.testing.assert_array_equal(
            native_sample.derivatives, reference.derivatives
        )
        for backend in ("native", "hip"):
            if backend == "hip" and not hip_modular_available():
                continue
            setup_started = perf_counter()
            result = plan.evaluate(
                points, derivative=True, threads=8, backend=backend
            )
            setup_and_first = perf_counter() - setup_started
            walls = []
            kernels = []
            for _ in range(args.repeats):
                started = perf_counter()
                result = plan.evaluate(
                    points, derivative=True, threads=8, backend=backend
                )
                walls.append(perf_counter() - started)
                kernels.append(result.elapsed_seconds)
            assert result.derivatives is not None
            polynomial_records[backend] = {
                "setup_and_first_seconds": setup_and_first,
                "seconds": walls,
                "median_seconds": median(walls),
                "kernel_median_seconds": median(kernels),
                "checksum": checksum(
                    result.values, result.derivatives, prime=prime
                ),
            }

    determinant_records: dict[str, object] = {}
    with ModularDeterminantPlan(8, prime=prime) as plan:
        reference = plan.determinants(
            matrices[:20], backend="reference"
        )
        for backend in ("native", "hip"):
            if backend == "hip" and not hip_modular_available():
                continue
            setup_started = perf_counter()
            result = plan.determinants(
                matrices, threads=8, backend=backend
            )
            setup_and_first = perf_counter() - setup_started
            walls = []
            kernels = []
            for _ in range(args.repeats):
                started = perf_counter()
                result = plan.determinants(
                    matrices, threads=8, backend=backend
                )
                walls.append(perf_counter() - started)
                kernels.append(result.elapsed_seconds)
            np.testing.assert_array_equal(
                result.determinants[:20], reference.determinants
            )
            determinant_records[backend] = {
                "setup_and_first_seconds": setup_and_first,
                "seconds": walls,
                "median_seconds": median(walls),
                "kernel_median_seconds": median(kernels),
                "checksum": checksum(result.determinants, prime=prime),
            }

    native_polynomial = polynomial_records["native"]
    native_determinant = determinant_records["native"]
    assert isinstance(native_polynomial, dict)
    assert isinstance(native_determinant, dict)
    assert native_polynomial["checksum"] == sage["polynomial_checksum"]
    assert native_determinant["checksum"] == sage["determinant_checksum"]
    assert native_determinant["checksum"] == flint_checksum
    payload = {
        "schema": 1,
        "operation": "exact-modular-batches",
        "prime": prime,
        "polynomial_shape": [128, 65, 1024],
        "determinant_shape": [1000, 8, 8],
        "sage": sage,
        "flint_determinant_seconds": flint_seconds,
        "polynomials": polynomial_records,
        "determinants": determinant_records,
    }
    if "hip" in polynomial_records:
        hip = polynomial_records["hip"]
        assert isinstance(hip, dict)
        payload["sage_polynomial_over_hip_best_wall"] = (
            sage["polynomial_seconds"] / min(hip["seconds"])
        )
        payload["native_polynomial_over_hip_best_wall"] = (
            min(native_polynomial["seconds"]) / min(hip["seconds"])
        )
    payload["sage_determinant_over_native_best_wall"] = (
        sage["determinant_seconds"] / min(native_determinant["seconds"])
    )
    payload["flint_determinant_over_native_best_wall"] = (
        flint_seconds / min(native_determinant["seconds"])
    )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="ascii")


if __name__ == "__main__":
    main()
