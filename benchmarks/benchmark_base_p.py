#!/usr/bin/env python3
"""Benchmark the base-p digit codec against the routes it replaces.

Routes per fixture and operation:

- ``reference``: the executable pure-Python specification in
  ``fast_math.base_p``.
- ``numpy``: the vectorized NumPy loops a scratch session writes by hand.
- ``native``: the fast-math native backend through the public API.
- ``hand_written_cxx``: the compiled standalone baseline program in
  ``benchmarks/baselines/base_p_hand_written.cpp``, a faithful transcription
  of the recurring census-program loops (full-space fixtures only).

All routes must produce identical complete outputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median
from time import perf_counter, process_time

import numpy as np

from fast_math import (
    base_p_decode,
    base_p_encode,
    base_p_negation_representatives,
    base_p_scalar_classes,
)

BASELINE_SOURCE = (
    Path(__file__).resolve().parent / "baselines" / "base_p_hand_written.cpp"
)


def numpy_decode(
    indices: np.ndarray,
    prime: int,
    width: int,
) -> np.ndarray:
    digits = np.empty((len(indices), width), dtype=np.uint64)
    remaining = indices.astype(np.uint64, copy=True)
    for dimension in range(width):
        digits[:, dimension] = remaining % prime
        remaining //= prime
    return digits


def numpy_encode(digits: np.ndarray, prime: int) -> np.ndarray:
    powers = prime ** np.arange(digits.shape[1], dtype=np.uint64)
    return (digits * powers).sum(axis=1)


def numpy_negation_representatives(
    indices: np.ndarray,
    prime: int,
    width: int,
) -> np.ndarray:
    negated = numpy_encode((prime - numpy_decode(indices, prime, width)) % prime, prime)
    return np.minimum(indices, negated)


def numpy_scalar_classes(
    indices: np.ndarray,
    prime: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    digits = numpy_decode(indices, prime, width)
    powers = prime ** np.arange(width, dtype=np.uint64)
    best = None
    for factor in range(1, prime):
        candidate = ((digits * factor % prime) * powers).sum(axis=1)
        best = candidate if best is None else np.minimum(best, candidate)
    representatives = best
    representatives[~digits.any(axis=1)] = 0
    unique, class_ids = np.unique(representatives, return_inverse=True)
    return class_ids.astype(np.uint32), representatives


def measure(
    call,
    repeats: int,
    iterations: int,
) -> tuple[list[float], list[float]]:
    wall_seconds: list[float] = []
    cpu_seconds: list[float] = []
    for _ in range(repeats):
        wall_started = perf_counter()
        cpu_started = process_time()
        for _ in range(iterations):
            result = call()
        cpu_seconds.append((process_time() - cpu_started) / iterations)
        wall_seconds.append((perf_counter() - wall_started) / iterations)
    return wall_seconds, cpu_seconds, result


def run_hand_written(
    prime: int,
    width: int,
    repeats: int,
    binary: Path | None,
) -> dict | None:
    if binary is None or not binary.is_file():
        return None
    completed = subprocess.run(
        [str(binary), str(prime), str(width), str(repeats)],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(20260823)
    full_spaces = {
        "p5_n6_full_space": (5, 6),
        "p13_n4_full_space": (13, 4),
        "p7_n6_full_space": (7, 6),
    }
    random_batches = {
        "p5_n6_random_1m": rng.choice(5**6, size=1_000_000, replace=True),
    }

    with tempfile.TemporaryDirectory() as temporary:
        binary: Path | None = None
        compiler = shutil.which("c++") or shutil.which("g++")
        if compiler is not None:
            binary = Path(temporary) / "base_p_hand_written"
            compiled = subprocess.run(
                [
                    compiler,
                    "-O2",
                    "-std=c++20",
                    str(BASELINE_SOURCE),
                    "-o",
                    str(binary),
                ],
                capture_output=True,
                text=True,
            )
            if compiled.returncode != 0:
                print(compiled.stderr, file=sys.stderr)
                binary = None

        results = {
            "repeats": args.repeats,
            "iterations_per_repeat": args.iterations,
            "threads": args.threads,
            "machine": sys.platform,
            "fixtures": {},
        }

        fixtures: dict[str, tuple[int, int, np.ndarray]] = {}
        for name, (prime, width) in full_spaces.items():
            fixtures[name] = (prime, width, np.arange(prime**width, dtype=np.uint64))
        for name, batch in random_batches.items():
            fixtures[name] = (5, 6, batch.astype(np.uint64))

        for name, (prime, width, indices) in fixtures.items():
            fixture_results: dict = {
                "prime": prime,
                "width": width,
                "element_count": len(indices),
            }
            operations = {
                "decode": lambda: base_p_decode(
                    indices, prime, width, backend="native", threads=args.threads
                ),
                "negation": lambda: base_p_negation_representatives(
                    indices, prime, width, backend="native", threads=args.threads
                ),
                "classes": lambda: base_p_scalar_classes(
                    indices, prime, width, backend="native", threads=args.threads
                ),
            }
            native_outputs = {
                key: call() for key, call in operations.items()
            }

            # NumPy hand-rolled route.
            numpy_operations = {
                "decode": lambda: numpy_decode(indices, prime, width),
                "negation": lambda: numpy_negation_representatives(
                    indices, prime, width
                ),
                "classes": lambda: numpy_scalar_classes(indices, prime, width),
            }

            # Executable reference specification (pure Python; small fixtures).
            reference_supported = len(indices) <= 40_000
            if reference_supported:
                reference_operations = {
                    "decode": lambda: base_p_decode(
                        indices, prime, width, backend="reference", threads=0
                    ),
                    "negation": lambda: base_p_negation_representatives(
                        indices, prime, width, backend="reference", threads=0
                    ),
                    "classes": lambda: base_p_scalar_classes(
                        indices, prime, width, backend="reference", threads=0
                    ),
                }

            for key in ("decode", "negation", "classes"):
                entry: dict = {}
                wall, cpu, native_output = measure(
                    operations[key], args.repeats, args.iterations
                )
                entry["native_wall_median"] = median(wall)
                entry["native_wall_seconds"] = wall
                entry["native_cpu_median"] = median(cpu)

                expected = native_outputs[key]
                if key == "classes":
                    np.testing.assert_array_equal(native_output.class_ids, expected.class_ids)
                    np.testing.assert_array_equal(
                        native_output.representatives, expected.representatives
                    )

                numpy_wall, _, numpy_output = measure(
                    numpy_operations[key], args.repeats, args.iterations
                )
                entry["numpy_wall_median"] = median(numpy_wall)
                entry["numpy_wall_seconds"] = numpy_wall
                if key == "classes":
                    numpy_ids, numpy_reps = numpy_output
                    np.testing.assert_array_equal(numpy_ids, expected.class_ids)
                    np.testing.assert_array_equal(numpy_reps, expected.representatives)
                else:
                    np.testing.assert_array_equal(numpy_output, expected)

                if reference_supported:
                    reference_wall, _, reference_output = measure(
                        reference_operations[key], max(args.repeats // 2, 3), 1
                    )
                    entry["reference_wall_median"] = median(reference_wall)
                    if key == "classes":
                        np.testing.assert_array_equal(
                            reference_output.class_ids, expected.class_ids
                        )
                        np.testing.assert_array_equal(
                            reference_output.representatives,
                            expected.representatives,
                        )
                    else:
                        np.testing.assert_array_equal(reference_output, expected)
                    entry["reference_speedup"] = (
                        median(reference_wall) / median(wall)
                    )

                entry["numpy_speedup"] = median(numpy_wall) / median(wall)

                if key == "classes":
                    classes = base_p_scalar_classes(indices, prime, width, backend="native")
                    entry["class_count"] = int(len(np.unique(classes.representatives)))

                fixture_results[key] = entry

            hand_written = None
            if name in full_spaces:
                hand_written = run_hand_written(
                    prime,
                    width,
                    max(args.repeats, 3),
                    binary,
                )
            if hand_written is not None:
                fixture_results["hand_written_cxx"] = {
                    "decode_wall_median": hand_written["decode_seconds"],
                    "negation_wall_median": hand_written["negation_seconds"],
                    "classes_wall_median": hand_written["classes_seconds"],
                    "class_count": hand_written["class_count"],
                    "native_vs_classes_ratio": (
                        fixture_results["classes"]["native_wall_median"]
                        / hand_written["classes_seconds"]
                    ),
                    "note": (
                        "standalone single-threaded C++ baseline; the native "
                        "route includes the Python call boundary"
                    ),
                }
                assert fixture_results["classes"]["class_count"] == hand_written[
                    "class_count"
                ], "class counts disagree with the hand-written baseline"

            results["fixtures"][name] = fixture_results
            print(f"completed {name}", file=sys.stderr)

        text = json.dumps(results, indent=2, sort_keys=True) + "\n"
        print(text, end="")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
