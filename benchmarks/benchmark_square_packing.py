from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from fast_math import (
    HipUnavailable,
    SquareCoverHipPlan,
    oriented_square_cover_words,
    oriented_square_weighted_scores,
)


def timed(work, repeats: int):
    records = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = work()
        records.append(time.perf_counter() - started)
    return min(records), records, result


def checksum(words: np.ndarray) -> int:
    return int(np.bitwise_xor.reduce(words.reshape(-1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", type=int, default=1_000_000)
    parser.add_argument("--points", type=int, nargs="+", default=[64, 289])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(87123)
    centers = np.ascontiguousarray(
        rng.uniform(0.71, 3.29, size=(args.poses, 2))
    )
    angles = rng.uniform(0.0, np.pi / 4, size=args.poses)
    directions = np.ascontiguousarray(
        np.column_stack((np.cos(angles), np.sin(angles)))
    )
    poses = np.ascontiguousarray(np.column_stack((centers, directions)))
    cases = []
    for point_count in args.points:
        points = np.ascontiguousarray(
            rng.uniform(0.75, 3.25, size=(point_count, 2))
        )
        reference_wall, reference_times, reference = timed(
            lambda: oriented_square_cover_words(
                points,
                centers,
                directions=directions,
                uncertainty=1e-12,
                backend="reference",
            ),
            max(1, min(2, args.repeats)),
        )
        native_wall, native_times, native = timed(
            lambda: oriented_square_cover_words(
                points,
                centers,
                directions=directions,
                uncertainty=1e-12,
                threads=args.threads,
                backend="native",
            ),
            args.repeats,
        )
        np.testing.assert_array_equal(
            native.inside_words, reference.inside_words
        )
        np.testing.assert_array_equal(
            native.uncertain_words, reference.uncertain_words
        )
        weights = rng.uniform(0.0, 1.0, size=point_count)
        score_reference_wall, _, score_reference = timed(
            lambda: oriented_square_weighted_scores(
                points,
                weights,
                centers,
                directions=directions,
                uncertainty=1e-12,
                backend="reference",
            ),
            1,
        )
        score_native_wall, _, score_native = timed(
            lambda: oriented_square_weighted_scores(
                points,
                weights,
                centers,
                directions=directions,
                uncertainty=1e-12,
                threads=args.threads,
                backend="native",
            ),
            args.repeats,
        )
        np.testing.assert_allclose(
            score_native.definite_scores,
            score_reference.definite_scores,
            atol=5e-13,
        )
        np.testing.assert_allclose(
            score_native.possible_scores,
            score_reference.possible_scores,
            atol=5e-13,
        )
        case = {
            "point_count": point_count,
            "pose_count": args.poses,
            "incidence_tests": point_count * args.poses,
            "output_bytes": int(
                native.inside_words.nbytes + native.uncertain_words.nbytes
            ),
            "reference_wall_seconds": reference_wall,
            "reference_times": reference_times,
            "native_wall_seconds": native_wall,
            "native_times": native_times,
            "native_kernel_seconds": native.elapsed_seconds,
            "native_threads": native.thread_count,
            "native_simd_lanes": native.simd_lanes,
            "native_speedup": reference_wall / native_wall,
            "native_tests_per_second": (
                point_count * args.poses / native_wall
            ),
            "checksum": checksum(native.inside_words),
            "score_reference_wall_seconds": score_reference_wall,
            "score_native_wall_seconds": score_native_wall,
            "score_native_kernel_seconds": score_native.elapsed_seconds,
            "score_native_speedup": score_reference_wall / score_native_wall,
        }
        try:
            plan = SquareCoverHipPlan(points)
        except (HipUnavailable, OSError, RuntimeError) as error:
            case["hip_unavailable"] = str(error)
        else:
            try:
                hip_wall, hip_times, hip = timed(
                    lambda: plan.evaluate(poses, uncertainty=1e-12),
                    args.repeats,
                )
                score_hip_wall, _, score_hip = timed(
                    lambda: plan.weighted_scores(
                        poses, weights, uncertainty=1e-12
                    ),
                    args.repeats,
                )
            finally:
                plan.close()
            hip_inside, hip_uncertain, hip_call = hip
            score_hip_definite, score_hip_possible, score_hip_call = score_hip
            np.testing.assert_array_equal(
                hip_inside, reference.inside_words
            )
            np.testing.assert_array_equal(
                hip_uncertain, reference.uncertain_words
            )
            np.testing.assert_allclose(
                score_hip_definite,
                score_reference.definite_scores,
                atol=5e-13,
            )
            np.testing.assert_allclose(
                score_hip_possible,
                score_reference.possible_scores,
                atol=5e-13,
            )
            case.update(
                {
                    "hip_persistent_wall_seconds": hip_wall,
                    "hip_times": hip_times,
                    "hip_call_seconds": hip_call,
                    "hip_speedup_over_reference": reference_wall / hip_wall,
                    "hip_speedup_over_native": native_wall / hip_wall,
                    "hip_tests_per_second": (
                        point_count * args.poses / hip_wall
                    ),
                    "score_hip_persistent_wall_seconds": score_hip_wall,
                    "score_hip_call_seconds": score_hip_call,
                    "score_hip_speedup_over_native": (
                        score_native_wall / score_hip_wall
                    ),
                }
            )
        cases.append(case)
    record = {
        "benchmark": "oriented-square-cover-words",
        "seed": 87123,
        "threads": args.threads,
        "uncertainty": 1e-12,
        "cases": cases,
    }
    payload = json.dumps(record, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
