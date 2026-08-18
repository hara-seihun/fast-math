from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from fast_math.ci import derivative_group_orbits


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = 3**6
    powers = np.asarray([3**index for index in range(6)], dtype=np.int64)
    vectors = np.asarray(
        [[(code // (3**index)) % 3 for index in range(6)] for code in range(order)],
        dtype=np.uint8,
    )
    table = ((vectors[:, None, :] + vectors[None, :, :]) % 3 @ powers).astype(
        np.uint32
    )
    inverses = (
        (-vectors.astype(np.int16) % 3) @ powers
    ).astype(np.uint32)
    image = vectors.copy()
    gates = (
        (0, (1, 2), (1, 2), 1),
        (3, (0, 4), (2, 1), 2),
        (5, (1, 3, 4), (0, 1, 2), 1),
        (2, (0, 5), (1, 2), 2),
        (1, (2, 3, 5), (2, 0, 1), 1),
    )
    for target, controls, values, increment in gates:
        selected = np.ones(order, dtype=np.bool_)
        for coordinate, value in zip(controls, values, strict=True):
            selected &= image[:, coordinate] == value
        image[selected, target] = (image[selected, target] + increment) % 3
    bijection = (image @ powers).astype(np.uint32)
    bijection = table[int(inverses[int(bijection[0])]), bijection]
    return table, inverses, bijection


def measured(
    table: np.ndarray,
    inverses: np.ndarray,
    bijection: np.ndarray,
    backend: str,
):
    started = perf_counter()
    result = derivative_group_orbits(
        table,
        inverses,
        bijection,
        backend=backend,
    )
    return result, perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/derivative-rank6-local.json"),
    )
    args = parser.parse_args()
    table, inverses, bijection = fixture()
    native_first, native_first_seconds = measured(
        table, inverses, bijection, "native"
    )
    native_second, native_second_seconds = measured(
        table, inverses, bijection, "native"
    )
    reference, reference_seconds = measured(
        table, inverses, bijection, "reference"
    )
    np.testing.assert_array_equal(
        native_first.orbit_labels, native_second.orbit_labels
    )
    np.testing.assert_array_equal(
        native_first.orbit_labels, reference.orbit_labels
    )
    best_native = min(native_first_seconds, native_second_seconds)
    report = {
        "deterministic": True,
        "labels_sha256": hashlib.sha256(
            native_first.orbit_labels.tobytes()
        ).hexdigest(),
        "native_seconds": best_native,
        "orbit_count": len(native_first.orbits),
        "order": len(table),
        "parity": True,
        "reference_seconds": reference_seconds,
        "shape": "C3^6",
        "speedup": reference_seconds / best_native,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
