#!/usr/bin/env python3
"""Benchmark the shift-divisor gate scan (fast_math.shift_gates).

Measures v/s throughput per backend on the Erdos 647 gate (modulus 360360)
at campaign scale.  HIP slabs are sized to fill each class-row chunk; the
reference backend runs a proportionally smaller window and reports its own
rate.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from fast_math._native import native_available, native_version  # noqa: E402
from fast_math.hip import hip_shift_gates_available  # noqa: E402
from fast_math.shift_gates import ShiftGateScanPlan, derive_shift_gate  # noqa: E402

CASES = {
    "smoke": {"chunk_fraction": 1 / 64, "reference_count": 100_000},
    "medium": {"chunk_fraction": 1 / 4, "reference_count": 1_000_000},
    "full": {"chunk_fraction": 1.0, "reference_count": 4_000_000},
}

V_START = 1_710_000_000_000  # just past the certified-empty region


def run(case: str) -> dict:
    spec = CASES[case]
    gate = derive_shift_gate(360360)
    plan = ShiftGateScanPlan(gate)
    slab = int(plan.wheel * (1 << 18) * spec["chunk_fraction"])
    results = {}

    started = time.perf_counter()
    survivors, stats = plan.scan_reference(V_START, spec["reference_count"])
    elapsed = time.perf_counter() - started
    results["reference"] = {
        "v_count": spec["reference_count"],
        "seconds": elapsed,
        "v_per_second": spec["reference_count"] / elapsed,
        "survivors": int(stats.survivors),
    }

    backends = []
    if native_available():
        backends.append("native")
    if hip_shift_gates_available():
        backends.append("hip")
    for backend in backends:
        started = time.perf_counter()
        survivors, stats = plan.scan(V_START, slab, backend=backend)
        elapsed = time.perf_counter() - started
        results[backend] = {
            "v_count": slab,
            "seconds": elapsed,
            "v_per_second": slab / elapsed,
            "survivors": int(stats.survivors),
            "sieve_survivors": int(stats.sieve_survivors),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), default="medium")
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args()

    payload = {
        "benchmark": "shift_gates",
        "case": arguments.case,
        "platform": platform.platform(),
        "native_version": native_version(),
        "results": run(arguments.case),
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if arguments.json is not None:
        arguments.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
