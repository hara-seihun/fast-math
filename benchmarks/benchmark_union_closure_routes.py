#!/usr/bin/env python3
"""Compare packed union-closure backends on both retained route fixtures."""

from __future__ import annotations

import argparse
from functools import partial
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter, process_time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fast-math/python"))

from fast_math import union_closed_family_masks


AUDIT_PATH = (
    ROOT
    / "problems/union-closed/scratch/"
    "minimal-counterexample--fast-math-packed-union-closure-audit/"
    "benchmark.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(
    callable_,
    repeats: int,
    iterations: int,
) -> tuple[list[float], list[float], object]:
    expected = callable_()
    wall_seconds = []
    cpu_seconds = []
    for _ in range(repeats):
        wall_started = perf_counter()
        cpu_started = process_time()
        for _ in range(iterations):
            actual = callable_()
        cpu_seconds.append((process_time() - cpu_started) / iterations)
        wall_seconds.append((perf_counter() - wall_started) / iterations)
        if actual != expected:
            raise AssertionError("route output changed between repeats")
    return wall_seconds, cpu_seconds, expected


def summarize(
    reference_wall: list[float],
    reference_cpu: list[float],
    native_wall: list[float],
    native_cpu: list[float],
) -> dict[str, object]:
    reference_wall_median = median(reference_wall)
    reference_cpu_median = median(reference_cpu)
    native_wall_median = median(native_wall)
    native_cpu_median = median(native_cpu)
    return {
        "reference_wall_seconds": reference_wall,
        "reference_cpu_seconds": reference_cpu,
        "native_wall_seconds": native_wall,
        "native_cpu_seconds": native_cpu,
        "reference_wall_median": reference_wall_median,
        "reference_cpu_median": reference_cpu_median,
        "native_wall_median": native_wall_median,
        "native_cpu_median": native_cpu_median,
        "wall_speedup": reference_wall_median / native_wall_median,
        "cpu_speedup": reference_cpu_median / native_cpu_median,
        "output_identical": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    audit = load_module("union_closure_route_audit", AUDIT_PATH)
    minimal = audit.load_module("minimal_members_route", audit.MINIMAL_PATH)
    channel = audit.load_module("channel_route", audit.CHANNEL_PATH)
    backend_call = {
        backend: partial(union_closed_family_masks, backend=backend)
        for backend in ("reference", "native")
    }

    minimal_results = {}
    minimal_outputs = {}
    for backend in ("reference", "native"):
        audit.union_closed_family_masks = backend_call[backend]
        wall, cpu, output = measure(
            lambda: audit.run_minimal_packed(minimal),
            args.repeats,
            args.iterations,
        )
        minimal_results[backend] = (wall, cpu)
        minimal_outputs[backend] = output
    if minimal_outputs["native"] != minimal_outputs["reference"]:
        raise AssertionError("minimal-members route output differs")

    channel_results = {}
    channel_outputs = {}
    for backend in ("reference", "native"):
        audit.union_closed_family_masks = backend_call[backend]
        channel.families_on_support = (
            lambda support: audit.packed_families_on_support(
                channel,
                support,
            )
        )
        wall, cpu, output = measure(
            lambda: audit.run_channel(channel),
            args.repeats,
            args.iterations,
        )
        channel_results[backend] = (wall, cpu)
        channel_outputs[backend] = output
    if channel_outputs["native"] != channel_outputs["reference"]:
        raise AssertionError("singleton-channel route output differs")

    result = {
        "repeats": args.repeats,
        "iterations_per_repeat": args.iterations,
        "minimal_members": summarize(
            *minimal_results["reference"],
            *minimal_results["native"],
        ),
        "singleton_channel": summarize(
            *channel_results["reference"],
            *channel_results["native"],
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
