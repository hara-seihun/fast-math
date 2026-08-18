#!/usr/bin/env python3
"""Benchmark one-worker versus ordered-Arb L=0.1479 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTE = (
    RESEARCH_ROOT
    / "problems/riemann-hypothesis/scratch"
    / "adjacent-unconditional--debruijn-newman-l01479-phase-faithful-"
    "finite-bridge"
)
DEFAULT_SCRATCH = (
    RESEARCH_ROOT
    / "problems/riemann-hypothesis/scratch"
    / "proof-system--fast-math-l01479-source-cache-integration"
)
CHECKPOINT_RELATIVE = Path("build/checkpoints/N863989-Q40")
CACHE_PATHS = (
    Path("source/primary-f64le.bin"),
    Path("source/low-dual-f64le.bin"),
    Path("source/transformed-complex-f64le.bin"),
    Path("weights/weights-u64-u64-f64le-lower-upper.bin"),
)
CERTIFICATE_PATHS = (
    Path("source/certificate.json"),
    Path("weights/certificate.json"),
    Path("l1/certificate.json"),
    Path("roundoff/certificate.json"),
    Path("scalar-certificate.json"),
    Path("replay/cache-input-certificate.json"),
    Path("replay/transcendental-certificate.json"),
    Path("replay/replay.jsonl"),
    Path("replay/repaired-certificate.json"),
)
TIMING_PATHS = (
    Path("source/timing.json"),
    Path("weights/timing.json"),
    Path("roundoff/timing.json"),
)
PROVENANCE_DIGEST_FIELDS = {
    "cache_derived_replay_sha256",
    "cache_input_certificate_sha256",
    "input_l1_certificate_sha256",
    "replay_sha256",
    "roundoff_certificate_sha256",
    "scalar_certificate_sha256",
    "source_cache_certificate_sha256",
    "source_certificate_sha256",
    "transcendental_certificate_sha256",
    "weight_certificate_sha256",
    "weight_interval_cache_certificate_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copytree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def stage_route(source_route: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    copytree(source_route / "source", destination / "source")
    if not (destination / "source/l01479_timing_sidecars.py").is_file():
        raise ValueError("retained route has no timing-sidecar implementation")
    runner = destination / "source/run_repaired_checkpoint.py"
    runner_text = runner.read_text(encoding="ascii")
    python_binding = 'PYTHON = ROOT / ".venv/bin/python"'
    if runner_text.count(python_binding) != 1:
        raise ValueError("staged runner Python binding changed")
    runner.write_text(
        runner_text.replace(
            python_binding,
            "PYTHON = Path(sys.executable)",
            1,
        ),
        encoding="ascii",
    )
    shutil.copy2(
        source_route / "rigorous_scalar_bounds.py",
        destination / "rigorous_scalar_bounds.py",
    )
    (destination / ".venv").symlink_to(
        Path(sys.prefix), target_is_directory=True
    )

    build = destination / "build"
    pinned = build / "pinned-kernel"
    pinned.mkdir(parents=True)
    retained_evidence = (
        source_route
        / CHECKPOINT_RELATIVE
        / "replay/lambda-fast-evidence"
    )
    retained_library = retained_evidence / "libfast_math.dylib"
    shutil.copy2(retained_library, pinned / retained_library.name)
    (pinned / "kernel.sha256").write_text(
        sha256_file(retained_library) + "\n", encoding="ascii"
    )
    shutil.copy2(
        source_route / "build/mollifier-L01479-P2000-full-f64le.bin",
        build / "mollifier-L01479-P2000-full-f64le.bin",
    )
    native_cache = source_route / "build/native-binary-cache"
    if native_cache.is_dir():
        copytree(native_cache, build / "native-binary-cache")

    lambda_root = destination / "lambda-fast-root"
    copytree(
        RESEARCH_ROOT / "fast-math/python/lambda_fast",
        lambda_root / "python/lambda_fast",
    )
    for relative in (
        "python/fast_math/_native.py",
        "python/lambda_fast/two_level.py",
        "cpp/include/lambda_fast.h",
        "cpp/src/two_level.cpp",
    ):
        target = lambda_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(retained_evidence / "source" / relative, target)


def run_checkpoint(
    *,
    source_route: Path,
    run_root: Path,
    threads: int,
    chunk_size: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    stage_route(source_route, run_root)
    pinned_library = run_root / "build/pinned-kernel/libfast_math.dylib"
    pinned_digest = (
        run_root / "build/pinned-kernel/kernel.sha256"
    ).read_text(encoding="ascii").strip()
    environment = os.environ.copy()
    python_path = str(RESEARCH_ROOT / "fast-math/python")
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    environment.update(
        {
            "FAST_MATH_ARB_THREADS": str(threads),
            "FAST_MATH_ARB_CHUNK_SIZE": str(chunk_size),
            "FAST_MATH_LIBRARY": str(pinned_library),
            "FAST_MATH_ROOT": str(RESEARCH_ROOT / "fast-math"),
            "LAMBDA_FAST_LIBRARY": str(pinned_library),
            "LAMBDA_FAST_PIN_SHA256": pinned_digest,
            "LAMBDA_FAST_ROOT": str(run_root / "lambda-fast-root"),
            "L01479_PREGENERATED_SUPPORT": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONPATH": python_path,
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    command = [
        sys.executable,
        str(run_root / "source/run_repaired_checkpoint.py"),
        "863989",
        "--ratio",
        "40",
        "--precision",
        "256",
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=run_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    wall_seconds = time.monotonic() - started
    (run_root / "runner.stdout").write_text(
        completed.stdout, encoding="ascii"
    )
    (run_root / "runner.stderr").write_text(
        completed.stderr, encoding="ascii"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"checkpoint failed with exit {completed.returncode}: "
            f"{completed.stderr[-4000:]}"
        )
    source_certificate_path = (
        run_root / CHECKPOINT_RELATIVE / "source/certificate.json"
    )
    source_timing = json.loads(
        (
            run_root / CHECKPOINT_RELATIVE / "source/timing.json"
        ).read_text(encoding="ascii")
    )
    weight_timing = json.loads(
        (
            run_root / CHECKPOINT_RELATIVE / "weights/timing.json"
        ).read_text(encoding="ascii")
    )
    return {
        "threads": threads,
        "chunk_size": chunk_size,
        "wall_seconds": wall_seconds,
        "source_elapsed_seconds": source_timing["elapsed_seconds"],
        "source_certificate_sha256": sha256_file(
            source_certificate_path
        ),
        "weight_elapsed_seconds": weight_timing["elapsed_seconds"],
        "runner_stdout_sha256": sha256_file(run_root / "runner.stdout"),
        "runner_stderr_sha256": sha256_file(run_root / "runner.stderr"),
    }


def canonical_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: canonical_json(child)
            for key, child in value.items()
            if key != "elapsed_seconds"
            and key not in PROVENANCE_DIGEST_FIELDS
        }
    if isinstance(value, list):
        return [canonical_json(child) for child in value]
    return value


def semantic_certificate(path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [
            canonical_json(json.loads(line))
            for line in path.read_text(encoding="ascii").splitlines()
        ]
    return canonical_json(json.loads(path.read_text(encoding="ascii")))


def compare_outputs(
    retained: Path, baseline: Path, integrated: Path
) -> dict[str, Any]:
    retained_checkpoint = retained / CHECKPOINT_RELATIVE
    baseline_checkpoint = baseline / CHECKPOINT_RELATIVE
    integrated_checkpoint = integrated / CHECKPOINT_RELATIVE
    caches: dict[str, Any] = {}
    for relative in CACHE_PATHS:
        paths = {
            "retained": retained_checkpoint / relative,
            "baseline": baseline_checkpoint / relative,
            "integrated": integrated_checkpoint / relative,
        }
        hashes = {name: sha256_file(path) for name, path in paths.items()}
        caches[str(relative)] = {
            "bytes": paths["integrated"].stat().st_size,
            "sha256": hashes,
            "exact_equal": len(set(hashes.values())) == 1,
        }

    certificates: dict[str, Any] = {}
    for relative in CERTIFICATE_PATHS:
        paths = {
            "retained": retained_checkpoint / relative,
            "baseline": baseline_checkpoint / relative,
            "integrated": integrated_checkpoint / relative,
        }
        hashes = {name: sha256_file(path) for name, path in paths.items()}
        semantic = {
            name: semantic_certificate(path) for name, path in paths.items()
        }
        certificates[str(relative)] = {
            "sha256": hashes,
            "fresh_exact_equal": (
                hashes["baseline"] == hashes["integrated"]
            ),
            "legacy_exact_equal": len(set(hashes.values())) == 1,
            "semantic_equal": (
                semantic["retained"]
                == semantic["baseline"]
                == semantic["integrated"]
            ),
        }
    timings: dict[str, Any] = {}
    for relative in TIMING_PATHS:
        baseline_payload = json.loads(
            (baseline_checkpoint / relative).read_text(encoding="ascii")
        )
        integrated_payload = json.loads(
            (integrated_checkpoint / relative).read_text(encoding="ascii")
        )
        baseline_semantic = dict(baseline_payload)
        integrated_semantic = dict(integrated_payload)
        baseline_semantic.pop("elapsed_seconds")
        integrated_semantic.pop("elapsed_seconds")
        timings[str(relative)] = {
            "baseline": baseline_payload,
            "integrated": integrated_payload,
            "semantic_equal": baseline_semantic == integrated_semantic,
        }
    return {
        "caches": caches,
        "certificates": certificates,
        "timings": timings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    route = args.route.resolve()
    scratch = args.scratch.resolve()
    if route == scratch or route in scratch.parents:
        raise ValueError("scratch path must be separate from the retained route")
    if args.threads != 2:
        raise ValueError("this bounded lane is pinned to exactly two workers")
    if args.chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    scratch.mkdir(parents=True, exist_ok=True)
    runs = scratch / "full-checkpoint"
    runs.mkdir(parents=True, exist_ok=True)
    baseline_root = runs / "baseline-route"
    integrated_root = runs / "integrated-route"

    baseline = run_checkpoint(
        source_route=route,
        run_root=baseline_root,
        threads=1,
        chunk_size=args.chunk_size,
        timeout_seconds=args.timeout_seconds,
    )
    integrated = run_checkpoint(
        source_route=route,
        run_root=integrated_root,
        threads=args.threads,
        chunk_size=args.chunk_size,
        timeout_seconds=args.timeout_seconds,
    )
    comparison = compare_outputs(route, baseline_root, integrated_root)
    wall_speedup = baseline["wall_seconds"] / integrated["wall_seconds"]
    cache_exact = all(
        value["exact_equal"] for value in comparison["caches"].values()
    )
    certificate_exact = all(
        value["fresh_exact_equal"]
        for value in comparison["certificates"].values()
    )
    legacy_certificate_exact = all(
        value["legacy_exact_equal"]
        for value in comparison["certificates"].values()
    )
    certificate_semantic = all(
        value["semantic_equal"]
        for value in comparison["certificates"].values()
    )
    timing_semantic = all(
        value["semantic_equal"] for value in comparison["timings"].values()
    )
    result = {
        "benchmark": "lambda_l01479_route_integration",
        "representative_checkpoint": {
            "N": 863989,
            "ratio": 40,
            "precision_bits": 256,
        },
        "baseline": baseline,
        "integrated": integrated,
        "wall_speedup": wall_speedup,
        "cache_files_exact": cache_exact,
        "certificate_files_exact": certificate_exact,
        "legacy_certificate_files_exact": legacy_certificate_exact,
        "certificate_semantics_exact": certificate_semantic,
        "timing_sidecars_semantically_equal": timing_semantic,
        "accepted": (
            wall_speedup >= 2.0
            and cache_exact
            and certificate_exact
            and timing_semantic
        ),
        "comparison": comparison,
        "retained_route": str(route),
        "scratch": str(scratch),
    }
    output = (
        args.output.resolve()
        if args.output is not None
        else scratch / "full-checkpoint-result.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
