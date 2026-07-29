#!/usr/bin/env python3
"""Build and validate fast-math on Modal CPU and GPU workers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, TypeVar

import modal


PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/opt/fast-math")
Result = TypeVar("Result")

dependency_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install(
        "build-essential",
        "cmake",
        "ninja-build",
        "pkg-config",
        "libflint-dev",
    )
    .pip_install("numpy>=2.0", "pytest>=8.0")
)

gpu_dependency_image = dependency_image.pip_install(
    "cupy-cuda12x[ctk]>=13,<15"
)


def add_sources(base: modal.Image) -> modal.Image:
    return (
        base
        .add_local_dir(
            PROJECT_ROOT / "cpp",
            str(REMOTE_ROOT / "cpp"),
            copy=True,
        )
        .add_local_dir(
            PROJECT_ROOT / "arb",
            str(REMOTE_ROOT / "arb"),
            copy=True,
        )
        .add_local_dir(
            PROJECT_ROOT / "python",
            str(REMOTE_ROOT / "python"),
            copy=True,
        )
        .add_local_dir(
            PROJECT_ROOT / "tests",
            str(REMOTE_ROOT / "tests"),
            copy=True,
        )
        .add_local_dir(
            PROJECT_ROOT / "benchmarks",
            str(REMOTE_ROOT / "benchmarks"),
            copy=True,
            ignore=["**/results/**", "**/__pycache__/**"],
        )
        .add_local_file(
            PROJECT_ROOT / "CMakeLists.txt",
            str(REMOTE_ROOT / "CMakeLists.txt"),
            copy=True,
        )
    )

image = add_sources(dependency_image)
gpu_image = add_sources(gpu_dependency_image)

app = modal.App("fast-math-modal-validation")


def run_checked(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
) -> float:
    started = time.perf_counter()
    subprocess.run(command, check=True, env=env)
    return time.perf_counter() - started


def best_of_three(function: Callable[[], Result]) -> tuple[Result, float]:
    started = time.perf_counter()
    output = function()
    durations = [time.perf_counter() - started]
    for _ in range(2):
        started = time.perf_counter()
        output = function()
        durations.append(time.perf_counter() - started)
    return output, min(durations)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=1200,
)
def cpu_smoke() -> str:
    build = Path("/tmp/fast-math-build")
    compatibility_project = REMOTE_ROOT.parent / "lambda-fast"
    if not compatibility_project.exists():
        compatibility_project.symlink_to(REMOTE_ROOT, target_is_directory=True)
    configure_seconds = run_checked(
        [
            "cmake",
            "-S",
            str(REMOTE_ROOT),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DFAST_MATH_NATIVE_ARCH=ON",
            "-DFAST_MATH_USE_COMMONCRYPTO=OFF",
            "-DFAST_MATH_USE_NAUTY=OFF",
        ]
    )
    build_seconds = run_checked(
        ["cmake", "--build", str(build), "--parallel", "4"]
    )
    ctest_seconds = run_checked(
        ["ctest", "--test-dir", str(build), "--output-on-failure"]
    )
    environment = os.environ.copy()
    environment.update(
        {
            "FAST_MATH_LIBRARY": str(build / "libfast_math.so"),
            "PYTHONPATH": str(REMOTE_ROOT / "python"),
        }
    )
    pytest_seconds = run_checked(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            str(REMOTE_ROOT / "tests"),
        ],
        env=environment,
    )
    return json.dumps(
        {
            "status": "ok",
            "platform": "modal-cpu",
            "configure_seconds": configure_seconds,
            "build_seconds": build_seconds,
            "ctest_seconds": ctest_seconds,
            "pytest_seconds": pytest_seconds,
        },
        sort_keys=True,
    )


@app.function(
    image=gpu_image,
    gpu="L4",
    cpu=2,
    memory=8192,
    timeout=600,
)
def gpu_smoke() -> str:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REMOTE_ROOT / "python")
    pytest_seconds = run_checked(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            str(REMOTE_ROOT / "tests" / "test_affine.py"),
            str(REMOTE_ROOT / "tests" / "test_cuda.py"),
        ],
        env=environment,
    )
    import cupy as cp
    import numpy as np

    sys.path.insert(0, str(REMOTE_ROOT / "python"))
    from fast_math import AffineCudaPlan, AffineNumpyPlan

    rng = np.random.default_rng(8401)
    base = (
        rng.normal(size=4097) + 1j * rng.normal(size=4097)
    ).astype(np.complex64)
    basis = (
        rng.normal(size=(20, base.size))
        + 1j * rng.normal(size=(20, base.size))
    ).astype(np.complex64)
    steps = rng.normal(size=(1024, basis.shape[0])).astype(np.float32)
    edge_slice = slice(1024, 3073)
    numpy_plan = AffineNumpyPlan(base, basis)
    cuda_plan = AffineCudaPlan(base, basis)
    numpy_plan.contour_metrics(
        steps[:32], edge_slice=edge_slice, batch_size=32
    )
    cuda_plan.contour_metrics(
        steps[:32], edge_slice=edge_slice, batch_size=32
    )
    numpy_metrics, numpy_seconds = best_of_three(
        lambda: numpy_plan.contour_metrics(
            steps, edge_slice=edge_slice, batch_size=256
        )
    )
    cuda_metrics, cuda_seconds = best_of_three(
        lambda: cuda_plan.contour_metrics(
            steps, edge_slice=edge_slice, batch_size=256
        )
    )
    winding_disagreements = int(
        np.count_nonzero(
            numpy_metrics.windings != cuda_metrics.windings
        )
    )
    if winding_disagreements:
        raise AssertionError(
            "CUDA affine metrics disagree with NumPy windings"
        )

    properties = cp.cuda.runtime.getDeviceProperties(0)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    return json.dumps(
        {
            "status": "ok",
            "platform": "modal-gpu",
            "device": device_name,
            "pytest_seconds": pytest_seconds,
            "affine_population": len(steps),
            "affine_point_count": len(base),
            "affine_numpy_seconds": numpy_seconds,
            "affine_cuda_seconds": cuda_seconds,
            "affine_cuda_speedup": numpy_seconds / cuda_seconds,
            "affine_winding_disagreements": winding_disagreements,
        },
        sort_keys=True,
    )


@app.local_entrypoint()
def main(target: str = "all", gpu: str = "L4") -> None:
    if target not in {"all", "cpu", "gpu"}:
        raise ValueError("target must be all, cpu, or gpu")
    if target in {"all", "cpu"}:
        print(cpu_smoke.remote())
    if target in {"all", "gpu"}:
        print(gpu_smoke.with_options(gpu=gpu).remote())
