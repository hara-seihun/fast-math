"""Deterministically separate elapsed telemetry from proof certificates."""

from __future__ import annotations

import hashlib
import json
from numbers import Real
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="ascii",
    )


def split_timed_payload(
    payload: dict[str, Any],
    proof_path: Path,
    timing_path: Path,
    *,
    stage: str,
) -> dict[str, Any]:
    proof = dict(payload)
    elapsed = proof.pop("elapsed_seconds", None)
    if not isinstance(elapsed, Real) or elapsed < 0:
        raise ValueError("timed certificate requires nonnegative elapsed_seconds")
    _write_json(proof_path, proof)
    timing = {
        "event": "l01479_certificate_timing",
        "role": "non_proof_telemetry",
        "stage": stage,
        "elapsed_seconds": float(elapsed),
        "proof_certificate_sha256": hashlib.sha256(
            proof_path.read_bytes()
        ).hexdigest(),
    }
    _write_json(timing_path, timing)
    return proof


def split_timed_file(
    raw_path: Path,
    proof_path: Path,
    timing_path: Path,
    *,
    stage: str,
) -> dict[str, Any]:
    payload = json.loads(raw_path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise ValueError("timed certificate must be a JSON object")
    return split_timed_payload(
        payload,
        proof_path,
        timing_path,
        stage=stage,
    )
