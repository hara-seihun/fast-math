from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/l01479_timing_sidecars.py"
)
SPEC = importlib.util.spec_from_file_location(
    "l01479_timing_sidecars",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
sidecars = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sidecars)

def test_split_timed_payload_keeps_proof_bytes_deterministic(tmp_path) -> None:
    proof_a = tmp_path / "a/certificate.json"
    proof_b = tmp_path / "b/certificate.json"
    timing_a = tmp_path / "a/timing.json"
    timing_b = tmp_path / "b/timing.json"
    payload = {"event": "source", "rigorous": True, "elapsed_seconds": 1.25}

    sidecars.split_timed_payload(
        payload,
        proof_a,
        timing_a,
        stage="source",
    )
    sidecars.split_timed_payload(
        {**payload, "elapsed_seconds": 9.5},
        proof_b,
        timing_b,
        stage="source",
    )

    assert proof_a.read_bytes() == proof_b.read_bytes()
    assert "elapsed_seconds" not in json.loads(proof_a.read_text())
    first_timing = json.loads(timing_a.read_text())
    second_timing = json.loads(timing_b.read_text())
    assert first_timing["elapsed_seconds"] == 1.25
    assert second_timing["elapsed_seconds"] == 9.5
    assert (
        first_timing["proof_certificate_sha256"]
        == second_timing["proof_certificate_sha256"]
    )


@pytest.mark.parametrize("elapsed", [None, -1, "slow"])
def test_split_timed_payload_rejects_invalid_elapsed(
    tmp_path,
    elapsed,
) -> None:
    with pytest.raises(ValueError, match="elapsed_seconds"):
        sidecars.split_timed_payload(
            {"event": "source", "elapsed_seconds": elapsed},
            tmp_path / "certificate.json",
            tmp_path / "timing.json",
            stage="source",
        )
