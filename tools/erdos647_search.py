#!/usr/bin/env python3
"""Erdos 647 witness campaign: scan n = 360360 * v for the divisor condition.

A witness settles Erdos problem 647 affirmatively: an n > 24 with
tau(n - j) <= j + 2 for every shift 1 <= j < n.  The scan runs the derived
shift gate (fast_math.shift_gates) over a v-interval in resumable slabs;
each gate survivor then gets an exact deep check by direct factorization,
ascending through the unsieved shifts until a budget is violated or the
witness cap is cleared.

Shifts covered by the gate's sieveable forms are guaranteed within budget
(coprime part prime or prime square implies tau(n - j) = tau(F) * 2 or
tau(F) * 3 <= j + 2), so the deep check visits only the others.  The check
runs ascending to --deep-cap, far past the largest tau any number of this
magnitude can reach; a v that clears it is reported as a WITNESS and the
campaign halts for verification.

State lives in a JSON file (--state); rerunning resumes after the last
completed slab.  Every survivor and its death shift is appended to the
survivor log (state path + .survivors.jsonl).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from flint import fmpz  # noqa: E402

from fast_math.shift_gates import ShiftGateScanPlan, derive_shift_gate  # noqa: E402

MODULUS = 360360
# Campaign default: from just past the certified-empty region (least witness
# > 6.16e17, i.e. v > 1.7095e12) through n ~ 9.75e19.
DEFAULT_V_START = 1_709_500_000_000
DEFAULT_V_END = 270_549_000_000_000


def tau(x: int) -> int:
    out = 1
    for _, e in fmpz(x).factor():
        out *= int(e) + 1
    return out


def deep_check(v: int, sieved_shifts: frozenset[int], cap: int) -> tuple[bool, int, int]:
    """Exact check of every unsieved shift up to cap.

    Returns (is_witness_candidate, death_shift, death_tau); death fields are
    (0, 0) when the cap is cleared.
    """
    n = MODULUS * v
    for j in range(1, cap + 1):
        if j in sieved_shifts:
            continue
        t = tau(n - j)
        if t > j + 2:
            return False, j, t
    return True, 0, 0


def load_state(path: Path, v_start: int, v_end: int, slab: int) -> dict:
    if path.exists():
        state = json.loads(path.read_text())
        if state["v_end"] != v_end or state["slab"] != slab:
            raise SystemExit(
                f"state file {path} was created with different bounds; "
                "delete it or match --v-start/--v-end/--slab"
            )
        return state
    return {
        "v_start": v_start,
        "v_end": v_end,
        "slab": slab,
        "next_v": v_start,
        "slabs_done": 0,
        "survivors": 0,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--v-start", type=int, default=DEFAULT_V_START)
    parser.add_argument("--v-end", type=int, default=DEFAULT_V_END)
    parser.add_argument("--slab", type=int, default=None, help="v per slab (default: wheel * 2^18)")
    parser.add_argument("--backend", default="hip", choices=["hip", "native", "reference"])
    parser.add_argument(
        "--deep-cap",
        type=int,
        default=50_000,
        help="deep-check shift cap; j + 2 beyond any tau at this magnitude",
    )
    parser.add_argument("--max-slabs", type=int, default=None, help="stop after N slabs (smoke runs)")
    arguments = parser.parse_args()

    gate = derive_shift_gate(MODULUS)
    plan = ShiftGateScanPlan(gate)
    slab = arguments.slab or plan.wheel * (1 << 18)
    sieved = frozenset(f.shift for f in gate.forms)

    state = load_state(arguments.state, arguments.v_start, arguments.v_end, slab)
    survivor_log = arguments.state.with_suffix(arguments.state.suffix + ".survivors.jsonl")

    total = state["v_end"] - state["v_start"]
    slabs_run = 0
    while state["next_v"] < state["v_end"]:
        if arguments.max_slabs is not None and slabs_run >= arguments.max_slabs:
            break
        v0 = state["next_v"]
        count = min(slab, state["v_end"] - v0)
        started = time.perf_counter()
        survivors, stats = plan.scan(v0, count, backend=arguments.backend)
        scan_seconds = time.perf_counter() - started

        for v in survivors.tolist():
            witness, death_shift, death_tau = deep_check(int(v), sieved, arguments.deep_cap)
            record = {
                "v": int(v),
                "n": MODULUS * int(v),
                "death_shift": death_shift,
                "death_tau": death_tau,
            }
            with survivor_log.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            state["survivors"] += 1
            if witness:
                print(
                    f"WITNESS CANDIDATE: v={v} n={MODULUS * int(v)} cleared "
                    f"every shift to {arguments.deep_cap}; halting for verification",
                    flush=True,
                )
                state["witness"] = int(v)
                arguments.state.write_text(json.dumps(state, indent=2) + "\n")
                raise SystemExit(647)

        state["next_v"] = v0 + count
        state["slabs_done"] += 1
        slabs_run += 1
        arguments.state.write_text(json.dumps(state, indent=2) + "\n")
        done = state["next_v"] - state["v_start"]
        print(
            f"slab {state['slabs_done']}: v=[{v0}, {v0 + count}) "
            f"{scan_seconds:.1f}s {count / scan_seconds:.2e} v/s "
            f"survivors +{len(survivors)} (total {state['survivors']}) "
            f"progress {done / total:.1%}",
            flush=True,
        )

    if state["next_v"] >= state["v_end"]:
        print(
            f"campaign complete: [{state['v_start']}, {state['v_end']}) scanned, "
            f"{state['survivors']} gate survivors, no witness"
            if "witness" not in state
            else f"campaign halted with witness candidate v={state['witness']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
