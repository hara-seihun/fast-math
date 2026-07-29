#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Callable
from hashlib import sha256
import json
from math import comb
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from fast_math import (  # noqa: E402
    digest_u64_rows,
    induced_subgraph_profile_stack,
    induced_subgraph_profiles,
)


CLASS_COUNTS = (1, 2, 4, 11, 34, 156)
NAMESPACE = b"induced-profile/n=9/orders=1..6/canonical-classes=v1"
EDGE_COUNT_NAMESPACE = (
    b"induced-profile/n=9/orders=1..6/edge-count-classes=v1"
)


def profile_rows(row_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = np.empty((row_count, sum(CLASS_COUNTS)), dtype=np.uint64)
    begin = 0
    for order, class_count in enumerate(CLASS_COUNTS, start=1):
        probabilities = rng.dirichlet(np.full(class_count, 0.35))
        rows[:, begin : begin + class_count] = rng.multinomial(
            comb(9, order),
            probabilities,
            size=row_count,
        )
        begin += class_count
    return rows


def random_graphs(
    graph_count: int,
    vertex_count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    graphs = np.zeros((graph_count, vertex_count), dtype=np.uint64)
    for left in range(vertex_count):
        for right in range(left + 1, vertex_count):
            present = rng.integers(
                0,
                2,
                size=graph_count,
                dtype=np.uint64,
            )
            graphs[:, left] |= present << np.uint64(right)
            graphs[:, right] |= present << np.uint64(left)
    return graphs


def edge_count_lookup(order: int) -> np.ndarray:
    edge_count = order * (order - 1) // 2
    return np.fromiter(
        (mask.bit_count() for mask in range(1 << edge_count)),
        dtype=np.uint32,
        count=1 << edge_count,
    )


def sparse_profile(row: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    entries = []
    begin = 0
    for order, class_count in enumerate(CLASS_COUNTS, start=1):
        segment = row[begin : begin + class_count]
        entries.extend(
            (order, class_id, int(count))
            for class_id, count in enumerate(segment)
            if count
        )
        begin += class_count
    return tuple(entries)


def legacy_digests(
    rows: np.ndarray,
    serializer: Callable[[object], bytes],
) -> tuple[str, int]:
    aggregate = sha256()
    payload_bytes = 0
    for row in rows:
        payload = serializer(sparse_profile(row))
        aggregate.update(sha256(payload).digest())
        payload_bytes += len(payload)
    return aggregate.hexdigest(), payload_bytes


def fixed_digests(
    rows: np.ndarray,
    backend: str,
    threads: int,
) -> tuple[str, int, float]:
    result = digest_u64_rows(
        rows,
        namespace=NAMESPACE,
        backend=backend,
        threads=threads,
    )
    return (
        sha256(result.digests).hexdigest(),
        rows.nbytes + len(rows) * (len(NAMESPACE) + 37),
        result.elapsed_seconds,
    )


def timed(function, repeats: int):
    values = []
    best = None
    for _ in range(repeats):
        started = time.perf_counter()
        value = function()
        elapsed = time.perf_counter() - started
        values.append(elapsed)
        if best is None or elapsed < best[0]:
            best = (elapsed, value)
    assert best is not None
    return best, values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--graph-rows", type=int, default=50_000)
    parser.add_argument("--threads", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.rows <= 0
        or args.graph_rows <= 0
        or args.threads < 0
        or args.repeats <= 0
    ):
        raise ValueError(
            "row counts and repeats must be positive; threads nonnegative"
        )

    rows = profile_rows(args.rows, 7612)
    benchmarks = [
        (
            "legacy_sparse_repr_sha256",
            lambda: legacy_digests(
                rows,
                lambda profile: repr(profile).encode("ascii"),
            ),
        ),
        (
            "legacy_sparse_json_sha256",
            lambda: legacy_digests(
                rows,
                lambda profile: json.dumps(
                    profile,
                    separators=(",", ":"),
                ).encode("ascii"),
            ),
        ),
        (
            "fixed_u64_reference_sha256",
            lambda: fixed_digests(rows, "reference", args.threads),
        ),
        (
            "fixed_u64_native_sha256",
            lambda: fixed_digests(rows, "native", args.threads),
        ),
    ]

    records = []
    for name, function in benchmarks:
        (wall, value), samples = timed(function, args.repeats)
        aggregate, payload_bytes, *kernel = value
        record = {
            "benchmark": name,
            "row_count": args.rows,
            "field_count": rows.shape[1],
            "threads": args.threads if "native" in name else 1,
            "wall_seconds": wall,
            "rows_per_second": args.rows / wall,
            "payload_bytes": payload_bytes,
            "aggregate_sha256": aggregate,
            "samples_seconds": samples,
            "fixture": "order-nine induced profiles through order six",
        }
        if kernel:
            record["kernel_seconds"] = kernel[0]
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    graphs = random_graphs(args.graph_rows, 9, 7613)
    orders = tuple(range(1, 7))
    lookups = tuple(edge_count_lookup(order) for order in orders)

    def separate_profiles() -> np.ndarray:
        return np.concatenate(
            [
                induced_subgraph_profiles(
                    graphs,
                    order,
                    lookup,
                    threads=args.threads,
                    backend="native",
                ).counts
                for order, lookup in zip(orders, lookups, strict=True)
            ],
            axis=1,
        )

    def stacked_profiles() -> np.ndarray:
        return induced_subgraph_profile_stack(
            graphs,
            orders,
            lookups,
            threads=args.threads,
            backend="native",
        ).counts

    (separate_wall, separate), separate_samples = timed(
        separate_profiles,
        args.repeats,
    )
    (stacked_wall, stacked), stacked_samples = timed(
        stacked_profiles,
        args.repeats,
    )
    np.testing.assert_array_equal(stacked, separate)
    for name, wall, samples in (
        (
            "separate_induced_profile_orders_1_to_6_native",
            separate_wall,
            separate_samples,
        ),
        (
            "stacked_induced_profile_orders_1_to_6_native",
            stacked_wall,
            stacked_samples,
        ),
    ):
        record = {
            "benchmark": name,
            "graph_count": args.graph_rows,
            "vertex_count": 9,
            "order_count": len(orders),
            "field_count": stacked.shape[1],
            "threads": args.threads,
            "wall_seconds": wall,
            "graphs_per_second": args.graph_rows / wall,
            "count_checksum": int(np.sum(stacked, dtype=np.uint64)),
            "samples_seconds": samples,
            "fixture": "edge-count classes for induced orders one through six",
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    def stacked_and_digested() -> np.ndarray:
        profile_counts = stacked_profiles()
        return digest_u64_rows(
            profile_counts,
            namespace=EDGE_COUNT_NAMESPACE,
            threads=args.threads,
            backend="native",
        ).digests

    (combined_wall, combined), combined_samples = timed(
        stacked_and_digested,
        args.repeats,
    )
    combined_record = {
        "benchmark": "stacked_induced_profiles_and_digest_native",
        "graph_count": args.graph_rows,
        "vertex_count": 9,
        "order_count": len(orders),
        "field_count": stacked.shape[1],
        "threads": args.threads,
        "wall_seconds": combined_wall,
        "graphs_per_second": args.graph_rows / combined_wall,
        "digest_aggregate_sha256": sha256(combined).hexdigest(),
        "samples_seconds": combined_samples,
        "fixture": "edge-count classes for induced orders one through six",
    }
    records.append(combined_record)
    print(json.dumps(combined_record, sort_keys=True), flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
