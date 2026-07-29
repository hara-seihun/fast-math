from __future__ import annotations

from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from lambda_fast._native import load_library

from fast_math.ci import (
    atom_subsets_to_element_words,
    canonicalize_cayley_graphs,
    enumerate_subset_orbits,
)

try:
    NAUTY_AVAILABLE = hasattr(
        load_library(),
        "fast_math_canonical_digraphs_nauty_u64",
    )
except (OSError, RuntimeError):
    NAUTY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not NAUTY_AVAILABLE,
    reason="native nauty canonicalization is not built",
)


ROOT = Path(__file__).resolve().parents[1]
ATLAS_ROUTE = (
    ROOT.parent
    / "problems"
    / "cayley-ci"
    / "scratch"
    / "atlas--dihedral-generalized-dihedral-first-exact-defect-atlas"
)


def _load_builder():
    name = "_fast_math_ci_atlas_builder"
    spec = importlib.util.spec_from_file_location(
        name,
        ATLAS_ROUTE / "build_atlas.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the retained atlas builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fiber_bytes(rows: list[dict[str, object]]) -> bytes:
    normalized = sorted(
        (
            {
                "ci_multiplicity": int(row["ci_multiplicity"]),
                "orbit_representatives": sorted(
                    int(value, 16)
                    for value in row["orbit_representatives"]
                ),
                "raw_connection_sets": int(row["raw_connection_sets"]),
            }
            for row in rows
        ),
        key=lambda row: row["orbit_representatives"],
    )
    return b"".join(
        (
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
        for row in normalized
    )


def test_native_pipeline_matches_complete_retained_ci_atlas() -> None:
    builder = _load_builder()
    atlas = json.loads(
        (ATLAS_ROUTE / "build" / "atlas.json").read_text(encoding="ascii")
    )
    summaries = {
        row["group"]: row
        for row in atlas["groups"]
    }
    rows_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    with (ATLAS_ROUTE / "build" / "fibers.jsonl").open(
        encoding="ascii"
    ) as handle:
        for line in handle:
            row = json.loads(line)
            rows_by_group[row["group"]].append(row)

    checked_orbits = 0
    checked_fibers = 0
    for group_spec in builder.atlas_specs():
        model = builder.build_group(group_spec)
        action = np.asarray(model.atom_permutations, dtype=np.uint32)
        partition = enumerate_subset_orbits(
            action,
            backend="native",
        )
        masks = np.asarray(partition.representatives, dtype=np.uint64)
        connections = atom_subsets_to_element_words(
            masks,
            model.atoms,
            group_order=len(model.elements),
        )
        canonical = canonicalize_cayley_graphs(
            np.asarray(model.multiply_table, dtype=np.uint32),
            connections,
            inverse_indices=np.asarray(
                model.inverse_indices,
                dtype=np.uint32,
            ),
            construction_backend="native",
            canonical_backend="native",
        )

        observed_fibers: dict[int, list[int]] = defaultdict(list)
        for mask, class_id in zip(
            map(int, masks),
            map(int, canonical.class_ids),
            strict=True,
        ):
            observed_fibers[class_id].append(mask)
        orbit_size = {
            int(mask): int(size)
            for mask, size in zip(
                masks,
                partition.orbit_sizes,
                strict=True,
            )
        }
        observed_rows = [
            {
                "ci_multiplicity": len(fiber),
                "orbit_representatives": [
                    f"0x{mask:x}"
                    for mask in sorted(fiber)
                ],
                "raw_connection_sets": sum(
                    orbit_size[mask]
                    for mask in fiber
                ),
            }
            for fiber in observed_fibers.values()
        ]

        expected_rows = rows_by_group[group_spec.slug]
        summary = summaries[group_spec.slug]
        assert len(masks) == summary["ci_orbits"]
        assert len(observed_rows) == summary["unlabeled_cayley_graphs"]
        assert _fiber_bytes(observed_rows) == _fiber_bytes(expected_rows)
        checked_orbits += len(masks)
        checked_fibers += len(observed_rows)

    assert checked_orbits == 11_664
    assert checked_fibers == 9_606
