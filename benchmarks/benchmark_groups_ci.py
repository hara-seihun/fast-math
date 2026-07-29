#!/usr/bin/env python3
"""Benchmark the group and Cayley-CI kernels against retained reference paths."""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
from itertools import permutations
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from fast_math.ci import (  # noqa: E402
    atom_subsets_to_element_words,
    cayley_graphs,
    coherent_configuration,
    derivative_orbit_partitions,
    enumerate_subset_orbits,
)
from fast_math.canonical import canonicalize_colored_digraphs  # noqa: E402
from fast_math.groups import (  # noqa: E402
    group_order,
    permutation_double_cosets,
    permutation_group_contains,
    permutation_orbits,
)


ATLAS_ROUTE = (
    RESEARCH_ROOT
    / "problems"
    / "cayley-ci"
    / "scratch"
    / "atlas--dihedral-generalized-dihedral-first-exact-defect-atlas"
)


def load_atlas_builder():
    name = "_fast_math_benchmark_ci_atlas_builder"
    spec = importlib.util.spec_from_file_location(
        name,
        ATLAS_ROUTE / "build_atlas.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load retained atlas builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def serialize_fibers(
    fibers: dict[int | str, list[int]],
    orbit_sizes: dict[int, int],
) -> bytes:
    rows = sorted(
        (
            {
                "ci_multiplicity": len(masks),
                "orbit_representatives": sorted(map(int, masks)),
                "raw_connection_sets": sum(
                    orbit_sizes[int(mask)]
                    for mask in masks
                ),
            }
            for masks in fibers.values()
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
        for row in rows
    )


def retained_atlas_profile(builder, models) -> tuple[bytes, int, int]:
    profiles = []
    orbit_count = 0
    fiber_count = 0
    for model in models:
        representatives, _, orbit_sizes = builder.connection_orbits(
            len(model.atoms),
            model.atom_permutations,
        )
        graphs = [
            builder.cayley_graph(model, mask)
            for mask in representatives
        ]
        labels = builder.canonical_records(graphs)
        fibers: dict[str, list[int]] = defaultdict(list)
        for mask, label in zip(representatives, labels, strict=True):
            fibers[label].append(mask)
        profiles.append(serialize_fibers(fibers, orbit_sizes))
        orbit_count += len(representatives)
        fiber_count += len(fibers)
    return b"".join(profiles), orbit_count, fiber_count


def native_atlas_profile(models, threads: int) -> tuple[bytes, int, int]:
    profiles = []
    orbit_count = 0
    fiber_count = 0
    for model in models:
        action = np.asarray(model.atom_permutations, dtype=np.uint32)
        partition = enumerate_subset_orbits(action, backend="native")
        masks = np.asarray(partition.representatives, dtype=np.uint64)
        connections = atom_subsets_to_element_words(
            masks,
            model.atoms,
            group_order=len(model.elements),
        )
        graphs = cayley_graphs(
            np.asarray(model.multiply_table, dtype=np.uint32),
            connections,
            inverse_indices=np.asarray(
                model.inverse_indices,
                dtype=np.uint32,
            ),
            threads=threads,
            backend="native",
        )
        canonical = canonicalize_colored_digraphs(
            graphs.adjacency_words,
            np.zeros(
                (len(masks), len(model.elements)),
                dtype=np.uint32,
            ),
            backend="native",
        )
        fibers: dict[int, list[int]] = defaultdict(list)
        for mask, class_id in zip(
            map(int, masks),
            map(int, canonical.class_ids),
            strict=True,
        ):
            fibers[class_id].append(mask)
        orbit_sizes = {
            int(mask): int(size)
            for mask, size in zip(
                masks,
                partition.orbit_sizes,
                strict=True,
            )
        }
        profiles.append(serialize_fibers(fibers, orbit_sizes))
        orbit_count += len(masks)
        fiber_count += len(fibers)
    return b"".join(profiles), orbit_count, fiber_count


def timed_repeats(function, repeats: int):
    values = []
    result = None
    for _ in range(repeats):
        started = perf_counter()
        result = function()
        values.append(perf_counter() - started)
    return values, result


def gap_group_query(
    builder,
    generators: np.ndarray,
    candidates: np.ndarray,
) -> tuple[int, int, tuple[bool, ...]]:
    generator_text = ",".join(
        builder.gap_cycle(tuple(map(int, generator)))
        for generator in generators
    )
    candidate_text = ",".join(
        builder.gap_cycle(tuple(map(int, candidate)))
        for candidate in candidates
    )
    degree = generators.shape[1]
    script = "\n".join(
        [
            f"g := Group([{generator_text}]);;",
            f"queries := [{candidate_text}];;",
            'Print("ORDER ", Size(g), "\\n");',
            f'Print("ORBITS ", Length(Orbits(g, [1..{degree}])), "\\n");',
            'Print("MEMBERS ");',
            (
                "for q in queries do "
                'if q in g then Print("1"); else Print("0"); fi; '
                "od;"
            ),
            'Print("\\n");',
        ]
    )
    completed = subprocess.run(
        ["mamba", "run", "-n", ".gap-env", "gap", "-q"],
        input=script + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    values = {}
    for line in completed.stdout.splitlines():
        if line.startswith("ORDER "):
            values["order"] = int(line.split()[1])
        elif line.startswith("ORBITS "):
            values["orbits"] = int(line.split()[1])
        elif line.startswith("MEMBERS "):
            values["members"] = tuple(
                value == "1"
                for value in line.split()[1]
            )
    if set(values) != {"order", "orbits", "members"}:
        raise RuntimeError(
            "could not parse GAP output:\n"
            + completed.stdout
            + completed.stderr
        )
    return values["order"], values["orbits"], values["members"]


def native_group_query(
    generators: np.ndarray,
    candidates: np.ndarray,
    threads: int,
) -> tuple[int, int, tuple[bool, ...]]:
    return (
        group_order(generators, backend="native"),
        len(permutation_orbits(generators, backend="native")),
        tuple(
            map(
                bool,
                permutation_group_contains(
                    generators,
                    candidates,
                    threads=threads,
                    backend="native",
                ),
            )
        ),
    )


def cycle_relations(order: int) -> np.ndarray:
    relations = np.full((order, order), 2, dtype=np.uint32)
    for vertex in range(order):
        relations[vertex, (vertex + 1) % order] = 1
        relations[(vertex + 1) % order, vertex] = 1
    np.fill_diagonal(relations, 0)
    return relations


def benchmark_reference_kernels(builder, models, repeats: int, threads: int):
    by_slug = {model.spec.slug: model for model in models}
    results = {}

    subset_model = by_slug["D16"]
    subset_action = np.asarray(
        subset_model.atom_permutations,
        dtype=np.uint32,
    )
    subset_outputs = {}
    subset_times = {}
    for backend in ("reference", "native"):
        subset_times[backend], subset_outputs[backend] = timed_repeats(
            lambda backend=backend: enumerate_subset_orbits(
                subset_action,
                backend=backend,
            ),
            repeats,
        )
    np.testing.assert_array_equal(
        subset_outputs["native"].representatives,
        subset_outputs["reference"].representatives,
    )
    np.testing.assert_array_equal(
        subset_outputs["native"].orbit_sizes,
        subset_outputs["reference"].orbit_sizes,
    )
    results["subset_orbits_d16"] = timing_record(
        subset_times,
        item_count=1 << len(subset_model.atoms),
    )

    cayley_model = by_slug["D24"]
    cayley_partition = enumerate_subset_orbits(
        np.asarray(cayley_model.atom_permutations, dtype=np.uint32),
        backend="native",
    )
    connections = atom_subsets_to_element_words(
        cayley_partition.representatives,
        cayley_model.atoms,
        group_order=len(cayley_model.elements),
    )
    cayley_outputs = {}
    cayley_times = {}
    for backend in ("reference", "native"):
        cayley_times[backend], cayley_outputs[backend] = timed_repeats(
            lambda backend=backend: cayley_graphs(
                np.asarray(cayley_model.multiply_table, dtype=np.uint32),
                connections,
                inverse_indices=np.asarray(
                    cayley_model.inverse_indices,
                    dtype=np.uint32,
                ),
                threads=threads,
                backend=backend,
            ),
            repeats,
        )
    np.testing.assert_array_equal(
        cayley_outputs["native"].adjacency_words,
        cayley_outputs["reference"].adjacency_words,
    )
    results["cayley_graphs_d24"] = timing_record(
        cayley_times,
        item_count=len(connections),
    )

    derivative_model = by_slug["D18"]
    derivative_maps = np.asarray(
        derivative_model.automorphisms[:32],
        dtype=np.uint32,
    )
    derivative_outputs = {}
    derivative_times = {}
    for backend in ("reference", "native"):
        derivative_times[backend], derivative_outputs[backend] = timed_repeats(
            lambda backend=backend: derivative_orbit_partitions(
                np.asarray(
                    derivative_model.multiply_table,
                    dtype=np.uint32,
                ),
                np.asarray(
                    derivative_model.inverse_indices,
                    dtype=np.uint32,
                ),
                derivative_maps,
                threads=threads,
                backend=backend,
            ),
            repeats,
        )
    np.testing.assert_array_equal(
        derivative_outputs["native"].orbit_labels,
        derivative_outputs["reference"].orbit_labels,
    )
    results["derivative_orbits_d18_batch32"] = timing_record(
        derivative_times,
        item_count=len(derivative_maps),
    )

    candidates = np.asarray(
        list(permutations(range(6))),
        dtype=np.uint32,
    )
    left = np.asarray([[1, 0, 2, 3, 4, 5]], dtype=np.uint32)
    right = np.asarray([[0, 1, 3, 2, 4, 5]], dtype=np.uint32)
    coset_outputs = {}
    coset_times = {}
    for backend in ("reference", "native"):
        coset_times[backend], coset_outputs[backend] = timed_repeats(
            lambda backend=backend: permutation_double_cosets(
                candidates,
                left,
                right,
                backend=backend,
            ),
            repeats,
        )
    np.testing.assert_array_equal(
        coset_outputs["native"].class_ids,
        coset_outputs["reference"].class_ids,
    )
    results["double_cosets_s6"] = timing_record(
        coset_times,
        item_count=len(candidates),
    )

    relations = cycle_relations(31)
    wl_outputs = {}
    wl_times = {}
    for backend in ("reference", "native"):
        wl_times[backend], wl_outputs[backend] = timed_repeats(
            lambda backend=backend: coherent_configuration(
                relations,
                backend=backend,
            ),
            repeats,
        )
    np.testing.assert_array_equal(
        wl_outputs["native"].relations,
        wl_outputs["reference"].relations,
    )
    np.testing.assert_array_equal(
        wl_outputs["native"].intersection_numbers,
        wl_outputs["reference"].intersection_numbers,
    )
    results["coherent_configuration_cycle31"] = timing_record(
        wl_times,
        item_count=31 * 31,
    )
    return results


def timing_record(
    times: dict[str, list[float]],
    *,
    item_count: int,
) -> dict[str, object]:
    reference_median = median(times["reference"])
    native_median = median(times["native"])
    return {
        "item_count": item_count,
        "reference_wall_seconds": times["reference"],
        "native_wall_seconds": times["native"],
        "reference_wall_median": reference_median,
        "native_wall_median": native_median,
        "wall_speedup": reference_median / native_median,
        "output_identical": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--kernel-repeats", type=int, default=5)
    parser.add_argument("--native-query-iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    builder = load_atlas_builder()
    models = [
        builder.build_group(spec)
        for spec in builder.atlas_specs()
    ]

    retained_times, retained_output = timed_repeats(
        lambda: retained_atlas_profile(builder, models),
        args.repeats,
    )
    native_times, native_output = timed_repeats(
        lambda: native_atlas_profile(models, args.threads),
        args.repeats,
    )
    if native_output != retained_output:
        raise AssertionError("native and retained atlas fibers differ")
    retained_median = median(retained_times)
    native_median = median(native_times)

    d24 = next(model for model in models if model.spec.slug == "D24")
    action = np.asarray(d24.atom_permutations, dtype=np.uint32)
    outsider = np.arange(action.shape[1], dtype=np.uint32)
    outsider[[0, 1]] = outsider[[1, 0]]
    candidates = np.vstack([action[:16], outsider])
    expected_query = native_group_query(action, candidates, args.threads)
    gap_times, gap_output = timed_repeats(
        lambda: gap_group_query(builder, action, candidates),
        args.repeats,
    )
    if gap_output != expected_query:
        raise AssertionError("native and GAP group-query outputs differ")

    native_query_times = []
    native_query_output = None
    for _ in range(args.repeats):
        started = perf_counter()
        for _ in range(args.native_query_iterations):
            native_query_output = native_group_query(
                action,
                candidates,
                args.threads,
            )
        native_query_times.append(
            (perf_counter() - started) / args.native_query_iterations
        )
    if native_query_output != expected_query:
        raise AssertionError("repeated native group query changed output")
    gap_median = median(gap_times)
    native_query_median = median(native_query_times)

    payload = {
        "schema": 1,
        "date": "2026-07-29",
        "repeats": args.repeats,
        "kernel_repeats": args.kernel_repeats,
        "threads": args.threads,
        "atlas_pipeline": {
            "groups": 13,
            "automorphism_orbits": retained_output[1],
            "graph_fibers": retained_output[2],
            "retained_python_labelg_wall_seconds": retained_times,
            "native_wall_seconds": native_times,
            "retained_python_labelg_wall_median": retained_median,
            "native_wall_median": native_median,
            "wall_speedup": retained_median / native_median,
            "output_identical": True,
        },
        "gap_group_query": {
            "degree": action.shape[1],
            "generator_count": len(action),
            "membership_query_count": len(candidates),
            "native_iterations_per_repeat": args.native_query_iterations,
            "gap_mamba_wall_seconds": gap_times,
            "native_wall_seconds": native_query_times,
            "gap_mamba_wall_median": gap_median,
            "native_wall_median": native_query_median,
            "wall_speedup": gap_median / native_query_median,
            "group_order": expected_query[0],
            "orbit_count": expected_query[1],
            "output_identical": True,
        },
        "reference_kernels": benchmark_reference_kernels(
            builder,
            models,
            args.kernel_repeats,
            args.threads,
        ),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="ascii")


if __name__ == "__main__":
    main()
