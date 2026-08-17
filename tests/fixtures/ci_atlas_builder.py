#!/usr/bin/env python3
"""Build exact undirected CI-defect fibers for small generalized dihedral groups."""

from __future__ import annotations

# Retained from the reviewed R-0703 atlas source so the native parity suite is
# self-contained after retirement of the predecessor research runtime.

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from itertools import product
import json
from math import gcd
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable

import networkx as nx


ROUTE_DIR = Path(__file__).resolve().parent
BUILD_DIR = ROUTE_DIR / "build"
GROUP_RE = re.compile(
    r"groupsize=([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)"
)

Vector = tuple[int, ...]
Element = tuple[Vector, int]
Matrix = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class GroupSpec:
    slug: str
    label: str
    kind: str
    modulus: int
    rank: int
    literature_expectation: str

    @property
    def kernel_order(self) -> int:
        return self.modulus**self.rank

    @property
    def group_order(self) -> int:
        return 2 * self.kernel_order


@dataclass
class GroupModel:
    spec: GroupSpec
    elements: list[Element]
    index: dict[Element, int]
    identity_index: int
    multiply_table: list[list[int]]
    inverse_indices: list[int]
    automorphisms: list[tuple[int, ...]]
    atoms: list[tuple[int, ...]]
    atom_of_element: dict[int, int]
    atom_permutations: list[tuple[int, ...]]


def nauty_command(name: str) -> str:
    for candidate in (name, f"nauty-{name}"):
        path = shutil.which(candidate)
        if path is not None:
            return path
    raise FileNotFoundError(f"nauty command not found: {name}")


def run_lines(command: list[str], records: list[str]) -> list[str]:
    completed = subprocess.run(
        command,
        input="\n".join(records) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.splitlines()


def canonical_records(graphs: list[nx.Graph]) -> list[str]:
    records = [
        nx.to_graph6_bytes(graph, header=False).decode("ascii").strip()
        for graph in graphs
    ]
    return run_lines(
        [nauty_command("labelg"), "-q", "-g", "-", "-"],
        records,
    )


def automorphism_orders(records: list[str]) -> list[int]:
    if not records:
        return []
    lines = run_lines(
        [nauty_command("countg"), "-q", "-V", "--a", "-"],
        records,
    )
    parsed: list[int | None] = []
    large_records = []
    for record, line in zip(records, lines, strict=True):
        match = GROUP_RE.search(line)
        if match is None:
            raise ValueError(f"bad countg line: {line}")
        if "e" in match.group(1):
            parsed.append(None)
            large_records.append(record)
            continue
        value = Decimal(match.group(1))
        order = int(value)
        assert value == order
        parsed.append(order)

    exact_large = exact_large_automorphism_orders(large_records)
    large_index = 0
    orders = []
    for value in parsed:
        if value is None:
            orders.append(exact_large[large_index])
            large_index += 1
        else:
            orders.append(value)
    assert large_index == len(exact_large)
    return orders


def dreadnaut_generators(record: str) -> list[tuple[int, ...]]:
    graph = nx.from_graph6_bytes(record.encode("ascii"))
    order = graph.number_of_nodes()
    lines = [f"n={order}", "g"]
    for vertex in range(order):
        lines.append(
            f"{vertex}: "
            + " ".join(str(neighbor) for neighbor in sorted(graph[vertex]))
            + ";"
        )
    lines.extend(["+a", "x", "q"])
    completed = subprocess.run(
        ["dreadnaut"],
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )

    blocks = []
    current = ""
    for line in completed.stdout.splitlines():
        if line.startswith("("):
            if current:
                blocks.append(current)
            current = line.strip()
        elif current and line.startswith("   "):
            current += " " + line.strip()
        elif current:
            blocks.append(current)
            current = ""
    if current:
        blocks.append(current)

    generators = []
    for block in blocks:
        permutation = list(range(order))
        for cycle_text in re.findall(r"\(([^()]*)\)", block):
            cycle = [int(value) for value in cycle_text.split()]
            for source, target in zip(cycle, cycle[1:] + cycle[:1]):
                permutation[source] = target
        generators.append(tuple(permutation))
    assert generators
    return generators


def gap_cycle(permutation: tuple[int, ...]) -> str:
    unseen = set(range(len(permutation)))
    cycles = []
    while unseen:
        seed = min(unseen)
        cycle = []
        value = seed
        while value not in cycle:
            cycle.append(value)
            unseen.discard(value)
            value = permutation[value]
        if len(cycle) > 1:
            cycles.append(
                "(" + ",".join(str(point + 1) for point in cycle) + ")"
            )
    return "".join(cycles) or "()"


def exact_large_automorphism_orders(records: list[str]) -> list[int]:
    if not records:
        return []
    lines = []
    for index, record in enumerate(records):
        generators = dreadnaut_generators(record)
        gap_generators = ",".join(
            gap_cycle(generator)
            for generator in generators
        )
        lines.extend(
            [
                f"g{index} := Group([{gap_generators}]);;",
                f'Print("SIZE {index} ", Size(g{index}), "\\n");',
            ]
        )
    completed = subprocess.run(
        ["gap", "-q"],
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    values = {
        int(match.group(1)): int(match.group(2))
        for match in re.finditer(
            r"^SIZE (\d+) (\d+)$",
            completed.stdout,
            re.MULTILINE,
        )
    }
    if len(values) != len(records):
        raise AssertionError(completed.stdout[-5000:] + completed.stderr)
    return [values[index] for index in range(len(records))]


def vector_add(left: Vector, right: Vector, modulus: int) -> Vector:
    return tuple(
        (left[index] + right[index]) % modulus
        for index in range(len(left))
    )


def vector_neg(vector: Vector, modulus: int) -> Vector:
    return tuple((-entry) % modulus for entry in vector)


def mat_vec(matrix: Matrix, vector: Vector, modulus: int) -> Vector:
    return tuple(
        sum(row[index] * vector[index] for index in range(len(vector)))
        % modulus
        for row in matrix
    )


def all_vectors(modulus: int, rank: int) -> list[Vector]:
    return list(product(range(modulus), repeat=rank))


def gl_order(modulus: int, rank: int) -> int:
    return product_int(
        modulus**rank - modulus**index
        for index in range(rank)
    )


def product_int(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def invertible_matrices(modulus: int, rank: int) -> list[Matrix]:
    vectors = all_vectors(modulus, rank)
    nonzero = vectors[1:]
    matrices: list[Matrix] = []
    for entries in product(range(modulus), repeat=rank * rank):
        matrix = tuple(
            tuple(entries[row * rank : (row + 1) * rank])
            for row in range(rank)
        )
        images = {mat_vec(matrix, vector, modulus) for vector in nonzero}
        if len(images) == len(nonzero) and (0,) * rank not in images:
            matrices.append(matrix)
    assert len(matrices) == gl_order(modulus, rank)
    return matrices


def euler_phi(number: int) -> int:
    return sum(gcd(candidate, number) == 1 for candidate in range(number))


def square_free(number: int) -> bool:
    prime = 2
    remaining = number
    while prime * prime <= remaining:
        exponent = 0
        while remaining % prime == 0:
            exponent += 1
            remaining //= prime
        if exponent > 1:
            return False
        prime += 1
    return True


def build_group(spec: GroupSpec) -> GroupModel:
    vectors = all_vectors(spec.modulus, spec.rank)
    zero = (0,) * spec.rank
    elements = [
        (vector, reflection)
        for reflection in (0, 1)
        for vector in vectors
    ]
    index = {element: position for position, element in enumerate(elements)}

    def multiply(left: Element, right: Element) -> Element:
        signed_right = (
            right[0]
            if left[1] == 0
            else vector_neg(right[0], spec.modulus)
        )
        return (
            vector_add(left[0], signed_right, spec.modulus),
            (left[1] + right[1]) % 2,
        )

    multiply_table = [
        [index[multiply(left, right)] for right in elements]
        for left in elements
    ]
    identity_index = index[(zero, 0)]
    inverse_indices: list[int] = []
    for element_index, element in enumerate(elements):
        inverse = next(
            candidate
            for candidate in range(len(elements))
            if multiply_table[element_index][candidate] == identity_index
            and multiply_table[candidate][element_index] == identity_index
        )
        inverse_indices.append(inverse)

    automorphisms = enumerate_automorphisms(
        spec,
        elements,
        index,
    )
    verify_automorphisms(
        automorphisms,
        multiply_table,
        identity_index,
    )

    unseen = set(range(len(elements)))
    unseen.remove(identity_index)
    atoms: list[tuple[int, ...]] = []
    while unseen:
        element_index = min(unseen)
        atom = tuple(
            sorted({element_index, inverse_indices[element_index]})
        )
        atoms.append(atom)
        unseen.difference_update(atom)
    atom_of_element = {
        element_index: atom_index
        for atom_index, atom in enumerate(atoms)
        for element_index in atom
    }

    atom_permutations_set: set[tuple[int, ...]] = set()
    for automorphism in automorphisms:
        atom_permutation: list[int] = []
        for atom in atoms:
            image_atoms = {
                atom_of_element[automorphism[element_index]]
                for element_index in atom
            }
            assert len(image_atoms) == 1
            atom_permutation.append(next(iter(image_atoms)))
        assert sorted(atom_permutation) == list(range(len(atoms)))
        atom_permutations_set.add(tuple(atom_permutation))

    return GroupModel(
        spec=spec,
        elements=elements,
        index=index,
        identity_index=identity_index,
        multiply_table=multiply_table,
        inverse_indices=inverse_indices,
        automorphisms=automorphisms,
        atoms=atoms,
        atom_of_element=atom_of_element,
        atom_permutations=sorted(atom_permutations_set),
    )


def enumerate_automorphisms(
    spec: GroupSpec,
    elements: list[Element],
    index: dict[Element, int],
) -> list[tuple[int, ...]]:
    zero = (0,) * spec.rank
    permutations: set[tuple[int, ...]] = set()

    if spec.kind == "cyclic" and spec.modulus >= 3:
        assert spec.rank == 1
        units = [
            unit
            for unit in range(spec.modulus)
            if gcd(unit, spec.modulus) == 1
        ]
        for unit in units:
            for translate in range(spec.modulus):
                image: list[int] = []
                for vector, reflection in elements:
                    coordinate = (
                        unit * vector[0] + reflection * translate
                    ) % spec.modulus
                    image.append(index[((coordinate,), reflection)])
                permutations.add(tuple(image))
        assert len(permutations) == spec.modulus * euler_phi(spec.modulus)
        return sorted(permutations)

    if spec.modulus == 2:
        dimension = spec.rank + 1
        matrices = invertible_matrices(2, dimension)
        for matrix in matrices:
            image = []
            for vector, reflection in elements:
                transformed = mat_vec(
                    matrix,
                    vector + (reflection,),
                    2,
                )
                image.append(
                    index[(transformed[:-1], transformed[-1])]
                )
            permutations.add(tuple(image))
        assert len(permutations) == gl_order(2, dimension)
        return sorted(permutations)

    matrices = invertible_matrices(spec.modulus, spec.rank)
    translations = all_vectors(spec.modulus, spec.rank)
    for matrix in matrices:
        for translate in translations:
            image = []
            for vector, reflection in elements:
                transformed = mat_vec(
                    matrix,
                    vector,
                    spec.modulus,
                )
                if reflection:
                    transformed = vector_add(
                        transformed,
                        translate,
                        spec.modulus,
                    )
                image.append(index[(transformed, reflection)])
            permutations.add(tuple(image))
    assert len(permutations) == (
        spec.kernel_order * gl_order(spec.modulus, spec.rank)
    )
    assert index[(zero, 0)] == 0
    return sorted(permutations)


def verify_automorphisms(
    automorphisms: list[tuple[int, ...]],
    multiply_table: list[list[int]],
    identity_index: int,
) -> None:
    order = len(multiply_table)
    for automorphism in automorphisms:
        assert sorted(automorphism) == list(range(order))
        assert automorphism[identity_index] == identity_index
        for left in range(order):
            for right in range(order):
                assert automorphism[multiply_table[left][right]] == (
                    multiply_table[automorphism[left]][automorphism[right]]
                )


def permute_mask(mask: int, permutation: tuple[int, ...]) -> int:
    image = 0
    remaining = mask
    while remaining:
        low_bit = remaining & -remaining
        atom_index = low_bit.bit_length() - 1
        image |= 1 << permutation[atom_index]
        remaining ^= low_bit
    return image


def connection_orbits(
    atom_count: int,
    atom_permutations: list[tuple[int, ...]],
) -> tuple[list[int], dict[int, int], dict[int, int]]:
    total = 1 << atom_count
    seen = bytearray(total)
    representative_of: dict[int, int] = {}
    orbit_size: dict[int, int] = {}
    representatives: list[int] = []

    for mask in range(total):
        if seen[mask]:
            continue
        orbit = {
            permute_mask(mask, permutation)
            for permutation in atom_permutations
        }
        representative = min(orbit)
        assert representative == mask
        representatives.append(representative)
        orbit_size[representative] = len(orbit)
        for image in orbit:
            seen[image] = 1
            representative_of[image] = representative

    assert all(seen)
    assert sum(orbit_size.values()) == total
    return representatives, representative_of, orbit_size


def mask_elements(model: GroupModel, mask: int) -> list[int]:
    return [
        element_index
        for atom_index, atom in enumerate(model.atoms)
        if mask & (1 << atom_index)
        for element_index in atom
    ]


def cayley_graph(model: GroupModel, mask: int) -> nx.Graph:
    connection = mask_elements(model, mask)
    graph = nx.Graph()
    graph.add_nodes_from(range(model.spec.group_order))
    for vertex in range(model.spec.group_order):
        for step in connection:
            graph.add_edge(
                vertex,
                model.multiply_table[step][vertex],
            )
    assert nx.number_of_selfloops(graph) == 0
    degrees = {degree for _, degree in graph.degree()}
    assert len(degrees) == 1
    assert next(iter(degrees)) == len(connection)
    return graph


def element_label(spec: GroupSpec, element: Element) -> str:
    vector, reflection = element
    if spec.kind == "cyclic":
        exponent = vector[0]
        base = "1" if exponent == 0 else (
            "r" if exponent == 1 else f"r^{exponent}"
        )
    else:
        base = "a(" + ",".join(map(str, vector)) + ")"
    if reflection:
        if base == "1":
            return "s"
        return f"{base}s"
    return base


def connection_labels(model: GroupModel, mask: int) -> list[str]:
    return sorted(
        element_label(model.spec, model.elements[element_index])
        for element_index in mask_elements(model, mask)
    )


def mask_from_elements(model: GroupModel, elements: list[Element]) -> int:
    mask = 0
    for element in elements:
        element_index = model.index[element]
        mask |= 1 << model.atom_of_element[element_index]
    return mask


def digest_rows(rows: list[dict[str, object]]) -> str:
    payload = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in rows
    )
    return sha256((payload + "\n").encode("ascii")).hexdigest()


def build_group_atlas(
    spec: GroupSpec,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model = build_group(spec)
    representatives, representative_of, orbit_size = connection_orbits(
        len(model.atoms),
        model.atom_permutations,
    )
    graphs = [cayley_graph(model, mask) for mask in representatives]
    canonical = canonical_records(graphs)
    assert len(canonical) == len(representatives)

    fibers: dict[str, list[int]] = {}
    graph_metadata: dict[str, tuple[int, bool]] = {}
    for mask, graph, record in zip(
        representatives,
        graphs,
        canonical,
        strict=True,
    ):
        fibers.setdefault(record, []).append(mask)
        metadata = (
            next(iter(dict(graph.degree()).values())),
            nx.is_connected(graph),
        )
        if record in graph_metadata:
            assert graph_metadata[record] == metadata
        else:
            graph_metadata[record] = metadata

    representative_index = {
        mask: index
        for index, mask in enumerate(representatives)
    }
    canonical_by_representative = {
        mask: canonical[index]
        for mask, index in representative_index.items()
    }
    full_mask = (1 << len(model.atoms)) - 1

    fiber_rows: list[dict[str, object]] = []
    defect_records = sorted(
        record
        for record, masks in fibers.items()
        if len(masks) > 1
    )
    defect_automorphism_orders = dict(
        zip(
            defect_records,
            automorphism_orders(defect_records),
            strict=True,
        )
    )

    defects: list[dict[str, object]] = []
    for record in sorted(fibers):
        masks = sorted(fibers[record])
        valency, connected = graph_metadata[record]
        complement_representative = representative_of[
            full_mask ^ masks[0]
        ]
        complement_record = canonical_by_representative[
            complement_representative
        ]
        row = {
            "group": spec.slug,
            "graph6": record,
            "ci_multiplicity": len(masks),
            "valency": valency,
            "connected": connected,
            "raw_connection_sets": sum(
                orbit_size[mask]
                for mask in masks
            ),
            "orbit_representatives": [
                f"0x{mask:x}"
                for mask in masks
            ],
            "complement_graph6": complement_record,
        }
        fiber_rows.append(row)
        if len(masks) == 1:
            continue
        defects.append(
            {
                **row,
                "graph_automorphisms": (
                    defect_automorphism_orders[record]
                ),
                "connection_orbits": [
                    {
                        "mask": f"0x{mask:x}",
                        "orbit_size": orbit_size[mask],
                        "stabilizer_order": (
                            len(model.automorphisms)
                            // orbit_size[mask]
                        ),
                        "connection": connection_labels(model, mask),
                    }
                    for mask in masks
                ],
            }
        )

    fiber_row_by_record = {
        row["graph6"]: row
        for row in fiber_rows
    }
    for row in fiber_rows:
        complement = fiber_row_by_record[row["complement_graph6"]]
        assert complement["complement_graph6"] == row["graph6"]
        assert complement["ci_multiplicity"] == row["ci_multiplicity"]
        assert complement["valency"] == (
            spec.group_order - 1 - row["valency"]
        )

    prism: dict[str, object] | None = None
    if spec.kind == "cyclic" and spec.modulus >= 4:
        n = spec.modulus
        if n % 2 == 0:
            prism_mask = mask_from_elements(
                model,
                [
                    ((1,), 0),
                    (((-1) % n,), 0),
                    ((0,), 1),
                ],
            )
            prism_representative = representative_of[prism_mask]
            prism_record = canonical_by_representative[
                prism_representative
            ]
            prism_fiber = sorted(fibers[prism_record])
            connected_cubic_defects = [
                defect
                for defect in defects
                if defect["connected"] and defect["valency"] == 3
            ]
            assert len(connected_cubic_defects) == 1
            assert connected_cubic_defects[0]["graph6"] == prism_record
            assert len(prism_fiber) > 1
            prism = {
                "graph6": prism_record,
                "ci_multiplicity": len(prism_fiber),
                "input_mask": f"0x{prism_mask:x}",
                "orbit_representative": (
                    f"0x{prism_representative:x}"
                ),
                "connection": connection_labels(
                    model,
                    prism_representative,
                ),
                "fiber_representatives": [
                    {
                        "mask": f"0x{mask:x}",
                        "connection": connection_labels(model, mask),
                    }
                    for mask in prism_fiber
                ],
            }

    if spec.kind == "cyclic":
        n = spec.modulus
        if n == 2 or (n % 2 == 1 and (square_free(n) or n == 9)):
            assert not defects
        if n >= 4 and n % 2 == 0:
            assert defects

    summary = {
        "group": spec.slug,
        "label": spec.label,
        "kernel_order": spec.kernel_order,
        "group_order": spec.group_order,
        "inversion_orbits": len(model.atoms),
        "inverse_closed_connection_sets": 1 << len(model.atoms),
        "group_automorphisms": len(model.automorphisms),
        "faithful_atom_action_size": len(model.atom_permutations),
        "ci_orbits": len(representatives),
        "unlabeled_cayley_graphs": len(fibers),
        "defect_graph_types": len(defects),
        "defect_excess": sum(
            defect["ci_multiplicity"] - 1
            for defect in defects
        ),
        "max_ci_multiplicity": max(
            len(masks)
            for masks in fibers.values()
        ),
        "literature_expectation": spec.literature_expectation,
        "fiber_digest": digest_rows(fiber_rows),
        "known_prism_defect": prism,
        "defects": defects,
    }
    return summary, fiber_rows


def atlas_specs() -> list[GroupSpec]:
    return [
        GroupSpec("D4", "D_4 = Dih(C_2)", "cyclic", 2, 1, "CI"),
        GroupSpec("D6", "D_6 = Dih(C_3)", "cyclic", 3, 1, "DCI"),
        GroupSpec("D8", "D_8 = Dih(C_4)", "cyclic", 4, 1, "non-CI"),
        GroupSpec("D10", "D_10 = Dih(C_5)", "cyclic", 5, 1, "DCI"),
        GroupSpec("D12", "D_12 = Dih(C_6)", "cyclic", 6, 1, "non-CI"),
        GroupSpec("D14", "D_14 = Dih(C_7)", "cyclic", 7, 1, "DCI"),
        GroupSpec("D16", "D_16 = Dih(C_8)", "cyclic", 8, 1, "non-CI"),
        GroupSpec("D18", "D_18 = Dih(C_9)", "cyclic", 9, 1, "CI"),
        GroupSpec("D20", "D_20 = Dih(C_10)", "cyclic", 10, 1, "non-CI"),
        GroupSpec("D22", "D_22 = Dih(C_11)", "cyclic", 11, 1, "DCI"),
        GroupSpec("D24", "D_24 = Dih(C_12)", "cyclic", 12, 1, "non-CI"),
        GroupSpec(
            "Dih-C2xC2",
            "Dih(C_2^2) = C_2^3",
            "elementary",
            2,
            2,
            "DCI",
        ),
        GroupSpec(
            "Dih-C3xC3",
            "Dih(C_3^2)",
            "elementary",
            3,
            2,
            "candidate CI family",
        ),
    ]


def main() -> None:
    specs = atlas_specs()

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    all_fibers: list[dict[str, object]] = []
    for spec in specs:
        summary, fiber_rows = build_group_atlas(spec)
        summaries.append(summary)
        all_fibers.extend(fiber_rows)
        print(
            f"{spec.slug}: sets={summary['inverse_closed_connection_sets']} "
            f"Aut-orbits={summary['ci_orbits']} "
            f"graphs={summary['unlabeled_cayley_graphs']} "
            f"defects={summary['defect_graph_types']} "
            f"excess={summary['defect_excess']} "
            f"max={summary['max_ci_multiplicity']}",
            flush=True,
        )

    payload = {
        "schema": 1,
        "scope": {
            "ci_or_dci": "undirected CI",
            "connection_sets": "all inverse-closed subsets excluding identity",
            "first_quotient": "full exact group automorphism action",
            "second_quotient": "nauty exact unlabeled graph canonicalization",
        },
        "groups": summaries,
    }
    atlas_path = BUILD_DIR / "atlas.json"
    atlas_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    fibers_path = BUILD_DIR / "fibers.jsonl"
    fibers_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in all_fibers
        ),
        encoding="ascii",
    )

    manifest = {
        "schema": 1,
        "atlas_sha256": sha256(atlas_path.read_bytes()).hexdigest(),
        "fibers_sha256": sha256(fibers_path.read_bytes()).hexdigest(),
        "group_count": len(summaries),
        "fiber_rows": len(all_fibers),
        "defect_graph_types": sum(
            summary["defect_graph_types"]
            for summary in summaries
        ),
    }
    (BUILD_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
