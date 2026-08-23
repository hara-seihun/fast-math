from __future__ import annotations

import subprocess
import sys

import fast_math
from fast_math.__main__ import main
from fast_math.index import entries, find


def test_every_public_export_carries_a_summary() -> None:
    missing = [entry.path for entry in entries() if not entry.summary]
    assert not missing, f"undocumented public names: {missing}"


def test_index_covers_the_root_exports() -> None:
    indexed = {entry.name for entry in entries()}
    assert set(fast_math.__all__) <= indexed


def test_find_matches_names_summaries_and_modules() -> None:
    assert any(e.name == "sparse_rank_mod_u32" for e in find("sparse_rank"))
    assert any(e.name == "csr_common_neighbors" for e in find("common neighbor"))
    assert all("graph64" in e.module for e in find("graph64"))
    assert find("no kernel does this") == ()


def test_find_requires_every_term() -> None:
    both = find("orbit", "subset")
    assert both
    assert all(e in find("orbit") for e in both)


def test_shell_entry_point_reports_a_miss(capsys) -> None:
    assert main(["--find", "certainly absent"]) == 1
    assert "CONTRIBUTING.md" in capsys.readouterr().out


def test_the_other_spelling_of_the_package_name_works() -> None:
    subprocess.run(
        [sys.executable, "-c", "import fastmath; assert fastmath.find('rank')"],
        check=True,
    )
