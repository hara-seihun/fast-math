"""What this library already has, searchable from inside a session.

The index is derived from the package itself: every public module's ``__all__``
and the first line of each object's docstring. Nothing here is a second list to
keep in sync, so a kernel is findable the moment it is exported.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
import typing
from typing import NamedTuple

__all__ = ["Entry", "entries", "find", "render"]

_SKIP_PREFIX = "_"


class Entry(NamedTuple):
    """One public name, where it lives, and what its docstring says it does."""

    name: str
    module: str
    kind: str
    summary: str

    @property
    def path(self) -> str:
        return f"{self.module}.{self.name}"


def _summary(obj: object, name: str) -> tuple[str, str]:
    if typing.get_origin(obj) is typing.Literal:
        return "type", "one of: " + ", ".join(str(a) for a in typing.get_args(obj))
    doc = (inspect.getdoc(obj) or "").strip()
    first = doc.split("\n", 1)[0].strip() if doc else ""
    if inspect.isclass(obj):
        auto = first.startswith(f"{name}(")
        if dataclasses.is_dataclass(obj) or auto:
            fields = [f.name for f in dataclasses.fields(obj)] if dataclasses.is_dataclass(obj) else []
            if auto or not first:
                shown = ", ".join(fields[:8]) + (", ..." if len(fields) > 8 else "")
                return "record", f"result record: {shown}" if shown else "result record"
            return "record", first
        return "class", first
    if callable(obj):
        return "function", first
    return "data", first


def _public_names(module: object, module_name: str, exported: frozenset[str]) -> list[str]:
    """What a module offers: its ``__all__``, or what the package re-exports.

    Half the modules here declare no ``__all__``. Falling back to ``dir()``
    would index every argument validator they happen to define, which is how an
    index stops being worth reading, so the root package's export list stands in
    as the curated surface.
    """
    declared = getattr(module, "__all__", None)
    if declared is not None:
        return list(declared)
    names = []
    for name in dir(module):
        if name.startswith(_SKIP_PREFIX) or name not in exported:
            continue
        if getattr(getattr(module, name, None), "__module__", None) != module_name:
            continue
        names.append(name)
    return names


def _modules() -> list[str]:
    root = importlib.import_module("fast_math")
    names = ["fast_math"]
    for found in pkgutil.iter_modules(root.__path__):
        if found.name.startswith(_SKIP_PREFIX):
            continue
        names.append(f"fast_math.{found.name}")
    return names


def entries() -> tuple[Entry, ...]:
    """Every public name the package exports, with its one-line summary.

    A submodule that cannot be imported here (an absent GPU toolchain, an
    optional transform library) contributes nothing and does not raise.
    """
    found: dict[str, Entry] = {}
    exported = frozenset(importlib.import_module("fast_math").__all__)
    for module_name in _modules():
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for name in _public_names(module, module_name, exported):
            obj = getattr(module, name, None)
            if obj is None:
                continue
            kind, summary = _summary(obj, name)
            entry = Entry(name=name, module=module_name, kind=kind, summary=summary)
            previous = found.get(name)
            if previous is None or previous.module == "fast_math":
                found[name] = entry
    return tuple(sorted(found.values(), key=lambda e: (e.module, e.name)))


def _score(entry: Entry, needle: str) -> int:
    name = entry.name.lower()
    if name == needle:
        return 0
    if name.startswith(needle):
        return 1
    if needle in name:
        return 2
    if needle in entry.summary.lower():
        return 3
    if needle in entry.module.lower():
        return 4
    return -1


def find(*terms: str) -> tuple[Entry, ...]:
    """Kernels whose name, summary, or module matches every term given.

    ``fast_math.find("csr")``, ``fast_math.find("orbit", "subset")``. Matching is
    case-insensitive and substring-based, and results come back with name
    matches ahead of summary matches.
    """
    needles = [t.lower() for t in terms if t.strip()]
    if not needles:
        return entries()
    scored = []
    for entry in entries():
        marks = [_score(entry, n) for n in needles]
        if any(m < 0 for m in marks):
            continue
        scored.append((min(marks), sum(marks), entry.path, entry))
    return tuple(entry for _, _, _, entry in sorted(scored))


def render(found: tuple[Entry, ...]) -> str:
    """The listing as printed by ``fast-math --find``."""
    if not found:
        return ""
    width = max(len(e.path) for e in found)
    lines = []
    for entry in found:
        summary = entry.summary or f"({entry.kind}, undocumented)"
        lines.append(f"{entry.path.ljust(width)}  {summary}")
    return "\n".join(lines)
