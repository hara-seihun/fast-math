"""``fast-math --find <term>`` and ``fast-math --index``, from the shell."""

from __future__ import annotations

import sys

from .index import entries, find, render

USAGE = """usage:
  fast-math script.py [args]     run a script with fast_math importable
  fast-math -c 'CODE'            run one line
  fast-math --find TERM [TERM]   kernels matching every term
  fast-math --index              every public name, one per line

Terms match names, summaries, and module names, so `--find csr` and
`--find "common neighbor"` both work.
"""

MISSING = """no kernel matches {terms}.

If you are about to hand-write this loop, that is the signal CONTRIBUTING.md is
written for: the kernel merged here is there for every session after yours, and
the one in your scratch directory dies with the session. `--index` lists
everything, in case it is here under a name you did not guess.
"""


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        sys.stdout.write(USAGE)
        return 0
    command, *terms = argv
    if command in ("--index", "index"):
        found = entries()
    elif command in ("--find", "find"):
        if not terms:
            sys.stderr.write("fast-math --find needs a term\n")
            return 2
        found = find(*terms)
    else:
        sys.stderr.write(USAGE)
        return 2
    if not found:
        sys.stdout.write(MISSING.format(terms=" ".join(repr(t) for t in terms)))
        return 1
    sys.stdout.write(render(found) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
