"""Check that every number in a report can be found in the project's records.

A report is only as trustworthy as its numbers. This lists every decimal number in REPORT
that does not occur, as a standalone token, in any of the SOURCES. It cannot prove a number
is used correctly, only that it was not invented or mistyped; a clean run is necessary, not
sufficient.

    python scripts/check_report_numbers.py docs/research/checkin.md docs/results/results.md docs/research/FINDINGS.md

Numbers a report legitimately derives itself must be declared in the report on a line of
the form

    <!-- derived: 0.0435 1.85 -->

so that they are visible as the author's own arithmetic rather than silently exempt.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# A decimal number with at least one digit after the point, optionally signed. Integers are
# ignored: they are mostly counts, dates and section numbers, and match sources by accident.
NUMBER = re.compile(r"(?<![\w.])[-+]?\d+\.\d+(?:e[-+]?\d+)?(?![\w]|\.\d)", re.IGNORECASE)
DERIVED = re.compile(r"<!--\s*derived:(.*?)-->", re.DOTALL)
# Two kinds of decimal-looking token are not results and are exempt, by rule rather than by
# list: arXiv identifiers (2512.23675), and the report's OWN section numbers, which are
# collected from its headings ("### 4.2 Context scaling" exempts "4.2" everywhere in it).
ARXIV = re.compile(r"^\d{4}\.\d{4,5}$")
HEADING_NUMBER = re.compile(r"^#+\s+(\d+\.\d+)\b", re.MULTILINE)


def canonical(tok: str) -> str:
    """'+0.0248' and '0.0248' are the same number for this purpose; sign conventions differ
    between tables (delta vs difference), so only the magnitude is matched."""
    return tok.lstrip("+-").lower()


def numbers(text: str) -> set[str]:
    return {canonical(m.group(0)) for m in NUMBER.finditer(text)}


def main() -> None:
    report = Path(sys.argv[1])
    sources = [Path(p) for p in sys.argv[2:]]
    assert sources, "give at least one source file"
    text = report.read_text()
    declared = {canonical(t) for block in DERIVED.findall(text) for t in block.split()}
    own_sections = set(HEADING_NUMBER.findall(text))
    known = set().union(*(numbers(p.read_text()) for p in sources))

    missing: dict[str, list[int]] = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        if DERIVED.search(line):
            continue
        for m in NUMBER.finditer(line):
            c = canonical(m.group(0))
            if c in own_sections or ARXIV.match(c):
                continue
            if c not in known and c not in declared:
                missing.setdefault(c, []).append(lineno)

    total = len(numbers(text))
    print(f"{report.name}: {total} distinct decimal numbers; {len(declared)} declared as derived; "
          f"{len(missing)} not found in {', '.join(p.name for p in sources)}")
    for c, lines in sorted(missing.items(), key=lambda kv: kv[1][0]):
        print(f"  NOT FOUND  {c:<12} line(s) {', '.join(map(str, lines[:6]))}")
    unused = declared - numbers(DERIVED.sub("", text))
    if unused:
        print(f"  declared as derived but never used: {sorted(unused)}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
