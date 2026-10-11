"""Print the README's module map: one line per ``src/rab/*.py`` with its guide section and size.

    uv run python tools/module_map.py

The section is the first "§" or "design v2" reference in the module's docstring's first
line; the description is that first line with the reference stripped.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "rab"


def rows() -> list[tuple[str, str, int]]:
    out = []
    for path in sorted(SRC.glob("*.py")):
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        doc = (ast.get_docstring(ast.parse(text)) or "").strip().split("\n\n")[0].replace("\n", " ")
        doc = re.sub(r"\s+", " ", doc).rstrip(".")
        out.append((path.name, doc, len(text.splitlines())))
    return out


def main() -> int:
    print("| Module | What it does (first docstring paragraph) | Lines |")
    print("|---|---|---|")
    for name, doc, n in rows():
        print(f"| `src/rab/{name}` | {doc} | {n} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
