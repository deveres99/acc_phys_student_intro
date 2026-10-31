"""
Build the student and solution notebooks from one source file per exercise.

    python tools/build_transverse_notebooks.py            # builds all
    python tools/build_transverse_notebooks.py ex05       # builds sources whose name contains 'ex05'

Source files live in ``transverse_beam_dynamics/_src/*.nbsrc``.  Format:

    %% md                 markdown cell (both versions)
    %% md solution        markdown cell, solutions only (e.g. answers)
    %% md student         markdown cell, student version only
    %% code               code cell (both versions)
    %% code solution      code cell, solutions only
    %% code student       code cell, student version only

Inside code cells, a block

    ### BEGIN SOLUTION
    ...reference code...
    ### END SOLUTION

is kept in the solutions and replaced in the student version by
``raise NotImplementedError(...)`` (same indentation).  Lines inside the
block starting with ``### STUDENT:`` are put in the student version instead
of the default ``raise`` line (e.g. ``### STUDENT: answer = None``).

Only the standard library is needed (notebooks are plain JSON, nbformat 4).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "transverse_beam_dynamics" / "_src"
OUT = {"student": ROOT / "transverse_beam_dynamics" / "student", "solutions": ROOT / "transverse_beam_dynamics" / "solutions"}

HEADER = re.compile(r"^%%\s*(md|code)\s*(solution|student)?\s*$")


def parse(text: str):
    """Split a source file into cells: (kind, only, lines)."""
    cells, cur = [], None
    for line in text.splitlines():
        m = HEADER.match(line)
        if m:
            if cur is not None:
                cells.append(cur)
            cur = (m.group(1), m.group(2), [])
        elif cur is not None:
            cur[2].append(line)
    if cur is not None:
        cells.append(cur)
    return cells


def transform_code(lines, version: str):
    out, in_sol, indent, student_lines = [], False, "", []
    for line in lines:
        stripped = line.strip()
        if stripped == "### BEGIN SOLUTION":
            in_sol, indent, student_lines = True, line[: len(line) - len(line.lstrip())], []
            continue
        if stripped == "### END SOLUTION":
            in_sol = False
            if version == "student":
                if student_lines:
                    out += student_lines
                else:
                    out += [indent + "# YOUR CODE HERE",
                            indent + 'raise NotImplementedError("Replace this line with your code")']
            continue
        if in_sol:
            if stripped.startswith("### STUDENT:"):
                student_lines.append(indent + stripped[len("### STUDENT:"):].strip())
            elif version == "solutions":
                out.append(line)
            continue
        out.append(line)
    if in_sol:
        raise ValueError("unterminated ### BEGIN SOLUTION block")
    return out


def strip_blank(lines):
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines


def to_source(lines):
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])


def build(src: Path, version: str) -> dict:
    cells = []
    for kind, only, lines in parse(src.read_text(encoding="utf-8")):
        keep = (only is None
                or (only == "solution" and version == "solutions")
                or (only == "student" and version == "student"))
        if not keep:
            continue
        if kind == "code":
            lines = transform_code(lines, version)
        lines = strip_blank(lines)
        if not lines:
            continue
        cell = {"cell_type": "markdown" if kind == "md" else "code",
                "metadata": {}, "source": to_source(lines)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    # banner after the title cell
    banner = ("> **Solutions version.** Contains reference code and answers. "
              "For the instructor; do not hand out before the session.") if version == "solutions" \
        else ("> **Student version.** Fill in the cells marked `# YOUR CODE HERE` and run "
              "the check cells after them: they tell you when you are right.")
    cells.insert(1, {"cell_type": "markdown", "metadata": {}, "source": [banner]})
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv):
    pattern = argv[1] if len(argv) > 1 else ""
    srcs = sorted(p for p in SRC.glob("*.nbsrc") if pattern in p.name)
    for src in srcs:
        for version in ("student", "solutions"):
            nb = build(src, version)
            # nbformat 4.5 wants cell ids
            for i, c in enumerate(nb["cells"]):
                c["id"] = f"{src.stem[:20]}-{i:03d}".replace("_", "-")
            OUT[version].mkdir(parents=True, exist_ok=True)
            path = OUT[version] / (src.stem + ".ipynb")
            path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"built {src.stem}")


if __name__ == "__main__":
    main(sys.argv)
