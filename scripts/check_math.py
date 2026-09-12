# -*- coding: utf-8 -*-
"""Math in the Markdown must survive GitHub's renderer.

Two rules, both learned the hard way:

1. **Display math is a fenced ``math`` block, never ``$$...$$``.**
   GitHub runs its emphasis parser before the math renderer, so every ``_``
   that can be paired inside a multi-line ``$$`` block is eaten as italics and
   the formula collapses - ``\\underbrace{x}_{a}`` renders as
   ``\\underbrace{x}{a}``.

2. **No line outside a fence carries two or more ``_`` inside inline math.**
   Same parser, same failure, one line at a time.

Run directly, or let ``tests/test_docs.py`` run it.
"""
from __future__ import annotations

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INLINE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")
SKIP_DIRS = {".git", ".venv", "runs", "__pycache__", "reports", ".pytest_cache"}


def md_files():
    out = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        out += [os.path.join(base, f) for f in files if f.endswith(".md")]
    return sorted(out)


def problems():
    found = []
    for path in md_files():
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        fenced = False
        for i, line in enumerate(io.open(path, encoding="utf-8").read().split("\n"), 1):
            stripped = line.lstrip()
            if stripped.startswith("```"):
                fenced = not fenced
                continue
            if fenced:
                continue
            if "$$" in line:
                found.append((rel, i, "display math must be a ```math fence",
                              line.strip()[:90]))
            n = sum(s.count("_") for s in INLINE.findall(line))
            if n >= 2:
                found.append((rel, i, f"{n} underscores in inline math on one "
                              "line - Markdown will italicise them",
                              line.strip()[:90]))
    return found


def main() -> int:
    found = problems()
    for rel, line, why, text in found:
        print(f"{rel}:{line}  <- {why}\n    {text}")
    print(f"\n{len(md_files())} markdown files, {len(found)} math problems")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
