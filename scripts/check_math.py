"""Math in the Markdown must be visible in every viewer, not only on github.com.

Three rules, all learned the hard way on the rendered page rather than in the
source:

1. **No ``$$...$$``.**  GitHub runs its emphasis parser before the math
   renderer, so every ``_`` that can be paired inside a multi-line ``$$`` block
   is eaten as italics: ``\\underbrace{x}_{a} ... \\underbrace{y}_{b}`` arrives
   at the renderer as ``\\underbrace{x}{a} ... \\underbrace{y}{b}`` and the
   formula collapses.

2. **No inline ``$...$`` in prose.**  It renders on github.com and silently
   does not in plenty of other Markdown viewers, where the reader is left
   looking at raw LaTeX.  Single symbols go in backticks instead.

3. **A formula a reader must see is a plain-text block**, with the LaTeX folded
   underneath for anyone who wants to paste it into a paper.  A fenced
   ``math`` block is fine *inside* such a ``<details>`` - it is then a bonus for
   viewers that support it, not the only way to read the formula.

Run directly, or let ``tests/test_docs.py`` run it.

Usage::

    python scripts/check_math.py
"""
from __future__ import annotations

import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INLINE = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")
CODE_SPAN = re.compile(r"`+[^`]*`+")
SKIP_DIRS = {".git", ".venv", "runs", "__pycache__", "reports", ".pytest_cache"}


def _prose(line: str) -> str:
    """The line with inline code removed.

    A document that *describes* this rule quotes ``$$...$$`` in backticks, and
    that is prose about math, not math.
    """
    return CODE_SPAN.sub(" ", line)


def md_files():
    out = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        out += [os.path.join(base, f) for f in files if f.endswith(".md")]
    return sorted(out)


def problems():
    """Every place a formula would fail to reach the reader."""
    found = []
    for path in md_files():
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        fenced = False
        math_fences = 0
        plain_fences = 0
        for i, line in enumerate(io.open(path, encoding="utf-8").read().split("\n"), 1):
            stripped = line.lstrip()
            if stripped.startswith("```"):
                lang = stripped[3:].strip().lower()
                if not fenced:
                    if lang == "math":
                        math_fences += 1
                    elif lang in ("text", "", "console"):
                        plain_fences += 1
                fenced = not fenced
                continue
            if fenced:
                continue
            prose = _prose(line)
            if "$$" in prose:
                found.append((rel, i, "display math must not use $$ - GitHub "
                              "eats its subscripts", line.strip()[:88]))
            for span in INLINE.findall(prose):
                found.append((rel, i, "inline $...$ does not render in every "
                              "viewer - use `backticks`", f"${span}$"))
        if math_fences and not plain_fences:
            found.append((rel, 0, "LaTeX with no plain-text formula next to it: "
                          "a reader whose viewer lacks math support sees nothing",
                          f"{math_fences} ```math blocks, 0 plain blocks"))
    return found


def main() -> int:
    found = problems()
    for rel, line, why, text in found:
        where = f"{rel}:{line}" if line else rel
        print(f"{where}  <- {why}\n    {text}")
    print(f"\n{len(md_files())} markdown files, {len(found)} math problems")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
