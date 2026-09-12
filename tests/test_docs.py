"""The documentation must stay internally consistent.

Prose rots more quietly than code: a file is renamed, a heading is reworded, a
number is quoted in two places and one of them is updated.  These tests catch
the kinds of rot that a reader would otherwise hit first.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _md_files():
    out = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs
                   if d not in (".git", ".venv", "runs", "__pycache__",
                                "reports", ".pytest_cache", "node_modules")]
        for f in files:
            if f.endswith(".md"):
                out.append(os.path.join(base, f))
    return sorted(out)


def test_every_link_and_anchor_resolves():
    """No dead relative link, no dead #anchor, anywhere in the documentation."""
    res = subprocess.run([sys.executable,
                          os.path.join(ROOT, "scripts", "check_links.py")],
                         capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stdout + res.stderr


def test_every_document_says_what_it_is():
    """A reader landing from a search result must see what the file is for."""
    exempt = {"docs/historical_results.md"}      # carries its own warning banner
    missing = []
    for path in _md_files():
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        if rel.startswith("results/") and rel != "results/README.md":
            continue
        if rel in exempt:
            continue
        text = io.open(path, encoding="utf-8").read()
        if "<!-- DOCNAV -->" not in text and rel not in ("README.md", "ROADMAP.md",
                                                         "LICENSE.md"):
            missing.append(rel)
    assert not missing, f"no navigation block: {missing}"


def test_the_roadmap_lists_every_experiment():
    """An experiment that exists but is not on the road map is invisible."""
    from avsec.program import PROGRAM

    text = io.open(os.path.join(ROOT, "ROADMAP.md"), encoding="utf-8").read()
    missing = [e.eid for e in PROGRAM if e.eid not in text]
    assert not missing, f"missing from ROADMAP.md: {missing}"


def test_the_roadmap_states_what_was_not_done():
    """The unfinished part is the part most worth being explicit about."""
    text = io.open(os.path.join(ROOT, "ROADMAP.md"), encoding="utf-8").read()
    assert "не виконано" in text
    assert "E11" in text
    assert "апаратн" in text.lower()


def test_headings_are_unique_within_a_file():
    """Duplicate top-level headings make anchors ambiguous and links fragile.

    Only levels 1 and 2 are checked: those are what documents link to.  A
    changelog legitimately repeats "Додано" under every version, so it is
    exempt from the deeper levels by construction.
    """
    head = re.compile(r"^#{1,2}\s+(.+?)\s*$", re.M)
    problems = []
    for path in _md_files():
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        if rel.startswith("results/") or rel == "CHANGELOG.md":
            continue
        titles = [t.strip().lower() for t in head.findall(
            io.open(path, encoding="utf-8").read())]
        dupes = {t for t in titles if titles.count(t) > 1}
        if dupes:
            problems.append((rel, sorted(dupes)))
    assert not problems, problems


@pytest.mark.parametrize("path,needle", [
    ("README.md", "B4t"),
    ("README.md", "ROADMAP.md"),
    ("docs/claims.md", "гіпотеза"),
    ("docs/conclusions.md", "B4t"),
    ("results/README.md", "avsec verify"),
])
def test_key_facts_are_present(path, needle):
    """A handful of facts that must not quietly disappear from the docs."""
    text = io.open(os.path.join(ROOT, path), encoding="utf-8").read()
    assert needle in text, f"{needle!r} missing from {path}"
