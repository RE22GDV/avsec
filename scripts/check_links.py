# -*- coding: utf-8 -*-
"""Every relative link and every in-page anchor in the Markdown must resolve.

Documentation rots quietly: a file is renamed, a heading is reworded, and a
link that used to work now lands on nothing.  This walks every Markdown file in
the repository, resolves each relative link and each #anchor against the
actual headings, and exits non-zero on the first broken one.

Run it directly, or let tests/test_docs.py run it.

Usage::

    python scripts/check_links.py
"""
import io
import os
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HREF = re.compile(r'href="([^"]+)"')
HEAD = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)


def slug(text: str) -> str:
    """GitHub's anchor slug: lowercase, drop punctuation/emoji, spaces -> '-'."""
    text = re.sub(r"`|\*|_", "", text).strip().lower()
    out = []
    for ch in text:
        if ch.isalnum() or ch in "-_ ":
            out.append(ch)
        elif unicodedata.category(ch).startswith(("P", "S")):
            continue
        else:
            out.append(ch)
    return "".join(out).replace(" ", "-")


def anchors(path):
    s = io.open(path, encoding="utf-8").read()
    seen, out = {}, set()
    for _, title in HEAD.findall(s):
        a = slug(title)
        n = seen.get(a, 0)
        seen[a] = n + 1
        out.add(a if n == 0 else f"{a}-{n}")
    return out


md_files = []
for base, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs
               if d not in (".git", ".venv", "runs", "__pycache__", "reports",
                            ".pytest_cache", "node_modules")]
    for f in files:
        if f.endswith(".md"):
            md_files.append(os.path.join(base, f))

bad = []
for path in sorted(md_files):
    text = io.open(path, encoding="utf-8").read()
    targets = [m.group(2) for m in LINK.finditer(text)]
    targets += HREF.findall(text)
    for t in targets:
        if t.startswith(("http://", "https://", "mailto:")):
            continue
        target, _, frag = t.partition("#")
        if not target:                      # in-page anchor
            if frag and frag not in anchors(path):
                bad.append((path, t, "anchor not found"))
            continue
        dest = os.path.normpath(os.path.join(os.path.dirname(path), target))
        if not os.path.exists(dest):
            bad.append((path, t, "missing file"))
        elif frag and dest.endswith(".md") and frag not in anchors(dest):
            bad.append((path, t, "anchor not found in target"))

for p, t, why in bad:
    print(f"{p}: {t}  <- {why}")
print(f"\n{len(md_files)} markdown files, {len(bad)} broken links")
sys.exit(1 if bad else 0)
