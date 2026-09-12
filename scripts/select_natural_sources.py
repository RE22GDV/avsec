"""Pick the independent UAV photographs the natural-imagery study uses.

This is the *discovery* step, run once.  It queries Wikimedia Commons for
files whose EXIF camera model is a DJI aerial camera, keeps one photograph per
author (so that two entries are never the same flight), checks the licence and
the native resolution, and writes ``data/real/sources.json``.

Selection is deterministic: candidates are ordered by page id and the first
acceptable one per author wins, so re-running reproduces the same registry.

Nothing here is downloaded.  ``scripts/fetch_natural_sources.py`` does that and
verifies each file against the hash recorded in the registry.

Usage::

    python scripts/select_natural_sources.py --target 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "data", "real", "sources.json")

API = "https://commons.wikimedia.org/w/api.php"
UA = "avsec-research/1.0 (dataset provenance; contact via repository)"

#: DJI aerial camera model identifiers, with the airframe they belong to.
DJI_CAMERAS = {
    "FC6310": "DJI Phantom 4 Pro",
    "FC6310S": "DJI Phantom 4 Pro V2.0",
    "FC330": "DJI Phantom 4",
    "FC220": "DJI Mavic Pro",
    "FC2103": "DJI Mavic Air",
    "FC3170": "DJI Mavic Air 2",
    "FC3582": "DJI Mini 3 Pro",
    "FC7303": "DJI Mini 2",
    "FC6540": "DJI Zenmuse X5S (Inspire 2)",
    "FC300X": "DJI Phantom 3 Professional",
}

#: Licences that allow redistribution and derivative use with attribution.
ALLOWED_LICENCES = (
    "cc0", "cc-by-1.0", "cc-by-2.0", "cc-by-2.5", "cc-by-3.0", "cc-by-4.0",
    "cc-by-sa-1.0", "cc-by-sa-2.0", "cc-by-sa-2.5", "cc-by-sa-3.0",
    "cc-by-sa-4.0", "fal", "public domain",
)

MIN_WIDTH = 3000
MIN_HEIGHT = 2000
MAX_BYTES = 14_000_000


def api(**params: Any) -> Dict[str, Any]:
    """One API call, polite about rate limits.

    Commons answers 429 when a client asks too fast; backing off is part of
    using the service correctly, not an error to swallow.
    """
    import time

    params.setdefault("action", "query")
    params.setdefault("format", "json")
    body = urllib.parse.urlencode(params).encode("utf-8")
    delay = 2.0
    for attempt in range(6):
        req = urllib.request.Request(
            API, data=body,
            headers={"User-Agent": UA,
                     "Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                out = json.loads(r.read())
            time.sleep(0.7)
            return out
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503) or attempt == 5:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _clean(text: Optional[str]) -> str:
    """Strip the HTML Commons puts in author and credit fields."""
    import re

    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def candidates(model: str, limit: int = 500) -> List[str]:
    r = api(list="categorymembers", cmtitle=f"Category:Taken with DJI {model}",
            cmlimit=limit, cmtype="file")
    return [m["title"] for m in r.get("query", {}).get("categorymembers", [])]


def info(titles: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(0, len(titles), 20):
        chunk = titles[i : i + 20]
        r = api(prop="imageinfo", titles="|".join(chunk),
                iiprop="url|size|mime|extmetadata|metadata",
                iimetadataversion="latest")
        for page in r.get("query", {}).get("pages", {}).values():
            ii = (page.get("imageinfo") or [{}])[0]
            if not ii:
                continue
            ext = ii.get("extmetadata", {})
            meta = {m.get("name"): m.get("value")
                    for m in (ii.get("metadata") or []) if isinstance(m, dict)}
            out.append({
                "pageid": page.get("pageid", 0),
                "title": page.get("title", ""),
                "url": ii.get("url", ""),
                "descriptionurl": ii.get("descriptionurl", ""),
                "width": int(ii.get("width", 0)),
                "height": int(ii.get("height", 0)),
                "size": int(ii.get("size", 0)),
                "mime": ii.get("mime", ""),
                "licence": _clean(ext.get("LicenseShortName", {}).get("value")),
                "licence_id": _clean(ext.get("License", {}).get("value")).lower(),
                "licence_url": _clean(ext.get("LicenseUrl", {}).get("value")),
                "author": _clean(ext.get("Artist", {}).get("value")),
                "credit": _clean(ext.get("Credit", {}).get("value")),
                "date": _clean(ext.get("DateTimeOriginal", {}).get("value"))[:32],
                "description": _clean(
                    ext.get("ImageDescription", {}).get("value"))[:220],
                "camera_model": _clean(str(meta.get("Model", ""))),
                "camera_make": _clean(str(meta.get("Make", ""))),
            })
    return out


def acceptable(c: Dict[str, Any]) -> bool:
    if c["mime"] != "image/jpeg":
        return False
    if c["width"] < MIN_WIDTH or c["height"] < MIN_HEIGHT:
        return False
    if not (0 < c["size"] <= MAX_BYTES):
        return False
    lic = (c["licence_id"] or c["licence"]).lower()
    if not any(a in lic for a in ALLOWED_LICENCES):
        return False
    return bool(c["author"])


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=24)
    ap.add_argument("--per-model", type=int, default=500)
    args = ap.parse_args(argv)

    # Spread the quota across airframes rather than filling it from whichever
    # category answers first: a set drawn from one camera model is one sensor
    # and one lens, which is exactly the kind of hidden dependence this
    # selection exists to avoid.
    import math
    import re

    per_model = max(1, math.ceil(args.target / len(DJI_CAMERAS)))
    picked: List[Dict[str, Any]] = []
    authors: set = set()

    def author_key(name: str) -> str:
        """Letters sorted, so "Balazs Mocsar" and "Mocsarbalazs" collide.

        Two accounts of one photographer are one photographer, and counting
        them as two independent sources would overstate the sample.  Sorting
        the characters catches reordered and run-together spellings of a name;
        it can over-merge, which is the safe direction to err in here.
        """
        return "".join(sorted(re.sub(r"[^a-z0-9]", "", name.lower())))[:48]

    pools: Dict[str, List[Dict[str, Any]]] = {}
    for model in DJI_CAMERAS:
        try:
            titles = candidates(model, args.per_model)
        except Exception as exc:
            print(f"  {model}: query failed ({type(exc).__name__})")
            pools[model] = []
            continue
        pools[model] = [c for c in sorted(info(titles), key=lambda c: c["pageid"])
                        if acceptable(c)]
        print(f"  {model:8s} {len(titles):4d} candidates -> "
              f"{len(pools[model])} acceptable")

    for quota in (per_model, args.target):     # fair share first, then top up
        for model, airframe in DJI_CAMERAS.items():
            taken = sum(1 for c in picked if c["camera_category"] == model)
            for c in pools.get(model, []):
                if len(picked) >= args.target or taken >= quota:
                    break
                key = author_key(c["author"])
                if not key or key in authors:
                    continue                   # one photograph per author
                authors.add(key)
                c["airframe"] = airframe
                c["camera_category"] = model
                picked.append(c)
                taken += 1

    picked.sort(key=lambda c: c["pageid"])
    os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
    with open(REGISTRY, "w", encoding="utf-8") as fh:
        json.dump({
            "note": ("реєстр незалежних джерел: кожен запис - окрема "
                     "фотографія, зроблена іншим автором іншою камерою "
                     "та/або в іншому місці"),
            "selection": {
                "query": "Wikimedia Commons, категорії 'Taken with DJI <model>'",
                "filters": {
                    "mime": "image/jpeg",
                    "min_width": MIN_WIDTH, "min_height": MIN_HEIGHT,
                    "max_bytes": MAX_BYTES,
                    "licences": list(ALLOWED_LICENCES),
                    "one_photograph_per_author": True,
                },
                "order": "за pageid, перший придатний на автора",
            },
            "sources": picked,
        }, fh, ensure_ascii=False, indent=2)
    print(f"\n{len(picked)} sources -> {REGISTRY}")
    for c in picked:
        print(f"  {c['camera_category']:7s} {c['width']}x{c['height']} "
              f"{c['size']/1e6:5.1f} MB  {c['licence']:12s} {c['author'][:40]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
