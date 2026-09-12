"""Download and verify the independent UAV photographs listed in the registry.

The registry (``data/real/sources.json``) is committed; the image files are
not - two dozen aerial photographs are well over a hundred megabytes, which does
not belong in a source repository.  What is committed is everything needed to
obtain exactly the same bytes: the Commons page, the rendering request, the
licence, the author and, once this script has run, the SHA-256 of the file the
results were computed from.

Which rendering, and why
------------------------
Wikimedia asks clients not to pull camera originals in bulk and answers ``429``
when they do, pointing at the thumbnail service instead.  So each photograph is
fetched through ``Special:FilePath?width=N``: a JPEG the Wikimedia thumbnailer
produced from the camera original.

That is a re-encode, and the manifests say so rather than calling it an
original.  It does not affect what is measured here: every clip is a crop of
roughly a thousand pixels across, area-averaged down to 256x192 before it
reaches the codec, so the working material is far below the rendering's
resolution either way.  The one photograph committed to this repository
(``data/real/curonian_spit_epha_dune.jpg``) *is* the camera original and stays
labelled as such.

Usage::

    python scripts/fetch_natural_sources.py             # download what is missing
    python scripts/fetch_natural_sources.py --verify    # check, download nothing
    python scripts/fetch_natural_sources.py --force     # re-download everything
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

REGISTRY = os.path.join(ROOT, "data", "real", "sources.json")
STORE = os.path.join(ROOT, "data", "real", "natural")
UA = "avsec-research/1.0 (dataset provenance; contact via repository)"

#: Rendered width requested from the thumbnailer.  Roughly three times the
#: widest crop any clip takes, so the downsampling in the pipeline - not the
#: rendering - is what limits detail.
RENDER_WIDTH = 3200


def render_url(title: str, width: int = RENDER_WIDTH) -> str:
    name = re.sub(r"^File:", "", title).replace(" ", "_")
    return ("https://commons.wikimedia.org/wiki/Special:FilePath/"
            + urllib.parse.quote(name) + f"?width={int(width)}")


def slug(title: str, pageid: int) -> str:
    base = re.sub(r"^File:", "", title)
    base = os.path.splitext(base)[0]
    base = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_").lower()[:60]
    return f"{pageid:09d}_{base}.jpg"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry() -> Dict[str, Any]:
    if not os.path.exists(REGISTRY):
        raise SystemExit(
            f"{REGISTRY} is missing - run scripts/select_natural_sources.py first")
    with open(REGISTRY, encoding="utf-8") as fh:
        return json.load(fh)


def save_registry(reg: Dict[str, Any]) -> None:
    with open(REGISTRY, "w", encoding="utf-8") as fh:
        json.dump(reg, fh, ensure_ascii=False, indent=2)


def download(url: str, dest: str, attempts: int = 6) -> None:
    """Fetch one file, backing off when the service asks us to."""
    tmp = dest + ".part"
    delay = 10.0
    for attempt in range(attempts):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            os.replace(tmp, dest)
            return
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            print(f"  {exc.code}; waiting {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 180.0)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 180.0)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass


def measure(path: str) -> Dict[str, Any]:
    """Rendered size and a detail statistic, both recorded with the file."""
    import cv2
    import numpy as np

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"rendered_width": 0, "rendered_height": 0}
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return {
        "rendered_width": int(img.shape[1]),
        "rendered_height": int(img.shape[0]),
        "mean_level": round(float(img.mean()), 2),
        "gradient_rms": round(float(np.sqrt((gx ** 2 + gy ** 2).mean())), 3),
    }


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check what is present; download nothing")
    ap.add_argument("--force", action="store_true", help="re-download everything")
    ap.add_argument("--width", type=int, default=RENDER_WIDTH)
    ap.add_argument("--pause", type=float, default=1.5,
                    help="seconds between files")
    args = ap.parse_args(argv)

    reg = load_registry()
    reg["rendering"] = {
        "service": "Wikimedia Special:FilePath thumbnailer",
        "requested_width_px": args.width,
        "note": ("це рендер з камерного оригіналу, а НЕ сам оригінал; "
                 "Wikimedia просить не тягнути оригінали пакетно. Для цієї "
                 "роботи це не має значення: кожен кліп - це вирізка близько "
                 "тисячі пікселів завширшки, усереднена до 256x192 ще до "
                 "кодека"),
    }
    os.makedirs(STORE, exist_ok=True)
    ok = missing = changed = fetched = 0

    for src in reg["sources"]:
        name = src.get("local") or slug(src["title"], src["pageid"])
        src["local"] = name
        src["render_url"] = render_url(src["title"], args.width)
        path = os.path.join(STORE, name)

        if args.force or (not os.path.exists(path) and not args.verify):
            print(f"downloading {name}", flush=True)
            try:
                download(src["render_url"], path)
                fetched += 1
                src.pop("sha256", None)     # a new rendering is a new file
                time.sleep(max(0.0, args.pause))
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

        if not os.path.exists(path):
            missing += 1
            print(f"MISSING {name}")
            continue

        digest = sha256(path)
        recorded = src.get("sha256", "")
        if not recorded:
            src["sha256"] = digest
            src["bytes_on_disk"] = os.path.getsize(path)
            src.update(measure(path))
            ok += 1
        elif recorded != digest:
            changed += 1
            print(f"CHANGED {name}\n  recorded {recorded}\n  on disk  {digest}",
                  file=sys.stderr)
        else:
            ok += 1

    save_registry(reg)
    print(f"\n{len(reg['sources'])} sources: {ok} verified, {fetched} downloaded, "
          f"{missing} missing, {changed} changed")
    if changed:
        print("A changed file is a different photograph, not a different opinion: "
              "re-run the experiments before quoting a number against it.",
              file=sys.stderr)
        return 1
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
