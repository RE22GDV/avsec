"""Download and verify the drone photograph the real-imagery tests use.

The photograph is committed to this repository (it is small enough and the
licence allows redistribution), so this script normally only *verifies* it.
Run it if the file is missing, or to confirm that the copy you have is the one
the published results were computed from.

Usage::

    python scripts/fetch_drone_photo.py            # verify, download if missing
    python scripts/fetch_drone_photo.py --force    # re-download
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from avsec.sources.drone import DRONE_PHOTO, PHOTO_CREDIT  # noqa: E402

URL = ("https://upload.wikimedia.org/wikipedia/commons/d/d2/"
       "Curonian_Spit_NP_05-2017_img17_aerial_view_at_Epha_Dune.jpg")

#: SHA-256 of the exact bytes every published real-imagery number was computed
#: from.  A different hash means a different file, not a different opinion.
EXPECTED_SHA256 = ("53fc9411409c8e2c7ecd097ee9026b61"
                   "779f68541b7ea08c157cb1cc321a086d")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list) -> int:
    force = "--force" in argv
    path = os.path.join(ROOT, DRONE_PHOTO)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if force or not os.path.exists(path):
        print(f"downloading {URL}", flush=True)
        req = urllib.request.Request(
            URL, headers={"User-Agent": "avsec-research/0.1 "
                                        "(+https://github.com/RE22GDV/avsec)"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as fh:
            fh.write(r.read())

    digest = sha256(path)
    size = os.path.getsize(path)
    print(f"file   : {path}")
    print(f"size   : {size / 1e6:.2f} MB")
    print(f"sha256 : {digest}")
    for k, v in PHOTO_CREDIT.items():
        print(f"{k:14s}: {v}")

    if EXPECTED_SHA256 and digest != EXPECTED_SHA256:
        print("\nNOTE: this copy differs from the one the published numbers were "
              "computed from. Re-run the experiments before quoting any figure "
              "against it.", file=sys.stderr)
        return 1
    print("\nOK: this is the copy the published real-imagery results came from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
