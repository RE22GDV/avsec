"""Independent natural UAV imagery: one sequence per source photograph.

What this is, exactly
---------------------
Twenty-four aerial photographs, each taken by a **different photographer** with
a DJI aerial camera, in a different place, under different light, and released
under a free licence.  ``data/real/sources.json`` records for each one: the
Commons page, the rendering request, the licence, the author, the camera model,
the SHA-256 of the file on disk and its measured size and detail.

Each photograph yields **one clip**, built by panning a crop window across it -
a *window-motion sequence over a natural image*.  That name is used everywhere,
because it is what the material is.  A window panned across a still photograph
is not a flight: it has no parallax, no rolling-shutter skew, no exposure
adaptation and no scene motion.  What it does give, and what it is used for, is
natural spatial statistics under a controlled, repeatable motion.

Why one clip per photograph
---------------------------
The unit of independence is the **source recording**, not the crop (defect
R07).  Seventeen crops of one photograph share a sensor, a lens, an hour of the
day and a landscape; a bootstrap over them says how uncertain a number is
*within that photograph*, which is not what a claim about drone video needs.
Cutting one clip from each of twenty-four photographs gives twenty-four
genuinely different origins, and splits are assigned by origin.

Content categories are assigned from the **measured** gradient energy of the
photograph, recorded at download time, before any result is looked at.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from avsec.sources import PROV_LOCAL_FILE, FrameSource

#: Where the registry and the image files live, relative to the repository root.
REGISTRY = os.path.join("data", "real", "sources.json")
STORE = os.path.join("data", "real", "natural")

#: Fractions of the *source* set that go to each split.  Splits are assigned to
#: sources, in a fixed order, so the same photograph is never on both sides of
#: a calibration/test boundary.
SPLIT_PLAN = (("calibration", 5), ("validation", 5), ("test", 14))

#: Crop window as a fraction of the photograph, and how far it travels.  One
#: window per source, so that "more data" means "more photographs".
WINDOW_FRACTION = 0.34
TRAVEL_FRACTION = 0.10


class SourcesMissing(FileNotFoundError):
    """The registry or the image files are not on disk yet."""


def load_registry(path: Optional[str] = None) -> Dict[str, Any]:
    p = path or REGISTRY
    if not os.path.exists(p):
        raise SourcesMissing(
            f"{p} is missing.  Obtain the natural sources once with:\n"
            "  python scripts/fetch_natural_sources.py")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def available(path: Optional[str] = None) -> bool:
    try:
        reg = load_registry(path)
    except SourcesMissing:
        return False
    return all(os.path.exists(os.path.join(STORE, s.get("local", "")))
               for s in reg.get("sources", []))


def _category(gradient_rms: float, cuts: Tuple[float, float]) -> str:
    """Detail class from measured gradient energy, not from the file name."""
    if gradient_rms <= cuts[0]:
        return "low-detail"
    if gradient_rms <= cuts[1]:
        return "structure"
    return "high-detail"


def _splits(n: int) -> List[str]:
    out: List[str] = []
    for name, count in SPLIT_PLAN:
        out.extend([name] * count)
    if len(out) < n:
        out.extend(["test"] * (n - len(out)))
    return out[:n]


def _window(src: Dict[str, Any], w_px: int, h_px: int
            ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Start and end crop rectangles, placed deterministically per source.

    The offset and the travel direction come from the Commons page id, so the
    window is reproducible and is not chosen after seeing a result.
    """
    rng = np.random.default_rng(int(src["pageid"]) & 0xFFFFFFFF)
    cw = int(round(w_px * WINDOW_FRACTION))
    ch = int(round(h_px * WINDOW_FRACTION))
    cw = max(64, min(cw, w_px - 2))
    ch = max(64, min(ch, h_px - 2))
    x0 = int(rng.integers(0, max(1, w_px - cw)))
    y0 = int(rng.integers(0, max(1, h_px - ch)))
    dx = int(round(cw * TRAVEL_FRACTION)) * int(rng.choice([-1, 1]))
    dy = int(round(ch * TRAVEL_FRACTION)) * int(rng.choice([-1, 1]))
    x1 = int(np.clip(x0 + dx, 0, w_px - cw))
    y1 = int(np.clip(y0 + dy, 0, h_px - ch))
    return (x0, y0, x0 + cw, y0 + ch), (x1, y1, x1 + cw, y1 + ch)


def natural_suite(h: int = 192, w: int = 256, n_frames: int = 16,
                  registry: Optional[str] = None,
                  limit: int = 0) -> List[FrameSource]:
    """One window-motion sequence per independent source photograph."""
    from avsec.sources.drone import _crop_resize, _to_luma

    reg = load_registry(registry)
    entries = sorted(reg.get("sources", []), key=lambda s: int(s["pageid"]))
    if limit > 0:
        entries = entries[:limit]
    grads = sorted(float(s.get("gradient_rms", 0.0)) for s in entries)
    cuts = (grads[len(grads) // 3] if grads else 0.0,
            grads[2 * len(grads) // 3] if grads else 0.0)
    splits = _splits(len(entries))

    import cv2

    out: List[FrameSource] = []
    for i, s in enumerate(entries):
        path = os.path.join(STORE, s.get("local", ""))
        if not os.path.exists(path):
            raise SourcesMissing(
                f"{path} is missing.  Run: python scripts/fetch_natural_sources.py")
        photo = cv2.imread(path, cv2.IMREAD_COLOR)
        if photo is None:
            raise ValueError(f"cannot decode {path}")
        H, W = photo.shape[:2]
        start, end = _window(s, W, H)
        frames = []
        for k in range(n_frames):
            t = k / max(n_frames - 1, 1)
            rect = tuple(float(a) + t * (float(b) - float(a))
                         for a, b in zip(start, end))
            frames.append(_to_luma(_crop_resize(photo, rect, h, w)))
        name = f"nat_{int(s['pageid']):09d}"
        out.append(FrameSource(
            name=name,
            frames=frames,
            provenance=PROV_LOCAL_FILE,
            description=(
                "window-motion sequence over a natural aerial photograph "
                f"({s.get('airframe', 'DJI')}, {s.get('licence', '?')}, "
                f"{s.get('author', '?')})"),
            meta={
                # identity
                "scene_id": f"natural:{int(s['pageid'])}",
                "source_id": f"commons:{int(s['pageid'])}",
                "category": _category(float(s.get("gradient_rms", 0.0)), cuts),
                "split": splits[i],
                "suite": "natural",
                # how this clip was derived from that source
                "derivation": ("послідовність зі штучним рухом вікна по "
                               "природному зображенні: вирізка панорамується "
                               "між двома прямокутниками і усереднюється до "
                               f"{w}x{h}"),
                "motion": "synthetic window pan (no parallax, no scene motion)",
                "crop_start": list(start),
                "crop_end": list(end),
                "path": path,
                # provenance of the source itself
                "source_sha256": s.get("sha256", ""),
                "source_bytes": s.get("bytes_on_disk", 0),
                "source_pixels": f"{s.get('rendered_width')}x{s.get('rendered_height')}",
                "native_pixels": f"{s.get('width')}x{s.get('height')}",
                "encoding": ("Wikimedia thumbnail rendering of the camera "
                             "original, not the original file"),
                "license": s.get("licence", ""),
                "license_url": s.get("licence_url", ""),
                "photo_author": s.get("author", ""),
                "credit": s.get("credit", ""),
                "camera": f"{s.get('camera_make','')} {s.get('camera_model','')}".strip(),
                "airframe": s.get("airframe", ""),
                "date": s.get("date", ""),
                "commons_page": s.get("descriptionurl", ""),
                "gradient_rms": s.get("gradient_rms", 0.0),
            },
        ))
    return out


def registry_digest(registry: Optional[str] = None) -> str:
    """One hash over every source file's hash, to name the whole dataset."""
    reg = load_registry(registry)
    h = hashlib.sha256()
    for s in sorted(reg.get("sources", []), key=lambda s: int(s["pageid"])):
        h.update(f"{s['pageid']}:{s.get('sha256','')}".encode())
    return h.hexdigest()[:16]


def describe(registry: Optional[str] = None) -> Dict[str, Any]:
    """What the natural set is, for a manifest or a report."""
    reg = load_registry(registry)
    entries = reg.get("sources", [])
    return {
        "n_sources": len(entries),
        "n_authors": len({s.get("author", "") for s in entries}),
        "n_camera_models": len({s.get("camera_category", "") for s in entries}),
        "airframes": sorted({s.get("airframe", "") for s in entries}),
        "licences": sorted({s.get("licence", "") for s in entries}),
        "clips_per_source": 1,
        "unit_of_independence": "source photograph",
        "rendering": reg.get("rendering", {}),
        "registry_digest": registry_digest(registry),
        "limits": [
            "вікно, що рухається по нерухомому знімку, - не політ: немає "
            "паралакса, скошування рядків, адаптації експозиції й руху в сцені",
            "усі знімки - JPEG-рендери з камерних оригіналів, а не оригінали",
            "усі камери - побутові DJI; це не вибірка «усіх БпЛА»",
        ],
    }


__all__ = ["REGISTRY", "STORE", "SPLIT_PLAN", "SourcesMissing", "available",
           "load_registry", "natural_suite", "registry_digest", "describe"]
