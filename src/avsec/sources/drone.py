"""One natural photograph, seventeen window-motion sequences cut from it.

What this module is, and what it is not
---------------------------------------
It builds clips from a real photograph taken by a real drone camera, so the
source-coding and burst results can be checked against content nobody in this
project designed.  It is a **demonstration set on one photograph**, not a
sample of drone imagery, and it is labelled that way everywhere:

* every clip is a *window-motion sequence over a natural image* - a crop window
  panned across a still photograph.  There is no parallax, no rolling-shutter
  skew, no exposure adaptation and no motion in the scene;
* all seventeen clips share one ``source_id``, because they share one sensor,
  one lens, one hour of one day and one landscape.  A bootstrap over them
  measures uncertainty **within this photograph** and does not generalise to a
  new recording (defect R07);
* the crop rectangles are **not** claimed to be disjoint.  Whether two of them
  overlap is measured by :meth:`avsec.dataset.DatasetManifest.measure_crop_overlap`
  and reported in the manifest, rather than asserted in prose.

For a set that *does* generalise across recordings, see
:mod:`avsec.sources.natural`: twenty-four photographs by twenty-four
photographers, one clip each.

The photograph
--------------
``data/real/curonian_spit_epha_dune.jpg`` - Curonian Spit National Park, aerial
view at the Epha Dune, 12 May 2017.  Author: A.Savin.  Camera: **DJI FC6310**
(the Phantom 4 Pro sensor).  4213 x 2633 px, the camera's own full-resolution
frame, downloaded from Wikimedia Commons and **not** re-encoded here.  Licence:
Free Art License 1.3 - free to use, copy, modify and distribute with
attribution.  The attribution travels with the data in every manifest this
repository writes.

It is a JPEG, because that is what the camera wrote and what is available under
a free licence; there is no free-licensed drone RAW of comparable quality.  Any
report that uses it therefore says "camera original", never "uncompressed".

How clips are made
------------------
Each clip pans and zooms a crop window across the photograph between a declared
start and end rectangle.  The window is then area-averaged down to the working
resolution - which is what a real sensor and lens do - and converted to luma
with the BT.601 weights the rest of the pipeline assumes.

The scenes cover different content classes, and the duplicate detector in
:mod:`avsec.dataset` is run over them like any other material; it must find
nothing.  Finding no duplicate is not the same as the crops being independent,
which is why they share a source id.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from avsec.sources import PROV_LOCAL_FILE, FrameSource

#: Where the photograph lives, relative to the repository root.
DRONE_PHOTO = os.path.join("data", "real", "curonian_spit_epha_dune.jpg")

#: Provenance recorded with every clip built from it.
PHOTO_CREDIT = {
    "title": "Curonian Spit NP, aerial view at the Epha Dune",
    "author": "A.Savin",
    "date": "2017-05-12",
    "camera": "DJI FC6310 (Phantom 4 Pro)",
    "source": "Wikimedia Commons",
    "source_url": ("https://commons.wikimedia.org/wiki/File:"
                   "Curonian_Spit_NP_05-2017_img17_aerial_view_at_Epha_Dune.jpg"),
    "license": "Free Art License 1.3",
    "license_url": "https://artlibre.org/licence/lal/en/",
    "native_pixels": "4213x2633",
    "encoding": "camera original JPEG, not re-encoded in this repository",
}

#: The one independent origin every clip in this module comes from (R07).
SOURCE_ID = "photo:curonian_spit_epha_dune"

#: Every clip is held out.  A single photograph cannot be divided into
#: calibration, validation and test: all three would share the same sensor,
#: light and landscape, which is precisely the leak the split exists to
#: prevent.  Nothing in this repository is tuned on this photograph - the
#: configurations come from the synthetic set - so holding all of it out is
#: both honest and accurate.
DECLARED_SPLIT = "test"

#: Scene definitions: ``(name, category, (x0, y0, x1, y1) start, end, motion)``.
#:
#: Rectangles are in the photograph's own pixel coordinates and cover different
#: content classes: open water and sky are almost flat, the pine canopy is the
#: hardest texture in the frame, the shoreline and the dune edge are long
#: high-contrast boundaries, and the village is a field of small bright objects.
#: Whether two windows overlap is measured in the dataset manifest, not
#: asserted here.
#:
#: ``motion`` scales how far the window travels; "static" still drifts a little,
#: because a hovering drone never holds perfectly still.
DRONE_SCENES: Tuple[Tuple[str, str, Tuple[int, int, int, int],
                          Tuple[int, int, int, int], str, str], ...] = (
    ("sea_open", "low-detail",
     (60, 620, 1340, 1580), (300, 700, 1580, 1660), "slow", "test"),
    ("sky_horizon", "low-detail",
     (1500, 40, 2780, 1000), (1740, 90, 3020, 1050), "slow", "test"),
    ("forest_canopy", "high-detail",
     (900, 1500, 2180, 2460), (1140, 1560, 2420, 2520), "slow", "test"),
    ("forest_fast", "high-detail",
     (300, 1420, 1580, 2380), (940, 1620, 2220, 2580), "fast", "test"),
    ("dune_ridge", "structure",
     (2500, 1450, 3780, 2410), (2740, 1520, 4020, 2480), "slow", "test"),
    ("dune_fast", "structure",
     (2800, 1300, 4080, 2260), (2180, 1560, 3460, 2520), "fast", "test"),
    ("shoreline", "structure",
     (120, 900, 1400, 1860), (520, 760, 1800, 1720), "slow", "test"),
    ("village", "high-detail",
     (2320, 700, 3080, 1270), (2480, 760, 3240, 1330), "slow", "test"),
    ("dune_forest_edge", "structure",
     (2000, 1700, 3280, 2660), (2240, 1660, 3520, 2620), "slow", "test"),
    ("spit_far", "low-detail",
     (1150, 300, 2430, 1260), (1390, 360, 2670, 1320), "slow", "test"),
    # ---- further regions of the same photograph -----------------------------
    # These were once labelled calibration/validation.  That was meaningless:
    # all of them are the same photograph, so a "calibration" crop shares its
    # sensor, light and terrain with every "test" crop (R07).  Nothing was ever
    # tuned on them - the configurations come from the synthetic set - so the
    # honest label is that the whole photograph is held out.
    ("cal_sea_edge", "low-detail",
     (0, 1100, 1280, 2060), (240, 1160, 1520, 2120), "slow", "test"),
    ("cal_canopy_mix", "high-detail",
     (1400, 1800, 2680, 2620), (1640, 1760, 2920, 2580), "slow", "test"),
    ("cal_dune_flat", "structure",
     (3200, 1800, 4200, 2560), (3000, 1860, 4000, 2620), "slow", "test"),
    ("cal_water_east", "low-detail",
     (3100, 600, 4180, 1410), (2980, 660, 4060, 1470), "slow", "test"),
    ("val_treeline", "structure",
     (700, 1150, 1980, 2110), (940, 1210, 2220, 2170), "slow", "test"),
    ("val_sand_texture", "high-detail",
     (2600, 2000, 3880, 2630), (2840, 1980, 4120, 2610), "fast", "test"),
    ("val_coast_curve", "structure",
     (200, 500, 1480, 1460), (440, 560, 1720, 1520), "slow", "test"),
)

_MOTION = {"static": 0.15, "slow": 1.0, "fast": 2.6}


def _to_luma(bgr: np.ndarray) -> np.ndarray:
    """BT.601 luma, the same convention the modem and the CVBS model assume."""
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    y = 0.299 * r.astype(np.float64) + 0.587 * g + 0.114 * b
    return np.clip(y, 0, 255).astype(np.uint8)


def _crop_resize(photo: np.ndarray, rect: Tuple[float, float, float, float],
                 h: int, w: int) -> np.ndarray:
    """Area-average a floating-point rectangle down to the working size.

    Area interpolation is the honest choice: it is what a sensor and lens do
    when they sample a scene, and unlike bilinear it does not invent detail or
    alias the pine canopy into a moire pattern that no camera would produce.
    """
    import cv2

    H, W = photo.shape[:2]
    x0, y0, x1, y1 = rect
    x0 = float(np.clip(x0, 0, W - 2))
    y0 = float(np.clip(y0, 0, H - 2))
    x1 = float(np.clip(x1, x0 + 2, W))
    y1 = float(np.clip(y1, y0 + 2, H))
    patch = photo[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))]
    return cv2.resize(patch, (w, h), interpolation=cv2.INTER_AREA)


def load_photo(path: Optional[str] = None) -> np.ndarray:
    import cv2

    p = path or DRONE_PHOTO
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"the drone photograph is missing: {p}\n"
            "Download it once with:\n"
            "  python scripts/fetch_drone_photo.py")
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode {p}")
    return img


def drone_suite(h: int = 192, w: int = 256, n_frames: int = 16,
                path: Optional[str] = None,
                scenes: Optional[Sequence] = None) -> List[FrameSource]:
    """Clips panned across the drone photograph, one per declared scene."""
    photo = load_photo(path)
    out: List[FrameSource] = []
    for name, category, start, end, motion, split in (scenes or DRONE_SCENES):
        k = _MOTION.get(motion, 1.0)
        frames = []
        for i in range(n_frames):
            t = (i / max(n_frames - 1, 1)) * k
            rect = tuple(float(a) + t * (float(b) - float(a))
                         for a, b in zip(start, end))
            frames.append(_to_luma(_crop_resize(photo, rect, h, w)))
        out.append(FrameSource(
            name=f"uav_{name}",
            frames=frames,
            provenance=PROV_LOCAL_FILE,
            description=(f"window-motion sequence over a natural image: pan across "
                         f"'{name}' in the Epha Dune photograph, motion={motion}, "
                         f"{w}x{h}, {n_frames} frames"),
            meta={
                "scene_id": f"uav:{name}",
                # Every clip here comes from ONE photograph, and says so (R07).
                "source_id": SOURCE_ID,
                "derivation": ("послідовність зі штучним рухом вікна по "
                               "природному зображенні: одна фотографія, "
                               "вирізка панорамується між двома прямокутниками"),
                "category": category,
                "split": split,
                "motion": f"synthetic window pan ({motion}); no parallax, "
                          "no scene motion",
                "suite": "drone",
                "crop_start": list(start),
                "crop_end": list(end),
                "path": DRONE_PHOTO,
                **{f"photo_{k2}": v for k2, v in PHOTO_CREDIT.items()},
            },
        ))
    return out


def showcase_frame(h: int = 288, w: int = 384, path: Optional[str] = None,
                   rect: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """One high-resolution frame of the whole scene, for the qualitative figures."""
    photo = load_photo(path)
    H, W = photo.shape[:2]
    r = rect or (0, 0, W, H)
    return _to_luma(_crop_resize(photo, r, h, w))


__all__ = ["DRONE_PHOTO", "PHOTO_CREDIT", "SOURCE_ID", "DRONE_SCENES", "drone_suite",
           "load_photo", "showcase_frame", "_to_luma"]
