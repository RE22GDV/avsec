"""Frame sources: procedural test material, image/video files, hardware adapters.

Every source yields ``uint8`` grayscale frames of a fixed size.  A source states
its own provenance so that a report can never silently claim that a synthetic
sequence is a real recording.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from avsec.utils import experiment_rng, sha256_array, sha256_file

Provenance = str
PROV_SYNTHETIC = "synthetic"          # procedurally generated in this repo
PROV_LOCAL_FILE = "local_file"        # user supplied image / video file
PROV_HARDWARE = "hardware_capture"    # real capture device (never faked)


@dataclass
class FrameSource:
    """A finite sequence of grayscale frames with declared provenance."""

    name: str
    frames: List[np.ndarray]
    provenance: Provenance
    description: str = ""
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for f in self.frames:
            if f.ndim != 2 or f.dtype != np.uint8:
                raise ValueError("frames must be 2-D uint8 grayscale")
        if self.frames:
            h, w = self.frames[0].shape
            for f in self.frames:
                if f.shape != (h, w):
                    raise ValueError("all frames of a source must share one size")

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.frames)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.frames[i]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.frames[0].shape if self.frames else (0, 0)

    def content_hash(self) -> str:
        return sha256_array(np.stack(self.frames)) if self.frames else ""

    def summary(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "provenance": self.provenance,
            "description": self.description,
            "n_frames": len(self.frames),
            "height": int(self.shape[0]),
            "width": int(self.shape[1]),
            "content_sha256": self.content_hash(),
            **{f"meta_{k}": v for k, v in self.meta.items()},
        }


# --------------------------------------------------------------------- patterns
def _grid(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w]
    return yy.astype(np.float64), xx.astype(np.float64)


def pattern_smooth(h: int, w: int, phase: float = 0.0) -> np.ndarray:
    yy, xx = _grid(h, w)
    v = (
        110
        + 55 * np.sin(2 * np.pi * (xx / max(w, 1) * 1.3 + phase))
        + 40 * np.cos(2 * np.pi * (yy / max(h, 1) * 0.9 - 0.4 * phase))
    )
    return np.clip(v, 0, 255).astype(np.uint8)


def pattern_edges(h: int, w: int, phase: float = 0.0) -> np.ndarray:
    img = np.full((h, w), 30, dtype=np.uint8)
    off = int(phase * w) % max(w, 1)
    for i, (x0, x1, val) in enumerate(
        [(0.05, 0.28, 235), (0.34, 0.52, 150), (0.58, 0.74, 90), (0.80, 0.96, 205)]
    ):
        a = (int(x0 * w) + off) % w
        b = (int(x1 * w) + off) % w
        y0, y1 = int(0.12 * h) + i * 3, int(0.88 * h) - i * 3
        if a < b:
            img[y0:y1, a:b] = val
        else:
            img[y0:y1, a:] = val
            img[y0:y1, :b] = val
    img[int(0.45 * h) : int(0.55 * h), :] = 255
    return img


def pattern_texture(h: int, w: int, phase: float = 0.0, seed: int = 7) -> np.ndarray:
    """Fine but *spatially correlated* texture, like gravel or foliage.

    Uncorrelated pixel noise is incompressible and would make every rate
    measurement meaningless, so the noise here is low-pass filtered - which is
    also what a real camera and lens deliver.
    """
    from scipy import ndimage

    rng = experiment_rng(seed, "texture", round(phase, 6))
    noise = ndimage.gaussian_filter(rng.normal(0.0, 1.0, size=(h, w)), 1.1)
    noise = noise / max(float(noise.std()), 1e-6) * 26.0
    yy, xx = _grid(h, w)
    fine = 22 * np.sin(2 * np.pi * (xx * 0.055 + yy * 0.031 + phase * 3))
    fine += 16 * np.sin(2 * np.pi * (xx * 0.013 - yy * 0.021))
    shade = 18 * np.cos(2 * np.pi * yy / max(h, 1))
    return np.clip(128 + noise + fine + shade, 0, 255).astype(np.uint8)


def pattern_text(h: int, w: int, phase: float = 0.0, text: Optional[str] = None) -> np.ndarray:
    """Rendered text - the hardest content for lossy coding and for scrambling attacks."""
    import cv2

    img = np.full((h, w), 235, dtype=np.uint8)
    lines = (text or "AVSEC TESTBED\nAEAD + FEC\nCVBS RASTER\n0123456789\nPSNR/SSIM").split("\n")
    scale = max(0.35, h / 260.0)
    y = int(0.16 * h)
    shift = int(phase * 12)
    for ln in lines:
        cv2.putText(img, ln, (max(2, 6 + shift), y), cv2.FONT_HERSHEY_SIMPLEX, scale, 15,
                    max(1, int(scale * 2)), cv2.LINE_AA)
        y += int(0.17 * h)
        if y > h - 4:
            break
    return img


def pattern_illumination(h: int, w: int, phase: float = 0.0) -> np.ndarray:
    base = pattern_edges(h, w, 0.0).astype(np.float64)
    yy, xx = _grid(h, w)
    gain = 0.55 + 0.75 * (0.5 + 0.5 * np.sin(2 * np.pi * phase)) * (0.4 + 0.9 * xx / max(w, 1))
    return np.clip(base * gain, 0, 255).astype(np.uint8)


def pattern_horizon(h: int, w: int, phase: float = 0.0) -> np.ndarray:
    """Sky over ground with a tilting horizon - the commonest UAV downlink frame."""
    yy, xx = _grid(h, w)
    tilt = 0.18 * np.sin(2 * np.pi * phase)
    horizon = h * (0.42 + 0.06 * np.cos(2 * np.pi * phase)) + tilt * (xx - w / 2)
    sky = 190 - 40 * yy / max(h, 1)
    ground = 95 + 30 * np.sin(2 * np.pi * (xx / max(w, 1) * 2.0 + phase))
    ground += 18 * np.sin(2 * np.pi * (yy / max(h, 1) * 5.0 - phase))
    img = np.where(yy < horizon, sky, ground)
    band = np.abs(yy - horizon) < 1.5
    img = np.where(band, 235, img)
    return np.clip(img, 0, 255).astype(np.uint8)


def pattern_road(h: int, w: int, phase: float = 0.0) -> np.ndarray:
    """Converging lines: strong geometry, sensitive to horizontal jitter."""
    yy, xx = _grid(h, w)
    depth = np.clip(yy / max(h, 1), 0.02, 1.0)
    centre = w * (0.5 + 0.12 * np.sin(2 * np.pi * phase))
    half = w * 0.06 + w * 0.42 * depth
    img = np.full((h, w), 120.0)
    img += 45 * np.sin(2 * np.pi * (yy / max(h, 1) * 3.0 + phase * 2))
    road = np.abs(xx - centre) < half
    img = np.where(road, 70.0, img)
    dash = (np.abs(xx - centre) < half * 0.06) & (((yy / max(h, 1) * 14 + phase * 9) % 2) < 1)
    img = np.where(dash, 240.0, img)
    edge = np.abs(np.abs(xx - centre) - half) < 1.2
    img = np.where(edge, 215.0, img)
    return np.clip(img, 0, 255).astype(np.uint8)


def pattern_targets(h: int, w: int, phase: float = 0.0, seed: int = 11) -> np.ndarray:
    """Small high-contrast objects on a soft background - detail that matters."""
    rng = experiment_rng(seed, "targets")
    img = pattern_smooth(h, w, 0.25 * phase).astype(np.float64) * 0.55 + 70.0
    n = 9
    for i in range(n):
        cy = float(rng.uniform(0.12, 0.88) * h)
        cx = float((rng.uniform(0.08, 0.92) + 0.15 * np.sin(2 * np.pi * (phase + i / n))) * w)
        r = float(rng.uniform(0.018, 0.05) * min(h, w))
        yy, xx = _grid(h, w)
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        val = 245.0 if i % 2 == 0 else 20.0
        img = np.where(d < r, val, img)
        img = np.where((d >= r) & (d < r + 1.2), 0.5 * (val + img), img)
    return np.clip(img, 0, 255).astype(np.uint8)


def pattern_grid(h: int, w: int, phase: float = 0.0) -> np.ndarray:
    """Regular grid - deliberately close to the symbol cell pitch, aliasing-prone."""
    yy, xx = _grid(h, w)
    period = 12.0
    gx = (np.mod(xx + phase * period * 4, period) < 2.0)
    gy = (np.mod(yy + phase * period * 2, period) < 2.0)
    img = np.full((h, w), 205.0)
    img = np.where(gx | gy, 40.0, img)
    img += 12 * np.sin(2 * np.pi * (xx + yy) / max(w, 1))
    return np.clip(img, 0, 255).astype(np.uint8)


def pattern_clouds(h: int, w: int, phase: float = 0.0, seed: int = 23) -> np.ndarray:
    """Low-frequency field: easy to compress, unforgiving of banding."""
    from scipy import ndimage

    rng = experiment_rng(seed, "clouds")
    base = rng.normal(0.0, 1.0, size=(h, w))
    field = np.zeros_like(base)
    for octave, weight in ((18.0, 1.0), (7.0, 0.5), (3.0, 0.25)):
        field += weight * ndimage.gaussian_filter(base, octave)
    field = field / max(float(field.std()), 1e-6)
    yy, xx = _grid(h, w)
    drift = ndimage.shift(field, (0.0, phase * w * 0.25), mode="wrap", order=1)
    return np.clip(150 + 45 * drift + 12 * np.cos(2 * np.pi * yy / max(h, 1)),
                   0, 255).astype(np.uint8)


def pattern_lowlight(h: int, w: int, phase: float = 0.0, seed: int = 31) -> np.ndarray:
    """Dark scene with sensor noise: little headroom above the black level."""
    from scipy import ndimage

    rng = experiment_rng(seed, "lowlight", round(phase, 6))
    yy, xx = _grid(h, w)
    # a few dim sources in the dark, not a darkened copy of another pattern
    base = np.full((h, w), 14.0)
    for i, (fy, fx, amp) in enumerate(((0.30, 0.22, 70.0), (0.68, 0.55, 46.0),
                                       (0.44, 0.81, 58.0))):
        cy = (fy + 0.03 * np.sin(2 * np.pi * (phase + i * 0.3))) * h
        cx = (fx + 0.05 * np.cos(2 * np.pi * (phase + i * 0.2))) * w
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        base += amp * np.exp(-d2 / (2 * (0.09 * min(h, w)) ** 2))
    grain = ndimage.gaussian_filter(rng.normal(0.0, 1.0, size=(h, w)), 0.7) * 9.0
    return np.clip(base + grain, 0, 255).astype(np.uint8)


def pattern_blocks(h: int, w: int, rows: int, cols: int) -> np.ndarray:
    """Uniquely identifiable tiles - the chosen-plaintext frame for permutation attacks."""
    img = np.zeros((h, w), dtype=np.uint8)
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    n = rows * cols
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            # low-frequency ramp unique per tile plus a per-tile corner marker
            val = 20 + int(215 * idx / max(n - 1, 1))
            tile = np.full((ys[r + 1] - ys[r], xs[c + 1] - xs[c]), val, dtype=np.uint8)
            if tile.shape[0] > 3 and tile.shape[1] > 3:
                tile[0:2, 0:2] = 255 if idx % 2 == 0 else 0
                tile[-2:, -2:] = (idx * 37) % 256
            img[ys[r] : ys[r + 1], xs[c] : xs[c + 1]] = tile
    return img


PATTERNS = {
    "smooth": pattern_smooth,
    "edges": pattern_edges,
    "texture": pattern_texture,
    "text": pattern_text,
    "illumination": pattern_illumination,
    "horizon": pattern_horizon,
    "road": pattern_road,
    "targets": pattern_targets,
    "grid": pattern_grid,
    "clouds": pattern_clouds,
    "lowlight": pattern_lowlight,
}

#: Content category of each pattern, used by the dataset manifest so that a
#: result can be reported per category instead of per individual clip.
PATTERN_CATEGORY = {
    "smooth": "low-detail", "clouds": "low-detail", "illumination": "low-detail",
    "edges": "structure", "road": "structure", "grid": "structure",
    "horizon": "structure",
    "texture": "high-detail", "targets": "high-detail", "lowlight": "high-detail",
    "text": "text",
}


# --------------------------------------------------------------------- builders
def synthetic_still(name: str, h: int, w: int, phase: float = 0.0) -> FrameSource:
    if name not in PATTERNS:
        raise KeyError(f"unknown pattern {name!r}; have {sorted(PATTERNS)}")
    frame = PATTERNS[name](h, w, phase)
    return FrameSource(
        name=f"still_{name}",
        frames=[frame],
        provenance=PROV_SYNTHETIC,
        description=f"procedurally generated still pattern '{name}' ({w}x{h})",
    )


def synthetic_sequence(
    name: str, h: int, w: int, n_frames: int = 8, motion: str = "slow"
) -> FrameSource:
    """Synthetic moving sequence.  ``motion`` in {'static','slow','fast'}."""
    speeds = {"static": 0.0, "slow": 0.02, "fast": 0.10}
    if motion not in speeds:
        raise KeyError(f"unknown motion {motion!r}")
    step = speeds[motion]
    frames = [PATTERNS[name](h, w, i * step) for i in range(n_frames)]
    return FrameSource(
        name=f"seq_{name}_{motion}",
        frames=frames,
        provenance=PROV_SYNTHETIC,
        description=f"procedurally generated sequence '{name}', motion={motion}, {w}x{h}",
        meta={"motion": motion, "phase_step": step},
    )


def chosen_plaintext_source(h: int, w: int, rows: int, cols: int) -> FrameSource:
    return FrameSource(
        name=f"chosen_blocks_{rows}x{cols}",
        frames=[pattern_blocks(h, w, rows, cols)],
        provenance=PROV_SYNTHETIC,
        description="chosen-plaintext frame with uniquely identifiable tiles",
        meta={"grid_rows": rows, "grid_cols": cols},
    )


def default_suite(h: int, w: int, n_frames: int = 6) -> List[FrameSource]:
    """The shared comparison material: content classes required by the protocol."""
    out = [
        synthetic_still("smooth", h, w),
        synthetic_still("text", h, w),
        synthetic_still("texture", h, w),
        synthetic_still("edges", h, w),
        synthetic_sequence("illumination", h, w, n_frames, "slow"),
        synthetic_sequence("edges", h, w, n_frames, "slow"),
        synthetic_sequence("texture", h, w, n_frames, "fast"),
    ]
    return out




def _scene_variant(frame: np.ndarray, variant: int) -> np.ndarray:
    """Deterministic viewpoint and exposure change that defines a *new scene*.

    Two clips of the same pattern with different motion are not two scenes -
    they are one scene filmed twice, and the duplicate detector says so.  A
    variant applies a fixed rotation, zoom, mirror and exposure, i.e. what
    actually distinguishes two shots of the same kind of terrain.  The same
    variant is applied to every frame of a clip, so motion is preserved.
    """
    if variant == 0:
        return frame
    import cv2

    h, w = frame.shape
    angle = ((variant * 37) % 21) - 10.0            # -10..+10 degrees
    zoom = 1.0 + 0.06 * ((variant * 13) % 5)        # 1.00..1.24
    mirror = bool(variant % 2)
    gain = 0.80 + 0.09 * ((variant * 7) % 5)        # 0.80..1.16
    offset = -14.0 + 7.0 * ((variant * 11) % 5)     # -14..+14
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, zoom)
    out = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    if mirror:
        out = out[:, ::-1]
    return np.clip(out.astype(np.float64) * gain + offset, 0, 255).astype(np.uint8)


#: The research material of E02, declared before anything is run.
#:
#: Each entry is ``(pattern, motion, phase0, viewpoint variant, split)``.  A
#: variant is a fixed rotation, zoom, mirror and exposure, so two clips of the
#: same pattern are two different shots of that kind of terrain - not the same
#: scene twice.  The duplicate detector in :mod:`avsec.dataset` is run over the
#: whole set and must find nothing; the splits below are fixed here rather than
#: drawn at run time so that a later run cannot move a scene between them.
#:
#: 20 test scenes, 8 calibration, 6 validation.  All of it is SYNTHETIC.
RESEARCH_SCENES = (
    # ---- test (20) -----------------------------------------------------
    ("horizon", "slow", 0.00, 0, "test"),
    ("horizon", "fast", 0.37, 3, "test"),
    ("road", "slow", 0.00, 0, "test"),
    ("road", "fast", 0.51, 5, "test"),
    ("targets", "slow", 0.00, 0, "test"),
    ("targets", "fast", 0.29, 7, "test"),
    ("texture", "slow", 0.00, 0, "test"),
    ("texture", "fast", 0.61, 2, "test"),
    ("edges", "slow", 0.00, 0, "test"),
    ("edges", "fast", 0.13, 9, "test"),
    ("clouds", "slow", 0.00, 0, "test"),
    ("clouds", "fast", 0.44, 4, "test"),
    ("grid", "slow", 0.00, 0, "test"),
    ("grid", "fast", 0.72, 6, "test"),
    ("lowlight", "slow", 0.00, 0, "test"),
    ("lowlight", "fast", 0.18, 8, "test"),
    ("text", "slow", 0.00, 0, "test"),
    ("text", "fast", 0.55, 11, "test"),
    ("illumination", "slow", 0.00, 0, "test"),
    ("smooth", "slow", 0.00, 0, "test"),
    # ---- calibration (8) -----------------------------------------------
    ("horizon", "slow", 0.21, 13, "calibration"),
    ("road", "slow", 0.33, 15, "calibration"),
    ("targets", "slow", 0.47, 17, "calibration"),
    ("texture", "slow", 0.09, 19, "calibration"),
    ("edges", "slow", 0.66, 21, "calibration"),
    ("clouds", "slow", 0.28, 23, "calibration"),
    ("text", "slow", 0.81, 25, "calibration"),
    ("lowlight", "slow", 0.05, 27, "calibration"),
    # ---- validation (6) ------------------------------------------------
    ("horizon", "fast", 0.74, 29, "validation"),
    ("road", "fast", 0.12, 31, "validation"),
    ("targets", "fast", 0.58, 33, "validation"),
    ("texture", "fast", 0.86, 35, "validation"),
    ("grid", "slow", 0.41, 37, "validation"),
    ("clouds", "fast", 0.63, 39, "validation"),
)


def research_suite(h: int, w: int, n_frames: int = 16,
                   scenes: Optional[Sequence[Tuple[str, str, float, int, str]]] = None
                   ) -> List[FrameSource]:
    """The research material of E02: continuous clips, one scene each.

    Every clip is a genuinely different scene - a different pattern, a
    different motion regime or a different starting phase - so that scenes can
    be treated as independent clusters in the statistics.  They remain
    *synthetic*: that limitation is carried in ``provenance`` and repeated in
    every report built from them.
    """
    speeds = {"static": 0.0, "slow": 0.02, "fast": 0.10}
    out: List[FrameSource] = []
    for pattern, motion, phase0, variant, split in (scenes or RESEARCH_SCENES):
        step = speeds[motion]
        frames = [_scene_variant(PATTERNS[pattern](h, w, phase0 + i * step), variant)
                  for i in range(n_frames)]
        out.append(FrameSource(
            name=f"clip_{pattern}_{motion}_v{variant}",
            frames=frames,
            provenance=PROV_SYNTHETIC,
            description=(f"procedural research clip '{pattern}', motion={motion}, "
                         f"phase0={phase0}, viewpoint variant {variant}, "
                         f"{w}x{h}, {n_frames} frames"),
            meta={"pattern": pattern, "motion": motion, "phase0": phase0,
                  "variant": variant, "scene_id": f"synthetic:{pattern}-v{variant}",
                  "category": PATTERN_CATEGORY.get(pattern, "other"),
                  "split": split, "suite": "research"},
        ))
    return out


# --------------------------------------------------------------------- file I/O
def from_image_file(path: str, h: Optional[int] = None, w: Optional[int] = None) -> FrameSource:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read image {path!r}")
    if h and w and img.shape != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return FrameSource(
        name=os.path.splitext(os.path.basename(path))[0],
        frames=[img],
        provenance=PROV_LOCAL_FILE,
        description=f"local image file {os.path.basename(path)}",
        meta={"path": os.path.abspath(path), "file_sha256": sha256_file(path)},
    )


def from_video_file(
    path: str, h: Optional[int] = None, w: Optional[int] = None, max_frames: int = 32,
    stride: int = 1,
) -> FrameSource:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path!r}")
    frames: List[np.ndarray] = []
    idx = 0
    try:
        while len(frames) < max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            if idx % max(1, stride) == 0:
                g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                if h and w and g.shape != (h, w):
                    g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
                frames.append(g)
            idx += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path!r}")
    return FrameSource(
        name=os.path.splitext(os.path.basename(path))[0],
        frames=frames,
        provenance=PROV_LOCAL_FILE,
        description=f"local video file {os.path.basename(path)} ({len(frames)} frames)",
        meta={"path": os.path.abspath(path), "file_sha256": sha256_file(path), "stride": stride},
    )


def from_directory(
    directory: str, h: Optional[int] = None, w: Optional[int] = None, max_items: int = 64
) -> List[FrameSource]:
    """Load a user supplied dataset directory (images and/or videos)."""
    out: List[FrameSource] = []
    img_ext = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.pgm")
    vid_ext = ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.y4m")
    for pat in img_ext:
        for p in sorted(glob.glob(os.path.join(directory, pat))):
            if len(out) >= max_items:
                return out
            out.append(from_image_file(p, h, w))
    for pat in vid_ext:
        for p in sorted(glob.glob(os.path.join(directory, pat))):
            if len(out) >= max_items:
                return out
            out.append(from_video_file(p, h, w))
    return out


# ------------------------------------------------------------- hardware adapter
class HardwareCaptureUnavailable(RuntimeError):
    """Raised when a real capture device was requested but is not present.

    The hardware path must never silently fall back to synthetic frames.
    """


def from_capture_device(index: int = 0, h: Optional[int] = None, w: Optional[int] = None,
                        max_frames: int = 32) -> FrameSource:
    """Read from a real V4L2/DirectShow capture device (e.g. a USB video grabber).

    Status: implemented but **not verified against physical hardware** in this
    repository.  See ``docs/hardware.md``.
    """
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise HardwareCaptureUnavailable(
            f"capture device index {index} is not available on this machine"
        )
    frames: List[np.ndarray] = []
    try:
        for _ in range(max_frames):
            ok, fr = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if h and w and g.shape != (h, w):
                g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
            frames.append(g)
    finally:
        cap.release()
    if not frames:
        raise HardwareCaptureUnavailable(f"capture device {index} returned no frames")
    return FrameSource(
        name=f"capture{index}",
        frames=frames,
        provenance=PROV_HARDWARE,
        description=f"live capture from device index {index}",
        meta={"device_index": index},
    )


def build_sources(spec: Dict[str, object]) -> List[FrameSource]:
    """Build the source list from a config block.

    Keys: ``kind`` in {'synthetic','directory','image','video','capture'},
    plus ``height``/``width`` and kind specific fields.
    """
    kind = str(spec.get("kind", "synthetic"))
    h = int(spec.get("height", 240))
    w = int(spec.get("width", 320))
    if kind == "research":
        return research_suite(h, w, int(spec.get("n_frames", 16)))
    if kind == "drone":
        from avsec.sources.drone import DRONE_SCENES, drone_suite

        want = spec.get("scene")
        picked = ([s for s in DRONE_SCENES if s[0] == str(want)] or None) if want else None
        return drone_suite(h, w, int(spec.get("n_frames", 16)),
                           path=spec.get("path"), scenes=picked)
    if kind == "natural":
        # Independent source photographs, one window-motion sequence each (R07).
        from avsec.sources.natural import natural_suite

        return natural_suite(h, w, int(spec.get("n_frames", 16)),
                             registry=spec.get("registry"),
                             limit=int(spec.get("limit", 0)))
    if kind == "synthetic":
        n = int(spec.get("n_frames", 6))
        names = spec.get("patterns")
        if names:
            out = []
            for nm in names:  # type: ignore[union-attr]
                out.append(synthetic_sequence(str(nm), h, w, n, str(spec.get("motion", "slow")))
                           if n > 1 else synthetic_still(str(nm), h, w))
            return out
        return default_suite(h, w, n)
    if kind == "directory":
        return from_directory(str(spec["path"]), h, w, int(spec.get("max_items", 64)))
    if kind == "image":
        return [from_image_file(str(spec["path"]), h, w)]
    if kind == "video":
        return [from_video_file(str(spec["path"]), h, w, int(spec.get("max_frames", 32)),
                                int(spec.get("stride", 1)))]
    if kind == "capture":
        return [from_capture_device(int(spec.get("index", 0)), h, w,
                                    int(spec.get("max_frames", 32)))]
    raise KeyError(f"unknown source kind {kind!r}")


__all__ = [
    "FrameSource", "PROV_SYNTHETIC", "PROV_LOCAL_FILE", "PROV_HARDWARE", "PATTERNS",
    "pattern_blocks", "synthetic_still", "synthetic_sequence", "chosen_plaintext_source",
    "default_suite", "research_suite", "RESEARCH_SCENES",
    "PATTERN_CATEGORY", "from_image_file", "from_video_file", "from_directory",
    "from_capture_device", "HardwareCaptureUnavailable", "build_sources",
]
