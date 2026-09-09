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
    "default_suite", "from_image_file", "from_video_file", "from_directory",
    "from_capture_device", "HardwareCaptureUnavailable", "build_sources",
]
