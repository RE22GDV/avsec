"""Dataset identity, duplicate control and source-level splits.

Three nested levels of identity, and they are not interchangeable:

``clip_id``
    the smallest thing a job may process end to end.

``parent_scene_id``
    clips that are the same content seen slightly differently - a still taken
    from a sequence, a rescaled copy, a motion variant.  Putting one in the
    calibration split and another in the test split leaks the answer, so whole
    scenes move together.

``source_id``
    the **independent origin**: one photograph, one recording, one flight.
    Several scenes cut from one photograph share a source id.  They are not
    independent samples of "drone imagery": they share the sensor, the optics,
    the lighting, the weather and the terrain.  A bootstrap over such scenes
    estimates the uncertainty *within that photograph*; only a bootstrap over
    sources generalises to a new recording (defect R07).

Splits are assigned by **source**, so no two crops of one photograph can end up
on opposite sides of a calibration/test boundary.  Exact and near duplicates
are detected explicitly, and crops of one image are additionally checked for
geometric overlap, so an overlap claim is measured rather than asserted.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from avsec.sources import FrameSource
from avsec.utils import ensure_dir, write_csv, write_json

SPLITS = ("calibration", "validation", "test")

# Content categories used to report effects per content class.  Assigned before
# any result is looked at.
CATEGORIES = ("smooth", "text", "fine_texture", "structure_edges", "vegetation_water",
              "motion_lighting", "other")

_SYNTHETIC_CATEGORY = {
    "smooth": "smooth",
    "text": "text",
    "texture": "fine_texture",
    "edges": "structure_edges",
    "illumination": "motion_lighting",
}


def dhash(frame: np.ndarray, size: int = 8) -> int:
    """64-bit difference hash - cheap, robust to rescaling and mild filtering."""
    import cv2

    small = cv2.resize(frame, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).astype(np.uint64).ravel()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def normalised_hash(frames: Sequence[np.ndarray]) -> str:
    """SHA-256 of the frames after the pipeline's own normalisation."""
    h = hashlib.sha256()
    for f in frames:
        h.update(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    return h.hexdigest()


@dataclass
class ClipRecord:
    """One clip: the smallest thing a job may process end to end."""

    clip_id: str
    parent_scene_id: str
    category: str
    provenance: str
    source_kind: str
    n_frames: int
    width: int
    height: int
    fps: float
    t_start_s: float
    t_end_s: float
    content_sha256: str
    first_frame_dhash: str
    license: str = "generated-in-repo"
    split: str = ""
    notes: str = ""
    #: Independent origin - one photograph, one recording, one flight (R07).
    source_id: str = ""
    #: How the clip was derived from that origin, in plain words.
    derivation: str = ""
    #: Crop rectangle in the original image, when the clip is a window on one.
    crop: str = ""
    source_sha256: str = ""
    source_license: str = ""
    source_credit: str = ""

    def __post_init__(self) -> None:
        if not self.source_id:
            self.source_id = self.parent_scene_id

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DuplicateFinding:
    kind: str                 # "exact" | "near"
    clip_a: str
    clip_b: str
    scene_a: str
    scene_b: str
    distance: int
    same_scene: bool

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetManifest:
    """Clips, their scenes, their splits, and the duplicate report."""

    def __init__(self, clips: Optional[List[ClipRecord]] = None) -> None:
        self.clips: List[ClipRecord] = list(clips or [])
        self.duplicates: List[DuplicateFinding] = []
        self.crop_overlaps: List[Dict[str, Any]] = []

    # ------------------------------------------------------------- building
    @staticmethod
    def from_sources(sources: Sequence[FrameSource], fps: float = 8.333,
                     scene_of=None) -> "DatasetManifest":
        """Build records from frame sources.

        ``scene_of(source) -> str`` decides which scene a clip belongs to.  The
        default groups the procedural material by its **pattern**, because
        ``still_edges`` and frame 0 of ``seq_edges_slow`` are literally the same
        picture; treating them as two scenes would split identical content
        across two splits.
        """
        scene_of = scene_of or default_scene_id
        clips: List[ClipRecord] = []
        for s in sources:
            h, w = s.shape
            clips.append(ClipRecord(
                clip_id=s.name,
                parent_scene_id=scene_of(s),
                category=default_category(s),
                provenance=s.provenance,
                source_kind=str(s.meta.get("path", s.description))[:200],
                n_frames=len(s), width=int(w), height=int(h), fps=float(fps),
                t_start_s=0.0, t_end_s=round(len(s) / max(fps, 1e-9), 4),
                content_sha256=normalised_hash(s.frames),
                first_frame_dhash=f"{dhash(s.frames[0]):016x}" if s.frames else "",
                license="generated-in-repo" if s.provenance == "synthetic" else
                        str(s.meta.get("license", "unknown")),
                split=str(s.meta.get("split", "")),
                notes=s.description,
                source_id=default_source_id(s, scene_of),
                derivation=str(s.meta.get("derivation", "")),
                crop=_crop_text(s.meta),
                source_sha256=str(s.meta.get("source_sha256", "")),
                source_license=str(s.meta.get("license", "")),
                source_credit=str(s.meta.get("photo_author",
                                             s.meta.get("credit", ""))),
            ))
        m = DatasetManifest(clips)
        m.detect_duplicates(sources)
        m.measure_crop_overlap(sources)
        return m

    # ---------------------------------------------------------- duplicates
    def detect_duplicates(self, sources: Optional[Sequence[FrameSource]] = None,
                          near_threshold: int = 5) -> List[DuplicateFinding]:
        """Exact (identical content hash) and near (dHash) duplicate pairs.

        Near duplicates are compared on the **first frame** of each clip, which
        is what catches a still that was taken from a sequence.
        """
        self.duplicates = []
        by_hash: Dict[str, List[ClipRecord]] = {}
        for c in self.clips:
            by_hash.setdefault(c.content_sha256, []).append(c)
        for group in by_hash.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    self.duplicates.append(DuplicateFinding(
                        "exact", a.clip_id, b.clip_id, a.parent_scene_id,
                        b.parent_scene_id, 0,
                        a.parent_scene_id == b.parent_scene_id))

        hashes = [(c, int(c.first_frame_dhash, 16)) for c in self.clips
                  if c.first_frame_dhash]
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                (a, ha), (b, hb) = hashes[i], hashes[j]
                if a.content_sha256 == b.content_sha256:
                    continue                       # already reported as exact
                d = hamming(ha, hb)
                if d <= near_threshold:
                    self.duplicates.append(DuplicateFinding(
                        "near", a.clip_id, b.clip_id, a.parent_scene_id,
                        b.parent_scene_id, d,
                        a.parent_scene_id == b.parent_scene_id))
        return self.duplicates

    # ---------------------------------------------------------- crop overlap
    def measure_crop_overlap(self, sources: Optional[Sequence[FrameSource]] = None
                             ) -> List[Dict[str, Any]]:
        """Measure, do not assert, whether two windows on one image overlap.

        Non-overlap of crops used to be claimed in prose.  Here it is computed:
        for every pair of clips that share a ``source_id`` and declare a crop
        rectangle, the intersection-over-union of the *swept* rectangles is
        recorded.  ``> 0`` means the two clips literally show some of the same
        pixels of the same photograph (R07).
        """
        self.crop_overlaps = []
        by_source: Dict[str, List[ClipRecord]] = {}
        for c in self.clips:
            if c.crop:
                by_source.setdefault(c.source_id, []).append(c)
        for source_id, group in sorted(by_source.items()):
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    ra = [int(v) for v in a.crop.split(",")]
                    rb = [int(v) for v in b.crop.split(",")]
                    ix = max(0, min(ra[2], rb[2]) - max(ra[0], rb[0]))
                    iy = max(0, min(ra[3], rb[3]) - max(ra[1], rb[1]))
                    inter = ix * iy
                    aa = (ra[2] - ra[0]) * (ra[3] - ra[1])
                    ab = (rb[2] - rb[0]) * (rb[3] - rb[1])
                    union = aa + ab - inter
                    self.crop_overlaps.append({
                        "source_id": source_id, "clip_a": a.clip_id,
                        "clip_b": b.clip_id, "intersection_px": int(inter),
                        "iou": round(inter / union, 4) if union else 0.0,
                        "overlaps": bool(inter > 0),
                    })
        return self.crop_overlaps

    # -------------------------------------------------------------- splits
    def assign_splits(self, seed: int = 7,
                      fractions: Tuple[float, float, float] = (0.4, 0.3, 0.3)
                      ) -> Dict[str, List[str]]:
        """Assign whole **sources** to splits, never individual clips (R07).

        A clip whose source declared a split keeps it: the research material
        fixes its calibration/validation/test membership before any run, and a
        random reassignment would silently break that pre-registration.

        When splits are drawn here, the unit drawn is the *source recording*,
        not the scene.  Two crops of one photograph on opposite sides of a
        calibration/test boundary would leak the answer just as surely as two
        clips of one scene would, and the photograph is the level at which the
        sensor, the lighting and the terrain are shared.
        """
        declared: Dict[str, str] = {}
        for c in self.clips:
            if c.split:
                declared.setdefault(c.parent_scene_id, c.split)
        if declared and all(c.parent_scene_id in declared for c in self.clips):
            for c in self.clips:
                c.split = declared[c.parent_scene_id]
            groups: Dict[str, List[str]] = {"calibration": [], "validation": [],
                                            "test": []}
            for scene, split in sorted(declared.items()):
                groups.setdefault(split, []).append(scene)
            return groups
        by_source: Dict[str, List[str]] = {}
        for c in self.clips:
            by_source.setdefault(c.source_id, []).append(c.parent_scene_id)
        sources = sorted(by_source)
        rng = np.random.default_rng(seed)
        order = list(sources)
        rng.shuffle(order)
        n = len(order)
        n_cal = max(1, int(round(fractions[0] * n)))
        n_val = max(1, int(round(fractions[1] * n))) if n - n_cal > 1 else 0
        groups = {
            "calibration": order[:n_cal],
            "validation": order[n_cal : n_cal + n_val],
            "test": order[n_cal + n_val :],
        }
        if not groups["test"]:
            groups["test"] = groups["validation"] or groups["calibration"]
        if not groups["validation"]:
            groups["validation"] = groups["calibration"]
        where = {s: k for k, v in groups.items() for s in v}
        for c in self.clips:
            c.split = where.get(c.source_id, "test")
        # report scenes, not sources, because that is what callers ask for
        return {k: sorted({s for src in v for s in by_source.get(src, [])})
                for k, v in groups.items()}

    def clips_in(self, split: str) -> List[ClipRecord]:
        return [c for c in self.clips if c.split == split]

    def scenes_in(self, split: str) -> List[str]:
        return sorted({c.parent_scene_id for c in self.clips if c.split == split})

    # ------------------------------------------------------------ validate
    def validate(self) -> Dict[str, Any]:
        """Everything that could make a result wrong, reported explicitly."""
        problems: List[str] = []
        warnings: List[str] = []

        scene_splits: Dict[str, set] = {}
        for c in self.clips:
            scene_splits.setdefault(c.parent_scene_id, set()).add(c.split)
        for scene, splits in sorted(scene_splits.items()):
            if len(splits) > 1:
                problems.append(
                    f"scene {scene!r} appears in several splits: {sorted(splits)}")

        for d in self.duplicates:
            a = next((c for c in self.clips if c.clip_id == d.clip_a), None)
            b = next((c for c in self.clips if c.clip_id == d.clip_b), None)
            if a is None or b is None:
                continue
            if a.split and b.split and a.split != b.split:
                problems.append(
                    f"{d.kind} duplicate across splits: {d.clip_a} ({a.split}) and "
                    f"{d.clip_b} ({b.split}), hamming={d.distance}")
            elif not d.same_scene:
                warnings.append(
                    f"{d.kind} duplicate in different scenes but the same split: "
                    f"{d.clip_a} / {d.clip_b} (hamming={d.distance})")

        for c in self.clips:
            if not c.parent_scene_id:
                problems.append(f"clip {c.clip_id!r} has no parent_scene_id")
            if c.split and c.split not in SPLITS:
                problems.append(f"clip {c.clip_id!r} has unknown split {c.split!r}")
            if not c.content_sha256:
                problems.append(f"clip {c.clip_id!r} has no content hash")

        # A source that lands in two splits leaks just as badly as a scene that
        # does, and it is the level at which sensor and lighting are shared.
        source_splits: Dict[str, set] = {}
        for c in self.clips:
            source_splits.setdefault(c.source_id, set()).add(c.split)
        for src, splits in sorted(source_splits.items()):
            if len(splits) > 1:
                problems.append(
                    f"source {src!r} appears in several splits: {sorted(splits)}")

        counts = {s: len(self.scenes_in(s)) for s in SPLITS}
        provenances = sorted({c.provenance for c in self.clips})
        if provenances == ["synthetic"]:
            warnings.append("every clip is procedurally generated (SYNTHETIC DATA)")

        n_sources = len(source_splits)
        real = [c for c in self.clips if c.provenance != "synthetic"]
        if real:
            real_sources = len({c.source_id for c in real})
            if real_sources < len({c.parent_scene_id for c in real}):
                warnings.append(
                    f"{len({c.parent_scene_id for c in real})} natural scenes come "
                    f"from only {real_sources} independent source(s): a bootstrap "
                    "over scenes does not generalise to a new recording")
        overlapping = [o for o in self.crop_overlaps if o["overlaps"]]
        if overlapping:
            warnings.append(
                f"{len(overlapping)} pair(s) of clips are windows on the same "
                "image and do overlap geometrically; non-overlap is not claimed")

        return {
            "ok": not problems,
            "problems": problems,
            "warnings": warnings,
            "n_clips": len(self.clips),
            "n_scenes": len(scene_splits),
            "n_sources": n_sources,
            "sources_per_split": {
                s: len({c.source_id for c in self.clips if c.split == s})
                for s in SPLITS},
            "scenes_per_split": counts,
            "clips_per_split": {s: len(self.clips_in(s)) for s in SPLITS},
            "duplicates": [d.to_row() for d in self.duplicates],
            "crop_overlaps": self.crop_overlaps,
            "n_crop_pairs_overlapping": len(overlapping),
            "categories": sorted({c.category for c in self.clips}),
            "provenances": provenances,
            "independence_note": (
                "одиниця незалежності для узагальнення на НОВІ записи - "
                f"source_id ({n_sources} шт.), а не сцена "
                f"({len(scene_splits)} шт.)"),
        }

    # ------------------------------------------------------------------ io
    def save(self, directory: str, name: str = "dataset_manifest") -> Dict[str, str]:
        ensure_dir(directory)
        csv_path = os.path.join(directory, f"{name}.csv")
        json_path = os.path.join(directory, f"{name}.json")
        write_csv(csv_path, [c.to_row() for c in self.clips])
        write_json(json_path, {
            "clips": [c.to_row() for c in self.clips],
            "validation": self.validate(),
        })
        return {"csv": csv_path, "json": json_path}

    @staticmethod
    def load(path: str) -> "DatasetManifest":
        import csv
        import json as _json

        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as fh:
                data = _json.load(fh)
            rows = data.get("clips", [])
        else:
            with open(path, "r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
        clips = []
        for r in rows:
            clips.append(ClipRecord(
                clip_id=r["clip_id"], parent_scene_id=r["parent_scene_id"],
                category=r.get("category", "other"),
                provenance=r.get("provenance", "unknown"),
                source_kind=r.get("source_kind", ""),
                n_frames=int(r["n_frames"]), width=int(r["width"]),
                height=int(r["height"]), fps=float(r.get("fps", 8.333)),
                t_start_s=float(r.get("t_start_s", 0.0)),
                t_end_s=float(r.get("t_end_s", 0.0)),
                content_sha256=r.get("content_sha256", ""),
                first_frame_dhash=r.get("first_frame_dhash", ""),
                license=r.get("license", "unknown"), split=r.get("split", ""),
                notes=r.get("notes", ""),
                source_id=r.get("source_id", ""),
                derivation=r.get("derivation", ""), crop=r.get("crop", ""),
                source_sha256=r.get("source_sha256", ""),
                source_license=r.get("source_license", ""),
                source_credit=r.get("source_credit", "")))
        m = DatasetManifest(clips)
        m.detect_duplicates()
        m.measure_crop_overlap()
        return m

    def sources_in(self, split: str) -> List[str]:
        return sorted({c.source_id for c in self.clips if c.split == split})

    def source_of(self) -> Dict[str, str]:
        """``clip_id -> source_id``, for runners that tag their rows."""
        return {c.clip_id: c.source_id for c in self.clips}


def _crop_text(meta: Dict[str, Any]) -> str:
    """``x0,y0,x1,y1`` of the union of a window's start and end rectangles."""
    a, b = meta.get("crop_start"), meta.get("crop_end")
    if not a and not b:
        return ""
    rects = [tuple(int(v) for v in r) for r in (a, b) if r]
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[2] for r in rects)
    y1 = max(r[3] for r in rects)
    return f"{x0},{y0},{x1},{y1}"


def default_source_id(source: FrameSource, scene_of=None) -> str:
    """The independent origin a clip came from (R07).

    A source that declares ``source_id`` keeps it - that is how seventeen crops
    of one photograph all say they came from that one photograph.  Otherwise
    the scene is its own origin, which is correct for material that really was
    generated or recorded independently.
    """
    declared = (source.meta or {}).get("source_id")
    if declared:
        return str(declared)
    path = (source.meta or {}).get("path")
    if path:
        return "file:" + os.path.splitext(os.path.basename(str(path)))[0]
    return (scene_of or default_scene_id)(source)


def default_scene_id(source: FrameSource) -> str:
    """Group procedural material by its generating pattern.

    ``still_edges`` and ``seq_edges_slow`` share frame 0 exactly, so they are one
    scene.  File-based material is grouped by the file it came from.
    """
    declared = (source.meta or {}).get("scene_id")
    if declared:
        return str(declared)

    name = source.name
    if source.provenance == "synthetic":
        # longest first: "still_texture" also contains "text"
        for pat in sorted(_SYNTHETIC_CATEGORY, key=len, reverse=True):
            if pat in name:
                return f"synthetic:{pat}"
        return f"synthetic:{name}"
    path = str(source.meta.get("path", name))
    return "file:" + os.path.splitext(os.path.basename(path))[0]


def default_category(source: FrameSource) -> str:
    declared = (source.meta or {}).get("category")
    if declared:
        return str(declared)
    # longest first, for the same reason as default_scene_id
    for pat in sorted(_SYNTHETIC_CATEGORY, key=len, reverse=True):
        if pat in source.name:
            return _SYNTHETIC_CATEGORY[pat]
    return "other"


def build_manifest(sources: Sequence[FrameSource], seed: int = 7,
                   fps: float = 8.333) -> DatasetManifest:
    m = DatasetManifest.from_sources(sources, fps=fps)
    m.assign_splits(seed=seed)
    return m


def select(sources: Sequence[FrameSource], manifest: DatasetManifest,
           split: str) -> List[FrameSource]:
    """The frame sources belonging to one split, in manifest order."""
    wanted = {c.clip_id for c in manifest.clips_in(split)}
    return [s for s in sources if s.name in wanted]


__all__ = [
    "SPLITS", "CATEGORIES", "ClipRecord", "DuplicateFinding", "DatasetManifest",
    "build_manifest", "select", "default_scene_id", "default_source_id", "default_category",
    "dhash", "hamming", "normalised_hash",
]
