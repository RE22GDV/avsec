"""Dataset identity, duplicate control and scene-level splits (defect F10).

The unit of independence is a **scene**, not a clip and certainly not a frame.
Two clips cut from the same footage, a still taken from a sequence, a rescaled
copy and a motion variant are all the *same* scene: putting one of them in the
calibration split and another in the test split leaks the answer.

Every clip therefore carries ``parent_scene_id``, and splits are assigned to
scenes.  Exact and near duplicates are detected explicitly, so a leak is
reported rather than assumed absent.
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
            ))
        m = DatasetManifest(clips)
        m.detect_duplicates(sources)
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

    # -------------------------------------------------------------- splits
    def assign_splits(self, seed: int = 7,
                      fractions: Tuple[float, float, float] = (0.4, 0.3, 0.3)
                      ) -> Dict[str, List[str]]:
        """Assign whole **scenes** to splits, never individual clips.

        A clip whose source declared a split keeps it: the research material
        fixes its calibration/validation/test membership before any run, and a
        random reassignment would silently break that pre-registration.  Whole
        scenes still move together, declared or not.
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
        scenes = sorted({c.parent_scene_id for c in self.clips})
        rng = np.random.default_rng(seed)
        order = list(scenes)
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
            c.split = where.get(c.parent_scene_id, "test")
        return groups

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

        counts = {s: len(self.scenes_in(s)) for s in SPLITS}
        provenances = sorted({c.provenance for c in self.clips})
        if provenances == ["synthetic"]:
            warnings.append("every clip is procedurally generated (SYNTHETIC DATA)")

        return {
            "ok": not problems,
            "problems": problems,
            "warnings": warnings,
            "n_clips": len(self.clips),
            "n_scenes": len(scene_splits),
            "scenes_per_split": counts,
            "clips_per_split": {s: len(self.clips_in(s)) for s in SPLITS},
            "duplicates": [d.to_row() for d in self.duplicates],
            "categories": sorted({c.category for c in self.clips}),
            "provenances": provenances,
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
                notes=r.get("notes", "")))
        m = DatasetManifest(clips)
        m.detect_duplicates()
        return m


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
    "build_manifest", "select", "default_scene_id", "default_category",
    "dhash", "hamming", "normalised_hash",
]
