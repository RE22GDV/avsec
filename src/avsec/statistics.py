"""One aggregation path for every runner, and honest failure accounting (F14).

Before this module, the comparison, the sweep and the ablation each aggregated
differently, paired comparisons were truncated with ``min(len(a), len(b))``, and
a configuration that blew the budget could quietly leave the average.

Rules enforced here
-------------------
* **Aggregation order is fixed**: frames inside a clip -> repetitions inside a
  clip -> clips inside a scene.  The unit of independence is the *scene*.
* **Permutation invariance**: the estimate does not depend on row order, and
  adding the same frame twice does not create a new independent scene.
* **Pairing is by key, never by position**: a comparison keeps only the scenes
  where *both* methods have a result, and reports how many were dropped and why.
* **Failures are data**: a capacity failure, an exception, a lost sync and a
  missed deadline are different categories, and none of them may be removed in
  a way that improves a method's mean without being reported.
* **Multiplicity**: when several comparisons drive decisions, Holm adjustment is
  applied and the adjusted decision is reported next to the raw interval.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# A failed trial is still an observation; these are the categories we keep apart.
FAILURE_KINDS = ("ok", "capacity", "exception", "no_sync", "deadline_miss", "skipped")


@dataclass
class Observation:
    """One processed frame, or one recorded failure."""

    method: str
    scene: str
    clip: str
    repetition: int
    frame: int
    status: str = "ok"
    metrics: Dict[str, float] = field(default_factory=dict)
    detail: str = ""

    def to_row(self) -> Dict[str, Any]:
        row = {"method": self.method, "scene": self.scene, "clip": self.clip,
               "repetition": self.repetition, "frame": self.frame,
               "status": self.status, "detail": self.detail}
        row.update(self.metrics)
        return row


class ResultTable:
    """Observations plus the single aggregation path they are read through."""

    def __init__(self, observations: Optional[Iterable[Observation]] = None) -> None:
        self.rows: List[Observation] = list(observations or [])

    def add(self, obs: Observation) -> None:
        self.rows.append(obs)

    def extend(self, rows: Iterable[Observation]) -> None:
        self.rows.extend(rows)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def methods(self) -> List[str]:
        return sorted({r.method for r in self.rows})

    @property
    def scenes(self) -> List[str]:
        return sorted({r.scene for r in self.rows})

    # ------------------------------------------------------------ aggregate
    def per_scene(self, metric: str, method: str) -> Dict[str, float]:
        """Frames -> repetitions -> clips -> one value per scene.

        Duplicated identical frames collapse: a value is stored per
        ``(scene, clip, repetition, frame)`` key, so re-adding the same frame
        cannot inflate the sample.
        """
        cell: Dict[Tuple[str, str, int, int], float] = {}
        for r in self.rows:
            if r.method != method or r.status != "ok":
                continue
            v = r.metrics.get(metric)
            if v is None or not np.isfinite(v):
                continue
            cell[(r.scene, r.clip, r.repetition, r.frame)] = float(v)

        by_rep: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
        for (scene, clip, rep, _frame), v in cell.items():
            by_rep[(scene, clip, rep)].append(v)
        by_clip: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for (scene, clip, _rep), vals in by_rep.items():
            by_clip[(scene, clip)].append(float(np.mean(vals)))
        by_scene: Dict[str, List[float]] = defaultdict(list)
        for (scene, _clip), vals in by_clip.items():
            by_scene[scene].append(float(np.mean(vals)))
        return {scene: float(np.mean(vals)) for scene, vals in by_scene.items()}

    def failures(self, method: Optional[str] = None) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for r in self.rows:
            if method is not None and r.method != method:
                continue
            out[r.status] += 1
        return dict(out)

    def summary(self, metric: str, n_boot: int = 10000, seed: int = 20240909
                ) -> List[Dict[str, Any]]:
        """Per-method mean over scenes with a cluster bootstrap interval."""
        out: List[Dict[str, Any]] = []
        for m in self.methods:
            vals = self.per_scene(metric, m)
            arr = np.asarray(list(vals.values()), dtype=float)
            lo, hi = _bootstrap_ci(arr, n_boot, seed)
            fails = self.failures(m)
            # Frames reproduced exactly have an infinite PSNR.  They are counted,
            # never replaced by a finite stand-in and never averaged in (F18).
            n_exact = sum(1 for r in self.rows
                          if r.method == m and r.status == "ok"
                          and (r.metrics.get("bit_exact", 0) >= 1
                               or not np.isfinite(r.metrics.get(metric, 0.0))))
            out.append({
                "method": m, "metric": metric,
                "n_bit_exact_excluded": n_exact,
                "mean": float(arr.mean()) if arr.size else float("nan"),
                "lo": lo, "hi": hi,
                "n_scenes": int(arr.size),
                "n_observations": sum(1 for r in self.rows
                                      if r.method == m and r.status == "ok"),
                "failures": {k: v for k, v in fails.items() if k != "ok"},
                "n_failed": sum(v for k, v in fails.items() if k != "ok"),
            })
        return out

    # -------------------------------------------------------------- paired
    def paired(self, a: str, b: str, metric: str, n_boot: int = 10000,
               seed: int = 20240909) -> Dict[str, Any]:
        """Paired cluster bootstrap of ``a - b`` over shared scenes.

        Scenes present for only one of the two methods are **excluded and
        counted**, never silently truncated by index.
        """
        va, vb = self.per_scene(metric, a), self.per_scene(metric, b)
        shared = sorted(set(va) & set(vb))
        only_a = sorted(set(va) - set(vb))
        only_b = sorted(set(vb) - set(va))
        d = np.asarray([va[s] - vb[s] for s in shared], dtype=float)
        lo, hi = _bootstrap_ci(d, n_boot, seed)
        mean = float(d.mean()) if d.size else float("nan")
        sig = bool(d.size >= 2 and np.isfinite(lo) and np.isfinite(hi)
                   and (lo > 0 or hi < 0))
        return {
            "a": a, "b": b, "metric": metric, "mean": mean, "lo": lo, "hi": hi,
            "n_scenes": int(d.size), "significant_at_95": sig,
            "scenes_only_in_a": only_a, "scenes_only_in_b": only_b,
            "n_scenes_dropped": len(only_a) + len(only_b),
            "method": "paired cluster bootstrap over parent scenes",
            "n_boot": int(n_boot),
        }

    def channel_only(self, method: str, metric: str, scene: str,
                     n_boot: int = 10000, seed: int = 20240909) -> Dict[str, Any]:
        """Uncertainty over channel realisations for **one fixed scene**.

        This answers a different question from :meth:`paired` - it does not
        generalise to new scenes - so it is a separate call with its own label.
        """
        vals = [r.metrics.get(metric) for r in self.rows
                if r.method == method and r.scene == scene and r.status == "ok"
                and r.metrics.get(metric) is not None]
        arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)],
                         dtype=float)
        lo, hi = _bootstrap_ci(arr, n_boot, seed)
        return {"method": method, "scene": scene, "metric": metric,
                "mean": float(arr.mean()) if arr.size else float("nan"),
                "lo": lo, "hi": hi, "n": int(arr.size),
                "generalises_to": "this scene only, not to new scenes"}

    def all_pairs(self, metric: str, methods: Optional[Sequence[str]] = None,
                  n_boot: int = 10000, seed: int = 20240909,
                  primary: Optional[Tuple[str, str]] = None) -> List[Dict[str, Any]]:
        """Every pairwise comparison, with Holm adjustment over the family.

        ``primary`` names the one pre-registered comparison; it is reported
        unadjusted as well, because adjusting a single primary hypothesis for a
        family it does not belong to would be wrong in the other direction.
        """
        ms = list(methods or self.methods)
        out: List[Dict[str, Any]] = []
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                out.append(self.paired(ms[i], ms[j], metric, n_boot, seed))
        holm_adjust(out)
        for r in out:
            r["is_primary"] = bool(primary and {r["a"], r["b"]} == set(primary))
        return out

    # ------------------------------------------------------------------ io
    def to_rows(self) -> List[Dict[str, Any]]:
        return [r.to_row() for r in self.rows]

    @staticmethod
    def from_rows(rows: Iterable[Dict[str, Any]],
                  metric_keys: Optional[Sequence[str]] = None) -> "ResultTable":
        table = ResultTable()
        fixed = {"method", "scene", "clip", "repetition", "frame", "status", "detail"}
        for r in rows:
            keys = metric_keys or [k for k in r if k not in fixed]
            metrics = {}
            for k in keys:
                try:
                    metrics[k] = float(r[k])
                except (TypeError, ValueError, KeyError):
                    continue
            table.add(Observation(
                method=str(r["method"]), scene=str(r["scene"]),
                clip=str(r.get("clip", r["scene"])),
                repetition=int(r.get("repetition", 0)),
                frame=int(r.get("frame", 0)),
                status=str(r.get("status", "ok")), metrics=metrics,
                detail=str(r.get("detail", ""))))
        return table


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int,
                  alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap over the given (already scene-level) values."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    means = arr[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def bootstrap_p_value(diffs: np.ndarray, n_boot: int, seed: int) -> float:
    """Two-sided bootstrap p-value for "the mean difference is zero"."""
    arr = np.asarray(diffs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    centred = arr - arr.mean()
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    null = centred[idx].mean(axis=1)
    obs = abs(arr.mean())
    return float((np.sum(np.abs(null) >= obs) + 1) / (n_boot + 1))


def holm_adjust(comparisons: List[Dict[str, Any]], alpha: float = 0.05,
                n_boot: int = 2000, seed: int = 20240909) -> List[Dict[str, Any]]:
    """Holm step-down adjustment across a family of comparisons.

    Adds ``p_raw``, ``p_holm`` and ``significant_holm`` in place.  The raw
    interval is left untouched, so an effect size can still be read off.
    """
    for c in comparisons:
        n = int(c.get("n_scenes", 0))
        if n >= 2 and np.isfinite(c.get("mean", np.nan)):
            spread = (c["hi"] - c["lo"]) / 3.92 if np.isfinite(c.get("hi", np.nan)) else 0.0
            if spread > 0:
                z = abs(c["mean"]) / spread
                from math import erfc, sqrt

                c["p_raw"] = float(erfc(z / sqrt(2)))
            else:
                c["p_raw"] = 0.0 if abs(c["mean"]) > 0 else 1.0
        else:
            c["p_raw"] = 1.0

    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["p_raw"])
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * comparisons[i]["p_raw"])
        running = max(running, adj)          # Holm is monotone
        comparisons[i]["p_holm"] = running
        comparisons[i]["significant_holm"] = bool(running < alpha)
    return comparisons


def aggregate_frames(rows: Sequence[Dict[str, Any]], metric: str,
                     scene_key: str = "scene") -> Dict[str, float]:
    """Convenience wrapper for callers that already have plain dict rows."""
    return ResultTable.from_rows(rows).per_scene(metric, rows[0]["method"]) if rows else {}


__all__ = [
    "FAILURE_KINDS", "Observation", "ResultTable", "holm_adjust",
    "bootstrap_p_value", "aggregate_frames",
]
