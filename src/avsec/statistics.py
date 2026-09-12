"""One aggregation path for every runner, and honest failure accounting.

Before this module the comparison, the sweep and the ablation each aggregated
differently, paired comparisons were truncated with ``min(len(a), len(b))``, and
a configuration that blew the budget could quietly leave the average.

Rules enforced here
-------------------
* **An observation is identified by its condition, not only by its frame.**
  The identity of a measurement is
  ``(method, condition, source, scene, clip, repetition, frame)`` where the
  *condition* is the channel point plus every declared experimental factor.
  Without the condition in the key, two channels writing the same frame index
  overwrite each other and a pooled estimate silently becomes "whichever row
  was read last" (defect R02).
* **The averaging order is fixed and declared**:
  ``frames -> repetitions -> clips -> scenes -> sources``.  Channels are *not*
  part of that chain: a combined number over channels exists only if somebody
  declares weights for it (:meth:`ResultTable.pooled_over_channels`), and no
  report in this repository does, because every research question here is asked
  per channel.
* **Permutation invariance**: the estimate does not depend on row order, and
  adding the same row twice does not create a new independent sample.
* **Pairing is by key, never by position**: a comparison keeps only the units
  where *both* methods have a result, and reports how many were dropped.
* **Failures are data**: a capacity failure, an exception, a lost sync and a
  missed deadline are different categories, and none of them may be removed in
  a way that improves a method's mean without being reported.
* **p-values come from a test on the paired differences** - never from the
  width of a bootstrap interval (defect R06).  Holm is then applied to a
  pre-declared family.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# A failed trial is still an observation; these are the categories we keep apart.
FAILURE_KINDS = ("ok", "capacity", "exception", "no_sync", "deadline_miss", "skipped")

#: The declared averaging chain.  Quoted into every table that uses it.
AGGREGATION_ORDER = ("frames", "repetitions", "clips", "scenes", "sources")

#: Metrics that are properties of a **scheduled display instant**, not of a
#: successful transmission, and are therefore averaged over every instant
#: including the failed ones (defect R05).  Restricting them to successes would
#: reward a configuration for not attempting the frames it cannot carry.
ALL_INSTANT_METRICS = frozenset({
    "availability", "psnr_displayed", "ssim_displayed", "operable",
})

#: Units the cluster bootstrap may resample over.  ``scene`` generalises to new
#: scenes *of the recordings already used*; ``source`` generalises to new
#: recordings, which is a strictly stronger and usually much smaller sample.
UNITS_OF_INDEPENDENCE = ("scene", "source")

Condition = Tuple[Tuple[str, str], ...]


@dataclass
class Observation:
    """One processed frame, or one recorded failure.

    ``channel`` and ``factors`` together make the *condition*.  They are part
    of the observation's identity, so rows from two channels never collapse
    onto one another (R02).

    ``source_id`` names the independent origin of the material - a flight, a
    recording, a photograph.  Several scenes cut from one photograph share it,
    which is what makes a bootstrap over scenes unable to generalise to new
    photographs (R07).  It defaults to the scene, which is correct when every
    scene is its own recording.
    """

    method: str
    scene: str
    clip: str
    repetition: int
    frame: int
    status: str = "ok"
    metrics: Dict[str, float] = field(default_factory=dict)
    detail: str = ""
    channel: str = ""
    config_id: str = ""
    factors: Condition = ()
    source_id: str = ""

    def __post_init__(self) -> None:
        self.factors = tuple(sorted((str(k), str(v)) for k, v in self.factors))
        if not self.source_id:
            self.source_id = self.scene

    @property
    def condition(self) -> Condition:
        """Everything that must match for two rows to be the same condition."""
        base = (("channel", self.channel), ("config_id", self.config_id))
        return tuple(sorted(base + tuple(self.factors)))

    @property
    def cell_key(self) -> Tuple[Any, ...]:
        """Full identity of one measurement (R02)."""
        return (self.condition, self.source_id, self.scene, self.clip,
                self.repetition, self.frame)

    def unit(self, unit_of_independence: str) -> str:
        return self.source_id if unit_of_independence == "source" else self.scene

    def to_row(self) -> Dict[str, Any]:
        row = {"method": self.method, "scene": self.scene, "clip": self.clip,
               "source_id": self.source_id, "channel": self.channel,
               "config_id": self.config_id,
               "factors": ";".join(f"{k}={v}" for k, v in self.factors),
               "repetition": self.repetition, "frame": self.frame,
               "status": self.status, "detail": self.detail}
        row.update(self.metrics)
        return row


class ResultTable:
    """Observations plus the single aggregation path they are read through."""

    def __init__(self, observations: Optional[Iterable[Observation]] = None,
                 unit_of_independence: str = "scene") -> None:
        if unit_of_independence not in UNITS_OF_INDEPENDENCE:
            raise ValueError(f"unit must be one of {UNITS_OF_INDEPENDENCE}")
        self.rows: List[Observation] = list(observations or [])
        self.unit_of_independence = unit_of_independence

    def add(self, obs: Observation) -> None:
        self.rows.append(obs)

    def extend(self, rows: Iterable[Observation]) -> None:
        self.rows.extend(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def view(self, rows: Iterable[Observation]) -> "ResultTable":
        return ResultTable(rows, self.unit_of_independence)

    @property
    def methods(self) -> List[str]:
        return sorted({r.method for r in self.rows})

    @property
    def scenes(self) -> List[str]:
        return sorted({r.scene for r in self.rows})

    @property
    def sources(self) -> List[str]:
        return sorted({r.source_id for r in self.rows})

    @property
    def channels(self) -> List[str]:
        return sorted({r.channel for r in self.rows if r.channel})

    @property
    def conditions(self) -> List[Condition]:
        return sorted({r.condition for r in self.rows})

    # ------------------------------------------------------------ aggregate
    def _cells(self, metric: str, method: str) -> Dict[Tuple[Any, ...], float]:
        """De-duplicated measurements keyed by full identity (R02).

        Quality metrics are read from successful instants only; availability
        and displayed quality are read from every scheduled instant (R05).
        """
        every_instant = metric in ALL_INSTANT_METRICS
        cell: Dict[Tuple[Any, ...], float] = {}
        for r in self.rows:
            if r.method != method:
                continue
            if not every_instant and r.status != "ok":
                continue
            if every_instant and r.frame < 0:
                continue                    # a job-level failure has no instant
            v = r.metrics.get(metric)
            if v is None or not np.isfinite(v):
                continue
            cell[r.cell_key] = float(v)
        return cell

    def per_scene(self, metric: str, method: str,
                  channel: Optional[str] = None) -> Dict[str, float]:
        """Frames -> repetitions -> clips -> one value per scene.

        Every step of the chain is a plain mean over the level below it, so a
        scene with many frames cannot dominate a scene with few.  ``channel``
        restricts the view; without it, a table holding several channels is
        averaged over all of them, which is only meaningful if the caller
        declared that it wanted that.
        """
        rows = self.rows if channel is None else [r for r in self.rows
                                                  if r.channel == channel]
        return self.view(rows)._per_key(metric, method, key="scene")

    def per_source(self, metric: str, method: str,
                   channel: Optional[str] = None) -> Dict[str, float]:
        """The same chain, carried one level further: scenes -> sources."""
        rows = self.rows if channel is None else [r for r in self.rows
                                                  if r.channel == channel]
        return self.view(rows)._per_key(metric, method, key="source")

    def per_unit(self, metric: str, method: str,
                 channel: Optional[str] = None) -> Dict[str, float]:
        """Per declared unit of independence (``scene`` or ``source``)."""
        if self.unit_of_independence == "source":
            return self.per_source(metric, method, channel)
        return self.per_scene(metric, method, channel)

    def _per_key(self, metric: str, method: str, key: str) -> Dict[str, float]:
        cell = self._cells(metric, method)
        by_row = {r.cell_key: r for r in self.rows if r.method == method}

        by_rep: Dict[Tuple[Any, ...], List[float]] = defaultdict(list)
        for ck, v in cell.items():
            r = by_row[ck]
            by_rep[(r.source_id, r.scene, r.clip, r.repetition)].append(v)
        by_clip: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
        for (src, scene, clip, _rep), vals in by_rep.items():
            by_clip[(src, scene, clip)].append(float(np.mean(vals)))
        by_scene: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for (src, scene, _clip), vals in by_clip.items():
            by_scene[(src, scene)].append(float(np.mean(vals)))
        scene_val = {k: float(np.mean(v)) for k, v in by_scene.items()}
        if key == "scene":
            out: Dict[str, List[float]] = defaultdict(list)
            for (_src, scene), v in scene_val.items():
                out[scene].append(v)
            return {s: float(np.mean(v)) for s, v in out.items()}
        by_source: Dict[str, List[float]] = defaultdict(list)
        for (src, _scene), v in scene_val.items():
            by_source[src].append(v)
        return {s: float(np.mean(v)) for s, v in by_source.items()}

    def pooled_over_channels(self, metric: str, method: str,
                             weights: Dict[str, float]) -> Dict[str, float]:
        """A combined indicator over channels, with **declared** weights.

        Channels are different operating conditions, not repeated draws of the
        same thing, so there is no default way to merge them.  This method
        exists for a caller that has a reason to combine them and can name the
        weights; it refuses to invent them.  Nothing in this repository calls it
        for a published claim - the research questions are all per channel.
        """
        missing = [c for c in self.channels if c not in weights]
        if missing:
            raise ValueError(
                "pooling over channels needs a declared weight for each of "
                f"{missing}; there is no defensible default")
        total = float(sum(weights[c] for c in self.channels))
        if total <= 0:
            raise ValueError("channel weights must sum to something positive")
        acc: Dict[str, float] = defaultdict(float)
        seen: Dict[str, float] = defaultdict(float)
        for ch in self.channels:
            w = weights[ch] / total
            for unit, v in self.per_unit(metric, method, channel=ch).items():
                acc[unit] += w * v
                seen[unit] += w
        return {u: acc[u] / seen[u] for u in acc if seen[u] > 0}

    def failures(self, method: Optional[str] = None) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for r in self.rows:
            if method is not None and r.method != method:
                continue
            out[r.status] += 1
        return dict(out)

    def summary(self, metric: str, n_boot: int = 10000, seed: int = 20240909
                ) -> List[Dict[str, Any]]:
        """Per-method mean over units with a cluster bootstrap interval."""
        out: List[Dict[str, Any]] = []
        for m in self.methods:
            vals = self.per_unit(metric, m)
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
                "unit_of_independence": self.unit_of_independence,
                "n_units": int(arr.size),
                "n_scenes": len(self.per_scene(metric, m)),
                "n_sources": len(self.per_source(metric, m)),
                "n_observations": sum(1 for r in self.rows
                                      if r.method == m and r.status == "ok"),
                "failures": {k: v for k, v in fails.items() if k != "ok"},
                "n_failed": sum(v for k, v in fails.items() if k != "ok"),
            })
        return out

    # -------------------------------------------------------------- paired
    def paired(self, a: str, b: str, metric: str, n_boot: int = 10000,
               seed: int = 20240909, equivalence_margin: float = 0.0
               ) -> Dict[str, Any]:
        """Paired difference ``a - b`` over the units both methods have.

        The interval is a paired cluster bootstrap; the p-value comes from a
        **test on those same differences** (R06), not from the interval's
        width.  Both a bootstrap test and the distribution-free Wilcoxon
        signed-rank test are reported, so a reader can see that the decision
        does not hinge on one of them.
        """
        va, vb = self.per_unit(metric, a), self.per_unit(metric, b)
        shared = sorted(set(va) & set(vb))
        only_a = sorted(set(va) - set(vb))
        only_b = sorted(set(vb) - set(va))
        d = np.asarray([va[s] - vb[s] for s in shared], dtype=float)
        lo, hi = _bootstrap_ci(d, n_boot, seed)
        mean = float(d.mean()) if d.size else float("nan")
        sig = bool(d.size >= 2 and np.isfinite(lo) and np.isfinite(hi)
                   and (lo > 0 or hi < 0))
        out = {
            "a": a, "b": b, "metric": metric, "mean": mean, "lo": lo, "hi": hi,
            "unit_of_independence": self.unit_of_independence,
            "n_units": int(d.size), "n_scenes": int(d.size),
            "units": shared,
            "diffs": [float(x) for x in d],
            "significant_at_95": sig,
            "units_only_in_a": only_a, "units_only_in_b": only_b,
            "scenes_only_in_a": only_a, "scenes_only_in_b": only_b,
            "n_units_dropped": len(only_a) + len(only_b),
            "n_scenes_dropped": len(only_a) + len(only_b),
            "method": (f"paired cluster bootstrap over {self.unit_of_independence}s"),
            "n_boot": int(n_boot),
        }
        out.update(difference_tests(d, n_boot, seed))
        if equivalence_margin > 0:
            out.update(equivalence(d, equivalence_margin, n_boot, seed))
        return out

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
                  primary: Optional[Tuple[str, str]] = None,
                  family: Optional[Sequence[Tuple[str, str]]] = None,
                  equivalence_margin: float = 0.0) -> List[Dict[str, Any]]:
        """Pairwise comparisons, Holm-adjusted over a **pre-declared** family.

        ``family`` names the comparisons the adjustment is for.  Everything
        outside it is still computed - an exploratory number is useful - but it
        is marked ``in_family=False`` and takes no part in the adjustment, so
        adding an exploratory pair can never change a pre-registered decision
        (R06).  ``primary`` names the one comparison also reported unadjusted.
        """
        ms = list(methods or self.methods)
        out: List[Dict[str, Any]] = []
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                out.append(self.paired(ms[i], ms[j], metric, n_boot, seed,
                                       equivalence_margin))
        fam = {frozenset(p) for p in family} if family else None
        for r in out:
            r["in_family"] = bool(fam is None or frozenset({r["a"], r["b"]}) in fam)
            r["is_primary"] = bool(primary and {r["a"], r["b"]} == set(primary))
        holm_adjust([r for r in out if r["in_family"]])
        for r in out:
            if not r["in_family"]:
                r["p_holm"] = float("nan")
                r["significant_holm"] = False
                r["holm_note"] = "поза наперед оголошеною сім'єю; не коригується"
        return out

    # ------------------------------------------------------------------ io
    def to_rows(self) -> List[Dict[str, Any]]:
        return [r.to_row() for r in self.rows]

    @staticmethod
    def from_rows(rows: Iterable[Dict[str, Any]],
                  metric_keys: Optional[Sequence[str]] = None,
                  unit_of_independence: str = "scene") -> "ResultTable":
        table = ResultTable(unit_of_independence=unit_of_independence)
        fixed = {"method", "scene", "clip", "repetition", "frame", "status",
                 "detail", "channel", "config_id", "factors", "source_id"}
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
                detail=str(r.get("detail", "")),
                channel=str(r.get("channel", "")),
                config_id=str(r.get("config_id", "")),
                factors=_parse_factors(r.get("factors")),
                source_id=str(r.get("source_id", "") or r["scene"])))
        return table


def _parse_factors(raw: Any) -> Condition:
    if not raw:
        return ()
    if isinstance(raw, dict):
        return tuple(sorted((str(k), str(v)) for k, v in raw.items()))
    out = []
    for part in str(raw).split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out.append((k.strip(), v.strip()))
    return tuple(sorted(out))


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int,
                  alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap over the given (already unit-level) values."""
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


def wilcoxon_p_value(diffs: np.ndarray) -> float:
    """Distribution-free two-sided signed-rank test on the paired differences."""
    arr = np.asarray(diffs, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr != 0.0]
    if arr.size < 2:
        return 1.0
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(arr, alternative="two-sided",
                              zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def difference_tests(diffs: np.ndarray, n_boot: int = 10000,
                     seed: int = 20240909) -> Dict[str, Any]:
    """The declared tests on a set of paired differences (R06).

    The *primary* test is the two-sided bootstrap test of "mean difference is
    zero", resampling the same independent units the interval resamples.  The
    Wilcoxon signed-rank test is reported next to it as a cross-check that does
    not assume the bootstrap's approximation.
    """
    arr = np.asarray(diffs, dtype=float)
    arr = arr[np.isfinite(arr)]
    p_boot = bootstrap_p_value(arr, n_boot, seed) if arr.size >= 2 else 1.0
    return {
        "p_raw": float(p_boot),
        "p_bootstrap": float(p_boot),
        "p_wilcoxon": wilcoxon_p_value(arr),
        "p_source": ("двобічний bootstrap-тест на парних різницях "
                     f"({int(n_boot)} перевибірок, seed {int(seed)})"),
        "n_diffs": int(arr.size),
    }


def equivalence(diffs: np.ndarray, margin: float, n_boot: int = 10000,
                seed: int = 20240909) -> Dict[str, Any]:
    """TOST: is the difference demonstrably *inside* +-``margin``?

    Absence of significance is not evidence of equality, so when a comparison
    comes out flat this says whether equality was actually shown.
    """
    arr = np.asarray(diffs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2 or margin <= 0:
        return {"equivalence_margin": float(margin), "equivalent": False,
                "equivalence_verdict": "не перевірялось"}
    lo, hi = _bootstrap_ci(arr, n_boot, seed, alpha=0.10)   # 90% for TOST
    ok = bool(lo > -margin and hi < margin)
    return {
        "equivalence_margin": float(margin),
        "equivalence_lo90": float(lo), "equivalence_hi90": float(hi),
        "equivalent": ok,
        "equivalence_verdict": (
            f"еквівалентність у межах ±{margin:g} показана (90% CI "
            f"[{lo:+.2f}; {hi:+.2f}])" if ok else
            f"еквівалентність у межах ±{margin:g} НЕ показана (90% CI "
            f"[{lo:+.2f}; {hi:+.2f}]): різниця просто не встановлена"),
    }


def holm_adjust(comparisons: List[Dict[str, Any]], alpha: float = 0.05,
                n_boot: int = 10000, seed: int = 20240909) -> List[Dict[str, Any]]:
    """Holm step-down adjustment across a pre-declared family.

    Each comparison must carry either a ``p_raw`` produced by
    :func:`difference_tests` or the ``diffs`` to compute one from.  A p-value is
    never reconstructed from an interval's width (R06): a comparison that
    supplies neither is marked unavailable and cannot be declared significant.
    """
    for c in comparisons:
        if "p_raw" not in c or not np.isfinite(c.get("p_raw", np.nan)):
            diffs = c.get("diffs")
            if diffs is not None and len(diffs) >= 2:
                c.update(difference_tests(np.asarray(diffs, dtype=float),
                                          n_boot, seed))
            else:
                c["p_raw"] = float("nan")
                c["p_source"] = ("парні різниці не збережені: p-value не "
                                 "обчислюється")

    testable = [c for c in comparisons if np.isfinite(c.get("p_raw", np.nan))]
    order = sorted(range(len(testable)), key=lambda i: testable[i]["p_raw"])
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * testable[i]["p_raw"])
        running = max(running, adj)          # Holm is monotone
        testable[i]["p_holm"] = running
        testable[i]["significant_holm"] = bool(running < alpha)
    for c in comparisons:
        if not np.isfinite(c.get("p_raw", np.nan)):
            c["p_holm"] = float("nan")
            c["significant_holm"] = False
    return comparisons


def aggregate_frames(rows: Sequence[Dict[str, Any]], metric: str,
                     scene_key: str = "scene") -> Dict[str, float]:
    """Convenience wrapper for callers that already have plain dict rows."""
    return ResultTable.from_rows(rows).per_scene(metric, rows[0]["method"]) if rows else {}


__all__ = [
    "FAILURE_KINDS", "AGGREGATION_ORDER", "UNITS_OF_INDEPENDENCE",
    "Observation", "ResultTable", "holm_adjust", "bootstrap_p_value",
    "wilcoxon_p_value", "difference_tests", "equivalence", "aggregate_frames",
]
