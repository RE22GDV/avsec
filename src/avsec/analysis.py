"""Turn a matrix run into the statistical tables the report and figures read.

``avsec analyze`` is the only place where effects are estimated.  It reads the
observations a run persisted, applies the fixed aggregation path from
:mod:`avsec.statistics` (frames -> repetitions -> clips -> scenes) and writes:

* ``summary.csv``          - per method and channel, mean over scenes with a CI;
* ``paired_effects.csv``   - every pairwise comparison, effect, CI, Holm status;
* ``operability.csv``      - fraction of displays meeting the declared criterion;
* ``failures_summary.csv`` - failures by category, per method and channel;
* ``analysis.json``        - the statistical plan actually used, and the primary
  hypothesis, so that a number can never be quoted without its plan.

No new experiment is run here.  If a comparison has no data, it is reported as
missing rather than filled in from somewhere else.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from avsec.statistics import Observation, ResultTable, holm_adjust
from avsec.utils import ensure_dir, write_csv, write_json

#: The operating criterion of E04/E06.  It is an engineering requirement of
#: this study, not a universal standard for drone video, and it is stated
#: wherever an operability number appears.
DEFAULT_CRITERION = {
    "coverage_verified_min": 0.90,
    "psnr_full_min": 20.0,
    "note": ("робочий критерій цього дослідження: поточне автентифіковане "
             "покриття >= 0.90 і PSNR усього показаного кадру >= 20 дБ; "
             "це інженерна вимога, а не універсальний стандарт"),
}

#: The one pre-registered comparison.  Everything else is exploratory and is
#: reported with a Holm-adjusted decision alongside the raw interval.
PRIMARY = ("P", "B4")

#: Methods that display a picture nobody authenticated.  Their "coverage" is
#: not comparable with an authenticated method's coverage and is labelled.
UNAUTHENTICATED = ("B0a", "B0a-R", "B0d", "B0d-W", "B1", "B2")


@dataclass
class AnalysisPlan:
    """The statistical plan, written next to the results it produced."""

    metric: str = "psnr_full"
    secondary: Tuple[str, ...] = ("ssim_full", "coverage", "psnr_verified")
    primary: Tuple[str, str] = PRIMARY
    n_boot: int = 10000
    seed: int = 20240909
    criterion: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_CRITERION))
    unit_of_independence: str = "parent scene"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric, "secondary": list(self.secondary),
            "primary_comparison": list(self.primary), "n_boot": self.n_boot,
            "seed": self.seed, "criterion": self.criterion,
            "unit_of_independence": self.unit_of_independence,
            "aggregation": "frames -> repetitions -> clips -> scenes, then a "
                           "paired cluster bootstrap over scenes",
            "multiplicity": "Holm step-down over the family of pairwise "
                            "comparisons; the primary comparison is also "
                            "reported unadjusted",
        }

    @staticmethod
    def load(path: Optional[str]) -> "AnalysisPlan":
        if not path or not os.path.exists(path):
            return AnalysisPlan()
        import yaml

        with open(path, encoding="utf-8") as fh:
            d = yaml.safe_load(fh) or {}
        plan = AnalysisPlan()
        plan.metric = str(d.get("metric", plan.metric))
        plan.secondary = tuple(d.get("secondary", plan.secondary))
        pr = d.get("primary") or d.get("primary_comparison")
        if pr:
            plan.primary = (str(pr[0]), str(pr[1]))
        plan.n_boot = int(d.get("n_boot", plan.n_boot))
        plan.seed = int(d.get("seed", plan.seed))
        if d.get("criterion"):
            plan.criterion.update(d["criterion"])
        return plan


# ------------------------------------------------------------------- loading
def load_observations(run_dir: str) -> Tuple[ResultTable, List[Dict[str, Any]]]:
    """Read a run's frame rows into the shared table, keeping the raw rows."""
    rows = _read_rows(run_dir)
    table = ResultTable()
    for r in rows:
        metrics: Dict[str, float] = {}
        for k, v in r.items():
            if k in ("method", "scene", "clip", "channel", "status", "detail",
                     "job_id", "trace_id", "provenance", "channel_axis"):
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(f):
                metrics[k] = f
        table.add(Observation(
            method=str(r.get("method", "")), scene=str(r.get("scene", "")),
            clip=str(r.get("clip", "")), repetition=int(float(r.get("repetition", 0) or 0)),
            frame=int(float(r.get("frame_id", 0) or 0)),
            status=str(r.get("status", "ok")), metrics=metrics,
            detail=str(r.get("channel", ""))))
    return table, rows


def _read_rows(run_dir: str) -> List[Dict[str, Any]]:
    import csv

    path = os.path.join(run_dir, "frames.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run `avsec matrix` before `avsec analyze`")
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    fail = os.path.join(run_dir, "failures.csv")
    if os.path.exists(fail):
        with open(fail, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                r.setdefault("frame_id", "-1")
                rows.append(r)
    return rows


def subset(table: ResultTable, channel: Optional[str]) -> ResultTable:
    """A view of one channel point.  ``detail`` carries the channel label."""
    if channel is None:
        return table
    return ResultTable([r for r in table.rows if r.detail == channel])


def channels_in(table: ResultTable) -> List[str]:
    return sorted({r.detail for r in table.rows if r.detail})


# ----------------------------------------------------------------- analysis
def analyse(run_dir: str, plan: Optional[AnalysisPlan] = None,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Produce every statistical table from one run directory."""
    plan = plan or AnalysisPlan()
    out_dir = ensure_dir(output_dir or run_dir)
    table, rows = load_observations(run_dir)
    channels = channels_in(table)
    metrics = [plan.metric] + [m for m in plan.secondary if m != plan.metric]

    # ---- per method x channel summary ----------------------------------
    summary: List[Dict[str, Any]] = []
    for ch in channels:
        sub = subset(table, ch)
        for metric in metrics:
            for rec in sub.summary(metric, plan.n_boot, plan.seed):
                rec["channel"] = ch
                rec["authenticated"] = rec["method"] not in UNAUTHENTICATED
                if metric.startswith("coverage") and not rec["authenticated"]:
                    rec["coverage_meaning"] = ("картинка показана, але НЕ "
                                               "автентифікована")
                rec["failures"] = json.dumps(rec.get("failures", {}), ensure_ascii=False)
                summary.append(rec)
    write_csv(os.path.join(out_dir, "summary.csv"), summary)

    # ---- paired effects, per channel and pooled -------------------------
    effects: List[Dict[str, Any]] = []
    for ch in channels:
        sub = subset(table, ch)
        for rec in sub.all_pairs(plan.metric, n_boot=plan.n_boot, seed=plan.seed,
                                 primary=plan.primary):
            rec["channel"] = ch
            rec["family"] = f"pairwise@{ch}"
            rec["scenes_only_in_a"] = ";".join(rec.get("scenes_only_in_a", []))
            rec["scenes_only_in_b"] = ";".join(rec.get("scenes_only_in_b", []))
            effects.append(rec)
    pooled = table.all_pairs(plan.metric, n_boot=plan.n_boot, seed=plan.seed,
                             primary=plan.primary)
    for rec in pooled:
        rec["channel"] = "ALL"
        rec["family"] = "pairwise@pooled"
        rec["scenes_only_in_a"] = ";".join(rec.get("scenes_only_in_a", []))
        rec["scenes_only_in_b"] = ";".join(rec.get("scenes_only_in_b", []))
        rec["caveat"] = ("об'єднання каналів усереднює різні умови; "
                         "основний висновок читати за окремими каналами")
        effects.append(rec)
    write_csv(os.path.join(out_dir, "paired_effects.csv"), effects)

    # ---- operability ----------------------------------------------------
    oper = operability(table, plan)
    write_csv(os.path.join(out_dir, "operability.csv"), oper)

    # ---- failures -------------------------------------------------------
    fails = failure_table(table)
    write_csv(os.path.join(out_dir, "failures_summary.csv"), fails)

    # Reorient every primary row to the declared direction before it is read.
    primary_rows = [orient(e, plan.primary[0]) for e in effects if e.get("is_primary")]
    result = {
        "kind": "analysis",
        "run_dir": os.path.abspath(run_dir),
        "plan": plan.to_dict(),
        "channels": channels,
        "methods": table.methods,
        "n_scenes": len(table.scenes),
        "n_observations": len(table.rows),
        "primary": primary_rows,
        "primary_verdict": _verdict(primary_rows, plan),
        "n_effects": len(effects),
        "operability_criterion": plan.criterion,
    }
    write_json(os.path.join(out_dir, "analysis.json"), result)
    return result


def operability(table: ResultTable, plan: AnalysisPlan) -> List[Dict[str, Any]]:
    """Fraction of displayed frames meeting the declared operating criterion.

    Computed per scene first, then averaged over scenes, so a scene with many
    frames cannot dominate.  Unauthenticated methods are marked, because for
    them "coverage" means "a picture was shown", not "verified data".
    """
    cov_min = float(plan.criterion["coverage_verified_min"])
    psnr_min = float(plan.criterion["psnr_full_min"])
    out: List[Dict[str, Any]] = []
    channels = channels_in(table) or [""]
    for ch in channels:
        sub = subset(table, ch or None)
        for method in sub.methods:
            per_scene: Dict[str, List[float]] = {}
            for r in sub.rows:
                if r.method != method or r.status != "ok":
                    continue
                cov = r.metrics.get("coverage", float("nan"))
                psnr = r.metrics.get("psnr_full", float("nan"))
                ok = bool(np.isfinite(cov) and np.isfinite(psnr)
                          and cov >= cov_min and psnr >= psnr_min)
                per_scene.setdefault(r.scene, []).append(1.0 if ok else 0.0)
            if not per_scene:
                continue
            vals = np.asarray([float(np.mean(v)) for v in per_scene.values()])
            out.append({
                "channel": ch, "method": method,
                "operable_fraction": float(vals.mean()),
                "n_scenes": int(vals.size),
                "sd_over_scenes": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "authenticated": method not in UNAUTHENTICATED,
                "criterion": (f"coverage>={cov_min} and psnr_full>={psnr_min} "
                              "(engineering requirement of this study)"),
            })
    return out


def failure_table(table: ResultTable) -> List[Dict[str, Any]]:
    """Failures kept apart by category, per method and channel."""
    out: List[Dict[str, Any]] = []
    channels = channels_in(table) or [""]
    for ch in channels:
        sub = subset(table, ch or None)
        for method in sub.methods:
            counts = sub.failures(method)
            total = sum(counts.values())
            row: Dict[str, Any] = {"channel": ch, "method": method,
                                   "n_observations": total}
            for kind, n in sorted(counts.items()):
                row[f"n_{kind}"] = n
            row["n_failed"] = sum(v for k, v in counts.items() if k != "ok")
            out.append(row)
    return out


def orient(rec: Dict[str, Any], first: str) -> Dict[str, Any]:
    """Return the comparison written as ``first - other``.

    :meth:`ResultTable.all_pairs` emits each pair once, in the method order it
    happened to iterate, so a row may be ``B4 - P`` when the pre-registered
    hypothesis is about ``P - B4``.  Reading the sign off such a row without
    reorienting it inverts the conclusion, so every consumer goes through here.
    """
    if rec.get("a") == first:
        return dict(rec)
    out = dict(rec)
    out["a"], out["b"] = rec.get("b"), rec.get("a")
    for lo_key, hi_key in (("lo", "hi"),):
        lo, hi = rec.get(lo_key), rec.get(hi_key)
        if lo is not None and hi is not None:
            out[lo_key], out[hi_key] = -float(hi), -float(lo)
    if rec.get("mean") is not None:
        out["mean"] = -float(rec["mean"])
    out["scenes_only_in_a"] = rec.get("scenes_only_in_b", "")
    out["scenes_only_in_b"] = rec.get("scenes_only_in_a", "")
    out["reoriented"] = True
    return out


def _verdict(primary_rows: Sequence[Dict[str, Any]], plan: AnalysisPlan) -> Dict[str, Any]:
    """State what the pre-registered comparison shows, without overclaiming."""
    per_channel = {r["channel"]: orient(r, plan.primary[0]) for r in primary_rows}
    wins = [c for c, r in per_channel.items()
            if c != "ALL" and r.get("significant_at_95") and r.get("mean", 0) > 0]
    losses = [c for c, r in per_channel.items()
              if c != "ALL" and r.get("significant_at_95") and r.get("mean", 0) < 0]
    flat = [c for c, r in per_channel.items()
            if c != "ALL" and not r.get("significant_at_95")]
    return {
        "comparison": f"{plan.primary[0]} - {plan.primary[1]}",
        "metric": plan.metric,
        "channels_where_a_is_better": sorted(wins),
        "channels_where_a_is_worse": sorted(losses),
        "channels_with_no_detectable_difference": sorted(flat),
        "statement": (
            "перевага підтверджена лише в перелічених каналах; там, де інтервал "
            "містить нуль, різниця не встановлена - це не доказ рівності"),
    }


__all__ = ["DEFAULT_CRITERION", "PRIMARY", "UNAUTHENTICATED", "AnalysisPlan",
           "load_observations", "subset", "channels_in", "analyse", "operability",
           "failure_table"]
