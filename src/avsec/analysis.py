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

#: The pre-declared family the Holm correction is for (R06).  It is fixed here,
#: in code, before any run; adding an exploratory pair afterwards cannot change
#: a decision inside it, because pairs outside the family take no part in the
#: adjustment.
#:
#: Every entry answers a question the study asked in advance:
#:   P vs B4       - does the proposed scheme beat the scheme it extends?
#:   P vs B4t      - does it beat that scheme once B4 is given P's transport?
#:   B4t vs B4     - how much of any difference is the transport parameters?
#:   P vs B3       - does it beat the strongest authenticated baseline?
#:   P vs B0d      - what does the whole protected path cost against a plain
#:                   digital-over-analog link that authenticates nothing?
PRIMARY_FAMILY = (("P", "B4"), ("P", "B4t"), ("B4t", "B4"), ("P", "B3"),
                  ("P", "B0d"))

#: Smallest difference in dB that the study treats as practically meaningful.
#: Used for the equivalence test, so "no difference found" and "shown to be
#: equivalent" are different statements (R06).
EQUIVALENCE_MARGIN_DB = 0.5

#: Methods that display a picture nobody authenticated.  Their "coverage" is
#: not comparable with an authenticated method's coverage and is labelled.
UNAUTHENTICATED = ("B0a", "B0a-R", "B0d", "B0d-W", "B1", "B2")


@dataclass
class AnalysisPlan:
    """The statistical plan, written next to the results it produced."""

    metric: str = "psnr_full"
    secondary: Tuple[str, ...] = ("ssim_full", "coverage", "psnr_verified",
                                  "availability", "psnr_displayed")
    primary: Tuple[str, str] = PRIMARY
    family: Tuple[Tuple[str, str], ...] = PRIMARY_FAMILY
    n_boot: int = 10000
    seed: int = 20240909
    criterion: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_CRITERION))
    #: ``scene`` or ``source``.  ``source`` is the level that generalises to a
    #: new recording; ``scene`` only to a new view of the recordings used (R07).
    unit_of_independence: str = "scene"
    equivalence_margin: float = EQUIVALENCE_MARGIN_DB
    #: The conditions the primary claim is about, declared before the run.
    primary_channels: Tuple[str, ...] = ("clean", "moderate", "bursty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric, "secondary": list(self.secondary),
            "primary_comparison": list(self.primary),
            "primary_channels": list(self.primary_channels),
            "family": [list(p) for p in self.family],
            "n_boot": self.n_boot,
            "seed": self.seed, "criterion": self.criterion,
            "unit_of_independence": self.unit_of_independence,
            "equivalence_margin": self.equivalence_margin,
            "aggregation": "frames -> repetitions -> clips -> scenes -> sources; "
                           f"the bootstrap resamples {self.unit_of_independence}s",
            "channel_pooling": ("не виконується: канали - це різні умови, а не "
                                "повторні реалізації однієї; об'єднаного "
                                "показника ALL немає"),
            "test": ("двобічний bootstrap-тест на парних різницях між методами "
                     "на однакових незалежних джерелах; поряд наводиться "
                     "критерій знакових рангів Вілкоксона"),
            "multiplicity": "Holm step-down over the pre-declared family above; "
                            "pairs outside it are exploratory and do not take "
                            "part in the adjustment",
            "negative_results": ("відсутність значущості не є доказом рівності; "
                                 "для цього окремо рахується TOST з межею "
                                 f"±{self.equivalence_margin} дБ"),
        }

    def fingerprint(self) -> str:
        """SHA-256 of the plan, so a published number names the plan it used."""
        import hashlib

        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

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
        if d.get("family"):
            plan.family = tuple((str(p[0]), str(p[1])) for p in d["family"])
        if d.get("primary_channels"):
            plan.primary_channels = tuple(str(c) for c in d["primary_channels"])
        plan.n_boot = int(d.get("n_boot", plan.n_boot))
        plan.seed = int(d.get("seed", plan.seed))
        plan.unit_of_independence = str(
            d.get("unit_of_independence", plan.unit_of_independence))
        plan.equivalence_margin = float(
            d.get("equivalence_margin", plan.equivalence_margin))
        if d.get("criterion"):
            plan.criterion.update(d["criterion"])
        return plan


# ------------------------------------------------------------------- loading
#: Row fields that describe *where* a measurement came from, not *what* it is.
_NON_METRIC = frozenset({
    "method", "scene", "clip", "channel", "status", "detail", "job_id",
    "trace_id", "provenance", "channel_axis", "channel_plan", "source_id",
    "config_id", "factors", "profile_id",
})


def load_observations(run_dir: str, unit_of_independence: str = "scene"
                      ) -> Tuple[ResultTable, List[Dict[str, Any]]]:
    """Read a run's frame rows into the shared table, keeping the raw rows.

    The channel label and the profile identity travel *into* the observation
    (R02).  Before that they were squeezed into ``detail``, which the
    aggregation key ignored, so two channels' rows for the same frame index
    overwrote one another and any pooled estimate was "whichever row was read
    last" rather than an average.
    """
    rows = _read_rows(run_dir)
    sources = _source_map(run_dir)
    table = ResultTable(unit_of_independence=unit_of_independence)
    for r in rows:
        metrics: Dict[str, float] = {}
        for k, v in r.items():
            if k in _NON_METRIC:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(f):
                metrics[k] = f
        clip = str(r.get("clip", ""))
        scene = str(r.get("scene", ""))
        channel = str(r.get("channel", ""))
        table.add(Observation(
            method=str(r.get("method", "")), scene=scene, clip=clip,
            repetition=int(float(r.get("repetition", 0) or 0)),
            frame=int(float(r.get("frame_id", 0) or 0)),
            status=str(r.get("status", "ok")), metrics=metrics,
            channel=channel,
            config_id=str(r.get("profile_id", "") or r.get("config_id", "")),
            source_id=sources.get(clip, "") or scene,
            detail=channel))
    return table, rows


def _source_map(run_dir: str) -> Dict[str, str]:
    """``clip_id -> source_id`` from the run's own dataset manifest."""
    import csv

    path = os.path.join(run_dir, "dataset_manifest.csv")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", newline="") as fh:
        return {r["clip_id"]: (r.get("source_id") or r.get("parent_scene_id", ""))
                for r in csv.DictReader(fh)}


def _read_rows(run_dir: str) -> List[Dict[str, Any]]:
    import csv

    path = os.path.join(run_dir, "frames.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run `avsec matrix` before `avsec analyze`")
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    # Every scheduled instant already has a row in frames.csv, with its own
    # status (R05).  ``failures.csv`` is per *job*, so only the jobs that
    # produced no rows at all are added here - otherwise a clip that hit
    # capacity would be counted twice.
    produced = {r.get("job_id") for r in rows}
    fail = os.path.join(run_dir, "failures.csv")
    if os.path.exists(fail):
        with open(fail, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("job_id") in produced:
                    continue
                r.setdefault("frame_id", "-1")
                rows.append(r)
    return rows


def subset(table: ResultTable, channel: Optional[str]) -> ResultTable:
    """A view of one channel point."""
    if channel is None:
        return table
    return table.view([r for r in table.rows if r.channel == channel])


def channels_in(table: ResultTable) -> List[str]:
    return sorted({r.channel for r in table.rows if r.channel})


# ----------------------------------------------------------------- analysis
def analyse(run_dir: str, plan: Optional[AnalysisPlan] = None,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Produce every statistical table from one run directory."""
    plan = plan or AnalysisPlan()
    out_dir = ensure_dir(output_dir or run_dir)
    table, rows = load_observations(run_dir, plan.unit_of_independence)
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

    # ---- paired effects, per channel ------------------------------------
    # There is deliberately no pooled "ALL" row (R02).  Channels are different
    # operating conditions; averaging them needs weights nobody has declared,
    # and the research questions here are all asked per channel.
    effects: List[Dict[str, Any]] = []
    for ch in channels:
        sub = subset(table, ch)
        for rec in sub.all_pairs(plan.metric, n_boot=plan.n_boot, seed=plan.seed,
                                 primary=plan.primary, family=plan.family,
                                 equivalence_margin=plan.equivalence_margin):
            rec["channel"] = ch
            rec["family_label"] = "pre-declared" if rec.get("in_family") else \
                                  "exploratory"
            rec["units_only_in_a"] = ";".join(rec.get("units_only_in_a", []))
            rec["units_only_in_b"] = ";".join(rec.get("units_only_in_b", []))
            rec["scenes_only_in_a"] = rec["units_only_in_a"]
            rec["scenes_only_in_b"] = rec["units_only_in_b"]
            rec["diffs"] = ";".join(f"{d:.6f}" for d in rec.get("diffs", []))
            rec["units"] = ";".join(rec.get("units", []))
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
    family_rows = [e for e in effects if e.get("in_family")]
    result = {
        "kind": "analysis",
        "run_dir": os.path.abspath(run_dir),
        "plan": plan.to_dict(),
        "plan_fingerprint": plan.fingerprint(),
        "channels": channels,
        "methods": table.methods,
        "n_scenes": len(table.scenes),
        "n_sources": len(table.sources),
        "unit_of_independence": plan.unit_of_independence,
        "n_observations": len(table.rows),
        "primary": primary_rows,
        "primary_verdict": _verdict(primary_rows, plan),
        "family_size": len(family_rows),
        "n_effects": len(effects),
        "significant_claims": significant_claims(effects, plan),
        "operability_criterion": plan.criterion,
    }
    write_json(os.path.join(out_dir, "analysis.json"), result)
    return result


def significant_claims(effects: Sequence[Dict[str, Any]], plan: AnalysisPlan
                       ) -> List[Dict[str, Any]]:
    """Every "significant" statement this analysis supports, in one list.

    ``avsec verify`` recomputes exactly these from the published tables, so a
    claim in the text can be checked one line at a time (R06, R11).
    """
    out: List[Dict[str, Any]] = []
    for e in effects:
        if not e.get("significant_at_95"):
            continue
        out.append({
            "channel": e.get("channel"), "a": e.get("a"), "b": e.get("b"),
            "metric": e.get("metric"),
            "mean": round(float(e.get("mean", float("nan"))), 4),
            "lo": round(float(e.get("lo", float("nan"))), 4),
            "hi": round(float(e.get("hi", float("nan"))), 4),
            "n_units": int(e.get("n_units", 0)),
            "unit_of_independence": plan.unit_of_independence,
            "p_raw": e.get("p_raw"), "p_holm": e.get("p_holm"),
            "in_family": bool(e.get("in_family")),
            "significant_holm": bool(e.get("significant_holm")),
        })
    return sorted(out, key=lambda r: (str(r["channel"]), str(r["a"]), str(r["b"])))


def operability(table: ResultTable, plan: AnalysisPlan) -> List[Dict[str, Any]]:
    """Fraction of **scheduled** display instants meeting the criterion.

    R05: the denominator is every instant the clip was supposed to show, not
    only the ones the method managed to process.  A frame that hit the capacity
    limit, lost synchronisation or missed its deadline counts as not operable -
    otherwise a configuration raises its score by failing early.

    Computed per unit of independence first, then averaged, so a scene with
    many frames cannot dominate.  Unauthenticated methods are marked, because
    for them "coverage" means "a picture was shown", not "verified data".
    """
    cov_min = float(plan.criterion["coverage_verified_min"])
    psnr_min = float(plan.criterion["psnr_full_min"])
    use_source = plan.unit_of_independence == "source"
    out: List[Dict[str, Any]] = []
    channels = channels_in(table) or [""]
    for ch in channels:
        sub = subset(table, ch or None)
        for method in sub.methods:
            per_unit: Dict[str, List[float]] = {}
            avail: Dict[str, List[float]] = {}
            reasons: Dict[str, int] = {}
            for r in sub.rows:
                if r.method != method or r.frame < 0:
                    continue
                unit = r.source_id if use_source else r.scene
                if r.status != "ok":
                    per_unit.setdefault(unit, []).append(0.0)
                    avail.setdefault(unit, []).append(0.0)
                    reasons[r.status] = reasons.get(r.status, 0) + 1
                    continue
                cov = r.metrics.get("coverage", float("nan"))
                psnr = r.metrics.get("psnr_full", float("nan"))
                ok = bool(np.isfinite(cov) and np.isfinite(psnr)
                          and cov >= cov_min and psnr >= psnr_min)
                per_unit.setdefault(unit, []).append(1.0 if ok else 0.0)
                avail.setdefault(unit, []).append(1.0)
                if not ok:
                    reasons["below_criterion"] = reasons.get("below_criterion", 0) + 1
            if not per_unit:
                continue
            vals = np.asarray([float(np.mean(v)) for v in per_unit.values()])
            av = np.asarray([float(np.mean(v)) for v in avail.values()])
            n_instants = sum(len(v) for v in per_unit.values())
            out.append({
                "channel": ch, "method": method,
                "operable_fraction": float(vals.mean()),
                "availability": float(av.mean()),
                "n_scheduled_instants": int(n_instants),
                "unit_of_independence": plan.unit_of_independence,
                "n_units": int(vals.size),
                "n_scenes": int(vals.size),
                "sd_over_units": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "sd_over_scenes": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "not_operable_because": json.dumps(reasons, ensure_ascii=False),
                "authenticated": method not in UNAUTHENTICATED,
                "criterion": (f"coverage>={cov_min} and psnr_full>={psnr_min} "
                              "on every scheduled display instant "
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
            if r.get("significant_at_95") and r.get("mean", 0) > 0]
    losses = [c for c, r in per_channel.items()
              if r.get("significant_at_95") and r.get("mean", 0) < 0]
    flat = [c for c, r in per_channel.items() if not r.get("significant_at_95")]
    # "Not significant" splits in two: shown to be equivalent within the
    # declared margin, or simply undetermined.  They are different findings.
    equivalent = sorted(c for c in flat if per_channel[c].get("equivalent"))
    undetermined = sorted(c for c in flat if not per_channel[c].get("equivalent"))
    return {
        "comparison": f"{plan.primary[0]} - {plan.primary[1]}",
        "metric": plan.metric,
        "unit_of_independence": plan.unit_of_independence,
        "declared_channels": list(plan.primary_channels),
        "channels_where_a_is_better": sorted(wins),
        "channels_where_a_is_worse": sorted(losses),
        "channels_with_no_detectable_difference": sorted(flat),
        "channels_shown_equivalent": equivalent,
        "channels_undetermined": undetermined,
        "equivalence_margin": plan.equivalence_margin,
        "statement": (
            "перевага підтверджена лише в перелічених каналах; там, де інтервал "
            "містить нуль, різниця не встановлена - це не доказ рівності, і "
            f"окремо позначено, де рівність справді показана (±{plan.equivalence_margin} дБ)"),
    }


__all__ = ["DEFAULT_CRITERION", "PRIMARY", "PRIMARY_FAMILY", "EQUIVALENCE_MARGIN_DB",
           "UNAUTHENTICATED", "AnalysisPlan", "load_observations", "subset",
           "channels_in", "analyse", "operability", "failure_table", "orient",
           "significant_claims"]
