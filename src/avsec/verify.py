"""Recompute every published claim from the published tables.

The point of this module is narrow and specific: a reader who has the contents
of ``results/main`` and nothing else must be able to check each "significant"
statement one line at a time, without re-running a single simulation (defect
R11, completion criterion of R06).

It answers four questions about a results directory:

``claims``
    every "P is better than X in channel Y" the analysis supports, recomputed
    from ``frames.csv`` and checked against what ``analysis.json`` recorded.

``tables``
    do the aggregates on disk (``summary.csv``, ``paired_effects.csv``) match
    what the raw rows produce when the declared plan is applied to them?

``provenance``
    was the code that produced this committed, are the seeds recorded, are the
    input hashes there?

``figures``
    does every figure that a report cites exist with its data table?

A mismatch is printed with both numbers.  Nothing is repaired silently.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from avsec.analysis import AnalysisPlan, channels_in, load_observations, orient, subset
from avsec.utils import write_json

#: How far a recomputed number may differ from the published one before it is
#: called a mismatch.  The bootstrap is seeded, so an exact match is the normal
#: case; the tolerance covers CSV rounding only.
TOLERANCE = 5e-3


def _read_csv(path: str) -> List[Dict[str, str]]:
    import csv

    from avsec.utils import open_table, table_exists

    if not table_exists(path):
        return []
    with open_table(path) as fh:
        return list(csv.DictReader(fh))


def _f(row: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _delta(a: float, b: float) -> float:
    """Difference between two published numbers, where "no value" matches.

    A comparison over an empty set has no mean, and two empty comparisons are
    the same result.  Plain subtraction says ``nan``, which is not a
    disagreement - it is the absence of a number on both sides.
    """
    if not np.isfinite(a) and not np.isfinite(b):
        return 0.0
    if not np.isfinite(a) or not np.isfinite(b):
        return float("inf")
    return abs(a - b)


def verify(run_dir: str, plan: Optional[AnalysisPlan] = None,
           output: Optional[str] = None) -> Dict[str, Any]:
    """Recompute the claims of ``run_dir`` and report every disagreement."""
    plan = plan or _plan_of(run_dir)
    checks: List[Dict[str, Any]] = []

    table, _rows = load_observations(run_dir, plan.unit_of_independence)
    published = _read_csv(os.path.join(run_dir, "paired_effects.csv"))
    by_key = {(r.get("channel", ""), r.get("a", ""), r.get("b", "")): r
              for r in published}

    # ---- 1. every significant claim, recomputed from the raw rows --------
    for ch in channels_in(table):
        sub = subset(table, ch)
        for key, pub in sorted(by_key.items()):
            if key[0] != ch:
                continue
            a, b = key[1], key[2]
            if a not in sub.methods or b not in sub.methods:
                continue
            fresh = sub.paired(a, b, plan.metric, plan.n_boot, plan.seed,
                               plan.equivalence_margin)
            pub_sig = str(pub.get("significant_at_95", "")).lower() == "true"
            agree_sig = bool(fresh["significant_at_95"]) == pub_sig
            # "No usable instant in either method" is a legitimate published
            # result, and two of them agree.  NaN != NaN, so it has to be said.
            d_mean = _delta(fresh["mean"], _f(pub, "mean"))
            d_lo = _delta(fresh["lo"], _f(pub, "lo"))
            d_hi = _delta(fresh["hi"], _f(pub, "hi"))
            empty = fresh["n_units"] == 0 and int(_f(pub, "n_units", 0)) == 0
            ok = (agree_sig and d_mean <= TOLERANCE
                  and d_lo <= TOLERANCE and d_hi <= TOLERANCE)
            checks.append({
                "kind": "paired_effect", "channel": ch, "a": a, "b": b,
                "published_mean": _f(pub, "mean"),
                "recomputed_mean": round(fresh["mean"], 4),
                "published_ci": [_f(pub, "lo"), _f(pub, "hi")],
                "recomputed_ci": [round(fresh["lo"], 4), round(fresh["hi"], 4)],
                "published_significant": pub_sig,
                "recomputed_significant": bool(fresh["significant_at_95"]),
                "n_units": fresh["n_units"],
                "p_raw": round(float(fresh.get("p_raw", float("nan"))), 5),
                "ok": ok,
                "no_data": empty,
                "detail": ("жоден метод не дав придатного моменту показу"
                           if empty and ok else
                           "" if ok else
                           f"Δmean={d_mean:.4f} Δlo={d_lo:.4f} Δhi={d_hi:.4f}"
                           + ("" if agree_sig else "; significance disagrees")),
            })

    # ---- 2. the per-method summary --------------------------------------
    pub_summary = _read_csv(os.path.join(run_dir, "summary.csv"))
    for ch in channels_in(table):
        sub = subset(table, ch)
        fresh = {r["method"]: r for r in sub.summary(plan.metric, plan.n_boot,
                                                     plan.seed)}
        for row in pub_summary:
            if row.get("channel") != ch or row.get("metric") != plan.metric:
                continue
            m = row.get("method", "")
            f = fresh.get(m)
            if f is None:
                checks.append({"kind": "summary", "channel": ch, "method": m,
                               "ok": False,
                               "detail": "у сирих рядках цього методу немає"})
                continue
            d = _delta(f["mean"], _f(row, "mean"))
            checks.append({
                "kind": "summary", "channel": ch, "method": m,
                "published_mean": _f(row, "mean"),
                "recomputed_mean": round(f["mean"], 4),
                "n_units": f["n_units"],
                "ok": bool(d <= TOLERANCE),
                "detail": "" if d <= TOLERANCE else f"Δ={d:.4f}",
            })

    # ---- 3. provenance ---------------------------------------------------
    prov = _provenance(run_dir)
    checks.extend(prov["checks"])

    # ---- 4. figures a report may cite ------------------------------------
    checks.extend(_figures(run_dir))

    failed = [c for c in checks if not c["ok"]]
    result = {
        "kind": "verify",
        "run_dir": os.path.abspath(run_dir),
        "plan": plan.to_dict(),
        "plan_fingerprint": plan.fingerprint(),
        "tolerance": TOLERANCE,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "ok": not failed,
        "checks": checks,
        "provenance": prov["record"],
        "statement": (
            "усі опубліковані числа відтворено з таблиць цієї теки"
            if not failed else
            f"{len(failed)} перевірок не збіглися - див. checks"),
    }
    if output:
        write_json(output, result)
    return result


def _plan_of(run_dir: str) -> AnalysisPlan:
    """Use the plan the run actually recorded, not today's default."""
    path = os.path.join(run_dir, "analysis.json")
    if not os.path.exists(path):
        return AnalysisPlan()
    with open(path, encoding="utf-8") as fh:
        d = (json.load(fh) or {}).get("plan") or {}
    plan = AnalysisPlan()
    plan.metric = str(d.get("metric", plan.metric))
    pr = d.get("primary_comparison")
    if pr:
        plan.primary = (str(pr[0]), str(pr[1]))
    if d.get("family"):
        plan.family = tuple((str(p[0]), str(p[1])) for p in d["family"])
    plan.n_boot = int(d.get("n_boot", plan.n_boot))
    plan.seed = int(d.get("seed", plan.seed))
    plan.unit_of_independence = str(d.get("unit_of_independence",
                                          plan.unit_of_independence))
    plan.equivalence_margin = float(d.get("equivalence_margin",
                                          plan.equivalence_margin))
    return plan


def _provenance(run_dir: str) -> Dict[str, Any]:
    """Is this run pinned to code, configuration, seeds and inputs? (R11)"""
    path = os.path.join(run_dir, "run_manifest.json")
    checks: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return {"record": {},
                "checks": [{"kind": "provenance", "item": "run_manifest.json",
                            "ok": False, "detail": "відсутній"}]}
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    env = rec.get("environment") or {}

    def _check(item: str, ok: bool, detail: str = "") -> None:
        checks.append({"kind": "provenance", "item": item, "ok": bool(ok),
                       "detail": detail})

    _check("git commit", bool(env.get("git_commit")), str(env.get("git_commit", "")))
    dirty = env.get("git_dirty")
    _check("working tree clean at run time", dirty is False,
           "стан невідомий (старий прогін без запису)" if dirty is None
           else ("DIRTY: " + ", ".join(env.get("git_dirty_files", [])[:5])
                 if dirty else "clean"))
    _check("resolved configuration", bool(rec.get("config")))
    seeds = rec.get("seeds") or {}
    _check("seeds recorded", all(k in seeds for k in
                                 ("seed", "channel_seed", "crypto_seed")),
           json.dumps(seeds, ensure_ascii=False))
    _check("key mode is the reproducible lab mode",
           str(seeds.get("key_mode", "")) == "lab", str(seeds.get("key_mode", "")))
    _check("dependency versions", bool(env.get("packages")))
    _check("exact package list (pip freeze)", bool(env.get("pip_freeze")),
           f"{len(env.get('pip_freeze') or [])} записів")
    _check("matrix plan", bool(rec.get("plan")))

    ds = rec.get("dataset") or {}
    _check("dataset validated", bool(ds.get("ok")),
           "; ".join(ds.get("problems", [])[:3]))
    manifest = _read_csv(os.path.join(run_dir, "dataset_manifest.csv"))
    _check("input content hashes",
           bool(manifest) and all(r.get("content_sha256") for r in manifest),
           f"{len(manifest)} кліпів")
    n_sources = len({r.get("source_id") or r.get("parent_scene_id")
                     for r in manifest})
    _check("independent sources recorded", n_sources > 0,
           f"{n_sources} джерел на {len(manifest)} кліпів")
    return {"record": {"commit": env.get("git_commit"),
                       "worktree": env.get("git_worktree"),
                       "n_clips": len(manifest), "n_sources": n_sources,
                       "seeds": seeds}, "checks": checks}


def _figures(run_dir: str) -> List[Dict[str, Any]]:
    """Does every figure this run *could* produce exist, with its data table?

    A run directory need not contain every experiment: the natural-imagery run
    is a transfer check and runs the matrix and E13, not the whole programme.
    A figure whose input tables are simply not in this directory is reported as
    not applicable rather than as a failure - but it is still listed, so the
    difference between "not run here" and "broken" stays visible.
    """
    from avsec.program import KEY, figure_status

    directory = os.path.join(run_dir, "figures")
    out: List[Dict[str, Any]] = []
    for fig in KEY:
        st = figure_status(directory, fig.gid)
        ready = st["status"] == "ready"
        has_input = any(os.path.exists(os.path.join(run_dir, t))
                        for t in (fig.tables or ()))
        out.append({"kind": "figure", "figure": fig.gid,
                    "ok": ready or not has_input,
                    "not_applicable": not ready and not has_input,
                    "detail": ("" if ready else
                               (f"{fig.source} не запускався у цій теці"
                                if not has_input else st["reason"])),
                    "data_table": st["data_table"]})
    return out


def render(result: Dict[str, Any], width: int = 96) -> str:
    """A plain-text report, one line per check."""
    lines = [f"verify {result['run_dir']}",
             f"план {result['plan_fingerprint']}  "
             f"метрика {result['plan']['metric']}  "
             f"одиниця незалежності: {result['plan']['unit_of_independence']}",
             "-" * width]
    for c in result["checks"]:
        mark = "ok  " if c["ok"] else "FAIL"
        if c["kind"] == "paired_effect":
            body = (f"{c['channel']:<22} {c['a']:>4} − {c['b']:<4} "
                    f"{c['recomputed_mean']:+7.3f} дБ "
                    f"[{c['recomputed_ci'][0]:+.2f}; {c['recomputed_ci'][1]:+.2f}] "
                    f"n={c['n_units']:<3} p={c['p_raw']:.4f} "
                    f"{'значуще' if c['recomputed_significant'] else 'не встановлено'}")
        elif c["kind"] == "summary":
            body = (f"{c['channel']:<22} {c['method']:<5} "
                    f"середнє {c.get('recomputed_mean', float('nan')):7.3f}")
        elif c["kind"] == "figure":
            body = f"рисунок {c['figure']}"
            if c.get("not_applicable"):
                mark = "n/a "
                body += f"  ({c['detail']})"
        else:
            body = f"{c.get('item', '')}"
        detail = f"  <- {c['detail']}" if c.get("detail") and not c["ok"] else ""
        lines.append(f"[{mark}] {body}{detail}")
    lines.append("-" * width)
    lines.append(f"{result['n_checks']} перевірок, {result['n_failed']} не збіглися")
    lines.append(result["statement"])
    return "\n".join(lines)


__all__ = ["TOLERANCE", "verify", "render"]
