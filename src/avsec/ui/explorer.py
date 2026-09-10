"""Server-side of the saved-run browser in the web interface.

The UI and the CLI must never disagree, so every function here reads the same
tables `avsec analyze` and `avsec plots` wrote - it recomputes nothing and runs
no experiment.  Where a table is missing, the answer says so instead of
returning an empty result that looks like a measurement of zero.
"""
from __future__ import annotations

import base64
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from avsec.program import CATALOGUE, FIGURES, programme_status

CHANNEL_ORDER = ("clean", "mild", "moderate", "bursty", "harsh")


def _ch_key(name: str) -> Tuple[int, str]:
    base = str(name).split("[")[0]
    return (CHANNEL_ORDER.index(base) if base in CHANNEL_ORDER else 99, str(name))


def _read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _f(row: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def list_runs(runs_dir: str = "runs") -> List[Dict[str, Any]]:
    """Every saved run directory, newest first, with what it actually contains."""
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(runs_dir):
        return out
    for name in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, name)
        if not os.path.isdir(d) or name.startswith("_"):
            continue
        manifest = _read_json(os.path.join(d, "run_manifest.json"))
        analysis = _read_json(os.path.join(d, "analysis.json"))
        figures = _read_csv(os.path.join(d, "figures", "figure_index.csv"))
        matrix = _read_json(os.path.join(d, "matrix.json"))
        out.append({
            "name": name,
            "path": d.replace(os.sep, "/"),
            "run_id": manifest.get("run_id", ""),
            "commit": manifest.get("commit", ""),
            "started": manifest.get("started", ""),
            "config_name": manifest.get("config_name", ""),
            "n_frames": matrix.get("n_frames", 0),
            "n_jobs": matrix.get("n_jobs_total", 0),
            "complete": (matrix.get("completeness") or {}).get("complete"),
            "n_scenes": analysis.get("n_scenes", 0),
            "methods": analysis.get("methods", []),
            "channels": analysis.get("channels", []),
            "figures_ready": sum(1 for r in figures if r.get("status") == "ready"),
            "figures_total": len(figures) or len(CATALOGUE),
            "has_analysis": bool(analysis),
            "mtime": os.path.getmtime(d),
        })
    return sorted(out, key=lambda r: -r["mtime"])


def run_overview(run_dir: str) -> Dict[str, Any]:
    """Header block: identity, scale, completeness and the standing caveats."""
    manifest = _read_json(os.path.join(run_dir, "run_manifest.json"))
    analysis = _read_json(os.path.join(run_dir, "analysis.json"))
    matrix = _read_json(os.path.join(run_dir, "matrix.json"))
    dataset = manifest.get("dataset", {})
    return {
        "run_id": manifest.get("run_id", ""),
        "commit": manifest.get("commit", ""),
        "config_name": manifest.get("config_name", ""),
        "environment": manifest.get("environment", {}),
        "seeds": manifest.get("seeds", {}),
        "plan": manifest.get("plan", {}),
        "dataset": {k: dataset.get(k) for k in
                    ("n_clips", "n_scenes", "clips_per_split", "scenes_per_split",
                     "provenances", "warnings", "problems")},
        "scale": {"n_frames": matrix.get("n_frames", 0),
                  "n_jobs": matrix.get("n_jobs_total", 0),
                  "n_scenes_analysed": analysis.get("n_scenes", 0),
                  "wall_s": matrix.get("wall_s_measured")},
        "completeness": matrix.get("completeness", {}),
        "methods": analysis.get("methods", []),
        "channels": sorted(analysis.get("channels", []), key=_ch_key),
        "primary_verdict": analysis.get("primary_verdict", {}),
        "caveats": [
            "увесь матеріал синтетичний - жодного кадру з реальної камери",
            "канал - програмна модель; однакова назва профілю на рівнях A і B "
            "не означає однаковий канал",
            "апаратних вимірювань немає (E11 не виконано)",
        ],
    }


def filters(run_dir: str) -> Dict[str, Any]:
    """What the run actually contains, for the scene/method/channel selectors."""
    frames = _read_csv(os.path.join(run_dir, "frames.csv"))
    clips = _read_csv(os.path.join(run_dir, "dataset_manifest.csv"))
    cat = {c["clip_id"]: c.get("category", "other") for c in clips}
    scenes = sorted({r.get("scene", "") for r in frames if r.get("scene")})
    return {
        "methods": sorted({r.get("method", "") for r in frames if r.get("method")}),
        "channels": sorted({r.get("channel", "") for r in frames
                            if r.get("channel")}, key=_ch_key),
        "scenes": scenes,
        "clips": sorted({r.get("clip", "") for r in frames if r.get("clip")}),
        "categories": sorted(set(cat.values())),
        "clip_category": cat,
        "metrics": ["psnr_full", "ssim_full", "coverage", "psnr_verified",
                    "max_age_frames", "stale_fraction", "mse_full"],
    }


def summary(run_dir: str, methods: Optional[Sequence[str]] = None,
            channels: Optional[Sequence[str]] = None,
            metric: str = "psnr_full") -> Dict[str, Any]:
    """The per-method table, filtered.  Read from `summary.csv`, not recomputed."""
    rows = [r for r in _read_csv(os.path.join(run_dir, "summary.csv"))
            if r.get("metric") == metric]
    if not rows:
        return {"error": "у цьому прогоні немає summary.csv - запустіть "
                         "`avsec analyze --input <run>`", "rows": []}
    if methods:
        rows = [r for r in rows if r["method"] in set(methods)]
    if channels:
        rows = [r for r in rows if r["channel"] in set(channels)]
    out = [{
        "method": r["method"], "channel": r["channel"], "metric": metric,
        "mean": _f(r, "mean"), "lo": _f(r, "lo"), "hi": _f(r, "hi"),
        "n_scenes": int(_f(r, "n_scenes", 0)),
        "n_observations": int(_f(r, "n_observations", 0)),
        "n_failed": int(_f(r, "n_failed", 0)),
        "authenticated": r.get("authenticated") == "True",
        "coverage_meaning": r.get("coverage_meaning", ""),
    } for r in rows]
    return {"rows": sorted(out, key=lambda r: (_ch_key(r["channel"]), r["method"])),
            "metric": metric}


def paired(run_dir: str, a: str, b: str, channel: Optional[str] = None,
           metric: str = "psnr_full") -> Dict[str, Any]:
    """One paired comparison, oriented as ``a - b``, straight from the analysis."""
    from avsec.analysis import orient

    rows = [r for r in _read_csv(os.path.join(run_dir, "paired_effects.csv"))
            if {r.get("a"), r.get("b")} == {a, b} and r.get("metric") == metric]
    if not rows:
        return {"error": f"немає порівняння {a} проти {b} за метрикою {metric}"}
    if channel:
        rows = [r for r in rows if r.get("channel") == channel]
    out = []
    for r in rows:
        o = orient({**r, "mean": _f(r, "mean"), "lo": _f(r, "lo"), "hi": _f(r, "hi")}, a)
        sig = r.get("significant_at_95") == "True"
        out.append({
            "channel": r.get("channel"), "a": a, "b": b,
            "mean": o["mean"], "lo": o["lo"], "hi": o["hi"],
            "n_scenes": int(_f(r, "n_scenes", 0)),
            "n_scenes_dropped": int(_f(r, "n_scenes_dropped", 0)),
            "significant_at_95": sig,
            "significant_holm": r.get("significant_holm") == "True",
            "p_holm": _f(r, "p_holm"),
            "verdict": ("різниця не встановлена" if not sig
                        else f"перевага {a}" if o["mean"] > 0 else f"перевага {b}"),
        })
    return {"rows": sorted(out, key=lambda r: _ch_key(r["channel"] or "")),
            "method": "парний кластерний бутстреп за сценами; сцени, де немає "
                      "результату одного з методів, виключені й полічені"}


def per_scene(run_dir: str, methods: Sequence[str], channel: str,
              metric: str = "psnr_full") -> Dict[str, Any]:
    """Scene-by-scene values, so a reader can see where a method loses."""
    from collections import defaultdict

    frames = _read_csv(os.path.join(run_dir, "frames.csv"))
    cells: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in frames:
        if r.get("channel") != channel or r.get("method") not in set(methods):
            continue
        v = _f(r, metric)
        if np.isfinite(v):
            cells[(r.get("scene", ""), r["method"])].append(v)
    scenes = sorted({k[0] for k in cells})
    return {
        "channel": channel, "metric": metric, "scenes": scenes,
        "series": {m: [float(np.mean(cells[(s, m)])) if (s, m) in cells else None
                       for s in scenes] for m in methods},
        "note": "значення усереднені за кадрами й повторами всередині сцени",
    }


def failures(run_dir: str) -> Dict[str, Any]:
    rows = _read_csv(os.path.join(run_dir, "failures.csv"))
    by_kind: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_kind.setdefault(r.get("status", "?"), []).append(r)
    return {
        "n_total": len(rows),
        "by_kind": {k: {"n": len(v),
                        "methods": sorted({x.get("method", "") for x in v}),
                        "clips": sorted({x.get("clip", "") for x in v}),
                        "example": (v[0].get("detail") or "")[:200]}
                    for k, v in sorted(by_kind.items())},
        "note": "відмова - це спостереження, а не пропуск: сцена, де метод не "
                "вміщується у бюджет, відсутня в його середньому",
    }


def figure_catalogue(run_dir: str) -> Dict[str, Any]:
    """The G01-G43 readiness table, with the reason for anything pending."""
    rows = _read_csv(os.path.join(run_dir, "figures", "figure_index.csv"))
    if not rows:
        rows = [{"figure": f.gid, "title_uk": f.title_uk, "title_en": f.title_en,
                 "experiment": f.source, "status": "pending",
                 "reason": "каталог не побудовано: `avsec plots --input <run>`",
                 "files": "", "data_table": ""} for f in CATALOGUE]
    for r in rows:
        fig = FIGURES.get(r.get("figure", ""))
        r["axes"] = fig.axes if fig else ""
        r["panels"] = fig.panels if fig else 1
    return {
        "rows": rows,
        "n_ready": sum(1 for r in rows if r.get("status") == "ready"),
        "n_total": len(rows),
        "programme": programme_status(rows),
        "note": "PNG, SVG і PDF одного рисунка - один результат, не три",
    }


def figure_image(run_dir: str, gid: str) -> Dict[str, Any]:
    """The PNG of one figure as a data URL, plus its data table and parameters."""
    base = os.path.join(run_dir, "figures", gid)
    png = base + ".png"
    if not os.path.exists(png):
        return {"error": f"{gid} не побудовано у цьому прогоні",
                "figure": gid, "status": "pending"}
    with open(png, "rb") as fh:
        data = base64.b64encode(fh.read()).decode("ascii")
    exports = {fmt: (base + "." + fmt).replace(os.sep, "/")
               for fmt in ("png", "svg", "pdf")
               if os.path.exists(base + "." + fmt)}
    return {
        "figure": gid, "status": "ready",
        "image": "data:image/png;base64," + data,
        "exports": exports,
        "params": _read_json(base + ".json"),
        "data_rows": _read_csv(base + ".csv")[:200],
        "data_table": (base + ".csv").replace(os.sep, "/"),
    }


def figure_export(run_dir: str, gid: str, fmt: str = "svg") -> Dict[str, Any]:
    """Raw bytes of one export, base64 encoded, for the browser to save."""
    path = os.path.join(run_dir, "figures", f"{gid}.{fmt}")
    if fmt not in ("png", "svg", "pdf", "csv", "json") or not os.path.exists(path):
        return {"error": f"немає {gid}.{fmt} у цьому прогоні"}
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode("ascii")
    ctype = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf",
             "csv": "text/csv", "json": "application/json"}[fmt]
    return {"figure": gid, "format": fmt, "content_type": ctype,
            "filename": f"{gid}.{fmt}", "base64": data,
            "bytes": os.path.getsize(path)}


def age_map(run_dir: str, clip: str, method: str, channel: str,
            frame_id: int = 0) -> Dict[str, Any]:
    """Re-render one recorded frame and return its availability overlay.

    The clip, method, channel and frame all come from the run's own table and
    the channel trace is rebuilt from the run's seed, so the picture shown is
    the one that produced the numbers next to it.
    """
    import dataclasses

    from avsec.channel import ChannelTrace
    from avsec.config import load_config
    from avsec.evaluation import availability_overlay
    from avsec.experiments import build_methods, build_sources
    from avsec.ui.server import png_data_url

    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        return {"error": "у прогоні немає config.yaml - його неможливо відтворити"}
    cfg = load_config(cfg_path)
    base = str(channel).split("[")[0] or cfg.channel_preset
    sub = dataclasses.replace(cfg, methods=(method,), channel_preset=base,
                              channel_overrides={})
    srcs = {s.name: s for s in build_sources(sub)}
    src = srcs.get(clip)
    if src is None:
        return {"error": f"кліп {clip} відсутній у наборі джерел прогону"}
    frame_id = max(0, min(int(frame_id), len(src.frames) - 1))
    try:
        m = build_methods(sub)[method]
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    trace = ChannelTrace(seed=cfg.channel_seed_value,
                         scene=f"{clip}|{channel}", repetition=0, profile=channel,
                         rasters_per_frame=cfg.budget.rasters_per_frame)
    m.reset()
    res = None
    try:
        for fi in range(frame_id + 1):
            res = m.process(src.frames[fi], fi, trace)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    stale = getattr(res, "stale", np.zeros_like(res.available))
    return {
        "clip": clip, "method": method, "channel": channel, "frame_id": frame_id,
        "original": png_data_url(res.original),
        "received": png_data_url(res.received),
        "reconstructed": png_data_url(res.reconstructed),
        "overlay": png_data_url(availability_overlay(res.reconstructed,
                                                     res.available, stale)),
        "coverage": float(res.available.mean()),
        "stale_fraction": float(np.asarray(stale).mean()),
        "estimated_fraction": float(1.0 - res.available.mean()
                                    - np.asarray(stale).mean()),
        "psnr_full": float(res.metrics.psnr_full),
        "authenticated": bool(getattr(m, "authenticated", False)),
        "legend": {"green": "поточне перевірене", "amber": "старе",
                   "red": "домальоване"},
    }


__all__ = ["list_runs", "run_overview", "filters", "summary", "paired",
           "per_scene", "failures", "figure_catalogue", "figure_image",
           "figure_export", "age_map"]
