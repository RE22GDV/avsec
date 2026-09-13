"""The figure engine for the G01-G43 catalogue.

Rules this module enforces, because the plan requires them and because they are
the difference between a figure and a decoration:

* a figure is built **only** from tables a run already wrote - never by quietly
  running a new experiment to fill a gap;
* every figure exports PNG (300 dpi), SVG and PDF *plus* the CSV it was drawn
  from and a JSON of its build parameters; those three image formats are one
  result, not three;
* a figure whose data is missing is not skipped - it is recorded as ``pending``
  with the reason, and the reason names the experiment that would supply it;
* method names, colours and markers are identical in every figure;
* captions switch between Ukrainian and English from one dictionary.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from avsec.program import CATALOGUE, EXPORT_FORMATS, FIGURES, catalogue_status
from avsec.utils import ensure_dir, write_csv, write_json

DPI = 300

#: One colour and one marker per method, everywhere.
METHOD_STYLE: Dict[str, Dict[str, Any]] = {
    "B0a":   {"color": "#9e9e9e", "marker": "o", "ls": ":"},
    "B0a-R": {"color": "#616161", "marker": "s", "ls": ":"},
    "B0d":   {"color": "#8d6e63", "marker": "^", "ls": "-."},
    "B0d-W": {"color": "#a1887f", "marker": "v", "ls": "-."},
    "B1":    {"color": "#e57373", "marker": "x", "ls": "--"},
    "B2":    {"color": "#ba68c8", "marker": "+", "ls": "--"},
    "B3":    {"color": "#4fc3f7", "marker": "D", "ls": "-"},
    "B4":    {"color": "#1976d2", "marker": "o", "ls": "-"},
    "P":     {"color": "#2e7d32", "marker": "*", "ls": "-"},
}
DEFAULT_STYLE = {"color": "#455a64", "marker": ".", "ls": "-"}

#: Fixed channel order, so every figure reads left to right by severity.
CHANNEL_ORDER = ("clean", "mild", "moderate", "bursty", "harsh")

L: Dict[str, Dict[str, str]] = {
    "uk": {
        "channel": "профіль каналу", "psnr": "PSNR усього кадру, дБ",
        "ssim": "SSIM усього кадру", "coverage": "поточне перевірене покриття",
        "method": "метод", "scene": "сцена", "diff": "різниця, дБ",
        "operable": "частка показів у межах критерію",
        "ecdf": "емпірична функція розподілу",
        "noise": "СКВ шуму, коди яскравості",
        "burst_len": "довжина burst, рядки растру",
        "burst_rate": "burst на растр", "jitter": "СКВ зсуву рядка, px",
        "lowpass": "sigma ФНЧ", "gain": "підсилення", "offset": "зміщення",
        "bits": "біти на кадр", "latency": "модельна затримка, с",
        "buffer": "логічний буфер, кБ", "rate": "корисна швидкість, біт/с",
        "time": "час, с", "age": "вік показаних даних, кадри",
        "stage": "етап конвеєра", "ms": "мс", "memory": "пам'ять, МБ",
        "symbols": "символи", "row": "рядок символів", "col": "стовпець символів",
        "damaged": "пошкоджені байти RS", "descs": "втрачені описи однієї смуги",
        "frac": "частка", "resolution": "роздільність",
        "synthetic": "СИНТЕТИЧНІ ДАНІ", "scenes": "сцен",
        "unauth": "без автентифікації",
        "ci": "95% довірчий інтервал за сценами",
    },
    "en": {
        "channel": "channel profile", "psnr": "full-frame PSNR, dB",
        "ssim": "full-frame SSIM", "coverage": "current verified coverage",
        "method": "method", "scene": "scene", "diff": "difference, dB",
        "operable": "fraction of displays meeting the criterion",
        "ecdf": "empirical CDF",
        "noise": "noise sigma, luma codes",
        "burst_len": "burst length, raster lines",
        "burst_rate": "bursts per raster", "jitter": "line shift sigma, px",
        "lowpass": "low-pass sigma", "gain": "gain", "offset": "offset",
        "bits": "bits per frame", "latency": "model latency, s",
        "buffer": "logical buffer, kB", "rate": "payload rate, bit/s",
        "time": "time, s", "age": "age of displayed data, frames",
        "stage": "pipeline stage", "ms": "ms", "memory": "memory, MB",
        "symbols": "symbols", "row": "symbol row", "col": "symbol column",
        "damaged": "damaged RS bytes", "descs": "descriptions lost in one stripe",
        "frac": "fraction", "resolution": "resolution",
        "synthetic": "SYNTHETIC DATA", "scenes": "scenes",
        "unauth": "not authenticated",
        "ci": "95% confidence interval over scenes",
    },
}


class FigurePending(Exception):
    """Raised by a builder when the data it needs was never produced."""


@dataclass
class FigureContext:
    """Everything a builder may read.  Builders never touch the filesystem
    outside this directory, and never run an experiment."""

    run_dir: str
    out_dir: str
    lang: str = "uk"
    formats: Sequence[str] = EXPORT_FORMATS
    #: filled in lazily from the run's analysis, for captions that need scale
    _n_scenes_hint: int = 0
    _cache: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            self._n_scenes_hint = int(self.json("analysis.json").get("n_scenes", 0))
        except Exception:
            self._n_scenes_hint = 0

    def t(self, key: str) -> str:
        return L.get(self.lang, L["uk"]).get(key, key)

    # ------------------------------------------------------------ loading
    def path(self, *parts: str) -> str:
        return os.path.join(self.run_dir, *parts)

    def table(self, name: str) -> List[Dict[str, Any]]:
        """A CSV from the run, or ``FigurePending`` naming what is missing."""
        if name in self._cache:
            return self._cache[name]
        from avsec.utils import open_table, table_exists

        p = self.path(name)
        if not table_exists(p):
            raise FigurePending(f"немає таблиці {name} - відповідний експеримент "
                                "не запускався у цьому прогоні")
        # Evidence directories store the per-frame tables gzipped: every row is
        # there, at a size a repository can carry.
        with open_table(p) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise FigurePending(f"таблиця {name} порожня")
        self._cache[name] = rows
        return rows

    def json(self, name: str) -> Dict[str, Any]:
        if name in self._cache:
            return self._cache[name]
        p = self.path(name)
        if not os.path.exists(p):
            raise FigurePending(f"немає файлу {name}")
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        self._cache[name] = data
        return data


BUILDERS: Dict[str, Callable[[FigureContext], Dict[str, Any]]] = {}


def figure(gid: str) -> Callable:
    def deco(fn: Callable[[FigureContext], Dict[str, Any]]) -> Callable:
        BUILDERS[gid] = fn
        return fn
    return deco


# ------------------------------------------------------------------ plumbing
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": DPI, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "legend.frameon": False, "figure.autolayout": False,
    })
    return plt


def style(method: str) -> Dict[str, Any]:
    return METHOD_STYLE.get(method, DEFAULT_STYLE)


def channel_key(name: str) -> Tuple[int, str]:
    base = name.split("[")[0]
    return (CHANNEL_ORDER.index(base) if base in CHANNEL_ORDER else 99, name)


#: Figures that are montages of raster frames rather than vector plots.  The
#: source images are 256x192; rendering them at 300 dpi only upsamples pixels
#: and multiplies the file size, and the SVG then carries the same upsampled
#: bitmap base64-encoded.  Line art keeps the full resolution.
RASTER_FIGURES = frozenset({"G01", "G02", "G27", "K07", "K08"})
RASTER_DPI = 140


def export(ctx: FigureContext, gid: str, fig, rows: Sequence[Dict[str, Any]],
           params: Dict[str, Any]) -> Dict[str, Any]:
    """Write PNG+SVG+PDF, the data table and the build parameters."""
    ensure_dir(ctx.out_dir)
    files = []
    dpi = RASTER_DPI if gid in RASTER_FIGURES else DPI
    for fmt in ctx.formats:
        p = os.path.join(ctx.out_dir, f"{gid}.{fmt}")
        fig.savefig(p, format=fmt, bbox_inches="tight", dpi=dpi)
        files.append(p)
    _plt().close(fig)
    write_csv(os.path.join(ctx.out_dir, f"{gid}.csv"), list(rows))
    meta = dict(params)
    fdef = FIGURES.get(gid)
    meta.update({"figure": gid, "lang": ctx.lang, "run_dir": ctx.run_dir,
                 "title_uk": fdef.title_uk if fdef else "",
                 "title_en": fdef.title_en if fdef else "",
                 "experiment": fdef.source if fdef else "",
                 "formats": list(ctx.formats), "dpi": dpi})
    write_json(os.path.join(ctx.out_dir, f"{gid}.json"), meta)
    return {"figure": gid, "status": "ready", "files": files, "n_rows": len(rows)}


def title_of(ctx: FigureContext, gid: str) -> str:
    f = FIGURES[gid]
    return f"{gid}. " + (f.title_uk if ctx.lang == "uk" else f.title_en)


def _float(row: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        v = float(row.get(key, ""))
    except (TypeError, ValueError):
        return default
    return v


def _by(rows: Iterable[Dict[str, Any]], **eq: Any) -> List[Dict[str, Any]]:
    return [r for r in rows if all(str(r.get(k, "")) == str(v) for k, v in eq.items())]


def footnote(ctx: FigureContext, fig, text: str) -> None:
    """Caveat line under the figure.

    Placed *below* the axes box (negative figure coordinate) so it can never
    overlap an axis label; ``bbox_inches="tight"`` grows the canvas to include
    it.  Long lines are wrapped rather than clipped, because a caveat that does
    not fit is a caveat nobody reads.
    """
    import textwrap

    width = int(max(80, fig.get_size_inches()[0] * 20))
    wrapped = chr(10).join(textwrap.wrap(text, width))
    fig.text(0.0, -0.02, wrapped, fontsize=6.5, color="#555555", va="top")


# =============================================================== E02 figures
def _summary_rows(ctx: FigureContext, metric: str) -> List[Dict[str, Any]]:
    rows = [r for r in ctx.table("summary.csv") if r.get("metric") == metric]
    if not rows:
        raise FigurePending(f"у summary.csv немає метрики {metric}")
    return rows


def _profile_bars(ctx: FigureContext, gid: str, metric: str, ylabel: str,
                  mark_unauthenticated: bool = False) -> Dict[str, Any]:
    plt = _plt()
    rows = _summary_rows(ctx, metric)
    channels = sorted({r["channel"] for r in rows}, key=channel_key)
    methods = sorted({r["method"] for r in rows},
                     key=lambda m: list(METHOD_STYLE).index(m)
                     if m in METHOD_STYLE else 99)
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(channels))
    out_rows: List[Dict[str, Any]] = []
    for i, m in enumerate(methods):
        vals, lo, hi = [], [], []
        for ch in channels:
            sel = _by(rows, channel=ch, method=m)
            v = _float(sel[0], "mean") if sel else float("nan")
            vals.append(v)
            lo.append(v - _float(sel[0], "lo") if sel else 0.0)
            hi.append(_float(sel[0], "hi") - v if sel else 0.0)
            if sel:
                out_rows.append({"channel": ch, "method": m, "metric": metric,
                                 "mean": v, "lo": _float(sel[0], "lo"),
                                 "hi": _float(sel[0], "hi"),
                                 "n_scenes": sel[0].get("n_scenes"),
                                 "authenticated": sel[0].get("authenticated")})
        st = style(m)
        auth = all(_by(rows, channel=ch, method=m)[0].get("authenticated") == "True"
                   for ch in channels if _by(rows, channel=ch, method=m))
        label = m if auth or not mark_unauthenticated else f"{m} ({ctx.t('unauth')})"
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.92,
               yerr=[np.abs(lo), np.abs(hi)], capsize=2, label=label,
               color=st["color"], hatch="" if auth else "//",
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_xlabel(ctx.t("channel"))
    ax.set_ylabel(ylabel)
    ax.set_title(title_of(ctx, gid))
    ax.legend(ncol=5, fontsize=7.5, loc="upper right")
    n_sc = max((int(r.get("n_scenes") or 0) for r in rows), default=0)
    # The hatch is drawn on every one of these figures, so it is always
    # explained: a reader must never have to guess whether a bar is a method
    # that authenticates what it displays.
    footnote(ctx, fig, f"{ctx.t('ci')}; n = {n_sc} {ctx.t('scenes')}; "
                       f"штрихування = {ctx.t('unauth')}; {ctx.t('synthetic')}")
    return export(ctx, gid, fig, out_rows,
                  {"metric": metric, "channels": channels, "methods": methods,
                   "source_table": "summary.csv"})


@figure("G03")
def g03(ctx: FigureContext) -> Dict[str, Any]:
    return _profile_bars(ctx, "G03", "psnr_full", ctx.t("psnr"))


@figure("G04")
def g04(ctx: FigureContext) -> Dict[str, Any]:
    return _profile_bars(ctx, "G04", "ssim_full", ctx.t("ssim"))


@figure("G05")
def g05(ctx: FigureContext) -> Dict[str, Any]:
    return _profile_bars(ctx, "G05", "coverage", ctx.t("coverage"),
                         mark_unauthenticated=True)


@figure("G06")
def g06(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    rows = ctx.table("operability.csv")
    channels = sorted({r["channel"] for r in rows}, key=channel_key)
    methods = sorted({r["method"] for r in rows},
                     key=lambda m: list(METHOD_STYLE).index(m)
                     if m in METHOD_STYLE else 99)
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    x = np.arange(len(channels))
    width = 0.8 / max(len(methods), 1)
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(methods):
        vals = []
        for ch in channels:
            sel = _by(rows, channel=ch, method=m)
            v = _float(sel[0], "operable_fraction") if sel else float("nan")
            vals.append(v)
            if sel:
                out.append(dict(sel[0]))
        auth = any(r.get("authenticated") == "True" for r in _by(rows, method=m))
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.92,
               color=style(m)["color"], label=m, hatch="" if auth else "//",
               edgecolor="white", linewidth=0.4)
    crit = rows[0].get("criterion", "")
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(ctx.t("channel"))
    ax.set_ylabel(ctx.t("operable"))
    ax.set_title(title_of(ctx, "G06"))
    ax.legend(ncol=5, fontsize=7.5)
    footnote(ctx, fig, f"{crit}; {ctx.t('synthetic')}")
    return export(ctx, "G06", fig, out,
                  {"criterion": crit, "source_table": "operability.csv"})


@figure("G07")
def g07(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    rows = [r for r in ctx.table("paired_effects.csv")
            if r.get("a") in ("P", "B4") and r.get("b") in ("P", "B4")
            and r.get("a") != r.get("b")]
    if not rows:
        raise FigurePending("у paired_effects.csv немає порівняння P проти B4")
    rows = sorted(rows, key=lambda r: channel_key(r["channel"]))
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ys = np.arange(len(rows))
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        sign = 1.0 if r["a"] == "P" else -1.0
        mean = sign * _float(r, "mean")
        lo, hi = sign * _float(r, "lo"), sign * _float(r, "hi")
        lo, hi = min(lo, hi), max(lo, hi)
        sig = r.get("significant_at_95") == "True"
        holm = r.get("significant_holm") == "True"
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o",
                    color="#2e7d32" if sig else "#9e9e9e", capsize=3,
                    markersize=6 if holm else 4)
        out.append({"channel": r["channel"], "mean_P_minus_B4": mean, "lo": lo,
                    "hi": hi, "n_scenes": r.get("n_scenes"),
                    "significant_at_95": sig, "significant_holm": holm,
                    "p_holm": r.get("p_holm")})
    ax.axvline(0.0, color="#b71c1c", lw=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["channel"] for r in rows])
    ax.set_xlabel("P − B4, " + ctx.t("diff"))
    ax.set_title(title_of(ctx, "G07"))
    footnote(ctx, fig, f"{ctx.t('ci')}; сірий = інтервал містить нуль (різниця не "
                       f"встановлена); великий маркер = значуще після поправки Holm; "
                       f"{ctx.t('synthetic')}")
    return export(ctx, "G07", fig, out, {"comparison": "P - B4",
                                         "source_table": "paired_effects.csv"})


def _scene_metric(ctx: FigureContext, metric: str) -> Dict[Tuple[str, str, str], float]:
    """(channel, method, scene) -> scene mean, from the raw observations."""
    from collections import defaultdict

    rows = ctx.table("frames.csv")
    cells: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in rows:
        v = _float(r, metric)
        if np.isfinite(v):
            cells[(r.get("channel", ""), r.get("method", ""),
                   r.get("scene", ""))].append(v)
    return {k: float(np.mean(v)) for k, v in cells.items()}


@figure("G08")
def g08(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    vals = _scene_metric(ctx, "psnr_full")
    channels = sorted({k[0] for k in vals}, key=channel_key)
    fig, axes = plt.subplots(1, len(channels), figsize=(2.6 * len(channels), 3.0),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    out: List[Dict[str, Any]] = []
    for ax, ch in zip(axes, channels):
        scenes = sorted({k[2] for k in vals if k[0] == ch})
        xs = [vals.get((ch, "B4", s), float("nan")) for s in scenes]
        ys = [vals.get((ch, "P", s), float("nan")) for s in scenes]
        ax.scatter(xs, ys, s=14, color="#2e7d32", alpha=0.8)
        finite = [v for v in xs + ys if np.isfinite(v)]
        if finite:
            lim = (min(finite) - 1, max(finite) + 1)
            ax.plot(lim, lim, color="#b71c1c", lw=0.9)
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
        ax.set_title(ch, fontsize=8)
        ax.set_xlabel("B4")
        for s, a, b in zip(scenes, xs, ys):
            out.append({"channel": ch, "scene": s, "psnr_B4": a, "psnr_P": b,
                        "P_wins": bool(np.isfinite(a) and np.isfinite(b) and b > a)})
    axes[0].set_ylabel("P")
    fig.suptitle(title_of(ctx, "G08"), y=1.02)
    n_win = sum(1 for r in out if r["P_wins"])
    footnote(ctx, fig, f"кожна точка - одна сцена; над діагоналлю P кращий "
                       f"({n_win}/{len(out)}); {ctx.t('synthetic')}")
    return export(ctx, "G08", fig, out, {"metric": "psnr_full",
                                         "source_table": "frames.csv"})


@figure("G09")
def g09(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    frames = ctx.table("frames.csv")
    cats = _clip_categories(ctx)
    from collections import defaultdict

    cells: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    per_scene: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    for r in frames:
        v = _float(r, "psnr_full")
        if not np.isfinite(v) or r.get("method") not in ("P", "B4"):
            continue
        cat = cats.get(r.get("clip", ""), "other")
        per_scene[(cat, r.get("channel", ""), r.get("scene", ""),
                   r["method"])].append(v)
    diffs: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    scenes = {(k[0], k[1], k[2]) for k in per_scene}
    for cat, ch, sc in scenes:
        a = per_scene.get((cat, ch, sc, "P"))
        b = per_scene.get((cat, ch, sc, "B4"))
        if a and b:
            diffs[(cat, ch)].append(float(np.mean(a)) - float(np.mean(b)))
    categories = sorted({k[0] for k in diffs})
    channels = sorted({k[1] for k in diffs}, key=channel_key)
    grid = np.full((len(categories), len(channels)), np.nan)
    out: List[Dict[str, Any]] = []
    for i, cat in enumerate(categories):
        for j, ch in enumerate(channels):
            vals = diffs.get((cat, ch), [])
            if vals:
                grid[i, j] = float(np.mean(vals))
            out.append({"category": cat, "channel": ch,
                        "mean_P_minus_B4": grid[i, j], "n_scenes": len(vals)})
    fig, ax = plt.subplots(figsize=(1.3 * len(channels) + 3.0,
                                    0.55 * len(categories) + 2.2))
    vmax = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0
    im = ax.imshow(grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=20, ha="right")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    for i in range(len(categories)):
        for j in range(len(channels)):
            txt = "—" if not np.isfinite(grid[i, j]) else f"{grid[i, j]:+.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5)
    fig.colorbar(im, ax=ax, label="P − B4, " + ctx.t("diff"))
    ax.set_title(title_of(ctx, "G09"))
    ax.grid(False)
    footnote(ctx, fig, f"«—» = немає даних для клітинки; {ctx.t('synthetic')}")
    return export(ctx, "G09", fig, out, {"source_table": "frames.csv"})


def _clip_categories(ctx: FigureContext) -> Dict[str, str]:
    try:
        rows = ctx.table("dataset_manifest.csv")
    except FigurePending:
        return {}
    return {r["clip_id"]: r.get("category", "other") for r in rows}


@figure("G10")
def g10(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    vals = _scene_metric(ctx, "psnr_full")
    channels = sorted({k[0] for k in vals}, key=channel_key)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    out: List[Dict[str, Any]] = []
    for ch in channels:
        scenes = sorted({k[2] for k in vals if k[0] == ch})
        d = [vals[(ch, "P", s)] - vals[(ch, "B4", s)] for s in scenes
             if (ch, "P", s) in vals and (ch, "B4", s) in vals]
        if not d:
            continue
        d = np.sort(np.asarray(d))
        y = np.arange(1, d.size + 1) / d.size
        ax.step(d, y, where="post", label=f"{ch} (n={d.size})")
        for s, v in zip(scenes, d):
            out.append({"channel": ch, "scene": s, "diff_P_minus_B4": float(v)})
    ax.axvline(0.0, color="#b71c1c", lw=1.0)
    ax.set_xlabel("P − B4, " + ctx.t("diff"))
    ax.set_ylabel(ctx.t("ecdf"))
    ax.set_title(title_of(ctx, "G10"))
    ax.legend(fontsize=7.5)
    footnote(ctx, fig, f"ліворуч від червоної лінії P програє; {ctx.t('synthetic')}")
    return export(ctx, "G10", fig, out, {"source_table": "frames.csv"})


# =============================================================== E01 figures
@figure("G11")
def g11(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    rows = [r for r in ctx.table("e01_rate_quality.csv") if r.get("admissible") == "True"]
    if not rows:
        raise FigurePending("усі клітинки E01 недопустимі")
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    colors = {"raw": "#8d6e63", "dct": "#1976d2", "jpeg": "#2e7d32"}
    marks = {"1": "o", "2": "s", "4": "^"}
    out: List[Dict[str, Any]] = []
    for codec in sorted({r["codec"] for r in rows}):
        for nd in sorted({r["n_descriptions"] for r in rows}):
            sel = sorted(_by(rows, codec=codec, n_descriptions=nd),
                         key=lambda r: _float(r, "bits_per_frame"))
            if not sel:
                continue
            ax.plot([_float(r, "bits_per_frame") for r in sel],
                    [_float(r, "psnr_full") for r in sel],
                    marker=marks.get(str(nd), "."), ms=4, lw=1.0,
                    color=colors.get(codec, "#455a64"),
                    alpha=0.45 + 0.25 * int(nd == "1"),
                    label=f"{codec}, {nd} опис(и)")
            out.extend({k: r.get(k) for k in
                        ("codec", "n_descriptions", "stripe_height", "quality",
                         "bits_per_frame", "psnr_full", "ssim_full")} for r in sel)
    ax.set_xscale("log")
    ax.set_xlabel(ctx.t("bits"))
    ax.set_ylabel(ctx.t("psnr"))
    ax.set_title(title_of(ctx, "G11"))
    ax.legend(fontsize=7, ncol=2)
    n_bad = len(ctx.table("e01_rate_quality.csv")) - len(rows)
    footnote(ctx, fig, f"кодеки зіставлені за ФАКТИЧНИМ бітрейтом, не за номером "
                       f"quality; недопустимих конфігурацій: {n_bad}; чистий канал; "
                       f"{ctx.t('synthetic')}")
    return export(ctx, "G11", fig, out, {"source_table": "e01_rate_quality.csv"})


@figure("G12")
def g12(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    budgets = ctx.json("budgets.json")
    methods = list(budgets.get("methods", {}))
    if not methods:
        raise FigurePending("у budgets.json немає методів")
    labels = ["payload", "aead_tag", "header", "header_parity", "payload_parity"]
    colors = ["#2e7d32", "#f9a825", "#1976d2", "#4fc3f7", "#ba68c8"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    bottom = np.zeros(len(methods))
    out: List[Dict[str, Any]] = []
    for lab, col in zip(labels, colors):
        vals = np.array([budgets["methods"][m]["overhead_breakdown_per_unit"].get(lab, 0)
                         for m in methods], dtype=float)
        ax.bar(methods, vals, bottom=bottom, label=lab, color=col)
        bottom += vals
    for m in methods:
        d = budgets["methods"][m]
        row = {"method": m, **d["overhead_breakdown_per_unit"],
               "unit_wire_bytes": d["unit_wire_bytes"],
               "units_per_raster": d["units_per_raster"],
               "unused_symbols": d["unused_symbols"],
               "payload_efficiency": d["payload_efficiency"]}
        row.update({f"raster_{k}": v for k, v in
                    d["waterfall"]["raster_overhead"]["fractions"].items()})
        out.append(row)
    ax.set_ylabel("байти на одиницю")
    ax.set_title("Одиниця передавання")
    ax.legend(fontsize=7.5)

    frac_keys = ["data_cells", "pilots", "sync", "blanking", "grid_remainder"]
    bottom2 = np.zeros(len(methods))
    for k, col in zip(frac_keys, ["#2e7d32", "#f9a825", "#e57373", "#90a4ae", "#455a64"]):
        vals = np.array([budgets["methods"][m]["waterfall"]["raster_overhead"]
                         ["fractions"].get(k, 0.0) for m in methods], dtype=float)
        ax2.bar(methods, vals, bottom=bottom2, label=k, color=col)
        bottom2 += vals
    ax2.set_ylabel("частка пікселів растру")
    ax2.set_title("Растр: куди йде кожен піксель")
    ax2.legend(fontsize=7.5)
    fig.suptitle(title_of(ctx, "G12"), y=1.02)
    footnote(ctx, fig, "ліворуч - накладні витрати одиниці, праворуч - гасіння, "
                       "синхронізація, пілоти й залишок сітки")
    return export(ctx, "G12", fig, out, {"source_table": "budgets.json"})


# =============================================================== E03 figures
def _sweep_panel(ctx: FigureContext, gid: str, axis: str, xlabel: str,
                 metrics: Sequence[str] = ("psnr_full", "coverage")) -> Dict[str, Any]:
    plt = _plt()
    rows = [r for r in ctx.table("sweeps.csv") if r.get("axis") == axis]
    if not rows:
        raise FigurePending(f"у sweeps.csv немає осі {axis} - E03 для неї не запускалась")
    try:
        raw = next((r for r in ctx.table("sweeps_frames.csv")
                    if r.get("axis") == axis), {})
        for r in rows:
            r.setdefault("cofactors", raw.get("cofactors", ""))
    except FigurePending:
        pass
    methods = sorted({r["method"] for r in rows},
                     key=lambda m: list(METHOD_STYLE).index(m)
                     if m in METHOD_STYLE else 99)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 3.8))
    axes = np.atleast_1d(axes)
    out: List[Dict[str, Any]] = []
    for ax, metric in zip(axes, metrics):
        for m in methods:
            sel = sorted(_by(rows, method=m), key=lambda r: _float(r, "channel_value"))
            xs = [_float(r, "channel_value") for r in sel]
            ys = [_float(r, metric) for r in sel]
            sd = [_float(r, f"{metric}_sd", 0.0) for r in sel]
            n = [max(int(r.get("n_scenes") or 1), 1) for r in sel]
            err = [s / np.sqrt(k) * 1.96 for s, k in zip(sd, n)]
            st = style(m)
            ax.errorbar(xs, ys, yerr=err, marker=st["marker"], ls=st["ls"],
                        color=st["color"], ms=4, lw=1.2, capsize=2, label=m)
            for r in sel:
                out.append({"axis": axis, "value": _float(r, "channel_value"),
                            "method": m, metric: _float(r, metric),
                            "n_scenes": r.get("n_scenes")})
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ctx.t("psnr") if metric == "psnr_full"
                      else ctx.t("coverage") if metric == "coverage"
                      else metric)
    axes[0].legend(fontsize=7.5)
    fig.suptitle(title_of(ctx, gid), y=1.03)
    # Name the co-factor: an axis that needs another field switched on to have
    # any effect must say which, and at what value.
    fixed = ""
    for r in rows:
        raw = r.get("cofactors") or ""
        if raw and raw not in ("{}", "null"):
            try:
                d = json.loads(raw)
            except Exception:
                d = {}
            if d:
                fixed = "; зафіксовано " + ", ".join(f"{k}={v:g}" for k, v in
                                                     sorted(d.items()))
            break
    footnote(ctx, fig, f"база профілю mild{fixed}; вісь числова, не назва "
                       f"профілю; смуги - 95% CI за сценами; {ctx.t('synthetic')}")
    return export(ctx, gid, fig, out, {"axis": axis, "metrics": list(metrics),
                                       "source_table": "sweeps.csv"})


@figure("G17")
def g17(ctx: FigureContext) -> Dict[str, Any]:
    return _sweep_panel(ctx, "G17", "noise_sigma", ctx.t("noise"))


@figure("G18")
def g18(ctx: FigureContext) -> Dict[str, Any]:
    return _sweep_panel(ctx, "G18", "burst_len_lines", ctx.t("burst_len"))


@figure("G19")
def g19(ctx: FigureContext) -> Dict[str, Any]:
    return _sweep_panel(ctx, "G19", "burst_rate_per_frame", ctx.t("burst_rate"))


@figure("G20")
def g20(ctx: FigureContext) -> Dict[str, Any]:
    return _sweep_panel(ctx, "G20", "line_jitter_sigma", ctx.t("jitter"))


@figure("G21")
def g21(ctx: FigureContext) -> Dict[str, Any]:
    return _sweep_panel(ctx, "G21", "lowpass_sigma", ctx.t("lowpass"),
                        metrics=("symbol_errors_pre_fec", "psnr_full"))


@figure("G13")
def g13(ctx: FigureContext) -> Dict[str, Any]:
    """Authenticated payload bytes actually delivered per second of channel time."""
    plt = _plt()
    rows = [r for r in ctx.table("sweeps.csv") if r.get("axis") == "noise_sigma"]
    if not rows:
        raise FigurePending("немає осі noise_sigma у sweeps.csv")
    frames = [r for r in ctx.table("sweeps_frames.csv") if r.get("channel_axis")
              == "noise_sigma"]
    if not frames:
        raise FigurePending("немає покадрових рядків E03")
    from collections import defaultdict

    cells: Dict[Tuple[str, float, str], List[float]] = defaultdict(list)
    for r in frames:
        v = _float(r, "payload_bytes")
        cov = _float(r, "coverage")
        if not np.isfinite(v):
            continue
        auth = r.get("method") in ("B3", "B4", "P")
        cells[(r.get("method", ""), _float(r, "channel_value"),
               r.get("scene", ""))].append(v * (cov if auth else 0.0))
    per_point: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    for (m, val, _sc), vals in cells.items():
        per_point[(m, val)].append(float(np.mean(vals)))
    fps = 25.0 / 3.0
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    out: List[Dict[str, Any]] = []
    for m in sorted({k[0] for k in per_point}):
        pts = sorted((v, float(np.mean(per_point[(m, v)])))
                     for v in {k[1] for k in per_point if k[0] == m})
        st = style(m)
        ax.plot([p[0] for p in pts], [p[1] * 8 * fps for p in pts],
                marker=st["marker"], ls=st["ls"], color=st["color"], label=m)
        out.extend({"method": m, "noise_sigma": p[0],
                    "authenticated_bits_per_s": p[1] * 8 * fps} for p in pts)
    ax.set_xlabel(ctx.t("noise"))
    ax.set_ylabel("автентифіковані корисні біт/с")
    ax.set_title(title_of(ctx, "G13"))
    ax.legend(fontsize=7.5)
    footnote(ctx, fig, "для неавтентифікованих методів величина дорівнює нулю за "
                       "означенням - вони не доставляють автентифікованих даних")
    return export(ctx, "G13", fig, out, {"source_table": "sweeps_frames.csv"})


# =============================================================== E04 figures
def _interaction(ctx: FigureContext, gid: str, grid: str, ax_x: str, ax_y: str,
                 value: str, method: Optional[str] = None,
                 diff: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
    plt = _plt()
    rows = [r for r in ctx.table("interaction.csv") if r.get("grid") == grid]
    if not rows:
        raise FigurePending(f"у interaction.csv немає сітки {grid}")
    xs = sorted({_float(r, ax_x) for r in rows})
    ys = sorted({_float(r, ax_y) for r in rows})
    m1 = np.full((len(ys), len(xs)), np.nan)
    m2 = np.full((len(ys), len(xs)), np.nan)
    out: List[Dict[str, Any]] = []
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            cell = [r for r in rows if _float(r, ax_x) == xv and _float(r, ax_y) == yv]
            if diff:
                a = [r for r in cell if r["method"] == diff[0]]
                b = [r for r in cell if r["method"] == diff[1]]
                if a and b:
                    m1[i, j] = _float(a[0], value) - _float(b[0], value)
                    m2[i, j] = np.hypot(_float(a[0], f"{value}_sd", 0.0),
                                        _float(b[0], f"{value}_sd", 0.0))
            else:
                sel = [r for r in cell if r["method"] == method]
                if sel:
                    m1[i, j] = _float(sel[0], value)
                    m2[i, j] = _float(sel[0], f"{value}_sd", 0.0)
            out.append({ax_x: xv, ax_y: yv, "value": m1[i, j], "sd": m2[i, j],
                        "method": method or f"{diff[0]}-{diff[1]}"})
    n_panels = 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.0))
    cmap = "RdYlGn" if diff else "viridis"
    lim = float(np.nanmax(np.abs(m1))) if np.isfinite(m1).any() else 1.0
    kw = dict(vmin=-lim, vmax=lim) if diff else {}
    im0 = axes[0].imshow(m1, cmap=cmap, aspect="auto", origin="lower", **kw)
    axes[0].set_title(("P − B4, " + ctx.t("diff")) if diff else f"{method}: {value}")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(m2, cmap="magma", aspect="auto", origin="lower")
    axes[1].set_title("невизначеність (СКВ за сценами)")
    fig.colorbar(im1, ax=axes[1])
    for ax in axes:
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{v:g}" for v in xs], rotation=0)
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels([f"{v:g}" for v in ys])
        ax.set_xlabel(ctx.t(ax_x) if ax_x in L[ctx.lang] else ax_x)
        ax.set_ylabel(ctx.t(ax_y) if ax_y in L[ctx.lang] else ax_y)
        ax.grid(False)
    fig.suptitle(title_of(ctx, gid), y=1.03)
    footnote(ctx, fig, "різниця й невизначеність показані окремими панелями; "
                       "дрібна різниця з широким інтервалом не є перемогою")
    return export(ctx, gid, fig, out, {"grid": grid, "x": ax_x, "y": ax_y,
                                       "source_table": "interaction.csv"})


@figure("G22")
def g22(ctx: FigureContext) -> Dict[str, Any]:
    return _interaction(ctx, "G22", "gain_x_offset", "gain", "offset",
                        "coverage", method="P")


@figure("G23")
def g23(ctx: FigureContext) -> Dict[str, Any]:
    return _interaction(ctx, "G23", "noise_x_burst", "noise_sigma", "burst_len_lines",
                        "psnr_full", diff=("P", "B4"))


# =============================================================== E05 figures
@figure("G27")
def g27(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    geo = ctx.table("placement_geometry.csv")
    from avsec.interleaving import Interleaver, InterleaverConfig

    cfgs = [("sequential", dict(scheme="sequential", depth=1)),
            ("block-278", dict(scheme="block", depth=278)),
            ("diagonal-64", dict(scheme="diagonal", depth=64)),
            ("bawp-8", dict(scheme="bawp", depth=1, burst_rows=8))]
    n_rows, n_cols = 60, 40                      # a legible sub-grid
    fig, axes = plt.subplots(1, len(cfgs), figsize=(3.0 * len(cfgs), 3.4))
    out: List[Dict[str, Any]] = []
    for ax, (name, kw) in zip(axes, cfgs):
        il = Interleaver(InterleaverConfig(column_twist=1, **kw), n_rows, n_cols)
        n_units = max(1, il.max_units(300, 2, 8))
        cells = il.place([300] * n_units, [i % 2 for i in range(n_units)], 2)
        img = np.full((n_rows, n_cols), np.nan)
        for u, c in enumerate(cells):
            arr = np.asarray(c)
            img.flat[arr] = u
        ax.imshow(img, cmap="tab20", interpolation="nearest", aspect="auto")
        ax.set_title(f"{name}\n{n_units} units, accum={il.accumulation_rows}",
                     fontsize=8)
        ax.set_xlabel(ctx.t("col"))
        ax.grid(False)
        out.append({"scheme": name, "n_units": n_units,
                    "accumulation_rows": il.accumulation_rows,
                    "grid": f"{n_rows}x{n_cols}"})
    axes[0].set_ylabel(ctx.t("row"))
    fig.suptitle(title_of(ctx, "G27"), y=1.04)
    footnote(ctx, fig, f"демонстраційна сітка {n_rows}x{n_cols}; колір - номер "
                       f"одиниці; повний растр у codewords.csv")
    return export(ctx, "G27", fig, out, {"grid_rows": n_rows, "grid_cols": n_cols,
                                         "source_table": "placement_geometry.csv"})


@figure("G28")
def g28(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    rows = [r for r in ctx.table("codewords.csv") if r.get("admissible") == "True"
            and r.get("n_descriptions") == "2" and r.get("column_twist") in ("1", "0")]
    if not rows:
        raise FigurePending("немає допустимих рядків E05 для двох описів")
    schemes = sorted({r["scheme"] for r in rows})
    lens = sorted({_float(r, "burst_lines") for r in rows})
    grid = np.full((len(schemes), len(lens)), np.nan)
    out: List[Dict[str, Any]] = []
    for i, sch in enumerate(schemes):
        for j, bl in enumerate(lens):
            sel = [r for r in rows if r["scheme"] == sch and _float(r, "burst_lines") == bl]
            if sel:
                grid[i, j] = _float(sel[0], "worst_damaged_bytes")
                out.append({"scheme": sch, "burst_lines": bl,
                            "worst_damaged_bytes": grid[i, j],
                            "rs_nsym": sel[0].get("rs_nsym"),
                            "survives_as_erasures": sel[0].get("survives_as_erasures")})
    fig, ax = plt.subplots(figsize=(1.0 * len(lens) + 3.5, 0.45 * len(schemes) + 2.4))
    im = ax.imshow(grid, cmap="magma_r", aspect="auto")
    nsym = _float(rows[0], "rs_nsym")
    # Whether a unit survives is decided **per RS block**, not by comparing a
    # whole unit's damage with one block's parity budget (R03), so the verdict
    # is read from the sweep rather than recomputed from the total here.
    verdict = {(r["scheme"], _float(r, "burst_lines")):
               _float(r, "frac_positions_uncorrectable", 0.0) for r in rows}
    for i in range(len(schemes)):
        for j in range(len(lens)):
            if np.isfinite(grid[i, j]):
                ok = verdict.get((schemes[i], lens[j]), 0.0) <= 0.0
                ax.text(j, i, f"{int(grid[i, j])}", ha="center", va="center",
                        fontsize=7, color="#1b5e20" if ok else "#ffffff",
                        weight="bold" if ok else "normal")
    ax.set_xticks(range(len(lens)))
    ax.set_xticklabels([f"{int(v)}" for v in lens])
    ax.set_yticks(range(len(schemes)))
    ax.set_yticklabels(schemes)
    ax.set_xlabel(ctx.t("burst_len"))
    ax.set_ylabel("розміщення")
    ax.set_title(title_of(ctx, "G28"))
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=ctx.t("damaged"))
    spb = rows[0].get("symbols_per_byte", "?")
    nblk = rows[0].get("n_rs_blocks_per_unit", "?")
    footnote(ctx, fig, f"числа - пошкоджені БАЙТИ RS-слова ({spb} клітинки модема "
                       f"= 1 байт), розкладені по {nblk} RS-словах одиниці; "
                       f"зелений жирний = жодна позиція burst не робить одиницю "
                       f"невідновною (nsym={int(nsym)} на слово); два описи, "
                       f"вичерпний перебір початків burst")
    return export(ctx, "G28", fig, out, {"source_table": "codewords.csv"})


@figure("G29")
def g29(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    rows = [r for r in ctx.table("codewords.csv")
            if r.get("admissible") == "True" and _float(r, "claimed_bound_bytes") > 0]
    if not rows:
        raise FigurePending("немає рядків з аналітичною межею (лише BAWP її має)")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    out: List[Dict[str, Any]] = []
    for sch in sorted({r["scheme"] for r in rows}):
        sel = sorted([r for r in rows if r["scheme"] == sch
                      and r["n_descriptions"] == "2" and r["column_twist"] == "1"],
                     key=lambda r: _float(r, "burst_lines"))
        if not sel:
            continue
        xs = [_float(r, "burst_lines") for r in sel]
        ax.plot(xs, [_float(r, "worst_damaged_bytes") for r in sel], "o-",
                label=f"{sch}: виміряний максимум")
        ax.plot(xs, [_float(r, "claimed_bound_bytes") for r in sel], "s--",
                alpha=0.7, label=f"{sch}: заявлена межа")
        for r in sel:
            out.append({"scheme": sch, "burst_lines": _float(r, "burst_lines"),
                        "measured": _float(r, "worst_damaged_bytes"),
                        "claimed_bound": _float(r, "claimed_bound_bytes"),
                        "bound_holds": _float(r, "worst_damaged_bytes")
                                       <= _float(r, "claimed_bound_bytes")})
    ax.set_xlabel(ctx.t("burst_len"))
    ax.set_ylabel(ctx.t("damaged"))
    ax.set_title(title_of(ctx, "G29"))
    ax.legend(fontsize=7.5)
    holds = all(r["bound_holds"] for r in out)
    footnote(ctx, fig, ("межа виконується в усіх перевірених точках"
                        if holds else "МЕЖА ПОРУШЕНА - див. CSV")
             + "; вичерпний перебір початків burst")
    return export(ctx, "G29", fig, out, {"source_table": "codewords.csv",
                                         "bound_holds": holds})


@figure("G30")
def g30(ctx: FigureContext) -> Dict[str, Any]:
    """Touched versus actually lost, as two curves.

    The old figure plotted "both descriptions hit by the burst" and called it
    loss.  A description the FEC repaired was never lost, so the two are drawn
    separately here and the gap between them *is* the FEC doing its job (R04).
    """
    plt = _plt()
    rows = [r for r in ctx.table("joint_loss.csv") if r.get("n_descriptions") == "2"
            and r.get("column_twist") == "1"]
    if not rows:
        raise FigurePending("немає joint_loss.csv для двох описів")
    has_outcome = any(r.get("outcome") for r in rows)
    schemes = sorted({r["scheme"] for r in rows})
    lens = sorted({_float(r, "burst_lines") for r in rows})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4), sharey=True)
    out: List[Dict[str, Any]] = []
    colours = {sch: f"C{i % 10}" for i, sch in enumerate(schemes)}
    for sch in schemes:
        lost_ys, touched_ys = [], []
        for bl in lens:
            sel = [r for r in rows if r["scheme"] == sch
                   and _float(r, "burst_lines") == bl]
            lost = sum(_float(r, "fraction", 0.0) for r in sel
                       if r.get("stripe_lost") == "True")
            touched = sum(
                _float(r, "fraction", 0.0) for r in sel
                if r.get("outcome") == "touched"
                and _float(r, "descriptions_in_one_stripe", 0.0) >= 2)
            lost_ys.append(lost)
            touched_ys.append(touched)
            out.append({"scheme": sch, "burst_lines": bl,
                        "p_both_descriptions_touched": touched,
                        "p_both_descriptions_lost": lost})
        ax.plot(lens, touched_ys, "o--", ms=4, color=colours[sch], alpha=0.8,
                label=sch)
        ax2.plot(lens, lost_ys, "o-", ms=4, color=colours[sch], label=sch)
    ax.set_title("зачеплено обидва описи смуги")
    ax2.set_title("ВТРАЧЕНО обидва описи смуги")
    for a in (ax, ax2):
        a.set_xlabel(ctx.t("burst_len"))
        a.set_ylim(-0.03, 1.03)
    ax.set_ylabel("частка позицій burst")
    ax2.legend(fontsize=7, ncol=2)
    fig.suptitle(title_of(ctx, "G30"))
    # Several curves coincide exactly; saying so is the finding, not clutter.
    coincident: Dict[Tuple[float, ...], List[str]] = {}
    for sch in schemes:
        key = tuple(round(r["p_both_descriptions_lost"], 6) for r in out
                    if r["scheme"] == sch)
        coincident.setdefault(key, []).append(sch)
    groups = ["=".join(v) for v in coincident.values() if len(v) > 1]
    extra = ("; збіжні криві справа: " + "; ".join(groups)) if groups else ""
    note = ("ліворуч - буря дістала обидва описи; праворуч - жоден опис не "
            "вдалося використати після FEC. Різниця між панелями і є робота FEC"
            + extra)
    if not has_outcome:
        note = ("таблиця зі старого прогону не розрізняє «зачеплено» і "
                "«втрачено»; перерахуйте E05" + extra)
    footnote(ctx, fig, note)
    return export(ctx, "G30", fig, out, {"source_table": "joint_loss.csv"})


# =============================================================== E08 figures
@figure("G38")
def g38(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    data = ctx.json("protocol_checks.json")
    checks = data.get("checks") or data.get("results") or []
    if not checks:
        raise FigurePending("у protocol_checks.json немає перевірок")
    names = [c.get("name", f"check{i}") for i, c in enumerate(checks)]
    ok = [bool(c.get("passed", c.get("ok", False))) for c in checks]
    fig, ax = plt.subplots(figsize=(7.0, 0.22 * len(names) + 1.8))
    ax.barh(range(len(names)), [1] * len(names),
            color=["#2e7d32" if o else "#b71c1c" for o in ok])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_title(title_of(ctx, "G38"))
    ax.grid(False)
    out = [{"check": n, "passed": o,
            "expected": c.get("expected", ""), "observed": c.get("observed", "")}
           for n, o, c in zip(names, ok, checks)]
    footnote(ctx, fig, f"зелений = очікуваний статус отримано; "
                       f"{sum(ok)}/{len(ok)} пройдено")
    return export(ctx, "G38", fig, out, {"source_table": "protocol_checks.json"})


@figure("G35")
def g35(ctx: FigureContext) -> Dict[str, Any]:
    """Boundary-compatibility attack: does the picture come back from the seams?"""
    plt = _plt()
    res = _attack_results(ctx, "boundary_compatibility_reassembly")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    labels = [str(r.get("content", i)) for i, r in enumerate(res)]
    direct = [float(r["metrics"].get("direct_accuracy", 0.0)) for r in res]
    neigh = [float(r["metrics"].get("neighbour_accuracy", 0.0)) for r in res]
    psnr = [float(r["metrics"].get("reconstruction_psnr_db", float("nan")))
            for r in res]
    x = np.arange(len(labels))
    a1.bar(x - 0.2, direct, 0.4, label="точна позиція блока", color="#8d6e63")
    a1.bar(x + 0.2, neigh, 0.4, label="правильні сусіди", color="#e57373")
    a1.set_xticks(x)
    a1.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.5)
    a1.set_ylim(0, 1.02)
    a1.set_ylabel("частка")
    a1.legend(fontsize=7.5)
    a1.set_title("Точність складання")
    a2.bar(x, psnr, 0.6, color="#1976d2")
    a2.set_xticks(x)
    a2.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.5)
    a2.set_ylabel(ctx.t("psnr"))
    a2.set_title("Якість відновленої зловмисником картинки")
    fig.suptitle(title_of(ctx, "G35"), y=1.02)
    out = [{"content": l, "direct_accuracy": d, "neighbour_accuracy": n,
            "reconstruction_psnr_db": p, "n_blocks": r["metrics"].get("n_blocks"),
            "assumptions": r.get("assumptions", "")}
           for l, d, n, p, r in zip(labels, direct, neigh, psnr, res)]
    footnote(ctx, fig, "атака лише за шифротекстом і публічною сіткою; висока "
                       "точність сусідства означає, що перестановка не приховує "
                       "структуру, навіть коли точна позиція блока не вгадана")
    return export(ctx, "G35", fig, out, {"source_table": "attacks.json",
                                         "attack": "boundary_compatibility"})


@figure("G36")
def g36(ctx: FigureContext) -> Dict[str, Any]:
    """How many known or chosen frames does recovering the permutation take?"""
    plt = _plt()
    known = _attack_results(ctx, "known_plaintext_tile_matching")
    chosen = _attack_results(ctx, "chosen_plaintext_permutation_recovery",
                             required=False)
    multi = _attack_results(ctx, "multi_frame_variance_fingerprint",
                            required=False)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    out: List[Dict[str, Any]] = []
    groups = [("1 відома пара", known), ("1 обраний кадр", chosen),
              ("багатокадрова", multi)]
    xs, ys, cols = [], [], []
    for gi, (label, rs) in enumerate(groups):
        for r in rs:
            acc = float(r["metrics"].get("permutation_accuracy",
                        r["metrics"].get("accuracy", 0.0)))
            xs.append(gi + np.random.default_rng(gi).uniform(-0.12, 0.12))
            ys.append(acc)
            cols.append("#2e7d32" if r.get("success") else "#b71c1c")
            out.append({"attack": r["name"], "group": label,
                        "content": r.get("content", ""),
                        "permutation_accuracy": acc,
                        "n_frames_used": r["metrics"].get("n_frames_used", 1),
                        "seconds": r.get("seconds"),
                        "success": r.get("success"),
                        "assumptions": r.get("assumptions", "")})
    ax.scatter(xs, ys, c=cols, s=45)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("частка правильно відновлених блоків")
    ax.set_title(title_of(ctx, "G36"))
    footnote(ctx, fig, "зелений - атака досягла своєї мети. Одна відома пара вже "
                       "дає повне відновлення сталої перестановки")
    return export(ctx, "G36", fig, out, {"source_table": "attacks.json"})


@figure("G37")
def g37(ctx: FigureContext) -> Dict[str, Any]:
    """Measured brute-force time for the lab LFSR, extrapolation marked as such."""
    plt = _plt()
    res = _attack_results(ctx, "lfsr_seed_bruteforce")
    r = res[0]
    rate = float(r["metrics"].get("seeds_per_second", 0.0)) or 1.0
    measured_width = int(np.log2(float(r["metrics"].get("key_space", 65535)) + 1))
    widths = [8, 12, 16, 20, 24, 32]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    out: List[Dict[str, Any]] = []
    for w in widths:
        space = 2 ** w - 1
        secs = space / rate
        measured = w <= measured_width
        # hatched, not only coloured: a projection must still read as a
        # projection in greyscale or in print (R11)
        ax.bar(str(w), secs, color="#2e7d32" if measured else "#f9a825",
               hatch="" if measured else "//", edgecolor="#37474f", lw=0.7)
        out.append({"seed_width_bits": w, "key_space": space,
                    "seconds": secs if measured else float("nan"),
                    "projected_seconds": secs,
                    "measured": measured,
                    "measured_rate_seeds_per_s": rate,
                    "status": "виміряно" if measured else
                              "ПРОЄКЦІЯ за виміряною швидкістю, не запуск"})
    ax.set_yscale("log")
    ax.set_xlabel("ширина seed, біт")
    ax.set_ylabel("час перебору, с")
    ax.set_title(title_of(ctx, "G37"))
    footnote(ctx, fig,
             f"зелений - фактично виміряний перебір ({rate:.0f} seed/с, "
             f"{r['seconds']:.1f} с до знайдення); жовтий - ПРОЄКЦІЯ за тією ж "
             f"швидкістю, а не запуск. Це власна лабораторна реалізація LFSR, "
             f"а не оцінка стійкості довільного шифру")
    return export(ctx, "G37", fig, out,
                  {"source_table": "attacks.json",
                   "measured_width_bits": measured_width,
                   "note": "проєкції позначені окремо і не називаються зламом"})


def _attack_results(ctx: FigureContext, name: str, required: bool = True,
                    target: str = "B1") -> List[Dict[str, Any]]:
    """Attack rows for one attack against one target.

    Since B2 is attacked as well, a row now names its ``target``; the catalogue
    figures are about the 2021 scheme itself, so they keep to ``B1``.  Older
    runs have no ``target`` field and are treated as B1, which is what they were.
    """
    data = ctx.json("attacks.json")
    res = [r for r in data.get("results", [])
           if r.get("name") == name and r.get("target", "B1") == target]
    if not res and required:
        raise FigurePending(f"у attacks.json немає результатів '{name}'")
    return res


# =============================================================== E09 figures
@figure("G39")
def g39(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    rows = ctx.table("timings.csv")
    rows = sorted(rows, key=lambda r: -_float(r, "median_ms"))[:18]
    fig, ax = plt.subplots(figsize=(8.4, 0.32 * len(rows) + 1.8))
    y = np.arange(len(rows))
    ax.barh(y, [_float(r, "median_ms") for r in rows], color="#1976d2", label="median")
    ax.barh(y, [_float(r, "p95_ms") - _float(r, "median_ms") for r in rows],
            left=[_float(r, "median_ms") for r in rows], color="#90caf9", label="p95")
    ax.barh(y, [_float(r, "p99_ms") - _float(r, "p95_ms") for r in rows],
            left=[_float(r, "p95_ms") for r in rows], color="#e3f2fd", label="p99")
    ax.set_yticks(y)
    ax.set_yticklabels([r.get("stage", "") for r in rows], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel(ctx.t("ms"))
    ax.set_title(title_of(ctx, "G39"))
    ax.legend(fontsize=7.5)
    footnote(ctx, fig, "wall-clock симулятора; це НЕ віртуальний розклад передавання")
    return export(ctx, "G39", fig, list(rows), {"source_table": "timings.csv"})


@figure("G41")
def g41(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    b = ctx.json("budgets.json")
    methods = list(b.get("methods", {}))
    if not methods:
        raise FigurePending("у budgets.json немає методів")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    logical = [b["methods"][m]["memory"]["logical_total_kb"] for m in methods]
    limit = b["methods"][methods[0]]["profile"] and None
    a1.bar(methods, logical, color="#2e7d32")
    lim_kb = b.get("budget", {}).get("max_receiver_buffer_kb")
    if lim_kb:
        a1.axhline(float(lim_kb), color="#b71c1c", ls="--",
                   label=f"ліміт протоколу {lim_kb:g} кБ")
        a1.legend(fontsize=7.5)
    a1.set_ylabel("логічний буфер протоколу, кБ")
    a1.set_title("Що має буферизувати протокол")
    proc = [b["methods"][m]["memory"]["process_peak_mb"] for m in methods]
    model = [b["methods"][m]["memory"]["model_total_mb"] for m in methods]
    x = np.arange(len(methods))
    a2.bar(x - 0.2, proc, 0.4, label="пік процесу", color="#455a64")
    a2.bar(x + 0.2, model, 0.4, label="масиви симулятора", color="#90a4ae")
    a2.set_xticks(x)
    a2.set_xticklabels(methods)
    a2.set_ylabel(ctx.t("memory"))
    a2.set_title("Пам'ять процесу й моделі")
    a2.legend(fontsize=7.5)
    fig.suptitle(title_of(ctx, "G41"), y=1.02)
    out = [{"method": m, "logical_kb": lg, "process_peak_mb": p, "model_mb": mo,
            "process_method": b["methods"][m]["memory"]["process_method"]}
           for m, lg, p, mo in zip(methods, logical, proc, model)]
    footnote(ctx, fig, "три різні величини; ліміт 512 кБ стосується ЛИШЕ лівої "
                       "панелі й не є піковою пам'яттю процесу")
    return export(ctx, "G41", fig, out, {"source_table": "budgets.json"})


@figure("G43")
def g43(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    b = ctx.json("budgets.json")
    methods = list(b.get("methods", {}))
    if not methods:
        raise FigurePending("у budgets.json немає методів")
    m = "P" if "P" in methods else methods[0]
    events = b["methods"][m].get("latency_events_sample") or []
    if not events:
        raise FigurePending("у budgets.json немає подій розкладу")
    kinds = ["capture", "stripe_ready", "unit_ready", "tx_start", "tx_end",
             "rx_end", "assembled", "display"]
    colors = {"capture": "#455a64", "stripe_ready": "#8d6e63",
              "unit_ready": "#f9a825", "tx_start": "#1976d2", "tx_end": "#4fc3f7",
              "rx_end": "#7e57c2", "assembled": "#2e7d32", "display": "#b71c1c"}
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    out: List[Dict[str, Any]] = []
    for i, k in enumerate(kinds):
        ts = [float(e["t"]) for e in events if e.get("kind") == k]
        ax.scatter(ts, [i] * len(ts), s=22, color=colors.get(k, "#333"), label=k)
        out.extend({"kind": k, "t_s": t} for t in ts)
    lat = b["methods"][m]["latency"]["latency_s"]["mean"]
    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels(kinds, fontsize=7.5)
    ax.set_xlabel(ctx.t("time"))
    ax.set_title(title_of(ctx, "G43") + f" ({m})")
    footnote(ctx, fig, f"затримка = display − capture = {lat*1e3:.0f} мс, різниця "
                       f"позначок часу, а не сума тривалостей етапів; "
                       f"адитивна верхня межа "
                       f"{b['methods'][m]['latency']['additive_upper_bound']['virtual_total_s']*1e3:.0f} мс")
    return export(ctx, "G43", fig, out, {"method": m, "source_table": "budgets.json"})



# =============================================================== E02 examples
def _percentile_cases(ctx: FigureContext, metric: str = "psnr_full",
                      method: str = "P") -> List[Dict[str, Any]]:
    """Pick the 10th / 50th / 90th percentile case, never the best frame.

    Choosing the nicest frame of the proposed method is the classic way to make
    a qualitative figure lie, so the cases are selected by percentile of a
    declared metric and the percentile is printed on the figure.
    """
    rows = [r for r in ctx.table("frames.csv") if r.get("method") == method]
    vals = [(_float(r, metric), r) for r in rows]
    vals = [(v, r) for v, r in vals if np.isfinite(v)]
    if not vals:
        raise FigurePending(f"немає кадрів методу {method} з метрикою {metric}")
    vals.sort(key=lambda vr: vr[0])
    out = []
    for q in (10, 50, 90):
        idx = min(len(vals) - 1, int(round((q / 100.0) * (len(vals) - 1))))
        v, r = vals[idx]
        out.append({"percentile": q, "value": v, "row": r})
    return out


def _render_case(ctx: FigureContext, case: Dict[str, Any]
                 ) -> Dict[str, np.ndarray]:
    """Re-render one recorded case from its identifiers.

    This is not a new experiment: the clip, method, channel point and frame
    index all come from the run's own table, and the channel trace is rebuilt
    from the same seed, so the pictures are the ones that produced the numbers.
    """
    import dataclasses

    from avsec.channel import ChannelTrace
    from avsec.config import load_config
    from avsec.experiments import build_methods, build_sources

    r = case["row"]
    cfg_path = os.path.join(ctx.run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FigurePending("немає config.yaml поряд із результатами прогону")
    cfg = load_config(cfg_path)
    base = str(r.get("channel", "")).split("[")[0] or cfg.channel_preset
    sub = dataclasses.replace(cfg, methods=(r["method"],), channel_preset=base,
                              channel_overrides={})
    srcs = {s.name: s for s in build_sources(sub)}
    src = srcs.get(r.get("clip", ""))
    if src is None:
        raise FigurePending(f"кліп {r.get('clip')} відсутній у наборі джерел")
    method = build_methods(sub)[r["method"]]
    trace = ChannelTrace(seed=cfg.channel_seed_value,
                         scene=f"{r['clip']}|{r.get('channel','')}",
                         repetition=int(float(r.get("repetition", 0) or 0)),
                         profile=str(r.get("channel", "")),
                         rasters_per_frame=cfg.budget.rasters_per_frame)
    method.reset()
    target = int(float(r.get("frame_id", 0) or 0))
    res = None
    for fi in range(target + 1):
        res = method.process(src.frames[fi], fi, trace)
    return {"original": res.original, "tx": res.transmitted, "rx": res.received,
            "recon": res.reconstructed, "available": res.available,
            "stale": getattr(res, "stale", np.zeros_like(res.available))}


@figure("G01")
def g01(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    cases = _percentile_cases(ctx)
    fig, axes = plt.subplots(len(cases), 4, figsize=(11.0, 2.6 * len(cases)))
    axes = np.atleast_2d(axes)
    out: List[Dict[str, Any]] = []
    for i, case in enumerate(cases):
        imgs = _render_case(ctx, case)
        for j, (key, title) in enumerate((("original", "оригінал"),
                                          ("tx", "переданий растр"),
                                          ("rx", "прийнятий растр"),
                                          ("recon", "реконструкція"))):
            axes[i, j].imshow(imgs[key], cmap="gray", vmin=0, vmax=255)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].grid(False)
            if i == 0:
                axes[i, j].set_title(title, fontsize=8)
        r = case["row"]
        axes[i, 0].set_ylabel(f"p{case['percentile']}\n{r.get('clip','')[:16]}\n"
                              f"{r.get('channel','')}", fontsize=6.5)
        out.append({"percentile": case["percentile"], "psnr_full": case["value"],
                    "clip": r.get("clip"), "channel": r.get("channel"),
                    "method": r.get("method"), "frame_id": r.get("frame_id")})
    fig.suptitle(title_of(ctx, "G01"), y=1.005)
    footnote(ctx, fig, "випадки обрані за 10/50/90-м процентилем PSNR методу P, "
                       "а не як найкращий кадр; ті самі сцени для всіх панелей")
    return export(ctx, "G01", fig, out, {"selection": "percentiles 10/50/90",
                                         "source_table": "frames.csv"})


@figure("G02")
def g02(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    from avsec.evaluation import availability_overlay

    cases = _percentile_cases(ctx)
    fig, axes = plt.subplots(1, len(cases), figsize=(3.4 * len(cases), 3.2))
    axes = np.atleast_1d(axes)
    out: List[Dict[str, Any]] = []
    for ax, case in zip(axes, cases):
        imgs = _render_case(ctx, case)
        ov = availability_overlay(imgs["recon"], imgs["available"], imgs["stale"])
        ax.imshow(ov[..., ::-1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        cov = float(imgs["available"].mean())
        stale = float(imgs["stale"].mean())
        ax.set_title(f"p{case['percentile']}  покриття {cov:.2f}", fontsize=8)
        out.append({"percentile": case["percentile"], "coverage": cov,
                    "stale_fraction": stale,
                    "estimated_fraction": 1.0 - cov - stale,
                    "clip": case["row"].get("clip"),
                    "channel": case["row"].get("channel")})
    fig.suptitle(title_of(ctx, "G02"), y=1.02)
    footnote(ctx, fig, "зелений - поточне перевірене; бурштиновий - старе; "
                       "червоний - домальоване. Старі й домальовані ділянки НЕ "
                       "входять до поточного перевіреного покриття")
    return export(ctx, "G02", fig, out, {"source_table": "frames.csv"})


# =============================================================== E09 Pareto
def _pareto(points: Sequence[Tuple[float, float]], maximise_y: bool = True
            ) -> List[int]:
    """Indices on the Pareto front: smaller x, larger y."""
    order = sorted(range(len(points)), key=lambda i: points[i][0])
    front, best = [], -np.inf
    for i in order:
        y = points[i][1]
        if not np.isfinite(y):
            continue
        if y > best:
            best = y
            front.append(i)
    return front


def _pareto_figure(ctx: FigureContext, gid: str, xkey: str, xlabel: str,
                   note: str) -> Dict[str, Any]:
    plt = _plt()
    b = ctx.json("budgets.json")
    quality = _summary_rows(ctx, "psnr_full")
    methods = [m for m in b.get("methods", {}) if any(
        r["method"] == m for r in quality)]
    if not methods:
        raise FigurePending("немає методів, спільних для budgets.json і summary.csv")
    channels = sorted({r["channel"] for r in quality}, key=channel_key)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    out: List[Dict[str, Any]] = []
    for ch in channels:
        pts, labels = [], []
        for m in methods:
            sel = _by(quality, channel=ch, method=m)
            if not sel:
                continue
            x = _xvalue(b["methods"][m], xkey)
            y = _float(sel[0], "mean")
            pts.append((x, y))
            labels.append(m)
            out.append({"channel": ch, "method": m, xkey: x, "psnr_full": y,
                        "lo": _float(sel[0], "lo"), "hi": _float(sel[0], "hi")})
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "o", ms=5, alpha=0.75, label=ch)
        front = _pareto(pts)
        if len(front) > 1:
            ax.plot([pts[i][0] for i in front], [pts[i][1] for i in front],
                    "-", lw=1.0, alpha=0.5)
        for (x, y), lab in zip(pts, labels):
            ax.annotate(lab, (x, y), fontsize=6, xytext=(3, 3),
                        textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ctx.t("psnr"))
    ax.set_title(title_of(ctx, gid))
    ax.legend(fontsize=7.5, title=ctx.t("channel"), title_fontsize=7)
    footnote(ctx, fig, note)
    return export(ctx, gid, fig, out, {"x": xkey, "source_table": "budgets.json"})


def _xvalue(entry: Dict[str, Any], key: str) -> float:
    if key == "payload_bitrate_bps":
        return float(entry.get("payload_bitrate_bps", float("nan")))
    if key == "latency_s":
        return float(entry["latency"]["latency_s"]["mean"])
    if key == "logical_kb":
        return float(entry["memory"]["logical_total_kb"])
    return float("nan")


@figure("G14")
def g14(ctx: FigureContext) -> Dict[str, Any]:
    return _pareto_figure(ctx, "G14", "payload_bitrate_bps",
                          ctx.t("rate"),
                          "фактична корисна швидкість профілю за однакової "
                          "зайнятості каналу; підпис - метод")


@figure("G15")
def g15(ctx: FigureContext) -> Dict[str, Any]:
    return _pareto_figure(ctx, "G15", "latency_s", ctx.t("latency"),
                          "затримка з подієвого розкладу (різниця позначок часу), "
                          "однаковий ресурс каналу для всіх точок")


@figure("G16")
def g16(ctx: FigureContext) -> Dict[str, Any]:
    return _pareto_figure(ctx, "G16", "logical_kb", ctx.t("buffer"),
                          "логічний буфер протоколу; пікова пам'ять процесу й "
                          "масиви симулятора - на G41, це різні величини")


@figure("G40")
def g40(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    rows = [r for r in ctx.table("scaling.csv") if r.get("admissible") == "True"]
    if not rows:
        raise FigurePending("немає допустимих рядків у scaling.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    out: List[Dict[str, Any]] = []
    for m in sorted({r["method"] for r in rows}):
        sel = sorted(_by(rows, method=m), key=lambda r: _float(r, "pixels"))
        px = [_float(r, "pixels") for r in sel]
        st = style(m)
        a1.plot(px, [_float(r, "frames_per_s") for r in sel], marker=st["marker"],
                ls=st["ls"], color=st["color"], label=m)
        a2.plot(px, [_float(r, "median_ms") / max(_float(r, "pixels"), 1) * 1e3
                     for r in sel], marker=st["marker"], ls=st["ls"],
                color=st["color"], label=m)
        out.extend({"method": m, "resolution": r.get("resolution"),
                    "pixels": _float(r, "pixels"),
                    "frames_per_s": _float(r, "frames_per_s"),
                    "median_ms": _float(r, "median_ms")} for r in sel)
    for ax, ylab in ((a1, "оброблених кадрів/с"), (a2, "мкс на піксель")):
        ax.set_xlabel("пікселів у кадрі")
        ax.set_ylabel(ylab)
        ax.set_xscale("log")
        ax.set_yscale("log")
    a1.legend(fontsize=7.5)
    fig.suptitle(title_of(ctx, "G40"), y=1.02)
    footnote(ctx, fig, "wall-clock на ПК, один процес, warm-up відкинуто; "
                       "це не вимір на Raspberry Pi 5")
    return export(ctx, "G40", fig, out, {"source_table": "scaling.csv"})


# =============================================================== E07 dynamics
@figure("G25")
def g25(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    rows = ctx.table("dynamics.csv")
    scen = sorted({r["scenario"] for r in rows})
    fig, axes = plt.subplots(2, len(scen), figsize=(4.2 * len(scen), 5.4),
                             sharex=True)
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:
        axes = axes.T
    out: List[Dict[str, Any]] = []
    for j, sc in enumerate(scen):
        sub = [r for r in rows if r["scenario"] == sc]
        impaired = sorted({_float(r, "t_frame") for r in sub
                           if r.get("impaired") in ("True", True)})
        for metric, ax in (("psnr_full", axes[0, j]), ("coverage", axes[1, j])):
            for m in sorted({r["method"] for r in sub}):
                pts: Dict[float, List[float]] = {}
                for r in _by(sub, method=m):
                    v = _float(r, metric)
                    if np.isfinite(v):
                        pts.setdefault(_float(r, "t_frame"), []).append(v)
                xs = sorted(pts)
                ys = [float(np.mean(pts[x])) for x in xs]
                st = style(m)
                ax.plot(xs, ys, color=st["color"], lw=1.2, label=m)
                out.extend({"scenario": sc, "method": m, "t_frame": x,
                            metric: y} for x, y in zip(xs, ys))
            if impaired:
                ax.axvspan(min(impaired), max(impaired), color="#ffcdd2", alpha=0.5,
                           zorder=0)
            ax.set_ylabel(ctx.t("psnr") if metric == "psnr_full" else ctx.t("coverage"))
        axes[0, j].set_title(sc, fontsize=8)
        axes[1, j].set_xlabel("кадр")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title_of(ctx, "G25"), y=1.01)
    footnote(ctx, fig, "рожеві смуги - інтервали завад, задані наперед")
    return export(ctx, "G25", fig, out, {"source_table": "dynamics.csv"})


@figure("G24")
def g24(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    rows = ctx.table("recovery.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    out: List[Dict[str, Any]] = []
    for m in sorted({r["method"] for r in rows}):
        sel = _by(rows, method=m)
        vals = [int(float(r["recovery_frames"])) for r in sel
                if r.get("censored") != "True"]
        n_all = len(sel)
        n_cens = sum(1 for r in sel if r.get("censored") == "True")
        if n_all == 0:
            continue
        xs = sorted(set(vals)) or [0]
        ys = [sum(1 for v in vals if v <= x) / n_all for x in xs]
        st = style(m)
        ax.step([0] + xs, [0] + ys, where="post", color=st["color"],
                label=f"{m} (цензуровано {n_cens}/{n_all})")
        out.extend({"method": m, "recovery_frames": x, "fraction_recovered": y,
                    "n_runs": n_all, "n_censored": n_cens}
                   for x, y in zip(xs, ys))
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.5, max(2.0, ax.get_xlim()[1]))
    ax.set_xlabel("кадрів після кінця завади")
    ax.set_ylabel("частка запусків, що відновились")
    ax.set_title(title_of(ctx, "G24"))
    ax.legend(fontsize=7.5)
    all_zero = out and all(r["recovery_frames"] == 0 for r in out)
    extra = (". Усі відновлення відбуваються вже на ПЕРШОМУ показаному кадрі "
             "після завади: одиниці захищені незалежно, міжкадрового "
             "передбачення немає, тож «хвоста» відновлення не існує. Роздільна "
             "здатність вимірювання - один кадр (120 мс); швидше за це метод "
             "нічого не розрізняє" if all_zero else "")
    footnote(ctx, fig, "крива не досягає 1, якщо частина запусків не відновилась "
                       "до кінця кліпу; такі випадки НЕ вилучені" + extra)
    return export(ctx, "G24", fig, out, {"source_table": "recovery.csv"})


@figure("G26")
def g26(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    rows = ctx.table("dynamics.csv")
    metrics = (("max_age_frames", ctx.t("age")),
               ("stale_fraction", "частка старих ділянок"),
               ("units_verified", "перевірених одиниць на кадр"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 3.6))
    axes = np.atleast_1d(axes)
    out: List[Dict[str, Any]] = []
    sc = sorted({r["scenario"] for r in rows})[0]
    sub = [r for r in rows if r["scenario"] == sc]
    impaired = sorted({_float(r, "t_frame") for r in sub
                       if r.get("impaired") in ("True", True)})
    for ax, (metric, label) in zip(axes, metrics):
        for m in sorted({r["method"] for r in sub}):
            pts: Dict[float, List[float]] = {}
            for r in _by(sub, method=m):
                v = _float(r, metric)
                if np.isfinite(v):
                    pts.setdefault(_float(r, "t_frame"), []).append(v)
            xs = sorted(pts)
            ys = [float(np.mean(pts[x])) for x in xs]
            st = style(m)
            ax.plot(xs, ys, color=st["color"], lw=1.2, label=m)
            out.extend({"scenario": sc, "method": m, "t_frame": x, metric: y}
                       for x, y in zip(xs, ys))
        if impaired:
            ax.axvspan(min(impaired), max(impaired), color="#ffcdd2", alpha=0.5,
                       zorder=0)
        ax.set_xlabel("кадр")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=7)
    fig.suptitle(title_of(ctx, "G26") + f" ({sc})", y=1.02)
    footnote(ctx, fig, "вік показаних даних - скільки кадрів тому вони були "
                       "перевірені; старі ділянки не входять у поточне покриття")
    return export(ctx, "G26", fig, out, {"scenario": sc,
                                         "source_table": "dynamics.csv"})


# =============================================================== E06 figures
@figure("G31")
def g31(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    rows = [r for r in ctx.table("ablations.csv") if r.get("study") == "factorial"]
    if not rows:
        raise FigurePending("у ablations.csv немає факторного дослідження")
    descs = sorted({r["n_descriptions"] for r in rows}, key=lambda v: int(v))
    places = sorted({r["placement"] for r in rows})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    x = np.arange(len(descs))
    out: List[Dict[str, Any]] = []
    for i, pl in enumerate(places):
        for ax, key, lab in ((a1, "psnr_full", ctx.t("psnr")),
                             (a2, "coverage", ctx.t("coverage"))):
            vals, errs = [], []
            for d in descs:
                sel = [r for r in rows if r["n_descriptions"] == d
                       and r["placement"] == pl]
                ok = sel and sel[0].get("admissible") == "True"
                vals.append(_float(sel[0], key) if ok else float("nan"))
                n = max(int(float(sel[0].get("n_scenes") or 1)), 1) if ok else 1
                errs.append(_float(sel[0], "psnr_sd", 0.0) / np.sqrt(n) * 1.96
                            if ok and key == "psnr_full" else 0.0)
            ax.bar(x + i * 0.35 - 0.18, vals, 0.32, yerr=errs, capsize=2,
                   label=pl, color=["#1976d2", "#2e7d32"][i % 2])
            ax.set_xticks(x)
            ax.set_xticklabels([f"{d} опис(и)" for d in descs])
            ax.set_ylabel(lab)
        for d in descs:
            sel = [r for r in rows if r["n_descriptions"] == d and r["placement"] == pl]
            if sel:
                out.append({"n_descriptions": d, "placement": pl,
                            "admissible": sel[0].get("admissible"),
                            "reason": sel[0].get("reason", ""),
                            "psnr_full": _float(sel[0], "psnr_full"),
                            "coverage": _float(sel[0], "coverage"),
                            "n_scenes": sel[0].get("n_scenes")})
    a1.legend(fontsize=7.5)
    fig.suptitle(title_of(ctx, "G31"), y=1.02)
    footnote(ctx, fig, "дослідження МЕХАНІЗМУ: решта транспорту зафіксована, "
                       "жодна клітинка не переналаштовувалась")
    return export(ctx, "G31", fig, out, {"source_table": "ablations.csv"})


@figure("G32")
def g32(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    rows = [r for r in ctx.table("ablations.csv") if r.get("study") == "payload_x_fec"]
    if not rows:
        raise FigurePending("у ablations.csv немає сітки payload x nsym")
    pls = sorted({int(float(r["max_unit_payload"])) for r in rows})
    nss = sorted({int(float(r["fec_nsym"])) for r in rows})
    grid = np.full((len(nss), len(pls)), np.nan)
    mask = np.zeros_like(grid, dtype=bool)
    out: List[Dict[str, Any]] = []
    for i, ns in enumerate(nss):
        for j, pl in enumerate(pls):
            sel = [r for r in rows if int(float(r["fec_nsym"])) == ns
                   and int(float(r["max_unit_payload"])) == pl]
            ok = bool(sel) and sel[0].get("admissible") == "True"
            if ok:
                grid[i, j] = _float(sel[0], "psnr_full")
            else:
                mask[i, j] = True
            out.append({"max_unit_payload": pl, "fec_nsym": ns,
                        "rs_k": 255 - ns, "admissible": ok,
                        "reason": sel[0].get("reason", "") if sel else "не запускалось",
                        "psnr_full": grid[i, j]})
    fig, ax = plt.subplots(figsize=(1.1 * len(pls) + 3.2, 0.6 * len(nss) + 2.4))
    im = ax.imshow(grid, cmap="viridis", aspect="auto", origin="lower")
    for i in range(len(nss)):
        for j in range(len(pls)):
            if mask[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           color="#9e9e9e"))
                ax.text(j, i, "×", ha="center", va="center", fontsize=9,
                        color="white")
            elif np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                        fontsize=7, color="white")
    ax.set_xticks(range(len(pls)))
    ax.set_xticklabels([str(v) for v in pls])
    ax.set_yticks(range(len(nss)))
    ax.set_yticklabels([f"nsym={v}\nk={255-v}" for v in nss], fontsize=7)
    ax.set_xlabel("корисне навантаження одиниці, Б")
    ax.set_title(title_of(ctx, "G32"))
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=ctx.t("psnr"))
    footnote(ctx, fig, "сірі клітинки з «×» недопустимі за спільним бюджетом; "
                       "причина в CSV. Для кожного nsym перераховано k і "
                       "блокування RS")
    return export(ctx, "G32", fig, out, {"source_table": "ablations.csv"})


@figure("G33")
def g33(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    rows = [r for r in ctx.table("ablations.csv") if r.get("study") == "window"]
    if not rows:
        raise FigurePending("у ablations.csv немає дослідження вікна")
    rows = sorted(rows, key=lambda r: (_float(r, "window_rows") or 1e9))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    xs = [int(_float(r, "window_rows", 0)) for r in rows]
    labels = [str(x) if x else "повний растр" for x in xs]
    ok = [r.get("admissible") == "True" for r in rows]
    ys = [_float(r, "psnr_full") if o else float("nan") for r, o in zip(rows, ok)]
    ax.plot(range(len(xs)), ys, "o-", color="#2e7d32", label=ctx.t("psnr"))
    # An inadmissible window is a result, not a gap: mark it and say why.
    for i, (o, r) in enumerate(zip(ok, rows)):
        if not o:
            ax.axvspan(i - 0.4, i + 0.4, color="#9e9e9e", alpha=0.28, zorder=0)
            ax.annotate("недопустимо", (i, 0.5), xycoords=("data", "axes fraction"),
                        rotation=90, ha="center", va="center", fontsize=7,
                        color="#455a64")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("висота вікна розміщення, рядки символів")
    ax.set_ylabel(ctx.t("psnr"))
    ax2 = ax.twinx()
    # measured by the experiment, not recomputed here from the label
    lat = [_float(r, "window_accumulation_ms") for r in rows]
    ax2.plot(range(len(xs)), lat, "s--", color="#1976d2",
             label="накопичення вікна")
    ax2.set_ylabel("накопичення вікна, мс")
    ax2.grid(False)
    ax.set_title(title_of(ctx, "G33"))
    out = [{"window_rows": x, "psnr_full": y, "window_accumulation_ms": l,
            "accumulation_rows": _float(r, "accumulation_rows"),
            "admissible": r.get("admissible") == "True",
            "reason": r.get("reason", "")}
           for x, y, l, r in zip(xs, ys, lat, rows)]
    n_bad = sum(1 for r in out if not r["admissible"])
    footnote(ctx, fig,
             f"сірі смуги - недопустимі вікна ({n_bad} з {len(out)}): за двох "
             f"описів одиниця не вміщується у вікно такої висоти, тому BAWP тут "
             f"вимагає буферизації всього растру і НЕ дає виграшу за затримкою "
             f"перед глибоким блочним перемежуванням. Затримка на графіку - лише "
             f"внесок вікна; повна наскрізна береться з подієвого розкладу (G43)")
    return export(ctx, "G33", fig, out, {"source_table": "ablations.csv"})


@figure("G34")
def g34(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    rows = [r for r in ctx.table("ablations.csv") if r.get("study") == "fill_policy"]
    if not rows:
        raise FigurePending("у ablations.csv немає порівняння політик заповнення")
    fills = [r["fill"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    keys = (("psnr_full", ctx.t("psnr")), ("coverage", ctx.t("coverage")),
            ("max_age_frames", ctx.t("age")))
    out: List[Dict[str, Any]] = []
    for ax, (k, lab) in zip(axes, keys):
        ax.bar(fills, [_float(r, k) for r in rows], color="#1976d2")
        ax.set_ylabel(lab)
    for r in rows:
        out.append({"fill": r["fill"], "psnr_full": _float(r, "psnr_full"),
                    "coverage": _float(r, "coverage"),
                    "max_age_frames": _float(r, "max_age_frames")})
    fig.suptitle(title_of(ctx, "G34"), y=1.02)
    footnote(ctx, fig, "покриття тут - лише поточні перевірені пікселі: старі й "
                       "домальовані ділянки до нього не входять за жодної політики")
    return export(ctx, "G34", fig, out, {"source_table": "ablations.csv"})


# =============================================================== E10 figure
@figure("G42")
def g42(ctx: FigureContext) -> Dict[str, Any]:
    plt = _plt()
    data = ctx.json("cvbs.json")
    rows = data.get("rows") or data.get("results") or []
    if not rows:
        raise FigurePending("у cvbs.json немає рядків рівня B")
    lvlA = _summary_rows(ctx, "psnr_full")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    out: List[Dict[str, Any]] = []
    labels, a_vals, b_vals = [], [], []
    for r in rows:
        method = str(r.get("method", "P"))
        cond = str(r.get("channel", r.get("preset", "")))
        b = float(r.get("psnr_full", r.get("psnr", float("nan"))))
        sel = _by(lvlA, channel=cond, method=method)
        a = _float(sel[0], "mean") if sel else float("nan")
        labels.append(f"{method}@{cond}")
        a_vals.append(a)
        b_vals.append(b)
        out.append({"method": method, "condition": cond, "level_A_psnr": a,
                    "level_B_psnr": b,
                    "mapped": bool(sel),
                    "note": "" if sel else "немає зіставної умови на рівні A"})
    x = np.arange(len(labels))
    ax.bar(x - 0.2, a_vals, 0.4, label="рівень A", color="#1976d2")
    ax.bar(x + 0.2, b_vals, 0.4, label="рівень B (CVBS)", color="#2e7d32")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel(ctx.t("psnr"))
    ax.set_title(title_of(ctx, "G42"))
    ax.legend(fontsize=7.5)
    n_unmapped = sum(1 for r in out if not r["mapped"])
    footnote(ctx, fig,
             "конфігурації зафіксовані на рівні A і НЕ переналаштовувались на "
             "рівні B. Однакова назва профілю на двох рівнях не означає однаковий "
             "канал: це різні фізичні моделі, і стовпці поруч показують поведінку "
             "за однаковою НАЗВОЮ умови, а не за однаковою умовою. "
             f"Рівень A — {ctx._n_scenes_hint} сцен × 5 повторів × 16 кадрів; "
             "рівень B — ОДИН кадр однієї сцени на клітинку (демонстрація "
             "перенесення, не статистика). Без зіставлення: "
             f"{n_unmapped}")
    return export(ctx, "G42", fig, out, {"source_table": "cvbs.json"})


# ----------------------------------------------------------------- building
def build(ctx: FigureContext, ids: Optional[Sequence[str]] = None
          ) -> List[Dict[str, Any]]:
    """Build what can be built; record the rest as pending with a reason."""
    import avsec.figures_key  # noqa: F401  (registers K01-K10)

    from avsec.program import KEY

    ensure_dir(ctx.out_dir)
    reasons: Dict[str, str] = {}
    wanted = list(ids or ([f.gid for f in KEY] + [f.gid for f in CATALOGUE]))
    for gid in wanted:
        builder = BUILDERS.get(gid)
        if builder is None:
            reasons[gid] = "будівник рисунка ще не реалізований у цьому прогоні"
            continue
        try:
            builder(ctx)
        except FigurePending as exc:
            reasons[gid] = str(exc)
        except Exception as exc:                    # a broken builder is a defect
            reasons[gid] = f"помилка побудови: {type(exc).__name__}: {exc}"
    rows = catalogue_status(ctx.out_dir, reasons)
    write_csv(os.path.join(ctx.out_dir, "figure_index.csv"), rows)
    return rows


__all__ = ["DPI", "METHOD_STYLE", "CHANNEL_ORDER", "FigurePending", "FigureContext",
           "BUILDERS", "figure", "build", "export", "style"]
