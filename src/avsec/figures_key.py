"""The key figures: K01-K13.

The G01-G43 catalogue is complete by construction - one number per research
question - but completeness is not the same as saying something.  Several of
those figures are bookkeeping: a bar chart of four values, or a flat line that
needs a paragraph to explain why it is flat.

These ten are the ones that carry the argument.  Each answers one question, is
readable at README width, and states its own caveat on the figure rather than in
a caption somewhere else.  They are built from exactly the same tables as the
catalogue - no figure here runs an experiment or recomputes a statistic.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from avsec.figures import (
    CHANNEL_ORDER,
    FigureContext,
    FigurePending,
    _by,
    _float,
    _plt,
    channel_key,
    export,
    figure,
    footnote,
    style,
)

#: One line each, so a reader who only looks at the figures still gets the point.
KEY_FIGURES: Tuple[Tuple[str, str, str], ...] = (
    ("K01", "Захист і якість за методами",
     "захист і якість поруч: висока якість без автентифікації не є захистом"),
    ("K02", "Точка перелому", "де саме запропонована схема починає вигравати"),
    ("K03", "Карта режимів", "хто виграє за якої комбінації шуму й пакетів"),
    ("K04", "Компроміс схем розміщення",
     "пошкодження кодового слова проти втрати обох описів"),
    ("K05", "Джерела виміряної переваги", "внесок кожного кроку від B4 до P"),
    ("K06", "Ціна стиснення", "фактичний бітрейт проти якості, межа каналу"),
    ("K07", "Наскрізний приклад", "реальний кадр з дрона через увесь тракт"),
    ("K08", "Склад показаного зображення",
     "перевірене, старе й домальоване на одному кадрі"),
    ("K09", "Бюджет, затримка, пам'ять", "куди йде кожен біт і кожна мілісекунда"),
    ("K10", "Атаки на реконструкцію B1 зі статичною перестановкою",
     "одна відома пара кадрів відновлює перестановку"),
    ("K11", "Залежність результату від порядку кроків",
     "розклад у двох порядках дає різний внесок механізмів"),
    ("K12", "24 незалежні джерела",
     "інтервал по сценах і по вихідних записах - це різні твердження"),
    ("K13", "Карта області працездатності",
     "довжина x частота пакетів, а не один поріг"),
)

OK = "#2e7d32"
BAD = "#b71c1c"
WARN = "#f9a825"
NEUTRAL = "#90a4ae"


def _annot(ax, text: str, xy, xytext, color: str = "#37474f") -> None:
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=7.5, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=0.6))


# ============================================================== K01
@figure("K01")
def k01(ctx: FigureContext) -> Dict[str, Any]:
    """Security properties and quality in one grid.

    The 2021 baselines look respectable on a quality chart and provide nothing.
    Putting both on one figure is the only way to stop a reader taking the
    quality number as a security number.
    """
    from avsec.experiments import SECURITY_PROPERTIES

    plt = _plt()
    rows = [r for r in ctx.table("summary.csv") if r.get("metric") == "psnr_full"]
    cov = {(r["channel"], r["method"]): _float(r, "mean")
           for r in ctx.table("summary.csv") if r.get("metric") == "coverage"}
    if not rows:
        raise FigurePending("немає summary.csv")
    methods = [m for m in ("B0a", "B0a-R", "B1", "B2", "B0d", "B3", "B4", "P")
               if any(r["method"] == m for r in rows)]
    ch = "bursty" if any(r["channel"] == "bursty" for r in rows) else rows[0]["channel"]

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.6, 4.4),
                                 gridspec_kw={"width_ratios": [1.15, 1.0]})
    # ---- left: what is actually protected ------------------------------
    props = ["конфіденційність", "автентифікація", "цілісність"]
    keys = ["confidentiality", "authentication", "integrity"]
    grid = np.zeros((len(methods), len(props)))
    labels = np.empty((len(methods), len(props)), dtype=object)
    for i, m in enumerate(methods):
        p = SECURITY_PROPERTIES.get(m, {})
        for j, k in enumerate(keys):
            v = str(p.get(k, "none"))
            if v.startswith("AEAD") or "128-bit" in v:
                grid[i, j], labels[i, j] = 2, "AEAD"
            elif v.startswith("none"):
                grid[i, j], labels[i, j] = 0, "немає"
            elif v.startswith("broken"):
                grid[i, j], labels[i, j] = 0, "відновлюється"
            elif v.startswith("weak"):
                grid[i, j], labels[i, j] = 1, "слабко"
            else:
                grid[i, j], labels[i, j] = 1, "лише FEC"
    from matplotlib.colors import ListedColormap

    a0.imshow(grid, cmap=ListedColormap([BAD, WARN, OK]), vmin=0, vmax=2,
              aspect="auto")
    for i in range(len(methods)):
        for j in range(len(props)):
            a0.text(j, i, labels[i, j], ha="center", va="center", fontsize=7.5,
                    color="white", weight="bold")
    a0.set_xticks(range(len(props)))
    a0.set_xticklabels(props, fontsize=8)
    a0.set_yticks(range(len(methods)))
    a0.set_yticklabels(methods, fontsize=9)
    a0.set_title("Що метод захищає", fontsize=10)
    a0.grid(False)

    # ---- right: quality and coverage on the damaged channel -------------
    y = np.arange(len(methods))
    psnr = [_float(next(iter(_by(rows, channel=ch, method=m)), {}), "mean")
            for m in methods]
    lo = [_float(next(iter(_by(rows, channel=ch, method=m)), {}), "lo") for m in methods]
    hi = [_float(next(iter(_by(rows, channel=ch, method=m)), {}), "hi") for m in methods]
    auth = [grid[i, 1] == 2 for i in range(len(methods))]
    a1.barh(y, psnr, xerr=[np.abs(np.array(psnr) - np.array(lo)),
                           np.abs(np.array(hi) - np.array(psnr))],
            color=[OK if a else NEUTRAL for a in auth], capsize=2,
            hatch=["" if a else "//" for a in auth], edgecolor="white", lw=0.5)
    a1.set_xlim(0, max(psnr) * 1.42)
    # coverage is printed in its own right-hand column, clear of the error bars
    x_text = a1.get_xlim()[1] * 0.985
    for i, m in enumerate(methods):
        c = cov.get((ch, m))
        if c is not None and np.isfinite(c):
            a1.text(x_text, i, f"покриття {c:.2f}", va="center", ha="right",
                    fontsize=7.5, color=OK if auth[i] else "#546e7a")
    a1.set_yticks(y)
    a1.set_yticklabels(methods, fontsize=9)
    a1.invert_yaxis()
    a1.set_xlabel(ctx.t("psnr"))
    a1.set_title(f"Якість на каналі «{ch}»", fontsize=10)

    fig.suptitle("K01. Захист і якість за методами", y=1.02, fontsize=12)
    footnote(ctx, fig,
             "B1 і B2 - це відтворена схема 2021 року та її «покращена» версія. "
             "Вони дають картинку, але НЕ дають ані конфіденційності, ані "
             "автентифікації, і при цьому програють навіть незахищеному "
             "аналоговому передаванню на пошкодженому каналі. Штрихування = "
             "показане ніким не автентифіковане.")
    out = [{"method": m, "channel": ch, "psnr_full": psnr[i],
            "coverage": cov.get((ch, m)),
            **{k: SECURITY_PROPERTIES.get(m, {}).get(k) for k in keys}}
           for i, m in enumerate(methods)]
    return export(ctx, "K01", fig, out, {"channel": ch,
                                         "source_table": "summary.csv"})


# ============================================================== K02
@figure("K02")
def k02(ctx: FigureContext) -> Dict[str, Any]:
    """The crossover: below it the simple scheme wins, above it the proposal."""
    plt = _plt()
    rows = [r for r in ctx.table("sweeps.csv") if r.get("axis") == "burst_len_lines"]
    if not rows:
        raise FigurePending("немає осі burst_len_lines у sweeps.csv")

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.2, 4.3))
    series: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
    for m in ("B0a-R", "B4", "P"):
        sel = sorted(_by(rows, method=m), key=lambda r: _float(r, "channel_value"))
        if not sel:
            continue
        xs = [_float(r, "channel_value") for r in sel]
        ys = [_float(r, "psnr_full") for r in sel]
        n = [max(int(float(r.get("n_scenes") or 1)), 1) for r in sel]
        err = [_float(r, "psnr_full_sd", 0.0) / np.sqrt(k) * 1.96
               for r, k in zip(sel, n)]
        series[m] = (xs, ys, err)
        st = style(m)
        a0.errorbar(xs, ys, yerr=err, marker=st["marker"], ls=st["ls"],
                    color=st["color"], ms=5, lw=1.6, capsize=2, label=m)
        cs = [_float(r, "coverage") for r in sel]
        a1.plot(xs, cs, marker=st["marker"], ls=st["ls"], color=st["color"],
                ms=5, lw=1.6, label=m)

    # find where P overtakes B4
    cross = None
    if "P" in series and "B4" in series:
        xs, yp, _ = series["P"]
        _, yb, _ = series["B4"]
        for i in range(1, min(len(yp), len(yb))):
            if yp[i - 1] < yb[i - 1] and yp[i] >= yb[i]:
                x0, x1 = xs[i - 1], xs[i]
                d0, d1 = yp[i - 1] - yb[i - 1], yp[i] - yb[i]
                cross = x0 + (x1 - x0) * (-d0) / max(d1 - d0, 1e-9)
                break
    if cross is not None:
        for ax in (a0, a1):
            ax.axvline(cross, color="#37474f", ls=":", lw=1.2)
        a0.axvspan(a0.get_xlim()[0], cross, color="#1976d2", alpha=0.06)
        a0.axvspan(cross, a0.get_xlim()[1], color=OK, alpha=0.07)
        _annot(a0, f"перелом ≈ {cross:.0f} рядків\nліворуч кращий B4,\nправоруч — P",
               (cross, a0.get_ylim()[0] + 0.12 * np.ptp(a0.get_ylim())),
               (cross + 6, a0.get_ylim()[0] + 0.30 * np.ptp(a0.get_ylim())))

    a0.set_xlabel(ctx.t("burst_len"))
    a0.set_ylabel(ctx.t("psnr"))
    a0.set_title("Якість", fontsize=10)
    a0.legend(fontsize=8)
    a1.set_xlabel(ctx.t("burst_len"))
    a1.set_ylabel(ctx.t("coverage"))
    a1.set_title("Поточне перевірене покриття", fontsize=10)
    a1.legend(fontsize=8)
    fig.suptitle("K02. Точка перелому", y=1.02, fontsize=12)
    footnote(ctx, fig,
             "Одна змінна - довжина пакета пошкодження; частота пакетів "
             "зафіксована на 4 за растр, решта каналу - профіль mild. Смуги - "
             "95% довірчий інтервал за незалежними сценами. Перелом означає, що "
             "перевага запропонованої схеми не загальна, а починається з певного "
             "рівня пакетного пошкодження.")
    out = [{"method": m, "burst_lines": x, "psnr_full": y, "ci95": e}
           for m, (xs, ys, es) in series.items() for x, y, e in zip(xs, ys, es)]
    return export(ctx, "K02", fig, out,
                  {"crossover_burst_lines": cross, "source_table": "sweeps.csv"})


# ============================================================== K03
@figure("K03")
def k03(ctx: FigureContext) -> Dict[str, Any]:
    """Which method wins where - with "not established" as its own colour."""
    plt = _plt()
    rows = [r for r in ctx.table("interaction.csv")
            if r.get("grid") == "noise_x_burst"]
    if not rows:
        raise FigurePending("немає сітки noise_x_burst у interaction.csv")
    xs = sorted({_float(r, "noise_sigma") for r in rows})
    ys = sorted({_float(r, "burst_len_lines") for r in rows})
    diff = np.full((len(ys), len(xs)), np.nan)
    unc = np.full((len(ys), len(xs)), np.nan)
    out: List[Dict[str, Any]] = []
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            cell = [r for r in rows if _float(r, "noise_sigma") == xv
                    and _float(r, "burst_len_lines") == yv]
            a = [r for r in cell if r["method"] == "P"]
            b = [r for r in cell if r["method"] == "B4"]
            if a and b:
                d = _float(a[0], "psnr_full") - _float(b[0], "psnr_full")
                na = max(int(float(a[0].get("n_scenes") or 1)), 1)
                sd = np.hypot(_float(a[0], "psnr_full_sd", 0.0),
                              _float(b[0], "psnr_full_sd", 0.0))
                diff[i, j] = d
                unc[i, j] = 1.96 * sd / np.sqrt(na)
                out.append({"noise_sigma": xv, "burst_len_lines": yv,
                            "P_minus_B4": d, "ci95_halfwidth": unc[i, j],
                            "decisive": bool(abs(d) > unc[i, j])})

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    # a decisive cell is coloured; an indecisive one is grey, not pale green
    shown = np.where(np.abs(diff) > unc, diff, np.nan)
    lim = float(np.nanmax(np.abs(shown))) if np.isfinite(shown).any() else 1.0
    ax.imshow(np.zeros_like(diff), cmap="Greys", vmin=0, vmax=1, aspect="auto",
              origin="lower", alpha=0.10)
    im = ax.imshow(shown, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto",
                   origin="lower")
    for i in range(len(ys)):
        for j in range(len(xs)):
            if not np.isfinite(diff[i, j]):
                txt, col = "—", "#607d8b"
            elif abs(diff[i, j]) <= unc[i, j]:
                txt, col = "≈", "#455a64"
            else:
                txt, col = f"{diff[i, j]:+.1f}", "#212121"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=col)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([f"{v:g}" for v in ys])
    ax.set_xlabel(ctx.t("noise"))
    ax.set_ylabel(ctx.t("burst_len"))
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="P − B4, дБ")
    fig.suptitle("K03. Карта режимів", y=0.99, fontsize=12)
    n_dec = sum(1 for r in out if r["decisive"])
    footnote(ctx, fig,
             f"«≈» — різниця менша за власний довірчий інтервал: не перемога "
             f"нікого. Вирішальних клітинок {n_dec} з {len(out)}. Зелене — "
             f"виграє P, червоне — B4. Колір нанесено ЛИШЕ там, де різниця "
             f"перевищує невизначеність.")
    return export(ctx, "K03", fig, out, {"source_table": "interaction.csv"})


# ============================================================== K04
@figure("K04")
def k04(ctx: FigureContext) -> Dict[str, Any]:
    """Two axes, no single winner: FEC survival against joint description loss."""
    plt = _plt()
    cw = [r for r in ctx.table("codewords.csv") if r.get("admissible") == "True"
          and r.get("n_descriptions") == "2" and r.get("column_twist") == "1"]
    jl = [r for r in ctx.table("joint_loss.csv") if r.get("n_descriptions") == "2"
          and r.get("column_twist") == "1"]
    if not cw or not jl:
        raise FigurePending("немає даних E05 для двох описів")
    burst = "16"
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    out: List[Dict[str, Any]] = []
    nsym = _float(cw[0], "rs_nsym")
    fam_colour = {"bawp": OK, "block": "#1976d2", "sequential": "#8d6e63",
                  "diagonal": "#7e57c2", "random": "#ef6c00"}

    # Several placements land on exactly the same point - that coincidence is
    # itself a finding (E05), so they are drawn once and labelled as a group
    # instead of being stacked into an unreadable pile of overlapping text.
    points: Dict[Tuple[float, float], List[str]] = {}
    for scheme in sorted({r["scheme"] for r in cw}):
        c = [r for r in cw if r["scheme"] == scheme and r["burst_lines"] == burst]
        if not c:
            continue
        j = [r for r in jl if r["scheme"] == scheme and r["burst_lines"] == burst
             and r["stripe_lost"] == "True"]
        dmg = _float(c[0], "worst_damaged_bytes")
        lost = sum(_float(r, "fraction", 0.0) for r in j)
        points.setdefault((round(dmg, 3), round(lost, 4)), []).append(scheme)
        out.append({"scheme": scheme, "burst_lines": int(burst),
                    "worst_damaged_bytes": dmg,
                    "p_both_descriptions_lost": lost,
                    "buffer_rows": _float(c[0], "accumulation_rows"),
                    "survives_as_erasures": c[0].get("survives_as_erasures")})

    span_x = max(k[0] for k in points) or 1.0
    placed = 0
    for (dmg, lost), names in sorted(points.items()):
        fam = names[0].split("-")[0]
        ax.scatter(dmg, lost, s=130, color=fam_colour.get(fam, "#455a64"),
                   zorder=3, edgecolor="white", lw=1.2)
        if len(names) == 1:
            label = names[0]
        else:
            # collapse bawp-8/16/32 into "bawp-8/16/32" rather than three labels
            heads = {n.split("-")[0] for n in names}
            if len(heads) == 1:
                tails = "/".join(n.split("-", 1)[1] for n in sorted(
                    names, key=lambda x: int(x.split("-")[1])))
                label = f"{names[0].split('-')[0]}-{tails}"
            else:
                label = " = ".join(sorted(names))
        # points that are close but not identical still collide as text, so
        # neighbours alternate above/below instead of overprinting each other
        near = sum(1 for (d2, l2) in points
                   if abs(d2 - dmg) < 0.08 * span_x and abs(l2 - lost) < 0.08)
        dy = 9 if (placed % 2 == 0 or near < 2) else -17
        ax.annotate(label, (dmg, lost), xytext=(9, dy),
                    textcoords="offset points", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                              alpha=0.8))
        placed += 1

    ax.axvline(nsym, color=BAD, ls="--", lw=1.3)
    ax.set_xlabel("найгірше пошкодження одного RS-слова, байт  →  гірше")
    ax.set_ylabel("частка позицій burst, де втрачено\nОБИДВА описи смуги  →  гірше")
    ax.set_ylim(-0.08, 1.18)
    ax.set_xlim(0, max(k[0] for k in points) * 1.12)
    ax.set_title("K04. Розміщення: компроміс без переможця", fontsize=12, pad=26)
    ax.text(nsym, 1.21, f"межа RS: {int(nsym)} Б стирань", ha="center", va="bottom",
            fontsize=8, color=BAD, transform=ax.get_xaxis_transform())
    ax.axvspan(0, nsym, color=OK, alpha=0.05)
    ax.text(nsym * 0.5, -0.055, "FEC ще рятує слово", ha="center", fontsize=7.5,
            color="#33691e")
    ax.scatter([], [], s=130, color="#455a64", label="одна точка = одне розміщення")
    _annot(ax, "ідеал недосяжний", (nsym * 0.25, 0.02),
           (nsym * 1.15, 0.30), "#37474f")

    return export(ctx, "K04", fig, out,
                  {"burst_lines": int(burst), "rs_nsym": nsym,
                   "source_table": "codewords.csv+joint_loss.csv"})


# ============================================================== K05
@figure("K05")
def k05(ctx: FigureContext) -> Dict[str, Any]:
    """Where the advantage actually comes from, one change at a time."""
    plt = _plt()
    rows = ctx.table("chain.csv")
    if not rows:
        raise FigurePending("немає chain.csv - E13 не запускалась")
    # chain.csv now holds both orderings; K05 shows the declared forward one
    # and K11 puts the two side by side.
    fwd = [r for r in rows if r.get("order", "fwd") == "fwd"] or rows
    rows = sorted(fwd, key=lambda r: int(float(r.get("step", 0))))
    labels = [r.get("label", "?") for r in rows]
    ok = [r.get("admissible") == "True" for r in rows]
    vals = [_float(r, "psnr_full") if o else float("nan") for r, o in zip(rows, ok)]
    deltas = [_float(r, "delta_psnr", 0.0) if o else float("nan")
              for r, o in zip(rows, ok)]
    lo = [_float(r, "delta_psnr_full_lo") for r in rows]
    hi = [_float(r, "delta_psnr_full_hi") for r in rows]

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(9.6, 6.2), sharex=True,
                                 gridspec_kw={"height_ratios": [1.25, 1.0]})
    x = np.arange(len(labels))
    a0.plot(x, vals, "o-", color="#37474f", lw=1.6, ms=7, zorder=3)
    for i, (v, o) in enumerate(zip(vals, ok)):
        if o:
            a0.annotate(f"{v:.2f}", (i, v), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=8)
    base, final = vals[0], next((v for v in reversed(vals) if np.isfinite(v)), np.nan)
    a0.axhline(base, color="#1976d2", ls=":", lw=1.0)
    a0.set_ylabel(ctx.t("psnr"))
    a0.set_title(f"Накопичена якість: {base:.2f} → {final:.2f} дБ "
                 f"({final - base:+.2f})", fontsize=10)

    colors = [OK if (np.isfinite(d) and d > 0) else BAD if np.isfinite(d) else NEUTRAL
              for d in deltas]
    # Each step's contribution carries its paired interval (R09): "+0.20 dB"
    # and "how sure" belong on the same axis.
    err = np.array([[max(0.0, d - l) if np.isfinite(l) and np.isfinite(d) else 0.0
                     for d, l in zip(deltas, lo)],
                    [max(0.0, h - d) if np.isfinite(h) and np.isfinite(d) else 0.0
                     for d, h in zip(deltas, hi)]])
    a1.bar(x, deltas, 0.6, color=colors,
           yerr=err if np.isfinite(err).all() and err.any() else None,
           capsize=3, ecolor="#37474f")
    a1.axhline(0, color="#37474f", lw=0.9)
    for i, d in enumerate(deltas):
        if np.isfinite(d) and i:
            a1.annotate(f"{d:+.2f}", (i, d), xytext=(0, 6 if d >= 0 else -13),
                        textcoords="offset points", ha="center", fontsize=8)
    a1.set_ylabel("внесок кроку, дБ")
    a1.set_xticks(x)
    a1.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)

    fig.suptitle("K05. Звідки береться перевага", y=0.985, fontsize=12)
    mech = sum(d for lbl, d in zip(labels, deltas)
               if np.isfinite(d) and ("опис" in lbl or "BAWP" in lbl))
    transport = sum(d for lbl, d in zip(labels, deltas)
                    if np.isfinite(d) and not ("опис" in lbl or "BAWP" in lbl))
    footnote(ctx, fig,
             f"Канал bursty, ті самі кліпи й ті самі реалізації каналу на всіх "
             f"кроках. Параметри транспорту, доступні будь-якій схемі, дають "
             f"{transport:+.2f} дБ; два запропоновані механізми (MDC і BAWP) — "
             f"{mech:+.2f} дБ. Вуса — парні bootstrap-інтервали різниці між "
             f"сусідніми кроками. Порядок кроків оголошено наперед; наскільки "
             f"розклад від нього залежить — див. K11.")
    out = [{"step": int(float(r.get("step", 0))), "label": r.get("label"),
            "admissible": r.get("admissible") == "True",
            "psnr_full": _float(r, "psnr_full"),
            "delta_psnr": _float(r, "delta_psnr"),
            "delta_lo": _float(r, "delta_psnr_full_lo"),
            "delta_hi": _float(r, "delta_psnr_full_hi"),
            "coverage": _float(r, "coverage"),
            "availability": _float(r, "availability")} for r in rows]
    return export(ctx, "K05", fig, out,
                  {"transport_gain_db": transport, "mechanism_gain_db": mech,
                   "source_table": "chain.csv"})


# ============================================================== K06
@figure("K06")
def k06(ctx: FigureContext) -> Dict[str, Any]:
    """Rate against quality, with the channel's actual capacity drawn on."""
    plt = _plt()
    rows = ctx.table("e01_rate_quality.csv")
    ok = [r for r in rows if r.get("admissible") == "True"]
    if not ok:
        raise FigurePending("усі клітинки E01 недопустимі")
    budget = ctx.json("budgets.json")
    cap_bps = None
    for m in ("B4", "P"):
        if m in budget.get("methods", {}):
            cap_bps = float(budget["methods"][m]["payload_bitrate_bps"])
            break
    fps = float(budget.get("budget", {}).get("source_fps", 8.333)) if budget else 8.333

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    marks = {"1": "o", "2": "s", "4": "^"}
    cols = {"dct": "#1976d2", "jpeg": OK}
    out: List[Dict[str, Any]] = []
    for codec in sorted({r["codec"] for r in ok}):
        for nd in sorted({r["n_descriptions"] for r in ok}):
            sel = sorted(_by(ok, codec=codec, n_descriptions=nd,
                             stripe_height="24"),
                         key=lambda r: _float(r, "bits_per_frame"))
            if not sel:
                continue
            ax.plot([_float(r, "bits_per_frame") for r in sel],
                    [_float(r, "psnr_full") for r in sel],
                    marker=marks.get(str(nd), "."), ms=5, lw=1.4,
                    color=cols.get(codec, "#8d6e63"),
                    alpha=1.0 if nd == "1" else 0.5,
                    label=f"{codec}, {nd} опис{'и' if nd != '1' else ''}")
            out.extend({"codec": codec, "n_descriptions": nd,
                        "quality": r.get("quality"),
                        "bits_per_frame": _float(r, "bits_per_frame"),
                        "psnr_full": _float(r, "psnr_full")} for r in sel)
    if cap_bps:
        cap_bits = cap_bps / max(fps, 1e-9)
        ax.axvline(cap_bits, color=BAD, ls="--", lw=1.3)
        ax.text(cap_bits, 0.03, f" місткість каналу\n {cap_bits/1e3:.0f} кбіт/кадр",
                rotation=90, va="bottom", fontsize=7.5, color=BAD,
                transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel(ctx.t("bits"))
    ax.set_ylabel(ctx.t("psnr"))
    ax.set_title("K06. Ціна стиснення", fontsize=12)
    ax.legend(fontsize=7.5, ncol=2)
    n_bad = len(rows) - len(ok)
    footnote(ctx, fig,
             f"Чистий канал. Кодеки зіставлені за ФАКТИЧНИМ бітрейтом: за "
             f"однакового номера quality DCT і JPEG витрачають утричі різну "
             f"кількість бітів, тож порівняння «за quality» безглузде. "
             f"Недопустимих конфігурацій {n_bad} з {len(rows)}; `raw` не "
             f"вміщується у бюджет у ЖОДНІЙ з перевірених.")
    return export(ctx, "K06", fig, out,
                  {"capacity_bits_per_frame": (cap_bps / fps) if cap_bps else None,
                   "source_table": "e01_rate_quality.csv"})


# ============================================================== K07 / K08
def _pick_case(ctx: FigureContext, method: str, channel: str,
               metric: str = "psnr_full") -> Dict[str, Any]:
    rows = [r for r in ctx.table("frames.csv")
            if r.get("method") == method and r.get("channel") == channel]
    vals = sorted(((_float(r, metric), r) for r in rows
                   if np.isfinite(_float(r, metric))), key=lambda vr: vr[0])
    if not vals:
        raise FigurePending(f"немає кадрів {method}@{channel}")
    return vals[len(vals) // 2][1]          # the median case, never the best


def _render(ctx: FigureContext, row: Dict[str, Any], method: str) -> Dict[str, Any]:
    import dataclasses

    from avsec.channel import ChannelTrace
    from avsec.config import load_config
    from avsec.experiments import build_methods, build_sources

    cfg_path = os.path.join(ctx.run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FigurePending("у прогоні немає config.yaml")
    cfg = load_config(cfg_path)
    base = str(row.get("channel", "")).split("[")[0] or cfg.channel_preset
    sub = dataclasses.replace(cfg, methods=(method,), channel_preset=base,
                              channel_overrides={})
    srcs = {s.name: s for s in build_sources(sub)}
    src = srcs.get(row.get("clip", ""))
    if src is None:
        raise FigurePending(f"кліп {row.get('clip')} відсутній")
    m = build_methods(sub)[method]
    trace = ChannelTrace(seed=cfg.channel_seed_value,
                         scene=f"{row['clip']}|{row.get('channel','')}",
                         repetition=int(float(row.get("repetition", 0) or 0)),
                         profile=str(row.get("channel", "")),
                         rasters_per_frame=cfg.budget.rasters_per_frame)
    m.reset()
    target = int(float(row.get("frame_id", 0) or 0))
    res = None
    for fi in range(target + 1):
        res = m.process(src.frames[fi], fi, trace)
    return {"res": res, "stale": getattr(res, "stale",
                                         np.zeros_like(res.available))}


@figure("K07")
def k07(ctx: FigureContext) -> Dict[str, Any]:
    """The whole path on one real frame: source, wire, arrival, reconstruction."""
    plt = _plt()
    channel = "bursty"
    row = _pick_case(ctx, "P", channel)
    pack = {m: _render(ctx, dict(row, method=m), m) for m in ("B4", "P")}

    fig, axes = plt.subplots(2, 4, figsize=(12.4, 5.4))
    titles = ("1. оригінал (джерело)", "2. переданий растр",
              "3. прийнятий растр", "4. реконструкція")
    out: List[Dict[str, Any]] = []
    for r, m in enumerate(("B4", "P")):
        res = pack[m]["res"]
        imgs = (res.original, res.transmitted, res.received, res.reconstructed)
        for c, (img, t) in enumerate(zip(imgs, titles)):
            axes[r, c].imshow(img, cmap="gray", vmin=0, vmax=255,
                              interpolation="nearest")
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].grid(False)
            if r == 0:
                axes[r, c].set_title(t, fontsize=9)
        axes[r, 0].set_ylabel(m, fontsize=12, rotation=0, labelpad=18,
                              va="center", weight="bold")
        axes[r, 3].set_xlabel(f"PSNR {res.metrics.psnr_full:.1f} дБ · "
                              f"покриття {res.metrics.coverage:.2f}", fontsize=8.5)
        out.append({"method": m, "clip": row.get("clip"), "channel": channel,
                    "frame_id": row.get("frame_id"),
                    "psnr_full": float(res.metrics.psnr_full),
                    "coverage": float(res.metrics.coverage)})
    fig.suptitle("K07. Наскрізний приклад: кадр з дрона через увесь тракт",
                 y=1.0, fontsize=12)
    footnote(ctx, fig,
             f"Кліп {row.get('clip')}, кадр {row.get('frame_id')}, канал "
             f"«{channel}», МЕДІАННИЙ випадок за PSNR — не найкращий кадр. "
             f"Обидва рядки бачать ту саму реалізацію каналу (спільна траса), "
             f"тому різниця між ними — це різниця схем, а не різних завад. "
             f"Стовпець 2 — те, що йде в аналоговий тракт: не зображення, а "
             f"рівні яскравості, у які закодовано зашифровані дані.")
    return export(ctx, "K07", fig, out,
                  {"clip": row.get("clip"), "channel": channel,
                   "selection": "median PSNR case", "source_table": "frames.csv"})


@figure("K08")
def k08(ctx: FigureContext) -> Dict[str, Any]:
    """What the viewer is actually looking at: verified, stale or invented."""
    plt = _plt()
    from avsec.evaluation import availability_overlay

    channel = "bursty"
    row = _pick_case(ctx, "P", channel)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.1))
    out: List[Dict[str, Any]] = []
    axes[0].imshow(_render(ctx, dict(row), "P")["res"].original, cmap="gray",
                   vmin=0, vmax=255)
    axes[0].set_title("оригінал", fontsize=10)
    for ax, m in zip(axes[1:], ("B4", "P")):
        pack = _render(ctx, dict(row), m)
        res, stale = pack["res"], pack["stale"]
        ov = availability_overlay(res.reconstructed, res.available, stale)
        ax.imshow(ov[..., ::-1])
        cov = float(res.available.mean())
        st = float(np.asarray(stale).mean())
        ax.set_title(f"{m}: перевірено {cov:.2f}", fontsize=10)
        ax.set_xlabel(f"старе {st:.2f} · домальовано {1 - cov - st:.2f}",
                      fontsize=8.5)
        out.append({"method": m, "coverage": cov, "stale_fraction": st,
                    "estimated_fraction": 1 - cov - st,
                    "psnr_full": float(res.metrics.psnr_full)})
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("K08. Що саме показано глядачу", y=1.0, fontsize=12)
    footnote(ctx, fig,
             "Зелене — поточні дані, які пройшли перевірку автентичності. "
             "Бурштинове — показано зі старого кадру. Червоне — домальовано "
             "інтерполяцією, за цим не стоїть жодного прийнятого біта. Ані "
             "старе, ані домальоване НЕ входить у «поточне перевірене покриття»; "
             "саме тому ця метрика, а не PSNR, є головною для захищеного тракту.")
    return export(ctx, "K08", fig, out,
                  {"clip": row.get("clip"), "channel": channel,
                   "source_table": "frames.csv"})


# ============================================================== K09
@figure("K09")
def k09(ctx: FigureContext) -> Dict[str, Any]:
    """Budget, latency and memory: three accountings that are often conflated."""
    plt = _plt()
    b = ctx.json("budgets.json")
    methods = [m for m in ("B0d", "B3", "B4", "P") if m in b.get("methods", {})]
    if not methods:
        raise FigurePending("немає budgets.json")
    ref = "P" if "P" in methods else methods[-1]
    d = b["methods"][ref]

    fig = plt.figure(figsize=(12.6, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.1, 1.0], wspace=0.34)

    # --- 1. where the raster's symbols go -------------------------------
    a0 = fig.add_subplot(gs[0, 0])
    steps = d["waterfall"]["steps"]
    names = [s["label"] for s in steps[1:]] + ["корисні дані"]
    vals = [abs(s["symbols"]) for s in steps[1:]] + [d["waterfall"]["payload_symbols"]]
    cap = steps[0]["symbols"]
    colors = [BAD, "#8d6e63", "#1976d2", "#4fc3f7", "#ba68c8", WARN, "#90a4ae", OK]
    left = 0.0
    for n, v, c in zip(names, vals, colors[:len(names)]):
        if v <= 0:
            continue
        a0.barh([0], [v], left=left, color=c, height=0.55, label=n)
        if v / cap > 0.06:
            a0.text(left + v / 2, 0, f"{100*v/cap:.0f}%", ha="center", va="center",
                    fontsize=7.5, color="white", weight="bold")
        left += v
    a0.set_xlim(0, cap)
    a0.set_yticks([])
    a0.set_xlabel(f"символи растру (усього {int(cap)})")
    a0.set_title(f"Куди йде місткість, {ref}", fontsize=10)
    a0.legend(fontsize=6.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    a0.grid(False)

    # --- 2. latency: schedule against the additive misconception ---------
    a1 = fig.add_subplot(gs[0, 1])
    ev = d.get("latency_events_sample") or []
    kinds = ["capture", "stripe_ready", "unit_ready", "tx_start", "rx_end",
             "assembled", "display"]
    cmap = {"capture": "#455a64", "stripe_ready": "#8d6e63", "unit_ready": WARN,
            "tx_start": "#1976d2", "rx_end": "#7e57c2", "assembled": OK,
            "display": BAD}
    for i, k in enumerate(kinds):
        ts = [float(e["t"]) * 1e3 for e in ev if e.get("kind") == k
              and int(e.get("frame_id", 0)) == 0]
        a1.scatter(ts, [i] * len(ts), s=26, color=cmap.get(k, "#333"), zorder=3)
    lat = d["latency"]["latency_s"]["mean"] * 1e3
    add = d["latency"]["additive_upper_bound"]["virtual_total_s"] * 1e3
    a1.axvline(lat, color=BAD, lw=1.3)
    a1.axvline(add, color=NEUTRAL, ls="--", lw=1.2)
    a1.text(lat, len(kinds) - 0.4, f" {lat:.0f} мс", color=BAD, fontsize=8)
    a1.text(add, len(kinds) - 1.3, f" сума етапів {add:.0f} мс", color="#546e7a",
            fontsize=7.5)
    a1.set_yticks(range(len(kinds)))
    a1.set_yticklabels(kinds, fontsize=7.5)
    a1.set_xlabel("час від захоплення кадру, мс")
    a1.set_title("Затримка = різниця позначок часу", fontsize=10)

    # --- 3. memory: three different quantities ---------------------------
    a2 = fig.add_subplot(gs[0, 2])
    lim_kb = float(b.get("budget", {}).get("max_receiver_buffer_kb", 512))
    q = [("логічний буфер\nпротоколу", d["memory"]["logical_total_kb"] / 1024, OK),
         ("масиви\nсимулятора", d["memory"]["model_total_mb"], NEUTRAL),
         ("пік пам'яті\nпроцесу", d["memory"]["process_peak_mb"], "#455a64")]
    a2.bar([x[0] for x in q], [x[1] for x in q], color=[x[2] for x in q], width=0.6)
    a2.axhline(lim_kb / 1024, color=BAD, ls="--", lw=1.2)
    a2.text(2.4, lim_kb / 1024, f"ліміт {lim_kb:.0f} кБ", ha="right", va="bottom",
            fontsize=7.5, color=BAD)
    for i, (_n, v, _c) in enumerate(q):
        a2.annotate(f"{v:.2f} МБ", (i, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8)
    a2.set_yscale("log")
    a2.set_ylabel(ctx.t("memory"))
    a2.set_title("Три різні величини пам'яті", fontsize=10)
    a2.tick_params(axis="x", labelsize=7.5)

    fig.suptitle("K09. Бюджет, затримка, пам'ять", y=1.03, fontsize=12)
    footnote(ctx, fig,
             f"Ліміт 512 кБ стосується ЛИШЕ логічного буфера протоколу — це не "
             f"пікова пам'ять процесу і не розмір масивів симулятора. Затримка "
             f"{lat:.0f} мс — різниця позначок часу display−capture; сума "
             f"тривалостей етапів дала б {add:.0f} мс, бо етапи перекриваються і "
             f"спільні інтервали рахувалися б двічі.")
    out = [{"method": ref, "latency_ms": lat, "additive_ms": add,
            "logical_kb": d["memory"]["logical_total_kb"],
            "process_peak_mb": d["memory"]["process_peak_mb"],
            "model_mb": d["memory"]["model_total_mb"],
            "payload_symbols": d["waterfall"]["payload_symbols"],
            "raster_capacity_symbols": cap}]
    return export(ctx, "K09", fig, out, {"method": ref,
                                         "source_table": "budgets.json"})


# ============================================================== K10
@figure("K10")
def k10(ctx: FigureContext) -> Dict[str, Any]:
    """The 2021 scheme, broken: the picture comes back from one known pair."""
    plt = _plt()
    data = ctx.json("attacks.json")
    res = data.get("results", [])
    known = [r for r in res if r.get("name") == "known_plaintext_tile_matching"]
    chosen = [r for r in res if r.get("name") == "chosen_plaintext_permutation_recovery"]
    boundary = [r for r in res if r.get("name") == "boundary_compatibility_reassembly"]
    brute = [r for r in res if r.get("name") == "lfsr_seed_bruteforce"]
    if not known:
        raise FigurePending("немає результатів атак у attacks.json")

    fig = plt.figure(figsize=(12.2, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.0], wspace=0.3)

    # --- 1. what the attacker gets ---------------------------------------
    a0 = fig.add_subplot(gs[0, 0])
    imgs = _scramble_demo(ctx)
    if imgs:
        strip = np.hstack([imgs["original"], imgs["scrambled"], imgs["recovered"]])
        a0.imshow(strip, cmap="gray", vmin=0, vmax=255)
        w = imgs["original"].shape[1]
        for i, t in enumerate(("оригінал", "передано (B1)", "відновлено атакою")):
            a0.text(i * w + w / 2, -6, t, ha="center", fontsize=8.5)
    a0.set_xticks([])
    a0.set_yticks([])
    a0.grid(False)
    a0.set_title("Одна відома пара кадрів", fontsize=10, pad=18)

    # --- 2. recovery accuracy --------------------------------------------
    a1 = fig.add_subplot(gs[0, 1])
    bars = []
    if chosen:
        bars.append(("обраний кадр",
                     float(chosen[0]["metrics"].get("permutation_accuracy", 0))))
    bars.append(("відома пара",
                 float(np.mean([float(r["metrics"].get("permutation_accuracy", 0))
                                for r in known]))))
    if boundary:
        bars.append(("лише шифротекст\n(сусідство)",
                     float(np.mean([float(r["metrics"].get("neighbour_accuracy", 0))
                                    for r in boundary]))))
    a1.bar([b[0] for b in bars], [b[1] for b in bars],
           color=[BAD if b[1] > 0.9 else WARN for b in bars], width=0.6)
    for i, b in enumerate(bars):
        a1.annotate(f"{b[1]*100:.0f}%", (i, b[1]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9,
                    weight="bold")
    a1.set_ylim(0, 1.15)
    a1.set_ylabel("частка правильно відновленого")
    a1.set_title("Успіх атаки", fontsize=10)
    a1.tick_params(axis="x", labelsize=7.5)

    # --- 3. brute force ---------------------------------------------------
    a2 = fig.add_subplot(gs[0, 2])
    if brute:
        met = brute[0]["metrics"]
        rate = float(met.get("seeds_per_second", 1)) or 1.0
        width = int(np.log2(float(met.get("key_space", 65535)) + 1))
        widths = [8, 12, 16, 24, 32]
        secs = [(2 ** w - 1) / rate for w in widths]
        a2.bar([str(w) for w in widths], secs,
               color=[BAD if w <= width else NEUTRAL for w in widths])
        a2.set_yscale("log")
        a2.set_xlabel("ширина seed, біт")
        a2.set_ylabel("час перебору, с")
        a2.set_title(f"Перебір: {float(brute[0]['seconds']):.1f} с на {width} біт",
                     fontsize=10)
    fig.suptitle("K10. Атаки на реконструкцію B1 зі статичною перестановкою",
                 y=1.02, fontsize=11)
    footnote(ctx, fig,
             "Перестановка блоків не приховує вміст: одна відома пара "
             "(оригінал, передане) повністю відновлює перестановку, а лише за "
             "шифротекстом сусідство блоків вгадується на ~95% просто з "
             "неперервності зображення. Сірі стовпці праворуч — ПРОЄКЦІЯ за "
             "виміряною швидкістю, а не виконаний перебір. Низька кореляція чи "
             "висока ентропія тут нічого не доводять.")
    out = [{"attack": b[0], "recovered_fraction": b[1]} for b in bars]
    if brute:
        out.append({"attack": "brute force", "seconds": float(brute[0]["seconds"]),
                    "seeds_per_second": float(brute[0]["metrics"]
                                              .get("seeds_per_second", 0))})
    return export(ctx, "K10", fig, out, {"source_table": "attacks.json"})


def _scramble_demo(ctx: FigureContext) -> Optional[Dict[str, np.ndarray]]:
    """Reproduce one scramble-and-attack on the run's own material."""
    try:
        from avsec.attacks import attack_known_pair
        from avsec.config import load_config
        from avsec.experiments import build_sources
        from avsec.lfsr import BlockScrambler
    except Exception:
        return None
    cfg_path = os.path.join(ctx.run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        return None
    cfg = load_config(cfg_path)
    srcs = build_sources(cfg)
    img = srcs[0].frames[0]
    scr = BlockScrambler(cfg.lfsr)
    scrambled = scr.scramble(img, 0)
    rows, cols = cfg.lfsr.grid_rows, cfg.lfsr.grid_cols
    try:
        res = attack_known_pair(img, scrambled, rows, cols)
        rec = res.reconstructed
    except Exception:
        rec = None
    if rec is None:                 # the attack is the point; never fake it
        return None
    h, w = img.shape
    return {"original": img, "scrambled": scrambled[:h, :w], "recovered": rec[:h, :w]}


__all__ = ["KEY_FIGURES"]


# ============================================================== K11
@figure("K11")
def k11(ctx: FigureContext) -> Dict[str, Any]:
    """The stepwise decomposition depends on the order the steps were applied.

    A waterfall reads as though each component "contributes" a fixed amount.
    It does not: whatever is applied first collects the gain that the later
    steps would also have produced.  Both orderings of the same two end points
    are drawn here - and a step that could not run at all is drawn as that,
    not as a zero-height bar (R09).
    """
    plt = _plt()
    rows = ctx.table("chain.csv")
    if not rows:
        raise FigurePending("немає chain.csv - E13 не запускалась")
    orders = sorted({r.get("order", "fwd") for r in rows})
    if len(orders) < 2:
        raise FigurePending("chain.csv має лише один порядок кроків; "
                            "перезапустіть E13")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4))
    fig.subplots_adjust(bottom=0.30)

    def _wrap(text: str, width: int) -> str:
        import textwrap

        return "\n".join(textwrap.wrap(text, width)) or text

    out: List[Dict[str, Any]] = []
    titles = {"fwd": "спершу параметри транспорту",
              "rev": "спершу два механізми"}
    totals: Dict[str, Dict[str, Any]] = {}
    for ax, order in zip(axes, ["fwd", "rev"]):
        sel = sorted([r for r in rows if r.get("order") == order],
                     key=lambda r: int(float(r.get("step", 0))))
        labels = [r.get("label", "?") for r in sel]
        ok = [r.get("admissible") == "True" for r in sel]
        deltas = [_float(r, "delta_psnr_full", 0.0) if o else float("nan")
                  for r, o in zip(sel, ok)]
        mech_mask = ["опис" in l or "BAWP" in l for l in labels]
        lo = [_float(r, "delta_psnr_full_lo") if o else float("nan")
              for r, o in zip(sel, ok)]
        hi = [_float(r, "delta_psnr_full_hi") if o else float("nan")
              for r, o in zip(sel, ok)]
        x = np.arange(len(labels))
        drawn = [0.0 if not np.isfinite(d) else d for d in deltas]
        colors = [NEUTRAL if not o else (BAD if m else "#1976d2")
                  for o, m in zip(ok, mech_mask)]
        err = np.array([[max(0.0, d - l) if np.isfinite(l) and np.isfinite(d)
                         else 0.0 for d, l in zip(deltas, lo)],
                        [max(0.0, h - d) if np.isfinite(h) and np.isfinite(d)
                         else 0.0 for d, h in zip(deltas, hi)]])
        ax.bar(x, drawn, 0.62, color=colors,
               yerr=err if err.any() else None, capsize=3, ecolor="#37474f")
        ax.axhline(0, color="#37474f", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([_wrap(l, 22) for l in labels], rotation=18,
                           ha="right", fontsize=7)
        ax.set_title(titles.get(order, order), fontsize=10)
        ax.set_ylabel("внесок кроку, дБ")
        mech = sum(d for d, m, o in zip(deltas, mech_mask, ok) if m and o)
        trans = sum(d for d, m, o in zip(deltas, mech_mask, ok) if not m and o)
        blocked = [l for l, o, m in zip(labels, ok, mech_mask) if not o and m]
        totals[order] = {"mechanisms": mech, "transport": trans,
                         "blocked": blocked}
        for i, (lbl, d, o) in enumerate(zip(labels, deltas, ok)):
            if not o:
                # A step that never ran is marked as such, at the axis, in
                # words - not as a bar of height zero.  Consecutive blocked
                # steps are staggered so the labels do not overlap.
                n_before = sum(1 for k in range(i) if not ok[k])
                ax.annotate("НЕДОПУСТИМО\nне вміщується\nу бюджет", (i, 0.0),
                            xytext=(0, 16 + 36 * (n_before % 2)),
                            textcoords="offset points", ha="center",
                            fontsize=6.5, color=BAD, weight="bold",
                            linespacing=1.3)
            elif i:
                ax.annotate(f"{d:+.2f}", (i, d),
                            xytext=(0, 6 if d >= 0 else -13),
                            textcoords="offset points", ha="center",
                            fontsize=7.5)
            out.append({"order": order, "step": i, "label": lbl,
                        "delta": d if o else None, "admissible": bool(o),
                        "is_mechanism": bool(mech_mask[i])})
    finite = [o["delta"] for o in out if o["delta"] is not None
              and np.isfinite(o["delta"])]
    lim = (max(abs(min(finite)), abs(max(finite))) * 1.45) if finite else 1.0
    for ax in axes:
        ax.set_ylim(-lim, lim)

    mf = totals.get("fwd", {}).get("mechanisms", float("nan"))
    mr = totals.get("rev", {}).get("mechanisms", float("nan"))
    blocked = totals.get("rev", {}).get("blocked", [])
    fig.suptitle("K11. Залежність результату від порядку кроків", y=0.99,
                 fontsize=12)
    if blocked:
        note = (f"Ті самі дві кінцеві точки (B4 і P), той самий матеріал, той "
                f"самий канал — різний лише порядок змін. Зворотний порядок "
                f"НЕ ПРОХОДИТЬСЯ: {', '.join(blocked)} не вміщуються у "
                f"растровий бюджет, доки не зменшено розмір одиниці. Тобто "
                f"механізми не є самостійним доповненням до базової схеми — "
                f"вони застосовні лише після тих самих змін транспорту, які й "
                f"дають увесь виграш ({mf:+.2f} дБ за прямого порядку).")
    else:
        same = np.isfinite(mf) and np.isfinite(mr) and (mf < 0) == (mr < 0)
        note = (f"Ті самі дві кінцеві точки, той самий матеріал і канал — "
                f"різний лише порядок. Внесок MDC+BAWP: {mf:+.2f} дБ за "
                f"прямого порядку і {mr:+.2f} за зворотного "
                f"(різниця {abs(mf - mr):.2f} дБ). "
                + ("Знак однаковий, тож висновок не є артефактом порядку."
                   if same else
                   "ЗНАК різний: жоден окремий розклад не може обґрунтувати "
                   "висновок про користь механізмів."))
    footnote(ctx, fig, note)
    return export(ctx, "K11", fig, out,
                  {"mechanism_gain_forward_db": mf,
                   "mechanism_gain_reverse_db": mr,
                   "reverse_order_blocked": blocked,
                   "source_table": "chain.csv"})


# ============================================================== K12
@figure("K12")
def k12(ctx: FigureContext) -> Dict[str, Any]:
    """The same effect measured over scenes and over independent recordings.

    Crops of one photograph are not independent samples of drone imagery.  This
    puts the interval computed over scenes next to the interval computed over
    source photographs, which is the one a claim about new footage needs (R07).
    """
    plt = _plt()
    rows = ctx.table("paired_effects.csv")
    if not rows:
        raise FigurePending("немає paired_effects.csv")
    primary = [r for r in rows if r.get("is_primary") == "True"]
    if not primary:
        primary = [r for r in rows if {r.get("a"), r.get("b")} == {"P", "B4"}]
    if not primary:
        raise FigurePending("у paired_effects.csv немає основного порівняння")

    unit = primary[0].get("unit_of_independence", "scene")
    chans = sorted({r.get("channel", "") for r in primary}, key=channel_key)
    fig, ax = plt.subplots(figsize=(9.0, 0.62 * len(chans) + 3.0))
    out: List[Dict[str, Any]] = []
    y = np.arange(len(chans))
    for i, ch in enumerate(chans):
        r = next((x for x in primary if x.get("channel") == ch), None)
        if r is None:
            continue
        a, b = r.get("a"), r.get("b")
        sign = 1.0 if a == "P" else -1.0
        m = sign * _float(r, "mean")
        if not np.isfinite(m):
            # A channel where no method delivered a single usable instant has
            # no effect to plot, and saying so beats an empty row (R05).
            ax.annotate("жоден метод не дав придатного моменту показу",
                        (0.0, i), xytext=(12, 0), textcoords="offset points",
                        fontsize=8, color=NEUTRAL, va="center")
            out.append({"channel": ch, "mean": None, "n_units": 0,
                        "unit_of_independence": unit, "significant": False,
                        "note": "немає даних"})
            continue
        lo = sign * _float(r, "hi" if sign < 0 else "lo")
        hi = sign * _float(r, "lo" if sign < 0 else "hi")
        n = int(_float(r, "n_units", _float(r, "n_scenes", 0)))
        colour = OK if (np.isfinite(lo) and lo > 0) else \
                 BAD if (np.isfinite(hi) and hi < 0) else NEUTRAL
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", ms=7,
                    color=colour, capsize=4, lw=1.8)
        ax.annotate(f"{m:+.2f} дБ  [{lo:+.2f}; {hi:+.2f}]   n={n}",
                    (m, i), xytext=(10, 8), textcoords="offset points",
                    fontsize=8, color=colour)
        out.append({"channel": ch, "mean": m, "lo": lo, "hi": hi,
                    "n_units": n, "unit_of_independence": unit,
                    "significant": bool(np.isfinite(lo) and (lo > 0 or hi < 0))})
    ax.axvline(0, color="#37474f", lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels(chans)
    ax.set_xlabel("P − B4, дБ")
    ax.set_title(f"K12. Ефект з інтервалом по незалежних одиницях ({unit})",
                 fontsize=12)
    ax.margins(x=0.28)
    footnote(ctx, fig,
             "Одиниця незалежності вказана в заголовку. Інтервал по СЦЕНАХ "
             "узагальнюється лише на нові вирізки тих самих записів; інтервал "
             "по ДЖЕРЕЛАХ — на новий запис. Для набору з однієї фотографії "
             "друге неможливе в принципі, і саме тому набір з 24 незалежних "
             "знімків існує окремо.")
    return export(ctx, "K12", fig, out,
                  {"unit_of_independence": unit,
                   "source_table": "paired_effects.csv"})


# ============================================================== K13
@figure("K13")
def k13(ctx: FigureContext) -> Dict[str, Any]:
    """Where the proposal beats the retuned baseline: a map, not a threshold."""
    plt = _plt()
    rows = ctx.table("operating_map.csv")
    if not rows:
        raise FigurePending("немає operating_map.csv - E15 не запускалась")
    rates = sorted({_float(r, "burst_rate_per_frame") for r in rows})
    lens = sorted({_float(r, "burst_len_lines") for r in rows})
    grid = np.full((len(rates), len(lens)), np.nan)
    sig = np.zeros_like(grid, dtype=bool)
    oper = np.full_like(grid, np.nan)
    out: List[Dict[str, Any]] = []
    for r in rows:
        i = rates.index(_float(r, "burst_rate_per_frame"))
        j = lens.index(_float(r, "burst_len_lines"))
        grid[i, j] = _float(r, "d_P_B4t")
        sig[i, j] = r.get("d_P_B4t_significant") == "True"
        oper[i, j] = _float(r, "operable_P")
        out.append({"burst_rate_per_frame": _float(r, "burst_rate_per_frame"),
                    "burst_len_lines": _float(r, "burst_len_lines"),
                    "d_P_B4t": grid[i, j], "significant": bool(sig[i, j]),
                    "operable_P": oper[i, j],
                    "operable_B4t": _float(r, "operable_B4t")})

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6))
    lim = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(rates)):
        for j in range(len(lens)):
            if not np.isfinite(grid[i, j]):
                continue
            txt = f"{grid[i, j]:+.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    weight="bold" if sig[i, j] else "normal",
                    color="#111111" if sig[i, j] else "#78909c")
    ax.set_xticks(range(len(lens)))
    ax.set_xticklabels([f"{int(v)}" for v in lens], fontsize=8)
    ax.set_yticks(range(len(rates)))
    ax.set_yticklabels([f"{v:g}" for v in rates])
    ax.set_xlabel("довжина пакета, рядків")
    ax.set_ylabel("пакетів на кадр")
    ax.set_title("P − переналаштований B4t, дБ", fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="дБ")

    for i, rate in enumerate(rates):
        ax2.plot(lens, oper[i], "o-", ms=4, label=f"{rate:g} пак./кадр")
    ax2.set_xlabel("довжина пакета, рядків")
    ax2.set_ylabel("частка працездатних моментів (P)")
    ax2.set_ylim(-0.03, 1.03)
    ax2.legend(fontsize=8)
    ax2.set_title("працездатність P за оголошеним критерієм", fontsize=10)

    n_sig_pos = int(np.sum(sig & (grid > 0)))
    n_sig_neg = int(np.sum(sig & (grid < 0)))
    n_flat = int(np.sum(~sig & np.isfinite(grid)))
    fig.suptitle("K13. Область працездатності - карта, а не поріг", y=0.99,
                 fontsize=12)
    footnote(ctx, fig,
             f"Жирним позначено клітинки, де різниця встановлена (парний "
             f"bootstrap по сценах): {n_sig_pos} на користь P, {n_sig_neg} на "
             f"користь B4t, у {n_flat} різниця не встановлена. Виміряно лише "
             f"показані вузли; значення між ними не вимірювалися. Це карта для "
             f"ЦІЄЇ моделі каналу і ЦЬОГО бюджету, а не характеристика "
             f"реального відеотракту.")
    return export(ctx, "K13", fig, out,
                  {"n_cells_P_better": n_sig_pos, "n_cells_B4t_better": n_sig_neg,
                   "n_cells_undetermined": n_flat,
                   "source_table": "operating_map.csv"})
