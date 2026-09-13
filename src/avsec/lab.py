"""ISITIA 2021 in one command: reproduce the scheme, then repeat it on photographs.

This is the lab bench for [docs/reconstruction_2021.md](../../docs/reconstruction_2021.md).
It does two things and writes them side by side:

**Part 1 - the paper's own tables.**  The article reports a "similarity" between
the original and the scrambled image for three seeds (44257, 1234, 1111) and
averages it to 3.94 %.  That similarity is ``|r| * 100``, the absolute Pearson
correlation coefficient in per cent.  The replica here runs the same three seeds
over three images and reports the same quantity, so the two numbers can be put
next to each other rather than asserted to agree.

**Part 2 - the same scheme on real photographs.**  The synthetic patterns of
part 1 are ours; the article used camera frames.  Part 2 runs the identical
scrambler over crops of an actual UAV photograph - open sea, pine canopy,
shoreline, a village - and reports the same similarity per scene.  The point is
not a better number: it is that **the number depends on the content**, not only
on the seed, which is the first hint that it is not measuring confidentiality.

Everything here is deterministic: the same command gives the same bytes.

Usage::

    avsec lab                       # -> results/lab/
    avsec lab --output runs/lab     # somewhere else
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from avsec.config import ExperimentConfig
from avsec.lfsr import (
    BlockScrambler,
    LFSRConfig,
    ScramblerConfig,
    correlation_similarity,
)
from avsec.utils import ensure_dir, environment_record, write_csv, write_json

#: The three seeds the article prints in its experiments.
ISITIA_SEEDS: Tuple[int, ...] = (44257, 1234, 1111)

#: The seed that appears in the caption of the article's Fig. 8 - the one the
#: brute-force attack recovers in :mod:`avsec.attacks`.
ISITIA_FIGURE_SEED = 44257

#: What the article reports, for putting our numbers next to rather than
#: instead of.  Both are averages over its own three experiments.
PAPER_MEAN_SCRAMBLED_PCT = 3.94
PAPER_MEAN_DECRYPTED_PCT = 98.24

#: Part 1 material: three procedural images, chosen before any result was seen
#: to span flat / structured / textured content.
SYNTHETIC_IMAGES: Tuple[str, ...] = ("smooth", "edges", "texture")

#: Part 2 material: four crops of the committed UAV photograph, named for what
#: they contain.  These are the content classes a drone downlink actually sees.
NATURAL_SCENES: Tuple[Tuple[str, str], ...] = (
    ("sky_horizon", "море й небо над косою — найплоскіший вміст набору"),
    ("village", "село — поле дрібних яскравих об'єктів"),
    ("shoreline", "берегова лінія — довга контрастна межа"),
    ("forest_canopy", "крона сосон — найважча текстура кадру"),
)


def _scrambler(cfg: ExperimentConfig, seed: int) -> BlockScrambler:
    """The reconstruction of the 2021 scheme, with only the seed varying."""
    base = cfg.lfsr
    return BlockScrambler(ScramblerConfig(
        grid_rows=base.grid_rows, grid_cols=base.grid_cols,
        lfsr=LFSRConfig(base.lfsr.width, base.lfsr.taps, seed, base.lfsr.form,
                        base.lfsr.zero_state_policy),
        variant=base.variant, size_policy=base.size_policy,
        per_frame=base.per_frame))


def gradient_energy(img: np.ndarray) -> float:
    """RMS Sobel gradient - how much detail a crop actually carries.

    Used to order the natural scenes by content rather than by impression: the
    whole point of part 2 is that the article's number moves with the content.
    """
    import cv2

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt((gx ** 2 + gy ** 2).mean()))


def _measure(sc: BlockScrambler, img: np.ndarray) -> Dict[str, Any]:
    """One row of the article's table: scramble, descramble, correlate both."""
    scrambled = sc.scramble(img, 0)
    restored = sc.descramble(scrambled, 0, img.shape)
    r_scr, sim_scr = correlation_similarity(sc._fit(img), scrambled)
    r_dec, sim_dec = correlation_similarity(img, restored)
    return {
        "corr_original_vs_scrambled": round(r_scr, 6),
        "similarity_scrambled_pct": round(sim_scr, 4),
        "corr_original_vs_decrypted": round(r_dec, 6),
        "similarity_decrypted_pct": round(sim_dec, 4),
        "exact_recovery": bool(np.array_equal(img, restored)),
        "_scrambled": scrambled,
        "_restored": restored,
    }


def natural_frames(height: int, width: int,
                   scenes: Sequence[Tuple[str, str]] = NATURAL_SCENES
                   ) -> List[Tuple[str, str, np.ndarray]]:
    """One frame per named crop of the committed UAV photograph."""
    from avsec.sources.drone import DRONE_SCENES, drone_suite

    wanted = [s[0] for s in scenes]
    picked = [d for d in DRONE_SCENES if d[0] in wanted]
    order = {name: i for i, name in enumerate(wanted)}
    picked.sort(key=lambda d: order[d[0]])
    sources = drone_suite(height, width, n_frames=1, scenes=picked)
    labels = dict(scenes)
    return [(s.name.replace("uav_", ""), labels[s.name.replace("uav_", "")],
             s.frames[0]) for s in sources]


def run_lab(cfg: ExperimentConfig, output_dir: str = "results/lab",
            height: int = 288, width: int = 384,
            seeds: Sequence[int] = ISITIA_SEEDS,
            progress: Optional[Any] = None) -> Dict[str, Any]:
    """Reproduce the ISITIA 2021 scheme and repeat it on real photographs."""
    from avsec import sources as src_mod

    # Read the code's provenance BEFORE writing anything: this directory is
    # tracked, so the run would otherwise report the working tree as dirty
    # because of its own output.
    env = environment_record()
    out = ensure_dir(output_dir)
    img_dir = ensure_dir(os.path.join(out, "images"))
    rows, cols = cfg.lfsr.grid_rows, cfg.lfsr.grid_cols

    def _say(stage: str, frac: float) -> None:
        if progress:
            try:
                progress(stage, frac, {})
            except Exception:
                pass

    # ---- part 1: the article's own tables --------------------------------
    _say("реплiка таблиць I–III", 0.05)
    table: List[Dict[str, Any]] = []
    for seed in seeds:
        sc = _scrambler(cfg, seed)
        for name in SYNTHETIC_IMAGES:
            img = src_mod.PATTERNS[name](height, width)
            m = _measure(sc, img)
            table.append({"seed": seed, "image": name,
                          **{k: v for k, v in m.items() if not k.startswith("_")}})

    scr_vals = [r["similarity_scrambled_pct"] for r in table]
    dec_vals = [r["similarity_decrypted_pct"] for r in table]
    part1 = {
        "n_cells": len(table),
        "seeds": list(seeds),
        "images": list(SYNTHETIC_IMAGES),
        "mean_similarity_scrambled_pct": round(float(np.mean(scr_vals)), 4),
        "mean_similarity_decrypted_pct": round(float(np.mean(dec_vals)), 4),
        "max_similarity_scrambled_pct": round(float(np.max(scr_vals)), 4),
        "min_similarity_scrambled_pct": round(float(np.min(scr_vals)), 4),
        "paper_mean_similarity_scrambled_pct": PAPER_MEAN_SCRAMBLED_PCT,
        "paper_mean_similarity_decrypted_pct": PAPER_MEAN_DECRYPTED_PCT,
        "all_exact_recovery": all(r["exact_recovery"] for r in table),
    }

    # ---- part 2: the same scheme on photographs --------------------------
    _say("природні знімки", 0.45)
    natural: List[Dict[str, Any]] = []
    thumbs: List[Dict[str, Any]] = []
    try:
        frames = natural_frames(height, width)
    except FileNotFoundError as exc:
        frames = []
        natural_note = str(exc)
    else:
        natural_note = ""
        sc = _scrambler(cfg, ISITIA_FIGURE_SEED)
        for scene, label, img in frames:
            m = _measure(sc, img)
            natural.append({"seed": ISITIA_FIGURE_SEED, "scene": scene,
                            "label": label,
                            "gradient_energy": round(gradient_energy(img), 3),
                            **{k: v for k, v in m.items()
                               if not k.startswith("_")}})
            thumbs.append({"scene": scene, "label": label, "original": img,
                           "scrambled": m["_scrambled"],
                           "restored": m["_restored"],
                           "gradient": gradient_energy(img),
                           "similarity": m["similarity_scrambled_pct"]})
            from avsec.evaluation import save_image

            for kind, arr in (("original", img), ("scrambled", m["_scrambled"]),
                              ("decrypted", m["_restored"])):
                save_image(os.path.join(img_dir, f"{scene}_{kind}.png"), arr)

    part2: Dict[str, Any] = {"n_scenes": len(natural), "note": natural_note}
    if natural:
        vals = [r["similarity_scrambled_pct"] for r in natural]
        part2.update({
            "seed": ISITIA_FIGURE_SEED,
            "mean_similarity_scrambled_pct": round(float(np.mean(vals)), 4),
            "min_similarity_scrambled_pct": round(float(np.min(vals)), 4),
            "max_similarity_scrambled_pct": round(float(np.max(vals)), 4),
            "spread_pct": round(float(np.max(vals) - np.min(vals)), 4),
            "all_exact_recovery": all(r["exact_recovery"] for r in natural),
            "source_photo": "data/real/curonian_spit_epha_dune.jpg",
        })

    # ---- the same photographs, attacked ----------------------------------
    # A number measured on a procedural pattern says what the attack does to a
    # procedural pattern.  These are the crops a UAV downlink actually carries.
    _say("атаки на тих самих знімках", 0.65)
    attacked: List[Dict[str, Any]] = []
    if frames:
        from avsec.attacks import attack_boundary_reassembly, attack_known_pair

        sc = _scrambler(cfg, ISITIA_FIGURE_SEED)
        perm = sc.permutation(0)
        for scene, label, img in frames:
            fitted = sc._fit(img)
            scrambled = sc.scramble(img, 0)
            kp = attack_known_pair(fitted, scrambled, rows, cols)
            ba = attack_boundary_reassembly(scrambled, rows, cols, perm, fitted)
            attacked.append({
                "scene": scene,
                "known_pair_permutation_accuracy": round(float(
                    (kp.recovered_permutation == perm).mean()), 4),
                "known_pair_seconds": round(kp.seconds, 4),
                "boundary_direct_accuracy": round(
                    float(ba.metrics.get("direct_accuracy", float("nan"))), 4),
                "boundary_neighbour_accuracy": round(
                    float(ba.metrics.get("neighbour_accuracy", float("nan"))), 4),
                "boundary_psnr_db": round(
                    float(ba.metrics.get("reconstruction_psnr_db", float("nan"))), 3),
                "boundary_seconds": round(ba.seconds, 4),
            })
            if ba.reconstructed is not None:
                from avsec.evaluation import save_image

                save_image(os.path.join(img_dir, f"{scene}_boundary_attack.png"),
                           ba.reconstructed)

    # ---- one figure covering both parts ----------------------------------
    _say("рисунок", 0.8)
    fig_paths = _figure(table, natural, thumbs, part1, part2, out, rows, cols)

    write_csv(os.path.join(out, "table_seeds.csv"), table)
    if natural:
        write_csv(os.path.join(out, "natural_photos.csv"), natural)
    if attacked:
        write_csv(os.path.join(out, "attacks_natural.csv"), attacked)

    result = {
        "kind": "lab",
        "title": "ISITIA 2021: відтворення і повтор на природних знімках",
        "grid": f"{rows}x{cols}",
        "frame": f"{width}x{height}",
        "lfsr": cfg.lfsr.describe(),
        "part1_replica": part1,
        "part2_natural": part2,
        "part3_attacks_on_photographs": attacked,
        "figures": fig_paths,
        "run_id": cfg.run_identity(),
        "commit": env.get("git_commit"),
        "git_worktree": env.get("git_worktree"),
        "environment": {k: env[k] for k in ("python", "platform", "packages")
                        if k in env},
        "similarity_definition": (
            "«подібність» статті = |r|·100, де r — коефіцієнт кореляції Пірсона "
            "між оригіналом і перемішаним зображенням. Це статистична міра "
            "схожості, а НЕ показник конфіденційності: перестановка не змінює "
            "жодного пікселя, тож гістограма й локальні статистики лишаються "
            "недоторканими"),
    }
    write_json(os.path.join(out, "lab.json"), result)
    _say("готово", 1.0)
    return result


def _figure(table: List[Dict[str, Any]], natural: List[Dict[str, Any]],
            thumbs: List[Dict[str, Any]], part1: Dict[str, Any],
            part2: Dict[str, Any], out_dir: str, rows: int, cols: int
            ) -> Dict[str, str]:
    """One picture that carries both parts, saved as PNG, SVG and PDF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    n_thumb = max(1, len(thumbs))
    fig = plt.figure(figsize=(11.6, 3.6 + 1.75 * n_thumb))
    gs = gridspec.GridSpec(2 + n_thumb, 3, figure=fig,
                           height_ratios=[1.45, 0.30] + [1.0] * n_thumb,
                           hspace=0.14, wspace=0.06,
                           left=0.085, right=0.985, top=0.93, bottom=0.075)

    # --- A: the replica of the article's tables ---------------------------
    ax = fig.add_subplot(gs[0, :2])
    seeds = part1["seeds"]
    images = part1["images"]
    width = 0.24
    x = np.arange(len(seeds))
    colours = {"smooth": "#4fc3f7", "edges": "#7e57c2", "texture": "#26a69a"}
    for k, name in enumerate(images):
        vals = [next(r["similarity_scrambled_pct"] for r in table
                     if r["seed"] == s and r["image"] == name) for s in seeds]
        bars = ax.bar(x + (k - 1) * width, vals, width,
                      color=colours.get(name, "#90a4ae"), label=name)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5)
    ours = part1["mean_similarity_scrambled_pct"]
    ax.axhline(PAPER_MEAN_SCRAMBLED_PCT, color="#b71c1c", ls="--", lw=1.4,
               label=f"стаття: {PAPER_MEAN_SCRAMBLED_PCT:.2f} %")
    ax.axhline(ours, color="#37474f", ls=":", lw=1.4,
               label=f"наше середнє: {ours:.2f} %")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("«подібність» = |r| · 100, %")
    ax.set_title("A. Репліка таблиць I–III статті: оригінал проти перемішаного",
                 fontsize=10.5)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, max(max(r["similarity_scrambled_pct"] for r in table),
                       PAPER_MEAN_SCRAMBLED_PCT) * 1.45)

    # --- B: the same quantity on photographs ------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    if natural:
        names = [r["scene"] for r in natural]
        vals = [r["similarity_scrambled_pct"] for r in natural]
        bars = ax2.barh(np.arange(len(names)), vals, color="#ef6c00")
        xmax = max(vals + [PAPER_MEAN_SCRAMBLED_PCT]) * 1.45
        for b, v, nm in zip(bars, vals, names):
            y = b.get_y() + b.get_height() / 2
            ax2.annotate(f"{v:.2f}", (v, y), xytext=(5, 0),
                         textcoords="offset points", va="center", fontsize=8)
            # the name rides on the bar when the bar is long enough to hold it,
            # and follows the value when it is not
            if v > 0.3 * xmax:
                ax2.annotate(nm, (0, y), xytext=(6, 0),
                             textcoords="offset points", va="center",
                             ha="left", fontsize=7.5, color="#ffffff",
                             weight="bold")
            else:
                ax2.annotate(nm, (v, y), xytext=(38, 0),
                             textcoords="offset points", va="center",
                             ha="left", fontsize=7.5, color="#37474f")
        ax2.axvline(PAPER_MEAN_SCRAMBLED_PCT, color="#b71c1c", ls="--", lw=1.3)
        ax2.set_yticks([])
        ax2.invert_yaxis()
        ax2.set_xlabel("|r| · 100, %", fontsize=8.5, labelpad=1)
        ax2.set_xlim(0, xmax)
        ax2.set_title(f"B. Ті самі обчислення на знімку\nз БпЛА "
                      f"(seed {ISITIA_FIGURE_SEED})", fontsize=10.5)
    else:
        ax2.text(0.5, 0.5, "знімок недоступний", ha="center", va="center",
                 fontsize=10, color="#b71c1c")
        ax2.set_axis_off()

    # --- C: original / scrambled / decrypted, per scene -------------------
    for i, t in enumerate(thumbs):
        for j, (kind, key) in enumerate((("оригінал", "original"),
                                         ("перемішано", "scrambled"),
                                         ("розшифровано", "restored"))):
            a = fig.add_subplot(gs[2 + i, j])
            a.imshow(t[key], cmap="gray", vmin=0, vmax=255)
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title(kind, fontsize=10)
            if j == 0:
                a.set_ylabel(f"{t['scene']}\n|r|·100 = {t['similarity']:.2f} %",
                             fontsize=8)
            if j == 2 and t.get("gradient") is not None:
                a.annotate(f"деталізація {t['gradient']:.0f}",
                           (0.98, 0.03), xycoords="axes fraction",
                           ha="right", fontsize=7, color="#ffffff",
                           bbox=dict(boxstyle="round,pad=0.22", fc="#37474f",
                                     ec="none", alpha=0.75))

    fig.suptitle("ISITIA 2021: відтворення схеми і повтор на природних знімках",
                 fontsize=13, y=0.995)
    spread = part2.get("spread_pct")
    foot = (
        f"Сітка {rows}×{cols}. Панель A — репліка таблиць статті: три seed-и × "
        f"три зображення, середнє {ours:.2f} % проти заявлених "
        f"{PAPER_MEAN_SCRAMBLED_PCT:.2f} %. Панель B — та сама величина на "
        f"чотирьох вирізках справжнього знімка з БпЛА"
        + (f": розкид {spread:.2f} відсоткових пунктів між сценами при ОДНОМУ "
           f"й тому самому seed." if spread is not None else ".")
        + " Розшифрування точне до біта в усіх клітинках. Низька кореляція "
          "НЕ означає конфіденційності: перестановка не змінює жодного пікселя."
    )
    fig.text(0.012, 0.012, foot, fontsize=7.6, color="#37474f", wrap=True)

    paths: Dict[str, str] = {}
    for ext in ("png", "svg", "pdf"):
        p = os.path.join(out_dir, f"lab_isitia.{ext}")
        fig.savefig(p, dpi=140 if ext == "png" else None)
        paths[ext] = p
    plt.close(fig)
    return paths


__all__ = ["gradient_energy", "ISITIA_SEEDS", "ISITIA_FIGURE_SEED", "PAPER_MEAN_SCRAMBLED_PCT",
           "PAPER_MEAN_DECRYPTED_PCT", "SYNTHETIC_IMAGES", "NATURAL_SCENES",
           "natural_frames", "run_lab"]
