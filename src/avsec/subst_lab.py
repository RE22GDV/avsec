"""B2s bench: the substitution table, the attacks against it, and the channel.

This is the runner behind ``docs/substitution_b2s.md``.  It answers four
questions in one command, and writes the evidence for each next to the number:

**1. Is the transform correct and is the table really key-only?**
   The S-box and its inverse are exported in full, and the round trip is
   checked for bit-exact recovery in every mode.

**2. What does substitution actually hide?**
   A permutation leaves the histogram of the frame untouched.  A substitution
   relabels the bins - so the histogram changes, but the *sorted* histogram
   does not.  Both facts are measured rather than asserted, because the second
   one is what the known-pair attack lives on.

**3. What do the attacks do now?**
   The five attacks of :mod:`avsec.attacks` are pointed at ``B2`` and ``B2s``
   side by side, plus the substitution-aware known-pair attack.  The reassembly
   attack is the interesting one: it is the only attack that needs no key and
   no known plaintext, and substitution is aimed exactly at it.

**4. What does it cost on an analog path?**
   A substitution table does not preserve closeness: neighbouring values are
   scattered across the alphabet, so an amplitude error of a few levels becomes
   an arbitrary value after the inverse table.  Every channel profile is run
   for ``B1``, ``B2`` and ``B2s`` so the size of that effect is a measurement.

Usage::

    avsec subst-lab                     # -> results/b2s/
    avsec subst-lab --output runs/b2s
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from avsec.attacks import (
    apply_recovered,
    attack_boundary_reassembly,
    attack_known_pair,
    attack_multi_frame_reuse,
    attack_substitution_known_pair,
    split_tiles,
)
from avsec.config import ExperimentConfig
from avsec.substitution import (
    SUBSTITUTION_MODES,
    SubstitutionPermutationScrambler,
    crypto_sbox,
    invert_sbox,
)
from avsec.utils import ensure_dir, environment_record, write_csv, write_json

#: Channel profiles, ordered by severity.  Not averaged: different conditions.
CHANNELS: Tuple[str, ...] = ("clean", "mild", "moderate", "bursty", "harsh")

#: Scenes used for the picture examples and the channel sweep.
BENCH_SCENES: Tuple[Tuple[str, str], ...] = (
    ("sky_horizon", "море й небо над косою"),
    ("village", "село — поле дрібних яскравих об'єктів"),
)


# --------------------------------------------------------------- the table
def sbox_tables(key: bytes, session_id: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """One demonstration S-box and its inverse, for printing in full."""
    box = crypto_sbox(key, b"|" + session_id + b"|session")
    return box, invert_sbox(box)


def _table_rows(table: np.ndarray, label: str) -> List[Dict[str, Any]]:
    """The 16x16 layout every cipher specification prints."""
    out = []
    for hi in range(16):
        row: Dict[str, Any] = {"table": label, "рядок": f"{hi:X}x"}
        for lo in range(16):
            row[f"{lo:X}"] = f"0x{int(table[hi * 16 + lo]):02X}"
        out.append(row)
    return out


def _sanity(box: np.ndarray, inv: np.ndarray) -> Dict[str, Any]:
    """Properties a substitution table must have, checked rather than claimed."""
    v = np.arange(256, dtype=np.int64)
    fixed = int((box.astype(np.int64) == v).sum())
    # how far the table moves a value: a table that mostly keeps values near
    # themselves would preserve the smoothness the reassembly attack needs
    shift = np.abs(box.astype(np.int64) - v)
    return {
        "bijective": bool(sorted(box.tolist()) == list(range(256))),
        "inverse_exact": bool(np.array_equal(inv[box.astype(np.int64)], v)),
        "fixed_points": fixed,
        "fixed_points_expected": 1.0,
        "mean_abs_shift": round(float(shift.mean()), 2),
        "mean_abs_shift_random": 85.33,
        "table_space": "256! ≈ 8,6 · 10^506",
    }


# ------------------------------------------------------------- what is hidden
def histogram_facts(original: np.ndarray, permuted: np.ndarray,
                    subst_global: np.ndarray,
                    subst_block: np.ndarray) -> Dict[str, Any]:
    """Which statistics each primitive leaves untouched.

    Three transforms of the same frame, because they hide different things.
    The permutation leaves the histogram byte for byte.  One global table
    permutes the bins - so the histogram changes while the *sorted* histogram
    does not, and that invariant is what the known-pair attack matches on.  A
    table per block destroys the invariant as well.
    """
    def h(x):
        return np.bincount(x.ravel().astype(np.int64), minlength=256)

    h0, hp, hg, hb = h(original), h(permuted), h(subst_global), h(subst_block)
    return {
        "hist_identical_after_permutation": bool(np.array_equal(h0, hp)),
        "hist_l1_after_permutation": int(np.abs(h0 - hp).sum()),
        "hist_identical_after_global_substitution": bool(np.array_equal(h0, hg)),
        "hist_l1_after_global_substitution": int(np.abs(h0 - hg).sum()),
        "sorted_hist_identical_after_global_substitution":
            bool(np.array_equal(np.sort(h0), np.sort(hg))),
        "sorted_hist_identical_after_block_substitution":
            bool(np.array_equal(np.sort(h0), np.sort(hb))),
        "sorted_hist_l1_after_block_substitution":
            int(np.abs(np.sort(h0) - np.sort(hb)).sum()),
        "note": ("перестановка не змінює жодного значення, тож гістограма "
                 "збігається побітово; одна глобальна таблиця переставляє "
                 "стовпчики гістограми, але впорядкована гістограма лишається "
                 "тією самою — саме на цьому інваріанті працює атака за "
                 "відомою парою; окрема таблиця на кожен блок руйнує й цей "
                 "інваріант"),
    }


def error_amplification(box: np.ndarray,
                        deltas: Sequence[int] = (1, 2, 4, 8, 16)) -> List[Dict[str, Any]]:
    """How an error on the wire grows after the inverse table.

    A permutation transports values unchanged, so an amplitude error of ``d``
    levels stays an error of ``d`` levels.  A substitution table is a bijection
    with no order structure: neighbouring values are scattered across the whole
    alphabet, so the same error becomes an essentially arbitrary value.  This is
    the mechanism behind the channel sweep, measured directly.

    ``mean_abs_error_random`` is the reference: the mean absolute difference of
    two independent uniform values on ``0..255`` is ``255 / 3 = 85``.
    """
    inv = invert_sbox(box).astype(np.int64)
    v = np.arange(256, dtype=np.int64)
    out: List[Dict[str, Any]] = []
    for d in deltas:
        shifted = np.clip(v + int(d), 0, 255)
        err = np.abs(inv[shifted] - inv[v])
        out.append({
            "wire_error_levels": int(d),
            "permutation_error_levels": int(d),
            "substitution_mean_abs_error": round(float(err.mean()), 2),
            "substitution_median_abs_error": float(np.median(err)),
            "mean_abs_error_random": 85.0,
        })
    return out


# ------------------------------------------------------------------ attacks
def _b2(cfg: ExperimentConfig, key: bytes, sid: bytes, per_frame: bool = True):
    from avsec.baselines import CryptoPermutationScrambler

    rows, cols = cfg.b2_grid
    return CryptoPermutationScrambler(rows, cols, key, sid, per_frame=per_frame)


class _Adapter:
    """Enough of the BlockScrambler surface for the multi-frame attack."""

    def __init__(self, inner, cfg):
        self._inner = inner
        self.cfg = cfg.lfsr

    def scramble(self, img, i):
        return self._inner.scramble(img, i)

    def permutation(self, i=0):
        return self._inner.permutation(i)

    def _fit(self, img):
        return self._inner._fit(img) if hasattr(self._inner, "_fit") else img


def attack_table(cfg: ExperimentConfig, key: bytes, sid: bytes,
                 frames: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
    """Every attack against B2 and against B2s, on identical content."""
    rows, cols = cfg.b2_grid
    img = frames[0]
    out: List[Dict[str, Any]] = []

    targets: List[Tuple[str, Any]] = [("B2", _b2(cfg, key, sid))]
    targets += [(f"B2s ({m})",
                 SubstitutionPermutationScrambler(rows, cols, key, sid, mode=m))
                for m in SUBSTITUTION_MODES]

    for label, sc in targets:
        fit = sc._fit(img) if hasattr(sc, "_fit") else img
        ciph = sc.scramble(img, 0)
        perm = sc.permutation(0)

        kp = attack_known_pair(fit, ciph, rows, cols)
        kp_acc = float((kp.recovered_permutation == perm).mean())

        sk = attack_substitution_known_pair(fit, ciph, rows, cols, perm)

        ba = attack_boundary_reassembly(ciph, rows, cols, perm, fit)

        mf = attack_multi_frame_reuse(list(frames), _Adapter(sc, cfg), rows, cols)

        out.append({
            "target": label,
            "known_pair_plain_accuracy": round(kp_acc, 4),
            "known_pair_plain_seconds": round(kp.seconds, 4),
            "known_pair_subst_accuracy":
                round(sk.metrics.get("permutation_accuracy", 0.0), 4),
            "known_pair_subst_seconds": round(sk.seconds, 4),
            "substitution_detected_global":
                int(sk.metrics.get("substitution_is_global", -1)),
            "merged_alphabet_covered":
                round(sk.metrics.get("merged_alphabet_covered", 0.0), 4),
            "boundary_neighbour_pct": round(ba.metrics["neighbour_accuracy"] * 100, 2),
            "boundary_direct_pct": round(ba.metrics["direct_accuracy"] * 100, 2),
            "boundary_psnr_db": round(ba.metrics["reconstruction_psnr_db"], 2),
            "boundary_seconds": round(ba.seconds, 4),
            "multi_frame_accuracy": round(mf.metrics["permutation_accuracy"], 4),
            "chance_level": round(1.0 / (rows * cols), 6),
        })
    return out


def reuse_matrix(cfg: ExperimentConfig, key: bytes, sid: bytes,
                 f0: np.ndarray, f1: np.ndarray) -> List[Dict[str, Any]]:
    """Does what the attacker learned on one frame decrypt the next one?

    Two axes, because two primitives can each be static or fresh: the block
    permutation and the substitution table.  The cell reports the quality of
    frame 1 decrypted with what frame 0 gave away.
    """
    rows, cols = cfg.b2_grid
    out: List[Dict[str, Any]] = []
    for static_perm in (True, False):
        for mode in SUBSTITUTION_MODES:
            sc = SubstitutionPermutationScrambler(
                rows, cols, key, sid, mode=mode, per_frame=not static_perm)
            c0, fit0 = sc.scramble(f0, 0), sc._fit(f0)
            r = attack_substitution_known_pair(fit0, c0, rows, cols, sc.permutation(0))
            c1, fit1 = sc.scramble(f1, 1), sc._fit(f1)
            rec = apply_recovered(c1, r.recovered_permutation, r.recovered_tables,
                                  rows, cols)
            mse = float(((rec.astype(np.float64) - fit1.astype(np.float64)) ** 2).mean())
            psnr = 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))
            out.append({
                "permutation": "стала" if static_perm else "щокадрова",
                "substitution_mode": mode,
                "frame0_permutation_accuracy":
                    round(r.metrics.get("permutation_accuracy", 0.0), 4),
                "frame1_psnr_db": round(psnr, 2),
                "frame1_readable": bool(psnr >= 20.0),
            })
    return out


# ------------------------------------------------------------ analog channel
def channel_sweep(cfg: ExperimentConfig, master, scenes: Sequence[Tuple[str, Any]],
                  n_frames: int = 3, progress=None) -> Tuple[List[Dict[str, Any]],
                                                             Dict[str, Any]]:
    """B1, B2 and B2s over every channel profile, on identical traces."""
    from avsec.baselines import make_b1_lfsr, make_b2_cryptoperm
    from avsec.channel import ChannelTrace, preset
    from avsec.substitution import make_b2s_subst_perm

    rows, cols = cfg.b2_grid
    H, W = cfg.frame_height, cfg.frame_width
    transport = cfg.profile("B4").transport_config(cfg.budget)
    out: List[Dict[str, Any]] = []
    examples: Dict[str, Any] = {}

    for ci, chan_name in enumerate(CHANNELS):
        if progress:
            progress(f"канал {chan_name}", 0.45 + 0.4 * ci / len(CHANNELS), {})
        chan = preset(chan_name)
        builders = {
            "B1": lambda: make_b1_lfsr(transport, chan, H, W, cfg.lfsr),
            "B2": lambda: make_b2_cryptoperm(transport, chan, H, W, rows, cols,
                                             master, session_id=b"B2-BNCH0"),
            "B2s": lambda: make_b2s_subst_perm(transport, chan, H, W, rows, cols,
                                               master, session_id=b"B2sBNCH0",
                                               mode="block"),
        }
        for scene_name, frames in scenes:
            trace = ChannelTrace(seed=cfg.channel_seed_value, scene=scene_name,
                                 repetition=0, profile=chan_name)
            for mname, build in builders.items():
                method = build()
                method.reset()
                psnrs, covs = [], []
                for fid in range(min(n_frames, len(frames))):
                    res = method.process(frames[fid], fid, trace)
                    psnrs.append(float(res.metrics.psnr_full))
                    covs.append(float(res.metrics.coverage))
                    if fid == 0 and scene_name == scenes[0][0]:
                        examples.setdefault(chan_name, {})[mname] = {
                            "transmitted": res.transmitted,
                            "received": res.received,
                            "reconstructed": res.reconstructed,
                            "original": res.original,
                        }
                finite = [p for p in psnrs if np.isfinite(p)]
                out.append({
                    "channel": chan_name, "method": mname, "scene": scene_name,
                    "n_frames": len(psnrs),
                    "psnr_full_db": round(float(np.mean(finite)), 2) if finite else None,
                    "psnr_min_db": round(float(np.min(finite)), 2) if finite else None,
                    "coverage": round(float(np.mean(covs)), 4),
                })
    return out, examples


# ------------------------------------------------------------------ the run
def run_subst_lab(cfg: ExperimentConfig, output_dir: str = "results/b2s",
                  n_frames: int = 3, progress=None) -> Dict[str, Any]:
    """Everything above, written to ``output_dir``."""
    from avsec.sources.drone import DRONE_SCENES, drone_suite

    # provenance first: this directory is tracked, so reading it after writing
    # would report the run's own output as a dirty working tree
    env = environment_record()
    out = ensure_dir(output_dir)
    img_dir = ensure_dir(os.path.join(out, "images"))
    rows, cols = cfg.b2_grid
    H, W = cfg.frame_height, cfg.frame_width

    def say(stage: str, frac: float) -> None:
        if progress:
            try:
                progress(stage, frac, {})
            except Exception:
                pass

    master = cfg.master_secret()
    sid = b"B2sBNCH0"
    from avsec.crypto import derive_session_keys

    key = derive_session_keys(master, sid).key

    # ---- 1. the table itself -------------------------------------------
    say("таблиця замін", 0.05)
    box, inv = sbox_tables(key, sid)
    write_csv(os.path.join(out, "sbox.csv"), _table_rows(box, "S-Box"))
    write_csv(os.path.join(out, "sbox_inverse.csv"), _table_rows(inv, "S-Box⁻¹"))
    sanity = _sanity(box, inv)

    # ---- 2. what each primitive hides -----------------------------------
    say("гістограми", 0.15)
    want = [s[0] for s in BENCH_SCENES]
    picked = [d for d in DRONE_SCENES if d[0] in want]
    picked.sort(key=lambda d: want.index(d[0]))
    suite = drone_suite(H, W, n_frames=max(4, n_frames), scenes=picked)
    scenes = [(s.name.replace("uav_", ""), s.frames) for s in suite]
    f0 = scenes[0][1][0]
    f1 = scenes[0][1][1]

    sc_demo = SubstitutionPermutationScrambler(rows, cols, key, sid, mode="block")
    sc_global = SubstitutionPermutationScrambler(rows, cols, key, sid, mode="session")
    b2_demo = _b2(cfg, key, sid)
    fit = sc_demo._fit(f0)
    hist = histogram_facts(fit, b2_demo.scramble(f0, 0),
                           sc_global.substitute(f0, 0), sc_demo.substitute(f0, 0))
    amplification = error_amplification(box)

    # ---- 3. correctness --------------------------------------------------
    say("перевірка оборотності", 0.25)
    correctness = []
    for mode in SUBSTITUTION_MODES:
        sc = SubstitutionPermutationScrambler(rows, cols, key, sid, mode=mode)
        ok = all(np.array_equal(sc.descramble(sc.scramble(scenes[0][1][i], i), i),
                                sc._fit(scenes[0][1][i]))
                 for i in range(min(3, len(scenes[0][1]))))
        correctness.append({
            "substitution_mode": mode,
            "tables_per_frame": sc.subst.n_tables(),
            "bit_exact_recovery": bool(ok),
            "frames_checked": min(3, len(scenes[0][1])),
        })

    # ---- 4. attacks ------------------------------------------------------
    say("атаки", 0.3)
    attacks = attack_table(cfg, key, sid, scenes[0][1][:12])
    reuse = reuse_matrix(cfg, key, sid, f0, f1)

    # ---- 5. the analog channel -------------------------------------------
    sweep, examples = channel_sweep(cfg, master, scenes, n_frames, progress)

    # ---- 6. pictures ------------------------------------------------------
    say("зображення", 0.88)
    stages = _stage_images(sc_demo, b2_demo, f0)
    _save_images(img_dir, stages, examples)

    say("рисунки", 0.93)
    figures = _figures(out, box, inv, stages, attacks, sweep, hist)

    summary = {
        "kind": "b2s",
        "title": "B2s: перестановка блоків разом із заміною значень пікселів",
        "grid": f"{rows}x{cols}",
        "frame": f"{W}x{H}",
        "sbox": sanity,
        "histograms": hist,
        "error_amplification": amplification,
        "correctness": correctness,
        "attacks": attacks,
        "reuse_matrix": reuse,
        "channel_sweep": sweep,
        "figures": figures,
        "run_id": cfg.run_identity(),
        "commit": env.get("git_commit"),
        "git_worktree": env.get("git_worktree"),
        "environment": env.get("environment", env),
    }
    write_json(os.path.join(out, "b2s.json"), summary)
    write_csv(os.path.join(out, "attacks.csv"), attacks)
    write_csv(os.path.join(out, "reuse_matrix.csv"), reuse)
    write_csv(os.path.join(out, "channel_sweep.csv"), sweep)
    write_csv(os.path.join(out, "correctness.csv"), correctness)
    write_csv(os.path.join(out, "error_amplification.csv"), amplification)
    say("готово", 1.0)
    return summary


def _stage_images(sc, b2, frame: np.ndarray) -> Dict[str, np.ndarray]:
    """The frame at each stage of the transform, for the picture examples."""
    fit = sc._fit(frame)
    return {
        "1_original": fit,
        "2_substituted": sc.substitute(frame, 0),
        "3_substituted_permuted": sc.scramble(frame, 0),
        "4_permutation_only_B2": b2.scramble(frame, 0),
        "5_restored": sc.descramble(sc.scramble(frame, 0), 0),
    }


def _save_images(img_dir: str, stages: Dict[str, np.ndarray],
                 examples: Dict[str, Any]) -> None:
    from PIL import Image

    for name, arr in stages.items():
        Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(
            os.path.join(img_dir, f"{name}.png"))
    for chan, per_method in examples.items():
        for method, imgs in per_method.items():
            for what, arr in imgs.items():
                if arr is None:
                    continue
                a = np.asarray(arr)
                a = np.clip(a, 0, 255).astype(np.uint8)
                Image.fromarray(a).save(
                    os.path.join(img_dir, f"{chan}_{method}_{what}.png"))


def _figures(out_dir: str, box: np.ndarray, inv: np.ndarray,
             stages: Dict[str, np.ndarray], attacks: List[Dict[str, Any]],
             sweep: List[Dict[str, Any]], hist: Dict[str, Any]) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: Dict[str, str] = {}

    # -- figure 1: the tables ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4))
    for ax, table, title in ((axes[0], box, "Таблиця замін S-Box"),
                             (axes[1], inv, "Обернена таблиця S-Box⁻¹")):
        grid = table.astype(np.int64).reshape(16, 16)
        ax.imshow(grid, cmap="viridis", vmin=0, vmax=255)
        for i in range(16):
            for j in range(16):
                ax.text(j, i, f"{grid[i, j]:02X}", ha="center", va="center",
                        fontsize=6.4,
                        color="#ffffff" if grid[i, j] < 150 else "#101010")
        ax.set_xticks(range(16))
        ax.set_xticklabels([f"{v:X}" for v in range(16)], fontsize=7)
        ax.set_yticks(range(16))
        ax.set_yticklabels([f"{v:X}x" for v in range(16)], fontsize=7)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("молодша півбайта", fontsize=8)
        ax.set_ylabel("старша півбайта", fontsize=8)
    fig.suptitle("Ключова таблиця замін: значення 0x00…0xFF і обернене перетворення",
                 fontsize=12.5)
    fig.text(0.012, 0.01,
             "Клітинка (рядок, стовпець) дає образ значення 0xРС. Таблиця "
             "бієктивна, тож обернена відновлює кожне значення точно. "
             "Обидві виводяться з ключа і без нього не відтворюються.",
             fontsize=7.6, color="#37474f", wrap=True)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    p = os.path.join(out_dir, "sbox_tables.png")
    fig.savefig(p, dpi=140)
    fig.savefig(p.replace(".png", ".svg"))
    plt.close(fig)
    paths["sbox_tables"] = p

    # -- figure 2: stages of the transform ------------------------------
    titles = {
        "1_original": "1. Вихідний кадр",
        "2_substituted": "2. Після заміни значень",
        "3_substituted_permuted": "3. Після заміни й перестановки\n(те, що йде в канал)",
        "4_permutation_only_B2": "4. Лише перестановка (B2),\nдля порівняння",
        "5_restored": "5. Відновлено з ключем",
    }
    order = ["1_original", "2_substituted", "3_substituted_permuted",
             "4_permutation_only_B2", "5_restored"]
    fig, axes = plt.subplots(2, 5, figsize=(15.5, 6.2),
                             gridspec_kw={"height_ratios": [3, 1.5]})
    for k, name in enumerate(order):
        arr = np.asarray(stages[name], dtype=np.uint8)
        axes[0, k].imshow(arr, cmap="gray", vmin=0, vmax=255)
        axes[0, k].set_xticks([])
        axes[0, k].set_yticks([])
        axes[0, k].set_title(titles[name], fontsize=9)
        axes[1, k].hist(arr.ravel(), bins=64, range=(0, 255), color="#5c6bc0")
        axes[1, k].set_yticks([])
        axes[1, k].tick_params(labelsize=6.5)
        axes[1, k].set_xlabel("значення пікселя", fontsize=7)
    axes[1, 0].set_ylabel("гістограма", fontsize=8)
    fig.suptitle("Що саме робить кожен примітив: кадр угорі, його гістограма внизу",
                 fontsize=12.5)
    fig.text(0.012, 0.012,
             "Перестановка (панель 4) не змінює жодного значення, тому її "
             "гістограма збігається з вихідною побітово. Заміна (панель 2) "
             "переставляє стовпчики гістограми. Панель 5 — точне відновлення "
             "з ключем.",
             fontsize=7.6, color="#37474f", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    p = os.path.join(out_dir, "stages.png")
    fig.savefig(p, dpi=140)
    fig.savefig(p.replace(".png", ".svg"))
    plt.close(fig)
    paths["stages"] = p

    # -- figure 3: attacks and the channel cost --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2))
    labels = [a["target"] for a in attacks]
    x = np.arange(len(labels))
    a0 = axes[0]
    a0.bar(x - 0.2, [a["boundary_neighbour_pct"] for a in attacks], 0.4,
           label="збирання за межами, % сусідств", color="#ef5350")
    a0.bar(x + 0.2, [a["known_pair_subst_accuracy"] * 100 for a in attacks], 0.4,
           label="відома пара з урахуванням заміни, % позицій", color="#42a5f5")
    a0.set_xticks(x)
    a0.set_xticklabels(labels, fontsize=8, rotation=12, ha="right")
    a0.set_ylabel("%", fontsize=9)
    a0.set_ylim(0, 105)
    a0.legend(fontsize=8)
    a0.set_title("Атаки: що заміна зупиняє, а що ні", fontsize=11)
    a0.grid(axis="y", alpha=0.25)

    a1 = axes[1]
    for mname, colour in (("B1", "#8d6e63"), ("B2", "#ba68c8"), ("B2s", "#26a69a")):
        ys = []
        for ch in CHANNELS:
            vals = [r["psnr_full_db"] for r in sweep
                    if r["channel"] == ch and r["method"] == mname
                    and r["psnr_full_db"] is not None]
            ys.append(float(np.mean(vals)) if vals else np.nan)
        a1.plot(range(len(CHANNELS)), ys, marker="o", label=mname, color=colour)
    a1.set_xticks(range(len(CHANNELS)))
    a1.set_xticklabels(CHANNELS, fontsize=9)
    a1.set_ylabel("PSNR відновленого кадру, дБ", fontsize=9)
    a1.set_title("Ціна заміни в аналоговому тракті", fontsize=11)
    a1.axhline(20.0, color="#b71c1c", ls=":", lw=1)
    a1.annotate("робочий поріг 20 дБ", (0.02, 20.6), fontsize=7.5, color="#b71c1c")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.25)
    fig.suptitle("B2s проти B2: що заміна дає і що вона коштує", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    p = os.path.join(out_dir, "attacks_and_channel.png")
    fig.savefig(p, dpi=140)
    fig.savefig(p.replace(".png", ".svg"))
    plt.close(fig)
    paths["attacks_and_channel"] = p
    return paths


__all__ = ["CHANNELS", "BENCH_SCENES", "sbox_tables", "histogram_facts",
           "attack_table", "reuse_matrix", "channel_sweep", "run_subst_lab",
           "error_amplification"]
