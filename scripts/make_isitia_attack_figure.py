"""Figures and tables for part 3 of docs/reconstruction_2021.md.

The boundary-compatibility reassembly is the one attack that needs **no key and
no known plaintext**, so it is the one that decides how the scheme should be
described.  Its strength depends entirely on the content, and that is the point
of this picture: the same attack, the same grid, the same seed, run on a
synthetic smooth gradient and on four crops of an actual UAV photograph.

It also answers the question the attack raises: does replacing the LFSR with a
cryptographic generator help?  Two of the five attacks never look at the
generator, so they are pointed at **B2** as well and the numbers are written
next to B1's.  That turns "the weakness is the permutation, not the LFSR" from
an assertion into a measurement.

Writes ``docs/img/isitia_attacks.png`` (plus SVG) and two CSVs beside it.

Usage::

    python scripts/make_isitia_attack_figure.py
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np  # noqa: E402

from avsec import sources as S  # noqa: E402
from avsec.attacks import attack_boundary_reassembly  # noqa: E402
from avsec.config import load_config  # noqa: E402
from avsec.lab import ISITIA_FIGURE_SEED, NATURAL_SCENES, natural_frames  # noqa: E402
from avsec.lfsr import BlockScrambler, LFSRConfig, ScramblerConfig  # noqa: E402
from avsec.utils import environment_record  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PNG = os.path.join(ROOT, "docs", "img", "isitia_attacks.png")
OUT_CSV = os.path.join(ROOT, "docs", "img", "isitia_attacks.csv")
OUT_B2 = os.path.join(ROOT, "docs", "img", "isitia_b1_vs_b2.csv")


def _b1_vs_b2(cfg, h, w, rows, cols, env) -> None:
    """The same attacks against B1 and against the CSPRNG variant B2.

    The boundary solver and tile matching never see a key, so they cannot tell
    the two generators apart; the variance fingerprint and the seed search can.
    Writing all four side by side is the evidence for the B1 -> B2 step of the
    improvement chain.
    """
    from avsec.attacks import (attack_known_pair, attack_lfsr_bruteforce,
                               attack_multi_frame_reuse)
    from avsec.baselines import CryptoPermutationScrambler
    from avsec.crypto import derive_session_keys

    sc1 = BlockScrambler(cfg.lfsr)
    sid = cfg.session_id_source().next("B2-doc")
    keys = derive_session_keys(cfg.master_secret(), sid)
    sc2 = CryptoPermutationScrambler(rows, cols, keys.key, sid, per_frame=True)

    class _Adapter:
        def __init__(self, inner):
            self._inner = inner
            self.cfg = cfg.lfsr

        def scramble(self, img, i):
            return self._inner.scramble(img, i)

        def permutation(self, i=0):
            return self._inner.permutation(i)

        def _fit(self, img):
            return img

    img = S.PATTERNS["smooth"](h, w)
    frames = [S.pattern_edges(h, w, i * 0.05) for i in range(12)]
    out = []
    for label, sc in (("B1 (LFSR, стала перестановка)", sc1),
                      ("B2 (CSPRNG, щокадрово)", _Adapter(sc2))):
        scrambled = sc.scramble(img, 0)
        kp = attack_known_pair(sc._fit(img), scrambled, rows, cols)
        kp_acc = float((kp.recovered_permutation == sc.permutation(0)).mean())
        ba = attack_boundary_reassembly(scrambled, rows, cols, sc.permutation(0),
                                        sc._fit(img))
        mf = attack_multi_frame_reuse(frames, sc, rows, cols)
        row = {
            "target": label,
            "known_pair_accuracy": round(kp_acc, 4),
            "boundary_neighbour_pct": round(
                ba.metrics["neighbour_accuracy"] * 100, 2),
            "boundary_psnr_db": round(ba.metrics["reconstruction_psnr_db"], 2),
            "multi_frame_accuracy": round(
                mf.metrics["permutation_accuracy"], 4),
        }
        if sc is sc1:
            bf = attack_lfsr_bruteforce(sc._fit(img), scrambled, cfg.lfsr)
            row["seed_search_key_space"] = int(bf.metrics["key_space"])
            row["seed_search_found"] = int(bf.metrics["seed_found"])
            row["seed_search_seconds"] = round(bf.seconds, 3)
        else:
            row["seed_search_key_space"] = 2 ** 256
            row["seed_search_found"] = -1
            row["seed_search_seconds"] = -1.0
        row["chance_level"] = round(1.0 / (rows * cols), 6)
        row["commit"] = str(env.get("git_commit"))
        out.append(row)

    with open(OUT_B2, "w", encoding="utf-8", newline="") as fh:
        wri = csv.DictWriter(fh, fieldnames=list(out[0]))
        wri.writeheader()
        wri.writerows(out)
    print("\n  B1 проти B2 (та сама сітка, той самий вміст):")
    for r in out:
        print(f"    {r['target']:32s} відома пара {r['known_pair_accuracy']:.3f} | "
              f"межі {r['boundary_neighbour_pct']:5.2f} % / {r['boundary_psnr_db']:.2f} дБ | "
              f"багатокадрова {r['multi_frame_accuracy']:.4f}")
    print(f"  {OUT_B2}")


LAB_DIR = os.path.join(ROOT, "results", "lab")
LAB_COPIES = (
    ("lab_isitia.png", "isitia_lab.png"),
    ("table_seeds.csv", "isitia_lab_table_seeds.csv"),
    ("natural_photos.csv", "isitia_lab_natural.csv"),
    ("attacks_natural.csv", "isitia_lab_attacks.csv"),
)


def _copy_lab() -> None:
    """Carry the ``avsec lab`` outputs into docs/img beside their captions.

    Copying by hand is how a figure and the numbers quoted next to it drift
    apart.  The document reads from docs/img, so docs/img is refreshed from the
    latest bench run whenever this script runs.
    """
    import shutil

    if not os.path.isdir(LAB_DIR):
        print("results/lab missing - run `run.bat lab` first", file=sys.stderr)
        return
    for src, dst in LAB_COPIES:
        s_path = os.path.join(LAB_DIR, src)
        if os.path.exists(s_path):
            shutil.copyfile(s_path, os.path.join(ROOT, "docs", "img", dst))
            print(f"  docs/img/{dst}  <- results/lab/{src}")


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = environment_record()
    cfg = load_config(os.path.join(ROOT, "configs", "research_main.yaml"))
    h, w = cfg.frame_height, cfg.frame_width
    rows, cols = cfg.lfsr.grid_rows, cfg.lfsr.grid_cols
    base = cfg.lfsr
    sc = BlockScrambler(ScramblerConfig(
        grid_rows=rows, grid_cols=cols,
        lfsr=LFSRConfig(base.lfsr.width, base.lfsr.taps, ISITIA_FIGURE_SEED,
                        base.lfsr.form, base.lfsr.zero_state_policy),
        variant=base.variant, size_policy=base.size_policy,
        per_frame=base.per_frame))
    perm = sc.permutation(0)

    cases = [("smooth (синтетичний градієнт)", S.PATTERNS["smooth"](h, w), True)]
    try:
        cases += [(f"{name} (знімок з БпЛА)", img, False)
                  for name, _label, img in natural_frames(h, w)]
    except FileNotFoundError as exc:
        print(f"natural photographs unavailable: {exc}", file=sys.stderr)

    rowsout = []
    panels = []
    for title, img, synthetic in cases:
        fitted = sc._fit(img)
        scrambled = sc.scramble(img, 0)
        ba = attack_boundary_reassembly(scrambled, rows, cols, perm, fitted)
        m = ba.metrics
        rowsout.append({
            "case": title, "synthetic": synthetic,
            "neighbour_accuracy_pct": round(m["neighbour_accuracy"] * 100, 2),
            "direct_accuracy_pct": round(m["direct_accuracy"] * 100, 2),
            "psnr_db": round(m["reconstruction_psnr_db"], 2),
            "seconds": round(ba.seconds, 4),
            "grid": f"{rows}x{cols}", "frame": f"{w}x{h}",
            "seed": ISITIA_FIGURE_SEED,
        })
        panels.append((title, fitted, scrambled, ba.reconstructed, m))

    n = len(panels)
    fig, axes = plt.subplots(n, 3, figsize=(9.6, 2.35 * n))
    if n == 1:
        axes = np.array([axes])
    for i, (title, orig, scr, rec, m) in enumerate(panels):
        for j, (lab, arr) in enumerate((("оригінал", orig),
                                        ("перемішано", scr),
                                        ("зібрано за межами", rec))):
            a = axes[i, j]
            a.imshow(arr, cmap="gray", vmin=0, vmax=255)
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title(lab, fontsize=10)
        axes[i, 0].set_ylabel(title.replace(" (", "\n("), fontsize=8)
        axes[i, 2].annotate(
            f"сусідств {m['neighbour_accuracy']*100:.2f} %\n"
            f"абс. позицій {m['direct_accuracy']*100:.2f} %\n"
            f"PSNR {m['reconstruction_psnr_db']:.2f} дБ",
            (0.98, 0.04), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=7.5, color="#ffffff",
            bbox=dict(boxstyle="round,pad=0.28", fc="#37474f", ec="none",
                      alpha=0.82))

    fig.suptitle("Збирання за сумісністю меж: без ключа і без відомої пари",
                 fontsize=12.5, y=0.995)
    worst_direct = max(r["direct_accuracy_pct"] for r in rowsout)
    fig.text(0.012, 0.008,
             f"Сітка {rows}×{cols}, кадр {w}×{h}, seed {ISITIA_FIGURE_SEED}. "
             f"Частка блоків у правильних АБСОЛЮТНИХ позиціях ніде не "
             f"перевищує {worst_direct:.2f} % — відновлюється не розташування, "
             f"а структура: локальні сусідства. На синтетичному градієнті атака "
             f"найсильніша, на справжніх знімках помітно слабша. "
             f"commit {str(env.get('git_commit'))[:12]}.",
             fontsize=7.6, color="#37474f", wrap=True)
    fig.tight_layout(rect=(0, 0.035, 1, 0.985))
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    fig.savefig(OUT_PNG.replace(".png", ".svg"))
    plt.close(fig)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        wri = csv.DictWriter(fh, fieldnames=list(rowsout[0]))
        wri.writeheader()
        wri.writerows(rowsout)

    _b1_vs_b2(cfg, h, w, rows, cols, env)

    for r in rowsout:
        print(f"  {r['case']:34s} сусідств {r['neighbour_accuracy_pct']:6.2f} %  "
              f"абс. {r['direct_accuracy_pct']:5.2f} %  "
              f"PSNR {r['psnr_db']:6.2f} дБ  ({r['seconds']:.3f} с)")
    print(f"\n{OUT_PNG}\n{OUT_CSV}")
    print("  копії стенду для docs/img:")
    _copy_lab()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
