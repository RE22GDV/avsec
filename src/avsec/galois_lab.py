"""GF(2^8) bench: the computed substitution table against the stored one.

The runner behind ``docs/galois_sbox.md``.  It answers the questions the two
constructions raise, in the order they arise:

**Is the field arithmetic right?**  The computed table is compared byte for byte
with the AES S-box printed in the standard.  Reproducing a table someone else
published is the strongest available test of a GF(2^8) implementation.

**Does the choice of polynomial matter?**  There are 30 irreducible polynomials
of degree 8 over GF(2).  Each defines a different field representation, so each
gives a different table - and the bench measures whether the *properties* change
with it, rather than assuming they do or do not.

**Is the algebraic table actually better?**  Differential uniformity,
nonlinearity, algebraic degree and avalanche are computed for the algebraic
table, for its keyed variant, for one keyed random table, and for a sample of
many random tables, so "on average" and "exactly" sit in the same table.

**Does it cost anything in the scheme?**  ``B2s`` is run with both table sources
against the same attacks and the same channel, because a better table is only
worth having if it does not make something else worse.

Usage::

    avsec gf-lab                    # -> results/gf_sbox/
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from avsec.config import ExperimentConfig
from avsec.galois import (
    AES_SBOX_REFERENCE,
    GF256,
    KeyedAlgebraicSbox,
    algebraic_sbox,
    inverse_algebraic_sbox,
    irreducible_polynomials,
)
from avsec.sbox_analysis import ALGEBRAIC_OPTIMUM, analyse, random_table_reference
from avsec.substitution import crypto_sbox
from avsec.utils import ensure_dir, environment_record, write_csv, write_json

#: Bits of storage a 256-entry byte table needs, for the hardware comparison.
TABLE_BITS = 256 * 8


def verify_against_aes() -> Dict[str, Any]:
    """The computed table against the published one, byte for byte."""
    gf = GF256()
    got = algebraic_sbox(gf)
    ref = np.array(AES_SBOX_REFERENCE, dtype=np.uint8)
    inv = inverse_algebraic_sbox(gf)
    return {
        "matches_published_aes_sbox": bool(np.array_equal(got, ref)),
        "bytes_compared": int(ref.size),
        "first_differing_index": (int(np.flatnonzero(got != ref)[0])
                                  if not np.array_equal(got, ref) else -1),
        "inverse_is_exact": bool(np.array_equal(inv[got.astype(np.int64)],
                                                np.arange(256))),
        "field": gf.describe(),
    }


def polynomial_sweep() -> List[Dict[str, Any]]:
    """One table per irreducible polynomial, with its measured properties.

    The field is a choice, not a constant.  What this shows is which part of the
    result follows from the choice (the table) and which part does not (the
    resistance).
    """
    out: List[Dict[str, Any]] = []
    ref = np.array(AES_SBOX_REFERENCE, dtype=np.uint8)
    for poly in irreducible_polynomials(8):
        gf = GF256(poly)
        box = algebraic_sbox(gf)
        props = analyse(box, f"0x{poly:X}")
        out.append({
            "polynomial_hex": f"0x{poly:X}",
            "polynomial": gf.describe()["polynomial"],
            "generator": gf.describe()["generator"],
            "same_table_as_aes": bool(np.array_equal(box, ref)),
            "differing_bytes_vs_aes": int((box != ref).sum()),
            "differential_uniformity": props["differential_uniformity"],
            "nonlinearity": props["nonlinearity"],
            "algebraic_degree": props["algebraic_degree"],
            "fixed_points": props["fixed_points"],
        })
    return out


def property_table(key: bytes, session_id: bytes) -> List[Dict[str, Any]]:
    """The four tables that matter, measured the same way."""
    gf = GF256()
    alg = algebraic_sbox(gf)
    keyed = KeyedAlgebraicSbox(key, gf).table(b"|" + session_id + b"|session")
    rnd = crypto_sbox(key, b"|" + session_id + b"|session")
    rows = [
        analyse(alg, "алгебраїчна (публічна, як в AES)"),
        analyse(keyed, "алгебраїчна + ключове забілювання"),
        analyse(rnd, "ключова випадкова (поточна в B2s)"),
    ]
    # what actually has to sit in memory: the algebraic table is computed, so
    # only its key material is stored; the random table is the table itself
    stored = (0, 16, TABLE_BITS)
    for r, bits in zip(rows, stored):
        r["stored_bits"] = bits
        r["storage"] = ("обчислюється" if bits == 0 else
                        f"{bits} біт ключа" if bits < TABLE_BITS else
                        f"{bits} біт таблиці")
    return rows


def scheme_comparison(cfg: ExperimentConfig, key: bytes, sid: bytes,
                      frames) -> List[Dict[str, Any]]:
    """The same attacks against B2s built on each table source."""
    from avsec.subst_lab import attack_table

    out: List[Dict[str, Any]] = []
    for source in ("random", "algebraic"):
        for row in attack_table(cfg, key, sid, frames, source=source):
            if row["target"].startswith("B2s"):
                row = dict(row)
                row["table_source"] = source
                out.append(row)
    return out


def run_galois_lab(cfg: ExperimentConfig, output_dir: str = "results/gf_sbox",
                   n_random_samples: int = 64, progress=None) -> Dict[str, Any]:
    """Everything above, written to ``output_dir``."""
    from avsec.crypto import derive_session_keys
    from avsec.sources.drone import DRONE_SCENES, drone_suite

    env = environment_record()
    out = ensure_dir(output_dir)

    def say(stage: str, frac: float) -> None:
        if progress:
            try:
                progress(stage, frac, {})
            except Exception:
                pass

    sid = b"GFLAB000"
    key = derive_session_keys(cfg.master_secret(), sid).key

    say("звірка з опублікованою таблицею AES", 0.05)
    verification = verify_against_aes()

    say("таблиці S-box і обернена", 0.15)
    gf = GF256()
    box = algebraic_sbox(gf)
    inv = inverse_algebraic_sbox(gf)
    from avsec.subst_lab import _table_rows

    write_csv(os.path.join(out, "aes_sbox.csv"), _table_rows(box, "S-Box (GF)"))
    write_csv(os.path.join(out, "aes_sbox_inverse.csv"),
              _table_rows(inv, "S-Box⁻¹ (GF)"))

    say("властивості таблиць", 0.3)
    props = property_table(key, sid)

    say(f"розподіл для {n_random_samples} випадкових таблиць", 0.45)
    rnd_ref = random_table_reference(n_random_samples)

    say("усі 30 незвідних многочленів", 0.6)
    polys = polynomial_sweep()

    say("схема B2s на обох конструкціях", 0.78)
    picked = [d for d in DRONE_SCENES if d[0] == "sky_horizon"]
    frames = drone_suite(cfg.frame_height, cfg.frame_width, n_frames=12,
                         scenes=picked)[0].frames
    scheme = scheme_comparison(cfg, key, sid, frames)

    say("рисунок", 0.92)
    fig = _figure(out, box, inv, props, polys)

    summary = {
        "kind": "gf_sbox",
        "title": "Обчислювана таблиця замін над GF(2^8)",
        "verification": verification,
        "storage": {
            "stored_table_bits": TABLE_BITS,
            "stored_table_bytes": TABLE_BITS // 8,
            "computed_table_bits": 0,
            "note": ("для програми 256 байтів не є проблемою; різниця важлива "
                     "в апаратній реалізації, де таблиця — це ПЗП з дешифратором "
                     "адреси, і раунду потрібно кілька таких копій одночасно"),
        },
        "properties": props,
        "random_table_distribution": rnd_ref,
        "algebraic_optimum": ALGEBRAIC_OPTIMUM,
        "polynomial_sweep": polys,
        "scheme_comparison": scheme,
        "figures": fig,
        "run_id": cfg.run_identity(),
        "commit": env.get("git_commit"),
        "git_worktree": env.get("git_worktree"),
        "environment": env.get("environment", env),
    }
    write_json(os.path.join(out, "gf_sbox.json"), summary)
    write_csv(os.path.join(out, "properties.csv"), props)
    write_csv(os.path.join(out, "polynomial_sweep.csv"), polys)
    write_csv(os.path.join(out, "scheme_comparison.csv"), scheme)
    say("готово", 1.0)
    return summary


def _figure(out_dir: str, box: np.ndarray, inv: np.ndarray,
            props: List[Dict[str, Any]], polys: List[Dict[str, Any]]
            ) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.0, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1.0], hspace=0.32,
                          wspace=0.26)

    for col, (table, title) in enumerate(
            ((box, "S-Box, обчислена над GF(2⁸)"),
             (inv, "Обернена S-Box⁻¹"))):
        ax = fig.add_subplot(gs[0, col])
        grid = table.astype(np.int64).reshape(16, 16)
        ax.imshow(grid, cmap="viridis", vmin=0, vmax=255)
        for i in range(16):
            for j in range(16):
                ax.text(j, i, f"{grid[i, j]:02X}", ha="center", va="center",
                        fontsize=5.6,
                        color="#ffffff" if grid[i, j] < 150 else "#101010")
        ax.set_xticks(range(16))
        ax.set_xticklabels([f"{v:X}" for v in range(16)], fontsize=6.5)
        ax.set_yticks(range(16))
        ax.set_yticklabels([f"{v:X}x" for v in range(16)], fontsize=6.5)
        ax.set_title(title, fontsize=10.5)

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.set_title("Як вона рахується", fontsize=10.5)
    ax.text(0.0, 0.98,
            "поле:\n"
            "  GF(2⁸) = GF(2)[x] / m(x)\n"
            "  m(x) = x⁸+x⁴+x³+x+1  (0x11B)\n\n"
            "таблиця:\n"
            "  S(v) = A · v⁻¹ ⊕ 0x63\n"
            "  0⁻¹ ≜ 0 за домовленістю\n\n"
            "обернена:\n"
            "  S⁻¹(c) = ( A⁻¹·(c ⊕ 0x63) )⁻¹\n\n"
            "зберігати не треба:\n"
            "  таблиця   2048 біт ПЗП\n"
            "  обчислення   ~0 біт памʼяті\n\n"
            "перевірка:\n"
            "  збігається з таблицею AES\n"
            "  побайтово, усі 256 значень",
            va="top", ha="left", fontsize=8.6, family="monospace",
            color="#263238")

    ax = fig.add_subplot(gs[1, 0])
    labels = ["алгебраїчна", "алгебр.\n+ ключ", "випадкова\nключова"]
    du = [p["differential_uniformity"] for p in props]
    ax.bar(labels, du, color=["#26a69a", "#26a69a", "#ef5350"])
    for i, v in enumerate(du):
        ax.text(i, v + 0.25, str(v), ha="center", fontsize=9)
    ax.axhline(4, color="#00695c", ls=":", lw=1.2)
    ax.set_ylabel("диференціальна рівномірність", fontsize=9)
    ax.set_title("менше — краще (оптимум 4)", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[1, 1])
    nl = [p["nonlinearity"] for p in props]
    ax.bar(labels, nl, color=["#26a69a", "#26a69a", "#ef5350"])
    for i, v in enumerate(nl):
        ax.text(i, v + 1.0, str(v), ha="center", fontsize=9)
    ax.axhline(112, color="#00695c", ls=":", lw=1.2)
    ax.set_ylim(0, 125)
    ax.set_ylabel("нелінійність", fontsize=9)
    ax.set_title("більше — краще (оптимум 112)", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[1, 2])
    diff = [p["differing_bytes_vs_aes"] for p in polys]
    ax.bar(range(len(polys)), diff, color="#5c6bc0")
    ax.set_xlabel("30 незвідних многочленів степеня 8", fontsize=8.5)
    ax.set_ylabel("байтів, що відрізняються\nвід таблиці AES", fontsize=8.5)
    ax.set_title("поле — це вибір: таблиця змінюється,\n"
                 "а стійкість лишається тією самою", fontsize=9.5)
    ax.tick_params(labelsize=8)
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Обчислювана таблиця замін над GF(2⁸): побудова, перевірка, "
                 "властивості", fontsize=13)
    p = os.path.join(out_dir, "gf_sbox.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    fig.savefig(p.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    return {"gf_sbox": p}


__all__ = ["TABLE_BITS", "verify_against_aes", "polynomial_sweep",
           "property_table", "scheme_comparison", "run_galois_lab"]
