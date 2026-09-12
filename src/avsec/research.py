"""The research programme E01-E10, built on the single matrix runner.

Each function here produces exactly one raw table, named in
:mod:`avsec.program`, from which the figure catalogue is drawn.  None of them
invents a new aggregation path: they all write per-observation rows and let
:mod:`avsec.analysis` do the statistics.

Scope is stated in each docstring, because a reduced grid is a legitimate
result only if the reduction is visible.  Where the plan's nominal grid was too
large for the available compute, the executed grid is written into the output
alongside the planned one.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from avsec.config import ExperimentConfig
from avsec.matrix import ChannelPoint, MatrixRunner, MatrixSpec
from avsec.utils import ensure_dir, write_csv, write_json

Progress = Optional[Callable[[str, float, Dict[str, Any]], None]]


#: Source material and its manifest, built once per (kind, geometry, length).
#: Regenerating them for every cell of a parameter grid costs more than the
#: cell itself, and they do not depend on the parameter under test.
_MATERIAL: Dict[Tuple, Tuple[List[Any], Any]] = {}


def _material(cfg: ExperimentConfig) -> Tuple[List[Any], Any]:
    from avsec.dataset import build_manifest
    from avsec.experiments import build_sources

    key = (json.dumps(cfg.sources, sort_keys=True), cfg.frame_width,
           cfg.frame_height, cfg.n_frames, cfg.seed)
    if key not in _MATERIAL:
        srcs = build_sources(cfg)
        _MATERIAL[key] = (srcs, build_manifest(srcs, seed=cfg.seed))
    return _MATERIAL[key]


def _run(cfg: ExperimentConfig, spec: MatrixSpec, out_dir: str, workers: int,
         progress: Progress = None) -> List[Dict[str, Any]]:
    sources, manifest = _material(cfg)
    runner = MatrixRunner(cfg, spec, out_dir, sources=sources, manifest=manifest,
                          workers=workers)
    runner.run(progress=progress)
    return runner._load_rows("frames.jsonl")


def _scene_mean(rows: Sequence[Dict[str, Any]], metric: str,
                keys: Sequence[str]) -> List[Dict[str, Any]]:
    """Frames -> scenes, keeping the requested grouping keys."""
    from collections import defaultdict

    cells: Dict[Tuple, List[float]] = defaultdict(list)
    for r in rows:
        try:
            v = float(r.get(metric, "nan"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        cells[tuple(r.get(k, "") for k in keys) + (r.get("scene", ""),)].append(v)
    per_scene: Dict[Tuple, List[float]] = defaultdict(list)
    for key, vals in cells.items():
        per_scene[key[:-1]].append(float(np.mean(vals)))
    out = []
    for key, vals in sorted(per_scene.items(), key=lambda kv: str(kv[0])):
        arr = np.asarray(vals)
        row = dict(zip(keys, key))
        row.update({metric: float(arr.mean()), f"{metric}_sd": float(arr.std(ddof=1))
                    if arr.size > 1 else 0.0, "n_scenes": int(arr.size)})
        out.append(row)
    return out


def _scene_values(rows, metric: str) -> Dict[str, float]:
    """``scene -> mean of the metric`` over that scene's frames and repetitions."""
    return {r["scene"]: float(r[metric])
            for r in _scene_mean(rows, metric, ["scene"])}


def _instant_values(rows, metric: str) -> Dict[str, float]:
    """Same, but over **every scheduled instant**, failures included (R05)."""
    from collections import defaultdict

    cells: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        try:
            v = float(r.get(metric, "nan"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            cells[str(r.get("scene", ""))].append(v)
    return {k: float(np.mean(v)) for k, v in cells.items() if v}


def _paired_ci(a: Dict[str, float], b: Dict[str, float], n_boot: int = 4000,
               seed: int = 20240909) -> Dict[str, Any]:
    """Paired difference ``a - b`` over the scenes both cells produced (R09).

    A step of an ablation chain without an interval is a number nobody can
    argue with, which is not the same as a number anybody should believe.
    """
    from avsec.statistics import _bootstrap_ci, difference_tests

    shared = sorted(set(a) & set(b))
    d = np.asarray([a[s] - b[s] for s in shared], dtype=float)
    if d.size == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_scenes": 0, "significant": False}
    lo, hi = _bootstrap_ci(d, n_boot, seed)
    out = {"delta": float(d.mean()), "lo": lo, "hi": hi,
           "n_scenes": int(d.size),
           "significant": bool(d.size >= 2 and np.isfinite(lo)
                               and (lo > 0 or hi < 0))}
    out.update({k: v for k, v in difference_tests(d, n_boot, seed).items()
                if k in ("p_raw", "p_wilcoxon")})
    return out


# ----------------------------------------------------------------------- E01
#: Nominal grid from the plan; the executed grid is recorded next to it.
E01_PLANNED = {
    "codec": ["raw", "dct", "jpeg"],
    "n_descriptions": [1, 2, 4],
    "stripe_height": [8, 16, 24, 32, 48],
    "quality": [8, 12, 18, 25, 35, 50, 70],
}


def run_e01(cfg: ExperimentConfig, out_dir: str, workers: int = 8,
            progress: Progress = None) -> Dict[str, Any]:
    """E01: what source coding costs before the channel is involved.

    Codecs are compared at their **actual** bitrate, never at an equal nominal
    quality number, because ``quality=12`` means something different to the DCT
    and the JPEG paths.  ``raw`` is attempted and reported as inadmissible
    wherever it does not fit the shared budget - that is a result, not an error.
    """
    executed = {
        "codec": ["raw", "dct", "jpeg"],
        "n_descriptions": [1, 2, 4],
        "stripe_height": [16, 24, 48],
        "quality": [8, 12, 18, 25, 35, 50, 70],
    }
    ensure_dir(out_dir)
    rows: List[Dict[str, Any]] = []
    clean = [ChannelPoint.named("clean")]
    total = (len(executed["codec"]) * len(executed["n_descriptions"])
             * len(executed["stripe_height"]) * len(executed["quality"]))
    done = 0
    for codec, nd, sh, q in itertools.product(
            executed["codec"], executed["n_descriptions"],
            executed["stripe_height"], executed["quality"]):
        done += 1
        if codec == "raw" and q != executed["quality"][0]:
            continue                      # raw ignores quality; run it once
        if progress:
            progress(f"E01 {codec}/d{nd}/h{sh}/q{q}", done / total, {})
        prof = dataclasses.replace(cfg.profile("B4"), codec=codec, quality=q,
                                   stripe_height=sh, n_descriptions=nd)
        sub = dataclasses.replace(cfg, methods=("B4",),
                                  profiles={**cfg.profiles, "B4": prof},
                                  channel_preset="clean", channel_overrides={})
        spec = MatrixSpec(methods=("B4",), channels=clean, repetitions=1,
                          splits=("calibration",), max_frames=4,
                          name=f"E01-{codec}")
        cell_dir = os.path.join(out_dir, "_jobs", f"{codec}_d{nd}_h{sh}_q{q}")
        try:
            frames = _run(sub, spec, cell_dir, workers)
        except Exception as exc:                       # pragma: no cover
            rows.append({"codec": codec, "n_descriptions": nd, "stripe_height": sh,
                         "quality": q, "admissible": False,
                         "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if not frames:
            rows.append({"codec": codec, "n_descriptions": nd, "stripe_height": sh,
                         "quality": q, "admissible": False,
                         "reason": "не вміщується у спільний бюджет каналу"})
            continue
        bits = [float(f.get("payload_bytes", 0)) * 8 for f in frames]
        psnr = [float(f.get("psnr_full", "nan")) for f in frames]
        ssim = [float(f.get("ssim_full", "nan")) for f in frames]
        rows.append({
            "codec": codec, "n_descriptions": nd, "stripe_height": sh, "quality": q,
            "admissible": True, "reason": "",
            "bits_per_frame": float(np.mean(bits)),
            "psnr_full": float(np.nanmean(psnr)),
            "ssim_full": float(np.nanmean(ssim)),
            "units_per_frame": float(np.mean([f.get("units_sent", 0) for f in frames])),
            "wire_bytes_per_frame": float(np.mean([f.get("wire_bytes", 0) for f in frames])),
            "n_frames": len(frames),
        })
    write_csv(os.path.join(out_dir, "e01_rate_quality.csv"), rows)
    out = {"kind": "E01", "planned_grid": E01_PLANNED, "executed_grid": executed,
           "n_cells": len(rows),
           "n_inadmissible": sum(1 for r in rows if not r["admissible"]),
           "note": ("кодеки порівнюються за фактичним бітрейтом, не за однаковим "
                    "номером quality; недопустимі клітинки збережені з причиною")}
    write_json(os.path.join(out_dir, "e01.json"), out)
    return out


# ----------------------------------------------------------------------- E03
E03_GRIDS: Dict[str, Sequence[float]] = {
    "noise_sigma": (0, 2, 4, 6, 8, 12, 16, 24),
    "burst_len_lines": (1, 2, 4, 8, 16, 24, 32, 48, 64),
    "burst_rate_per_frame": (0, 1, 2, 4, 6, 8),
    "line_jitter_sigma": (0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    "lowpass_sigma": (0.5, 1.0, 1.5, 2.0, 2.5),
    "gain": (0.6, 0.75, 0.9, 1.0, 1.1),
    "offset": (-20, -10, 0, 10, 20, 40),
    "frame_drop_prob": (0.0, 0.01, 0.05, 0.1, 0.2),
}

#: Dense sweeps use a small method set, as the plan requires: the strong analog
#: control, segmented AEAD, the proposal, and the proposal's placement ablation.
E03_METHODS = ("B0a-R", "B4", "P")

#: Co-factors that must be non-zero for an axis to have any effect at all.
#:
#: The first version of this sweep varied ``burst_len_lines`` on the ``mild``
#: base, where ``burst_rate_per_frame`` is zero - so there were no bursts to
#: lengthen and every point of that axis returned an identical number.  A flat
#: curve caused by a mis-specified sweep is not a finding, so each axis now
#: declares what has to be switched on for it to bite, and the fixed value is
#: printed on the figure.
E03_COFACTORS: Dict[str, Dict[str, float]] = {
    "burst_len_lines": {"burst_rate_per_frame": 4.0},
    "burst_rate_per_frame": {"burst_len_lines": 16.0},
}


def run_e03(cfg: ExperimentConfig, out_dir: str, workers: int = 8,
            axes: Optional[Sequence[str]] = None, n_scenes: int = 6,
            repetitions: int = 3, max_frames: int = 6,
            progress: Progress = None) -> Dict[str, Any]:
    """E03: one impairment parameter at a time, everything else fixed.

    The base is ``mild`` for every axis, so a point on the noise axis differs
    from a point on the burst axis in exactly one field.  A dense sweep is run
    on a fixed subset of scenes chosen before the run, not on all 20.
    """
    ensure_dir(out_dir)
    axes = list(axes or E03_GRIDS)
    all_rows: List[Dict[str, Any]] = []
    for i, axis in enumerate(axes):
        fixed = tuple(sorted(E03_COFACTORS.get(axis, {}).items()))
        points = [ChannelPoint(base="mild",
                               overrides=tuple(sorted(((axis, float(v)),) + fixed)),
                               axis=axis, value=float(v))
                  for v in E03_GRIDS[axis]]
        spec = MatrixSpec(methods=E03_METHODS, channels=points,
                          repetitions=repetitions, splits=("test",),
                          max_frames=max_frames, max_clips=n_scenes,
                          name=f"E03-{axis}")
        sub = cfg
        if progress:
            progress(f"E03 {axis}", i / max(len(axes), 1), {})
        frames = _run(sub, spec, os.path.join(out_dir, "_jobs", axis), workers)
        for r in frames:
            r["axis"] = axis
            r["cofactors"] = json.dumps(dict(fixed), sort_keys=True)
        all_rows.extend(frames)
    agg = _scene_mean(all_rows, "psnr_full", ["axis", "channel_value", "method"])
    for metric in ("coverage", "ssim_full", "symbol_errors_pre_fec"):
        extra = {(_k(r)): r[metric] for r in _scene_mean(
            all_rows, metric, ["axis", "channel_value", "method"])}
        for row in agg:
            row[metric] = extra.get(_k(row), float("nan"))
    write_csv(os.path.join(out_dir, "sweeps.csv"), agg)
    write_csv(os.path.join(out_dir, "sweeps_frames.csv"), all_rows)
    out = {"kind": "E03", "axes": axes, "grids": {k: list(v) for k, v in E03_GRIDS.items()},
           "methods": list(E03_METHODS), "n_rows": len(agg),
           "n_scenes_used": n_scenes,
           "cofactors": {k: dict(v) for k, v in E03_COFACTORS.items()},
           "base_preset": "mild",
           "note": ("усі осі числові; назви профілів не використовуються як вісь. "
                    "Для осей, що потребують увімкненого супутнього фактора, він "
                    "зафіксований і вказаний у cofactors - інакше крива була б "
                    "плоскою через хибно заданий sweep, а не через результат")}
    write_json(os.path.join(out_dir, "e03.json"), out)
    return out


def _k(row: Dict[str, Any]) -> Tuple:
    return (row.get("axis"), row.get("channel_value"), row.get("method"))


# The dense-sweep scene subset is the first ``n_scenes`` clips of the test
# split in manifest order - fixed before the run and identical every time.


# ----------------------------------------------------------------------- E04
def run_e04(cfg: ExperimentConfig, out_dir: str, workers: int = 8,
            repetitions: int = 3, max_frames: int = 4, n_clips: int = 6,
            progress: Progress = None) -> Dict[str, Any]:
    """E04: two interaction matrices and the operating boundary.

    Every method sees the same random events in every cell, and the metrics of
    *all* compared methods are kept, not just the winner's.
    """
    ensure_dir(out_dir)
    rows: List[Dict[str, Any]] = []

    grids = [
        ("noise_x_burst", "noise_sigma", (0, 4, 8, 12, 16),
         "burst_len_lines", (1, 4, 8, 16, 32, 48)),
        ("gain_x_offset", "gain", (0.6, 0.75, 0.9, 1.0, 1.1),
         "offset", (-20, -10, 0, 10, 20, 40)),
    ]
    for gi, (name, ax, avals, bx, bvals) in enumerate(grids):
        points = [ChannelPoint(base="mild",
                               overrides=((ax, float(a)), (bx, float(b))),
                               axis=f"{ax}x{bx}", value=float(i))
                  for i, (a, b) in enumerate(itertools.product(avals, bvals))]
        meta = {p.label: (a, b) for p, (a, b) in
                zip(points, itertools.product(avals, bvals))}
        spec = MatrixSpec(methods=("B0a-R", "B4", "P"), channels=points,
                          repetitions=repetitions, splits=("test",),
                          max_frames=max_frames, max_clips=n_clips,
                          name=f"E04-{name}")
        if progress:
            progress(f"E04 {name}", gi / len(grids), {})
        frames = _run(cfg, spec, os.path.join(out_dir, "_jobs", name), workers)
        agg = _scene_mean(frames, "psnr_full", ["channel", "method"])
        cov = {(r["channel"], r["method"]): r["coverage"]
               for r in _scene_mean(frames, "coverage", ["channel", "method"])}
        for r in agg:
            a, b = meta.get(r["channel"], (float("nan"), float("nan")))
            r.update({"grid": name, ax: a, bx: b,
                      "coverage": cov.get((r["channel"], r["method"]), float("nan"))})
            rows.append(r)
    write_csv(os.path.join(out_dir, "interaction.csv"), rows)
    out = {"kind": "E04", "grids": [g[0] for g in grids], "n_cells": len(rows),
           "note": ("у кожній клітинці збережено метрики всіх методів; дрібні "
                    "різниці без інтервалу не оголошуються перемогою")}
    write_json(os.path.join(out_dir, "e04.json"), out)
    return out


# ----------------------------------------------------------------------- E05
E05_SCHEMES = (
    ("sequential", dict(scheme="sequential", depth=1)),
    ("block-16", dict(scheme="block", depth=16)),
    ("block-64", dict(scheme="block", depth=64)),
    ("block-278", dict(scheme="block", depth=278)),
    ("diagonal-64", dict(scheme="diagonal", depth=64)),
    ("random-64", dict(scheme="random", depth=64)),
    ("bawp-8", dict(scheme="bawp", depth=1, burst_rows=8)),
    ("bawp-16", dict(scheme="bawp", depth=1, burst_rows=16)),
    ("bawp-32", dict(scheme="bawp", depth=1, burst_rows=32)),
)
E05_BURST_LINES = (1, 2, 4, 8, 16, 24, 32, 48, 64)


def run_e05(cfg: ExperimentConfig, out_dir: str) -> Dict[str, Any]:
    """E05: BAWP studied on the placement itself, with no video involved.

    The plan's first level: identical coded units through different placements,
    with lengths, FEC, modem and description count held fixed.  A single burst
    is swept over **every** starting row and every length in the grid.

    Damage is counted in the units the code actually works in (defect R03).
    A burst hits modem *cells*; ``bits_per_symbol`` cells make one byte; the
    bytes fall into the header's RS blocks and the payload's RS blocks, each of
    which corrects ``nsym/2`` unknown errors or ``nsym`` erasures **on its own**.
    :mod:`avsec.damage` does that conversion, and
    :func:`avsec.damage.decode_check` compares its verdict against a real
    Reed-Solomon decode, so the model is checked rather than asserted.

    "Touched" and "lost" are also kept apart (R04): a description a burst
    reached and the FEC repaired was not lost, and counting it as lost is what
    made the old joint-loss figure unreadable.
    """
    from avsec.budget import compute_budget
    from avsec.damage import layout_for, sweep_burst
    from avsec.interleaving import (Interleaver, InterleaverConfig, PlacementError,
                                    burst_rows_from_lines)

    ensure_dir(out_dir)
    prof = cfg.profile("P")
    tr = prof.transport_config(cfg.budget)
    modem = tr.modem
    n_rows, n_cols = modem.n_data_rows, modem.n_data_cols
    b = compute_budget(modem, tr.fec_payload, tr.fec_header, prof.max_unit_payload,
                       cfg.budget.raster_rate_hz)
    unit_symbols = b.unit_symbols
    nsym = tr.fec_payload.nsym
    layout = layout_for(tr, prof.max_unit_payload)
    n_blocks = len(layout.blocks())

    rows: List[Dict[str, Any]] = []
    joint_rows: List[Dict[str, Any]] = []
    geometry: List[Dict[str, Any]] = []

    for scheme_name, kw in E05_SCHEMES:
        for n_desc in (1, 2, 4):
            twists = (0, 1, 3) if kw["scheme"] == "bawp" else (kw.get("column_twist", 1),)
            for twist in twists:
                il_cfg = InterleaverConfig(column_twist=twist, **kw)
                try:
                    il = Interleaver(il_cfg, n_rows, n_cols)
                    n_units = il.max_units(unit_symbols, n_desc, b.units_per_raster)
                    if n_units <= 0:
                        raise PlacementError("no unit fits this placement")
                    descs = [i % n_desc for i in range(n_units)]
                    cells = il.place([unit_symbols] * n_units, descs, n_desc)
                except Exception as exc:
                    rows.append({"scheme": scheme_name, "n_descriptions": n_desc,
                                 "column_twist": twist, "admissible": False,
                                 "reason": f"{type(exc).__name__}: {exc}",
                                 "burst_lines": -1})
                    continue
                # hash the whole placement table: two configurations are
                # aliases only if every cell of every unit coincides
                import hashlib

                h = hashlib.sha256()
                for c in cells:
                    h.update(np.ascontiguousarray(c, dtype=np.int64).tobytes())
                sig = h.hexdigest()[:16]
                # rows alone decide burst damage, so record that separately:
                # a parameter that only permutes columns cannot change it
                hr = hashlib.sha256()
                for c in cells:
                    hr.update(np.ascontiguousarray(
                        np.asarray(c, dtype=np.int64) // n_cols).tobytes())
                sig_rows = hr.hexdigest()[:16]
                geometry.append({"scheme": scheme_name, "n_descriptions": n_desc,
                                 "column_twist": twist, "n_units": n_units,
                                 "accumulation_rows": il.accumulation_rows,
                                 "placement_signature": sig,
                                 "row_signature": sig_rows})
                for lines in E05_BURST_LINES:
                    br = burst_rows_from_lines(lines, modem.symbol_height)
                    dmg = sweep_burst(cells, n_cols, n_rows, br, layout,
                                      descs, n_desc)
                    rows.append({
                        "scheme": scheme_name, "n_descriptions": n_desc,
                        "column_twist": twist, "admissible": True, "reason": "",
                        "burst_lines": lines, "burst_symbol_rows": br,
                        "worst_damaged_bytes": dmg["worst_damaged_bytes"],
                        "worst_damaged_symbols": dmg["worst_damaged_symbols"],
                        "mean_damaged_bytes": round(dmg["mean_damaged_bytes"], 2),
                        "worst_unit": dmg["worst_unit"],
                        "rs_nsym": nsym, "rs_k": tr.fec_payload.k,
                        "bits_per_symbol": modem.bits_per_symbol,
                        "symbols_per_byte": layout.symbols_per_byte,
                        "n_rs_blocks_per_unit": n_blocks,
                        "correctable_errors": nsym // 2,
                        "correctable_erasures": nsym,
                        "frac_positions_uncorrectable":
                            round(dmg["frac_positions_uncorrectable"], 4),
                        "survives_as_errors":
                            bool(dmg["n_positions_uncorrectable"] == 0),
                        "frac_positions_stripe_lost":
                            round(dmg["frac_positions_stripe_lost"], 4),
                        "claimed_bound_bytes": _claimed_bound(
                            il_cfg, n_desc, br, unit_symbols, n_rows,
                            layout.symbols_per_byte),
                        "n_units_placed": n_units,
                        "accumulation_rows": il.accumulation_rows,
                        "n_burst_positions": dmg["n_burst_positions"],
                    })
                    # Two histograms, not one: how many descriptions of a
                    # stripe a burst *reached*, and how many it actually cost.
                    for kind, hist in (("touched", dmg["touched_histogram"]),
                                       ("lost", dmg["lost_histogram"])):
                        for k, count in sorted(hist.items()):
                            joint_rows.append({
                                "scheme": scheme_name, "n_descriptions": n_desc,
                                "column_twist": twist, "burst_lines": lines,
                                "outcome": kind,
                                "descriptions_in_one_stripe": k,
                                "descriptions_lost_in_one_stripe":
                                    k if kind == "lost" else -1,
                                "n_burst_positions": count,
                                "fraction": round(
                                    count / max(dmg["n_burst_positions"], 1), 4),
                                "stripe_lost": bool(kind == "lost" and k >= n_desc),
                            })
    write_csv(os.path.join(out_dir, "codewords.csv"), rows)
    write_csv(os.path.join(out_dir, "joint_loss.csv"), joint_rows)
    write_csv(os.path.join(out_dir, "placement_geometry.csv"), geometry)

    # The analytic count is only worth publishing if the decoder agrees with it.
    checks = _rs_cross_check(layout)
    write_csv(os.path.join(out_dir, "rs_cross_check.csv"), checks)

    aliases = _placement_aliases(geometry)
    out = {"kind": "E05", "schemes": [s[0] for s in E05_SCHEMES],
           "burst_lines": list(E05_BURST_LINES), "n_rows": len(rows),
           "n_inadmissible": sum(1 for r in rows if not r["admissible"]),
           "placement_aliases": aliases,
           "unit_layout": layout.describe(),
           "rs_cross_check": {
               "n_cases": len(checks),
               "n_disagreements": sum(1 for c in checks if not c["agree"]),
           },
           "note": ("пошкодження рахуються в байтах RS-слова: клітинка модема "
                    f"несе {modem.bits_per_symbol} біт, тому "
                    f"{layout.symbols_per_byte} пошкоджені клітинки - це один "
                    "пошкоджений байт; байти розкладаються по фактичних RS-словах "
                    f"({n_blocks} на одиницю: заголовок + payload), і кожне "
                    "слово має власний бюджет виправлення. Окремо рахується, "
                    "скільки описів смуги буря ЗАЧЕПИЛА і скільки справді "
                    "ВТРАТИЛА - це різні події")}
    write_json(os.path.join(out_dir, "e05.json"), out)
    return out


def _rs_cross_check(layout: Any, n_cases: int = 24) -> List[Dict[str, Any]]:
    """Analytic verdict versus a real RS decode, on a grid of control cases."""
    from avsec.damage import decode_check

    out: List[Dict[str, Any]] = []
    total = layout.wire_symbols
    for i in range(n_cases):
        n = int(round(total * (i + 1) / (n_cases + 1)))
        start = (i * 137) % max(1, total - n)
        ords = list(range(start, start + n))
        for as_er in (False, True):
            r = decode_check(ords, layout, seed=1000 + i, as_erasures=as_er)
            r.pop("per_block_damaged_bytes", None)
            r.update({"case": i, "first_symbol": start, "n_symbols": n})
            out.append(r)
    return out


def _claimed_bound(il_cfg: Any, n_desc: int, burst_rows: int, unit_symbols: int,
                   n_rows: int, symbols_per_byte: int = 1) -> int:
    """What the analysis *claims* a burst can destroy, for the G29 check.

    Returned in **bytes**, so it is comparable with the measured damage;
    the naive derivation counts modem cells, and those are not bytes (R03).
    """
    if il_cfg.scheme != "bawp":
        return -1
    from avsec.interleaving import bawp_bands

    window = il_cfg.window_rows or n_rows
    try:
        bands = bawp_bands(window, n_desc, il_cfg.burst_rows)
    except Exception:
        return -1
    h = min(h for _, h in bands)
    per_row = int(np.ceil(unit_symbols / max(h, 1)))
    return int(np.ceil(burst_rows * per_row / max(1, symbols_per_byte)))


def _placement_aliases(geometry: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parameters that produce the *same* placement are reported as such."""
    from collections import defaultdict

    exact: Dict[Tuple, List[str]] = defaultdict(list)
    rowwise: Dict[Tuple, List[str]] = defaultdict(list)
    for g in geometry:
        name = f"{g['scheme']}/twist={g['column_twist']}"
        exact[(g["n_descriptions"], g["placement_signature"])].append(name)
        rowwise[(g["n_descriptions"], g["row_signature"])].append(name)
    out = [{"kind": "identical placement", "n_descriptions": k[0], "variants": v}
           for k, v in sorted(exact.items(), key=lambda kv: str(kv[0])) if len(v) > 1]
    for k, v in sorted(rowwise.items(), key=lambda kv: str(kv[0])):
        if len(v) > 1 and not any(set(v) <= set(e["variants"]) for e in out):
            out.append({"kind": "same rows, different columns - identical damage "
                                "under a row burst", "n_descriptions": k[0],
                        "variants": v})
    return out



# ----------------------------------------------------------------------- E06
def run_e06(cfg: ExperimentConfig, out_dir: str, workers: int = 8,
            repetitions: int = 3, max_frames: int = 6, channel: str = "bursty",
            n_clips: int = 6, progress: Progress = None) -> Dict[str, Any]:
    """E06: ablations, with the mechanism study and the retuned study separated.

    The 2x2 factorial (one/two descriptions x deep block/BAWP) is a *mechanism*
    study: everything else in the transport is held fixed, so the cells differ
    only in the factor under test.  The parameter grids that follow are also
    mechanism studies for the same reason; a "best system per budget" comparison
    would have to retune every cell and is reported separately by ``avsec tune``.
    """
    ensure_dir(out_dir)
    base = cfg.profile("P")
    rows: List[Dict[str, Any]] = []
    scene_rows: List[Dict[str, Any]] = []
    cells: Dict[str, Dict[str, Dict[str, float]]] = {}
    point = [ChannelPoint.named(channel)]

    def _cell(name: str, prof, extra: Dict[str, Any], sub_dir: str) -> None:
        sub = dataclasses.replace(cfg, methods=("P",),
                                  profiles={**cfg.profiles, "P": prof},
                                  channel_preset=channel, channel_overrides={},
                                  fill=extra.get("fill", cfg.fill))
        # Record what the placement actually costs, so the figure reads a
        # measured buffer depth instead of recomputing it from the label.
        try:
            from avsec.interleaving import Interleaver

            tr = prof.transport_config(cfg.budget)
            il = Interleaver(tr.interleaver, tr.modem.n_data_rows,
                             tr.modem.n_data_cols)
            extra = dict(extra, accumulation_rows=il.accumulation_rows,
                         window_accumulation_ms=round(
                             il.accumulation_rows * tr.modem.symbol_height
                             / (cfg.budget.raster_rate_hz * tr.modem.raster_height)
                             * 1e3, 3))
        except Exception:
            pass
        spec = MatrixSpec(methods=("P",), channels=point, repetitions=repetitions,
                          splits=("test",), max_frames=max_frames,
                          max_clips=n_clips, name=name)
        try:
            frames = _run(sub, spec, os.path.join(out_dir, "_jobs", sub_dir), workers)
        except Exception as exc:
            rows.append({"study": name, "admissible": False,
                         "reason": f"{type(exc).__name__}: {exc}", **extra})
            return
        if not frames:
            rows.append({"study": name, "admissible": False,
                         "reason": "не вміщується у спільний бюджет", **extra})
            return
        agg = _scene_mean(frames, "psnr_full", [])
        cov = _scene_mean(frames, "coverage", [])
        age = _scene_mean(frames, "max_age_frames", [])
        # Per-scene values are kept, not just their mean: without them no step
        # of an ablation can carry a paired interval (R09).
        q = _scene_values(frames, "psnr_full")
        c = _scene_values(frames, "coverage")
        av = _instant_values(frames, "availability")
        dis = _instant_values(frames, "psnr_displayed")
        cells[name + "|" + json.dumps(extra, sort_keys=True, default=str)] = {
            "psnr_full": q, "coverage": c, "availability": av,
            "psnr_displayed": dis}
        for scene in sorted(q):
            scene_rows.append({"study": name, **extra, "scene": scene,
                               "psnr_full": q[scene],
                               "coverage": c.get(scene, float("nan")),
                               "availability": av.get(scene, float("nan")),
                               "psnr_displayed": dis.get(scene, float("nan"))})
        rows.append({
            "study": name, "admissible": True, "reason": "", **extra,
            "psnr_full": agg[0]["psnr_full"] if agg else float("nan"),
            "psnr_sd": agg[0]["psnr_full_sd"] if agg else float("nan"),
            "coverage": cov[0]["coverage"] if cov else float("nan"),
            "availability": float(np.mean(list(av.values()))) if av else float("nan"),
            "psnr_displayed": float(np.mean(list(dis.values()))) if dis
                              else float("nan"),
            "max_age_frames": age[0]["max_age_frames"] if age else float("nan"),
            "n_scenes": agg[0]["n_scenes"] if agg else 0,
            "cell_key": name + "|" + json.dumps(extra, sort_keys=True, default=str),
        })

    # --- 2x2 factorial: descriptions x placement -------------------------
    for nd in (1, 2, 4):
        for scheme, depth, burst in (("block", 278, 0), ("bawp", 0, 8)):
            prof = dataclasses.replace(base, n_descriptions=nd,
                                       interleaver=(scheme, depth, burst))
            _cell("factorial", prof,
                  {"n_descriptions": nd, "placement": scheme},
                  f"fact_d{nd}_{scheme}")

    # --- unit size x FEC --------------------------------------------------
    for payload in (160, 240, 320, 480, 640, 960):
        for nsym in (32, 64, 96, 128):
            prof = dataclasses.replace(base, max_unit_payload=payload, fec_nsym=nsym)
            _cell("payload_x_fec", prof,
                  {"max_unit_payload": payload, "fec_nsym": nsym,
                   "rs_k": max(1, 255 - nsym)},
                  f"pl{payload}_ns{nsym}")

    # --- interleaver window ----------------------------------------------
    # The window is a field of its own: it is not the block interleaver's depth,
    # and setting only the scheme (as an earlier version did) would have made
    # every cell of this study identical.
    for window in (16, 32, 64, 128, 0):
        prof = dataclasses.replace(base, interleaver=("bawp", 0, 8),
                                   interleaver_window=window)
        _cell("window", prof, {"window_rows": window, "placement": "bawp"},
              f"win{window}")

    # --- gap-fill policy --------------------------------------------------
    for fill in ("neutral", "interpolate", "previous"):
        _cell("fill_policy", base, {"fill": fill}, f"fill_{fill}")

    write_csv(os.path.join(out_dir, "ablations.csv"), rows)
    write_csv(os.path.join(out_dir, "ablation_scenes.csv"), scene_rows)

    # --- paired contrasts inside the factorial (R09) ----------------------
    # The reference cell is declared, not chosen after looking: it is the
    # configuration of P itself (two descriptions, BAWP).
    contrasts: List[Dict[str, Any]] = []
    ref_key = next((r["cell_key"] for r in rows
                    if r.get("study") == "factorial"
                    and r.get("n_descriptions") == 2
                    and r.get("placement") == "bawp" and r.get("admissible")), "")
    for r in rows:
        if r.get("study") != "factorial" or not r.get("admissible"):
            continue
        if not ref_key or r["cell_key"] == ref_key:
            continue
        row = {"study": "factorial", "against": "P (2 descriptions, BAWP)",
               "n_descriptions": r.get("n_descriptions"),
               "placement": r.get("placement")}
        for metric in ("psnr_full", "coverage", "availability"):
            ci = _paired_ci(cells[r["cell_key"]][metric], cells[ref_key][metric])
            row.update({f"{metric}_delta": round(ci["delta"], 4),
                        f"{metric}_lo": round(ci["lo"], 4),
                        f"{metric}_hi": round(ci["hi"], 4),
                        f"{metric}_significant": ci["significant"]})
        row["n_scenes"] = ci["n_scenes"]
        contrasts.append(row)
    write_csv(os.path.join(out_dir, "ablation_contrasts.csv"), contrasts)

    interaction = _payload_fec_interaction(rows)
    out = {"kind": "E06", "channel": channel, "n_cells": len(rows),
           "n_inadmissible": sum(1 for r in rows if not r["admissible"]),
           "studies": sorted({r["study"] for r in rows}),
           "factorial_contrasts": contrasts,
           "payload_fec_interaction": interaction,
           "note": ("це дослідження МЕХАНІЗМУ: решта транспорту зафіксована. "
                    "Порівняння найкращих систем за однаковим бюджетом вимагає "
                    "переналаштування кожної клітинки і звітується окремо. "
                    "Кожна клітинка збережена посценно (ablation_scenes.csv), "
                    "а різниці до P - з парними інтервалами "
                    "(ablation_contrasts.csv)")}
    write_json(os.path.join(out_dir, "e06.json"), out)
    return out


def _payload_fec_interaction(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Does the benefit of stronger FEC depend on the unit size? (R09)

    Reported as the spread of the FEC effect across unit sizes: if the gain
    from the weakest to the strongest code is the same at every payload, the
    two factors are additive on this grid and the claim says so.
    """
    grid: Dict[Tuple[int, int], float] = {}
    for r in rows:
        if r.get("study") != "payload_x_fec" or not r.get("admissible"):
            continue
        try:
            grid[(int(r["max_unit_payload"]), int(r["fec_nsym"]))] = float(r["psnr_full"])
        except (KeyError, TypeError, ValueError):
            continue
    if not grid:
        return {"available": False}
    payloads = sorted({k[0] for k in grid})
    nsyms = sorted({k[1] for k in grid})
    effects: Dict[int, float] = {}
    for pl in payloads:
        have = [n for n in nsyms if (pl, n) in grid]
        if len(have) >= 2:
            effects[pl] = grid[(pl, have[-1])] - grid[(pl, have[0])]
    if len(effects) < 2:
        return {"available": False, "reason": "замало допустимих клітинок"}
    vals = np.asarray(list(effects.values()), dtype=float)
    return {
        "available": True,
        "payloads": payloads, "fec_nsym": nsyms,
        "fec_effect_per_payload_db": {str(k): round(v, 3) for k, v in effects.items()},
        "spread_db": round(float(vals.max() - vals.min()), 3),
        "additive_on_this_grid": bool(float(vals.max() - vals.min()) < 1.0),
        "statement": (
            "ефект сильнішого FEC змінюється на "
            f"{float(vals.max() - vals.min()):.2f} дБ між розмірами одиниці; "
            "тобто фактори "
            + ("практично адитивні на цій сітці"
               if float(vals.max() - vals.min()) < 1.0
               else "взаємодіють: виграш від FEC залежить від розміру одиниці")),
        "caveat": ("клітинки, недопустимі за бюджетом, виключені; сітка "
                   "нерівномірно заповнена, і це не «повний факторний план»"),
    }


# ----------------------------------------------------------------------- E07
#: Impairment scripts, with the onset frames fixed before the run.
E07_SCENARIOS = (
    ("clean_burst_clean", "чисто -> пакет завад -> чисто"),
    ("dropouts", "випадіння растрів"),
    ("degrade_recover", "поступове погіршення і відновлення"),
)


def e07_schedule(name: str, n_frames: int
                 ) -> Tuple[Tuple[Tuple[int, ChannelPoint], ...], Tuple[int, int]]:
    """The channel script of one scenario, plus the impairment window.

    Onsets are a fixed fraction of the clip, chosen before the run, so every
    clip and every repetition is impaired at the same frames.
    """
    t0, t1 = n_frames // 4, n_frames // 2
    mild = ChannelPoint.named("mild")
    if name == "clean_burst_clean":
        script = ((0, mild), (t0, ChannelPoint.named("harsh")), (t1, mild))
    elif name == "dropouts":
        drop = ChannelPoint(base="mild", overrides=(("frame_drop_prob", 0.25),))
        script = ((0, mild), (t0, drop), (t1, mild))
    else:
        step = max(1, (t1 - t0) // 3)
        script = ((0, ChannelPoint.named("clean")), (t0, mild),
                  (t0 + step, ChannelPoint.named("moderate")),
                  (t0 + 2 * step, ChannelPoint.named("harsh")), (t1, mild))
    return script, (t0, t1)


def run_e07(cfg: ExperimentConfig, out_dir: str, workers: int = 8,
            n_frames: int = 40, repetitions: int = 3, n_clips: int = 4,
            progress: Progress = None) -> Dict[str, Any]:
    """E07: dynamics on longer clips, with scripted impairment intervals.

    The whole clip is **one job**: the receiver is never reset at a phase
    boundary, only the impairment model is swapped. That is what makes recovery
    time meaningful - an earlier version ran each phase as its own job, so every
    measurement started from a fresh session on a clean channel and every
    recovery time came out as zero.

    Recovery is defined exactly: the interval from the end of an impairment to
    the first displayed frame whose current verified coverage reaches the
    criterion. A run that never recovers before the clip ends stays in the
    distribution as a censored observation.
    """
    ensure_dir(out_dir)
    methods = ("B0a-R", "B4", "P")
    # E07 needs LONGER clips than the main matrix: recovery cannot be observed
    # if the clip ends while the impairment is still running.  The material is
    # regenerated at `n_frames`, not truncated from the 16-frame test clips.
    cfg = dataclasses.replace(cfg, n_frames=n_frames,
                              sources={"kind": "research", "n_frames": n_frames},
                              max_frames_per_source=n_frames)
    rows: List[Dict[str, Any]] = []
    windows: Dict[str, Tuple[int, int]] = {}
    for si, (name, label) in enumerate(E07_SCENARIOS):
        script, window = e07_schedule(name, n_frames)
        windows[name] = window
        spec = MatrixSpec(methods=methods, channels=[script[0][1]],
                          repetitions=repetitions, splits=("test",),
                          max_frames=n_frames, max_clips=n_clips,
                          schedule=script, name=f"E07-{name}")
        if progress:
            progress(f"E07 {name}", si / len(E07_SCENARIOS), {})
        frames = _run(cfg, spec, os.path.join(out_dir, "_jobs", name), workers)
        impaired = {f for f, p in script if p.label != script[0][1].label}
        first_bad, last_bad = window
        for r in frames:
            r = dict(r)
            r["scenario"] = name
            r["scenario_label"] = label
            r["t_frame"] = int(float(r.get("frame_id", 0) or 0))
            r["impaired"] = first_bad <= r["t_frame"] < last_bad
            rows.append(r)
    write_csv(os.path.join(out_dir, "dynamics.csv"), rows)
    recov = _recovery_times(rows, windows)
    write_csv(os.path.join(out_dir, "recovery.csv"), recov)
    out = {"kind": "E07", "scenarios": [s[0] for s in E07_SCENARIOS],
           "n_frames": n_frames, "impairment_windows":
               {k: list(v) for k, v in windows.items()},
           "schedules": {name: [[f, p.label] for f, p in e07_schedule(name, n_frames)[0]]
                         for name, _ in E07_SCENARIOS},
           "n_rows": len(rows), "n_recovery_cases": len(recov),
           "n_censored": sum(1 for r in recov if r["censored"]),
           "session_continuity": ("увесь кліп - одне завдання; приймач не "
                                  "скидається на межі фази, змінюється лише "
                                  "модель спотворень"),
           "recovery_definition": ("від кінця завади до першого показаного кадру "
                                   "з поточним покриттям >= 0.9; випадки без "
                                   "відновлення до кінця кліпу залишені в "
                                   "розподілі як цензуровані")}
    write_json(os.path.join(out_dir, "e07.json"), out)
    return out


def _recovery_times(rows: Sequence[Dict[str, Any]],
                    windows: Dict[str, Tuple[int, int]],
                    coverage_min: float = 0.9) -> List[Dict[str, Any]]:
    from collections import defaultdict

    runs: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        runs[(r["scenario"], r.get("method"), r.get("clip"),
              r.get("repetition"))].append(r)
    out: List[Dict[str, Any]] = []
    for key, items in sorted(runs.items(), key=lambda kv: str(kv[0])):
        items.sort(key=lambda r: r["t_frame"])
        t_end = windows.get(key[0], (0, 0))[1]
        during = [r for r in items if r["impaired"]]
        after = [r for r in items if r["t_frame"] >= t_end]
        # a run that never dropped below the criterion was never impaired
        # enough to recover from; it is reported separately, not as "recovered
        # in zero frames", which would flatter every method equally
        degraded = any(float(r.get("coverage", 1) or 0) < coverage_min
                       for r in during)
        rec = next((r for r in after
                    if float(r.get("coverage", 0) or 0) >= coverage_min), None)
        out.append({
            "scenario": key[0], "method": key[1], "clip": key[2],
            "repetition": key[3],
            "degraded_during_impairment": degraded,
            "min_coverage_during": min(
                (float(r.get("coverage", 1) or 0) for r in during), default=1.0),
            "recovery_frames": (rec["t_frame"] - t_end) if (rec and degraded) else -1,
            "censored": bool(degraded and rec is None),
            "n_frames_observed": len(after),
            "criterion": f"coverage >= {coverage_min}",
        })
    return out


# ----------------------------------------------------------------------- E09
E09_RESOLUTIONS = ((128, 96), (256, 192), (320, 240), (384, 288))


def run_e09(cfg: ExperimentConfig, out_dir: str, workers: int = 1,
            warmup: int = 1, repeats: int = 3,
            progress: Progress = None) -> Dict[str, Any]:
    """E09: measured wall-clock cost per stage, and scaling with resolution.

    Timing runs single-process on purpose: a per-stage median measured while
    twelve workers compete for the same cores would not be the cost of the
    stage.  Warm-up iterations are discarded and their count is reported.
    """
    from avsec.channel import ChannelTrace
    from avsec.experiments import build_methods, build_sources
    from avsec.utils import StageTimer, environment_record

    ensure_dir(out_dir)
    rows: List[Dict[str, Any]] = []
    scaling: List[Dict[str, Any]] = []
    for ri, (w, h) in enumerate(E09_RESOLUTIONS):
        sub = dataclasses.replace(cfg, frame_width=w, frame_height=h,
                                  methods=("B0a-R", "B0d", "B3", "B4", "P"),
                                  channel_preset="moderate", channel_overrides={},
                                  sources={"kind": "research", "n_frames": 4})
        if progress:
            progress(f"E09 {w}x{h}", ri / len(E09_RESOLUTIONS), {})
        srcs = build_sources(sub)
        src = srcs[0]
        timer = StageTimer()
        methods = build_methods(sub, timer)
        trace = ChannelTrace(seed=sub.channel_seed_value, scene="e09",
                             repetition=0, profile="moderate",
                             rasters_per_frame=sub.budget.rasters_per_frame)
        for name, m in methods.items():
            m.reset()
            for _ in range(warmup):
                try:
                    m.process(src.frames[0], 0, trace)
                except Exception:
                    break
            per_frame: List[float] = []
            m.reset()
            ok = True
            for rep in range(repeats):
                t0 = time.perf_counter()
                try:
                    m.process(src.frames[rep % len(src.frames)], rep, trace)
                except Exception as exc:
                    scaling.append({"resolution": f"{w}x{h}", "method": name,
                                    "admissible": False,
                                    "reason": f"{type(exc).__name__}: {exc}"})
                    ok = False
                    break
                per_frame.append(time.perf_counter() - t0)
            if ok and per_frame:
                a = np.asarray(per_frame)
                scaling.append({
                    "resolution": f"{w}x{h}", "width": w, "height": h,
                    "method": name, "admissible": True, "reason": "",
                    "median_ms": float(np.median(a) * 1e3),
                    "p95_ms": float(np.percentile(a, 95) * 1e3),
                    "mean_ms": float(a.mean() * 1e3),
                    "frames_per_s": float(1.0 / max(a.mean(), 1e-9)),
                    "pixels": w * h, "repeats": repeats, "warmup": warmup,
                })
        for stage, st in timer.summary().items():
            rows.append({"resolution": f"{w}x{h}", "stage": stage,
                         "warmup_discarded": warmup, **st})
    write_csv(os.path.join(out_dir, "timings.csv"), rows)
    write_csv(os.path.join(out_dir, "scaling.csv"), scaling)
    out = {"kind": "E09", "resolutions": [f"{w}x{h}" for w, h in E09_RESOLUTIONS],
           "warmup": warmup, "repeats": repeats, "workers": workers,
           "environment": environment_record(),
           "note": ("wall-clock симулятора на ПК; це не вимір на Raspberry Pi 5 і "
                    "не оцінка апаратної реалізації. Наявність GPU не прискорює "
                    "цей код: він виконується на CPU у NumPy та чистому Python")}
    write_json(os.path.join(out_dir, "e09.json"), out)
    return out


# ----------------------------------------------------------------------- E10
def run_e10(cfg: ExperimentConfig, out_dir: str,
            progress: Progress = None) -> Dict[str, Any]:
    """E10: the level-A configurations, unchanged, evaluated on level B.

    Nothing is retuned on level B.  The physical mapping between the two models
    is stated explicitly, and where a level-A parameter has no level-B
    counterpart the two condition sets are reported separately instead of being
    forced onto one axis.
    """
    from avsec.experiments import run_cvbs

    ensure_dir(out_dir)
    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    presets = ("clean", "mild", "moderate", "harsh")
    methods = ("B4", "P")
    total = len(presets) * len(methods)
    done = 0
    for method in methods:
        for preset in presets:
            done += 1
            if progress:
                progress(f"E10 {method}/{preset}", done / total, {})
            cell_dir = os.path.join(out_dir, "_cvbs", f"{method}_{preset}")
            try:
                res = run_cvbs(cfg, cell_dir, preset_name=preset, method=method)
            except Exception as exc:
                rows.append({"method": method, "channel": preset,
                             "admissible": False,
                             "reason": f"{type(exc).__name__}: {exc}"})
                continue
            q = res.get("quality", {})
            stats = res.get("cvbs_stats") or [{}]
            rows.append({
                "method": method, "channel": preset, "admissible": True,
                "reason": "",
                "psnr_full": q.get("psnr_full"), "ssim_full": q.get("ssim_full"),
                "coverage": q.get("coverage"),
                "mse_full": q.get("mse_full"), "bit_exact": q.get("bit_exact"),
                "line_recovery_fraction": stats[0].get("line_recovery_fraction"),
                "sync_edges_found": stats[0].get("sync_edges_found"),
                "status_counts": json.dumps(res.get("status_counts", {})),
            })
            details[f"{method}/{preset}"] = {
                k: res[k] for k in ("status_counts", "compliance_note", "timing")
                if k in res}

    out = {
        "kind": "E10", "rows": rows, "presets": list(presets),
        "methods": list(methods), "details": details,
        "n_inadmissible": sum(1 for r in rows if not r["admissible"]),
        "transfer_note": (
            "конфігурації зафіксовані на рівні A і НЕ переналаштовувались на "
            "рівні B; однакова назва профілю на двох рівнях не означає однаковий "
            "канал - зіставлення фізичних параметрів описане в "
            "docs/architecture.md. Де зіставлення немає, умови подані окремо."),
    }
    write_json(os.path.join(out_dir, "cvbs.json"), out)
    write_csv(os.path.join(out_dir, "cvbs_transfer.csv"), rows)
    return out




# ----------------------------------------------------------------------- E13
#: The path from the strong baseline to the proposal, one change per step.
#:
#: The 2x2 factorial of E06 answers "does MDC x placement help, all else equal".
#: It does not answer "where does the measured advantage of P over B4 actually
#: come from", because P differs from B4 in four things at once: unit size, FEC
#: strength, description count and placement.  This walks that difference one
#: step at a time, on the same clips and the same channel realisations, so each
#: step's contribution is a measured number rather than an attribution.
#:
#: Order matters and is declared: it goes from the transport parameters that any
#: scheme could adopt to the two mechanisms this work proposes, so the proposal
#: is charged with whatever is left after the cheap changes have been made.
E13_CHAIN = (
    ("B4 (база)", dict()),
    ("+ менша одиниця 640→320 Б", dict(max_unit_payload=320)),
    ("+ сильніший FEC nsym 96→128", dict(max_unit_payload=320, fec_nsym=128)),
    ("+ нижча якість 12→8", dict(max_unit_payload=320, fec_nsym=128, quality=8)),
    ("+ два описи (MDC)", dict(max_unit_payload=320, fec_nsym=128, quality=8,
                               n_descriptions=2)),
    ("+ BAWP = P", dict(max_unit_payload=320, fec_nsym=128, quality=8,
                        n_descriptions=2, interleaver=("bawp", 0, 8))),
)

#: The same two end points, walked the other way round: the two proposed
#: mechanisms first, the transport parameters afterwards.  A stepwise
#: attribution is a property of the path taken, so running both paths is the
#: only way to say how much of a step's credit is the step and how much is its
#: position (R09).
E13_CHAIN_REVERSE = (
    ("B4 (база)", dict()),
    ("+ два описи (MDC)", dict(n_descriptions=2)),
    ("+ BAWP", dict(n_descriptions=2, interleaver=("bawp", 0, 8))),
    ("+ менша одиниця 640→320 Б", dict(n_descriptions=2,
                                       interleaver=("bawp", 0, 8),
                                       max_unit_payload=320)),
    ("+ сильніший FEC nsym 96→128", dict(n_descriptions=2,
                                         interleaver=("bawp", 0, 8),
                                         max_unit_payload=320, fec_nsym=128)),
    ("+ нижча якість 12→8 = P", dict(n_descriptions=2,
                                     interleaver=("bawp", 0, 8),
                                     max_unit_payload=320, fec_nsym=128,
                                     quality=8)),
)


def _chain_variant(cfg: ExperimentConfig, out_dir: str, tag: str,
                   chain, workers: int, channel: str, repetitions: int,
                   max_frames: int, n_clips: int,
                   progress: Progress = None) -> Dict[str, Any]:
    """Walk one ordering of the chain and return everything it measured."""
    base = cfg.profile("B4")
    point = [ChannelPoint.named(channel)]
    rows: List[Dict[str, Any]] = []
    per_scene: List[Dict[str, Dict[str, float]]] = []

    for i, (label, changes) in enumerate(chain):
        if progress:
            progress(f"E13 {tag} {label}", i / max(1, len(chain)), {})
        prof = dataclasses.replace(base, **changes)
        sub = dataclasses.replace(cfg, methods=("B4",),
                                  profiles={**cfg.profiles, "B4": prof},
                                  channel_preset=channel, channel_overrides={})
        spec = MatrixSpec(methods=("B4",), channels=point, repetitions=repetitions,
                          splits=("test",), max_frames=max_frames,
                          max_clips=n_clips, name=f"E13-{tag}-{i}")
        row: Dict[str, Any] = {"order": tag, "step": i, "label": label,
                               **_flat(changes)}
        try:
            frames = _run(sub, spec,
                          os.path.join(out_dir, "_jobs", f"{tag}{i}"), workers)
        except Exception as exc:
            row.update({"admissible": False,
                        "reason": f"{type(exc).__name__}: {exc}"})
            rows.append(row)
            per_scene.append({})
            continue
        if not frames:
            row.update({"admissible": False,
                        "reason": "не вміщується у спільний бюджет"})
            rows.append(row)
            per_scene.append({})
            continue
        cell = {
            "psnr_full": _scene_values(frames, "psnr_full"),
            "coverage": _scene_values(frames, "coverage"),
            "availability": _instant_values(frames, "availability"),
            "psnr_displayed": _instant_values(frames, "psnr_displayed"),
        }
        per_scene.append(cell)
        arr = np.asarray(list(cell["psnr_full"].values()), dtype=float)
        row.update({"admissible": True, "reason": "",
                    "psnr_all_scenes": float(arr.mean()) if arr.size else float("nan"),
                    "n_scenes_available": int(arr.size)})
        rows.append(row)
    return {"rows": rows, "per_scene": per_scene}


def _finish_chain(rows, per_scene, metrics=("psnr_full", "coverage",
                                            "availability", "psnr_displayed")):
    """Restrict every step to the common scene set and attach paired intervals."""
    sets = [set(c.get("psnr_full", {})) for c in per_scene if c.get("psnr_full")]
    common = sorted(set.intersection(*sets)) if sets else []
    dropped = sorted(set.union(*sets) - set(common)) if sets else []

    scene_rows: List[Dict[str, Any]] = []
    prev: Optional[Dict[str, Dict[str, float]]] = None
    for row, cell in zip(rows, per_scene):
        if not row.get("admissible") or not common:
            continue
        row["n_scenes"] = len(common)
        for metric in metrics:
            vals = np.asarray([cell.get(metric, {}).get(s, np.nan)
                               for s in common], dtype=float)
            row[metric] = float(np.nanmean(vals)) if vals.size else float("nan")
            if metric == "psnr_full":
                row["psnr_sd"] = (float(np.nanstd(vals, ddof=1))
                                  if vals.size > 1 else 0.0)
            if prev is None:
                row[f"delta_{metric}"] = 0.0
                continue
            a = {s: cell.get(metric, {}).get(s) for s in common
                 if s in cell.get(metric, {})}
            b = {s: prev.get(metric, {}).get(s) for s in common
                 if s in prev.get(metric, {})}
            ci = _paired_ci(a, b)
            row[f"delta_{metric}"] = round(ci["delta"], 4)
            row[f"delta_{metric}_lo"] = round(ci["lo"], 4)
            row[f"delta_{metric}_hi"] = round(ci["hi"], 4)
            row[f"delta_{metric}_significant"] = ci["significant"]
            if metric == "psnr_full":
                row["delta_psnr"] = row["delta_psnr_full"]
                row["delta_coverage"] = row.get("delta_coverage", 0.0)
                row["p_raw"] = ci.get("p_raw")
        for scene in common:
            scene_rows.append({
                "order": row.get("order", ""), "step": row["step"],
                "label": row["label"], "scene": scene,
                **{m: cell.get(m, {}).get(scene, float("nan")) for m in metrics}})
        prev = cell
    return common, dropped, scene_rows


def run_e13(cfg: ExperimentConfig, out_dir: str, workers: int = 8,
            channel: str = "bursty", repetitions: int = 3, max_frames: int = 8,
            n_clips: int = 0, progress: Progress = None) -> Dict[str, Any]:
    """E13: decompose the B4 -> P difference, and say how much the order matters.

    Every step is averaged over the **same** scenes.  A step that changes the
    unit size or the quality can push the hardest scenes over the capacity
    limit; averaging each step over whatever it happened to fit would credit it
    with an easier subset.  The per-step contributions therefore come from the
    intersection of scenes that every admissible step produced, and the scenes
    dropped that way are counted and named.

    Three things this adds over a bare waterfall (R09):

    * every step is stored **per scene** (``chain_scenes.csv``) and every step
      difference carries a **paired interval**, so "+0.20 dB" can be read
      together with how sure that is;
    * coverage, availability and displayed quality are decomposed alongside
      PSNR, because a step that buys quality by dropping frames must not look
      like a free gain;
    * the same chain is walked in the **reverse order** as well.  A stepwise
      decomposition is a property of the path, not of the components: a factor
      applied first gets credit a factor applied last does not.  Running both
      orders turns that caveat into a measurement.
    """
    ensure_dir(out_dir)
    forward = _chain_variant(cfg, out_dir, "fwd", E13_CHAIN, workers, channel,
                             repetitions, max_frames, n_clips, progress)
    reverse = _chain_variant(cfg, out_dir, "rev", E13_CHAIN_REVERSE, workers,
                             channel, repetitions, max_frames, n_clips, progress)

    common, dropped, scene_rows = _finish_chain(forward["rows"],
                                                forward["per_scene"])
    r_common, r_dropped, r_scene_rows = _finish_chain(reverse["rows"],
                                                      reverse["per_scene"])

    rows = forward["rows"] + reverse["rows"]
    write_csv(os.path.join(out_dir, "chain.csv"), rows)
    write_csv(os.path.join(out_dir, "chain_scenes.csv"), scene_rows + r_scene_rows)

    def _split(step_rows) -> Tuple[float, float]:
        ok = [r for r in step_rows if r.get("admissible") and "delta_psnr" in r]
        mech = sum(r["delta_psnr"] for r in ok
                   if ("опис" in r["label"] or "BAWP" in r["label"]))
        trans = sum(r["delta_psnr"] for r in ok
                    if not ("опис" in r["label"] or "BAWP" in r["label"]))
        return trans, mech

    t_fwd, m_fwd = _split(forward["rows"])
    t_rev, m_rev = _split(reverse["rows"])

    # The end points are identical by construction, so the two orders differ
    # only in how the same total is attributed.
    out = {"kind": "E13", "channel": channel,
           "steps": [lbl for lbl, _ in E13_CHAIN],
           "steps_reverse": [lbl for lbl, _ in E13_CHAIN_REVERSE],
           "n_rows": len(rows),
           "n_scenes_common": len(common),
           "scenes_dropped": dropped,
           "n_scenes_dropped": len(dropped),
           "transport_gain_db": round(t_fwd, 3),
           "mechanism_gain_db": round(m_fwd, 3),
           "reverse_order": {
               "n_scenes_common": len(r_common),
               "transport_gain_db": round(t_rev, 3),
               "mechanism_gain_db": round(m_rev, 3),
           },
           "order_sensitivity_db": round(abs(m_fwd - m_rev), 3),
           "order_statement": (
               "покроковий внесок залежить від порядку застосування змін. "
               f"За прямого порядку механізми дають {m_fwd:+.2f} дБ, за "
               f"зворотного {m_rev:+.2f} дБ; розбіжність "
               f"{abs(m_fwd - m_rev):.2f} дБ - це і є ціна вибору порядку, "
               "а не властивість самих механізмів. Спільним для обох є знак"
               if (m_fwd < 0) == (m_rev < 0) else
               "покроковий внесок залежить від порядку, і в цьому випадку "
               "навіть ЗНАК внеску механізмів різний - висновок про їхню "
               "користь не може спиратися на один розклад"),
           "note": ("кожен крок змінює рівно одну річ відносно попереднього; "
                    "кліпи, повтори й реалізації каналу спільні для всіх кроків, "
                    "а середні рахуються по спільному набору сцен. Різниці "
                    "супроводжуються парними bootstrap-інтервалами, і поряд з "
                    "PSNR розкладаються покриття й доступність")}
    write_json(os.path.join(out_dir, "e13.json"), out)
    return out


# ----------------------------------------------------------------------- E15
#: The operating map's axes.  Burst **length** and burst **rate** are varied
#: together, because a threshold quoted on one of them alone is a threshold for
#: whatever the other one happened to be (defect R10).
#:
#: The length axis is dense where the methods change places - that is where a
#: reader needs resolution - and coarse elsewhere.
E15_BURST_LINES = (1, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64)
E15_BURST_RATES = (1.0, 2.0, 4.0, 8.0)

#: The methods the map compares.  ``B4t`` is the retuned baseline, which is the
#: only honest comparison for "where does the proposal help" (R08).
E15_METHODS = ("B4", "B4t", "P")

#: Fixed before the run: what counts as operable.  Stated here rather than
#: chosen after looking at the surface.
E15_CRITERION = {"coverage_verified_min": 0.90, "psnr_full_min": 20.0}


def run_e15(cfg: ExperimentConfig, out_dir: str, workers: int = 8,
            repetitions: int = 3, max_frames: int = 6, n_clips: int = 6,
            progress: Progress = None) -> Dict[str, Any]:
    """E15: where does the proposal help, as a map rather than a threshold.

    The earlier report said the advantage "breaks at 28 burst lines".  That is
    one number read off one curve, at one burst rate, one budget and one model,
    and it was quoted as though it described a video link.  Here the same
    question is asked over a **grid** of burst length and burst rate, with:

    * the paired ``P - B4t`` difference and its interval in every cell, so the
      map shows where the difference is established and where it is not;
    * the operability criterion fixed in :data:`E15_CRITERION` before the run;
    * measured cells marked as measured.  Nothing between them is filled in;
      a figure may interpolate for the eye, and it says when it does.
    """
    ensure_dir(out_dir)
    cells: List[Dict[str, Any]] = []
    scene_rows: List[Dict[str, Any]] = []
    total = len(E15_BURST_RATES)

    for i, rate in enumerate(E15_BURST_RATES):
        if progress:
            progress(f"E15 rate={rate:g}", i / max(1, total), {})
        points = [ChannelPoint(
            base="mild",
            overrides=(("burst_len_lines", float(n)),
                       ("burst_rate_per_frame", float(rate))),
            axis="burst_len_lines", value=float(n)) for n in E15_BURST_LINES]
        spec = MatrixSpec(methods=E15_METHODS, channels=points,
                          repetitions=repetitions, splits=("test",),
                          max_frames=max_frames, max_clips=n_clips,
                          name=f"E15-rate{rate:g}")
        frames = _run(cfg, spec, os.path.join(out_dir, "_jobs", f"map{rate:g}"),
                      workers)
        by_point: Dict[Tuple[float, str], List[Dict[str, Any]]] = {}
        for r in frames:
            key = (float(r.get("channel_value", float("nan"))),
                   str(r.get("method", "")))
            by_point.setdefault(key, []).append(r)

        for lines in E15_BURST_LINES:
            per_method: Dict[str, Dict[str, Dict[str, float]]] = {}
            for m in E15_METHODS:
                rows = by_point.get((float(lines), m), [])
                if not rows:
                    continue
                per_method[m] = {
                    "psnr_full": _scene_values(rows, "psnr_full"),
                    "coverage": _scene_values(rows, "coverage"),
                    "availability": _instant_values(rows, "availability"),
                    "operable": _operable_by_scene(rows, E15_CRITERION),
                }
                for scene, v in per_method[m]["psnr_full"].items():
                    scene_rows.append({
                        "burst_rate_per_frame": rate, "burst_len_lines": lines,
                        "method": m, "scene": scene, "psnr_full": v,
                        "coverage": per_method[m]["coverage"].get(scene),
                        "availability": per_method[m]["availability"].get(scene),
                        "operable": per_method[m]["operable"].get(scene)})
            cell: Dict[str, Any] = {
                "burst_rate_per_frame": rate, "burst_len_lines": lines,
                "measured": True,
            }
            for m in E15_METHODS:
                d = per_method.get(m)
                cell[f"psnr_{m}"] = (float(np.mean(list(d["psnr_full"].values())))
                                     if d and d["psnr_full"] else float("nan"))
                cell[f"operable_{m}"] = (float(np.mean(list(d["operable"].values())))
                                         if d and d["operable"] else float("nan"))
                cell[f"availability_{m}"] = (
                    float(np.mean(list(d["availability"].values())))
                    if d and d["availability"] else float("nan"))
            for a, b in (("P", "B4t"), ("P", "B4")):
                if a in per_method and b in per_method:
                    ci = _paired_ci(per_method[a]["psnr_full"],
                                    per_method[b]["psnr_full"])
                    cell.update({
                        f"d_{a}_{b}": round(ci["delta"], 4),
                        f"d_{a}_{b}_lo": round(ci["lo"], 4),
                        f"d_{a}_{b}_hi": round(ci["hi"], 4),
                        f"d_{a}_{b}_significant": ci["significant"],
                        f"d_{a}_{b}_n": ci["n_scenes"]})
            cells.append(cell)

    write_csv(os.path.join(out_dir, "operating_map.csv"), cells)
    write_csv(os.path.join(out_dir, "operating_map_scenes.csv"), scene_rows)

    crossings = _crossings(cells)
    out = {
        "kind": "E15",
        "axes": {"burst_len_lines": list(E15_BURST_LINES),
                 "burst_rate_per_frame": list(E15_BURST_RATES)},
        "methods": list(E15_METHODS),
        "criterion": dict(E15_CRITERION),
        "n_cells": len(cells),
        "n_cells_where_P_beats_B4t": sum(
            1 for c in cells if c.get("d_P_B4t_significant")
            and c.get("d_P_B4t", 0) > 0),
        "n_cells_where_B4t_beats_P": sum(
            1 for c in cells if c.get("d_P_B4t_significant")
            and c.get("d_P_B4t", 0) < 0),
        "n_cells_undetermined": sum(
            1 for c in cells if not c.get("d_P_B4t_significant")),
        "crossings": crossings,
        "note": ("це карта умов для ЦІЄЇ моделі каналу і ЦЬОГО бюджету. "
                 "Поріг не є характеристикою реального відеотракту: він "
                 "залежить від частоти пакетів так само, як від їхньої "
                 "довжини, і в кожній клітинці наведено інтервал різниці. "
                 "Виміряні лише перелічені точки; будь-яке значення між ними "
                 "- інтерполяція для ока, а не вимірювання"),
    }
    write_json(os.path.join(out_dir, "e15.json"), out)
    return out


def _operable_by_scene(rows, criterion: Dict[str, float]) -> Dict[str, float]:
    """Fraction of a scene's **scheduled instants** meeting the criterion."""
    from collections import defaultdict

    cov_min = float(criterion["coverage_verified_min"])
    psnr_min = float(criterion["psnr_full_min"])
    per: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        scene = str(r.get("scene", ""))
        if str(r.get("status", "ok")) != "ok":
            per[scene].append(0.0)
            continue
        try:
            cov = float(r.get("coverage", "nan"))
            psnr = float(r.get("psnr_full", "nan"))
        except (TypeError, ValueError):
            per[scene].append(0.0)
            continue
        per[scene].append(1.0 if (np.isfinite(cov) and np.isfinite(psnr)
                                  and cov >= cov_min and psnr >= psnr_min) else 0.0)
    return {k: float(np.mean(v)) for k, v in per.items() if v}


def _crossings(cells: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Where the sign of ``P - B4t`` changes along the length axis.

    Reported as the **bracket** between the two measured points it happens
    between, never as a single interpolated number pretending to be measured.
    """
    out: List[Dict[str, Any]] = []
    for rate in sorted({c["burst_rate_per_frame"] for c in cells}):
        row = sorted([c for c in cells if c["burst_rate_per_frame"] == rate],
                     key=lambda c: c["burst_len_lines"])
        for a, b in zip(row, row[1:]):
            da, db = a.get("d_P_B4t"), b.get("d_P_B4t")
            if da is None or db is None:
                continue
            if np.isfinite(da) and np.isfinite(db) and (da > 0) != (db > 0):
                out.append({
                    "burst_rate_per_frame": rate,
                    "between_lines": [a["burst_len_lines"], b["burst_len_lines"]],
                    "delta_before": da, "delta_after": db,
                    "both_significant": bool(a.get("d_P_B4t_significant")
                                             and b.get("d_P_B4t_significant")),
                    "note": ("перетин лежить між двома ВИМІРЯНИМИ точками; "
                             "точне значення всередині не вимірювалося"),
                })
    return out


__all__ = ["run_e01", "run_e03", "run_e04", "run_e05", "run_e06",
           "run_e07", "run_e09", "run_e10", "run_e13", "run_e15", "E13_CHAIN",
           "E13_CHAIN_REVERSE", "E15_BURST_LINES", "E15_BURST_RATES",
           "E15_CRITERION",
           "E01_PLANNED", "E03_GRIDS", "E03_METHODS"]
