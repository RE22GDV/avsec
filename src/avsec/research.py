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


def _codeword_damage(cells_per_unit, n_cols: int, n_rows: int, burst_rows: int,
                     descs) -> Dict[str, Any]:
    """Exhaustive single-burst damage, per unit and per stripe.

    For every possible burst start row this counts how many symbols of each
    placed unit fall inside the burst, and how many *distinct descriptions of
    the same image stripe* are hit at once.  The second number is the one that
    matters for MDC: hitting two of two descriptions of one stripe loses that
    stripe entirely, however few classes were touched elsewhere.
    """
    starts = max(1, n_rows - burst_rows + 1)
    per_unit_worst = np.zeros(len(cells_per_unit), dtype=np.int64)
    worst_joint = 0
    worst_joint_start = -1
    joint_hist: Dict[int, int] = {}
    rows_of = [np.asarray(c, dtype=np.int64) // int(n_cols) for c in cells_per_unit]
    n_desc = max(1, int(max(descs) + 1)) if len(descs) else 1
    # units are laid out description-major: unit u carries stripe u // n_desc
    stripe_of = [i // max(1, n_desc) for i in range(len(cells_per_unit))]

    for s0 in range(starts):
        s1 = s0 + burst_rows
        hit_per_stripe: Dict[int, set] = {}
        for u, rows in enumerate(rows_of):
            k = int(np.count_nonzero((rows >= s0) & (rows < s1)))
            if k > per_unit_worst[u]:
                per_unit_worst[u] = k
            if k:
                hit_per_stripe.setdefault(stripe_of[u], set()).add(int(descs[u]))
        joint = max((len(v) for v in hit_per_stripe.values()), default=0)
        joint_hist[joint] = joint_hist.get(joint, 0) + 1
        if joint > worst_joint:
            worst_joint, worst_joint_start = joint, s0
    return {
        "worst_bytes": int(per_unit_worst.max()) if per_unit_worst.size else 0,
        "worst_unit": int(per_unit_worst.argmax()) if per_unit_worst.size else -1,
        "mean_bytes": float(per_unit_worst.mean()) if per_unit_worst.size else 0.0,
        "worst_joint_descriptions": int(worst_joint),
        "worst_joint_start_row": int(worst_joint_start),
        "joint_histogram": joint_hist,
        "n_burst_positions": int(starts),
    }


def run_e05(cfg: ExperimentConfig, out_dir: str) -> Dict[str, Any]:
    """E05: BAWP studied on the placement itself, with no video involved.

    The plan's first level: identical coded units through different placements,
    with lengths, FEC, modem and description count held fixed.  A single burst
    is swept over **every** starting row and every length in the grid, and the
    actual damaged RS bytes are recorded - not just how many description
    classes were touched somewhere in the raster.
    """
    from avsec.budget import compute_budget
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
                    dmg = _codeword_damage(cells, n_cols, n_rows, br, descs)
                    rows.append({
                        "scheme": scheme_name, "n_descriptions": n_desc,
                        "column_twist": twist, "admissible": True, "reason": "",
                        "burst_lines": lines, "burst_symbol_rows": br,
                        "worst_damaged_bytes": dmg["worst_bytes"],
                        "mean_damaged_bytes": round(dmg["mean_bytes"], 2),
                        "worst_unit": dmg["worst_unit"],
                        "rs_nsym": nsym, "rs_k": tr.fec_payload.k,
                        "correctable_errors": nsym // 2,
                        "correctable_erasures": nsym,
                        "survives_as_errors": bool(2 * dmg["worst_bytes"] <= nsym),
                        "survives_as_erasures": bool(dmg["worst_bytes"] <= nsym),
                        "claimed_bound_bytes": _claimed_bound(
                            il_cfg, n_desc, br, unit_symbols, n_rows),
                        "n_units_placed": n_units,
                        "accumulation_rows": il.accumulation_rows,
                        "n_burst_positions": dmg["n_burst_positions"],
                    })
                    for k, count in sorted(dmg["joint_histogram"].items()):
                        joint_rows.append({
                            "scheme": scheme_name, "n_descriptions": n_desc,
                            "column_twist": twist, "burst_lines": lines,
                            "descriptions_lost_in_one_stripe": k,
                            "n_burst_positions": count,
                            "fraction": round(count / max(dmg["n_burst_positions"], 1), 4),
                            "stripe_lost": bool(k >= n_desc),
                        })
    write_csv(os.path.join(out_dir, "codewords.csv"), rows)
    write_csv(os.path.join(out_dir, "joint_loss.csv"), joint_rows)
    write_csv(os.path.join(out_dir, "placement_geometry.csv"), geometry)

    aliases = _placement_aliases(geometry)
    out = {"kind": "E05", "schemes": [s[0] for s in E05_SCHEMES],
           "burst_lines": list(E05_BURST_LINES), "n_rows": len(rows),
           "n_inadmissible": sum(1 for r in rows if not r["admissible"]),
           "placement_aliases": aliases,
           "note": ("вимірюються фактично пошкоджені байти RS-слова, а не лише "
                    "число зачеплених класів описів; за двох описів межа "
                    "«зачеплено не більше двох» сама по собі не гарантує "
                    "виживання жодного опису - тому окремо рахується втрата "
                    "обох описів однієї смуги")}
    write_json(os.path.join(out_dir, "e05.json"), out)
    return out


def _claimed_bound(il_cfg: Any, n_desc: int, burst_rows: int, unit_symbols: int,
                   n_rows: int) -> int:
    """What the analysis *claims* a burst can destroy, for the G29 check."""
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
    return int(burst_rows * per_row)


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
        rows.append({
            "study": name, "admissible": True, "reason": "", **extra,
            "psnr_full": agg[0]["psnr_full"] if agg else float("nan"),
            "psnr_sd": agg[0]["psnr_full_sd"] if agg else float("nan"),
            "coverage": cov[0]["coverage"] if cov else float("nan"),
            "max_age_frames": age[0]["max_age_frames"] if age else float("nan"),
            "n_scenes": agg[0]["n_scenes"] if agg else 0,
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
    out = {"kind": "E06", "channel": channel, "n_cells": len(rows),
           "n_inadmissible": sum(1 for r in rows if not r["admissible"]),
           "studies": sorted({r["study"] for r in rows}),
           "note": ("це дослідження МЕХАНІЗМУ: решта транспорту зафіксована. "
                    "Порівняння найкращих систем за однаковим бюджетом вимагає "
                    "переналаштування кожної клітинки і звітується окремо")}
    write_json(os.path.join(out_dir, "e06.json"), out)
    return out


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



__all__ = ["run_e01", "run_e03", "run_e04", "run_e05", "run_e06",
           "run_e07", "run_e09", "run_e10",
           "E01_PLANNED", "E03_GRIDS", "E03_METHODS"]
