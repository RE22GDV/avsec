"""Experiment runners: demo, comparison, attacks, tuning, CVBS and research.

Every runner writes, next to its results: the resolved configuration, the
environment record, the input hashes, the seeds of the reproducible non-secret
parts, and the counts of successful / failed / skipped trials.  Secret session
keys are never written.
"""
from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from avsec import evaluation as ev
from avsec import sources as src_mod
from avsec.attacks import (
    attack_boundary_reassembly,
    attack_chosen_plaintext,
    attack_known_pair,
    attack_lfsr_bruteforce,
    attack_multi_frame_reuse,
)
from avsec.baselines import (
    DigitalMethod,
    Method,
    MethodResult,
    WholeFrameAEADMethod,
    make_b0_analog,
    make_b1_lfsr,
    make_b2_cryptoperm,
)
from avsec.budget import capacity_check, compute_latency, raw_video_bitrate
from avsec.config import ALL_METHODS, ExperimentConfig, MethodProfile
from avsec.crypto import AuthenticationFailed, ReplayDetected
from avsec.lfsr import BlockScrambler, correlation_similarity
from avsec.sources import FrameSource
from avsec.transmitter import CapacityExceeded
from avsec.utils import (
    RunRecord,
    StageTimer,
    append_jsonl,
    ensure_dir,
    environment_record,
    experiment_rng,
    mean_ci95,
    write_csv,
    write_json,
)

Progress = Optional[Callable[[str, float, Dict[str, Any]], None]]


def _emit(progress: Progress, stage: str, frac: float, **info: Any) -> None:
    if progress:
        try:
            progress(stage, float(frac), info)
        except Exception:
            pass


# --------------------------------------------------------------------- setup
def build_sources(cfg: ExperimentConfig) -> List[FrameSource]:
    spec = dict(cfg.sources or {})
    spec.setdefault("kind", "synthetic")
    spec["height"] = cfg.frame_height
    spec["width"] = cfg.frame_width
    spec.setdefault("n_frames", cfg.n_frames)
    return src_mod.build_sources(spec)


def build_methods(cfg: ExperimentConfig, timer: Optional[StageTimer] = None
                  ) -> Dict[str, Method]:
    """Instantiate the requested methods under the shared budget."""
    timer = timer or StageTimer()
    master = cfg.master_secret()
    chan = cfg.channel_config()
    H, W = cfg.frame_height, cfg.frame_width
    ref_transport = cfg.profile("B4").transport_config(cfg.budget)
    out: Dict[str, Method] = {}

    for name in cfg.methods:
        if name == "B0a":
            out[name] = make_b0_analog(ref_transport, chan, H, W, timer)
        elif name == "B1":
            out[name] = make_b1_lfsr(ref_transport, chan, H, W, cfg.lfsr, timer)
        elif name == "B2":
            out[name] = make_b2_cryptoperm(ref_transport, chan, H, W, cfg.b2_grid[0],
                                           cfg.b2_grid[1], master, timer=timer)
        elif name in ("B0d", "B4", "P"):
            prof = cfg.profile(name)
            out[name] = DigitalMethod(
                name, prof.source_config(), prof.transport_config(cfg.budget), chan,
                master, H, W, cfg.crypto, secure=(name != "B0d"), fill=cfg.fill,
                timer=timer,
                notes=("diagnostic transport without cryptography - authenticates "
                       "nothing" if name == "B0d" else ""))
        elif name == "B3":
            prof = cfg.profile("B3")
            out[name] = WholeFrameAEADMethod(
                prof.source_config(), prof.transport_config(cfg.budget), chan, master,
                H, W, cfg.crypto, fill=cfg.fill, timer=timer)
        else:
            raise KeyError(f"unknown method {name!r}; have {ALL_METHODS}")
    return out


# ----------------------------------------------------------------- demo run
def run_demo(cfg: ExperimentConfig, output_dir: str,
             progress: Progress = None) -> Dict[str, Any]:
    """One frame through every requested method, with images and metrics."""
    ensure_dir(output_dir)
    timer = StageTimer()
    srcs = build_sources(cfg)
    methods = build_methods(cfg, timer)
    frame = srcs[0].frames[0]
    rows: List[Dict[str, Any]] = []
    images: Dict[str, Dict[str, str]] = {}
    errors: Dict[str, str] = {}

    for i, (name, m) in enumerate(methods.items()):
        _emit(progress, f"method {name}", i / max(len(methods), 1), method=name)
        m.reset()
        rng = experiment_rng(cfg.seed, "demo", name, 0)
        try:
            res = m.process(frame, 0, rng)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        rows.append(res.metrics.to_row())
        if cfg.save_images:
            d = ensure_dir(os.path.join(output_dir, "images", name))
            images[name] = {k: ev.save_image(os.path.join(d, f"{k}.png"), v)
                            for k, v in res.images().items()}

    record = RunRecord(config_id=cfg.name, config=cfg.to_dict())
    record.inputs = {s.name: s.content_hash() for s in srcs}
    out = {
        "kind": "demo", "rows": rows, "images": images, "errors": errors,
        "timing": timer.summary(), "record": record.to_dict(),
        "source": srcs[0].summary(),
    }
    write_json(os.path.join(output_dir, "demo.json"), out)
    write_csv(os.path.join(output_dir, "demo_metrics.csv"), rows)
    return out


# ----------------------------------------------------------- comparison run
def run_comparison(cfg: ExperimentConfig, output_dir: str,
                   progress: Progress = None) -> Dict[str, Any]:
    """B0-B4 and P on a shared source set, shared budget and shared channel seeds."""
    ensure_dir(output_dir)
    timer = StageTimer()
    srcs = build_sources(cfg)
    methods = build_methods(cfg, timer)
    rows: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}
    per_sequence: Dict[str, Dict[str, List[float]]] = {}
    counters = {"ok": 0, "failed": 0, "skipped": 0}
    total = max(1, len(methods) * len(srcs) * max(cfg.channel_realisations, 1))
    done = 0
    jsonl = os.path.join(output_dir, "frames.jsonl")
    if os.path.exists(jsonl):
        os.remove(jsonl)

    for name, m in methods.items():
        per_sequence[name] = {}
        for si, src in enumerate(srcs):
            for rep in range(max(cfg.channel_realisations, 1)):
                done += 1
                _emit(progress, f"{name} / {src.name} / run {rep}", done / total,
                      method=name, source=src.name)
                m.reset()
                seq_psnr: List[float] = []
                seq_cov: List[float] = []
                for fi, frame in enumerate(src.frames[: cfg.max_frames_per_source]):
                    rng = experiment_rng(cfg.seed, "cmp", name, si, rep, fi)
                    try:
                        res = m.process(frame, fi, rng)
                    except CapacityExceeded as exc:
                        errors[f"{name}/{src.name}"] = f"capacity: {exc}"
                        counters["skipped"] += 1
                        break
                    except Exception as exc:
                        errors[f"{name}/{src.name}"] = f"{type(exc).__name__}: {exc}"
                        counters["failed"] += 1
                        break
                    counters["ok"] += 1
                    row = res.metrics.to_row()
                    row.update({"source": src.name, "provenance": src.provenance,
                                "realisation": rep,
                                "authenticated": bool(getattr(m, "authenticated", False))})
                    rows.append(row)
                    append_jsonl(jsonl, row)
                    seq_psnr.append(res.metrics.psnr_full)
                    seq_cov.append(res.metrics.coverage)
                if seq_psnr:
                    key = f"{src.name}#{rep}"
                    per_sequence[name][key] = [float(np.mean(seq_psnr)),
                                               float(np.mean(seq_cov))]

    # aggregate over *independent sequences*, not over neighbouring frames
    summary: List[Dict[str, Any]] = []
    for name, seqs in per_sequence.items():
        p = [v[0] for v in seqs.values()]
        c = [v[1] for v in seqs.values()]
        summary.append({
            "method": name,
            "sequences": len(seqs),
            "authenticated": bool(getattr(methods[name], "authenticated", False)),
            **{f"psnr_{k}": v for k, v in mean_ci95(p).items()},
            **{f"coverage_{k}": v for k, v in mean_ci95(c).items()},
        })

    paired: List[Dict[str, Any]] = []
    keys = sorted({k for s in per_sequence.values() for k in s})
    for a in per_sequence:
        for b in per_sequence:
            if a >= b:
                continue
            va = [per_sequence[a][k][0] for k in keys if k in per_sequence[a] and k in per_sequence[b]]
            vb = [per_sequence[b][k][0] for k in keys if k in per_sequence[a] and k in per_sequence[b]]
            if len(va) >= 2:
                d = ev.paired_difference(va, vb)
                d.update({"a": a, "b": b, "metric": "psnr_full"})
                paired.append(d)

    record = RunRecord(config_id=cfg.name, config=cfg.to_dict())
    record.inputs = {s.name: s.content_hash() for s in srcs}
    record.counters.update(counters)
    out = {
        "kind": "comparison", "summary": summary, "paired": paired, "errors": errors,
        "timing": timer.summary(), "record": record.to_dict(),
        "sources": [s.summary() for s in srcs],
        "budget": cfg.budget.describe(),
        "n_rows": len(rows),
        "provenance_warning": (
            "усі джерела процедурно згенеровані (SYNTHETIC DATA)" if all(
                s.provenance == src_mod.PROV_SYNTHETIC for s in srcs) else ""),
    }
    write_json(os.path.join(output_dir, "comparison.json"), out)
    write_csv(os.path.join(output_dir, "frames.csv"), rows)
    write_csv(os.path.join(output_dir, "summary.csv"), summary)
    return out


# --------------------------------------------------------------- attack run
def run_attacks(cfg: ExperimentConfig, output_dir: str,
                progress: Progress = None) -> Dict[str, Any]:
    """Cryptanalysis of the permutation baselines plus AEAD protocol checks."""
    ensure_dir(output_dir)
    H, W = cfg.frame_height, cfg.frame_width
    rows, cols = cfg.lfsr.grid_rows, cfg.lfsr.grid_cols
    scr = BlockScrambler(cfg.lfsr)
    results: List[Dict[str, Any]] = []

    _emit(progress, "chosen plaintext", 0.05)
    r = attack_chosen_plaintext(scr, 0, H, W)
    results.append(r.to_dict())

    _emit(progress, "known pair / boundary", 0.2)
    for name in ("smooth", "edges", "text", "texture"):
        img = src_mod.PATTERNS[name](H, W)
        s = scr.scramble(img, 0)
        kp = attack_known_pair(scr._fit(img), s, rows, cols)
        kp.metrics["permutation_accuracy"] = float(
            (kp.recovered_permutation == scr.permutation(0)).mean())
        d = kp.to_dict(); d["content"] = name
        results.append(d)
        ba = attack_boundary_reassembly(s, rows, cols, scr.permutation(0), scr._fit(img))
        d = ba.to_dict(); d["content"] = name
        results.append(d)
        if cfg.save_images and ba.reconstructed is not None:
            dd = ensure_dir(os.path.join(output_dir, "images"))
            ev.save_image(os.path.join(dd, f"b1_{name}_original.png"), img)
            ev.save_image(os.path.join(dd, f"b1_{name}_scrambled.png"), s)
            ev.save_image(os.path.join(dd, f"b1_{name}_boundary_attack.png"),
                          ba.reconstructed)

    _emit(progress, "multi frame reuse", 0.5)
    frames = [src_mod.pattern_edges(H, W, i * 0.05) for i in range(12)]
    results.append(attack_multi_frame_reuse(frames, scr, rows, cols).to_dict())

    _emit(progress, "lfsr brute force", 0.6)
    img = src_mod.pattern_texture(H, W)
    bf = attack_lfsr_bruteforce(scr._fit(img), scr.scramble(img, 0), cfg.lfsr,
                                max_states=min((1 << cfg.lfsr.lfsr.width) - 1, 70000))
    results.append(bf.to_dict())

    _emit(progress, "similarity table", 0.75)
    similarity: List[Dict[str, Any]] = []
    from avsec.lfsr import LFSRConfig, ScramblerConfig

    for seed in (44257, 1234, 1111):
        for name in ("smooth", "edges", "texture"):
            img = src_mod.PATTERNS[name](H, W)
            sc = BlockScrambler(ScramblerConfig(
                grid_rows=rows, grid_cols=cols,
                lfsr=LFSRConfig(cfg.lfsr.lfsr.width, cfg.lfsr.lfsr.taps, seed,
                                cfg.lfsr.lfsr.form),
                variant=cfg.lfsr.variant, size_policy=cfg.lfsr.size_policy))
            s = sc.scramble(img, 0)
            rec = sc.descramble(s, 0, (H, W))
            r1, sim1 = correlation_similarity(sc._fit(img), s)
            r2, sim2 = correlation_similarity(img, rec)
            similarity.append({
                "seed": seed, "content": name,
                "corr_original_vs_scrambled": r1, "similarity_scrambled_pct": sim1,
                "corr_original_vs_decrypted": r2, "similarity_decrypted_pct": sim2,
                "exact_recovery": bool(np.array_equal(img, rec)),
            })

    _emit(progress, "protocol checks", 0.85)
    protocol = run_protocol_checks(cfg)

    record = RunRecord(config_id=cfg.name, config=cfg.to_dict())
    out = {
        "kind": "attacks", "results": results, "similarity_table": similarity,
        "protocol_checks": protocol, "record": record.to_dict(),
        "lfsr": cfg.lfsr.describe(),
        "caveat": ("відновлення однієї сталої перестановки нічого не говорить про "
                   "незалежно перегенеровану перестановку; невдала атака не є "
                   "доказом безпеки"),
    }
    write_json(os.path.join(output_dir, "attacks.json"), out)
    write_csv(os.path.join(output_dir, "similarity.csv"), similarity)
    write_csv(os.path.join(output_dir, "attacks.csv"), results)
    return out


def run_protocol_checks(cfg: ExperimentConfig) -> List[Dict[str, Any]]:
    """Tamper / replay / session checks against the real AEAD pipeline."""
    from avsec.crypto import CryptoProfile, lab_master_secret, make_session
    from avsec.framing import Geometry, UnitHeader, parse_header

    master = lab_master_secret(cfg.seed)
    keys, sealer, opener = make_session(master, cfg.crypto)
    hdr = UnitHeader(profile_id=1, session_id=keys.session_id, stream_id=0, codec_id=1,
                     frame_id=3, stripe_id=2, desc_id=0, seg_id=0, n_segs=1, n_descs=1,
                     unit_seq=0, geometry=Geometry(0, 16, 320, 8), payload_len=10)
    payload = b"0123456789" + bytes(54)
    seq, ct = sealer.seal(payload, hdr.with_seq(0).core_bytes(), counter=0)
    aad = hdr.with_seq(seq).core_bytes()

    checks: List[Dict[str, Any]] = []

    def _check(name: str, fn, expect: str) -> None:
        try:
            fn()
            checks.append({"check": name, "expected": expect, "outcome": "ACCEPTED",
                           "passed": expect == "accept"})
        except Exception as exc:
            checks.append({"check": name, "expected": expect,
                           "outcome": f"REJECTED ({type(exc).__name__})",
                           "passed": expect == "reject", "detail": str(exc)[:120]})

    _check("valid unit", lambda: opener.open(seq, ct, aad), "accept")
    _check("replay of an accepted unit", lambda: opener.open(seq, ct, aad), "reject")
    _check("modified ciphertext",
           lambda: opener.open(seq + 1, bytes([ct[0] ^ 1]) + ct[1:], aad), "reject")
    _check("modified tag",
           lambda: opener.open(seq + 2, ct[:-1] + bytes([ct[-1] ^ 0x80]), aad), "reject")
    _check("modified authenticated coordinates",
           lambda: opener.open(seq + 3, ct, hdr.with_seq(seq).core_bytes().replace(
               b"\x00\x03", b"\x00\x04", 1)), "reject")
    _check("unit moved to another stripe",
           lambda: opener.open(seq + 4, ct,
                               UnitHeader(**{**hdr.__dict__, "stripe_id": 9,
                                             "unit_seq": seq}).core_bytes()), "reject")
    _check("unit moved to another frame",
           lambda: opener.open(seq + 5, ct,
                               UnitHeader(**{**hdr.__dict__, "frame_id": 99,
                                             "unit_seq": seq}).core_bytes()), "reject")

    keys2, sealer2, opener2 = make_session(master, cfg.crypto)
    _check("unit from another session", lambda: opener2.open(seq, ct, aad), "reject")

    master2 = lab_master_secret(cfg.seed + 1)
    _, _, opener3 = make_session(master2, cfg.crypto, keys.session_id)
    _check("wrong master key", lambda: opener3.open(seq, ct, aad), "reject")

    def _bad_version():
        import dataclasses

        parse_header(dataclasses.replace(hdr, version=7).to_bytes())

    _check("unknown protocol version", _bad_version, "reject")
    _check("out-of-range payload length",
           lambda: parse_header(hdr.with_payload_len(0).to_bytes()), "reject")
    _check("geometry outside the frame",
           lambda: Geometry(0, 236, 160, 6, 2, 2, 1, 1).validate(320, 240), "reject")
    _check("header CRC corruption", lambda: parse_header(
        bytes([hdr.to_bytes()[0] ^ 0xFF]) + hdr.to_bytes()[1:]), "reject")

    # FEC-corrected modification must NOT count as a forgery
    from avsec.fec import FECConfig, RSCodec

    rs = RSCodec(FECConfig(k=191, nsym=64))
    enc = bytearray(rs.encode(ct))
    for i in range(10):
        enc[i] ^= 0xFF
    dec, _ = rs.try_decode(bytes(enc), len(ct))
    checks.append({
        "check": "modification fully repaired by FEC",
        "expected": "not a forgery (identical protected message restored)",
        "outcome": "restored" if dec == ct else "not restored",
        "passed": dec == ct,
    })
    enc2 = bytearray(rs.encode(ct))
    for i in range(0, 200, 2):
        if i < len(enc2):
            enc2[i] ^= 0xA5
    dec2, _ = rs.try_decode(bytes(enc2), len(ct))
    ok = True
    detail = "uncorrectable, discarded by FEC"
    if dec2 is not None and dec2 != ct:
        try:
            opener.open(seq + 20, dec2, aad)
            ok = False
            detail = "AEAD ACCEPTED a miscorrected message"
        except Exception as exc:
            detail = f"miscorrected by FEC, rejected by AEAD ({type(exc).__name__})"
    checks.append({
        "check": "modification beyond FEC capability",
        "expected": "rejected by FEC or by AEAD", "outcome": detail, "passed": ok,
    })
    return checks


# ---------------------------------------------------------------- CVBS demo
def run_cvbs(cfg: ExperimentConfig, output_dir: str, preset_name: str = "mild",
             method: str = "B4", progress: Progress = None) -> Dict[str, Any]:
    """Short end-to-end demonstration through the level-B composite model."""
    from avsec.modem.cvbs import (
        CVBSChannel,
        CVBSGenerator,
        CVBSProfile,
        CVBSReceiver,
        cvbs_preset,
    )

    ensure_dir(output_dir)
    timer = StageTimer()
    profile = CVBSProfile()
    gen = CVBSGenerator(profile)
    rec = CVBSReceiver(profile)
    chan = CVBSChannel(cvbs_preset(preset_name), profile)

    prof = cfg.profile(method if method in cfg.profiles else "B4")
    source_cfg = prof.source_config()
    transport = prof.transport_config(cfg.budget)
    master = cfg.master_secret()
    from avsec.receiver import Receiver
    from avsec.source_coding import FrameAssembler
    from avsec.transmitter import Transmitter

    H, W = cfg.frame_height, cfg.frame_width
    tx = Transmitter(source_cfg, transport, master, cfg.crypto, frame_width=W,
                     frame_height=H, timer=timer)
    rx = Receiver(source_cfg, transport, master, cfg.crypto, frame_width=W,
                  frame_height=H, timer=timer)
    assembler = FrameAssembler(H, W, cfg.fill)

    srcs = build_sources(cfg)
    frame = srcs[0].frames[0]
    _emit(progress, "transmit", 0.1)
    txf = tx.encode_frame(frame, 0)

    placements = []
    status_counts: Dict[str, int] = {}
    cvbs_stats: List[Dict[str, Any]] = []
    rx_raster0 = None
    for i, raster in enumerate(txf.rasters):
        _emit(progress, f"cvbs raster {i}", 0.2 + 0.6 * i / max(len(txf.rasters), 1))
        with timer("cvbs.generate"):
            sig = gen.generate_frame(raster)
        rng = experiment_rng(cfg.seed, "cvbs", preset_name, i)
        with timer("cvbs.channel"):
            y, _ = chan.apply(sig, rng)
        with timer("cvbs.receive"):
            got = rec.receive_frame(y, transport.modem.raster_width,
                                    transport.modem.raster_height)
        cvbs_stats.append(got.summary())
        if rx_raster0 is None:
            rx_raster0 = got.raster
        out = rx.receive_raster(got.raster)
        for o in out.outcomes:
            status_counts[o.status.value] = status_counts.get(o.status.value, 0) + 1
            if o.ok and o.samples is not None and o.header is not None:
                placements.append((o.header.geometry, o.samples))

    asm = assembler.assemble(placements)
    q = ev.quality_pair(frame, asm.image, asm.available)
    images: Dict[str, str] = {}
    if cfg.save_images:
        d = ensure_dir(os.path.join(output_dir, "images"))
        images["original"] = ev.save_image(os.path.join(d, "original.png"), frame)
        images["transmitted"] = ev.save_image(os.path.join(d, "transmitted_raster.png"),
                                              txf.rasters[0])
        if rx_raster0 is not None:
            images["received"] = ev.save_image(os.path.join(d, "received_raster.png"),
                                               rx_raster0)
        images["reconstructed"] = ev.save_image(os.path.join(d, "reconstructed.png"),
                                                asm.image)
        images["availability"] = ev.save_image(
            os.path.join(d, "availability.png"),
            ev.availability_overlay(asm.image, asm.available, asm.from_previous))

    out = {
        "kind": "cvbs", "preset": preset_name, "method": method,
        "profile": profile.describe(), "channel": chan.cfg.describe(),
        "cvbs_stats": cvbs_stats, "status_counts": status_counts,
        "quality": q, "images": images, "timing": timer.summary(),
        "record": RunRecord(config_id=cfg.name, config=cfg.to_dict()).to_dict(),
        "compliance_note": (
            "спрощена структура 625/50 - перелік відмінностей у "
            "CVBSProfile.describe()['simplifications']; це не сертифікована реалізація "
            "стандарту і вона не містить моделі радіочастотного тракту"),
    }
    write_json(os.path.join(output_dir, "cvbs.json"), out)
    return out


# --------------------------------------------------------------- tuning run
def run_tuning(cfg: ExperimentConfig, output_dir: str, space: Optional[Any] = None,
               max_frames: int = 1, progress: Progress = None) -> Dict[str, Any]:
    """Parameter search on the calibration split, selection on validation."""
    from avsec.optimization import ParameterSpace, Tuner, split_sources

    ensure_dir(output_dir)
    srcs = build_sources(cfg)
    splits = split_sources(srcs, seed=cfg.seed)
    space = space or ParameterSpace()
    master = cfg.master_secret()
    tuner = Tuner(cfg.budget, master, cfg.channel_config(), cfg.frame_height,
                  cfg.frame_width, seed=cfg.seed)
    cost = Tuner.estimate_cost(space, splits["calibration"], max_frames)
    _emit(progress, "estimated cost", 0.0, **cost)

    scored: List[Any] = []

    def _cb(i: int, n: int, sc: Any) -> None:
        _emit(progress, f"candidate {i}/{n}", i / max(n, 1), key=sc.candidate.key(),
              admissible=sc.admissible)

    scored = tuner.search(space, splits["calibration"], "tune", max_frames, _cb)
    rows = [s.to_row() for s in scored]
    admissible = [s for s in scored if s.admissible]

    finalists = sorted(admissible, key=lambda s: -s.psnr_full)[:5]
    validated: List[Dict[str, Any]] = []
    for s in finalists:
        v = tuner.score(s.candidate, splits["validation"], max_frames,
                        split="validation")
        row = v.to_row()
        row["calibration_psnr"] = s.psnr_full
        validated.append(row)
    best = max(validated, key=lambda r: r.get("psnr_full", -1e9)) if validated else None

    out = {
        "kind": "tuning", "estimated_cost": cost,
        "n_candidates": len(scored), "n_admissible": len(admissible),
        "calibration": rows, "validation": validated, "selected": best,
        "splits": {k: [s.name for s in v] for k, v in splits.items()},
        "budget": cfg.budget.describe(),
        "record": RunRecord(config_id=cfg.name, config=cfg.to_dict()).to_dict(),
        "discipline": ("параметри добиралися на калібрувальній частині, переможець "
                       "обраний на валідаційній, тестова частина цією командою не "
                       "використовується"),
    }
    write_json(os.path.join(output_dir, "tuning.json"), out)
    write_csv(os.path.join(output_dir, "candidates.csv"), rows)
    write_csv(os.path.join(output_dir, "validation.csv"), validated)
    return out


# -------------------------------------------------------------- ablation run
ABLATIONS: Dict[str, Dict[str, Any]] = {
    "P (proposed)": {},
    "single description": {"n_descriptions": 1},
    "sequential placement": {"interleaver": ("sequential", 1, 0)},
    "plain block interleaving": {"interleaver": ("block", 32, 0)},
    "deep block interleaving": {"interleaver": ("block", 278, 0)},
    "smaller AEAD unit": {"max_unit_payload": 192},
    "larger AEAD unit": {"max_unit_payload": 896},
    "weaker FEC": {"fec_nsym": 48},
    "stronger FEC": {"fec_nsym": 160},
    "shallow BAWP window": {"interleaver": ("bawp", 0, 4)},
}


def run_ablations(cfg: ExperimentConfig, output_dir: str,
                  progress: Progress = None) -> Dict[str, Any]:
    """Ablate the proposed configuration one factor at a time."""
    ensure_dir(output_dir)
    srcs = build_sources(cfg)
    base = cfg.profile("P")
    master = cfg.master_secret()
    chan = cfg.channel_config()
    H, W = cfg.frame_height, cfg.frame_width
    rows: List[Dict[str, Any]] = []

    for i, (name, delta) in enumerate(ABLATIONS.items()):
        _emit(progress, name, i / max(len(ABLATIONS), 1), variant=name)
        prof = MethodProfile(**{**base.to_dict(),
                                **{k: v for k, v in delta.items()}})
        prof = MethodProfile.from_dict(prof.to_dict())
        try:
            m = DigitalMethod(name, prof.source_config(),
                              prof.transport_config(cfg.budget), chan, master, H, W,
                              cfg.crypto, fill=cfg.fill)
        except Exception as exc:
            rows.append({"variant": name, "admissible": False,
                         "reason": f"{type(exc).__name__}: {exc}"})
            continue
        psnrs: List[float] = []
        covs: List[float] = []
        rasters: List[int] = []
        failed = ""
        for si, s in enumerate(srcs):
            m.reset()
            for fi, frame in enumerate(s.frames[: cfg.max_frames_per_source]):
                try:
                    res = m.process(frame, fi, experiment_rng(cfg.seed, "abl", name, si, fi))
                except Exception as exc:
                    failed = f"{type(exc).__name__}: {exc}"
                    break
                psnrs.append(res.metrics.psnr_full)
                covs.append(res.metrics.coverage)
                rasters.append(res.metrics.rasters)
            if failed:
                break
        if failed:
            rows.append({"variant": name, "admissible": False, "reason": failed})
            continue
        rows.append({
            "variant": name, "admissible": True,
            "psnr_mean": float(np.mean(psnrs)), "psnr_sd": float(np.std(psnrs, ddof=1))
            if len(psnrs) > 1 else 0.0,
            "coverage_mean": float(np.mean(covs)),
            "rasters_mean": float(np.mean(rasters)),
            "aead_tag_bytes": 16, "n_frames": len(psnrs),
            "config": prof.to_dict(),
        })
    out = {
        "kind": "ablations", "rows": rows,
        "note": ("криптографічна міцність і довжина тега (16 байтів) зафіксовані в "
                 "усіх варіантах; зменшення захисту ніколи не використовується як "
                 "спосіб показати виграш"),
        "record": RunRecord(config_id=cfg.name, config=cfg.to_dict()).to_dict(),
    }
    write_json(os.path.join(output_dir, "ablations.json"), out)
    write_csv(os.path.join(output_dir, "ablations.csv"),
              [{k: v for k, v in r.items() if k != "config"} for r in rows])
    return out


# ---------------------------------------------------------------- sweeps
def run_sweeps(cfg: ExperimentConfig, output_dir: str,
               presets: Sequence[str] = ("clean", "mild", "moderate", "bursty", "harsh"),
               progress: Progress = None) -> Dict[str, Any]:
    """Quality vs impairment strength, and quality vs channel occupancy."""
    ensure_dir(output_dir)
    srcs = build_sources(cfg)
    rows: List[Dict[str, Any]] = []
    skipped: Dict[str, str] = {}
    total = max(1, len(presets) * len(cfg.methods))
    done = 0
    for pi, pname in enumerate(presets):
        sub = copy.deepcopy(cfg)
        sub.channel_preset = pname
        sub.channel_overrides = {}
        methods = build_methods(sub)
        for name, m in methods.items():
            done += 1
            _emit(progress, f"{pname} / {name}", done / total)
            psnrs: List[float] = []
            covs: List[float] = []
            rast: List[int] = []
            for si, s in enumerate(srcs):
                m.reset()
                for fi, frame in enumerate(s.frames[: cfg.max_frames_per_source]):
                    try:
                        res = m.process(frame, fi,
                                        experiment_rng(cfg.seed, "sweep", pname, name, si, fi))
                    except Exception as exc:
                        skipped[f"{pname}/{name}/{s.name}"] = f"{type(exc).__name__}: {exc}"
                        break
                    psnrs.append(res.metrics.psnr_full)
                    covs.append(res.metrics.coverage)
                    rast.append(res.metrics.rasters)
            if psnrs:
                rows.append({"channel": pname, "method": name,
                             "psnr_mean": float(np.mean(psnrs)),
                             "coverage_mean": float(np.mean(covs)),
                             "rasters_mean": float(np.mean(rast)),
                             "n": len(psnrs)})
    plots: Dict[str, str] = {}
    if rows:
        methods = sorted({r["method"] for r in rows})
        xs = list(range(len(presets)))
        series = {m: [next((r["psnr_mean"] for r in rows
                            if r["method"] == m and r["channel"] == p), float("nan"))
                      for p in presets] for m in methods}
        plots["psnr_vs_channel"] = ev.save_line_plot(
            os.path.join(output_dir, "psnr_vs_channel.png"), xs, series,
            "channel preset (" + ", ".join(presets) + ")", "PSNR, dB",
            "Quality vs impairment strength")
        series_c = {m: [next((r["coverage_mean"] for r in rows
                              if r["method"] == m and r["channel"] == p), float("nan"))
                        for p in presets] for m in methods}
        plots["coverage_vs_channel"] = ev.save_line_plot(
            os.path.join(output_dir, "coverage_vs_channel.png"), xs, series_c,
            "channel preset (" + ", ".join(presets) + ")",
            "verified coverage", "Verified coverage vs impairment strength")
    out = {"kind": "sweeps", "rows": rows, "presets": list(presets), "plots": plots,
           "skipped": skipped,
           "record": RunRecord(config_id=cfg.name, config=cfg.to_dict()).to_dict()}
    write_json(os.path.join(output_dir, "sweeps.json"), out)
    write_csv(os.path.join(output_dir, "sweeps.csv"), rows)
    return out


# --------------------------------------------------------------- budget run
def run_budget(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Pure accounting: capacity, overhead breakdown and virtual latency."""
    out: Dict[str, Any] = {
        "raw_grayscale_reference_bps": raw_video_bitrate(
            cfg.frame_width, cfg.frame_height, cfg.budget.source_fps),
        "raw_grayscale_reference_note": (
            f"нестиснене {cfg.frame_width}x{cfg.frame_height}, 8 біт, "
            f"{cfg.budget.source_fps} кадр/с; ця величина НЕ вважається автоматично "
            "доступною в обраному тракті"),
        "budget": cfg.budget.describe(),
        "methods": {},
    }
    from avsec.budget import compute_budget
    from avsec.interleaving import Interleaver

    for name in ("B0d", "B3", "B4", "P"):
        prof = cfg.profile(name)
        tr = prof.transport_config(cfg.budget)
        b = compute_budget(tr.modem, tr.fec_payload, tr.fec_header,
                           prof.max_unit_payload, cfg.budget.raster_rate_hz)
        il = Interleaver(tr.interleaver, tr.modem.n_data_rows, tr.modem.n_data_cols)
        placeable = il.max_units(b.unit_symbols, prof.n_descriptions, b.units_per_raster)
        lat = compute_latency(tr.modem, il.accumulation_rows,
                              cfg.budget.rasters_per_frame, 1.0 / cfg.budget.source_fps,
                              prof.stripe_height, cfg.frame_height,
                              cfg.budget.raster_rate_hz)
        d = b.to_dict()
        d["units_per_raster_after_placement"] = placeable
        d["accumulation_rows"] = il.accumulation_rows
        d["latency"] = lat.to_dict()
        d["profile"] = prof.to_dict()
        out["methods"][name] = d
    return out


__all__ = [
    "build_sources", "build_methods", "run_demo", "run_comparison", "run_attacks",
    "run_protocol_checks", "run_cvbs", "run_tuning", "run_ablations", "run_sweeps",
    "run_budget", "ABLATIONS",
]
