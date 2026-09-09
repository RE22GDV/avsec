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
from avsec.channel import ChannelTrace
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


def _trace(cfg: ExperimentConfig, scene: str, repetition: int) -> "ChannelTrace":
    """Channel realisation for one (scene, repetition) under this profile.

    Deliberately independent of which method is being measured: that is what
    makes the paired comparison paired (defect F08).
    """
    return ChannelTrace(seed=cfg.seed, scene=scene, repetition=int(repetition),
                        profile=cfg.channel_preset,
                        rasters_per_frame=cfg.budget.rasters_per_frame)


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
        trace = _trace(cfg, srcs[0].name, 0)
        try:
            res = m.process(frame, 0, trace)
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
                # One trace per (scene, repetition): every method meets the
                # same damage in the same time slot (defect F08).
                trace = _trace(cfg, src.name, rep)
                seq_psnr: List[float] = []
                seq_cov: List[float] = []
                for fi, frame in enumerate(src.frames[: cfg.max_frames_per_source]):
                    try:
                        res = m.process(frame, fi, trace)
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
                                "realisation": rep, "channel_trace": trace.trace_id,
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
    """Isolated tamper / replay / parser checks.

    Defect F07: the previous version reused one opener and walked the counter
    forward for every mutation, so a rejection could have come from the nonce
    rather than from the field under test, and any ``Exception`` counted as a
    cryptographic rejection.

    Each check below therefore

    * builds a **fresh** sealer/opener pair and seals exactly once,
    * mutates exactly one thing,
    * opens with an opener that has not yet seen this unit,
    * and demands a **specific** exception type, not just "something raised".

    Replay is tested on its own, and the suite is self-checking: a positive
    control seals and opens an untouched unit, and a negative control confirms
    that a field really is inside the associated data.
    """
    import dataclasses

    from avsec.crypto import (
        AuthenticationFailed,
        CryptoError,
        NonceExhausted,
        ReplayDetected,
        Sealer,
        Opener,
        derive_session_keys,
        lab_master_secret,
        new_session_id,
    )
    from avsec.fec import FECConfig, RSCodec
    from avsec.framing import (
        FramingError,
        Geometry,
        UnitHeader,
        UnknownProfile,
        UnknownVersion,
        parse_header,
    )

    master = lab_master_secret(cfg.seed)
    payload = b"0123456789" + bytes(54)

    def _fresh(session_id: Optional[bytes] = None, epoch: int = 0,
               master_secret=None, stream_id: int = 0):
        """A sealer and an independent opener over the same key context."""
        sid = session_id or new_session_id()
        keys = derive_session_keys(master_secret or master, sid,
                                   cfg.crypto.direction, stream_id,
                                   cfg.crypto.algorithm, epoch)
        return sid, epoch, Sealer(keys), Opener(keys, cfg.crypto.replay_window)

    def _header(sid: bytes, epoch: int, **over: Any) -> UnitHeader:
        base = dict(profile_id=1, session_id=sid, session_epoch=epoch, stream_id=0,
                    codec_id=1, frame_id=3, stripe_id=2, desc_id=0, seg_id=0,
                    n_segs=2, n_descs=2, unit_seq=0,
                    geometry=Geometry(0, 16, 320, 8), payload_len=10)
        base.update(over)
        return UnitHeader(**base)

    def _seal(sid: bytes, epoch: int, sealer: Sealer, **over: Any):
        """Seal one unit; the AAD is built inside the counter allocation."""
        box: Dict[str, UnitHeader] = {}

        def _aad(counter: int) -> bytes:
            h = _header(sid, epoch, unit_seq=counter, **over)
            box["h"] = h
            return h.core_bytes()

        seq, ct = sealer.seal(payload, _aad)
        return seq, ct, box["h"]

    checks: List[Dict[str, Any]] = []

    def _expect(name: str, fn, outcome: str, exc: Optional[type] = None,
                note: str = "") -> None:
        """``outcome`` is 'accept' or the name of the required exception."""
        try:
            fn()
            ok = outcome == "accept"
            checks.append({"check": name, "expected": outcome, "outcome": "ACCEPTED",
                           "passed": ok, "detail": note})
        except Exception as e:  # noqa: BLE001 - the type is what we are asserting
            ok = exc is not None and isinstance(e, exc)
            checks.append({
                "check": name, "expected": outcome,
                "outcome": f"REJECTED ({type(e).__name__})", "passed": ok,
                "detail": (note + " " if note else "") + str(e)[:110],
            })

    # ---- positive control: an untouched unit must open -------------------
    sid, ep, sealer, opener = _fresh()
    seq, ct, hdr = _seal(sid, ep, sealer)
    _expect("valid unit (positive control)", lambda: opener.open(seq, ct, hdr.core_bytes()),
            "accept", note="must accept, otherwise every rejection below is meaningless")

    # ---- replay, on its own ---------------------------------------------
    sid, ep, sealer, opener = _fresh()
    seq, ct, hdr = _seal(sid, ep, sealer)
    opener.open(seq, ct, hdr.core_bytes())
    _expect("replay of an accepted unit", lambda: opener.open(seq, ct, hdr.core_bytes()),
            "ReplayDetected", ReplayDetected)

    # ---- ciphertext / tag, same nonce, fresh opener ----------------------
    for name, mutate in (
        ("modified ciphertext", lambda c: bytes([c[0] ^ 1]) + c[1:]),
        ("modified tag", lambda c: c[:-1] + bytes([c[-1] ^ 0x80])),
        ("truncated ciphertext", lambda c: c[:-1]),
    ):
        sid, ep, sealer, opener = _fresh()
        seq, ct, hdr = _seal(sid, ep, sealer)
        bad = mutate(ct)
        _expect(name, lambda s=seq, b=bad, h=hdr: opener.open(s, b, h.core_bytes()),
                "AuthenticationFailed", AuthenticationFailed,
                note="same nonce and same AAD as the genuine unit")

    # ---- one authenticated header field at a time ------------------------
    FIELDS = {
        "frame_id": 99, "stripe_id": 9, "desc_id": 1, "seg_id": 1,
        "codec_id": 2, "profile_id": 7, "payload_len": 11, "flags": 0x02,
        "session_epoch": 5, "stream_id": 1, "n_segs": 3, "n_descs": 3,
    }
    for field_name, value in FIELDS.items():
        sid, ep, sealer, opener = _fresh()
        seq, ct, hdr = _seal(sid, ep, sealer)
        forged = dataclasses.replace(hdr, **{field_name: value})
        if forged.core_bytes() == hdr.core_bytes():
            # Changing the field did not change the associated data, so the field
            # is NOT authenticated.  Reporting this as a pass (or skipping it)
            # is precisely the blind spot F07 is about.
            checks.append({
                "check": f"modified authenticated field '{field_name}'",
                "expected": "AuthenticationFailed",
                "outcome": "FIELD NOT COVERED BY THE AAD",
                "passed": False,
                "detail": f"changing {field_name} leaves core_bytes() identical, so "
                          "the tag cannot possibly bind it",
            })
            continue
        _expect(f"modified authenticated field '{field_name}'",
                lambda s=seq, c=ct, f=forged: opener.open(s, c, f.core_bytes()),
                "AuthenticationFailed", AuthenticationFailed,
                note="only this field differs; nonce and ciphertext are untouched")

    # geometry is authenticated too
    sid, ep, sealer, opener = _fresh()
    seq, ct, hdr = _seal(sid, ep, sealer)
    moved = dataclasses.replace(hdr, geometry=Geometry(0, 24, 320, 8))
    _expect("modified authenticated geometry",
            lambda: opener.open(seq, ct, moved.core_bytes()),
            "AuthenticationFailed", AuthenticationFailed)

    # ---- wrong key context ----------------------------------------------
    sid, ep, sealer, _ = _fresh()
    seq, ct, hdr = _seal(sid, ep, sealer)
    _, _, _, other_session = _fresh()
    _expect("unit replayed into another session",
            lambda: other_session.open(seq, ct, hdr.core_bytes()),
            "AuthenticationFailed", AuthenticationFailed)
    _, _, _, other_epoch = _fresh(session_id=sid, epoch=ep + 1)
    _expect("unit replayed into another epoch of the same session",
            lambda: other_epoch.open(seq, ct, hdr.core_bytes()),
            "AuthenticationFailed", AuthenticationFailed)
    _, _, _, other_key = _fresh(session_id=sid, master_secret=lab_master_secret(cfg.seed + 1))
    _expect("wrong master key", lambda: other_key.open(seq, ct, hdr.core_bytes()),
            "AuthenticationFailed", AuthenticationFailed)
    _, _, _, other_stream = _fresh(session_id=sid, stream_id=1)
    _expect("unit replayed into another stream",
            lambda: other_stream.open(seq, ct, hdr.core_bytes()),
            "AuthenticationFailed", AuthenticationFailed)

    # ---- negative control: is the AAD really being used? -----------------
    sid, ep, sealer, opener = _fresh()
    seq, ct, hdr = _seal(sid, ep, sealer)
    _expect("negative control: empty AAD instead of the header",
            lambda: opener.open(seq, ct, b""),
            "AuthenticationFailed", AuthenticationFailed,
            note="if this were accepted, the header would not be authenticated at all")

    # ---- one-shot nonce allocation (F03) ---------------------------------
    _, _, sealer_only, _ = _fresh()
    a, _ = sealer_only.seal(b"x" * 16, b"aad")
    b, _ = sealer_only.seal(b"x" * 16, b"aad")
    checks.append({
        "check": "counters are allocated once and never repeat",
        "expected": "strictly increasing, no caller-chosen counter",
        "outcome": f"{a} then {b}; seal() takes no counter argument",
        "passed": b == a + 1 and "counter" not in Sealer.seal.__code__.co_varnames,
    })
    from avsec.crypto import COUNTER_MAX

    exhausted = Sealer(derive_session_keys(master, new_session_id()),
                       start_counter=COUNTER_MAX + 1)
    _expect("counter beyond the profile limit",
            lambda: exhausted.seal(b"x", b"a"), "NonceExhausted", NonceExhausted)

    # ---- parser bounds ---------------------------------------------------
    sid0 = new_session_id()
    _expect("unknown protocol version",
            lambda: parse_header(dataclasses.replace(_header(sid0, 0), version=7).to_bytes()),
            "UnknownVersion", UnknownVersion)
    _expect("unknown transport profile",
            lambda: parse_header(_header(sid0, 0, profile_id=200).to_bytes(),
                                 accepted_profiles=(1,)),
            "UnknownProfile", UnknownProfile)
    _expect("zero payload length",
            lambda: parse_header(_header(sid0, 0).with_payload_len(0).to_bytes()),
            "FramingError", FramingError)
    _expect("payload length above the profile bound",
            lambda: parse_header(_header(sid0, 0).with_payload_len(9000).to_bytes(),
                                 max_payload_len=1024),
            "FramingError", FramingError)
    _expect("segment index outside the segment count",
            lambda: parse_header(_header(sid0, 0, seg_id=3, n_segs=2).to_bytes()),
            "FramingError", FramingError)
    _expect("corrupted header CRC",
            lambda: parse_header(bytes([_header(sid0, 0).to_bytes()[0] ^ 0xFF])
                                 + _header(sid0, 0).to_bytes()[1:]),
            "FramingError", FramingError)
    _expect("geometry outside the frame",
            lambda: Geometry(0, 236, 160, 6, 2, 2, 1, 1).validate(320, 240),
            "FramingError", FramingError)
    _expect("geometry phase inconsistent with the step",
            lambda: Geometry(0, 0, 64, 8, step_x=2, phase_x=2).validate(320, 240),
            "FramingError", FramingError)

    # ---- FEC versus forgery ---------------------------------------------
    sid, ep, sealer, opener = _fresh()
    seq, ct, hdr = _seal(sid, ep, sealer)
    rs = RSCodec(FECConfig(k=191, nsym=64))
    enc = bytearray(rs.encode(ct))
    for i in range(10):
        enc[i] ^= 0xFF
    dec, _ = rs.try_decode(bytes(enc), len(ct))
    checks.append({
        "check": "modification fully repaired by FEC",
        "expected": "not a forgery: the identical protected message is restored",
        "outcome": "restored" if dec == ct else "not restored",
        "passed": dec == ct,
        "detail": "a corrected error is not evidence of a broken tag",
    })
    enc2 = bytearray(rs.encode(ct))
    for i in range(0, 200, 2):
        if i < len(enc2):
            enc2[i] ^= 0xA5
    dec2, _ = rs.try_decode(bytes(enc2), len(ct))
    detail = "uncorrectable, discarded by FEC"
    ok = True
    if dec2 is not None and dec2 != ct:
        try:
            opener.open(seq, dec2, hdr.core_bytes())
            ok, detail = False, "AEAD ACCEPTED a miscorrected message"
        except AuthenticationFailed:
            detail = "miscorrected by FEC, rejected by AEAD"
        except ReplayDetected:
            detail = "miscorrected by FEC, rejected as replay"
    elif dec2 == ct:
        detail = "still repaired by FEC"
    checks.append({
        "check": "modification beyond the FEC capability",
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
            trace = _trace(cfg, s.name, 0)
            for fi, frame in enumerate(s.frames[: cfg.max_frames_per_source]):
                try:
                    res = m.process(frame, fi, trace)
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
                trace = _trace(sub, s.name, 0)
                for fi, frame in enumerate(s.frames[: cfg.max_frames_per_source]):
                    try:
                        res = m.process(frame, fi, trace)
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
