"""One regression per fixed defect, each reproducing the original failure.

Every test here fails against the code as of commit 22a831f and passes after the
corresponding fix.  The defect identifiers follow the review document.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from avsec import sources as S
from avsec.channel import ChannelTrace, RasterChannel, RasterChannelConfig, preset
from avsec.crypto import CryptoProfile, Sealer, derive_session_keys, lab_master_secret
from avsec.fec import FECConfig
from avsec.framing import Geometry, UnitHeader
from avsec.interleaving import (
    Interleaver,
    InterleaverConfig,
    PlacementError,
    PlacementInfo,
    bawp_placement,
    burst_rows_from_lines,
    worst_case_codeword_damage,
)
from avsec.optimization import (
    SharedBudget,
    build_candidate,
    effective_fingerprint,
    placement_guarantee,
)
from avsec.receiver import Receiver, UnitStatus
from avsec.source_coding import IdentityMismatch, SourceCodingConfig
from avsec.transmitter import TransportConfig, Transmitter
from avsec.utils import bytes_to_symbols

LAB = lab_master_secret(1)
H, W = 192, 256
BUDGET = SharedBudget()


def _pipeline(n_desc=1, unit=640, scheme="block", depth=278, burst=8, nsym=96,
              channel=None, **rx_kw):
    src = SourceCodingConfig(stripe_height=24, n_descriptions=n_desc, codec="dct",
                             quality=12, max_unit_payload=unit)
    tr = TransportConfig(
        modem=BUDGET.modem(4, 8, 2),
        fec_payload=FECConfig(k=255 - nsym, nsym=nsym),
        fec_header=FECConfig(k=52, nsym=48),
        interleaver=InterleaverConfig(scheme=scheme, depth=depth, burst_rows=burst),
        max_rasters_per_frame=BUDGET.rasters_per_frame)
    tx = Transmitter(src, tr, LAB, CryptoProfile(), frame_width=W, frame_height=H)
    rx = Receiver(src, tr, LAB, CryptoProfile(), frame_width=W, frame_height=H, **rx_kw)
    return src, tr, tx, rx


def _forged_raster(tx, rx, tr, session_id: bytes, epoch: int = 0) -> np.ndarray:
    """A raster whose slot 0 carries a well-formed but unauthentic unit."""
    hdr = UnitHeader(profile_id=1, session_id=session_id, session_epoch=epoch,
                     stream_id=0, codec_id=1, frame_id=0, stripe_id=0, desc_id=0,
                     seg_id=0, n_segs=1, n_descs=1, unit_seq=0,
                     geometry=Geometry(0, 0, 64, 8), payload_len=10)
    wire = rx.rs_header.encode(hdr.to_bytes()) + rx.rs_payload.encode(
        bytes(tx.source_cfg.max_unit_payload + 16))
    sym = np.zeros(tr.modem.capacity_symbols, dtype=np.uint8)
    s = bytes_to_symbols(wire, tr.modem.bits_per_symbol)
    cells = tx.placement[0]
    sym[cells[: s.size]] = s[: cells.size]
    return tx.modem.modulate(sym)


# ------------------------------------------------------------------------ F01
def test_f01_forged_headers_cannot_evict_a_session_and_reopen_a_replay():
    """Unauthenticated headers must not touch the session cache.

    Original defect: ``_opener()`` inserted and LRU-evicted before the AEAD
    check, so N+1 forged session ids flushed the genuine session and its replay
    history, after which a replayed genuine raster was accepted again.
    """
    src, tr, tx, rx = _pipeline(rx_kw_placeholder := None) if False else _pipeline()
    rx.max_sessions = 4
    raster = tx.encode_frame(S.pattern_edges(H, W), 0).rasters[0]

    assert rx.receive_raster(raster).counts().get("verified", 0) > 0
    assert rx.receive_raster(raster).counts().get("replay", 0) > 0

    for i in range(rx.max_sessions * 3):
        rx.receive_raster(_forged_raster(tx, rx, tr, bytes([i + 1]) * 8))

    assert len(rx._openers) <= rx.max_sessions
    assert any(ctx[0] == tx.session_id for ctx in rx._openers), \
        "the genuine session was evicted by unauthenticated traffic"
    counts = rx.receive_raster(raster).counts()
    assert counts.get("verified", 0) == 0
    assert counts.get("replay", 0) > 0


def test_f01_unrecoverable_units_do_not_disturb_state():
    _, tr, tx, rx = _pipeline()
    raster = tx.encode_frame(S.pattern_edges(H, W), 0).rasters[0]
    rx.receive_raster(raster)
    before_sessions = dict(rx._openers)
    before_frames = dict(rx._newest_frame)

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, raster.shape).astype(np.uint8)
    rx.receive_raster(noise)
    assert dict(rx._openers) == before_sessions
    assert dict(rx._newest_frame) == before_frames


# ------------------------------------------------------------------------ F02
def test_f02_a_new_session_restarts_frame_numbering():
    """Frame 0 of a newly agreed session is accepted after frame 10 of the old one."""
    _, _, tx, rx = _pipeline(rx_kw_placeholder := None) if False else _pipeline()
    frame = S.pattern_edges(H, W)
    for fid in range(11):
        for r in tx.encode_frame(frame, fid).rasters:
            rx.receive_raster(r)

    tx.new_session()
    counts = {}
    for r in tx.encode_frame(frame, 0).rasters:
        for k, v in rx.receive_raster(r).counts().items():
            counts[k] = counts.get(k, 0) + v
    assert counts.get("verified", 0) > 0, counts
    assert counts.get("stale", 0) == 0


def test_f02_a_retired_epoch_cannot_be_reopened():
    _, _, tx, rx = _pipeline()
    frame = S.pattern_edges(H, W)
    old = tx.encode_frame(frame, 0)
    for r in old.rasters:
        rx.receive_raster(r)

    tx.new_epoch()
    for r in tx.encode_frame(frame, 0).rasters:
        assert rx.receive_raster(r).counts().get("verified", 0) > 0

    counts = {}
    for r in old.rasters:                      # a recording of the retired epoch
        for k, v in rx.receive_raster(r).counts().items():
            counts[k] = counts.get(k, 0) + v
    assert counts.get("verified", 0) == 0
    assert counts.get("stale_epoch", 0) + counts.get("replay", 0) > 0


# ------------------------------------------------------------------------ F03
def test_f03_a_failed_build_still_consumes_the_counter():
    keys = derive_session_keys(LAB, b"\x01" * 8)
    sealer = Sealer(keys)
    first, _ = sealer.seal(b"x" * 8, b"aad")

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        sealer.seal(b"x" * 8, lambda c: (_ for _ in ()).throw(Boom()))
    second, _ = sealer.seal(b"x" * 8, b"aad")
    assert second > first + 1, "the counter of the failed attempt was handed out again"


# ------------------------------------------------------------------------ F04
def test_f04_frames_cannot_be_merged_into_one_picture():
    """A late frame 0 must not be counted as coverage of frame 1."""
    from avsec.source_coding import FrameAssembler
    from avsec.receiver import VerifiedUnit

    asm = FrameAssembler(H, W)
    a = VerifiedUnit(b"\x01" * 8, 0, 0, 0, 0, 0, 0, Geometry(0, 0, W, 8),
                     np.zeros((8, W), np.uint8), 0.0)
    b = dataclasses.replace(a, frame_id=1, geometry=Geometry(0, 8, W, 8))
    with pytest.raises(IdentityMismatch):
        asm.assemble_verified([a, b])
    out = asm.assemble_verified([a])
    assert out.identity == a.identity and out.n_units_used == 1


def test_f04_only_the_current_frame_counts_as_coverage():
    from avsec.baselines import DigitalMethod

    src = SourceCodingConfig(stripe_height=24, n_descriptions=1, codec="dct",
                             quality=12, max_unit_payload=640)
    tr = TransportConfig(modem=BUDGET.modem(4, 8, 2),
                         fec_payload=FECConfig(k=159, nsym=96),
                         fec_header=FECConfig(k=52, nsym=48),
                         interleaver=InterleaverConfig(scheme="block", depth=278),
                         max_rasters_per_frame=BUDGET.rasters_per_frame)
    m = DigitalMethod("t", src, tr, RasterChannelConfig(), LAB, H, W)
    m.reset()
    res = m.process(S.pattern_edges(H, W), 7, ChannelTrace(1, "s", 0, "clean", 3))
    assert res.metrics.extra["units_from_other_frames"] == 0
    assert res.metrics.coverage == 1.0


# ------------------------------------------------------------------------ F05
def test_f05_b3_receiver_rejects_a_tampered_frame_id_with_valid_crc():
    """The B3 receiver must not take length, counter or AAD from the transmitter."""
    from avsec.baselines import B3Receiver, B3Transmitter, _B3Fragment

    src = SourceCodingConfig(stripe_height=24, n_descriptions=1, codec="dct",
                             quality=12, max_unit_payload=640)
    tr = TransportConfig(modem=BUDGET.modem(4, 8, 2),
                         fec_payload=FECConfig(k=159, nsym=96),
                         fec_header=FECConfig(k=52, nsym=48),
                         interleaver=InterleaverConfig(scheme="block", depth=278),
                         max_rasters_per_frame=BUDGET.rasters_per_frame)
    tx = B3Transmitter(src, tr, LAB, H, W)
    rx = B3Receiver(src, tr, LAB, H, W)
    rasters, _, _, _, _ = tx.encode_frame(S.pattern_edges(H, W), 0)

    clean = []
    for r in rasters:
        clean.extend(rx.receive_raster(r))
    assert rx.offer(clean), "the untouched frame must verify"

    rx2 = B3Receiver(src, tr, LAB, H, W)
    tampered = []
    for r in rasters:
        for f in rx2.receive_raster(r):
            # a well-formed header with a recomputed CRC, but a different frame
            tampered.append(_B3Fragment(dataclasses.replace(f.header, frame_id=99),
                                        f.payload))
    assert rx2.offer(tampered) == [], "a tampered frame_id was accepted"
    assert rx2.status_counts.get("auth_failed", 0) > 0


def test_f05_b3_receiver_holds_no_transmitter_state():
    import inspect
    import re

    from avsec.baselines import B3Receiver

    text = inspect.getsource(B3Receiver)
    for leak in ("occupied", "ref_syms", "coded_len", "n_frags_tx"):
        assert not re.search(rf"\b{leak}\b", text), \
            f"B3Receiver still refers to transmitter state {leak!r}"
    # the ciphertext length must come from the received header
    assert "h.payload_len + TAG_LEN" in text


# ------------------------------------------------------------------------ F06
def test_f06_dropped_rasters_produce_no_coverage():
    """With every raster dropped, current coverage must be zero, not one."""
    from avsec.baselines import DigitalMethod

    cfg = RasterChannelConfig(frame_drop_prob=1.0)
    src = SourceCodingConfig(stripe_height=24, n_descriptions=1, codec="dct",
                             quality=12, max_unit_payload=640)
    tr = TransportConfig(modem=BUDGET.modem(4, 8, 2),
                         fec_payload=FECConfig(k=159, nsym=96),
                         fec_header=FECConfig(k=52, nsym=48),
                         interleaver=InterleaverConfig(scheme="block", depth=278),
                         max_rasters_per_frame=BUDGET.rasters_per_frame)
    m = DigitalMethod("t", src, tr, cfg, LAB, H, W)
    m.reset()
    res = m.process(S.pattern_edges(H, W), 0, ChannelTrace(3, "s", 0, "drop", 3))
    assert res.metrics.coverage == 0.0
    assert res.metrics.units_verified == 0
    assert res.metrics.extra["rasters_delivered"] == 0


def test_f06_duplicates_do_not_create_new_units():
    from avsec.baselines import DigitalMethod

    src = SourceCodingConfig(stripe_height=24, n_descriptions=1, codec="dct",
                             quality=12, max_unit_payload=640)
    tr = TransportConfig(modem=BUDGET.modem(4, 8, 2),
                         fec_payload=FECConfig(k=159, nsym=96),
                         fec_header=FECConfig(k=52, nsym=48),
                         interleaver=InterleaverConfig(scheme="block", depth=278),
                         max_rasters_per_frame=BUDGET.rasters_per_frame)
    base = DigitalMethod("a", src, tr, RasterChannelConfig(), LAB, H, W)
    base.reset()
    ref = base.process(S.pattern_edges(H, W), 0, ChannelTrace(4, "s", 0, "clean", 3))

    dup = DigitalMethod("b", src, tr, RasterChannelConfig(frame_duplicate_prob=1.0),
                        LAB, H, W)
    dup.reset()
    got = dup.process(S.pattern_edges(H, W), 0, ChannelTrace(4, "s", 0, "dup", 3))
    assert got.metrics.units_verified == ref.metrics.units_verified
    assert got.metrics.status_counts.get("replay", 0) > 0


def test_f06_channel_time_is_not_reset_every_frame():
    ch = RasterChannel(RasterChannelConfig(frame_duplicate_prob=1.0))
    ch.reset_stream()
    rng = np.random.default_rng(0)
    img = np.zeros((16, 16), np.uint8)
    ch.apply_stream(img, rng)
    assert ch._raster_index == 1
    ch.apply_stream(img, rng)
    assert ch._raster_index == 2, "the stream clock restarted"


# ------------------------------------------------------------------------ F08
def test_f08_every_method_meets_the_same_channel_realisation():
    """The trace must depend on time and scene, never on the method."""
    tr = ChannelTrace(seed=11, scene="edges", repetition=2, profile="bursty",
                      rasters_per_frame=3)
    a = tr.rng(1, 0).normal(size=8)
    b = tr.rng(1, 0).normal(size=8)
    assert np.array_equal(a, b)

    ch = RasterChannel(preset("bursty"))
    img = np.full((64, 64), 128, np.uint8)
    r1, t1 = ch.apply(img, tr.rng(1, 0))
    r2, t2 = ch.apply(img, tr.rng(1, 0))
    assert np.array_equal(r1, r2)
    assert np.array_equal(t1.damaged_lines, t2.damaged_lines)
    assert not np.array_equal(a, tr.rng(1, 1).normal(size=8))


def test_f08_method_name_is_absent_from_the_channel_seed():
    import inspect

    from avsec import experiments

    src = inspect.getsource(experiments.run_comparison)
    assert 'experiment_rng(cfg.seed, "cmp", name' not in src
    assert "_trace(cfg, src.name, rep)" in src


# ------------------------------------------------------------------ F11 / F12
def test_f11_guarantee_is_enforced_not_discarded():
    import inspect

    from avsec import optimization

    body = inspect.getsource(optimization.static_admissibility)
    assert "pass" not in body.split("guarantee")[0].splitlines()[-1:], "leftover no-op"
    cand = build_candidate(BUDGET, 24, 1, "dct", 12, 640, 32, ("sequential", 1, 0),
                           (4, 8, 2))
    ok, why = placement_guarantee(cand, 1, burst_lines=64)
    assert not ok and "burst" in why


def test_f11_units_are_converted_not_mixed():
    assert burst_rows_from_lines(16, 2) == 9      # 16 lines / 2 lines per row, +1
    assert burst_rows_from_lines(1, 8) == 1
    cells = np.arange(0, 40) * 7 % 400
    d = worst_case_codeword_damage(cells, 20, 3, 20)
    assert 0 < d <= cells.size


def test_f12_description_isolation_counterexample():
    """The reviewed counterexample must be detected, not silently mis-claimed."""
    info = PlacementInfo()
    pl = bawp_placement(16, 16, 16, 4, 4, [64, 2, 1, 1, 1], [0, 0, 1, 2, 3], info=info)
    touched = {d for d, cells in zip([0, 0, 1, 2, 3], pl)
               if set(range(5, 9)) & set((cells // 16).tolist())}
    assert len(touched) == 3                      # more than the naive bound of two
    assert info.spilled and not info.isolation_guaranteed

    with pytest.raises(PlacementError):
        bawp_placement(16, 16, 16, 4, 4, [64, 2, 1, 1, 1], [0, 0, 1, 2, 3],
                       strict_isolation=True)


def test_f12_strict_isolation_keeps_one_class_per_band():
    info = PlacementInfo()
    n_cols, D, B = 60, 4, 4
    lens, descs = [40] * 8, [i % D for i in range(8)]
    pl = bawp_placement(120, n_cols, 64, D, B, lens, descs,
                        strict_isolation=True, info=info)
    assert not info.spilled and info.isolation_guaranteed
    worst = 0
    for s in range(120 - B + 1):
        hit = {descs[u] for u, cells in enumerate(pl)
               if ((cells // n_cols >= s) & (cells // n_cols < s + B)).any()}
        worst = max(worst, len(hit))
    assert worst <= 2


# ------------------------------------------------------------------------ F13
def test_f13_equivalent_configurations_collapse_to_one():
    fps = {b: effective_fingerprint(
        build_candidate(BUDGET, 24, 2, "dct", 8, 320, 128, ("bawp", 0, b), (4, 8, 2)))
        for b in (8, 12, 24)}
    assert len(set(fps.values())) == 1, fps
    other = effective_fingerprint(
        build_candidate(BUDGET, 24, 1, "dct", 8, 320, 128, ("block", 278, 0), (4, 8, 2)))
    assert other not in set(fps.values())
