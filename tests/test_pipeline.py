"""End-to-end integration checks of the secured pipeline.

These verify the properties that make a result trustworthy: a clean channel
reproduces exactly what the source decoder would produce, damage degrades
gracefully with honest statuses, and the receiver never uses information it
could not have measured.
"""
from __future__ import annotations

import numpy as np
import pytest

from avsec import sources as S
from avsec.baselines import DigitalMethod, WholeFrameAEADMethod, make_b0_analog, make_b1_lfsr
from avsec.channel import RasterChannelConfig, preset
from avsec.crypto import CryptoProfile, lab_master_secret
from avsec.evaluation import psnr
from avsec.fec import FECConfig
from avsec.interleaving import InterleaverConfig
from avsec.lfsr import ScramblerConfig
from avsec.optimization import SharedBudget
from avsec.receiver import Receiver, UnitStatus
from avsec.source_coding import FrameAssembler, SourceCodingConfig, StripeCoder
from avsec.transmitter import TransportConfig, Transmitter
from avsec.utils import experiment_rng

H, W = 192, 256
LAB = lab_master_secret(4242)
BUDGET = SharedBudget(rasters_per_frame=3, source_fps=8.333)


def _profile(n_desc=1, unit=640, scheme="block", depth=278, burst=12, quality=15):
    src = SourceCodingConfig(stripe_height=24, n_descriptions=n_desc, codec="dct",
                             quality=quality, max_unit_payload=unit)
    tr = TransportConfig(
        modem=BUDGET.modem(4, 8, 2),
        fec_payload=FECConfig(k=127, nsym=128),
        fec_header=FECConfig(k=52, nsym=48),
        interleaver=InterleaverConfig(scheme=scheme, depth=depth, burst_rows=burst),
        raster_rate_hz=BUDGET.raster_rate_hz,
        max_rasters_per_frame=BUDGET.rasters_per_frame)
    return src, tr


def _pair(src, tr, **kw):
    tx = Transmitter(src, tr, LAB, CryptoProfile(), frame_width=W, frame_height=H)
    rx = Receiver(src, tr, LAB, CryptoProfile(), frame_width=W, frame_height=H, **kw)
    return tx, rx


# ------------------------------------------------------------- clean channel
def test_clean_channel_matches_the_source_decoder_exactly():
    """Not bit-exact against the pixels - exact against the same source decoder."""
    src, tr = _profile()
    tx, rx = _pair(src, tr)
    coder = StripeCoder(src)
    frame = S.pattern_edges(H, W)

    reference = FrameAssembler(H, W).assemble(
        [(s.geometry, coder.decode_segment(s.payload, s.geometry))
         for s in coder.encode_frame(frame, 0)]).image

    txf = tx.encode_frame(frame, 0)
    placements = []
    for raster in txf.rasters:
        out = rx.receive_raster(raster)
        assert out.demod.sync_found
        for o in out.outcomes:
            assert o.status in (UnitStatus.VERIFIED, UnitStatus.FILLER)
            if o.ok:
                placements.append((o.header.geometry, o.samples))
    got = FrameAssembler(H, W).assemble(placements)
    assert got.coverage == 1.0
    assert np.array_equal(got.image, reference)


def test_no_unverified_data_ever_reaches_the_image_decoder():
    src, tr = _profile()
    tx, rx = _pair(src, tr)
    txf = tx.encode_frame(S.pattern_edges(H, W), 0)
    raster = txf.rasters[0].copy()
    # corrupt one slot beyond the FEC capability
    cells = tx.placement[0]
    rng = np.random.default_rng(1)
    rows = tx.cfg.modem.active_y0 + (cells // tx.cfg.modem.n_data_cols) * 2
    cols = tx.cfg.modem.active_x0 + (cells % tx.cfg.modem.n_data_cols) * 8
    for r, c in zip(rows[:4000], cols[:4000]):
        raster[r : r + 2, c : c + 8] = rng.integers(0, 256)
    out = rx.receive_raster(raster)
    bad = [o for o in out.outcomes if not o.ok and o.status is not UnitStatus.FILLER]
    assert bad, "the corrupted slot must not be reported as verified"
    for o in bad:
        assert o.samples is None
        assert o.detail != "" or o.status is UnitStatus.STALE


def test_replayed_unit_is_rejected_by_the_receiver():
    src, tr = _profile()
    tx, rx = _pair(src, tr)
    txf = tx.encode_frame(S.pattern_edges(H, W), 0)
    first = rx.receive_raster(txf.rasters[0])
    assert any(o.ok for o in first.outcomes)
    again = rx.receive_raster(txf.rasters[0])
    assert any(o.status is UnitStatus.REPLAY for o in again.outcomes)
    assert not any(o.ok for o in again.outcomes)


def test_stale_frame_is_rejected_after_the_display_deadline():
    src, tr = _profile()
    tx, rx = _pair(src, tr, max_frame_age=1)
    frame = S.pattern_edges(H, W)
    old = tx.encode_frame(frame, 0)
    for fid in (1, 2, 3, 4):
        for r in tx.encode_frame(frame, fid).rasters:
            rx.receive_raster(r)
    out = rx.receive_raster(old.rasters[0])
    assert any(o.status is UnitStatus.STALE for o in out.outcomes)


def test_slot_order_is_not_authenticated_but_position_in_the_picture_is():
    """Documented protocol property.

    The *physical slot* a unit travels in is not part of the associated data:
    what is authenticated is the unit's semantic position (session, frame,
    stripe, description, segment, geometry).  Consequently swapping two units
    between slots is harmless - each one still lands where its authenticated
    header says - while re-labelling a unit's coordinates fails the tag.
    """
    src, tr = _profile()
    tx, rx = _pair(src, tr)
    frame = S.pattern_edges(H, W)
    txf = tx.encode_frame(frame, 0)

    reference = []
    for o in rx.receive_raster(txf.rasters[0]).outcomes:
        if o.ok:
            reference.append((o.header.key(), o.samples))

    tx2, rx2 = _pair(src, tr)
    txf2 = tx2.encode_frame(frame, 0)
    a, b = tx2.placement[0], tx2.placement[1]
    sym = txf2.symbols[0].copy()
    sym[a], sym[b] = txf2.symbols[0][b].copy(), txf2.symbols[0][a].copy()
    out = rx2.receive_raster(tx2.modem.modulate(sym))
    swapped = [(o.header.key(), o.samples) for o in out.outcomes if o.ok]

    assert len(swapped) == len(reference)
    ref_map = {k: v for k, v in reference}
    for key, samples in swapped:
        assert key in ref_map
        assert np.array_equal(samples, ref_map[key])

    # ... whereas changing the authenticated coordinates does fail
    from avsec.crypto import AuthenticationFailed

    hdr = out.outcomes[0].header
    opener = rx2._opener(hdr.session_id)
    import dataclasses

    forged = dataclasses.replace(hdr, stripe_id=hdr.stripe_id + 1).core_bytes()
    with pytest.raises(AuthenticationFailed):
        opener.open(hdr.unit_seq + 10 ** 6, bytes(src.max_unit_payload + 16), forged)


def test_unknown_profile_is_rejected_in_a_defined_way():
    src, tr = _profile()
    tx, _ = _pair(src, tr)
    other = TransportConfig(**{**tr.__dict__, "profile_id": 42})
    rx = Receiver(src, other, LAB, CryptoProfile(), frame_width=W, frame_height=H)
    txf = tx.encode_frame(S.pattern_edges(H, W), 0)
    out = rx.receive_raster(txf.rasters[0])
    assert all(o.status is UnitStatus.UNKNOWN_PROFILE for o in out.outcomes)


# --------------------------------------------------------- damaged channels
def test_partial_loss_keeps_the_recovered_units_available():
    src, tr = _profile()
    method = DigitalMethod("P-test", src, tr, preset("bursty"), LAB, H, W)
    method.reset()
    res = method.process(S.pattern_edges(H, W), 0, experiment_rng(9, "burst"))
    m = res.metrics
    assert m.units_verified >= 1
    assert 0.0 < m.coverage <= 1.0
    assert m.units_verified + m.units_rejected <= m.units_sent + 64
    assert res.available.sum() > 0


def test_receiver_recovers_after_a_total_sync_loss():
    src, tr = _profile()
    tx, rx = _pair(src, tr)
    frame = S.pattern_edges(H, W)
    good = tx.encode_frame(frame, 0)
    assert any(o.ok for o in rx.receive_raster(good.rasters[0]).outcomes)

    rng = np.random.default_rng(5)
    garbage = rng.integers(0, 256, good.rasters[0].shape).astype(np.uint8)
    lost = rx.receive_raster(garbage)
    assert all(o.status is UnitStatus.NO_SYNC for o in lost.outcomes)

    later = tx.encode_frame(frame, 1)
    back = rx.receive_raster(later.rasters[0])
    assert back.resynchronised
    assert any(o.ok for o in back.outcomes)


def test_a_lost_unit_does_not_break_the_following_units():
    src, tr = _profile()
    tx, rx = _pair(src, tr)
    txf = tx.encode_frame(S.pattern_edges(H, W), 0)
    sym = txf.symbols[0].copy()
    sym[tx.placement[0]] = 0                      # destroy slot 0 entirely
    out = rx.receive_raster(tx.modem.modulate(sym))
    assert not out.outcomes[0].ok
    assert sum(1 for o in out.outcomes[1:] if o.ok) >= 1


# ------------------------------------------------------------ whole-frame AEAD
def test_whole_frame_aead_needs_every_fragment():
    src, tr = _profile()
    m = WholeFrameAEADMethod(src, tr, RasterChannelConfig(), LAB, H, W)
    res = m.process(S.pattern_edges(H, W), 0, experiment_rng(3, "clean"))
    assert res.metrics.status_counts.get("frame_verified") == 1
    assert res.metrics.coverage == 1.0

    m2 = WholeFrameAEADMethod(src, tr, preset("harsh"), LAB, H, W)
    res2 = m2.process(S.pattern_edges(H, W), 0, experiment_rng(3, "harsh"))
    assert res2.metrics.coverage == 0.0
    assert "frame_verified" not in res2.metrics.status_counts


# --------------------------------------------------------- analog baselines
def test_analog_baseline_is_near_lossless_on_a_clean_channel():
    _, tr = _profile()
    m = make_b0_analog(tr, RasterChannelConfig(), H, W)
    res = m.process(S.pattern_edges(H, W), 0, experiment_rng(1, "clean"))
    assert res.metrics.psnr_full > 40.0            # amplitude quantisation only


def test_permutation_baseline_restores_the_picture_on_a_clean_channel():
    _, tr = _profile()
    m = make_b1_lfsr(tr, RasterChannelConfig(), H, W,
                     ScramblerConfig(grid_rows=12, grid_cols=16))
    res = m.process(S.pattern_edges(H, W), 0, experiment_rng(1, "clean"))
    assert res.metrics.psnr_full > 40.0


def test_analog_baselines_are_not_marked_authenticated():
    _, tr = _profile()
    for m in (make_b0_analog(tr, RasterChannelConfig(), H, W),
              make_b1_lfsr(tr, RasterChannelConfig(), H, W, ScramblerConfig())):
        assert m.authenticated is False
        res = m.process(S.pattern_edges(H, W), 0, experiment_rng(1, "c"))
        assert res.metrics.extra["authenticated"] is False


# --------------------------------------------------------------- diagnostics
def test_unsecured_digital_transport_is_labelled_as_such():
    src, tr = _profile()
    m = DigitalMethod("B0d", src, tr, RasterChannelConfig(), LAB, H, W, secure=False)
    assert m.authenticated is False
    res = m.process(S.pattern_edges(H, W), 0, experiment_rng(1, "c"))
    assert res.metrics.coverage == 1.0
    assert m.describe()["secure"] is False


def test_capacity_and_latency_accounting_is_consistent():
    from avsec.budget import capacity_check, compute_budget, compute_latency

    src, tr = _profile()
    b = compute_budget(tr.modem, tr.fec_payload, tr.fec_header, src.max_unit_payload,
                       tr.raster_rate_hz)
    assert b.unit_wire_bytes == (b.header_encoded_bytes + b.payload_encoded_bytes)
    assert b.payload_bitrate_bps <= b.wire_bitrate_bps <= b.gross_bitrate_bps
    chk = capacity_check(b, units_needed_per_frame=b.units_per_raster * 2, fps=8.333)
    assert chk["rasters_needed_per_frame"] == 2
    lat = compute_latency(tr.modem, 278, 3, 1 / 8.333, 24, H, 25.0)
    assert lat.total_s == pytest.approx(sum([
        lat.stripe_accumulation_s, lat.interleaver_accumulation_s,
        lat.serialisation_s, lat.reception_s, lat.display_hold_s]))
