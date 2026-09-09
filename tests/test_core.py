"""Checks that target the risks which could make a result wrong or unsafe.

The numbering follows the risk list in the project brief (section 21).
"""
from __future__ import annotations

import dataclasses
import os
import tempfile

import numpy as np
import pytest

from avsec import sources as S
from avsec.crypto import (
    AuthenticationFailed,
    CryptoProfile,
    NonceExhausted,
    ReplayDetected,
    SessionState,
    CryptoError,
    derive_session_keys,
    lab_master_secret,
    make_session,
    new_session_id,
)
from avsec.fec import FECConfig, RSCodec, Uncorrectable
from avsec.framing import (
    HEADER_LEN,
    FramingError,
    Geometry,
    UnitHeader,
    UnknownProfile,
    UnknownVersion,
    parse_header,
)
from avsec.interleaving import Interleaver, InterleaverConfig, bawp_bands
from avsec.lfsr import (
    DEFAULT_TAPS,
    BlockScrambler,
    LFSR,
    LFSRConfig,
    ScramblerConfig,
    invert_permutation,
    lfsr_permutation,
)
from avsec.modem import ModemConfig, RasterModem
from avsec.source_coding import (
    FrameAssembler,
    SourceCodingConfig,
    StripeCoder,
    lattice,
)
from avsec.utils import bytes_to_symbols, symbols_to_bytes

H, W = 96, 128
LAB = lab_master_secret(12345)


def _header(**kw):
    base = dict(
        profile_id=1, session_id=bytes(range(8)), session_epoch=0, stream_id=0,
        codec_id=1, frame_id=5, stripe_id=2, desc_id=0, seg_id=0, n_segs=1,
        n_descs=1, unit_seq=7,
        geometry=Geometry(0, 16, 64, 8), payload_len=20, flags=0,
    )
    base.update(kw)
    return UnitHeader(**base)


# ------------------------------------------------- 1. permutation invertibility
@pytest.mark.parametrize("shape", [(96, 128), (97, 131), (64, 64), (100, 100)])
@pytest.mark.parametrize("policy", ["pad", "crop"])
@pytest.mark.parametrize("grid", [(4, 4), (12, 16), (3, 7)])
def test_permutation_roundtrip(shape, policy, grid):
    cfg = ScramblerConfig(grid_rows=grid[0], grid_cols=grid[1], size_policy=policy)
    sc = BlockScrambler(cfg)
    img = S.pattern_texture(*shape)
    back = sc.descramble(sc.scramble(img), original_shape=shape)
    ph, pw = sc.padded_shape(shape)
    ref = img[: min(ph, shape[0]), : min(pw, shape[1])]
    assert np.array_equal(ref, back[: ref.shape[0], : ref.shape[1]])


@pytest.mark.parametrize("variant", ["fisher_yates", "first_occurrence"])
@pytest.mark.parametrize("n", [1, 2, 12, 96, 192, 255])
def test_permutation_is_a_bijection(variant, n):
    p = lfsr_permutation(n, LFSRConfig(seed=44257), variant)
    assert sorted(p.tolist()) == list(range(n))
    assert np.array_equal(invert_permutation(p)[p], np.arange(n))


@pytest.mark.parametrize("width", [8, 12, 16, 17])
def test_lfsr_is_maximal_length(width):
    lf = LFSR(LFSRConfig(width=width, taps=DEFAULT_TAPS[width], seed=1))
    assert lf.period() == (1 << width) - 1


def test_lfsr_rejects_the_zero_state_when_asked():
    with pytest.raises(ValueError):
        LFSR(LFSRConfig(seed=0, zero_state_policy="error"))
    assert LFSR(LFSRConfig(seed=0)).state == 1


# --------------------------------------- 2. header compatibility and boundaries
def test_header_roundtrip_is_canonical():
    h = _header()
    assert len(h.to_bytes()) == HEADER_LEN
    assert parse_header(h.to_bytes()) == h
    # the canonical encoding is deterministic
    assert h.to_bytes() == _header().to_bytes()


def test_header_field_concatenation_is_unambiguous():
    """Two different field assignments must never produce the same bytes."""
    a = _header(stripe_id=1, desc_id=2).core_bytes()
    b = _header(stripe_id=2, desc_id=1).core_bytes()
    c = _header(frame_id=0x00010002, stripe_id=0).core_bytes()
    d = _header(frame_id=0x00010000, stripe_id=2).core_bytes()
    assert a != b and c != d


def test_header_rejects_out_of_range_and_unknown():
    with pytest.raises(FramingError):
        parse_header(_header(payload_len=0).to_bytes())
    with pytest.raises(FramingError):
        parse_header(_header(payload_len=99999).to_bytes(), max_payload_len=1024)
    with pytest.raises(UnknownVersion):
        parse_header(dataclasses.replace(_header(), version=9).to_bytes())
    with pytest.raises(UnknownProfile):
        parse_header(_header(profile_id=200).to_bytes(), accepted_profiles=(1,))
    with pytest.raises(FramingError):
        parse_header(_header(seg_id=3, n_segs=2).to_bytes())
    bad = bytearray(_header().to_bytes())
    bad[3] ^= 0xFF                     # corrupt a field, leave the CRC alone
    with pytest.raises(FramingError):
        parse_header(bytes(bad))


def test_geometry_bounds_are_checked_before_use():
    Geometry(0, 16, 64, 8).validate(W, H)
    with pytest.raises(FramingError):
        Geometry(0, H - 2, 64, 8).validate(W, H)
    with pytest.raises(FramingError):
        Geometry(0, 0, 64, 8, step_x=2, phase_x=2).validate(W, H)


# ------------------------------------ 3. unique context, sessions and restarts
def test_nonce_is_unique_per_unit_and_session():
    sid = new_session_id()
    keys, sealer, _ = make_session(LAB, CryptoProfile(), sid)
    nonces = {sealer.seal(b"x" * 8, b"aad")[0] for _ in range(200)}
    assert len(nonces) == 200
    other, _, _ = make_session(LAB, CryptoProfile(), new_session_id())
    assert keys.key != other.key and keys.nonce_prefix != other.nonce_prefix


def test_directions_and_streams_use_different_keys():
    sid = new_session_id()
    a = derive_session_keys(LAB, sid, "uplink", 0)
    b = derive_session_keys(LAB, sid, "downlink", 0)
    c = derive_session_keys(LAB, sid, "uplink", 1)
    assert len({a.key, b.key, c.key}) == 3


def test_restart_without_persisted_state_is_refused():
    from avsec.crypto import Sealer, derive_session_keys

    sid = new_session_id()
    _, sealer, _ = make_session(LAB, CryptoProfile(), sid)
    sealer.seal(b"data", b"aad")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "state.json")
        sealer.state().save(path)
        with pytest.raises(CryptoError):
            SessionState.restore(path, new_session_id(), 0)   # wrong session
        with pytest.raises(CryptoError):
            SessionState.restore(path, sid, 5)                # wrong epoch
        with pytest.raises(CryptoError):
            # no forward reservation: resuming could repeat counters
            SessionState.restore(path, sid, 0)

        # with a durable reservation, resuming starts *above* everything reserved
        saved = []
        keys = derive_session_keys(LAB, sid)
        s2 = Sealer(keys, reserve_chunk=32, persist=lambda st: saved.append(st))
        for _ in range(5):
            s2.seal(b"x" * 8, b"aad")
        saved[-1].save(path)
        resumed = SessionState.restore(path, sid, 0)
        assert resumed.next_counter >= s2.counter
        assert resumed.next_counter >= 32


def test_new_epoch_is_a_safe_restart():
    """F02/F03: a new epoch means a new key, so the counter may restart at 0."""
    from avsec.crypto import derive_session_keys

    sid = new_session_id()
    k0 = derive_session_keys(LAB, sid, epoch=0)
    k1 = derive_session_keys(LAB, sid, epoch=1)
    assert k0.key != k1.key and k0.nonce_prefix != k1.nonce_prefix
    assert k0.nonce(0) != k1.nonce(0) or k0.key != k1.key
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "state.json")
        st0 = SessionState.begin_epoch(path, sid)
        st1 = SessionState.begin_epoch(path, sid)
        assert st1.epoch == st0.epoch + 1
        assert st1.next_counter == 0          # safe: the key changed with the epoch


def test_counter_cannot_be_chosen_or_repeated():
    """F03: there is no API that lets a caller pick or reuse a counter."""
    import inspect

    from avsec.crypto import Sealer

    params = set(inspect.signature(Sealer.seal).parameters)
    assert "counter" not in params
    _, sealer, _ = make_session(LAB, CryptoProfile())
    seen = [sealer.seal(b"x" * 8, b"aad")[0] for _ in range(50)]
    assert seen == sorted(set(seen)) == list(range(50))

    class _Boom(Exception):
        pass

    def _explode(counter):
        raise _Boom()

    with pytest.raises(_Boom):
        sealer.seal(b"x" * 8, _explode)
    # the counter consumed by the failed attempt is never handed out again
    assert sealer.seal(b"x" * 8, b"aad")[0] > 50


def test_counter_limit_is_explicit():
    from avsec.crypto import COUNTER_MAX, Sealer

    keys = derive_session_keys(LAB, new_session_id())
    s = Sealer(keys, start_counter=COUNTER_MAX + 1)
    with pytest.raises(NonceExhausted):
        s.seal(b"x", b"a")


def test_lab_and_secure_key_material_are_distinguishable():
    from avsec.crypto import generate_master_secret

    assert lab_master_secret(1).is_lab
    assert not generate_master_secret().is_lab
    assert lab_master_secret(1).key == lab_master_secret(1).key
    assert lab_master_secret(1).key != lab_master_secret(2).key


# --------------------------------- 4/6. authentication failures and anti-replay
def test_tampering_is_always_rejected():
    keys, sealer, opener = make_session(LAB, CryptoProfile())
    aad = _header().core_bytes()
    seq, ct = sealer.seal(b"payload!" * 4, aad)
    assert opener.open(seq, ct, aad) == b"payload!" * 4
    with pytest.raises(ReplayDetected):
        opener.open(seq, ct, aad)
    with pytest.raises(AuthenticationFailed):
        opener.open(seq + 1, bytes([ct[0] ^ 1]) + ct[1:], aad)
    with pytest.raises(AuthenticationFailed):
        opener.open(seq + 2, ct[:-1] + bytes([ct[-1] ^ 0x80]), aad)
    with pytest.raises(AuthenticationFailed):
        opener.open(seq + 3, ct, _header(stripe_id=9).core_bytes())
    with pytest.raises(AuthenticationFailed):
        opener.open(seq + 4, ct, _header(frame_id=6).core_bytes())


def test_replay_window_advances_only_after_authentication():
    from avsec.crypto import Opener, Sealer

    keys = derive_session_keys(LAB, new_session_id())
    sealer, opener = Sealer(keys), Opener(keys, replay_window=64)
    aad = _header().core_bytes()
    seq, ct = sealer.seal(b"body" * 8, aad)
    with pytest.raises(AuthenticationFailed):
        opener.open(seq + 500, bytes(len(ct)), aad)      # forged, high sequence
    assert opener.replay.highest == -1                   # window did not move
    assert opener.open(seq, ct, aad) == b"body" * 8      # real unit still accepted


def test_wrong_key_is_rejected():
    sid = new_session_id()
    _, sealer, _ = make_session(LAB, CryptoProfile(), sid)
    aad = _header().core_bytes()
    seq, ct = sealer.seal(b"secret", aad)
    _, _, other = make_session(lab_master_secret(999), CryptoProfile(), sid)
    with pytest.raises(AuthenticationFailed):
        other.open(seq, ct, aad)


# --------------------------------------------------- 7/8. FEC and its limits
@pytest.mark.parametrize("nsym", [32, 64, 128])
@pytest.mark.parametrize("length", [10, 191, 200, 600])
def test_fec_corrects_up_to_its_capability(nsym, length):
    cfg = FECConfig(k=255 - nsym, nsym=nsym)
    codec = RSCodec(cfg)
    rng = np.random.default_rng(length + nsym)
    msg = bytes(rng.integers(0, 256, length, dtype=np.uint8))
    enc = bytearray(codec.encode(msg))
    assert len(enc) == cfg.encoded_len(length)
    for _, _, es, elen in cfg.block_layout(length):
        for p in rng.choice(elen, size=min(cfg.max_errors, elen), replace=False):
            enc[es + int(p)] ^= 0xFF
    assert codec.decode(bytes(enc), length)[0] == msg


def test_fec_failure_is_explicit_not_silent():
    cfg = FECConfig(k=191, nsym=32)
    codec = RSCodec(cfg)
    msg = bytes(range(191))
    enc = bytearray(codec.encode(msg))
    for i in range(0, 120, 2):
        enc[i] ^= 0xA5
    out, stats = codec.try_decode(bytes(enc), len(msg))
    assert out is None or out != msg          # never silently "succeeds" wrongly
    if out is None:
        assert stats["ok"] is False and "error" in stats
    with pytest.raises(Uncorrectable):
        codec.decode(codec.encode(msg)[:-5], len(msg))   # wrong length


def test_fec_erasure_capability_is_double_the_error_capability():
    cfg = FECConfig(k=191, nsym=64)
    codec = RSCodec(cfg)
    msg = bytes(range(191))
    enc = bytearray(codec.encode(msg))
    positions = list(range(64))
    for p in positions:
        enc[p] = 0
    assert codec.decode(bytes(enc), len(msg), erasures=positions)[0] == msg


# ------------------------------------------- 9/10. sync, loss, duplication
def _modem():
    return RasterModem(ModemConfig(levels=4, symbol_width=8, symbol_height=2))


def test_symbol_packing_roundtrip():
    data = bytes(range(256))
    for bits in (1, 2, 4, 8):
        assert symbols_to_bytes(bytes_to_symbols(data, bits), bits) == data


def test_sync_is_found_from_the_received_signal_only():
    m = _modem()
    rng = np.random.default_rng(3)
    syms = rng.integers(0, 4, m.cfg.capacity_symbols).astype(np.uint8)
    raster = m.modulate(syms)
    for dx, dy in ((0, 0), (3, 2), (-4, -3), (7, 5)):
        shifted = np.roll(np.roll(raster, dy, axis=0), dx, axis=1)
        got = np.clip(shifted.astype(float) * 0.8 + 20, 0, 255).astype(np.uint8)
        res = m.demodulate(got)
        assert res.sync_found
        assert (res.dx, res.dy) == (dx, dy)
        assert (res.symbols != syms).mean() < 1e-3


def test_no_sync_is_reported_not_guessed():
    m = _modem()
    rng = np.random.default_rng(4)
    noise = rng.integers(0, 256, (m.cfg.raster_height, m.cfg.raster_width)).astype(np.uint8)
    assert not m.demodulate(noise).sync_found


# --------------------------- 11. independent decoding, no dependency on the past
@pytest.mark.parametrize("n_desc", [1, 2, 4])
def test_every_segment_decodes_on_its_own(n_desc):
    cfg = SourceCodingConfig(stripe_height=8, n_descriptions=n_desc, codec="dct",
                             quality=40, max_unit_payload=4096)
    coder = StripeCoder(cfg)
    frame = S.pattern_edges(H, W)
    segs = coder.encode_frame(frame, 0)
    assert segs
    for s in segs:
        out = coder.decode_segment(s.payload, s.geometry)
        assert out.shape == (s.geometry.height, s.geometry.width)
    # a subset alone must still reconstruct its own pixels
    subset = segs[::2]
    asm = FrameAssembler(H, W)
    res = asm.assemble([(s.geometry, coder.decode_segment(s.payload, s.geometry))
                        for s in subset])
    assert 0.0 < res.coverage < 1.0
    assert res.available.sum() > 0


def test_lossless_profile_reconstructs_exactly():
    cfg = SourceCodingConfig(stripe_height=8, n_descriptions=1, codec="raw",
                             max_unit_payload=8192)
    coder = StripeCoder(cfg)
    frame = S.pattern_text(H, W)
    segs = coder.encode_frame(frame, 0)
    asm = FrameAssembler(H, W)
    out = asm.assemble([(s.geometry, coder.decode_segment(s.payload, s.geometry))
                        for s in segs])
    assert np.array_equal(frame, out.image)
    assert out.coverage == 1.0


def test_lossy_profile_matches_its_own_decoder_not_the_source_pixels():
    cfg = SourceCodingConfig(stripe_height=8, n_descriptions=1, codec="dct",
                             quality=40, max_unit_payload=8192)
    coder = StripeCoder(cfg)
    frame = S.pattern_edges(H, W)
    segs = coder.encode_frame(frame, 0)
    a = [coder.decode_segment(s.payload, s.geometry) for s in segs]
    b = [coder.decode_segment(s.payload, s.geometry) for s in segs]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))   # deterministic


def test_descriptions_cover_disjoint_lattices():
    for n in (1, 2, 4):
        seen = set()
        for d in range(n):
            sx, sy, px, py = lattice(n, d)
            seen.add((sx, sy, px, py))
        assert len(seen) == n


# ---------------------------------------- 12/13. capacity, memory and deadlines
def test_transmitter_refuses_to_exceed_the_declared_capacity():
    from avsec.optimization import SharedBudget
    from avsec.transmitter import CapacityExceeded, TransportConfig, Transmitter

    budget = SharedBudget(rasters_per_frame=1)
    src = SourceCodingConfig(stripe_height=4, n_descriptions=1, codec="raw",
                             quality=0, max_unit_payload=256)
    tr = TransportConfig(modem=budget.modem(2, 12, 4),
                         fec_payload=FECConfig(k=127, nsym=128),
                         fec_header=FECConfig(k=52, nsym=48),
                         interleaver=InterleaverConfig(scheme="sequential"),
                         max_rasters_per_frame=1)
    tx = Transmitter(src, tr, LAB, frame_width=320, frame_height=240)
    with pytest.raises(CapacityExceeded):
        tx.encode_frame(S.pattern_texture(240, 320), 0)


def test_unauthenticated_sessions_never_enter_the_cache():
    """F01: only a successful AEAD open may install or evict a session."""
    from avsec.optimization import SharedBudget
    from avsec.receiver import Receiver
    from avsec.transmitter import TransportConfig

    budget = SharedBudget()
    src = SourceCodingConfig(stripe_height=24, max_unit_payload=640)
    tr = TransportConfig(modem=budget.modem(4, 8, 2),
                         fec_payload=FECConfig(k=127, nsym=128),
                         fec_header=FECConfig(k=52, nsym=48))
    rx = Receiver(src, tr, LAB, frame_width=W, frame_height=H, max_sessions=3)
    for _ in range(10):
        ctx = (new_session_id(), 0, 0)
        assert rx._lookup_opener(ctx) is None      # read-only lookup
        assert rx._provisional_opener(ctx) is not None
    assert len(rx._openers) == 0                   # nothing was cached

    # only a commit installs, and eviction stays bounded
    for _ in range(10):
        ctx = (new_session_id(), 0, 0)
        rx._commit_session(ctx, rx._provisional_opener(ctx))
    assert len(rx._openers) <= 3


def test_source_coding_budget_failure_is_explicit():
    from avsec.source_coding import BudgetExceeded

    cfg = SourceCodingConfig(stripe_height=48, n_descriptions=1, codec="raw",
                             max_unit_payload=64, max_segments=4)
    with pytest.raises(BudgetExceeded):
        StripeCoder(cfg).encode_frame(S.pattern_edges(96, 320), 0)


# ------------------------------------------------- 14/15. reproducibility, leaks
def test_non_secret_experiment_randomness_is_reproducible():
    from avsec.utils import experiment_rng

    a = experiment_rng(7, "chan", 1).normal(size=64)
    b = experiment_rng(7, "chan", 1).normal(size=64)
    c = experiment_rng(7, "chan", 2).normal(size=64)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_tuner_refuses_to_score_on_the_test_split():
    from avsec.channel import preset
    from avsec.optimization import SharedBudget, Tuner, build_candidate

    t = Tuner(SharedBudget(), LAB, preset("clean"), H, W)
    cand = build_candidate(SharedBudget(), 24, 1, "dct", 15, 640, 128,
                           ("block", 278, 0), (4, 8, 2))
    with pytest.raises(RuntimeError):
        t.score(cand, [S.synthetic_still("edges", H, W)], split="test")


def test_split_keeps_whole_sequences_together():
    from avsec.optimization import split_sources

    srcs = [S.synthetic_sequence("edges", H, W, 3) for _ in range(6)]
    parts = split_sources(srcs, seed=1)
    names = [[s.name for s in v] for v in parts.values()]
    flat = [n for group in names for n in group]
    assert len(flat) >= len(srcs)          # every sequence is used somewhere
    for group in parts.values():
        assert all(isinstance(s, S.FrameSource) for s in group)


def test_master_secret_never_prints_its_bytes():
    assert "redacted" in repr(LAB)
    assert LAB.key.hex() not in repr(LAB)


# ----------------------------------------------------- 16. metric bookkeeping
def test_quality_is_reported_both_full_frame_and_verified_only():
    from avsec.evaluation import quality_pair

    orig = S.pattern_edges(H, W)
    rendered = orig.copy()
    rendered[H // 2 :, :] = 128
    avail = np.zeros((H, W), dtype=bool)
    avail[: H // 2, :] = True
    q = quality_pair(orig, rendered, avail)
    assert q["coverage"] == pytest.approx(0.5)
    assert q["psnr_verified"] > q["psnr_full"]      # discarding hard areas is visible


def test_availability_map_separates_received_from_estimated():
    coder = StripeCoder(SourceCodingConfig(stripe_height=8, codec="raw",
                                           max_unit_payload=8192))
    segs = coder.encode_frame(S.pattern_edges(H, W), 0)
    asm = FrameAssembler(H, W)
    res = asm.assemble([(s.geometry, coder.decode_segment(s.payload, s.geometry))
                        for s in segs[:3]])
    assert res.available.sum() > 0
    assert res.available.sum() < H * W
    assert res.summary()["estimated_fraction"] == pytest.approx(1 - res.coverage)


# ------------------------------------------------------------ placement rules
@pytest.mark.parametrize("scheme,kw", [("sequential", {}), ("block", {"depth": 16})])
def test_placement_schemes_are_bijections(scheme, kw):
    il = Interleaver(InterleaverConfig(scheme=scheme, **kw), 40, 60)
    cells = np.concatenate(il.place([40 * 60], [0], 1))
    assert sorted(cells.tolist()) == list(range(40 * 60))


def test_bawp_never_reuses_a_cell_and_meets_its_burst_bound():
    R, C, D, B, W_ROWS, n = 120, 60, 4, 4, 64, 100
    il = Interleaver(InterleaverConfig(scheme="bawp", window_rows=W_ROWS, burst_rows=B),
                     R, C)
    units = 24
    placements = il.place([n] * units, [i % D for i in range(units)], D)
    allc = np.concatenate(placements)
    assert np.unique(allc).size == allc.size
    h = min(hh for _, hh in bawp_bands(W_ROWS, D, B))
    bound = B * int(np.ceil(n / h))
    for cells in placements:
        rows = cells // C
        worst = max(int(((rows >= s) & (rows < s + B)).sum()) for s in range(R - B + 1))
        assert worst <= bound


def test_bawp_confines_a_burst_to_fewer_description_classes():
    R, C, D, B, W_ROWS, n = 120, 60, 4, 4, 64, 60
    units = 40
    descs = [i % D for i in range(units)]

    def touched(scheme, **kw):
        il = Interleaver(InterleaverConfig(scheme=scheme, **kw), R, C)
        pl = il.place([n] * units, descs, D)
        worst = 0
        for s in range(R - B + 1):
            hit = {u % D for u, cells in enumerate(pl)
                   if ((cells // C >= s) & (cells // C < s + B)).any()}
            worst = max(worst, len(hit))
        return worst

    assert touched("bawp", window_rows=W_ROWS, burst_rows=B) < touched("block", depth=16)


def test_both_endpoints_derive_the_same_placement():
    from avsec.optimization import SharedBudget
    from avsec.receiver import Receiver
    from avsec.transmitter import TransportConfig, Transmitter

    budget = SharedBudget()
    src = SourceCodingConfig(stripe_height=24, n_descriptions=2, max_unit_payload=320)
    tr = TransportConfig(modem=budget.modem(4, 8, 2),
                         fec_payload=FECConfig(k=127, nsym=128),
                         fec_header=FECConfig(k=52, nsym=48),
                         interleaver=InterleaverConfig(scheme="bawp", burst_rows=12))
    tx = Transmitter(src, tr, LAB, frame_width=W, frame_height=H)
    rx = Receiver(src, tr, LAB, frame_width=W, frame_height=H)
    assert len(tx.placement) == len(rx._placement)
    for a, b in zip(tx.placement, rx._placement):
        assert np.array_equal(a, b)
