"""One regression per item of the second review round (R01-R12).

Each test reproduces the reported failure against the code that preceded the
fix and passes after it.  The identifiers match the review document.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from avsec import sources as S
from avsec.crypto import CryptoProfile, lab_master_secret
from avsec.fec import FECConfig
from avsec.interleaving import InterleaverConfig
from avsec.optimization import SharedBudget
from avsec.receiver import Receiver
from avsec.receiver.sessions import Admission, KeyCache, SessionLedger
from avsec.source_coding import SourceCodingConfig
from avsec.transmitter import TransportConfig, Transmitter

LAB = lab_master_secret(1)
H, W = 192, 256
BUDGET = SharedBudget()


def _pipeline(**rx_kw):
    src = SourceCodingConfig(stripe_height=24, n_descriptions=1, codec="dct",
                             quality=12, max_unit_payload=640)
    tr = TransportConfig(
        modem=BUDGET.modem(4, 8, 2),
        fec_payload=FECConfig(k=255 - 96, nsym=96),
        fec_header=FECConfig(k=52, nsym=48),
        interleaver=InterleaverConfig(scheme="block", depth=278, burst_rows=8),
        max_rasters_per_frame=BUDGET.rasters_per_frame)
    tx = Transmitter(src, tr, LAB, CryptoProfile(), frame_width=W, frame_height=H)
    rx = Receiver(src, tr, LAB, CryptoProfile(), frame_width=W, frame_height=H,
                  **rx_kw)
    return src, tr, tx, rx


def _counts(rx, rasters):
    out = {}
    for r in rasters:
        for k, v in rx.receive_raster(r).counts().items():
            out[k] = out.get(k, 0) + v
    return out


# ========================================================================= R01
# Displacing a session from the cache used to discard its replay window, after
# which a recorded raster authenticated again.
def test_r01_displaced_session_cannot_be_replayed():
    """Accept a frame -> four session changes -> replay the original frame."""
    frame = S.pattern_edges(H, W)
    _, _, tx, rx = _pipeline(max_sessions=4)

    recorded = tx.encode_frame(frame, 0).rasters
    assert _counts(rx, recorded).get("verified", 0) > 0
    victim = (tx.session_id, tx.session_epoch, tx.crypto_profile.stream_id)

    for i in range(4):                       # four *genuine* new sessions
        tx.new_session()
        assert _counts(rx, tx.encode_frame(frame, 0).rasters).get("verified", 0) > 0

    assert len(rx.sessions) <= 4
    assert rx.sessions.state_of(victim).value == "closed", \
        "the displaced session must be closed, not forgotten"

    replayed = _counts(rx, recorded)
    assert replayed.get("verified", 0) == 0, \
        "a raster recorded before the displacement was accepted again"
    assert replayed.get("session_closed", 0) > 0


def test_r01_key_cache_eviction_changes_no_decision():
    """Evicting *keys* must cost an HKDF derivation and nothing else."""
    frame = S.pattern_edges(H, W)
    _, _, tx, rx = _pipeline(max_sessions=4, key_cache_size=1)
    recorded = tx.encode_frame(frame, 0).rasters
    assert _counts(rx, recorded).get("verified", 0) > 0

    for _ in range(8):                       # thrash the one-entry key cache
        rx._session_keys((bytes(8), 0, 0))
        rx._session_keys((bytes([1]) * 8, 0, 0))
    assert rx._keys.evictions > 0
    assert len(rx.sessions) == 1             # admissibility state untouched

    assert _counts(rx, recorded).get("verified", 0) == 0
    assert _counts(rx, recorded).get("replay", 0) > 0


def test_r01_unauthenticated_traffic_cannot_move_session_state():
    _, _, tx, rx = _pipeline(max_sessions=4)
    frame = S.pattern_edges(H, W)
    _counts(rx, tx.encode_frame(frame, 0).rasters)
    before = rx.sessions.export_state()

    rng = np.random.default_rng(7)
    shape = tx.encode_frame(frame, 1).rasters[0].shape
    for _ in range(20):
        rx.receive_raster(rng.integers(0, 256, shape).astype(np.uint8))
    after = rx.sessions.export_state()
    assert after["active"] == before["active"]
    assert after["closed"] == before["closed"]
    assert after["highest_epoch"] == before["highest_epoch"]


def test_r01_restart_with_persisted_state_still_rejects_the_replay():
    """The defined restart behaviour: freshness comes from the saved ledger."""
    frame = S.pattern_edges(H, W)
    src, tr, tx, rx = _pipeline()
    recorded = tx.encode_frame(frame, 0).rasters
    assert _counts(rx, recorded).get("verified", 0) > 0
    state = rx.export_session_state()

    warm = Receiver(src, tr, LAB, CryptoProfile(), frame_width=W, frame_height=H,
                    session_state=state)
    assert warm.sessions.cold_start == "persisted"
    assert _counts(warm, recorded).get("verified", 0) == 0

    # and the honest converse: a cold receiver has no history, which is why
    # the mode is reported rather than assumed away
    cold = Receiver(src, tr, LAB, CryptoProfile(), frame_width=W, frame_height=H)
    assert cold.sessions.describe()["cold_start"] == "tofu"
    assert _counts(cold, recorded).get("verified", 0) > 0


def test_r01_ledger_classify_is_read_only():
    ledger = SessionLedger(active_capacity=2, retired_capacity=4)
    ctx = (bytes(8), 0, 0)
    for _ in range(5):
        verdict, _ = ledger.classify(ctx, 0)
        assert verdict is Admission.ADMIT
    assert len(ledger) == 0 and len(ledger.closed) == 0


def test_r01_ledger_reports_when_it_can_no_longer_prove_freshness():
    """Bounded memory is a stated limit, not a silent one."""
    ledger = SessionLedger(active_capacity=1, retired_capacity=2)
    for i in range(6):
        ctx = (bytes([i]) * 8, 0, 0)
        ledger.commit(ctx, ledger.window_for(ctx), 0, float(i))
    d = ledger.describe()
    assert d["counters"]["forgotten_closures"] > 0
    assert "свіжість більше не доводиться" in d["exactness"]


def test_r01_key_cache_holds_no_security_state():
    cache = KeyCache(capacity=2)
    assert cache.get((bytes(8), 0, 0)) is None
    assert cache.describe()["capacity"] == 2


# ========================================================================= R02
# The observation key did not include the channel, so two channels writing the
# same frame index overwrote one another and any pooled estimate was whichever
# row happened to be read last.
def _obs(method, scene, frame, psnr, channel="", rep=0, status="ok",
         source=None, **metrics):
    from avsec.statistics import Observation

    m = {"psnr_full": psnr}
    m.update(metrics)
    return Observation(method=method, scene=scene, clip=scene + "-c",
                       repetition=rep, frame=frame, status=status, metrics=m,
                       channel=channel, source_id=source or scene)


def test_r02_two_channels_do_not_overwrite_each_other():
    from avsec.statistics import ResultTable

    rows = [_obs("P", "s0", 0, 30.0, channel="clean"),
            _obs("P", "s0", 0, 10.0, channel="bursty")]
    t = ResultTable(rows)
    assert t.per_scene("psnr_full", "P", channel="clean") == {"s0": 30.0}
    assert t.per_scene("psnr_full", "P", channel="bursty") == {"s0": 10.0}
    # and without a channel filter both rows contribute, rather than one of
    # them silently replacing the other
    assert t.per_scene("psnr_full", "P")["s0"] == pytest.approx(20.0)


def test_r02_row_order_does_not_change_the_estimate():
    from avsec.statistics import ResultTable

    rows = [_obs("P", f"s{i}", f, 20.0 + i + f, channel=ch)
            for i in range(4) for f in range(3) for ch in ("clean", "bursty")]
    a = ResultTable(rows).paired("P", "P", "psnr_full", n_boot=200)
    fwd = ResultTable(rows).per_scene("psnr_full", "P", channel="bursty")
    rev = ResultTable(list(reversed(rows))).per_scene("psnr_full", "P",
                                                      channel="bursty")
    assert fwd == rev and a["mean"] == 0.0


def test_r02_re_adding_identical_rows_adds_no_information():
    from avsec.statistics import ResultTable

    rows = [_obs("P", "s0", f, 30.0, channel="clean") for f in range(3)]
    once = ResultTable(rows).summary("psnr_full", n_boot=200)[0]
    twice = ResultTable(rows + rows).summary("psnr_full", n_boot=200)[0]
    assert once["n_units"] == twice["n_units"] == 1
    assert once["mean"] == pytest.approx(twice["mean"])


def test_r02_pooling_over_channels_requires_declared_weights():
    from avsec.statistics import ResultTable

    t = ResultTable([_obs("P", "s0", 0, 30.0, channel="clean"),
                     _obs("P", "s0", 0, 10.0, channel="bursty")])
    with pytest.raises(ValueError):
        t.pooled_over_channels("psnr_full", "P", {"clean": 1.0})
    got = t.pooled_over_channels("psnr_full", "P", {"clean": 3.0, "bursty": 1.0})
    assert got["s0"] == pytest.approx(25.0)


def test_r02_analysis_writes_no_pooled_all_row(tmp_path):
    """Channels are conditions, not repeated draws; there is no ALL row."""
    import csv

    from avsec.analysis import analyse

    rows = [{"method": m, "scene": f"s{i}", "clip": f"c{i}", "repetition": 0,
             "frame_id": 0, "channel": ch, "status": "ok",
             "psnr_full": 20.0 + i + (2.0 if m == "P" else 0.0)
                          + (0.0 if ch == "clean" else -4.0),
             "coverage": 1.0}
            for i in range(5) for m in ("P", "B4") for ch in ("clean", "bursty")]
    with open(tmp_path / "frames.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = analyse(str(tmp_path))
    assert "ALL" not in res["channels"]
    with open(tmp_path / "paired_effects.csv", encoding="utf-8", newline="") as fh:
        published = {r["channel"] for r in csv.DictReader(fh)}
    assert published == {"clean", "bursty"}


# ========================================================================= R03
# Modem cells were counted as RS bytes, damaged cells of one byte were counted
# several times, and a whole unit was compared against one block's parity.
def test_r03_cells_are_not_bytes():
    from avsec.damage import UnitLayout, assess_unit
    from avsec.fec import FECConfig

    lay = UnitLayout(bits_per_symbol=2, fec_header=FECConfig(k=52, nsym=48),
                     fec_payload=FECConfig(k=127, nsym=128), payload_len=336)
    assert lay.symbols_per_byte == 4
    d = assess_unit(list(range(40)), lay)
    assert d.damaged_symbols == 40
    assert d.damaged_bytes == 10            # not 40


def test_r03_several_hits_in_one_byte_count_once():
    from avsec.damage import UnitLayout, assess_unit
    from avsec.fec import FECConfig

    lay = UnitLayout(bits_per_symbol=2, fec_header=FECConfig(k=52, nsym=48),
                     fec_payload=FECConfig(k=127, nsym=128), payload_len=336)
    assert assess_unit([0, 1, 2, 3], lay).damaged_bytes == 1
    assert assess_unit([0], lay).damaged_bytes == 1


def test_r03_a_unit_is_several_codewords():
    from avsec.damage import UnitLayout
    from avsec.fec import FECConfig

    lay = UnitLayout(bits_per_symbol=2, fec_header=FECConfig(k=52, nsym=48),
                     fec_payload=FECConfig(k=127, nsym=128), payload_len=336)
    blocks = lay.blocks()
    assert len(blocks) > 1
    assert {b[0] for b in blocks} == {"header", "payload"}
    # every block is accounted for, including the shortened last one
    assert sum(b[2] for b in blocks) == lay.wire_bytes


def test_r03_model_and_real_decoder_agree():
    """The analytic verdict must match an actual Reed-Solomon decode."""
    from avsec.damage import UnitLayout, decode_check
    from avsec.fec import FECConfig

    lay = UnitLayout(bits_per_symbol=2, fec_header=FECConfig(k=52, nsym=48),
                     fec_payload=FECConfig(k=127, nsym=128), payload_len=336)
    total = lay.wire_symbols
    disagreements = []
    for i in range(1, 16):
        n = int(total * i / 17)
        start = (i * 211) % max(1, total - n)
        for as_er in (False, True):
            r = decode_check(range(start, start + n), lay, seed=i,
                             as_erasures=as_er)
            if not r["agree"]:
                disagreements.append((i, n, as_er, r))
    assert not disagreements, disagreements[:3]


def test_r03_erasures_and_errors_are_different_budgets():
    from avsec.damage import UnitLayout, assess_unit
    from avsec.fec import FECConfig

    lay = UnitLayout(bits_per_symbol=8, fec_header=FECConfig(k=52, nsym=48),
                     fec_payload=FECConfig(k=127, nsym=128), payload_len=127)
    base = lay.header_encoded
    # 70 damaged payload bytes in one block: too many as unknown errors
    # (2*70 > 128) but inside the erasure budget (70 <= 128)
    d = assess_unit(list(range(base, base + 70)), lay)
    assert d.damaged_bytes == 70
    assert not d.survives_as_errors
    assert d.survives_as_erasures


# ========================================================================= R04
# "Both descriptions touched by a burst" was reported as "both lost".
def test_r04_a_repaired_description_is_not_a_lost_one():
    from avsec.evaluation import DescriptionOutcome, description_accounting

    touched_but_fine = [
        DescriptionOutcome(stripe_id=0, desc_id=0, n_segs=1, n_touched=1,
                           n_verified=1),
        DescriptionOutcome(stripe_id=0, desc_id=1, n_segs=1, n_touched=1,
                           n_verified=1),
    ]
    acc = description_accounting(touched_but_fine, 2)
    assert acc["desc_touched"] == 2
    assert acc["desc_unusable"] == 0
    assert acc["band_no_usable_description"] == 0
    assert acc["band_lost_frac"] == 0.0


def test_r04_three_separate_metrics():
    from avsec.evaluation import DescriptionOutcome, description_accounting

    acc = description_accounting([
        DescriptionOutcome(0, 0, n_segs=2, n_touched=2, n_fec_failed=1,
                           n_verified=1),               # partial
        DescriptionOutcome(0, 1, n_segs=1, n_touched=1, n_fec_failed=1,
                           n_verified=0),               # unusable
        DescriptionOutcome(1, 0, n_segs=1, n_verified=1),
        DescriptionOutcome(1, 1, n_segs=1, n_verified=1),
    ], 2)
    assert acc["desc_touched"] == 2
    assert acc["desc_fec_failed"] == 2
    assert acc["desc_unusable"] == 2
    assert acc["desc_partial"] == 1
    # band 0 has no fully usable description; band 1 is complete
    assert acc["band_no_usable_description"] == 1
    assert acc["band_complete"] == 1


def test_r04_band_states_are_exhaustive():
    from avsec.evaluation import BAND_OUTCOMES, DescriptionOutcome, \
        description_accounting

    acc = description_accounting(
        [DescriptionOutcome(i, 0, n_segs=1, n_verified=1) for i in range(3)], 1)
    assert sum(acc[f"band_{k}"] for k in BAND_OUTCOMES) == acc["bands_total"]


def test_r04_pipeline_reports_the_three_metrics():
    from avsec.channel import ChannelTrace

    src, tr, tx, rx = _pipeline()
    from avsec.baselines import DigitalMethod
    from avsec.channel import preset

    m = DigitalMethod("P", src, tr, preset("bursty"), LAB, H, W, CryptoProfile())
    m.reset()
    trace = ChannelTrace(seed=1, scene="x", repetition=0, profile="bursty",
                         rasters_per_frame=tr.max_rasters_per_frame)
    res = m.process(S.pattern_edges(H, W), 0, trace)
    for key in ("desc_touched", "desc_fec_failed", "desc_unusable",
                "band_no_usable_description"):
        assert key in res.metrics.extra


# ========================================================================= R05
# A clip stopped at the first capacity failure, so the frames a configuration
# could not carry vanished from its average instead of counting against it.
def test_r05_every_scheduled_instant_has_a_row():
    from avsec.matrix import _gap_row

    class _Src:
        provenance = "synthetic"

    job = type("J", (), {"method": "P", "job_id": "j", "clip": "c",
                         "scene": "s", "repetition": 0,
                         "channel": type("C", (), {"label": "bursty", "axis": "",
                                                   "value": float("nan")})()})()
    row = _gap_row(job, _Src(), 3, np.zeros((8, 8), np.uint8), None,
                   "capacity", "too big", "bursty", "t")
    assert row["availability"] == 0.0
    assert row["status"] == "capacity"
    assert row["displayed_source"] == "neutral"
    assert "psnr_displayed" in row and "coverage" in row


def test_r05_availability_counts_failed_instants():
    from avsec.statistics import ResultTable

    rows = [_obs("P", "s0", 0, 30.0, channel="c", availability=1.0),
            _obs("P", "s0", 1, float("nan"), channel="c", status="capacity",
                 availability=0.0)]
    t = ResultTable(rows)
    # quality uses the successful instant only ...
    assert t.per_scene("psnr_full", "P") == {"s0": 30.0}
    # ... availability uses both
    assert t.per_scene("availability", "P")["s0"] == pytest.approx(0.5)


def test_r05_failing_early_cannot_raise_operability():
    from avsec.analysis import AnalysisPlan, operability
    from avsec.statistics import ResultTable

    plan = AnalysisPlan()
    good = [_obs("A", "s0", f, 30.0, channel="c", coverage=1.0) for f in range(4)]
    quitter = [_obs("B", "s0", 0, 30.0, channel="c", coverage=1.0)] + [
        _obs("B", "s0", f, float("nan"), channel="c", status="capacity")
        for f in range(1, 4)]
    rows = {r["method"]: r for r in operability(ResultTable(good + quitter), plan)}
    assert rows["A"]["operable_fraction"] == pytest.approx(1.0)
    assert rows["B"]["operable_fraction"] == pytest.approx(0.25)
    assert rows["A"]["n_scheduled_instants"] == rows["B"]["n_scheduled_instants"]


# ========================================================================= R06
# p-values were reconstructed from the width of a bootstrap interval.
def test_r06_p_values_come_from_the_differences():
    import inspect

    from avsec import statistics

    src = inspect.getsource(statistics.holm_adjust)
    assert "3.92" not in src and "erfc" not in src
    d = np.array([2.0, 2.2, 1.8, 2.1, 1.9, 2.3, 2.0, 2.1])
    tests = statistics.difference_tests(d, n_boot=2000, seed=1)
    assert tests["p_raw"] < 0.01
    assert "bootstrap" in tests["p_source"]


def test_r06_a_comparison_without_differences_cannot_be_significant():
    from avsec.statistics import holm_adjust

    comps = [{"n_scenes": 10, "mean": 3.0, "lo": 2.0, "hi": 4.0}]
    holm_adjust(comps)
    assert not comps[0]["significant_holm"]
    assert not np.isfinite(comps[0]["p_raw"])


def test_r06_holm_family_is_declared_in_advance():
    from avsec.analysis import PRIMARY, PRIMARY_FAMILY

    assert PRIMARY in PRIMARY_FAMILY
    assert ("P", "B4t") in PRIMARY_FAMILY and ("B4t", "B4") in PRIMARY_FAMILY


def test_r06_exploratory_pairs_do_not_change_a_declared_decision():
    from avsec.statistics import ResultTable

    rng = np.random.default_rng(5)
    rows = []
    for i in range(12):
        base = 20.0 + rng.normal(0, 2)
        rows.append(_obs("P", f"s{i}", 0, base + 1.2, channel="c"))
        rows.append(_obs("B4", f"s{i}", 0, base, channel="c"))
        rows.append(_obs("X1", f"s{i}", 0, base + rng.normal(0, 3), channel="c"))
        rows.append(_obs("X2", f"s{i}", 0, base + rng.normal(0, 3), channel="c"))
    fam = [("P", "B4")]
    small = ResultTable(rows[:2] + [r for r in rows if r.method in ("P", "B4")])
    a = [c for c in small.all_pairs("psnr_full", ["P", "B4"], n_boot=2000,
                                    family=fam) if c["in_family"]][0]
    b = [c for c in ResultTable(rows).all_pairs(
        "psnr_full", ["P", "B4", "X1", "X2"], n_boot=2000, family=fam)
        if c["in_family"]][0]
    assert a["significant_holm"] == b["significant_holm"]
    assert a["p_holm"] == pytest.approx(b["p_holm"])


def test_r06_no_difference_is_not_equality():
    from avsec.statistics import equivalence

    wide = equivalence(np.array([-3.0, 3.0, -2.5, 2.7, -3.1, 2.9]), 0.5,
                       n_boot=2000)
    assert not wide["equivalent"] and "НЕ показана" in wide["equivalence_verdict"]
    tight = equivalence(np.array([0.02, -0.01, 0.0, 0.01, -0.02, 0.01]), 0.5,
                        n_boot=2000)
    assert tight["equivalent"]


# ========================================================================= R07
def test_r07_one_photograph_is_one_source():
    from avsec.sources.drone import SOURCE_ID, DRONE_SCENES

    assert all(sc[-1] == "test" for sc in DRONE_SCENES), \
        "one photograph cannot be divided into calibration/validation/test"
    assert SOURCE_ID.startswith("photo:")


def test_r07_crop_overlap_is_measured_not_claimed():
    from avsec.dataset import DatasetManifest, ClipRecord

    def _clip(name, crop):
        return ClipRecord(clip_id=name, parent_scene_id=name, category="other",
                          provenance="local_file", source_kind="", n_frames=1,
                          width=8, height=8, fps=1.0, t_start_s=0.0, t_end_s=1.0,
                          content_sha256=name, first_frame_dhash="",
                          source_id="photo:x", crop=crop)

    m = DatasetManifest([_clip("a", "0,0,100,100"), _clip("b", "50,50,150,150"),
                         _clip("c", "200,200,300,300")])
    got = {(o["clip_a"], o["clip_b"]): o for o in m.measure_crop_overlap()}
    assert got[("a", "b")]["overlaps"] and got[("a", "b")]["iou"] > 0
    assert not got[("a", "c")]["overlaps"]


def test_r07_scene_and_source_are_different_units():
    from avsec.statistics import ResultTable

    rows = [_obs("P", f"s{i}", 0, 20.0 + i, source="one-photo") for i in range(6)]
    t = ResultTable(rows, unit_of_independence="source")
    assert len(t.per_scene("psnr_full", "P")) == 6
    assert len(t.per_source("psnr_full", "P")) == 1
    assert t.summary("psnr_full", n_boot=200)[0]["n_units"] == 1


def test_r07_manifest_warns_when_scenes_share_one_source():
    from avsec.dataset import DatasetManifest, ClipRecord

    clips = [ClipRecord(clip_id=f"c{i}", parent_scene_id=f"s{i}", category="other",
                        provenance="local_file", source_kind="", n_frames=1,
                        width=8, height=8, fps=1.0, t_start_s=0.0, t_end_s=1.0,
                        content_sha256=f"h{i}", first_frame_dhash="",
                        split="test", source_id="photo:one") for i in range(5)]
    rep = DatasetManifest(clips).validate()
    assert rep["n_sources"] == 1 and rep["n_scenes"] == 5
    assert any("does not generalise" in w for w in rep["warnings"])


def test_r07_splits_are_assigned_by_source():
    from avsec.dataset import DatasetManifest, ClipRecord

    clips = [ClipRecord(clip_id=f"c{i}", parent_scene_id=f"s{i}", category="other",
                        provenance="local_file", source_kind="", n_frames=1,
                        width=8, height=8, fps=1.0, t_start_s=0.0, t_end_s=1.0,
                        content_sha256=f"h{i}", first_frame_dhash="",
                        source_id=f"photo:{i // 3}") for i in range(12)]
    m = DatasetManifest(clips)
    m.assign_splits(seed=11)
    by_source = {}
    for c in m.clips:
        by_source.setdefault(c.source_id, set()).add(c.split)
    assert all(len(v) == 1 for v in by_source.values())
    assert m.validate()["problems"] == []


# ========================================================================= R08
def test_r08_the_retuned_baseline_exists_and_is_the_right_ablation():
    from avsec.config import ALL_METHODS, DEFAULT_PROFILES

    assert "B4t" in ALL_METHODS
    b4t, p = DEFAULT_PROFILES["B4t"], DEFAULT_PROFILES["P"]
    # same transport as P ...
    assert (b4t.max_unit_payload, b4t.fec_nsym, b4t.quality, b4t.stripe_height,
            b4t.modulation) == (p.max_unit_payload, p.fec_nsym, p.quality,
                                p.stripe_height, p.modulation)
    # ... and neither proposed mechanism
    assert b4t.n_descriptions == 1 and b4t.interleaver[0] == "block"
    assert p.n_descriptions == 2 and p.interleaver[0] == "bawp"


def test_r08_the_retuned_baseline_is_built_and_authenticates():
    import dataclasses

    from avsec.config import load_config
    from avsec.experiments import build_methods

    cfg = dataclasses.replace(load_config("configs/smoke.yaml"),
                              methods=("B4t",))
    m = build_methods(cfg)["B4t"]
    assert m.authenticated
    assert "retuned" in m.notes


def test_r08_the_research_configs_include_it():
    from avsec.config import load_config

    for path in ("configs/research_main.yaml", "configs/research_drone.yaml",
                 "configs/research_natural.yaml"):
        cfg = load_config(path)
        assert "B4t" in cfg.methods, path
        assert "B4t" in cfg.profiles, path


# ========================================================================= R09
def test_r09_the_chain_is_walked_in_both_orders():
    from avsec.research import E13_CHAIN, E13_CHAIN_REVERSE

    def _final(chain):
        return chain[-1][1]

    assert _final(E13_CHAIN) == _final(E13_CHAIN_REVERSE), \
        "both orders must end at the same configuration"
    assert E13_CHAIN[1][0] != E13_CHAIN_REVERSE[1][0]


def test_r09_step_differences_carry_intervals():
    from avsec.research import _finish_chain

    rng = np.random.default_rng(2)
    scenes = [f"s{i}" for i in range(8)]
    cells = []
    rows = []
    for step in range(3):
        cells.append({m: {s: 20.0 + step + rng.normal(0, 0.2) for s in scenes}
                      for m in ("psnr_full", "coverage", "availability",
                                "psnr_displayed")})
        rows.append({"order": "fwd", "step": step, "label": f"step{step}",
                     "admissible": True})
    common, dropped, scene_rows = _finish_chain(rows, cells)
    assert common == scenes and not dropped
    assert len(scene_rows) == 3 * len(scenes)
    assert rows[1]["delta_psnr_full_lo"] < rows[1]["delta_psnr_full"] \
        < rows[1]["delta_psnr_full_hi"]
    assert rows[1]["delta_psnr_full_significant"]


def test_r09_interaction_of_unit_size_and_fec_is_reported():
    from avsec.research import _payload_fec_interaction

    rows = [{"study": "payload_x_fec", "admissible": True,
             "max_unit_payload": pl, "fec_nsym": ns,
             "psnr_full": 20.0 + 0.01 * ns + (0.02 * ns if pl == 320 else 0)}
            for pl in (160, 320) for ns in (32, 128)]
    res = _payload_fec_interaction(rows)
    assert res["available"]
    assert set(res["fec_effect_per_payload_db"]) == {"160", "320"}
    assert "взаємод" in res["statement"] or "адитив" in res["statement"]


# ========================================================================= R10
def test_r10_the_operating_region_is_a_grid_not_a_threshold():
    from avsec.research import E15_BURST_LINES, E15_BURST_RATES, E15_CRITERION

    assert len(E15_BURST_RATES) >= 3
    assert len(E15_BURST_LINES) >= 8
    # dense where the methods change places
    near = [n for n in E15_BURST_LINES if 12 <= n <= 32]
    assert len(near) >= 5
    assert set(E15_CRITERION) == {"coverage_verified_min", "psnr_full_min"}


def test_r10_crossings_are_reported_as_brackets():
    from avsec.research import _crossings

    cells = [{"burst_rate_per_frame": 4.0, "burst_len_lines": n,
              "d_P_B4t": 1.0 - 0.1 * n, "d_P_B4t_significant": True}
             for n in (4, 8, 12, 16)]
    got = _crossings(cells)
    assert len(got) == 1
    assert got[0]["between_lines"] == [8, 12]
    assert "ВИМІРЯНИМИ" in got[0]["note"]


# ========================================================================= R11
def test_r11_the_worktree_state_is_recorded():
    from avsec.utils import environment_record

    rec = environment_record()
    assert rec["git_worktree"] in ("clean", "DIRTY", "unknown (no git)")
    assert "git_dirty_files" in rec
    assert isinstance(rec.get("pip_freeze"), list)


def test_r11_the_plan_has_a_fingerprint_that_tracks_changes():
    import dataclasses

    from avsec.analysis import AnalysisPlan

    a = AnalysisPlan()
    assert a.fingerprint() == AnalysisPlan().fingerprint()
    b = dataclasses.replace(a, metric="ssim_full")
    assert a.fingerprint() != b.fingerprint()


def test_r11_the_declared_plan_file_matches_the_code_defaults():
    from avsec.analysis import PRIMARY, PRIMARY_FAMILY, AnalysisPlan

    plan = AnalysisPlan.load("configs/analysis_plan.yaml")
    assert plan.primary == PRIMARY
    assert plan.family == PRIMARY_FAMILY


def test_r11_verify_reruns_the_claims_and_notices_a_tampered_number(tmp_path):
    import csv
    import json as _json

    from avsec.analysis import analyse
    from avsec.verify import verify

    rng = np.random.default_rng(9)
    run = tmp_path / "run"
    run.mkdir()
    rows = []
    for i in range(10):
        base = 22.0 + rng.normal(0, 2)
        for m, bump in (("P", 1.5), ("B4", 0.0)):
            for ch in ("clean", "bursty"):
                rows.append({"method": m, "scene": f"s{i}", "clip": f"c{i}",
                             "repetition": 0, "frame_id": 0, "channel": ch,
                             "status": "ok",
                             "psnr_full": base + bump + (0 if ch == "clean" else -3),
                             "coverage": 1.0, "availability": 1.0})
    with open(run / "frames.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    analyse(str(run))

    good = verify(str(run))
    effects = [c for c in good["checks"] if c["kind"] == "paired_effect"]
    assert effects and all(c["ok"] for c in effects)

    # now edit a published number and make sure verification catches it
    path = run / "paired_effects.csv"
    with open(path, encoding="utf-8", newline="") as fh:
        pub = list(csv.DictReader(fh))
        fields = list(pub[0])
    pub[0]["mean"] = str(float(pub[0]["mean"]) + 5.0)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(pub)
    bad = verify(str(run))
    assert not bad["ok"]
    assert any(c["kind"] == "paired_effect" and not c["ok"]
               for c in bad["checks"])
    _json.dumps(bad)          # the report must be serialisable


# ========================================================================= R12
def test_r12_no_document_claims_a_new_cipher_or_hardware_validation():
    import glob
    import io as _io

    banned = ("новий стійкий шифр", "доведено дешевизну",
              "перевірено на реальному відеолінку",
              "перевірено на реальному відеотракті")
    offenders = []
    for path in glob.glob("docs/*.md") + ["README.md", "results/README.md"]:
        if not os.path.exists(path):
            continue
        text = _io.open(path, encoding="utf-8").read().lower()
        for phrase in banned:
            if phrase in text:
                offenders.append((path, phrase))
    assert not offenders, offenders
