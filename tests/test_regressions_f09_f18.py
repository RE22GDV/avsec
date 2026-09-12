"""One regression per fixed defect, F09 through F18.

Each test reproduces the specific failure the review described, so that a
future change that re-introduces it fails here rather than in a published
number.  They deliberately avoid long runs: every one of them is a targeted
check of the invariant, not an end-to-end experiment.
"""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import pytest

from avsec import sources as S


# ------------------------------------------------------------------- F09
def test_f09_lab_session_ids_are_deterministic_and_labelled():
    """A benchmark run must be byte-reproducible; a secure one must not be."""
    from avsec.crypto import LabSessionIds, SecureSessionIds, make_session_id_source

    a = make_session_id_source("lab", crypto_seed=7, run_identity="cfg-x")
    b = make_session_id_source("lab", crypto_seed=7, run_identity="cfg-x")
    assert [a.next("s") for _ in range(4)] == [b.next("s") for _ in range(4)]
    assert a.origin == "lab"

    # a different configuration must not reuse the same session ids
    c = make_session_id_source("lab", crypto_seed=7, run_identity="cfg-y")
    assert c.next("s") != b.next("s")

    # contexts are independent streams
    d = make_session_id_source("lab", crypto_seed=7, run_identity="cfg-x")
    assert d.next("stream-a") != d.next("stream-b")

    secure = make_session_id_source("secure")
    assert secure.origin == "secure"
    assert secure.next() != secure.next()
    assert isinstance(secure, SecureSessionIds)
    assert isinstance(a, LabSessionIds)


def test_f09_run_identity_ignores_free_text_notes():
    """Two configurations that differ only in prose are the same experiment."""
    from avsec.config import ExperimentConfig

    a = ExperimentConfig(notes="first attempt")
    b = ExperimentConfig(notes="second attempt, same settings")
    assert a.run_identity() == b.run_identity()
    c = dataclasses.replace(a, channel_preset="harsh")
    assert c.run_identity() != a.run_identity()


# ------------------------------------------------------------------- F10
def test_f10_near_duplicates_across_splits_are_reported():
    """A still taken from a sequence must not sit in a different split."""
    from avsec.dataset import build_manifest

    srcs = S.default_suite(96, 128, n_frames=4)
    manifest = build_manifest(srcs, seed=3)
    pairs = {(d.clip_a, d.clip_b) for d in manifest.duplicates}
    assert ("still_texture", "seq_texture_fast") in pairs
    assert ("still_edges", "seq_edges_slow") in pairs
    # and the scene grouping keeps them together, so no split can separate them
    scene = {c.clip_id: c.parent_scene_id for c in manifest.clips}
    assert scene["still_texture"] == scene["seq_texture_fast"]
    assert scene["still_edges"] == scene["seq_edges_slow"]


def test_f10_scene_id_patterns_are_matched_longest_first():
    """'still_texture' contains 'text'; the longer pattern must win."""
    from avsec.dataset import default_category, default_scene_id

    src = S.synthetic_still("texture", 64, 64)
    assert "text" not in default_scene_id(src).split(":")[-1].replace("texture", "")
    assert default_category(src) != default_category(S.synthetic_still("text", 64, 64))


def test_f10_research_suite_has_no_duplicates_and_no_split_leak():
    from avsec.dataset import build_manifest

    srcs = S.research_suite(96, 128, n_frames=4)
    report = build_manifest(srcs, seed=3).validate()
    assert report["problems"] == []
    assert report["n_scenes"] == report["n_clips"] == 34
    assert report["scenes_per_split"]["test"] == 20


def test_f10_declared_splits_survive_reassignment():
    """A pre-registered split must not be redrawn from a seed at run time."""
    from avsec.dataset import build_manifest

    srcs = S.research_suite(64, 96, n_frames=2)
    a = build_manifest(srcs, seed=1)
    b = build_manifest(srcs, seed=999)
    assert {c.clip_id: c.split for c in a.clips} == {c.clip_id: c.split for c in b.clips}


# ------------------------------------------------------------------- F14
def _obs(method, scene, clip, rep, frame, psnr, status="ok"):
    from avsec.statistics import Observation

    return Observation(method=method, scene=scene, clip=clip, repetition=rep,
                       frame=frame, status=status, metrics={"psnr_full": psnr})


def test_f14_aggregation_is_permutation_invariant():
    from avsec.statistics import ResultTable

    rows = [_obs("P", f"s{i}", f"c{i}", 0, f, 20.0 + i + f)
            for i in range(4) for f in range(3)]
    a = ResultTable(rows).per_scene("psnr_full", "P")
    b = ResultTable(list(reversed(rows))).per_scene("psnr_full", "P")
    assert a == b


def test_f14_duplicate_frames_do_not_add_independent_samples():
    from avsec.statistics import ResultTable

    rows = [_obs("P", "s0", "c0", 0, f, 30.0) for f in range(3)]
    once = ResultTable(rows).summary("psnr_full", n_boot=200)[0]
    twice = ResultTable(rows + rows).summary("psnr_full", n_boot=200)[0]
    assert once["n_scenes"] == twice["n_scenes"] == 1
    assert once["mean"] == pytest.approx(twice["mean"])


def test_f14_paired_comparison_drops_by_key_not_by_index():
    """Truncating with min(len(a), len(b)) would pair unrelated scenes."""
    from avsec.statistics import ResultTable

    rows = [_obs("P", "s0", "c0", 0, 0, 30.0), _obs("P", "s1", "c1", 0, 0, 32.0),
            _obs("B4", "s1", "c1", 0, 0, 28.0), _obs("B4", "s2", "c2", 0, 0, 10.0)]
    res = ResultTable(rows).paired("P", "B4", "psnr_full", n_boot=200)
    assert res["n_scenes"] == 1                    # only s1 is shared
    assert res["scenes_only_in_a"] == ["s0"]
    assert res["scenes_only_in_b"] == ["s2"]
    assert res["n_scenes_dropped"] == 2
    assert res["mean"] == pytest.approx(4.0)


def test_f14_failures_are_kept_apart_by_category():
    from avsec.statistics import ResultTable

    rows = [_obs("P", "s0", "c0", 0, 0, 30.0),
            _obs("P", "s1", "c1", 0, -1, float("nan"), status="capacity"),
            _obs("P", "s2", "c2", 0, -1, float("nan"), status="exception")]
    summary = ResultTable(rows).summary("psnr_full", n_boot=200)[0]
    assert summary["failures"] == {"capacity": 1, "exception": 1}
    assert summary["n_failed"] == 2
    assert summary["n_scenes"] == 1


def test_f14_holm_is_monotone_and_never_below_the_raw_p():
    from avsec.statistics import holm_adjust

    # R06: the p-value comes from the paired differences themselves, so a
    # comparison must carry them; an interval's width is not a substitute.
    rng = np.random.default_rng(3)
    comps = [{"n_scenes": 10, "mean": 3.0, "lo": 2.0, "hi": 4.0,
              "diffs": list(3.0 + rng.normal(0, 0.5, 10))},
             {"n_scenes": 10, "mean": 0.2, "lo": -0.1, "hi": 0.5,
              "diffs": list(0.2 + rng.normal(0, 0.5, 10))},
             {"n_scenes": 10, "mean": 1.0, "lo": 0.4, "hi": 1.6,
              "diffs": list(1.0 + rng.normal(0, 0.5, 10))}]
    holm_adjust(comps, n_boot=2000)
    for c in comps:
        assert c["p_holm"] >= c["p_raw"] - 1e-12
    ordered = sorted(comps, key=lambda c: c["p_raw"])
    assert all(ordered[i]["p_holm"] <= ordered[i + 1]["p_holm"] + 1e-12
               for i in range(len(ordered) - 1))


def test_f14_channel_only_uncertainty_is_labelled_as_not_generalising():
    from avsec.statistics import ResultTable

    rows = [_obs("P", "s0", "c0", r, 0, 30.0 + r) for r in range(5)]
    res = ResultTable(rows).channel_only("P", "psnr_full", "s0", n_boot=200)
    assert "this scene only" in res["generalises_to"]


# ------------------------------------------------------------------- F15
def test_f15_public_whitening_provides_no_confidentiality():
    """The control must be reproducible by anyone - that is the whole point."""
    from avsec.utils import public_whiten

    data = b"secret-looking payload" * 4
    once = public_whiten(data, 7)
    assert once != data
    assert public_whiten(once, 7) == data          # involution, no key involved
    assert public_whiten(data, 8) != once          # sequence-dependent


def test_f15_equal_amplitude_window_is_not_equal_signal_power():
    from avsec.evaluation import signal_statistics

    rng = np.random.default_rng(0)
    two_level = rng.choice([40, 216], size=(64, 64)).astype(np.uint8)
    four_level = rng.choice([40, 99, 158, 216], size=(64, 64)).astype(np.uint8)
    a, b = signal_statistics(two_level), signal_statistics(four_level)
    assert a["min_level"] == b["min_level"] and a["max_level"] == b["max_level"]
    assert a["variance"] > b["variance"] * 1.2     # same window, different power


def test_f15_security_properties_cover_every_method():
    from avsec.config import ALL_METHODS
    from avsec.experiments import SECURITY_PROPERTIES

    for m in ALL_METHODS:
        assert m in SECURITY_PROPERTIES, m
        entry = SECURITY_PROPERTIES[m]
        assert {"confidentiality", "authentication", "integrity"} <= set(entry)
    assert SECURITY_PROPERTIES["B0d-W"]["confidentiality"].startswith("none")
    assert SECURITY_PROPERTIES["P"]["authentication"].startswith("AEAD")


# ------------------------------------------------------------------- F16
def test_f16_latency_is_a_timestamp_difference_not_a_sum():
    from avsec.timeline import Timeline, TimingProfile

    tl = Timeline(TimingProfile(source_fps=8.333, rasters_per_frame=3))
    info = tl.schedule_frame(0, stripes=8, units=12, rasters=3,
                             interleaver_rows=278)
    assert info["latency_s"] == pytest.approx(info["display"] - info["capture"])
    # overlapping stages must not be added: the schedule is shorter than the sum
    stage_sum = (1.0 / 8.333) + 278 * 2 / (25.0 * 576) + 3 / 25.0 + 1 / 25.0
    assert info["latency_s"] < stage_sum


def test_f16_memory_quantities_are_reported_separately():
    from avsec.timeline import MemoryReport, logical_buffers, model_arrays

    rep = MemoryReport(
        process_peak_bytes=42_000_000, process_method="test",
        logical_buffers=logical_buffers(1288, 4, 3, 278, 76, 2, 1, 256 * 192),
        model_arrays=model_arrays(576, 720, 3))
    d = rep.to_dict()
    assert d["logical_total_kb"] < 512            # the protocol limit applies here
    assert d["process_peak_mb"] > d["logical_total_kb"] / 1024
    assert d["model_total_mb"] > 0
    assert "NOT a limit on process memory" in d["note"]
    # the three numbers are genuinely different quantities
    assert len({round(d["logical_total_kb"] * 1024), d["process_peak_bytes"],
                sum(d["model_arrays"].values())}) == 3


def test_f16_level_a_and_level_b_have_different_line_timing():
    from avsec.timeline import CVBS_625_50, TimingProfile

    a = TimingProfile(total_lines=576, active_lines=560)
    b = CVBS_625_50
    assert a.total_lines != b.total_lines
    assert a.line_period_s != b.line_period_s
    assert b.describe()["line_period_us"] == pytest.approx(64.0, abs=1e-6)


def test_f16_process_peak_memory_is_actually_measured():
    from avsec.timeline import process_peak_bytes

    value, method = process_peak_bytes()
    assert value > 0, f"peak memory query failed: {method}"
    assert method != "unavailable"


# ------------------------------------------------------------------- F17
def _tiny_plan(tmp_path, methods=("B0d",), reps=1, frames=1, clips=2):
    from avsec.config import ExperimentConfig
    from avsec.matrix import ChannelPoint, MatrixSpec

    cfg = ExperimentConfig(frame_width=128, frame_height=96, n_frames=4,
                           sources={"kind": "research", "n_frames": 4})
    spec = MatrixSpec(methods=methods, channels=[ChannelPoint.named("mild")],
                      repetitions=reps, splits=("test",), max_frames=frames,
                      max_clips=clips, name="test")
    return cfg, spec


def test_f17_job_id_is_content_derived_not_positional():
    from avsec.matrix import ChannelPoint, Job

    a = Job(clip="c", scene="s", method="P", profile_id="p", repetition=0,
            n_frames=4, clip_hash="h", config_id="cfg",
            channel=ChannelPoint.named("mild"))
    b = Job(clip="c", scene="s", method="P", profile_id="p", repetition=0,
            n_frames=4, clip_hash="h", config_id="cfg",
            channel=ChannelPoint.named("mild"))
    assert a.job_id == b.job_id
    for change in (dict(clip_hash="other"), dict(repetition=1),
                   dict(config_id="cfg2"), dict(method="B4"), dict(seed=5)):
        assert dataclasses.replace(a, **change).job_id != a.job_id
    assert dataclasses.replace(
        a, channel=ChannelPoint.named("harsh")).job_id != a.job_id


def test_f17_resume_does_not_duplicate_rows(tmp_path):
    from avsec.matrix import MatrixRunner

    cfg, spec = _tiny_plan(tmp_path)
    out = str(tmp_path / "run")
    first = MatrixRunner(cfg, spec, out, workers=1).run()
    assert first["completeness"]["complete"]
    n_frames = first["n_frames"]

    second = MatrixRunner(cfg, spec, out, workers=1)
    assert second.pending() == []
    again = second.run()
    assert again["n_jobs_this_call"] == 0
    assert again["n_frames"] == n_frames        # rebuilt from what was persisted


def test_f17_partial_run_reports_the_missing_jobs_by_name(tmp_path):
    from avsec.matrix import MatrixRunner

    cfg, spec = _tiny_plan(tmp_path, methods=("B0d",), clips=3)
    runner = MatrixRunner(cfg, spec, str(tmp_path / "run"), workers=1)
    res = runner.run(limit=1)
    c = res["completeness"]
    assert not c["complete"]
    assert c["n_missing"] == len(runner.jobs) - 1
    assert c["missing"] and "job_id" in c["missing"][0]


def test_f17_sweep_axes_are_numeric_not_preset_names():
    from avsec.matrix import ChannelPoint

    pts = ChannelPoint.sweep("mild", "noise_sigma", [0, 4, 8])
    assert [p.value for p in pts] == [0.0, 4.0, 8.0]
    assert all(p.axis == "noise_sigma" for p in pts)
    assert pts[1].config().noise_sigma == 4.0
    with pytest.raises(ValueError):
        ChannelPoint.sweep("mild", "preset", [1, 2])


def test_f17_dry_run_measures_each_method_separately(tmp_path):
    from avsec.matrix import MatrixRunner

    cfg, spec = _tiny_plan(tmp_path, methods=("B0a", "B4"), clips=1)
    est = MatrixRunner(cfg, spec, str(tmp_path / "run"), workers=1).estimate()
    d = est.to_dict()
    assert "measured one frame per method" in d["cost_model"]
    assert "B0a=" in d["cost_model"] and "B4=" in d["cost_model"]
    assert d["n_jobs"] == 2


def test_f17_parallel_estimate_is_not_serial_time_over_workers(tmp_path):
    from avsec.matrix import MatrixRunner

    cfg, spec = _tiny_plan(tmp_path, methods=("B0a",), clips=2)
    one = MatrixRunner(cfg, spec, str(tmp_path / "a"), workers=1).estimate()
    many = MatrixRunner(cfg, spec, str(tmp_path / "b"), workers=8).estimate()
    assert many.est_wall_s > one.est_wall_s / 8      # not a linear division
    assert "measured efficiency" in many.measured_from


def test_f17_provenance_is_written_before_the_run(tmp_path):
    from avsec.matrix import MatrixRunner

    cfg, spec = _tiny_plan(tmp_path)
    out = str(tmp_path / "run")
    MatrixRunner(cfg, spec, out, workers=1)         # constructed, not run
    for name in ("config.yaml", "run_manifest.json", "dataset_manifest.csv"):
        assert os.path.exists(os.path.join(out, name)), name
    with open(os.path.join(out, "run_manifest.json"), encoding="utf-8") as fh:
        m = json.load(fh)
    assert m["run_id"] and m["seeds"]["key_mode"] == "lab"
    assert m["plan"]["methods"] == list(spec.methods)


def test_f17_receiver_state_is_continuous_across_a_clip(tmp_path):
    """One reset per job, not per frame: frame 1 must not look like a new session."""
    from avsec.matrix import MatrixRunner

    cfg, spec = _tiny_plan(tmp_path, methods=("B4",), frames=3, clips=1)
    runner = MatrixRunner(cfg, spec, str(tmp_path / "run"), workers=1)
    res = runner._run_job(runner.jobs[0])
    assert res.status == "ok"
    assert [int(r["frame_id"]) for r in res.frames] == [0, 1, 2]
    # a restarted session would reject later frames as stale
    assert all(float(r["coverage"]) > 0 for r in res.frames)


# ------------------------------------------------------------------- F18
def test_f18_bit_exact_frames_are_not_reported_as_99_db():
    from avsec.evaluation import psnr, quality_pair

    img = np.full((32, 32), 128, dtype=np.uint8)
    assert psnr(img, img) == float("inf")
    q = quality_pair(img, img, np.ones_like(img, dtype=bool))
    assert q["bit_exact"] is True
    assert q["mse_full"] == 0.0
    assert q["psnr_full"] == float("inf")


def test_f18_bit_exact_frames_are_counted_not_averaged_in():
    from avsec.statistics import Observation, ResultTable

    rows = [Observation("P", "s0", "c0", 0, 0, metrics={"psnr_full": 30.0,
                                                        "bit_exact": 0.0}),
            Observation("P", "s1", "c1", 0, 0, metrics={"psnr_full": float("inf"),
                                                        "bit_exact": 1.0})]
    s = ResultTable(rows).summary("psnr_full", n_boot=200)[0]
    assert s["mean"] == pytest.approx(30.0)      # inf never enters the mean
    assert s["n_bit_exact_excluded"] == 1


def test_f18_documents_are_generated_from_one_run(tmp_path):
    from avsec.publish import README_BEGIN, README_END, RunView, build_readme_block

    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({
        "run_id": "abc123", "commit": "deadbeef",
        "environment": {"platform": "test", "python": "3.12"},
        "dataset": {"n_clips": 4, "n_scenes": 4},
        "plan": {"splits": ["test"]}}), encoding="utf-8")
    (run / "analysis.json").write_text(json.dumps({
        "n_scenes": 4, "n_observations": 40, "channels": ["mild"],
        "methods": ["P", "B4"],
        "primary": [{"channel": "mild", "mean": 1.0, "lo": 0.5, "hi": 1.5,
                     "n_scenes": 4, "significant_at_95": True,
                     "significant_holm": True}],
        "primary_verdict": {"channels_where_a_is_better": ["mild"],
                            "channels_where_a_is_worse": [],
                            "channels_with_no_detectable_difference": []}}),
        encoding="utf-8")
    block = build_readme_block(RunView(str(run)), str(run / "figures"))
    assert block.startswith(README_BEGIN) and block.rstrip().endswith(README_END)
    assert "abc123" in block and "deadbeef" in block
    assert "Апаратних вимірювань немає" in block


def test_f18_a_loss_is_reported_not_hidden(tmp_path):
    """The generated text must name the channels where the proposal loses."""
    from avsec.publish import RunView, primary_section

    run = tmp_path / "run"
    run.mkdir()
    (run / "analysis.json").write_text(json.dumps({
        "plan": {"primary_comparison": ["P", "B4"], "metric": "psnr_full",
                 "unit_of_independence": "scene", "aggregation": "x"},
        "primary": [{"channel": "clean", "mean": -3.5, "lo": -4.0, "hi": -3.0,
                     "n_scenes": 19, "significant_at_95": True,
                     "significant_holm": True}],
        "primary_verdict": {"channels_where_a_is_better": [],
                            "channels_where_a_is_worse": ["clean"],
                            "channels_with_no_detectable_difference": []}}),
        encoding="utf-8")
    text = primary_section(RunView(str(run)))
    assert "перевага B4" in text
    assert "Значуща поразка P" in text


def test_f18_pending_figures_carry_a_reason(tmp_path):
    from avsec.program import CATALOGUE, KEY, catalogue_status

    rows = catalogue_status(str(tmp_path))
    assert len(rows) == len(KEY) + len(CATALOGUE) == 56
    # the ten key figures come first: a reader who stops after one screen
    # should have seen the ones that carry the argument
    assert [r["figure"] for r in rows[:len(KEY)]] == [f.gid for f in KEY]
    assert all(r["status"] == "pending" for r in rows)
    assert all(r["reason"] for r in rows)
    e11 = [r for r in rows if r["experiment"] == "E11"]
    assert not e11, "E11 has no software figures; its outputs are H01-H04"

    catalogue_only = catalogue_status(str(tmp_path), include_key=False)
    assert len(catalogue_only) == 43


def test_f18_hardware_experiment_is_declared_not_done():
    from avsec.program import EXPERIMENTS

    e11 = EXPERIMENTS["E11"]
    assert e11.status == "not_done"
    assert "недоступн" in e11.reason
    assert "дешевий" in e11.reason      # the word is explicitly disclaimed


# ------------------------------------- follow-up defects found while running
def test_e07_schedule_switches_channel_without_resetting_the_session():
    """Recovery time is a property of a session that keeps running.

    Running each impairment phase as its own job reset the receiver at every
    boundary, so the first frame after the impairment was always a fresh
    session on a clean channel and every recovery time came out as zero.
    """
    from avsec.matrix import ChannelPoint, Job

    script = ((0, ChannelPoint.named("mild")),
              (10, ChannelPoint.named("harsh")),
              (20, ChannelPoint.named("mild")))
    job = Job(clip="c", scene="s", method="P", profile_id="p",
              channel=script[0][1], repetition=0, n_frames=30, schedule=script)
    assert [job.channel_at(f).label for f in (0, 9, 10, 19, 20, 29)] == [
        "mild", "mild", "harsh", "harsh", "mild", "mild"]
    # the schedule is part of the job's identity, or a resumed run would reuse
    # results produced under a different impairment script
    plain = dataclasses.replace(job, schedule=())
    assert plain.job_id != job.job_id


def test_e07_clips_are_long_enough_to_observe_recovery():
    """The impairment must end before the clip does."""
    from avsec.research import E07_SCENARIOS, e07_schedule

    n_frames = 48
    for name, _ in E07_SCENARIOS:
        script, (t0, t1) = e07_schedule(name, n_frames)
        assert t1 < n_frames, name
        assert n_frames - t1 >= 8, f"{name}: only {n_frames - t1} frames after the fault"
        assert script[0][0] == 0
        assert max(f for f, _ in script) <= t1


def test_e03_axes_that_need_a_cofactor_declare_it():
    """A flat curve caused by a mis-specified sweep is not a finding.

    Sweeping burst_len_lines on a base whose burst rate is zero produced an
    identical number at every point: there were no bursts to lengthen.
    """
    from avsec.channel import preset
    from avsec.matrix import ChannelPoint
    from avsec.research import E03_COFACTORS

    assert preset("mild").burst_rate_per_frame == 0.0
    assert "burst_rate_per_frame" in E03_COFACTORS["burst_len_lines"]
    fixed = tuple(sorted(E03_COFACTORS["burst_len_lines"].items()))
    point = ChannelPoint(base="mild",
                         overrides=tuple(sorted((("burst_len_lines", 32.0),) + fixed)),
                         axis="burst_len_lines", value=32.0)
    cfg = point.config()
    assert cfg.burst_len_lines == 32
    assert cfg.burst_rate_per_frame > 0, "the axis would have no effect"


def test_e06_window_is_a_real_parameter_of_the_profile():
    """Setting only the scheme left every window cell identical."""
    import dataclasses as dc

    from avsec.config import MethodProfile
    from avsec.interleaving import Interleaver
    from avsec.optimization import SharedBudget

    budget = SharedBudget()
    depths = []
    for window in (16, 64, 0):
        prof = dc.replace(MethodProfile(interleaver=("bawp", 0, 8)),
                          interleaver_window=window)
        tr = prof.transport_config(budget)
        il = Interleaver(tr.interleaver, tr.modem.n_data_rows, tr.modem.n_data_cols)
        depths.append(il.accumulation_rows)
    assert depths[0] == 16 and depths[1] == 64
    assert len(set(depths)) == 3, f"the window did not change anything: {depths}"
