"""One runner for the whole experiment matrix (defect F17).

Before this module the demo, the comparison, the sweep and the ablation each
had their own loop, their own aggregation and their own idea of what a "run"
was.  That made it impossible to say whether two published numbers came from
the same data, and a crashed sweep had to be restarted from scratch.

The unit of work here is a **job**::

    job = clip x method x profile x channel point x repetition

A job has a stable identifier derived from the resolved configuration, the
content hash of its clip and the seeds - not from its position in a list - so:

* re-running the matrix **resumes** and never duplicates a completed job;
* a job that failed is recorded with its error and can be retried alone;
* a completeness check can name exactly which jobs are missing.

Receiver state is continuous across the frames of a clip: the transport is
reset once per job, then every frame of that clip is processed in order, so
replay windows, epoch watermarks and reassembly buffers behave as they would
in a real session.

Channel axes are **numeric**.  A sweep varies ``noise_sigma`` or
``burst_rate_per_frame`` over real values; preset names are labels for a
starting point, never an axis of measurement.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from avsec.channel import ChannelTrace, PRESETS, RasterChannelConfig
from avsec.channel import preset as channel_preset
from avsec.config import ExperimentConfig
from avsec.dataset import DatasetManifest
from avsec.sources import FrameSource
from avsec.statistics import Observation, ResultTable
from avsec.transmitter import CapacityExceeded
from avsec.utils import (NpEncoder, StageTimer, append_jsonl, ensure_dir, write_csv,
                         write_json)

Progress = Optional[Callable[[str, float, Dict[str, Any]], None]]

#: Numeric axes a sweep may vary.  Anything else is a fixed part of the
#: configuration, not an experimental factor.
NUMERIC_AXES = (
    "noise_sigma", "burst_rate_per_frame", "burst_len_lines", "gain", "offset",
    "shift_x", "shift_y", "line_jitter_sigma", "slow_drift_px",
    "region_dropout_rate", "frame_drop_prob", "lowpass_sigma",
)


# --------------------------------------------------------------- channel point
@dataclass(frozen=True)
class ChannelPoint:
    """One point of the channel axis: a base preset plus numeric overrides.

    ``label`` is for humans.  Comparisons and plots use ``axis``/``value``,
    which are real numbers, so a figure never puts preset names on an
    axis that is claimed to be SNR or burst rate.
    """

    base: str = "clean"
    overrides: Tuple[Tuple[str, float], ...] = ()
    axis: str = ""
    value: float = float("nan")

    @property
    def label(self) -> str:
        if self.axis:
            return f"{self.base}[{self.axis}={self.value:g}]"
        if self.overrides:
            body = ",".join(f"{k}={v:g}" for k, v in self.overrides)
            return f"{self.base}[{body}]"
        return self.base

    def config(self) -> RasterChannelConfig:
        cfg = channel_preset(self.base)
        for k, v in self.overrides:
            if not hasattr(cfg, k):
                raise KeyError(f"unknown channel field {k!r}")
            cur = getattr(cfg, k)
            setattr(cfg, k, type(cur)(v) if isinstance(cur, int) else float(v))
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {"base": self.base, "overrides": dict(self.overrides),
                "axis": self.axis, "value": self.value, "label": self.label}

    @staticmethod
    def sweep(base: str, axis: str, values: Sequence[float]) -> List["ChannelPoint"]:
        if axis not in NUMERIC_AXES:
            raise ValueError(f"{axis!r} is not a numeric axis; have {NUMERIC_AXES}")
        return [ChannelPoint(base=base, overrides=((axis, float(v)),),
                             axis=axis, value=float(v)) for v in values]

    @staticmethod
    def named(name: str) -> "ChannelPoint":
        if name not in PRESETS:
            raise KeyError(f"unknown preset {name!r}")
        return ChannelPoint(base=name)


# ------------------------------------------------------------------------ job
@dataclass
class Job:
    """One unit of work.  Its id does not depend on where it sits in a list."""

    clip: str
    scene: str
    method: str
    profile_id: str
    channel: ChannelPoint
    repetition: int
    n_frames: int
    clip_hash: str = ""
    config_id: str = ""
    seed: int = 0
    #: Optional per-frame channel script: ``((first_frame, ChannelPoint), ...)``,
    #: sorted by first_frame.  The receiver is **not** reset at a boundary - that
    #: is the whole point, because recovery time is a property of a session that
    #: keeps running.
    schedule: Tuple[Tuple[int, ChannelPoint], ...] = ()

    def channel_at(self, frame_id: int) -> ChannelPoint:
        current = self.channel
        for first, point in self.schedule:
            if frame_id >= first:
                current = point
        return current

    @property
    def job_id(self) -> str:
        material = json.dumps({
            "clip": self.clip, "clip_hash": self.clip_hash,
            "method": self.method, "profile": self.profile_id,
            "channel": self.channel.to_dict(), "repetition": self.repetition,
            "frames": self.n_frames, "config": self.config_id, "seed": self.seed,
            "schedule": [[f, p.to_dict()] for f, p in self.schedule],
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {"job_id": self.job_id, "clip": self.clip, "scene": self.scene,
                "method": self.method, "profile_id": self.profile_id,
                "channel": self.channel.label, "channel_axis": self.channel.axis,
                "channel_value": self.channel.value,
                "repetition": self.repetition, "n_frames": self.n_frames,
                "clip_hash": self.clip_hash, "config_id": self.config_id}


@dataclass
class JobResult:
    job: Job
    status: str = "ok"                       # ok | capacity | exception | skipped
    detail: str = ""
    frames: List[Dict[str, Any]] = field(default_factory=list)
    units: List[Dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0
    trace_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = self.job.to_dict()
        d.update({"status": self.status, "detail": self.detail,
                  "n_frames_done": len(self.frames), "wall_s": round(self.wall_s, 4),
                  "trace_id": self.trace_id})
        return d


# ------------------------------------------------------------------- the plan
@dataclass
class MatrixSpec:
    """What to run.  Kept separate from *how* it is executed."""

    methods: Sequence[str]
    channels: Sequence[ChannelPoint]
    repetitions: int = 3
    splits: Sequence[str] = ("test",)
    max_frames: int = 0                       # 0 = every frame of the clip
    max_clips: int = 0                        # 0 = every clip of the split
    #: Optional per-frame channel script shared by every job of this plan.
    schedule: Tuple[Tuple[int, ChannelPoint], ...] = ()
    name: str = "matrix"

    workers: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "methods": list(self.methods),
                "channels": [c.to_dict() for c in self.channels],
                "repetitions": self.repetitions, "splits": list(self.splits),
                "max_frames": self.max_frames, "max_clips": self.max_clips,
                "schedule": [[f, p.to_dict()] for f, p in self.schedule],
                "workers": self.workers}

    @staticmethod
    def from_plan(path: str, cfg: Optional[ExperimentConfig] = None) -> Tuple[
            ExperimentConfig, "MatrixSpec"]:
        """Read a plan file: an ordinary config with a ``matrix:`` block.

        The plan is part of the configuration, so the run identity that names
        the output already covers the matrix that produced it.
        """
        import yaml

        from avsec.config import load_config

        cfg = cfg or load_config(path)
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        block = raw.get("matrix") or {}
        channels: List[ChannelPoint] = []
        for entry in block.get("channels", [{"base": cfg.channel_preset}]):
            if isinstance(entry, str):
                channels.append(ChannelPoint.named(entry))
                continue
            base = str(entry.get("base", "clean"))
            axis = str(entry.get("axis", ""))
            if axis:
                channels.extend(ChannelPoint.sweep(base, axis,
                                                   [float(v) for v in entry["values"]]))
            else:
                ov = tuple(sorted((str(k), float(v))
                                  for k, v in (entry.get("overrides") or {}).items()))
                channels.append(ChannelPoint(base=base, overrides=ov))
        spec = MatrixSpec(
            methods=tuple(block.get("methods", cfg.methods)),
            channels=channels,
            repetitions=int(block.get("repetitions", cfg.channel_realisations)),
            splits=tuple(block.get("splits", ("test",))),
            max_frames=int(block.get("max_frames", cfg.max_frames_per_source)),
            max_clips=int(block.get("max_clips", 0)),
            name=str(block.get("name", cfg.name)),
            workers=int(block.get("workers", 1)),
        )
        return cfg, spec


def build_jobs(cfg: ExperimentConfig, spec: MatrixSpec,
               sources: Sequence[FrameSource],
               manifest: Optional[DatasetManifest] = None) -> List[Job]:
    """Expand the specification into jobs, in a deterministic order."""
    by_name = {s.name: s for s in sources}
    if manifest is None:
        from avsec.dataset import build_manifest

        manifest = build_manifest(sources, seed=cfg.seed)
    config_id = cfg.run_identity()
    jobs: List[Job] = []
    # A reduced study takes the FIRST clips of the split in manifest order, not a
    # random draw: the subset must be the same every time the plan is re-read.
    eligible = [c for c in manifest.clips
                if not spec.splits or c.split in spec.splits]
    if spec.max_clips > 0:
        eligible = eligible[: spec.max_clips]
    for clip in eligible:
        src = by_name.get(clip.clip_id)
        if src is None:
            continue
        n = len(src.frames) if spec.max_frames <= 0 else min(spec.max_frames,
                                                             len(src.frames))
        for method in spec.methods:
            for ch in spec.channels:
                for rep in range(max(1, spec.repetitions)):
                    jobs.append(Job(
                        clip=clip.clip_id, scene=clip.parent_scene_id,
                        method=method, profile_id=_profile_id(cfg, method),
                        channel=ch, repetition=rep, n_frames=n,
                        clip_hash=clip.content_sha256, config_id=config_id,
                        seed=cfg.channel_seed_value,
                        schedule=tuple(spec.schedule)))
    return jobs


def _profile_id(cfg: ExperimentConfig, method: str) -> str:
    prof = cfg.profile("B0d" if method == "B0d-W" else method)
    blob = json.dumps(prof.to_dict(), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


# ------------------------------------------------------------------- estimate
@dataclass
class DryRun:
    n_jobs: int
    n_frames: int
    est_wall_s: float
    est_bytes: int
    per_method: Dict[str, int]
    per_channel: Dict[str, int]
    measured_from: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_jobs": self.n_jobs, "n_frames_total": self.n_frames,
            "estimated_wall_s": round(self.est_wall_s, 1),
            "estimated_wall_human": _hms(self.est_wall_s),
            "estimated_output_bytes": self.est_bytes,
            "estimated_output_mb": round(self.est_bytes / 1e6, 2),
            "jobs_per_method": self.per_method,
            "jobs_per_channel": self.per_channel,
            "cost_model": self.measured_from,
        }


def _hms(seconds: float) -> str:
    s = int(max(0.0, seconds))
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


#: Per-method per-frame wall cost, measured once per configuration and cached,
#: so a dry run is a measurement rather than a guess: ``{run_id: {method: s}}``.
_COST_CACHE: Dict[str, Dict[str, float]] = {}

#: Measured parallel efficiency of the process pool on this pipeline.
#: 16 identical jobs took 13.9 s with one worker, 5.8 s with four and 4.4 s
#: with eight - a 3.2x speedup on 8 workers, i.e. 0.40 efficiency.  The code is
#: CPU-bound Python (Reed-Solomon, placement, per-cell demodulation), so wall
#: time must never be estimated as serial time divided by the worker count.
_MEASURED_EFFICIENCY = 0.40


class MatrixRunner:
    """Executes jobs, persists rows, resumes, and checks completeness."""

    #: bytes of CSV per frame row, measured empirically on the smoke config
    BYTES_PER_FRAME_ROW = 420

    #: Predicted serial seconds below which a process pool costs more than it
    #: saves.  Each worker re-generates the source material on start-up
    #: (~1.3 s here), so a batch of a few short jobs is faster in-process.
    POOL_BREAK_EVEN_S = 45.0

    def __init__(self, cfg: ExperimentConfig, spec: MatrixSpec, output_dir: str,
                 sources: Optional[Sequence[FrameSource]] = None,
                 manifest: Optional[DatasetManifest] = None,
                 workers: int = 1) -> None:
        from avsec.experiments import build_sources

        self.cfg = cfg
        self.spec = spec
        self.output_dir = ensure_dir(output_dir)
        self.sources = list(sources) if sources is not None else build_sources(cfg)
        if manifest is None:
            from avsec.dataset import build_manifest

            manifest = build_manifest(self.sources, seed=cfg.seed)
        self.manifest = manifest
        self.workers = max(1, int(workers))
        self.jobs = build_jobs(cfg, spec, self.sources, manifest)
        self._by_name = {s.name: s for s in self.sources}
        self.state_path = os.path.join(self.output_dir, "matrix_state.json")
        self.state: Dict[str, Any] = self._load_state()
        self._write_provenance()

    def _write_provenance(self) -> None:
        """Everything needed to re-run this matrix, written before it starts.

        A result whose configuration, dataset and commit are not recorded next
        to it cannot be reproduced, so this is written at construction time -
        not at the end, where a crash would lose it.
        """
        from avsec.config import dump_config
        from avsec.utils import environment_record

        dump_config(self.cfg, os.path.join(self.output_dir, "config.yaml"))
        self.manifest.save(self.output_dir)
        env = environment_record()
        write_json(os.path.join(self.output_dir, "run_manifest.json"), {
            "run_id": self.cfg.run_identity(),
            "config_name": self.cfg.name,
            "commit": env.get("git_commit"),
            "environment": env,
            "config": self.cfg.to_dict(),
            "seeds": {"seed": self.cfg.seed,
                      "channel_seed": self.cfg.channel_seed_value,
                      "crypto_seed": self.cfg.crypto_seed_value,
                      "key_mode": self.cfg.key_mode},
            "plan": self.spec.to_dict(),
            "n_jobs_planned": len(self.jobs),
            "dataset": self.manifest.validate(),
            "workers": self.workers,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    # ------------------------------------------------------------ persistence
    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, encoding="utf-8") as fh:
                    st = json.load(fh)
                if st.get("config_id") == self.cfg.run_identity():
                    return st
                st["stale_reason"] = ("configuration changed; previous results are "
                                      "kept but not reused")
                st["done"] = {}
                return st
            except Exception:
                pass
        return {"config_id": self.cfg.run_identity(), "spec": self.spec.to_dict(),
                "done": {}, "started": time.time()}

    def _save_state(self) -> None:
        write_json(self.state_path, self.state)

    def pending(self) -> List[Job]:
        """Jobs still to run.  A job that finished - including one that legitimately
        exceeded capacity, which is a *result*, not a crash - is never repeated."""
        done = self.state.get("done", {})
        return [j for j in self.jobs
                if done.get(j.job_id, {}).get("status") not in ("ok", "capacity")]

    def completed(self) -> List[str]:
        return [k for k, v in self.state.get("done", {}).items()
                if v.get("status") in ("ok", "capacity")]

    # --------------------------------------------------------------- dry run
    def estimate(self, measure: bool = True, parallel_efficiency: float = 0.0
                 ) -> DryRun:
        """Predict wall time and output size before committing to the matrix.

        The cost is measured **per method**, because they differ by more than an
        order of magnitude (B0a ~18 ms/frame, P ~630 ms/frame on this machine);
        one probe scaled by a guess would be off by 10x.  The parallel speedup
        is also measured rather than assumed to equal the worker count - this
        pipeline is CPU-bound Python and does not scale linearly.
        """
        pending = self.pending()
        n_frames = sum(j.n_frames for j in pending)
        per_method: Dict[str, int] = {}
        per_channel: Dict[str, int] = {}
        frames_by_method: Dict[str, int] = {}
        for j in pending:
            per_method[j.method] = per_method.get(j.method, 0) + 1
            per_channel[j.channel.label] = per_channel.get(j.channel.label, 0) + 1
            frames_by_method[j.method] = frames_by_method.get(j.method, 0) + j.n_frames

        costs, source = self._method_costs(pending, measure)
        serial_s = sum(frames_by_method.get(m, 0) * costs.get(m, 0.35)
                       for m in frames_by_method)
        eff = parallel_efficiency if parallel_efficiency > 0 else _MEASURED_EFFICIENCY
        speedup = 1.0 if self.workers <= 1 else max(1.0, self.workers * eff)
        source += (f"; parallel speedup {speedup:.1f}x assumed for "
                   f"{self.workers} process workers (measured efficiency {eff:.2f}, "
                   "not the worker count)")
        return DryRun(n_jobs=len(pending), n_frames=n_frames,
                      est_wall_s=serial_s / speedup,
                      est_bytes=n_frames * self.BYTES_PER_FRAME_ROW,
                      per_method=per_method, per_channel=per_channel,
                      measured_from=source)

    def _method_costs(self, pending: Sequence[Job], measure: bool
                      ) -> Tuple[Dict[str, float], str]:
        methods = sorted({j.method for j in pending})
        key = self.cfg.run_identity()
        cached = _COST_CACHE.get(key, {})
        if not measure:
            return ({m: cached.get(m, 0.35) for m in methods},
                    "declared default 0.35 s/frame (no measurement requested)")
        measured: List[str] = []
        for m in methods:
            if m in cached:
                continue
            job = next(j for j in pending if j.method == m)
            probe = dataclasses.replace(job, n_frames=1)
            t0 = time.perf_counter()
            res = self._run_job(probe, store_units=False)
            dt = time.perf_counter() - t0
            cached[m] = dt if res.status == "ok" else 0.35
            measured.append(f"{m}={dt*1e3:.0f}ms")
        _COST_CACHE[key] = cached
        return ({m: cached[m] for m in methods},
                "measured one frame per method (" + ", ".join(measured) + ")"
                if measured else "cached per-method measurements")

    # ------------------------------------------------------------------- run
    def run(self, progress: Progress = None, limit: int = 0) -> Dict[str, Any]:
        pending = self.pending()
        if limit > 0:
            pending = pending[:limit]
        total = max(1, len(pending))
        results: List[JobResult] = []
        t0 = time.perf_counter()

        def _emit(done: int, job: Job, status: str) -> None:
            if progress:
                try:
                    progress(f"{job.method} / {job.clip} / {job.channel.label}",
                             done / total, {"job_id": job.job_id, "status": status})
                except Exception:
                    pass

        # Spinning up a process pool costs about a second per worker, because
        # each one rebuilds the source material.  For a small batch that is more
        # than the work itself, so the pool is used only when the predicted
        # serial time is worth paying for it.
        workers = self.workers
        if workers > 1:
            costs, _ = self._method_costs(pending, measure=False)
            serial_s = sum(j.n_frames * costs.get(j.method, 0.35) for j in pending)
            if serial_s < self.POOL_BREAK_EVEN_S:
                workers = 1
            else:
                workers = max(1, min(workers, (len(pending) + 1) // 2))

        if workers == 1:
            for i, job in enumerate(pending, 1):
                r = self._run_job(job)
                results.append(r)
                self._record(r)
                _emit(i, job, r.status)
        else:
            # Threads give no speedup here - the pipeline is CPU-bound Python
            # (Reed-Solomon, placement, per-cell demodulation), so the GIL
            # serialises them.  Measured on this machine: 1/2/4 thread workers
            # all took 13.5-14.6 s for the same 16 jobs.  Processes are used
            # instead, and the worker pool is warmed once per process.
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_pool_init,
                                     initargs=(self.cfg, self.spec)) as pool:
                futs = {pool.submit(_pool_run_job, j): j for j in pending}
                for i, fut in enumerate(as_completed(futs), 1):
                    job = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as exc:            # pragma: no cover
                        r = JobResult(job, "exception", f"{type(exc).__name__}: {exc}")
                    results.append(r)
                    self._record(r)
                    _emit(i, job, r.status)
        self._save_state()
        wall = time.perf_counter() - t0
        return self._finalise(results, wall)

    def _record(self, r: JobResult) -> None:
        """Persist a job's rows immediately, so a crash loses at most one job."""
        self.state.setdefault("done", {})[r.job.job_id] = {
            "status": r.status, "detail": r.detail, "wall_s": round(r.wall_s, 4),
            "n_frames": len(r.frames), "finished": time.time(),
        }
        for row in r.frames:
            append_jsonl(os.path.join(self.output_dir, "frames.jsonl"), row)
        for row in r.units:
            append_jsonl(os.path.join(self.output_dir, "units.jsonl"), row)
        if r.status != "ok":
            append_jsonl(os.path.join(self.output_dir, "failures.jsonl"), r.to_dict())
        append_jsonl(os.path.join(self.output_dir, "jobs.jsonl"), r.to_dict())
        if len(self.state["done"]) % 25 == 0:
            self._save_state()

    def _load_rows(self, name: str) -> List[Dict[str, Any]]:
        """Every row written for this matrix so far, later writes winning.

        Deduplication is by the row's own identity, so a job re-run after a
        crash replaces its rows instead of adding a second copy of them.
        """
        path = os.path.join(self.output_dir, name)
        if not os.path.exists(path):
            return []
        keyed: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        order: List[Tuple[Any, ...]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (row.get("job_id"), row.get("frame_id"), row.get("status"),
                       row.get("method"))
                if key not in keyed:
                    order.append(key)
                keyed[key] = row
        return [keyed[k] for k in order]

    # ------------------------------------------------------------- one job
    def _run_job(self, job: Job, store_units: bool = True) -> JobResult:
        from avsec.experiments import build_methods

        src = self._by_name.get(job.clip)
        if src is None:
            return JobResult(job, "skipped", "clip not in the loaded source set")

        # The channel point is applied through the ordinary configuration path,
        # so the method is built against exactly the channel it is measured on.
        cfg = dataclasses.replace(
            self.cfg, methods=(job.method,), channel_preset=job.channel.base,
            channel_overrides=dict(job.channel.overrides))
        timer = StageTimer()
        try:
            method = build_methods(cfg, timer)[job.method]
        except Exception as exc:
            return JobResult(job, "exception", f"build: {type(exc).__name__}: {exc}")

        # The channel realisation depends on the clip, the repetition and the
        # channel point - never on the method (defect F08), so every method
        # meets the same damage in the same raster slot.
        trace = ChannelTrace(seed=job.seed, scene=f"{job.clip}|{job.channel.label}",
                             repetition=job.repetition, profile=job.channel.label,
                             rasters_per_frame=self.cfg.budget.rasters_per_frame)

        res = JobResult(job, trace_id=trace.trace_id)
        t0 = time.perf_counter()
        method.reset()                       # once per clip: state stays continuous
        current_label = job.channel.label
        for fi in range(job.n_frames):
            frame = src.frames[fi]
            if job.schedule:
                point = job.channel_at(fi)
                if point.label != current_label:
                    # swap the impairment model, keep every bit of session state
                    method.channel.cfg = point.config()
                    current_label = point.label
            try:
                out = method.process(frame, fi, trace)
            except CapacityExceeded as exc:
                res.status, res.detail = "capacity", str(exc)
                break
            except Exception as exc:
                res.status = "exception"
                res.detail = f"{type(exc).__name__}: {exc}"
                res.traceback = traceback.format_exc()   # type: ignore[attr-defined]
                break
            row = out.metrics.to_row()
            row.update({
                "job_id": job.job_id, "clip": job.clip, "scene": job.scene,
                "repetition": job.repetition,
                "channel": current_label if job.schedule else job.channel.label,
                "channel_plan": job.channel.label,
                "channel_axis": job.channel.axis, "channel_value": job.channel.value,
                "trace_id": trace.trace_id, "provenance": src.provenance,
            })
            res.frames.append(row)
            if store_units:
                res.units.extend(_unit_rows(out, job, fi))
        res.wall_s = time.perf_counter() - t0
        return res

    # ---------------------------------------------------------- finalisation
    def _finalise(self, results: List[JobResult], wall: float) -> Dict[str, Any]:
        """Rebuild the tables from **everything persisted**, not just this call.

        A resumed matrix would otherwise publish a table containing only the
        jobs that happened to run in the last invocation.
        """
        frames = self._load_rows("frames.jsonl")
        units = self._load_rows("units.jsonl")
        failures = self._load_rows("failures.jsonl")
        jobs_rows = self._load_rows("jobs.jsonl")

        write_csv(os.path.join(self.output_dir, "frames.csv"), frames)
        if units:
            write_csv(os.path.join(self.output_dir, "units.csv"), units)
        write_csv(os.path.join(self.output_dir, "jobs.csv"), jobs_rows)
        if failures:
            write_csv(os.path.join(self.output_dir, "failures.csv"), failures)

        table = ResultTable()
        for row in failures:
            table.add(Observation(
                method=str(row.get("method", "")), scene=str(row.get("scene", "")),
                clip=str(row.get("clip", "")), repetition=int(row.get("repetition", 0)),
                frame=-1, status=_status_kind(str(row.get("status", "exception"))),
                detail=str(row.get("detail", ""))))
        for row in frames:
            table.add(Observation(
                method=str(row.get("method", "")), scene=str(row.get("scene", "")),
                clip=str(row.get("clip", "")), repetition=int(row.get("repetition", 0)),
                frame=int(row.get("frame_id", 0)), status="ok",
                metrics={k: float(v) for k, v in row.items()
                         if isinstance(v, (int, float, np.floating))
                         and not isinstance(v, bool)}))
        write_csv(os.path.join(self.output_dir, "observations.csv"), table.to_rows())
        self._write_channel_traces(jobs_rows)

        completeness = self.check_completeness()
        out = {
            "kind": "matrix",
            "spec": self.spec.to_dict(),
            "config_id": self.cfg.run_identity(),
            "n_jobs_this_call": len(results),
            "n_jobs_total": len(jobs_rows),
            "n_frames": len(frames),
            "wall_s_this_call": round(wall, 2),
            "failures": failures,
            "completeness": completeness,
            "channel_axes": sorted({j.channel.axis for j in self.jobs if j.channel.axis}),
        }
        write_json(os.path.join(self.output_dir, "matrix.json"), out)
        self._save_state()
        return out

    def _write_channel_traces(self, jobs_rows: Sequence[Dict[str, Any]]) -> None:
        """A reproducible description of every channel realisation used.

        The impairment arrays are not stored - they are megabytes per raster and
        fully determined by the trace identity plus the seed.  What is stored is
        everything needed to regenerate them exactly, plus which jobs shared a
        trace, which is what makes the comparison verifiably paired.
        """
        path = os.path.join(self.output_dir, "channel_traces.jsonl")
        if os.path.exists(path):
            os.remove(path)
        by_trace: Dict[str, List[Dict[str, Any]]] = {}
        for r in jobs_rows:
            by_trace.setdefault(str(r.get("trace_id", "")), []).append(r)
        by_label = {c.label: c for c in self.spec.channels}
        for trace_id, rows in sorted(by_trace.items()):
            if not trace_id:
                continue
            first = rows[0]
            point = by_label.get(str(first.get("channel", "")))
            frames = max(int(r.get("n_frames", 0) or 0) for r in rows)
            append_jsonl(path, {
                "trace_id": trace_id,
                "generator": "avsec.utils.experiment_rng (SHA-256 seeded PCG64)",
                "seed": first.get("config_id") and self.cfg.channel_seed_value,
                "scene_key": f"{first.get('clip')}|{first.get('channel')}",
                "repetition": first.get("repetition"),
                "channel": point.to_dict() if point else {"label": first.get("channel")},
                "impairments": point.config().describe() if point else {},
                "raster_index": {
                    "formula": "frame_id * rasters_per_frame + slot",
                    "rasters_per_frame": self.cfg.budget.rasters_per_frame,
                    "first": 0,
                    "last": max(0, frames * self.cfg.budget.rasters_per_frame - 1),
                },
                "shared_by_jobs": sorted({str(r.get("job_id")) for r in rows}),
                "methods_sharing_it": sorted({str(r.get("method")) for r in rows}),
                "note": "масиви спотворень не зберігаються: вони повністю "
                        "визначені trace_id і seed; наведеного достатньо, щоб "
                        "відтворити їх байт у байт",
            })

    def check_completeness(self) -> Dict[str, Any]:
        """Name the jobs that are missing instead of reporting a fraction."""
        done = self.state.get("done", {})
        missing = [j.to_dict() for j in self.jobs if j.job_id not in done]
        failed = [dict(v, job_id=k) for k, v in done.items()
                  if v.get("status") not in ("ok", "capacity")]
        return {
            "n_planned": len(self.jobs),
            "n_done": len(done),
            "n_missing": len(missing),
            "n_failed": len(failed),
            "complete": not missing and not failed,
            "missing": missing[:200],
            "failed": failed[:200],
            "truncated": len(missing) > 200 or len(failed) > 200,
        }


# ------------------------------------------------------------- process pool
#: Per-process state, built once by the pool initializer.  Rebuilding the
#: material for every job would cost more than the job itself.
_WORKER: Dict[str, Any] = {}


def _pool_init(cfg: ExperimentConfig, spec: MatrixSpec) -> None:  # pragma: no cover
    from avsec.experiments import build_sources

    sources = build_sources(cfg)
    _WORKER["cfg"] = cfg
    _WORKER["spec"] = spec
    _WORKER["by_name"] = {s.name: s for s in sources}


def _pool_run_job(job: Job) -> JobResult:  # pragma: no cover
    runner = MatrixRunner.__new__(MatrixRunner)
    runner.cfg = _WORKER["cfg"]
    runner.spec = _WORKER["spec"]
    runner._by_name = _WORKER["by_name"]
    return runner._run_job(job)


def _status_kind(status: str) -> str:
    return {"capacity": "capacity", "exception": "exception",
            "skipped": "skipped"}.get(status, status)


def _unit_rows(result: Any, job: Job, frame_id: int) -> List[Dict[str, Any]]:
    """Per-unit rows, when the method exposes them."""
    rows: List[Dict[str, Any]] = []
    counts = getattr(result.metrics, "status_counts", None) or {}
    for status, n in counts.items():
        rows.append({"job_id": job.job_id, "clip": job.clip, "scene": job.scene,
                     "method": job.method, "repetition": job.repetition,
                     "channel": job.channel.label, "frame_id": frame_id,
                     "status": status, "count": int(n)})
    return rows


__all__ = ["NUMERIC_AXES", "ChannelPoint", "Job", "JobResult", "MatrixSpec",
           "build_jobs", "DryRun", "MatrixRunner"]
