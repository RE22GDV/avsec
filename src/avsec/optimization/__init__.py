"""Joint parameter selection under one shared, explicitly defined budget.

Optimisation problem
--------------------
Minimise the expected distortion of the picture reconstructed **from data that
was verified before the display deadline**, subject to hard constraints that are
identical for every method under comparison:

* channel occupancy - at most ``rasters_per_frame`` transmitted rasters per
  source frame, on the same raster geometry, the same number of levels and the
  same amplitude window;
* latency - the placement accumulation window and the resulting virtual
  end-to-end latency must stay under the configured limits;
* memory - the receiver's reassembly buffer must stay under its limit.

The search is an exhaustive scan of a small finite space with early pruning of
inadmissible configurations.  No optimiser more complex than that is used
because nothing has yet shown a measurable benefit from one.

Data discipline
---------------
Parameters are fitted on the *calibration* split, the winner among finalists is
chosen on the *validation* split, and the *test* split is touched only after the
configuration is frozen.  :class:`Tuner` refuses to score on the test split.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from avsec import budget as budget_mod
from avsec import evaluation as ev
from avsec.channel import ChannelTrace, RasterChannelConfig
from avsec.crypto import MasterSecret
from avsec.fec import FECConfig
from avsec.crypto import TAG_LEN
from avsec.framing import HEADER_LEN
from avsec.interleaving import (
    Interleaver,
    InterleaverConfig,
    PlacementError,
    bawp_admissible,
    burst_rows_from_lines,
    worst_case_codeword_damage,
)
from avsec.modem import ModemConfig
from avsec.source_coding import SourceCodingConfig, SourceCodingError
from avsec.sources import FrameSource
from avsec.transmitter import CapacityExceeded, TransportConfig
from avsec.utils import experiment_rng


@dataclass
class SharedBudget:
    """The budget every compared method must live inside.  Identical by design.

    Fixed for everyone: the raster geometry, the active window, the amplitude
    window (hence peak and mean signal power), the raster rate, the number of
    rasters per source frame, the latency ceiling and the memory ceiling.

    Chosen per configuration: the modulation order and the symbol cell size.
    Those change how the same time / bandwidth / amplitude budget is used, not
    how much of it is used, so the comparison stays fair.
    """

    rasters_per_frame: int = 3
    raster_rate_hz: float = 25.0
    source_fps: float = 8.333             # 25 / 3
    max_accumulation_rows: int = 300      # placement window, in symbol rows
    max_virtual_latency_s: float = 0.80
    max_receiver_buffer_kb: float = 512.0
    raster_width: int = 720
    raster_height: int = 576
    active_x0: int = 24
    active_x1: int = 696
    active_y0: int = 8
    active_y1: int = 568
    level_low: int = 40
    level_high: int = 216

    def modem(self, levels: int = 4, symbol_width: int = 8,
              symbol_height: int = 2) -> ModemConfig:
        return ModemConfig(
            raster_width=self.raster_width, raster_height=self.raster_height,
            active_x0=self.active_x0, active_x1=self.active_x1,
            active_y0=self.active_y0, active_y1=self.active_y1,
            levels=levels, symbol_width=symbol_width, symbol_height=symbol_height,
            level_low=self.level_low, level_high=self.level_high,
        )

    @property
    def reference_modem(self) -> ModemConfig:
        """Geometry-only reference, used by the analog baselines."""
        return self.modem()

    def describe(self) -> Dict[str, Any]:
        return {
            "rasters_per_frame": self.rasters_per_frame,
            "raster_rate_hz": self.raster_rate_hz,
            "source_fps": self.source_fps,
            "effective_display_fps": self.raster_rate_hz / max(self.rasters_per_frame, 1),
            "max_accumulation_rows": self.max_accumulation_rows,
            "max_virtual_latency_s": self.max_virtual_latency_s,
            "max_receiver_buffer_kb": self.max_receiver_buffer_kb,
            "raster": f"{self.raster_width}x{self.raster_height}",
            "active_window": [self.active_x0, self.active_y0, self.active_x1,
                              self.active_y1],
            "amplitude_window": [self.level_low, self.level_high],
            "note": "однакові геометрія растру, частота растрів і амплітудне вікно "
                    "для всіх методів; порядок модуляції та розмір комірки символу "
                    "входять до простору пошуку, тому жодна схема не отримує "
                    "непомітно додаткового часу, смуги чи потужності сигналу",
        }


@dataclass
class ParameterSpace:
    """The finite grid searched by :class:`Tuner`."""

    stripe_height: Sequence[int] = (8, 16, 24, 32)
    n_descriptions: Sequence[int] = (1, 2, 4)
    codec: Sequence[str] = ("dct",)
    quality: Sequence[int] = (8, 12, 18, 25, 35)
    max_unit_payload: Sequence[int] = (192, 384, 640, 896)
    fec_nsym: Sequence[int] = (32, 64, 96, 128)
    modulation: Sequence[Tuple[int, int, int]] = (   # (levels, cell_width, cell_height)
        (2, 4, 2), (4, 4, 2), (4, 6, 2), (4, 8, 4),
    )
    interleaver: Sequence[Tuple[str, int, int]] = (
        ("sequential", 1, 0),
        ("block", 32, 0),
        ("block", 128, 0),
        ("block", 278, 0),
        ("bawp", 0, 8),
        ("bawp", 0, 16),
    )

    def size(self) -> int:
        return (len(self.stripe_height) * len(self.n_descriptions) * len(self.codec)
                * len(self.quality) * len(self.max_unit_payload) * len(self.fec_nsym)
                * len(self.modulation) * len(self.interleaver))

    def restrict(self, **kwargs: Sequence[Any]) -> "ParameterSpace":
        return replace(self, **kwargs)


@dataclass
class Candidate:
    source: SourceCodingConfig
    transport: TransportConfig
    label: str

    def key(self) -> str:
        s, t = self.source, self.transport
        il, m = t.interleaver, t.modem
        return (f"sh{s.stripe_height}_d{s.n_descriptions}_{s.codec}q{s.quality}"
                f"_u{s.max_unit_payload}_n{t.fec_payload.nsym}"
                f"_m{m.levels}c{m.symbol_width}x{m.symbol_height}"
                f"_{il.scheme}{il.depth if il.scheme == 'block' else il.burst_rows}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label, "key": self.key(),
            "source_coding": self.source.describe(),
            "transport": self.transport.describe(),
        }


@dataclass
class ScoredCandidate:
    candidate: Candidate
    admissible: bool
    reason: str = ""
    psnr_full: float = float("nan")
    ssim_full: float = float("nan")
    coverage: float = 0.0
    rasters_per_frame: float = 0.0
    accumulation_rows: int = 0
    virtual_latency_s: float = 0.0
    payload_efficiency: float = 0.0
    units_per_raster: int = 0
    n_frames: int = 0
    guarantee_satisfied: bool = False
    guarantee_note: str = ""
    fingerprint: str = ""

    def to_row(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "candidate"}
        d["key"] = self.candidate.key()
        d["label"] = self.candidate.label
        return d


def build_candidate(
    shared: SharedBudget, stripe_height: int, n_desc: int, codec: str, quality: int,
    unit_payload: int, fec_nsym: int, interleaver: Tuple[str, int, int],
    modulation: Tuple[int, int, int] = (4, 4, 2),
    label: str = "", header_nsym: int = 48,
) -> Candidate:
    scheme, depth, burst_rows = interleaver
    levels, cell_w, cell_h = modulation
    il = InterleaverConfig(
        scheme=scheme,
        depth=max(1, depth),
        window_rows=0,
        burst_rows=max(1, burst_rows) if scheme == "bawp" else 4,
    )
    src = SourceCodingConfig(stripe_height=stripe_height, n_descriptions=n_desc,
                             codec=codec, quality=quality, max_unit_payload=unit_payload)
    tr = TransportConfig(
        modem=shared.modem(levels, cell_w, cell_h),
        fec_payload=FECConfig(k=max(1, 255 - fec_nsym), nsym=fec_nsym),
        fec_header=FECConfig(k=52, nsym=header_nsym),
        interleaver=il,
        raster_rate_hz=shared.raster_rate_hz,
        max_rasters_per_frame=shared.rasters_per_frame,
    )
    return Candidate(src, tr, label or "candidate")


def effective_fingerprint(cand: Candidate) -> str:
    """Hash of everything that actually changes the transmitted signal.

    Defect F13: at the current geometry, ``burst_rows`` of 8, 12 and 24 all
    produce the same number of BAWP bands and therefore the *same placement
    table*.  Searching them as three candidates wastes the budget and makes the
    search look larger than it is.  The fingerprint covers the codec, the
    segment geometry, the protected unit, the FEC blocking, the modem and the
    realised placement table, so equivalent configurations collapse to one and
    the rest are recorded as aliases.
    """
    import hashlib

    s, t = cand.source, cand.transport
    m, il_cfg = t.modem, t.interleaver
    h = hashlib.sha256()
    h.update(repr((
        s.stripe_height, s.n_descriptions, s.codec, s.quality, s.max_unit_payload,
        t.fec_payload.k, t.fec_payload.nsym, t.fec_header.k, t.fec_header.nsym,
        m.levels, m.symbol_width, m.symbol_height, m.raster_width, m.raster_height,
        m.active_x0, m.active_x1, m.active_y0, m.active_y1, m.level_low, m.level_high,
        t.profile_id, t.raster_rate_hz, t.send_filler, t.max_rasters_per_frame,
    )).encode())
    bud = budget_mod.compute_budget(m, t.fec_payload, t.fec_header,
                                    s.max_unit_payload, t.raster_rate_hz)
    il = Interleaver(il_cfg, m.n_data_rows, m.n_data_cols)
    n = il.max_units(bud.unit_symbols, s.n_descriptions, bud.units_per_raster)
    h.update(repr(n).encode())
    if n >= 1:
        try:
            tables = il.place([bud.unit_symbols] * n,
                              [i % max(1, s.n_descriptions) for i in range(n)],
                              s.n_descriptions)
            for tab in tables:
                h.update(np.asarray(tab, dtype=np.int64).tobytes())
        except PlacementError as exc:
            h.update(f"unplaceable:{exc}".encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ pruning
def placement_guarantee(cand: Candidate, n_descriptions: int,
                        burst_lines: Optional[int] = None) -> Tuple[bool, str]:
    """Is the analytic burst guarantee actually established for this candidate?

    Defect F11: the old code computed a predicate and then discarded it with a
    bare ``pass``, and it mixed up raster lines, symbol rows and RS byte symbols.
    This version converts units explicitly and checks **every real RS codeword**
    of a transport unit - the header block and each payload block, including the
    shortened last one - against its own parity budget, using the exact
    worst-case damage over all burst positions rather than a bound.

    Erasures whose positions the receiver can flag cost ``s <= nsym``; symbol
    errors at unknown positions cost ``2e <= nsym``.  The conservative test used
    here is the error case, because the receiver cannot always localise damage.
    """
    src, tr = cand.source, cand.transport
    modem = tr.modem
    il_cfg = tr.interleaver
    bud = budget_mod.compute_budget(modem, tr.fec_payload, tr.fec_header,
                                    src.max_unit_payload, tr.raster_rate_hz)
    il = Interleaver(il_cfg, modem.n_data_rows, modem.n_data_cols)
    n_units = il.max_units(bud.unit_symbols, n_descriptions, bud.units_per_raster)
    if n_units < 1:
        return False, "no unit fits the raster"

    # raster lines -> symbol rows, once, in one place
    lines = int(burst_lines) if burst_lines is not None else \
        il_cfg.burst_rows * modem.symbol_height
    b_rows = burst_rows_from_lines(lines, modem.symbol_height)

    try:
        placements = il.place([bud.unit_symbols] * n_units,
                              [i % max(1, n_descriptions) for i in range(n_units)],
                              n_descriptions)
    except PlacementError as exc:
        return False, str(exc)

    if il_cfg.scheme == "bawp" and il.last_info is not None and il.last_info.spilled:
        return False, ("units spilled across description bands, so the isolation "
                       "bound does not follow (F12)")

    # byte offsets of every real RS codeword inside one transport unit
    sym_per_byte = 8 // modem.bits_per_symbol
    blocks: List[Tuple[int, int, int]] = []          # (start_byte, length, nsym)
    off = 0
    for _, _, es, elen in tr.fec_header.block_layout(HEADER_LEN):
        blocks.append((off + es, elen, tr.fec_header.nsym))
    off += tr.fec_header.encoded_len(HEADER_LEN)
    pay_len = src.max_unit_payload + TAG_LEN
    for _, _, es, elen in tr.fec_payload.block_layout(pay_len):
        blocks.append((off + es, elen, tr.fec_payload.nsym))

    for cells in placements:
        for start, length, nsym in blocks:
            lo, hi = start * sym_per_byte, (start + length) * sym_per_byte
            if hi > cells.size:
                return False, "codeword layout does not fit the placed unit"
            dmg_symbols = worst_case_codeword_damage(
                cells[lo:hi], modem.n_data_cols, b_rows, modem.n_data_rows)
            dmg_bytes = int(np.ceil(dmg_symbols / sym_per_byte))
            if 2 * dmg_bytes > nsym:
                return False, (
                    f"a {lines}-line burst ({b_rows} symbol rows) can destroy "
                    f"{dmg_bytes} bytes of a {length}-byte codeword with nsym={nsym}; "
                    f"2*{dmg_bytes} > {nsym}")
    return True, ""


def static_admissibility(cand: Candidate, shared: SharedBudget,
                         frame_h: int, require_guarantee: bool = False
                         ) -> Tuple[bool, str]:
    """Cheap checks that do not need any frame to be transmitted.

    ``require_guarantee=True`` selects the *guaranteed* family: a configuration
    whose analytic burst guarantee is not established is rejected.  The default
    research family keeps such configurations but records the fact.
    """
    src, tr = cand.source, cand.transport
    try:
        src.validate()
        tr.interleaver.validate()
        tr.fec_payload.validate()
        tr.fec_header.validate()
    except Exception as exc:
        return False, f"invalid config: {exc}"

    bud = budget_mod.compute_budget(tr.modem, tr.fec_payload, tr.fec_header,
                                    src.max_unit_payload, tr.raster_rate_hz)
    if bud.units_per_raster < 1:
        return False, "one unit does not fit into a raster"

    il = Interleaver(tr.interleaver, tr.modem.n_data_rows, tr.modem.n_data_cols)
    if il.accumulation_rows > shared.max_accumulation_rows:
        return False, (f"accumulation window {il.accumulation_rows} rows exceeds the "
                       f"shared limit {shared.max_accumulation_rows}")
    placeable = il.max_units(bud.unit_symbols, src.n_descriptions, bud.units_per_raster)
    if placeable < 1:
        return False, "the placement scheme cannot fit a single unit"
    if placeable < src.n_descriptions:
        return False, (f"only {placeable} slots per raster for {src.n_descriptions} "
                       "descriptions")

    ok_guarantee, why_guarantee = placement_guarantee(cand, src.n_descriptions)
    if require_guarantee and not ok_guarantee:
        return False, f"guarantee not established: {why_guarantee}"

    lat = budget_mod.compute_latency(
        tr.modem, il.accumulation_rows, shared.rasters_per_frame,
        1.0 / shared.source_fps, src.stripe_height, frame_h, shared.raster_rate_hz)
    if lat.total_s > shared.max_virtual_latency_s:
        return False, (f"virtual latency {lat.total_s*1e3:.0f} ms exceeds the shared "
                       f"limit {shared.max_virtual_latency_s*1e3:.0f} ms")

    buf_kb = (placeable * shared.rasters_per_frame * bud.unit_wire_bytes
              + tr.modem.raster_height * tr.modem.raster_width) / 1024.0
    if buf_kb > shared.max_receiver_buffer_kb:
        return False, f"receiver buffer {buf_kb:.0f} kB exceeds the shared limit"
    return True, ""


# -------------------------------------------------------------------- tuner
class Tuner:
    """Exhaustive search with pruning on a calibration split."""

    def __init__(self, shared: SharedBudget, master: MasterSecret,
                 channel_cfg: RasterChannelConfig, frame_h: int, frame_w: int,
                 seed: int = 20240909, require_guarantee: bool = False) -> None:
        self.require_guarantee = bool(require_guarantee)
        self.shared = shared
        self.master = master
        self.channel_cfg = channel_cfg
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.seed = seed

    # -- scoring ----------------------------------------------------------
    def score(self, cand: Candidate, sources: Sequence[FrameSource],
              max_frames: int = 4, split: str = "calibration") -> ScoredCandidate:
        if split == "test":
            raise RuntimeError(
                "the test split must not be used for parameter selection; "
                "freeze the configuration first and run it once through the "
                "experiment runner")
        ok, why = static_admissibility(cand, self.shared, self.frame_h,
                                       require_guarantee=self.require_guarantee)
        if not ok:
            return ScoredCandidate(cand, False, why)
        guaranteed, gnote = placement_guarantee(cand, cand.source.n_descriptions)

        from avsec.baselines import DigitalMethod

        try:
            method = DigitalMethod(cand.label or cand.key(), cand.source, cand.transport,
                                   self.channel_cfg, self.master, self.frame_h,
                                   self.frame_w)
        except CapacityExceeded as exc:
            return ScoredCandidate(cand, False, f"setup: {exc}")

        psnrs: List[float] = []
        ssims: List[float] = []
        covs: List[float] = []
        rasters: List[int] = []
        n = 0
        for si, src in enumerate(sources):
            method.reset()
            for fi, frame in enumerate(src.frames[:max_frames]):
                trace = ChannelTrace(seed=self.seed, scene=src.name,
                                     repetition=0, profile="tune",
                                     rasters_per_frame=self.shared.rasters_per_frame)
                try:
                    res = method.process(frame, fi, trace)
                except CapacityExceeded as exc:
                    return ScoredCandidate(cand, False,
                                           f"does not fit the shared budget: {exc}")
                except SourceCodingError as exc:
                    return ScoredCandidate(cand, False, f"source coding: {exc}")
                psnrs.append(res.metrics.psnr_full)
                ssims.append(res.metrics.ssim_full)
                covs.append(res.metrics.coverage)
                rasters.append(res.metrics.rasters)
                n += 1
        il = Interleaver(cand.transport.interleaver, cand.transport.modem.n_data_rows,
                         cand.transport.modem.n_data_cols)
        lat = budget_mod.compute_latency(
            cand.transport.modem, il.accumulation_rows, int(np.max(rasters or [1])),
            1.0 / self.shared.source_fps, cand.source.stripe_height, self.frame_h,
            self.shared.raster_rate_hz)
        return ScoredCandidate(
            cand, True, "",
            psnr_full=float(np.mean(psnrs)) if psnrs else float("nan"),
            ssim_full=float(np.mean(ssims)) if ssims else float("nan"),
            coverage=float(np.mean(covs)) if covs else 0.0,
            rasters_per_frame=float(np.mean(rasters)) if rasters else 0.0,
            accumulation_rows=il.accumulation_rows,
            virtual_latency_s=lat.total_s,
            payload_efficiency=method.tx.budget.efficiency,
            units_per_raster=method.tx.budget.units_per_raster,
            n_frames=n,
            guarantee_satisfied=guaranteed, guarantee_note=gnote,
            fingerprint=effective_fingerprint(cand),
        )

    # -- search -----------------------------------------------------------
    def search(self, space: ParameterSpace, sources: Sequence[FrameSource],
               label: str = "search", max_frames: int = 2,
               progress: Optional[Callable[[int, int, ScoredCandidate], None]] = None,
               max_candidates: Optional[int] = None) -> List[ScoredCandidate]:
        combos = list(itertools.product(
            space.stripe_height, space.n_descriptions, space.codec, space.quality,
            space.max_unit_payload, space.fec_nsym, space.modulation, space.interleaver))
        if max_candidates is not None:
            combos = combos[:max_candidates]
        out: List[ScoredCandidate] = []
        seen: Dict[str, ScoredCandidate] = {}
        self.aliases: Dict[str, List[str]] = {}
        for i, (sh, nd, codec, q, up, nsym, mo, il) in enumerate(combos):
            cand = build_candidate(self.shared, sh, nd, codec, q, up, nsym, il, mo, label)
            fp = effective_fingerprint(cand)
            twin = seen.get(fp)
            if twin is not None:
                # identical realised configuration: score it once, record the rest
                # as aliases instead of spending the search budget again (F13)
                self.aliases.setdefault(twin.candidate.key(), []).append(cand.key())
                sc = replace(twin, candidate=cand)
                out.append(sc)
                if progress:
                    progress(i + 1, len(combos), sc)
                continue
            sc = self.score(cand, sources, max_frames)
            sc.fingerprint = fp
            seen[fp] = sc
            out.append(sc)
            if progress:
                progress(i + 1, len(combos), sc)
        return out

    @staticmethod
    def distinct(scored: Sequence[ScoredCandidate]) -> List[ScoredCandidate]:
        """One entry per *realised* configuration, aliases removed."""
        out, seen = [], set()
        for s in scored:
            if s.fingerprint and s.fingerprint in seen:
                continue
            seen.add(s.fingerprint)
            out.append(s)
        return out

    @staticmethod
    def best(scored: Sequence[ScoredCandidate], metric: str = "psnr_full",
             require_coverage: float = 0.0) -> Optional[ScoredCandidate]:
        pool = [s for s in scored if s.admissible and s.coverage >= require_coverage
                and np.isfinite(getattr(s, metric))]
        if not pool:
            return None
        return max(pool, key=lambda s: getattr(s, metric))

    @staticmethod
    def estimate_cost(space: ParameterSpace, sources: Sequence[FrameSource],
                      max_frames: int, seconds_per_frame: float = 0.35
                      ) -> Dict[str, Any]:
        """Rough cost estimate, printed before any long search starts."""
        n_frames = sum(min(len(s), max_frames) for s in sources)
        n = space.size()
        return {
            "candidates": n,
            "frames_per_candidate": n_frames,
            "estimated_seconds": round(n * n_frames * seconds_per_frame, 1),
            "estimated_minutes": round(n * n_frames * seconds_per_frame / 60.0, 1),
        }


def split_sources(sources: Sequence[FrameSource], seed: int = 7,
                  fractions: Tuple[float, float, float] = (0.4, 0.3, 0.3)
                  ) -> Dict[str, List[FrameSource]]:
    """Split *whole sequences* (never frames of one sequence) into three parts."""
    idx = np.arange(len(sources))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n = len(sources)
    n_cal = max(1, int(round(fractions[0] * n)))
    n_val = max(1, int(round(fractions[1] * n))) if n - n_cal > 1 else 0
    cal = [sources[i] for i in idx[:n_cal]]
    val = [sources[i] for i in idx[n_cal : n_cal + n_val]]
    test = [sources[i] for i in idx[n_cal + n_val :]]
    if not test:
        test = val or cal
    return {"calibration": cal, "validation": val or cal, "test": test}


__all__ = [
    "SharedBudget", "ParameterSpace", "Candidate", "ScoredCandidate",
    "build_candidate", "static_admissibility", "Tuner", "split_sources",
    "placement_guarantee", "effective_fingerprint",
]
