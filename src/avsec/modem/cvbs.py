"""Level B: monochrome composite baseband (CVBS) generator and receiver, 625/50.

Timing source
-------------
Line and field structure follow the 625/50 analogue system described in
ITU-R BT.470 / BT.1700: line period 64 us, line sync 4.7 us, front porch 1.5 us,
back porch 5.7 us, 52 us active line, 625 lines in two interlaced fields at
50 fields/s; sync tip -0.3 V, blanking 0 V, peak white +0.7 V.

Sampling follows the ITU-R BT.601 13.5 MHz structure for 625/50: 864 samples per
line, digital active line of 720 samples.  That window is slightly wider than
the 52 us analogue active line, which is the normal digitisation convention and
is exactly what a USB capture device delivers.

Declared simplifications (this is NOT a full standard implementation)
--------------------------------------------------------------------
* The vertical interval carries five equalising half-lines, five broad
  (field-sync) half-lines and five equalising half-lines, then blanked lines.
  Teletext, VITS and VITC lines are not generated.
* Field one and field two use the same vertical-interval waveform; the
  half-line offset that distinguishes them in the standard is applied to the
  *line phase* only.
* There is no colour subcarrier, no burst and no PAL phase alternation: this is
  a monochrome baseband model on purpose.
* No RF/FM link model.  Nothing here should be read as a property of a real
  radio channel.

Anything estimated by the receiver is estimated from the signal: sync edges,
field boundaries, blanking level and gain.  The profile constants below are the
only values it knows in advance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CVBSProfile:
    """625/50 monochrome baseband profile."""

    sample_rate_hz: float = 13.5e6
    line_samples: int = 864
    total_lines: int = 625
    fields_per_frame: int = 2
    active_samples: int = 720
    active_start: int = 132          # samples after the sync leading edge (BT.601 0H)
    sync_samples: int = 64           # 4.74 us
    front_porch_samples: int = 12
    sync_level: float = -0.3         # volts
    blank_level: float = 0.0
    white_level: float = 0.7
    first_active_line_f1: int = 23   # 1-based line numbers inside the frame
    first_active_line_f2: int = 336
    active_lines_per_field: int = 288
    equalising_pulse_samples: int = 32   # 2.37 us
    broad_pulse_samples: int = 368       # 27.3 us

    @property
    def line_period_s(self) -> float:
        return self.line_samples / self.sample_rate_hz

    @property
    def frame_period_s(self) -> float:
        return self.total_lines * self.line_period_s

    @property
    def active_lines(self) -> int:
        return self.active_lines_per_field * self.fields_per_frame

    @property
    def samples_per_frame(self) -> int:
        return self.line_samples * self.total_lines

    def describe(self) -> Dict[str, object]:
        return {
            "system": "625/50 monochrome baseband (CVBS luminance only)",
            "timing_reference": "ITU-R BT.470 / BT.1700 line and field structure",
            "sampling_reference": "ITU-R BT.601 13.5 MHz, 864 samples/line, "
                                  "720-sample digital active line",
            "sample_rate_hz": self.sample_rate_hz,
            "line_period_us": round(self.line_period_s * 1e6, 3),
            "frame_period_ms": round(self.frame_period_s * 1e3, 3),
            "active_lines": self.active_lines,
            "active_samples_per_line": self.active_samples,
            "levels_volts": {"sync": self.sync_level, "blanking": self.blank_level,
                             "white": self.white_level},
            "simplifications": [
                "vertical interval: 5 equalising + 5 broad + 5 equalising half-lines only",
                "no teletext / VITS / VITC lines",
                "no colour subcarrier, no burst, no PAL phase alternation",
                "no RF or FM link model",
            ],
        }


# ------------------------------------------------------------------ generator
class CVBSGenerator:
    """Render a 720x576 luminance raster as a 625/50 composite baseband signal."""

    def __init__(self, profile: Optional[CVBSProfile] = None) -> None:
        self.p = profile or CVBSProfile()

    def _blank_line(self) -> np.ndarray:
        p = self.p
        line = np.full(p.line_samples, p.blank_level, dtype=np.float32)
        line[: p.sync_samples] = p.sync_level
        return line

    def _half_line_pulses(self, n_half_lines: int, pulse_samples: int) -> np.ndarray:
        """``n_half_lines`` half-lines, each starting with a low pulse."""
        p = self.p
        half = p.line_samples // 2
        out = np.full(n_half_lines * half, p.blank_level, dtype=np.float32)
        for i in range(n_half_lines):
            out[i * half : i * half + pulse_samples] = p.sync_level
        return out

    def vertical_interval(self) -> np.ndarray:
        p = self.p
        return np.concatenate([
            self._half_line_pulses(5, p.equalising_pulse_samples),
            self._half_line_pulses(5, p.broad_pulse_samples),
            self._half_line_pulses(5, p.equalising_pulse_samples),
        ])

    def _active_line(self, row: np.ndarray) -> np.ndarray:
        p = self.p
        line = self._blank_line()
        v = p.blank_level + (row.astype(np.float32) / 255.0) * (p.white_level - p.blank_level)
        n = min(p.active_samples, v.size)
        line[p.active_start : p.active_start + n] = v[:n]
        return line

    def generate_frame(self, raster: np.ndarray) -> np.ndarray:
        """Interlaced frame: raster row ``r`` goes to field ``r % 2``, line ``r // 2``."""
        p = self.p
        if raster.shape[0] < p.active_lines:
            pad = np.full((p.active_lines - raster.shape[0], raster.shape[1]),
                          0, dtype=raster.dtype)
            raster = np.vstack([raster, pad])
        if raster.shape[1] < p.active_samples:
            raster = np.hstack([raster, np.zeros(
                (raster.shape[0], p.active_samples - raster.shape[1]), dtype=raster.dtype)])

        out: List[np.ndarray] = []
        for field in range(p.fields_per_frame):
            out.append(self.vertical_interval())
            blanked = (p.first_active_line_f1 - 1 - 8) if field == 0 else \
                      (p.first_active_line_f2 - p.first_active_line_f1 - 8)
            for _ in range(max(0, blanked)):
                out.append(self._blank_line())
            for li in range(p.active_lines_per_field):
                r = li * 2 + field
                out.append(self._active_line(raster[r, : p.active_samples]))
            out.append(self._blank_line())
        return np.concatenate(out).astype(np.float32)


# ------------------------------------------------------------------- channel
@dataclass
class CVBSChannelConfig:
    """Baseband path impairments applied to the composite signal itself."""

    impulse_response: Tuple[float, ...] = (1.0,)
    noise_sigma_v: float = 0.0
    gain: float = 1.0
    offset_v: float = 0.0
    nonlinearity: float = 0.0        # x -> x + a*x^2 on the 0..1 luma range
    timing_jitter_samples: float = 0.0
    dropout_rate_per_frame: float = 0.0
    dropout_len_us: float = 40.0
    clip_low_v: float = -0.45
    clip_high_v: float = 1.05

    def describe(self) -> Dict[str, object]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


class CVBSChannel:
    """Applies the level-B impairments in blocks so memory stays bounded."""

    def __init__(self, cfg: CVBSChannelConfig, profile: Optional[CVBSProfile] = None) -> None:
        self.cfg = cfg
        self.p = profile or CVBSProfile()

    def apply(self, signal: np.ndarray, rng: np.random.Generator,
              block_samples: int = 1 << 18) -> Tuple[np.ndarray, Dict[str, object]]:
        cfg = self.cfg
        x = np.asarray(signal, dtype=np.float32)
        h = np.asarray(cfg.impulse_response, dtype=np.float32)
        events: List[Dict[str, object]] = []

        out = np.empty_like(x)
        tail = np.zeros(max(0, h.size - 1), dtype=np.float32)
        for start in range(0, x.size, block_samples):
            blk = x[start : start + block_samples]
            if h.size > 1:
                ext = np.concatenate([tail, blk])
                conv = np.convolve(ext, h, mode="full")[: ext.size]
                y = conv[tail.size :]
                tail = blk[-(h.size - 1) :].copy()
            else:
                y = blk * float(h[0])
            out[start : start + blk.size] = y

        y = out
        if cfg.nonlinearity:
            u = (y - self.p.blank_level) / (self.p.white_level - self.p.blank_level)
            u = u + cfg.nonlinearity * u * u
            y = self.p.blank_level + u * (self.p.white_level - self.p.blank_level)
        y = y * cfg.gain + cfg.offset_v
        if cfg.noise_sigma_v > 0:
            y = y + rng.normal(0.0, cfg.noise_sigma_v, y.shape).astype(np.float32)
        if cfg.timing_jitter_samples > 0:
            n = y.size
            idx = np.arange(n, dtype=np.float64)
            phase = rng.normal(0.0, cfg.timing_jitter_samples, size=(n // self.p.line_samples) + 1)
            per_sample = np.repeat(phase, self.p.line_samples)[:n]
            y = np.interp(idx - per_sample, idx, y).astype(np.float32)
        if cfg.dropout_rate_per_frame > 0:
            n_drop = int(rng.poisson(cfg.dropout_rate_per_frame))
            length = int(cfg.dropout_len_us * 1e-6 * self.p.sample_rate_hz)
            for _ in range(n_drop):
                s0 = int(rng.integers(0, max(1, y.size - length)))
                y[s0 : s0 + length] = rng.uniform(self.p.blank_level, self.p.white_level)
                events.append({"type": "dropout", "start_sample": s0, "length": length})
        y = np.clip(y, cfg.clip_low_v, cfg.clip_high_v)
        return y.astype(np.float32), {"events": events}


# ------------------------------------------------------------------- receiver
@dataclass
class CVBSReceiveResult:
    raster: np.ndarray
    lines_recovered: int
    lines_expected: int
    field_starts: List[int]
    sync_edges: int
    estimated_blank_v: float
    estimated_white_v: float
    line_available: np.ndarray

    def summary(self) -> Dict[str, object]:
        return {
            "lines_recovered": self.lines_recovered,
            "lines_expected": self.lines_expected,
            "line_recovery_fraction": round(self.lines_recovered / max(self.lines_expected, 1), 4),
            "sync_edges_found": self.sync_edges,
            "fields_found": len(self.field_starts),
            "estimated_blank_v": round(self.estimated_blank_v, 4),
            "estimated_white_v": round(self.estimated_white_v, 4),
        }


class CVBSReceiver:
    """Sync separator, field detector and active-line sampler.

    Knows only the profile constants; the sync edge positions, the field
    boundaries and the amplitude reference are all measured from the signal.
    """

    def __init__(self, profile: Optional[CVBSProfile] = None) -> None:
        self.p = profile or CVBSProfile()
        self._last_widths = None

    # -- sync separation ---------------------------------------------------
    def _rough_level(self, x: np.ndarray) -> float:
        """First-pass slicer, safely between the sync tip and the blanking level."""
        lo = float(np.percentile(x, 0.5))
        mid = float(np.percentile(x, 50.0))
        return lo + 0.25 * (mid - lo)

    def _separated(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Low-pass the signal, then slice half-way between sync tip and blanking.

        Pass 1 uses a rough threshold to find candidate sync pulses; pass 2
        measures the sync tip and the back-porch blanking level from those
        pulses and re-slices exactly half-way between them, so the detected edge
        does not drift with the picture content.
        """
        k = 9
        kern = np.ones(k, dtype=np.float32) / k
        sm = np.convolve(x, kern, mode="same").astype(np.float32)
        thr = self._rough_level(sm)

        below = sm < thr
        starts = np.flatnonzero(below[1:] & ~below[:-1]) + 1
        p = self.p
        tips: List[float] = []
        blanks: List[float] = []
        for e in starts[: min(starts.size, 300)]:
            a, b = int(e) + 8, int(e) + p.sync_samples - 8
            c, d = int(e) + p.sync_samples + 10, int(e) + p.active_start - 8
            if b > a and b < sm.size:
                tips.append(float(np.mean(sm[a:b])))
            if d > c and d < sm.size:
                blanks.append(float(np.mean(sm[c:d])))
        if tips and blanks:
            tip = float(np.median(tips))
            blank = float(np.median(blanks))
            if blank - tip > 1e-3:
                thr = tip + 0.5 * (blank - tip)
        return sm, thr

    def _slice_level(self, x: np.ndarray) -> float:
        return self._separated(x)[1]

    def sync_edges(self, x: np.ndarray, min_width: Optional[int] = None) -> np.ndarray:
        """Falling edges of pulses that stay low for at least ``min_width`` samples."""
        sm, thr = self._separated(x)
        below = sm < thr
        starts = np.flatnonzero(below[1:] & ~below[:-1]) + 1
        ends = np.flatnonzero(~below[1:] & below[:-1]) + 1
        if starts.size == 0:
            return starts
        if ends.size and ends[0] < starts[0]:
            ends = ends[1:]
        n = min(starts.size, ends.size)
        starts, ends = starts[:n], ends[:n]
        w = ends - starts
        mw = min_width if min_width is not None else max(
            8, int(self.p.equalising_pulse_samples * 0.5))
        keep = w >= mw
        self._last_widths = w[keep]
        return starts[keep]

    def _pulse_widths(self, x: np.ndarray, edges: np.ndarray) -> np.ndarray:
        if getattr(self, "_last_widths", None) is not None and \
                len(self._last_widths) == len(edges):
            return np.asarray(self._last_widths)
        sm, thr = self._separated(x)
        below = sm < thr
        widths = np.zeros(edges.size, dtype=np.int64)
        for i, e in enumerate(edges):
            j = int(e)
            limit = min(x.size, int(e) + self.p.line_samples)
            while j < limit and below[j]:
                j += 1
            widths[i] = j - int(e)
        return widths

    def find_fields(self, x: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """Locate field starts by the broad (field-sync) pulses."""
        edges = self.sync_edges(x)
        if edges.size == 0:
            return [], edges
        widths = self._pulse_widths(x, edges)
        broad = widths > (self.p.broad_pulse_samples * 0.6)
        starts: List[int] = []
        prev = -10 ** 9
        for i in np.flatnonzero(broad):
            if int(edges[i]) - prev > self.p.line_samples * 4:
                starts.append(int(edges[i]))
            prev = int(edges[i])
        return starts, edges

    # -- amplitude ---------------------------------------------------------
    def estimate_levels(self, x: np.ndarray, edges: np.ndarray) -> Tuple[float, float]:
        """Blanking level from the back porch, sync tip from the pulse bottom."""
        p = self.p
        if edges.size < 8:
            return p.blank_level, p.white_level
        bp: List[float] = []
        st: List[float] = []
        for e in edges[: min(edges.size, 400)]:
            a = e + p.sync_samples + 8
            b = e + p.active_start - 8
            if b < x.size and b > a:
                bp.append(float(np.mean(x[a:b])))
            c = e + 8
            d = e + p.sync_samples - 8
            if d < x.size and d > c:
                st.append(float(np.mean(x[c:d])))
        if not bp or not st:
            return p.blank_level, p.white_level
        blank = float(np.median(bp))
        sync = float(np.median(st))
        span = blank - sync                      # nominal 0.3 V
        white = blank + span * ((p.white_level - p.blank_level) /
                                (p.blank_level - p.sync_level))
        return blank, white

    # -- main --------------------------------------------------------------
    def receive_frame(self, x: np.ndarray, raster_width: int = 720,
                      raster_height: int = 576) -> CVBSReceiveResult:
        p = self.p
        starts, edges = self.find_fields(x)
        blank, white = self.estimate_levels(x, edges)
        scale = 255.0 / max(white - blank, 1e-6)

        raster = np.zeros((raster_height, raster_width), dtype=np.uint8)
        available = np.zeros(raster_height, dtype=bool)
        recovered = 0

        edge_list = edges.tolist()
        for field, fs in enumerate(starts[: p.fields_per_frame]):
            after = [e for e in edge_list if e >= fs]
            if len(after) < 20:
                continue
            # leave the vertical interval: find the first edge that starts a run of
            # full-line spacing (measured, not assumed)
            k = 0
            for i in range(len(after) - 2):
                g1 = after[i + 1] - after[i]
                g2 = after[i + 2] - after[i + 1]
                if g1 > 0.75 * p.line_samples and g2 > 0.75 * p.line_samples:
                    k = i          # first edge whose spacing is a full line
                    break
            after = np.asarray(after[k:], dtype=np.int64)
            blanked = (p.first_active_line_f1 - 1 - 8) if field == 0 else \
                      (p.first_active_line_f2 - p.first_active_line_f1 - 8)
            if after.size <= blanked:
                continue
            # Flywheel line timing: predict the next line one line period ahead and
            # snap to the nearest measured sync edge inside a bounded window.  A
            # missing edge no longer shifts every following line.
            pos = int(after[max(0, blanked)])
            tol = int(0.15 * p.line_samples)
            for li in range(p.active_lines_per_field):
                j = int(np.searchsorted(after, pos))
                cand = [after[t] for t in (j - 1, j) if 0 <= t < after.size]
                snapped = None
                if cand:
                    e_best = min(cand, key=lambda e: abs(int(e) - pos))
                    if abs(int(e_best) - pos) <= tol:
                        snapped = int(e_best)
                line_start = snapped if snapped is not None else pos
                a = line_start + p.active_start
                b = a + min(raster_width, p.active_samples)
                r = li * 2 + field
                if r >= raster_height or b > x.size:
                    break
                seg = (x[a:b] - blank) * scale
                raster[r, : b - a] = np.clip(np.round(seg), 0, 255).astype(np.uint8)
                if snapped is not None:
                    available[r] = True
                    recovered += 1
                pos = line_start + p.line_samples

        return CVBSReceiveResult(
            raster=raster, lines_recovered=recovered, lines_expected=p.active_lines,
            field_starts=starts, sync_edges=int(edges.size),
            estimated_blank_v=blank, estimated_white_v=white, line_available=available,
        )


CVBS_PRESETS: Dict[str, CVBSChannelConfig] = {
    "clean": CVBSChannelConfig(),
    "mild": CVBSChannelConfig(
        impulse_response=(0.15, 0.70, 0.15), noise_sigma_v=0.004, gain=0.96,
        offset_v=0.01, timing_jitter_samples=0.3),
    "moderate": CVBSChannelConfig(
        impulse_response=(0.10, 0.22, 0.36, 0.22, 0.10), noise_sigma_v=0.012, gain=0.88,
        offset_v=0.03, nonlinearity=0.06, timing_jitter_samples=1.0,
        dropout_rate_per_frame=1.0, dropout_len_us=60.0),
    "harsh": CVBSChannelConfig(
        impulse_response=(0.08, 0.14, 0.20, 0.20, 0.20, 0.10, 0.08),
        noise_sigma_v=0.03, gain=0.78, offset_v=0.05, nonlinearity=0.12,
        timing_jitter_samples=2.5, dropout_rate_per_frame=4.0, dropout_len_us=250.0),
}


def cvbs_preset(name: str) -> CVBSChannelConfig:
    if name not in CVBS_PRESETS:
        raise KeyError(f"unknown CVBS preset {name!r}; have {sorted(CVBS_PRESETS)}")
    import copy

    return copy.deepcopy(CVBS_PRESETS[name])


__all__ = [
    "CVBSProfile", "CVBSGenerator", "CVBSChannelConfig", "CVBSChannel",
    "CVBSReceiver", "CVBSReceiveResult", "CVBS_PRESETS", "cvbs_preset",
]
