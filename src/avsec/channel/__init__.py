"""Channel models, split into two clearly separated levels.

Level A - :class:`RasterChannel`
    A model of the raster *as it comes out of a video capture device*: horizontal
    band limitation, additive noise, gain/offset, clipping, resampling and
    sub-pixel shift, slow and per-line horizontal jitter, damaged or missing
    runs of lines, region and frame dropouts, and reordering/duplication of
    received frames.  Fast enough for parameter search.  Adding noise to a PNG
    is *not* a CVBS simulation and this class never claims to be one.

Level B - :mod:`avsec.channel.cvbs` (see :class:`avsec.modem.cvbs.CVBSProfile`)
    A monochrome composite baseband generator and receiver with line timing,
    sync pulses, blanking and field structure, plus a path impulse response.

Every impairment returns a ``truth`` record.  That record is for the
*evaluator* only; the receiver never sees it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RasterChannelConfig:
    """Level A impairments.  All defaults are 'no impairment'."""

    # amplitude
    gain: float = 1.0
    offset: float = 0.0
    noise_sigma: float = 0.0            # additive Gaussian, in luma codes
    clip_low: int = 0
    clip_high: int = 255

    # bandwidth
    lowpass_taps: int = 1               # horizontal moving-average length (1 = off)
    lowpass_kind: str = "boxcar"        # 'boxcar' | 'gaussian'
    lowpass_sigma: float = 1.0

    # geometry
    shift_x: float = 0.0                # constant sub-pixel horizontal shift
    shift_y: float = 0.0
    scale_x: float = 1.0                # horizontal resampling factor
    line_jitter_sigma: float = 0.0      # per-line random horizontal shift, px
    slow_drift_px: float = 0.0          # slow sinusoidal horizontal drift, px
    slow_drift_period_lines: float = 200.0

    # burst damage
    burst_rate_per_frame: float = 0.0   # expected number of line bursts per frame
    burst_len_lines: int = 8            # mean burst height in raster lines
    burst_len_jitter: int = 4
    burst_mode: str = "noise"           # 'noise' | 'blank' | 'hold'
    region_dropout_rate: float = 0.0    # expected number of rectangular dropouts
    region_dropout_size: Tuple[int, int] = (32, 96)

    # sequence level
    frame_drop_prob: float = 0.0
    frame_duplicate_prob: float = 0.0
    frame_swap_prob: float = 0.0

    def describe(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


@dataclass
class ChannelTruth:
    """Ground truth of what the channel did - visible to the evaluator only."""

    damaged_lines: np.ndarray                 # bool per raster line
    damaged_pixels: np.ndarray                # bool per pixel
    applied_gain: float = 1.0
    applied_offset: float = 0.0
    applied_shift_x: float = 0.0
    applied_shift_y: float = 0.0
    dropped: bool = False
    duplicated: bool = False
    events: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        return {
            "damaged_line_fraction": float(self.damaged_lines.mean()),
            "damaged_pixel_fraction": float(self.damaged_pixels.mean()),
            "gain": self.applied_gain, "offset": self.applied_offset,
            "shift_x": self.applied_shift_x, "shift_y": self.applied_shift_y,
            "dropped": float(self.dropped), "duplicated": float(self.duplicated),
        }


class RasterChannel:
    """Level A: impairments applied to a raster of luminance codes."""

    def __init__(self, cfg: RasterChannelConfig) -> None:
        self.cfg = cfg

    # -- helpers ----------------------------------------------------------
    def _lowpass(self, img: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        if cfg.lowpass_kind == "gaussian" and cfg.lowpass_sigma > 0 and cfg.lowpass_taps > 1:
            from scipy import ndimage

            return ndimage.gaussian_filter1d(img, cfg.lowpass_sigma, axis=1, mode="nearest")
        if cfg.lowpass_taps > 1:
            k = np.ones(cfg.lowpass_taps) / cfg.lowpass_taps
            pad = cfg.lowpass_taps // 2
            padded = np.pad(img, ((0, 0), (pad, pad)), mode="edge")
            out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, padded)
            return out[:, : img.shape[1]]
        return img

    def _geometry(self, img: np.ndarray, rng: np.random.Generator,
                  truth: ChannelTruth) -> np.ndarray:
        cfg = self.cfg
        h, w = img.shape
        xs = np.arange(w, dtype=np.float64)

        per_line = np.zeros(h)
        if cfg.line_jitter_sigma > 0:
            per_line += rng.normal(0, cfg.line_jitter_sigma, h)
        if cfg.slow_drift_px > 0:
            per_line += cfg.slow_drift_px * np.sin(
                2 * np.pi * np.arange(h) / max(cfg.slow_drift_period_lines, 1.0)
            )
        total_dx = cfg.shift_x + per_line
        truth.applied_shift_x = float(cfg.shift_x)

        out = np.empty_like(img, dtype=np.float64)
        for y in range(h):
            src = (xs - total_dx[y]) / max(cfg.scale_x, 1e-6)
            out[y] = np.interp(src, xs, img[y], left=img[y, 0], right=img[y, -1])

        if abs(cfg.shift_y) > 1e-9:
            ys = np.arange(h, dtype=np.float64)
            src_y = ys - cfg.shift_y
            i0 = np.clip(np.floor(src_y).astype(int), 0, h - 1)
            i1 = np.clip(i0 + 1, 0, h - 1)
            frac = (src_y - i0)[:, None]
            out = out[i0] * (1 - frac) + out[i1] * frac
            truth.applied_shift_y = float(cfg.shift_y)
        return out

    def _bursts(self, img: np.ndarray, rng: np.random.Generator,
                truth: ChannelTruth) -> np.ndarray:
        cfg = self.cfg
        h, w = img.shape
        out = img
        if cfg.burst_rate_per_frame > 0:
            n = int(rng.poisson(cfg.burst_rate_per_frame))
            for _ in range(n):
                length = max(1, int(cfg.burst_len_lines
                                    + rng.integers(-cfg.burst_len_jitter,
                                                   cfg.burst_len_jitter + 1)))
                y0 = int(rng.integers(0, max(1, h - length)))
                y1 = min(h, y0 + length)
                if cfg.burst_mode == "blank":
                    out[y0:y1, :] = 16.0
                elif cfg.burst_mode == "hold":
                    out[y0:y1, :] = out[max(0, y0 - 1) : max(1, y0), :]
                else:
                    out[y0:y1, :] = rng.uniform(0, 255, size=(y1 - y0, w))
                truth.damaged_lines[y0:y1] = True
                truth.damaged_pixels[y0:y1, :] = True
                truth.events.append({"type": "line_burst", "y0": y0, "y1": y1,
                                     "mode": cfg.burst_mode})
        if cfg.region_dropout_rate > 0:
            n = int(rng.poisson(cfg.region_dropout_rate))
            rh, rw = cfg.region_dropout_size
            for _ in range(n):
                y0 = int(rng.integers(0, max(1, h - rh)))
                x0 = int(rng.integers(0, max(1, w - rw)))
                out[y0 : y0 + rh, x0 : x0 + rw] = rng.uniform(0, 255, size=(min(rh, h - y0),
                                                                           min(rw, w - x0)))
                truth.damaged_pixels[y0 : y0 + rh, x0 : x0 + rw] = True
                truth.events.append({"type": "region_dropout", "y0": y0, "x0": x0,
                                     "h": rh, "w": rw})
        return out

    # -- main -------------------------------------------------------------
    def apply(self, raster: np.ndarray, rng: np.random.Generator
              ) -> Tuple[np.ndarray, ChannelTruth]:
        cfg = self.cfg
        h, w = raster.shape
        truth = ChannelTruth(
            damaged_lines=np.zeros(h, dtype=bool),
            damaged_pixels=np.zeros((h, w), dtype=bool),
            applied_gain=cfg.gain, applied_offset=cfg.offset,
        )
        img = raster.astype(np.float64)
        img = self._lowpass(img)
        img = self._geometry(img, rng, truth)
        img = img * cfg.gain + cfg.offset
        if cfg.noise_sigma > 0:
            img = img + rng.normal(0, cfg.noise_sigma, img.shape)
        img = self._bursts(img, rng, truth)
        img = np.clip(img, cfg.clip_low, cfg.clip_high)
        return np.round(img).astype(np.uint8), truth

    def apply_sequence(self, rasters: Sequence[np.ndarray], rng: np.random.Generator
                       ) -> Tuple[List[Optional[np.ndarray]], List[ChannelTruth]]:
        """Frame-level impairments: drop, duplicate and swap received rasters."""
        cfg = self.cfg
        out: List[Optional[np.ndarray]] = []
        truths: List[ChannelTruth] = []
        for r in rasters:
            img, truth = self.apply(r, rng)
            if cfg.frame_drop_prob > 0 and rng.random() < cfg.frame_drop_prob:
                truth.dropped = True
                truth.events.append({"type": "frame_drop"})
                out.append(None)
                truths.append(truth)
                continue
            out.append(img)
            truths.append(truth)
            if cfg.frame_duplicate_prob > 0 and rng.random() < cfg.frame_duplicate_prob:
                truth.duplicated = True
                truth.events.append({"type": "frame_duplicate"})
                out.append(img.copy())
                truths.append(truth)
        if cfg.frame_swap_prob > 0:
            for i in range(len(out) - 1):
                if rng.random() < cfg.frame_swap_prob:
                    out[i], out[i + 1] = out[i + 1], out[i]
                    truths[i].events.append({"type": "frame_swap", "with": i + 1})
        return out, truths


# ---------------------------------------------------------------- named presets
PRESETS: Dict[str, RasterChannelConfig] = {
    "clean": RasterChannelConfig(),
    "mild": RasterChannelConfig(
        gain=0.94, offset=6.0, noise_sigma=3.0, lowpass_taps=3,
        shift_x=1.3, line_jitter_sigma=0.2,
    ),
    "moderate": RasterChannelConfig(
        gain=0.85, offset=14.0, noise_sigma=8.0, lowpass_taps=5, lowpass_kind="gaussian",
        lowpass_sigma=1.2, shift_x=2.7, shift_y=1.0, line_jitter_sigma=0.6,
        slow_drift_px=1.5, burst_rate_per_frame=1.5, burst_len_lines=8,
    ),
    "harsh": RasterChannelConfig(
        gain=0.75, offset=22.0, noise_sigma=16.0, lowpass_taps=7, lowpass_kind="gaussian",
        lowpass_sigma=1.8, shift_x=4.0, shift_y=2.0, line_jitter_sigma=1.2,
        slow_drift_px=3.0, burst_rate_per_frame=4.0, burst_len_lines=16,
        burst_len_jitter=8, region_dropout_rate=1.0, frame_drop_prob=0.05,
    ),
    "bursty": RasterChannelConfig(
        gain=0.9, offset=10.0, noise_sigma=4.0, lowpass_taps=3,
        burst_rate_per_frame=6.0, burst_len_lines=24, burst_len_jitter=8,
    ),
}


def preset(name: str) -> RasterChannelConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown channel preset {name!r}; have {sorted(PRESETS)}")
    import copy

    return copy.deepcopy(PRESETS[name])


# ------------------------------------------------------- diagnostic bit channel
@dataclass
class GilbertElliottConfig:
    """Optional *diagnostic* bit/symbol error model.

    It does not replace the amplitude and geometry model above; it exists only
    to sanity-check FEC behaviour against a classic burst-error process.
    """

    p_good_to_bad: float = 0.002
    p_bad_to_good: float = 0.1
    err_good: float = 1e-4
    err_bad: float = 0.3


def gilbert_elliott_mask(n: int, cfg: GilbertElliottConfig,
                         rng: np.random.Generator) -> np.ndarray:
    state_bad = False
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        p = cfg.err_bad if state_bad else cfg.err_good
        out[i] = rng.random() < p
        if state_bad:
            state_bad = rng.random() >= cfg.p_bad_to_good
        else:
            state_bad = rng.random() < cfg.p_good_to_bad
    return out


__all__ = [
    "RasterChannelConfig", "ChannelTruth", "RasterChannel", "PRESETS", "preset",
    "GilbertElliottConfig", "gilbert_elliott_mask",
]
