"""Raster modem: protected data carried as luminance levels in the active picture.

Nothing here assumes that an arbitrary byte survives as one 8-bit pixel.  The
carrier is a grid of *symbol cells* of ``symbol_width x symbol_height`` pixels
using ``levels`` amplitude levels with a margin to black and white, and every
receiver parameter that can be estimated is estimated from transmitted pilots.

Raster layout (public profile)
------------------------------
::

    +-----------------------------------------------------------+
    |  sync rows (known PRBS over the full symbol width)         |
    +-----+-----------------------------------------------+-----+
    | pil |                  data cells                   | pil |
    | ot  |                                               | ot  |
    +-----+-----------------------------------------------+-----+

* ``sync_rows`` symbol rows at the top carry a fixed pseudo-random pattern.
  The receiver *searches* for it (2-D correlation over a bounded offset range);
  it is never told where the raster starts.
* ``pilot_cols`` cells on the left and right of every data row carry a known
  level ladder, from which the receiver fits a per-line gain and offset.
* Everything else is payload.  Which logical symbol goes into which data cell
  is decided by :mod:`avsec.interleaving`, not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

BLANK_LEVEL = 16


def prbs_symbols(n: int, levels: int, seed: int = 0x5A5A) -> np.ndarray:
    """Deterministic public pattern used for sync and pilots (not secret)."""
    state = seed & 0xFFFF or 1
    out = np.empty(n, dtype=np.uint8)
    bits = int(np.log2(levels))
    for i in range(n):
        v = 0
        for _ in range(bits):
            fb = ((state >> 15) ^ (state >> 14) ^ (state >> 12) ^ (state >> 3)) & 1
            state = ((state << 1) | fb) & 0xFFFF
            v = (v << 1) | fb
        out[i] = v
    return out


@dataclass
class ModemConfig:
    """Public transport profile of the raster modem."""

    raster_width: int = 720
    raster_height: int = 576
    active_x0: int = 24
    active_x1: int = 696
    active_y0: int = 8
    active_y1: int = 568
    levels: int = 2                 # 2 or 4
    symbol_width: int = 4           # pixels per symbol, horizontally
    symbol_height: int = 2          # raster lines per symbol
    level_low: int = 40             # margin to black
    level_high: int = 216           # margin to white
    sync_rows: int = 2
    pilot_cols: int = 4
    sample_margin_frac: float = 0.25   # ignore this fraction of a cell at each edge
    erasure_threshold: float = 0.32    # fraction of the level spacing
    tx_shaping: bool = False           # optional 3-tap horizontal pulse shaping

    def validate(self) -> None:
        if self.levels not in (2, 4):
            raise ValueError("levels must be 2 or 4")
        if self.symbol_width < 1 or self.symbol_height < 1:
            raise ValueError("symbol size must be >= 1 pixel")
        if not (0 <= self.active_x0 < self.active_x1 <= self.raster_width):
            raise ValueError("active horizontal window out of raster")
        if not (0 <= self.active_y0 < self.active_y1 <= self.raster_height):
            raise ValueError("active vertical window out of raster")
        if self.level_low >= self.level_high:
            raise ValueError("level_low must be below level_high")
        if self.n_cols <= 2 * self.pilot_cols:
            raise ValueError("not enough symbol columns for the configured pilots")
        if self.n_rows <= self.sync_rows:
            raise ValueError("not enough symbol rows for the configured sync")

    # -- grid -------------------------------------------------------------
    @property
    def bits_per_symbol(self) -> int:
        return 1 if self.levels == 2 else 2

    @property
    def n_cols(self) -> int:
        return (self.active_x1 - self.active_x0) // self.symbol_width

    @property
    def n_rows(self) -> int:
        return (self.active_y1 - self.active_y0) // self.symbol_height

    @property
    def n_data_rows(self) -> int:
        return self.n_rows - self.sync_rows

    @property
    def n_data_cols(self) -> int:
        return self.n_cols - 2 * self.pilot_cols

    @property
    def capacity_symbols(self) -> int:
        return self.n_data_rows * self.n_data_cols

    @property
    def capacity_bits(self) -> int:
        return self.capacity_symbols * self.bits_per_symbol

    @property
    def capacity_bytes(self) -> int:
        return self.capacity_bits // 8

    def lines_per_symbol_row(self) -> int:
        return self.symbol_height

    def levels_array(self) -> np.ndarray:
        return np.linspace(self.level_low, self.level_high, self.levels)

    def describe(self) -> Dict[str, object]:
        return {
            "raster": f"{self.raster_width}x{self.raster_height}",
            "active_window": [self.active_x0, self.active_y0, self.active_x1, self.active_y1],
            "levels": self.levels,
            "bits_per_symbol": self.bits_per_symbol,
            "symbol_cell_px": [self.symbol_width, self.symbol_height],
            "symbol_grid": [self.n_rows, self.n_cols],
            "sync_rows": self.sync_rows,
            "pilot_cols_each_side": self.pilot_cols,
            "level_low": self.level_low,
            "level_high": self.level_high,
            "capacity_symbols_per_raster": self.capacity_symbols,
            "capacity_bytes_per_raster": self.capacity_bytes,
            "tx_shaping": self.tx_shaping,
        }


@dataclass
class DemodResult:
    symbols: np.ndarray            # decided data symbols, logical raster order
    erasures: np.ndarray           # bool, receiver's own reliability flag
    sync_found: bool
    dx: int                        # estimated horizontal offset in pixels
    dy: int                        # estimated vertical offset in pixels
    sync_score: float
    gain: float
    offset: float
    line_gain: Optional[np.ndarray] = None
    soft_margin: Optional[np.ndarray] = None

    def summary(self) -> Dict[str, float]:
        return {
            "sync_found": bool(self.sync_found), "dx": int(self.dx), "dy": int(self.dy),
            "sync_score": float(self.sync_score), "gain": float(self.gain),
            "offset": float(self.offset),
            "erasure_fraction": float(self.erasures.mean()) if self.erasures.size else 0.0,
        }


class RasterModem:
    """Modulate a symbol stream into a raster and demodulate it back."""

    def __init__(self, cfg: ModemConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self._levels = cfg.levels_array()
        self._sync = prbs_symbols(cfg.sync_rows * cfg.n_cols, cfg.levels, 0x5A5A)
        self._pilot = prbs_symbols(cfg.n_data_rows * 2 * cfg.pilot_cols, cfg.levels, 0x1234)

    # -- helpers ----------------------------------------------------------
    def _cell_slice(self, r: int, c: int) -> Tuple[slice, slice]:
        cfg = self.cfg
        y = cfg.active_y0 + r * cfg.symbol_height
        x = cfg.active_x0 + c * cfg.symbol_width
        return slice(y, y + cfg.symbol_height), slice(x, x + cfg.symbol_width)

    def data_cell_index(self, k: int) -> Tuple[int, int]:
        """Logical data cell ``k`` -> (symbol row, symbol column) in the raster."""
        cfg = self.cfg
        r = k // cfg.n_data_cols
        c = k % cfg.n_data_cols
        return cfg.sync_rows + r, cfg.pilot_cols + c

    def data_cell_line(self, k: int) -> int:
        """First raster line occupied by logical data cell ``k`` (for burst analysis)."""
        r, _ = self.data_cell_index(k)
        return self.cfg.active_y0 + r * self.cfg.symbol_height

    # -- modulate ---------------------------------------------------------
    def modulate(self, symbols: np.ndarray, fill_symbol: int = 0) -> np.ndarray:
        """Render ``symbols`` (logical data-cell order) into a full raster."""
        cfg = self.cfg
        if symbols.size > cfg.capacity_symbols:
            raise ValueError(
                f"{symbols.size} symbols exceed the raster capacity {cfg.capacity_symbols}; "
                "the transmitter must not send beyond the declared capacity"
            )
        raster = np.full((cfg.raster_height, cfg.raster_width), BLANK_LEVEL, dtype=np.uint8)

        grid = np.full((cfg.n_rows, cfg.n_cols), fill_symbol, dtype=np.uint8)
        grid[: cfg.sync_rows, :] = self._sync.reshape(cfg.sync_rows, cfg.n_cols)
        pil = self._pilot.reshape(cfg.n_data_rows, 2 * cfg.pilot_cols)
        grid[cfg.sync_rows :, : cfg.pilot_cols] = pil[:, : cfg.pilot_cols]
        grid[cfg.sync_rows :, cfg.n_cols - cfg.pilot_cols :] = pil[:, cfg.pilot_cols :]

        flat = np.full(cfg.capacity_symbols, fill_symbol, dtype=np.uint8)
        flat[: symbols.size] = symbols
        grid[cfg.sync_rows :, cfg.pilot_cols : cfg.n_cols - cfg.pilot_cols] = flat.reshape(
            cfg.n_data_rows, cfg.n_data_cols
        )

        levels = self._levels[grid].astype(np.float64)
        block = np.repeat(np.repeat(levels, cfg.symbol_height, axis=0), cfg.symbol_width, axis=1)
        y0, x0 = cfg.active_y0, cfg.active_x0
        raster[y0 : y0 + block.shape[0], x0 : x0 + block.shape[1]] = np.round(block).astype(np.uint8)

        if cfg.tx_shaping:
            region = raster[y0 : y0 + block.shape[0], x0 : x0 + block.shape[1]].astype(np.float64)
            kern = np.array([0.25, 0.5, 0.25])
            sm = np.apply_along_axis(lambda m: np.convolve(m, kern, mode="same"), 1, region)
            raster[y0 : y0 + block.shape[0], x0 : x0 + block.shape[1]] = np.clip(
                np.round(sm), 0, 255
            ).astype(np.uint8)
        return raster

    # -- demodulate -------------------------------------------------------
    def _sample_grid(self, raster: np.ndarray, dy: int, dx: int,
                     rows_limit: Optional[int] = None) -> np.ndarray:
        """Mean of the central part of every symbol cell, as a (n_rows, n_cols) array."""
        cfg = self.cfg
        n_rows = cfg.n_rows if rows_limit is None else min(rows_limit, cfg.n_rows)
        my = max(0, int(round(cfg.symbol_height * cfg.sample_margin_frac)))
        mx = max(0, int(round(cfg.symbol_width * cfg.sample_margin_frac)))
        hy = max(1, cfg.symbol_height - 2 * my)
        hx = max(1, cfg.symbol_width - 2 * mx)

        y0 = cfg.active_y0 + dy + my
        x0 = cfg.active_x0 + dx + mx
        rows = y0 + np.arange(n_rows)[:, None] * cfg.symbol_height + np.arange(hy)[None, :]
        cols = x0 + np.arange(cfg.n_cols)[:, None] * cfg.symbol_width + np.arange(hx)[None, :]
        rows = np.clip(rows, 0, raster.shape[0] - 1)
        cols = np.clip(cols, 0, raster.shape[1] - 1)
        # gather: (n_rows, hy, n_cols, hx) -> mean over hy, hx
        patch = raster[rows[:, :, None, None], cols[None, None, :, :]]
        return patch.astype(np.float64).mean(axis=(1, 3))

    def _sync_score(self, grid: np.ndarray) -> float:
        cfg = self.cfg
        obs = grid[: cfg.sync_rows, :].ravel()
        ref = self._levels[self._sync].astype(np.float64)
        o = obs - obs.mean()
        r = ref - ref.mean()
        den = float(np.sqrt((o * o).sum() * (r * r).sum()))
        return float((o * r).sum() / den) if den > 0 else -1.0

    def find_sync(self, raster: np.ndarray, search_x: int = 12, search_y: int = 8
                  ) -> Tuple[int, int, float]:
        """Search a bounded offset window for the transmitted sync pattern."""
        scores: Dict[Tuple[int, int], float] = {}
        best = -2.0
        for dy in range(-search_y, search_y + 1):
            for dx in range(-search_x, search_x + 1):
                g = self._sample_grid(raster, dy, dx, rows_limit=self.cfg.sync_rows)
                s = self._sync_score(g)
                scores[(dy, dx)] = s
                best = max(best, s)
        # A cell is several pixels wide, so a whole plateau of offsets samples the
        # same symbols and scores identically.  Report the centre of that plateau
        # instead of whichever corner the scan happened to reach first.
        tol = 1e-9 + 1e-4 * max(abs(best), 1.0)
        plateau = [k for k, v in scores.items() if v >= best - tol]
        dy = int(round(float(np.mean([p[0] for p in plateau]))))
        dx = int(round(float(np.mean([p[1] for p in plateau]))))
        return dy, dx, float(scores.get((dy, dx), best))

    def demodulate(self, raster: np.ndarray, search_x: int = 12, search_y: int = 8,
                   sync_threshold: float = 0.35) -> DemodResult:
        cfg = self.cfg
        dy, dx, score = self.find_sync(raster, search_x, search_y)
        grid = self._sample_grid(raster, dy, dx)
        sync_found = score >= sync_threshold

        # global amplitude fit from the sync rows (known symbols)
        ref_sync = self._levels[self._sync].astype(np.float64)
        obs_sync = grid[: cfg.sync_rows, :].ravel()
        gain, offset = _robust_affine(ref_sync, obs_sync)

        # per-symbol-row refinement from the pilot columns
        pil = self._pilot.reshape(cfg.n_data_rows, 2 * cfg.pilot_cols)
        ref_pil = self._levels[pil].astype(np.float64)
        obs_pil = np.concatenate(
            [grid[cfg.sync_rows :, : cfg.pilot_cols],
             grid[cfg.sync_rows :, cfg.n_cols - cfg.pilot_cols :]], axis=1
        )
        line_gain = np.empty(cfg.n_data_rows)
        line_off = np.empty(cfg.n_data_rows)
        for i in range(cfg.n_data_rows):
            g_i, o_i = _robust_affine(ref_pil[i], obs_pil[i], fallback=(gain, offset))
            line_gain[i], line_off[i] = g_i, o_i

        data = grid[cfg.sync_rows :, cfg.pilot_cols : cfg.n_cols - cfg.pilot_cols]
        corrected = (data - line_off[:, None]) / np.where(np.abs(line_gain) < 1e-6, 1.0,
                                                          line_gain)[:, None]
        dist = np.abs(corrected[..., None] - self._levels[None, None, :])
        idx = np.argmin(dist, axis=-1).astype(np.uint8)
        margin = np.min(dist, axis=-1)
        spacing = float(self._levels[1] - self._levels[0]) if cfg.levels > 1 else 1.0
        erasures = margin > (cfg.erasure_threshold * spacing)

        return DemodResult(
            symbols=idx.ravel(), erasures=erasures.ravel(), sync_found=sync_found,
            dx=dx, dy=dy, sync_score=score, gain=gain, offset=offset,
            line_gain=line_gain, soft_margin=margin.ravel() / max(spacing, 1e-9),
        )


def _robust_affine(ref: np.ndarray, obs: np.ndarray,
                   fallback: Tuple[float, float] = (1.0, 0.0)) -> Tuple[float, float]:
    """Least-squares fit ``obs ~ gain*ref + offset`` with a degenerate-case fallback."""
    r = np.asarray(ref, dtype=np.float64).ravel()
    o = np.asarray(obs, dtype=np.float64).ravel()
    if r.size < 2 or float(r.std()) < 1e-9:
        return fallback
    rm, om = r.mean(), o.mean()
    den = float(((r - rm) ** 2).sum())
    if den < 1e-9:
        return fallback
    g = float(((r - rm) * (o - om)).sum() / den)
    if abs(g) < 1e-3:
        return fallback
    return g, float(om - g * rm)


__all__ = ["BLANK_LEVEL", "prbs_symbols", "ModemConfig", "DemodResult", "RasterModem"]
