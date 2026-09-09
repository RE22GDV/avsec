"""Placement of coded symbols in the transmitted raster.

Three schemes share one interface.  Interleaving here is a *burst-damage*
countermeasure only - it is public, it is never relied on for confidentiality.

``sequential``
    Symbols go into data cells in raster order.  Accumulation depth 1 symbol row.

``block``
    Classic block interleaver of depth ``depth``: write the logical stream into
    a ``depth x L`` matrix by rows, read it out by columns.  Accumulation depth
    ``depth`` symbol rows.

``bawp`` - Burst-Aware Window Placement (proposed)
    See :func:`bawp_placement` for the exact rule, its admissibility predicate
    and its complexity.  Accumulation depth = the window height in symbol rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCHEMES = ("sequential", "block", "bawp")


class PlacementError(Exception):
    pass


@dataclass
class InterleaverConfig:
    scheme: str = "sequential"
    depth: int = 1                 # block interleaver depth, in symbol rows
    window_rows: int = 0           # BAWP window height in symbol rows (0 = whole raster)
    burst_rows: int = 4            # assumed max burst height, in symbol rows
    column_twist: int = 1          # BAWP public constant sigma

    def validate(self) -> None:
        if self.scheme not in SCHEMES:
            raise PlacementError(f"scheme must be one of {SCHEMES}")
        if self.depth < 1:
            raise PlacementError("depth must be >= 1")
        if self.burst_rows < 1:
            raise PlacementError("burst_rows must be >= 1")

    def describe(self) -> Dict[str, object]:
        return {
            "scheme": self.scheme,
            "depth_symbol_rows": self.depth if self.scheme == "block" else None,
            "window_rows": self.window_rows if self.scheme == "bawp" else None,
            "assumed_burst_rows": self.burst_rows if self.scheme == "bawp" else None,
            "column_twist": self.column_twist if self.scheme == "bawp" else None,
        }


# --------------------------------------------------------------------- schemes
def sequential_placement(n_cells: int) -> np.ndarray:
    return np.arange(n_cells, dtype=np.int64)


def block_placement(n_rows: int, n_cols: int, depth: int) -> np.ndarray:
    """Return ``cells[l]`` = data-cell index for logical symbol ``l``.

    Write by rows into ``depth x (depth_block_len)`` matrices, read by columns.
    Applied per group of ``depth`` symbol rows so the accumulation depth is
    exactly ``depth`` symbol rows.
    """
    depth = max(1, min(depth, n_rows))
    out = np.empty(n_rows * n_cols, dtype=np.int64)
    pos = 0
    for g0 in range(0, n_rows, depth):
        g = min(depth, n_rows - g0)
        block = np.arange(g0 * n_cols, (g0 + g) * n_cols, dtype=np.int64).reshape(g, n_cols)
        out[pos : pos + g * n_cols] = block.T.ravel()
        pos += g * n_cols
    return out


def bawp_bands(window_rows: int, n_descriptions: int, burst_rows: int) -> List[Tuple[int, int]]:
    """Contiguous row bands of a BAWP window: ``[(row_start, height), ...]``.

    ``G = max(1, min(D, floor(W / B)))`` bands of height ``W // G`` (the last one
    absorbs the remainder).  A burst of at most ``B`` symbol rows therefore
    intersects at most two bands.
    """
    W = int(window_rows)
    B = max(1, int(burst_rows))
    D = max(1, int(n_descriptions))
    G = max(1, min(D, W // B))
    h = W // G
    bands: List[Tuple[int, int]] = []
    r = 0
    for g in range(G):
        hh = h if g < G - 1 else W - r
        bands.append((r, hh))
        r += hh
    return bands


def bawp_placement(
    n_rows: int, n_cols: int, window_rows: int, n_descriptions: int, burst_rows: int,
    unit_lengths: Sequence[int], unit_descs: Sequence[int], column_twist: int = 1,
) -> List[np.ndarray]:
    """Burst-Aware Window Placement (BAWP), profile v1.

    Rule
    ----
    1. The raster's data grid is processed in windows of ``W`` symbol rows.
       Each window is split into ``G = max(1, min(D, floor(W / B)))`` contiguous
       bands (:func:`bawp_bands`).
    2. A transport unit carrying description ``d`` is appended to band
       ``d mod G``; if that band is full the unit spills into the next band
       with free capacity, in band order (deterministic).
    3. Inside a band with rows ``[r0, r0 + h)`` and ``C`` columns, the ``l``-th
       symbol appended to the band is written to::

           row = r0 + (l mod h)
           col = ((l // h) + sigma * (l mod h)) mod C

       which is a bijection onto the band's cells for ``l < h * C``.

    Guarantee
    ---------
    Consecutive symbols of one codeword land in distinct rows, cycling through
    the band, so a burst of ``b <= h`` symbol rows removes at most
    ``b * ceil(n / h)`` symbols of any codeword of length ``n``.  A configuration
    is *admissible* for a code with ``nsym`` parity symbols iff
    ``B * ceil(n / h) <= nsym`` (see :func:`bawp_admissible`).
    Because bands are contiguous and at least ``B`` rows high, a burst of at
    most ``B`` rows touches at most two bands, hence at most
    ``ceil(2 * D / G)`` description classes - the remaining descriptions of the
    affected stripes stay intact, which is what multiple descriptions are for.

    Complexity
    ----------
    ``O(total_symbols)`` time, ``O(total_symbols)`` memory for the table.
    """
    W = n_rows if window_rows <= 0 else min(window_rows, n_rows)
    if W < 1:
        raise PlacementError("window_rows resolves to zero")
    n_windows = int(np.ceil(n_rows / W))
    bands_template = bawp_bands(W, n_descriptions, burst_rows)

    # per-window, per-band fill counters
    fill = [[0] * len(bands_template) for _ in range(n_windows)]
    out: List[np.ndarray] = []

    for length, desc in zip(unit_lengths, unit_descs):
        placed: Optional[np.ndarray] = None
        pref = int(desc) % len(bands_template)
        for wi in range(n_windows):
            row_base = wi * W
            rows_here = min(W, n_rows - row_base)
            order = [(pref + i) % len(bands_template) for i in range(len(bands_template))]
            for bi in order:
                r0, h = bands_template[bi]
                h = min(h, max(0, rows_here - r0))
                if h <= 0:
                    continue
                cap = h * n_cols
                used = fill[wi][bi]
                if cap - used < length:
                    continue
                idx = np.arange(used, used + length, dtype=np.int64)
                rr = row_base + r0 + (idx % h)
                cc = ((idx // h) + column_twist * (idx % h)) % n_cols
                placed = rr * n_cols + cc
                fill[wi][bi] = used + length
                break
            if placed is not None:
                break
        if placed is None:
            raise PlacementError(
                f"raster capacity exhausted while placing a {length}-symbol unit "
                "(the transmitter must reduce the payload, never truncate it)"
            )
        out.append(placed)
    return out


def bawp_admissible(window_rows: int, n_descriptions: int, burst_rows: int,
                    codeword_len: int, nsym: int) -> bool:
    """Predicate the parameter search uses to prune inadmissible configurations."""
    bands = bawp_bands(window_rows, n_descriptions, burst_rows)
    h = min(h for _, h in bands)
    if h < 1:
        return False
    per_row = int(np.ceil(codeword_len / h))
    return burst_rows * per_row <= nsym


# ------------------------------------------------------------------ interleaver
class Interleaver:
    """Assign data-cell indices to the symbols of a sequence of transport units."""

    def __init__(self, cfg: InterleaverConfig, n_rows: int, n_cols: int) -> None:
        cfg.validate()
        self.cfg = cfg
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.capacity = self.n_rows * self.n_cols
        self._table: Optional[np.ndarray] = None
        if cfg.scheme == "sequential":
            self._table = sequential_placement(self.capacity)
        elif cfg.scheme == "block":
            self._table = block_placement(self.n_rows, self.n_cols, cfg.depth)

    @property
    def accumulation_rows(self) -> int:
        """Symbol rows that must be buffered before the first cell can be sent."""
        if self.cfg.scheme == "sequential":
            return 1
        if self.cfg.scheme == "block":
            return max(1, min(self.cfg.depth, self.n_rows))
        return self.n_rows if self.cfg.window_rows <= 0 else min(self.cfg.window_rows, self.n_rows)

    def max_units(self, unit_symbols: int, n_descriptions: int = 1,
                  upper: Optional[int] = None) -> int:
        """How many equal-size units this scheme can actually place in one raster.

        BAWP packs each band contiguously and never splits a unit across bands,
        so it can place fewer units than the raw capacity allows.  Both
        endpoints call this, so the effective schedule stays identical.
        """
        if unit_symbols <= 0:
            return 0
        hi = upper if upper is not None else self.capacity // unit_symbols
        for k in range(int(hi), 0, -1):
            try:
                self.place([unit_symbols] * k,
                           [i % max(1, n_descriptions) for i in range(k)], n_descriptions)
                return k
            except PlacementError:
                continue
        return 0

    def place(self, unit_lengths: Sequence[int], unit_descs: Sequence[int],
              n_descriptions: int = 1) -> List[np.ndarray]:
        total = int(sum(unit_lengths))
        if total > self.capacity:
            raise PlacementError(
                f"{total} symbols exceed the raster data capacity {self.capacity}; "
                "reduce quality/resolution or change the profile"
            )
        if self.cfg.scheme == "bawp":
            return bawp_placement(
                self.n_rows, self.n_cols, self.cfg.window_rows, n_descriptions,
                self.cfg.burst_rows, unit_lengths, unit_descs, self.cfg.column_twist,
            )
        assert self._table is not None
        out: List[np.ndarray] = []
        pos = 0
        for length in unit_lengths:
            out.append(self._table[pos : pos + length])
            pos += length
        return out


__all__ = [
    "SCHEMES", "PlacementError", "InterleaverConfig", "Interleaver",
    "sequential_placement", "block_placement", "bawp_bands", "bawp_placement",
    "bawp_admissible",
]
