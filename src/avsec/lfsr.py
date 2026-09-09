"""B1 baseline: block permutation driven by a Linear Feedback Shift Register.

Reconstruction notice
---------------------
Mardiyanto et al., ISITIA 2021 (DOI 10.1109/ISITIA52817.2021.9502241) describe
"a permutation table created with a Pseudo Random LFSR by seed input" and report
seeds 44257 / 1234 / 1111, a 4x3 block diagram ("will be increased in future
development up to 12x10") and a Raspberry Pi run with a "12x16 block divider".

The paper does **not** state:
  * the register width,
  * the feedback polynomial,
  * Fibonacci vs Galois form or the shift direction,
  * how register states are turned into a permutation,
  * the block traversal order,
  * the handling of frame sizes that are not multiples of the grid,
  * whether the permutation changes between frames.

Everything below is therefore *our reconstruction of the principle*.  Each such
choice is an explicit, documented parameter of :class:`LFSRConfig` /
:class:`ScramblerConfig` and must not be attributed to the original work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Maximal-length feedback polynomials, given as tap positions (1-based, MSB first).
# Source: standard tables of primitive polynomials over GF(2).
DEFAULT_TAPS: Dict[int, Tuple[int, ...]] = {
    8: (8, 6, 5, 4),
    12: (12, 11, 10, 4),
    16: (16, 15, 13, 4),
    17: (17, 14),
    20: (20, 17),
    24: (24, 23, 22, 17),
    32: (32, 22, 2, 1),
}


@dataclass(frozen=True)
class LFSRConfig:
    """Fully explicit LFSR parameters (all values are our reconstruction)."""

    width: int = 16
    taps: Tuple[int, ...] = DEFAULT_TAPS[16]
    seed: int = 44257
    form: str = "fibonacci"          # 'fibonacci' | 'galois'
    zero_state_policy: str = "force_one"   # 'force_one' | 'error'

    def validate(self) -> None:
        if not 2 <= self.width <= 64:
            raise ValueError("LFSR width must be in [2, 64]")
        if not self.taps or max(self.taps) > self.width or min(self.taps) < 1:
            raise ValueError(f"invalid taps {self.taps} for width {self.width}")
        if self.form not in ("fibonacci", "galois"):
            raise ValueError("form must be 'fibonacci' or 'galois'")
        if self.zero_state_policy not in ("force_one", "error"):
            raise ValueError("zero_state_policy must be 'force_one' or 'error'")

    @property
    def state_space(self) -> int:
        """Number of reachable non-zero states = size of this demo key space."""
        return (1 << self.width) - 1

    def describe(self) -> Dict[str, object]:
        poly = " + ".join([f"x^{t}" for t in self.taps] + ["1"])
        return {
            "width": self.width,
            "taps": list(self.taps),
            "polynomial": poly,
            "form": self.form,
            "seed": self.seed,
            "zero_state_policy": self.zero_state_policy,
            "state_space": self.state_space,
            "note": "reconstruction: the 2021 paper does not specify these parameters",
        }


class LFSR:
    """Binary LFSR with an explicit update rule.

    Fibonacci form: ``feedback = XOR(bits at tap positions)``; the register is
    shifted **left** by one and the feedback bit enters at bit 0::

        state = ((state << 1) | feedback) & mask

    where bit position ``t`` of ``taps`` is read as ``(state >> (t - 1)) & 1``.

    Galois form: shift **right**; if the bit that leaves is 1 the tap mask is
    XOR-ed into the state.
    """

    def __init__(self, cfg: LFSRConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self.mask = (1 << cfg.width) - 1
        self._tap_mask = 0
        for t in cfg.taps:
            self._tap_mask |= 1 << (t - 1)
        state = int(cfg.seed) & self.mask
        if state == 0:
            if cfg.zero_state_policy == "error":
                raise ValueError("LFSR seed maps to the zero state, which is a fixed point")
            state = 1
        self.state = state
        self.initial_state = state

    def step(self) -> int:
        """Advance one clock; return the new state."""
        if self.cfg.form == "fibonacci":
            fb = bin(self.state & self._tap_mask).count("1") & 1
            self.state = ((self.state << 1) | fb) & self.mask
        else:  # galois
            lsb = self.state & 1
            self.state >>= 1
            if lsb:
                self.state ^= self._tap_mask
            self.state &= self.mask
        if self.state == 0:  # unreachable for a primitive polynomial; defensive
            self.state = 1
        return self.state

    def states(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.uint64)
        for i in range(n):
            out[i] = self.step()
        return out

    def period(self, limit: Optional[int] = None) -> int:
        """Measure the actual cycle length starting from the current state."""
        limit = limit or (self.state_space_limit() + 2)
        start = self.state
        for i in range(1, limit + 1):
            if self.step() == start:
                return i
        return -1

    def state_space_limit(self) -> int:
        return (1 << self.cfg.width) - 1


# ------------------------------------------------------------------ permutation
PERMUTATION_VARIANTS = ("fisher_yates", "first_occurrence")


def lfsr_permutation(n: int, cfg: LFSRConfig, variant: str = "fisher_yates",
                     max_draws: Optional[int] = None) -> np.ndarray:
    """Build a permutation of ``range(n)`` from an LFSR sequence.

    Two documented reconstructions of "a permutation table created with a
    Pseudo Random LFSR":

    ``fisher_yates``
        Fisher-Yates / Knuth shuffle of ``[0..n-1]``.  Index ``j`` in
        ``[0, i]`` is drawn by *rejection sampling* over the register states so
        that the draw is uniform on the state space restriction (no modulo bias).

    ``first_occurrence``
        The naive table-filling variant often written in embedded code: take
        ``state mod n``; if that slot is still free use it, otherwise clock
        again.  Biased, but it is a realistic implementation of the sentence in
        the paper, so it is available for comparison.

    Both are deterministic given ``(n, cfg, variant)`` and both are invertible.
    """
    if variant not in PERMUTATION_VARIANTS:
        raise ValueError(f"variant must be one of {PERMUTATION_VARIANTS}")
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    lf = LFSR(cfg)
    budget = max_draws if max_draws is not None else max(1000, 200 * n)

    if variant == "fisher_yates":
        perm = np.arange(n, dtype=np.int64)
        span = lf.state_space_limit()  # states are in [1, 2^w - 1]
        for i in range(n - 1, 0, -1):
            m = i + 1
            limit = (span // m) * m  # rejection bound for an unbiased draw
            j = None
            for _ in range(budget):
                s = lf.step()
                if s <= limit:
                    j = (s - 1) % m
                    break
            if j is None:  # pragma: no cover - only for pathological widths
                j = (lf.state - 1) % m
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    # first_occurrence
    perm = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    filled = 0
    draws = 0
    while filled < n and draws < budget:
        s = int(lf.step())
        draws += 1
        j = s % n
        if not used[j]:
            used[j] = True
            perm[filled] = j
            filled += 1
    if filled < n:
        # deterministic completion so that the map stays a bijection
        rest = np.flatnonzero(~used)
        perm[filled:] = rest
    return perm


def invert_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size, dtype=perm.dtype)
    return inv


# -------------------------------------------------------------------- scrambler
@dataclass
class ScramblerConfig:
    """Block scrambling parameters (reconstruction of the 2021 principle)."""

    grid_rows: int = 12
    grid_cols: int = 16
    lfsr: LFSRConfig = field(default_factory=LFSRConfig)
    variant: str = "fisher_yates"
    size_policy: str = "pad"          # 'pad' | 'crop'
    pad_value: int = 0
    per_frame: bool = False           # False = one static permutation for the whole stream
    frame_seed_mode: str = "seed_plus_index"  # used only when per_frame is True

    def describe(self) -> Dict[str, object]:
        return {
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "n_blocks": self.grid_rows * self.grid_cols,
            "variant": self.variant,
            "size_policy": self.size_policy,
            "per_frame": self.per_frame,
            "frame_seed_mode": self.frame_seed_mode if self.per_frame else None,
            "block_order": "row-major (raster) over the block grid",
            "lfsr": self.lfsr.describe(),
        }


class BlockScrambler:
    """Split a frame into ``grid_rows x grid_cols`` equal tiles and permute them.

    Block traversal order is row-major over the tile grid (our choice).  Frames
    whose size is not a multiple of the grid are padded (default) or cropped;
    the policy is part of the public profile, and the inverse restores the
    original size exactly.
    """

    def __init__(self, cfg: ScramblerConfig) -> None:
        if cfg.grid_rows < 1 or cfg.grid_cols < 1:
            raise ValueError("grid dimensions must be >= 1")
        self.cfg = cfg
        self._perm_cache: Dict[int, np.ndarray] = {}

    # -- geometry ---------------------------------------------------------
    def padded_shape(self, shape: Tuple[int, int]) -> Tuple[int, int]:
        h, w = shape
        r, c = self.cfg.grid_rows, self.cfg.grid_cols
        if self.cfg.size_policy == "pad":
            return (int(np.ceil(h / r) * r), int(np.ceil(w / c) * c))
        return ((h // r) * r, (w // c) * c)

    def block_size(self, shape: Tuple[int, int]) -> Tuple[int, int]:
        ph, pw = self.padded_shape(shape)
        return ph // self.cfg.grid_rows, pw // self.cfg.grid_cols

    def _fit(self, img: np.ndarray) -> np.ndarray:
        ph, pw = self.padded_shape(img.shape)
        h, w = img.shape
        if (ph, pw) == (h, w):
            return img
        if self.cfg.size_policy == "crop":
            return img[:ph, :pw]
        out = np.full((ph, pw), self.cfg.pad_value, dtype=img.dtype)
        out[:h, :w] = img
        return out

    # -- permutation ------------------------------------------------------
    def permutation(self, frame_index: int = 0) -> np.ndarray:
        key = frame_index if self.cfg.per_frame else 0
        if key not in self._perm_cache:
            cfg = self.cfg.lfsr
            if self.cfg.per_frame:
                if self.cfg.frame_seed_mode == "seed_plus_index":
                    new_seed = (cfg.seed + frame_index) & ((1 << cfg.width) - 1)
                elif self.cfg.frame_seed_mode == "seed_xor_index":
                    new_seed = (cfg.seed ^ frame_index) & ((1 << cfg.width) - 1)
                else:
                    raise ValueError(f"unknown frame_seed_mode {self.cfg.frame_seed_mode!r}")
                cfg = LFSRConfig(cfg.width, cfg.taps, new_seed or 1, cfg.form,
                                 cfg.zero_state_policy)
            n = self.cfg.grid_rows * self.cfg.grid_cols
            self._perm_cache[key] = lfsr_permutation(n, cfg, self.cfg.variant)
        return self._perm_cache[key]

    # -- transform --------------------------------------------------------
    def _tiles(self, img: np.ndarray) -> np.ndarray:
        r, c = self.cfg.grid_rows, self.cfg.grid_cols
        bh, bw = img.shape[0] // r, img.shape[1] // c
        return (img.reshape(r, bh, c, bw).swapaxes(1, 2).reshape(r * c, bh, bw))

    def _untiles(self, tiles: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        r, c = self.cfg.grid_rows, self.cfg.grid_cols
        bh, bw = tiles.shape[1], tiles.shape[2]
        return tiles.reshape(r, c, bh, bw).swapaxes(1, 2).reshape(r * bh, c * bw)[: shape[0], : shape[1]]

    def scramble(self, img: np.ndarray, frame_index: int = 0) -> np.ndarray:
        """``out[i] = in[perm[i]]`` over row-major tile indices."""
        fitted = self._fit(img)
        perm = self.permutation(frame_index)
        tiles = self._tiles(fitted)
        return self._untiles(tiles[perm], fitted.shape)

    def descramble(self, img: np.ndarray, frame_index: int = 0,
                   original_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        perm = self.permutation(frame_index)
        inv = invert_permutation(perm)
        fitted = self._fit(img) if img.shape != self.padded_shape(img.shape) else img
        tiles = self._tiles(fitted)
        out = self._untiles(tiles[inv], fitted.shape)
        if original_shape is not None:
            out = out[: original_shape[0], : original_shape[1]]
        return out


# ------------------------------------------------------------------- similarity
def correlation_similarity(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """The paper's "similarity": Pearson correlation coefficient, reported as |r| in %.

    Returned as ``(r, similarity_percent)``.  This is a *statistical* similarity
    number, not a security measure - a low value does not imply confidentiality.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    if n == 0:
        return 0.0, 0.0
    xs, ys = x - x.mean(), y - y.mean()
    den = float(np.sqrt((xs * xs).sum() * (ys * ys).sum()))
    if den == 0.0:
        return 0.0, 0.0
    r = float((xs * ys).sum() / den)
    return r, abs(r) * 100.0


__all__ = [
    "DEFAULT_TAPS", "LFSRConfig", "LFSR", "PERMUTATION_VARIANTS", "lfsr_permutation",
    "invert_permutation", "ScramblerConfig", "BlockScrambler", "correlation_similarity",
]
