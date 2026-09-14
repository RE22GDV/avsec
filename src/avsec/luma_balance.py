"""Luminance-balanced encryption: a ciphertext of constant perceived brightness.

The idea
--------
Equal numbers in different colour channels do not look equally bright.  Under
BT.601 the perceived luminance of a pixel is

    Y = 0.299 R + 0.587 G + 0.114 B

so green at 0x80 looks more than five times brighter than blue at 0x80.  That
is the same reason three LEDs on one supply need three different resistors.

This module builds a cipher around that weighting: the ciphertext is
constrained so that **every pixel has the same luminance**.  One equation on
three unknowns leaves a two-dimensional set of colours per pixel, and the
message lives in that set.  To an observer who sees only luminance - a
monochrome sensor, a black-and-white monitor, or the luma-only analog path
modelled elsewhere in this project - the ciphertext is a uniform grey field
carrying no information at all.

The capacity argument, which decides everything else
----------------------------------------------------
The set of 8-bit colours whose rounded luminance equals a given level is
finite and can be counted exactly.  At the best level it has **111 749**
members, that is **16.77 bits** per pixel.  Therefore

* a **monochrome** source needs 8 bits per pixel and fits with room to spare;
  the transform is exactly invertible;
* a **colour** source needs 24 bits per pixel and does **not** fit.  The
  largest power of two below the capacity is ``2^16``, so a colour image can
  keep at most 16 of its 24 bits.

That is not an implementation limit; it follows from counting lattice points,
and no cleverer encoding removes it.  The two cases are therefore implemented
as two separate classes rather than one with a flag, because they are two
different situations: one lossless, one lossy by construction.

What this is and is not
-----------------------
Constant luminance is **perceptual concealment**, not a cryptographic
guarantee.  The chroma planes still carry the structure of the plaintext, and
an attack that looks at them is unaffected - this is measured in
``docs/luma_balance.md`` rather than argued.  The construction belongs on top
of the keyed substitution and permutation, not instead of them.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

#: BT.601 luma weights - the same convention the modem and the CVBS model use.
BT601 = (0.299, 0.587, 0.114)

#: Luminance level with the largest number of 8-bit colours; see module docstring.
BEST_LEVEL = 130


def luma(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luminance of an RGB array, as floating point."""
    a = np.asarray(rgb, dtype=np.float64)
    return BT601[0] * a[..., 0] + BT601[1] * a[..., 1] + BT601[2] * a[..., 2]


def constant_luma_palette(level: int = BEST_LEVEL) -> np.ndarray:
    """Every 8-bit colour whose rounded luminance equals ``level``.

    Returned sorted by hue angle in the chroma plane, then by saturation, so
    the ordering is deterministic, public, and places perceptually similar
    colours next to each other.  That matters: the codebook is drawn from this
    order at even spacing, which maximises the distance between neighbouring
    codewords.
    """
    r = np.arange(256, dtype=np.float64)
    y = (BT601[0] * r[:, None, None] + BT601[1] * r[None, :, None]
         + BT601[2] * r[None, None, :])
    sel = np.argwhere(np.rint(y).astype(np.int64) == int(level))
    rgb = sel.astype(np.uint8)
    # order by (hue, saturation) in the Cb/Cr plane of the same colour model
    f = rgb.astype(np.float64)
    cb = f[:, 2] - luma(f)
    cr = f[:, 0] - luma(f)
    order = np.lexsort((np.hypot(cb, cr), np.arctan2(cr, cb)))
    return np.ascontiguousarray(rgb[order])


def palette_capacity(level: int = BEST_LEVEL) -> Dict[str, object]:
    """How many bits one pixel of constant-luminance ciphertext can carry."""
    pal = constant_luma_palette(level)
    n = int(pal.shape[0])
    usable = int(2 ** int(np.floor(np.log2(n))))
    return {
        "level": int(level),
        "colours_with_this_luma": n,
        "capacity_bits": round(float(np.log2(n)), 3),
        "usable_codewords_power_of_two": usable,
        "usable_bits": int(np.log2(usable)),
        "enough_for_monochrome_8_bit": n >= 256,
        "enough_for_colour_24_bit": n >= 2 ** 24,
        "colour_bits_lost": 24 - int(np.log2(usable)),
    }


def _keyed_codebook(palette: np.ndarray, n_codewords: int, key: bytes,
                    context: bytes, keyed: bool = True) -> np.ndarray:
    """``n_codewords`` colours from the palette, in a key-dependent order.

    The *selection* is public and evenly spaced through the palette order, so
    neighbouring codewords stay as far apart as the palette allows.  The
    *assignment* of message values to those colours is keyed.  This is the same
    split as the algebraic S-box in :mod:`avsec.galois`: public structure, key
    where the secrecy has to be.
    """
    from avsec.baselines import crypto_permutation

    n = palette.shape[0]
    if n_codewords > n:
        raise ValueError(f"need {n_codewords} codewords, palette has {n}")
    picked = palette[np.linspace(0, n - 1, n_codewords).astype(np.int64)]
    if not keyed:
        # control: value v takes the v-th colour of the public palette order,
        # so neighbouring values stay neighbouring colours
        return np.ascontiguousarray(picked)
    order = crypto_permutation(n_codewords, key, b"avsec/luma|" + context)
    return np.ascontiguousarray(picked[order])


def _pack(rgb: np.ndarray) -> np.ndarray:
    """RGB triples to one integer each, for vectorised lookup."""
    a = np.asarray(rgb, dtype=np.int64)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


class _Codebook:
    """A keyed value-to-colour map with vectorised encode and decode."""

    def __init__(self, n_codewords: int, key: bytes, context: bytes,
                 level: int, keyed: bool = True) -> None:
        self.level = int(level)
        self.keyed = bool(keyed)
        self.colours = _keyed_codebook(constant_luma_palette(level),
                                       n_codewords, key, context, keyed)
        packed = _pack(self.colours)
        self._order = np.argsort(packed)
        self._sorted = packed[self._order]

    def encode(self, values: np.ndarray) -> np.ndarray:
        return self.colours[np.asarray(values, dtype=np.int64)]

    def decode(self, rgb: np.ndarray) -> np.ndarray:
        """Exact reverse lookup; unknown colours decode to index 0."""
        want = _pack(rgb)
        pos = np.searchsorted(self._sorted, want)
        pos = np.clip(pos, 0, self._sorted.size - 1)
        hit = self._sorted[pos] == want
        return np.where(hit, self._order[pos], 0)

    def decode_nearest(self, rgb: np.ndarray) -> np.ndarray:
        """Nearest codeword in RGB distance, for a perturbed ciphertext."""
        from scipy.spatial import cKDTree

        if not hasattr(self, "_tree"):
            self._tree = cKDTree(self.colours.astype(np.float64))
        flat = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
        _, idx = self._tree.query(flat, k=1)
        return idx.reshape(np.asarray(rgb).shape[:-1])


# --------------------------------------------------------------- monochrome
class LumaBalancedMono:
    """Monochrome source, constant-luminance colour ciphertext, **lossless**.

    A greyscale pixel carries 8 bits and the palette offers 16.77, so every
    value gets its own colour and the transform is a bijection.  Recovery is
    exact to the bit.
    """

    def __init__(self, key: bytes, session_id: bytes, level: int = BEST_LEVEL,
                 per_frame: bool = True, keyed: bool = True) -> None:
        self.key = key
        self.session_id = session_id
        self.level = int(level)
        self.per_frame = bool(per_frame)
        self.keyed = bool(keyed)
        self._cache: Dict[int, _Codebook] = {}

    def codebook(self, frame_id: int = 0) -> _Codebook:
        k = int(frame_id) if self.per_frame else 0
        if k not in self._cache:
            ctx = b"mono|" + self.session_id + b"|" + int(k).to_bytes(8, "big")
            self._cache[k] = _Codebook(256, self.key, ctx, self.level, self.keyed)
        return self._cache[k]

    def encrypt(self, img: np.ndarray, frame_id: int = 0) -> np.ndarray:
        a = np.asarray(img, dtype=np.uint8)
        if a.ndim != 2:
            raise ValueError("monochrome path expects a HxW frame")
        return self.codebook(frame_id).encode(a).astype(np.uint8)

    def decrypt(self, ct: np.ndarray, frame_id: int = 0,
                nearest: bool = False) -> np.ndarray:
        cb = self.codebook(frame_id)
        idx = cb.decode_nearest(ct) if nearest else cb.decode(ct)
        return idx.astype(np.uint8)

    def describe(self) -> Dict[str, object]:
        return {
            "scheme": "B2l-mono",
            "source": "монохромний кадр, 8 біт на піксель",
            "ciphertext": "RGB зі сталою яскравістю",
            "luma_level": self.level,
            "codewords": 256,
            "keyed_assignment": self.keyed,
            "lossless": True,
            "per_frame_codebook": self.per_frame,
        }


# ------------------------------------------------------------------- colour
class LumaBalancedColour:
    """Colour source, constant-luminance ciphertext, **lossy by construction**.

    A colour pixel carries 24 bits and the palette offers 16.77, so 16 bits is
    the most that can survive.  The channels are reduced to 5-6-5 bits before
    encryption, which is exactly 16, and the loss is the quantisation of that
    reduction - measured, not estimated.
    """

    #: Bits kept per channel.  Their sum must not exceed the palette capacity.
    BITS = (5, 6, 5)

    def __init__(self, key: bytes, session_id: bytes, level: int = BEST_LEVEL,
                 bits: Tuple[int, int, int] = BITS,
                 per_frame: bool = True, keyed: bool = True) -> None:
        self.key = key
        self.session_id = session_id
        self.level = int(level)
        self.bits = tuple(int(b) for b in bits)
        self.n_codewords = 1 << sum(self.bits)
        cap = palette_capacity(level)
        if self.n_codewords > int(cap["colours_with_this_luma"]):
            raise ValueError(
                f"{sum(self.bits)} біт потребує {self.n_codewords} кодових слів, "
                f"а палітра має {cap['colours_with_this_luma']}")
        self.per_frame = bool(per_frame)
        self.keyed = bool(keyed)
        self._cache: Dict[int, _Codebook] = {}

    # -- the reduction that makes the data fit ----------------------------
    def quantise(self, img: np.ndarray) -> np.ndarray:
        """The frame as it will be recovered: 24 bits reduced to 16."""
        a = np.asarray(img, dtype=np.int64)
        out = np.empty_like(a)
        for c, b in enumerate(self.bits):
            shift = 8 - b
            q = a[..., c] >> shift
            # replicate the high bits downward so the range still reaches 255
            out[..., c] = (q << shift) | (q >> max(0, 2 * b - 8))
        return out.astype(np.uint8)

    def _index(self, img: np.ndarray) -> np.ndarray:
        a = np.asarray(img, dtype=np.int64)
        idx = np.zeros(a.shape[:2], dtype=np.int64)
        shift = 0
        for c in (2, 1, 0):
            b = self.bits[c]
            idx |= (a[..., c] >> (8 - b)) << shift
            shift += b
        return idx

    def _unindex(self, idx: np.ndarray) -> np.ndarray:
        out = np.empty(idx.shape + (3,), dtype=np.int64)
        shift = 0
        for c in (2, 1, 0):
            b = self.bits[c]
            q = (idx >> shift) & ((1 << b) - 1)
            out[..., c] = (q << (8 - b)) | (q >> max(0, 2 * b - 8))
            shift += b
        return out.astype(np.uint8)

    def codebook(self, frame_id: int = 0) -> _Codebook:
        k = int(frame_id) if self.per_frame else 0
        if k not in self._cache:
            ctx = b"colour|" + self.session_id + b"|" + int(k).to_bytes(8, "big")
            self._cache[k] = _Codebook(self.n_codewords, self.key, ctx,
                                       self.level, self.keyed)
        return self._cache[k]

    def encrypt(self, img: np.ndarray, frame_id: int = 0) -> np.ndarray:
        a = np.asarray(img, dtype=np.uint8)
        if a.ndim != 3 or a.shape[2] < 3:
            raise ValueError("colour path expects a HxWx3 frame")
        return self.codebook(frame_id).encode(self._index(a[..., :3])).astype(np.uint8)

    def decrypt(self, ct: np.ndarray, frame_id: int = 0,
                nearest: bool = False) -> np.ndarray:
        cb = self.codebook(frame_id)
        idx = cb.decode_nearest(ct) if nearest else cb.decode(ct)
        return self._unindex(idx)

    def describe(self) -> Dict[str, object]:
        return {
            "scheme": "B2l-colour",
            "source": "кольоровий кадр, 24 біти на піксель",
            "ciphertext": "RGB зі сталою яскравістю",
            "luma_level": self.level,
            "bits_kept": list(self.bits),
            "bits_kept_total": sum(self.bits),
            "bits_lost": 24 - sum(self.bits),
            "codewords": self.n_codewords,
            "keyed_assignment": self.keyed,
            "lossless": False,
            "per_frame_codebook": self.per_frame,
        }


# ------------------------------------------------------------- measurements
def luma_statistics(ct: np.ndarray) -> Dict[str, float]:
    """What a luminance-only observer sees in the ciphertext."""
    y = luma(ct)
    yr = np.rint(y)
    hist = np.bincount(yr.astype(np.int64).ravel(), minlength=256)
    p = hist[hist > 0] / hist.sum()
    return {
        "luma_min": round(float(y.min()), 4),
        "luma_max": round(float(y.max()), 4),
        "luma_mean": round(float(y.mean()), 4),
        "luma_std": round(float(y.std()), 6),
        "rounded_levels_used": int((hist > 0).sum()),
        "luma_entropy_bits": round(float(-(p * np.log2(p)).sum()), 6),
    }


def chroma_planes(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """The two planes the message actually lives in, for attacks and figures."""
    a = np.asarray(rgb, dtype=np.float64)
    y = luma(a)
    return {"Y": y, "Cb": a[..., 2] - y, "Cr": a[..., 0] - y}


__all__ = [
    "BT601", "BEST_LEVEL", "luma", "constant_luma_palette", "palette_capacity",
    "LumaBalancedMono", "LumaBalancedColour", "luma_statistics", "chroma_planes",
]
