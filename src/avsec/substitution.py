"""B2s: a keyed value substitution added to the block permutation.

Why this module exists
---------------------
``B1`` and ``B2`` are built from a single primitive: a permutation of blocks.
A permutation moves pixels and never changes them, so everything a pixel value
carries survives the transform - the histogram of a scrambled frame is the
histogram of the original, byte for byte.  A cipher built from one primitive of
one kind is not a cipher, and the attacks in :mod:`avsec.attacks` measure
exactly that.

This module adds the second classical primitive: **substitution**.  Each pixel
value ``v`` in ``0..255`` is replaced by ``S(v)`` for a key-derived bijection
``S``.  Permutation is diffusion - it spreads information across the frame;
substitution is confusion - it destroys the relation between a value and what
it means.  Together they form the smallest honest substitution-permutation
network over an image.

Three substitution modes, and why the difference matters
--------------------------------------------------------
``session``
    One table for the whole session.  This is a **monoalphabetic** cipher on
    pixel values: the table is a relabelling, so the *sorted* histogram of the
    frame is unchanged and a single known frame recovers the whole table.  It
    is implemented because measuring its weakness is the point.

``frame``
    A new table for every frame.  A table recovered from one frame is worthless
    for the next one.

``block``
    A separate table for every block of every frame, so two blocks with the
    same content encrypt differently - the image analogue of moving from a
    monoalphabetic to a polyalphabetic cipher.

Two ways to build the table
---------------------------
``random`` (default)
    A keyed shuffle of ``0..255``.  Secret, and it has to be stored - 256 bytes.
    Its resistance to differential and linear analysis is whatever the draw
    gave.

``algebraic``
    The AES construction from :mod:`avsec.galois`: the multiplicative inverse in
    GF(2^8) followed by an affine map, **computed** rather than stored, with
    key material added before and after.  The table itself is public; the key
    enters where a block cipher puts it.  Its resistance is a theorem rather
    than a draw, and :mod:`avsec.sbox_analysis` measures both.

What this does and does not provide
-----------------------------------
Substitution removes the local continuity that the key-free reassembly attack
in :mod:`avsec.attacks` lives on, which is a real gain and is measured in
``docs/substitution_b2s.md``.  It does **not** turn the scheme into
authenticated encryption: there is no integrity tag, no freshness counter, and
a known (plaintext, ciphertext) pair still recovers the transform of the frame
it came from, because substitution is pointwise and the block permutation is
recoverable from a statistic that substitution leaves invariant.  Those limits
are measured too, by :func:`avsec.attacks.attack_substitution_known_pair`.

The substitution is applied **before** the permutation, over the original block
geometry.  The receiver therefore undoes the permutation first, which puts
every block back at the position whose table was used, and only then inverts
the substitution.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from avsec.baselines import crypto_permutation

#: Value alphabet of an 8-bit luma sample.
ALPHABET = 256

#: How the substitution table may vary.  See the module docstring.
SUBSTITUTION_MODES = ("session", "frame", "block")

#: How the table is built: a keyed shuffle, or the computed algebraic table.
TABLE_SOURCES = ("random", "algebraic")

#: Colour planes, in the order the arrays carry them.  ``A`` is optional: 24
#: bits are enough for the picture, and the alpha plane only exists when the
#: source actually has one.
RGB_CHANNELS = ("R", "G", "B")
RGBA_CHANNELS = ("R", "G", "B", "A")


def crypto_sbox(key: bytes, context: bytes) -> np.ndarray:
    """A bijective table ``0..255 -> 0..255`` derived from the key.

    The table is a permutation of the alphabet drawn with the same keyed
    Fisher-Yates shuffle the block permutation uses, so there is one sampling
    routine in the project and one place where modulo bias could hide.
    """
    return crypto_permutation(ALPHABET, key, b"avsec/b2s|sbox|" + context).astype(
        np.uint8)


def invert_sbox(sbox: np.ndarray) -> np.ndarray:
    """The table that undoes ``sbox``."""
    inv = np.empty(ALPHABET, dtype=np.uint8)
    inv[sbox.astype(np.int64)] = np.arange(ALPHABET, dtype=np.uint8)
    return inv


def apply_sbox(img: np.ndarray, sbox: np.ndarray) -> np.ndarray:
    """Replace every sample of ``img`` through ``sbox``."""
    return sbox[img.astype(np.int64)]


class SubstitutionTables:
    """The key-derived substitution tables of one session.

    ``mode`` decides how many tables exist and what they are indexed by; see
    the module docstring.  Tables are cached per frame, because deriving 192 of
    them costs a keyed shuffle each.
    """

    def __init__(self, key: bytes, session_id: bytes, n_blocks: int,
                 mode: str = "block", source: str = "random") -> None:
        if mode not in SUBSTITUTION_MODES:
            raise ValueError(
                f"mode must be one of {SUBSTITUTION_MODES}, got {mode!r}")
        if source not in TABLE_SOURCES:
            raise ValueError(
                f"source must be one of {TABLE_SOURCES}, got {source!r}")
        self.key = key
        self.session_id = session_id
        self.n_blocks = int(n_blocks)
        self.mode = mode
        self.source = source
        self._algebraic = None
        if source == "algebraic":
            from avsec.galois import KeyedAlgebraicSbox

            self._algebraic = KeyedAlgebraicSbox(key)
        self._cache: Dict[int, np.ndarray] = {}

    def _one(self, context: bytes) -> np.ndarray:
        """One table for this context, by whichever construction is selected."""
        if self._algebraic is not None:
            return self._algebraic.table(context)
        return crypto_sbox(self.key, context)

    def n_tables(self) -> int:
        """How many distinct tables are in use at one instant."""
        return self.n_blocks if self.mode == "block" else 1

    def tables(self, frame_id: int) -> np.ndarray:
        """Shape ``(n_tables, 256)``; row ``b`` is the table of block ``b``."""
        key = 0 if self.mode == "session" else int(frame_id)
        if key not in self._cache:
            head = b"|" + self.session_id + b"|"
            if self.mode == "session":
                rows = [self._one(head + b"session")]
            elif self.mode == "frame":
                rows = [self._one(head + int(frame_id).to_bytes(8, "big"))]
            else:
                stem = head + int(frame_id).to_bytes(8, "big") + b"|blk"
                rows = [self._one(stem + i.to_bytes(4, "big"))
                        for i in range(self.n_blocks)]
            self._cache[key] = np.stack(rows).astype(np.uint8)
        return self._cache[key]

    def inverse_tables(self, frame_id: int) -> np.ndarray:
        t = self.tables(frame_id)
        return np.stack([invert_sbox(row) for row in t])

    def describe(self) -> Dict[str, object]:
        derivation = ("keyed Fisher-Yates over a ChaCha20 keystream"
                      if self.source == "random" else
                      "computed: inverse in GF(2^8) + affine map, with key "
                      "material XORed before and after")
        return {
            "primitive": "keyed value substitution (S-box) over 0..255",
            "mode": self.mode,
            "source": self.source,
            "tables_per_instant": self.n_tables(),
            "alphabet": ALPHABET,
            "table_space": ("256! per table" if self.source == "random"
                            else "one public table; 2^16 keyed variants per context"),
            "derivation": derivation,
            "note": ("bijective, so decryption is exact; a table is not "
                     "recoverable without the key, but see the measured "
                     "known-pair attack before treating that as confidentiality"),
        }


class SubstitutionPermutationScrambler:
    """Keyed substitution followed by the keyed block permutation of ``B2``.

    The permutation half is exactly ``B2``'s
    :class:`~avsec.baselines.CryptoPermutationScrambler`, so a comparison
    between ``B2`` and ``B2s`` isolates one change: the substitution.
    """

    def __init__(self, grid_rows: int, grid_cols: int, key: bytes,
                 session_id: bytes, mode: str = "block",
                 per_frame: bool = True, source: str = "random") -> None:
        from avsec.baselines import CryptoPermutationScrambler

        self.rows, self.cols = int(grid_rows), int(grid_cols)
        self.n_blocks = self.rows * self.cols
        self.permuter = CryptoPermutationScrambler(
            grid_rows, grid_cols, key, session_id, per_frame=per_frame)
        self.subst = SubstitutionTables(key, session_id, self.n_blocks, mode,
                                        source)
        self.mode = mode
        self.source = source

    # -- geometry, shared with the permutation half ------------------------
    def permutation(self, frame_id: int = 0) -> np.ndarray:
        return self.permuter.permutation(frame_id)

    def _tiles(self, img: np.ndarray) -> np.ndarray:
        return self.permuter._tiles(img)

    def _join(self, tiles: np.ndarray) -> np.ndarray:
        return self.permuter._join(tiles)

    def _fit(self, img: np.ndarray) -> np.ndarray:
        """The frame cropped to whole blocks, as the permutation half sees it."""
        return self._join(self._tiles(img))

    # -- the transform -----------------------------------------------------
    def substitute(self, img: np.ndarray, frame_id: int = 0) -> np.ndarray:
        """Substitution only, over the original block geometry."""
        tiles = self._tiles(img)
        tables = self.subst.tables(frame_id)
        if tables.shape[0] == 1:
            return self._join(tables[0][tiles.astype(np.int64)])
        idx = tiles.astype(np.int64)
        out = np.empty_like(tiles)
        for b in range(tiles.shape[0]):
            out[b] = tables[b][idx[b]]
        return self._join(out)

    def unsubstitute(self, img: np.ndarray, frame_id: int = 0) -> np.ndarray:
        tiles = self._tiles(img)
        inv = self.subst.inverse_tables(frame_id)
        if inv.shape[0] == 1:
            return self._join(inv[0][tiles.astype(np.int64)])
        idx = tiles.astype(np.int64)
        out = np.empty_like(tiles)
        for b in range(tiles.shape[0]):
            out[b] = inv[b][idx[b]]
        return self._join(out)

    def scramble(self, img: np.ndarray, frame_id: int = 0) -> np.ndarray:
        """Substitute values, then permute blocks."""
        return self.permuter.scramble(self.substitute(img, frame_id), frame_id)

    def descramble(self, img: np.ndarray, frame_id: int = 0) -> np.ndarray:
        """Undo the permutation, then undo the substitution."""
        return self.unsubstitute(self.permuter.descramble(img, frame_id), frame_id)

    def describe(self) -> Dict[str, object]:
        return {
            "scheme": "B2s",
            "primitives": ["keyed block permutation", "keyed value substitution"],
            "order": "substitute, then permute",
            "table_source": self.source,
            "grid": f"{self.rows}x{self.cols}",
            "n_blocks": self.n_blocks,
            "substitution": self.subst.describe(),
        }


# --------------------------------------------------------------- colour
def channel_correlation(img: np.ndarray) -> Dict[str, float]:
    """Pearson correlation between the colour planes of one frame.

    A natural photograph has strongly correlated channels: the same scene
    lights all three.  One table applied to all three would preserve that
    correlation exactly, so measuring it is how the per-channel construction is
    justified rather than asserted.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError("expected a HxWxC colour frame")
    names = RGBA_CHANNELS[: a.shape[2]]
    out: Dict[str, float] = {}
    for i in range(a.shape[2]):
        for j in range(i + 1, a.shape[2]):
            x, y = a[..., i].ravel(), a[..., j].ravel()
            sx, sy = x.std(), y.std()
            r = 0.0 if sx == 0 or sy == 0 else float(
                ((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
            out[f"{names[i]}{names[j]}"] = round(r, 4)
    return out


def channel_equality(img: np.ndarray) -> Dict[str, float]:
    """Fraction of pixels where two colour planes carry the same byte.

    This is the discriminator between a shared table and per-channel tables.
    Correlation is not: any substitution is a relabelling without order
    structure, so it collapses the linear relation between planes whichever
    table is used.  Equality is different - one table maps equal values to
    equal values, always.
    """
    a = np.asarray(img)
    if a.ndim != 3:
        raise ValueError("expected a HxWxC colour frame")
    names = RGBA_CHANNELS[: a.shape[2]]
    out: Dict[str, float] = {}
    for i in range(a.shape[2]):
        for j in range(i + 1, a.shape[2]):
            out[f"{names[i]}={names[j]}"] = round(
                float((a[..., i] == a[..., j]).mean()), 4)
    return out


class ColourSubstitutionPermutation:
    """``B2s`` over a colour frame: one substitution table per channel.

    The block permutation is **shared** by the channels by default, because the
    blocks are one image and moving them apart per channel would be a different
    transform rather than a stronger one.  The substitution tables are
    independent: each channel derives its own from the key, so a value that is
    the same in two channels does not encrypt to the same byte.

    ``per_channel_permutation=True`` is available for comparison; the bench
    measures what it changes instead of the documentation claiming it.
    """

    def __init__(self, grid_rows: int, grid_cols: int, key: bytes,
                 session_id: bytes, mode: str = "block", per_frame: bool = True,
                 source: str = "random", n_channels: int = 3,
                 per_channel_permutation: bool = False) -> None:
        from avsec.baselines import CryptoPermutationScrambler

        if n_channels not in (3, 4):
            raise ValueError("n_channels must be 3 (RGB) or 4 (RGBA)")
        self.rows, self.cols = int(grid_rows), int(grid_cols)
        self.n_blocks = self.rows * self.cols
        self.n_channels = int(n_channels)
        self.channels = RGBA_CHANNELS[:n_channels]
        self.mode = mode
        self.source = source
        self.per_channel_permutation = bool(per_channel_permutation)

        self.permuters = []
        for ch in self.channels:
            sid = session_id if not per_channel_permutation else (
                session_id[:7] + ch.encode()[:1])
            self.permuters.append(CryptoPermutationScrambler(
                grid_rows, grid_cols, key, sid, per_frame=per_frame))
        # one table set per channel: the channel tag enters the context, so the
        # three tables are independent draws from the same key
        self.subst = [
            SubstitutionTables(key, session_id + b"|" + ch.encode(),
                               self.n_blocks, mode, source)
            for ch in self.channels
        ]

    # -- geometry ----------------------------------------------------------
    def _fit(self, img: np.ndarray) -> np.ndarray:
        planes = [self._plane(img, c) for c in range(self.n_channels)]
        fitted = [self.permuters[c]._join(self.permuters[c]._tiles(p))
                  for c, p in enumerate(planes)]
        return np.stack(fitted, axis=2)

    def _plane(self, img: np.ndarray, c: int) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(img)[..., c])

    def permutation(self, frame_id: int = 0, channel: int = 0) -> np.ndarray:
        return self.permuters[channel].permutation(frame_id)

    def tables(self, frame_id: int = 0) -> Dict[str, np.ndarray]:
        """The substitution tables in use, one entry per channel."""
        return {ch: self.subst[c].tables(frame_id)
                for c, ch in enumerate(self.channels)}

    # -- the transform -----------------------------------------------------
    def _apply(self, plane: np.ndarray, c: int, frame_id: int,
               inverse: bool) -> np.ndarray:
        perm = self.permuters[c]
        tiles = perm._tiles(plane)
        tabs = (self.subst[c].inverse_tables(frame_id) if inverse
                else self.subst[c].tables(frame_id))
        idx = tiles.astype(np.int64)
        if tabs.shape[0] == 1:
            return perm._join(tabs[0][idx])
        out = np.empty_like(tiles)
        for b in range(tiles.shape[0]):
            out[b] = tabs[b][idx[b]]
        return perm._join(out)

    def substitute(self, img: np.ndarray, frame_id: int = 0) -> np.ndarray:
        planes = [self._apply(self._plane(img, c), c, frame_id, False)
                  for c in range(self.n_channels)]
        return np.stack(planes, axis=2)

    def scramble(self, img: np.ndarray, frame_id: int = 0) -> np.ndarray:
        sub = self.substitute(img, frame_id)
        planes = [self.permuters[c].scramble(sub[..., c], frame_id)
                  for c in range(self.n_channels)]
        return np.stack(planes, axis=2)

    def descramble(self, img: np.ndarray, frame_id: int = 0) -> np.ndarray:
        planes = []
        for c in range(self.n_channels):
            un = self.permuters[c].descramble(np.asarray(img)[..., c], frame_id)
            planes.append(self._apply(un, c, frame_id, True))
        return np.stack(planes, axis=2)

    def describe(self) -> Dict[str, object]:
        return {
            "scheme": "B2s-colour",
            "channels": list(self.channels),
            "bits_per_pixel": 8 * self.n_channels,
            "tables_per_frame": self.n_blocks * self.n_channels
                                if self.mode == "block" else self.n_channels,
            "table_shape": "16x16 = 256 значень, тобто всі варіації байта",
            "shared_permutation": not self.per_channel_permutation,
            "table_source": self.source,
            "substitution_mode": self.mode,
        }


def make_b2s_subst_perm(transport, channel_cfg, frame_h: int, frame_w: int,
                        grid_rows: int, grid_cols: int, master,
                        session_id: Optional[bytes] = None, mode: str = "block",
                        timer=None, session_ids=None, source: str = "random"):
    """Build ``B2s`` as an analog picture method, like ``B1`` and ``B2``."""
    from avsec.baselines import AnalogPictureMethod, SecureSessionIds
    from avsec.crypto import derive_session_keys

    sid = session_id or (session_ids or SecureSessionIds()).next("B2s")
    keys = derive_session_keys(master, sid)
    sc = SubstitutionPermutationScrambler(grid_rows, grid_cols, keys.key, sid,
                                          mode=mode, per_frame=True,
                                          source=source)
    method = AnalogPictureMethod(
        "B2s", transport, channel_cfg, frame_h, frame_w,
        transform=lambda img, fid: sc.scramble(img, fid),
        inverse=lambda img, fid: sc.descramble(img, fid),
        authenticated=False, timer=timer,
        notes=(f"keyed block permutation plus keyed value substitution "
               f"(mode={mode}, source={source}); two primitives, still no "
               f"integrity tag and no freshness counter"),
    )
    method.scrambler = sc
    return method


__all__ = [
    "ALPHABET", "SUBSTITUTION_MODES", "TABLE_SOURCES",
    "RGB_CHANNELS", "RGBA_CHANNELS", "channel_correlation",
    "channel_equality",
    "ColourSubstitutionPermutation",
    "crypto_sbox", "invert_sbox", "apply_sbox",
    "SubstitutionTables", "SubstitutionPermutationScrambler",
    "make_b2s_subst_perm",
]
