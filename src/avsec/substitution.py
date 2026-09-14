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
    "crypto_sbox", "invert_sbox", "apply_sbox",
    "SubstitutionTables", "SubstitutionPermutationScrambler",
    "make_b2s_subst_perm",
]
