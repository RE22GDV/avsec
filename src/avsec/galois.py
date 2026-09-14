"""GF(2^8): the arithmetic behind a computed substitution table.

Why a second way to build the table
-----------------------------------
:mod:`avsec.substitution` builds its S-box by shuffling ``0..255`` with a keyed
generator.  That table is secret, and it has to be stored: 256 bytes, which is
nothing for a computer.  In hardware it is not nothing - a table is a ROM plus
an address decoder, and a round that needs sixteen of them in parallel pays for
sixteen copies.

The other construction computes the table instead.  Every byte is read as a
polynomial over GF(2), the 256 bytes form the field GF(2^8) modulo an
irreducible polynomial, and the table is

    S(v) = affine( v^-1 ),        with 0^-1 defined as 0

The inverse is the only nonlinear step and is what resists differential and
linear analysis; the affine map after it removes fixed points and breaks the
too-simple algebraic form.

The crucial difference, and the mistake it invites
--------------------------------------------------
**A computed table is a fixed table.**  If a formula produces it without a key,
the same formula produces it for everyone - the AES S-box is printed in the
standard.  Confidentiality in AES comes from adding round keys, never from the
S-box.  A scheme that replaces a keyed random table with the AES table and
changes nothing else has a *public* substitution and gains no secrecy at all.

:class:`KeyedAlgebraicSbox` therefore keeps the algebra and puts the key back
where it belongs: ``S_k(v) = S(v XOR k1) XOR k2``.  That is an affine-equivalent
transform, so it provably keeps the differential and linear properties of the
algebraic table, which :mod:`avsec.sbox_analysis` then measures rather than
assumes.

Three appearances of the same algebra in this project
-----------------------------------------------------
* the LFSR of ``B1`` uses a **primitive** polynomial over GF(2), which is what
  gives it the maximal period ``2^16 - 1``;
* Reed-Solomon in ``B3``/``B4`` is a code over **GF(256)** - the same field as
  here, and it needs a primitive element as generator;
* the algebraic S-box needs the multiplicative inverse in GF(2^8).

Irreducible is what makes the quotient a field; primitive is the stronger
property that a root generates all 255 non-zero elements.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

#: The reduction polynomial of AES: x^8 + x^4 + x^3 + x + 1.
AES_POLY = 0x11B

#: The affine constant of the AES S-box.
AES_AFFINE_CONST = 0x63

#: Rotation amounts of the AES affine map, as a circulant over GF(2).
AES_AFFINE_ROTATIONS: Tuple[int, ...] = (0, 1, 2, 3, 4)

#: Rotations and constant of the inverse affine map.
AES_INV_AFFINE_ROTATIONS: Tuple[int, ...] = (1, 3, 6)
AES_INV_AFFINE_CONST = 0x05

#: The published AES S-box, used only to check that our field arithmetic is
#: right.  Reproducing a table someone else printed is the strongest available
#: test of a GF(2^8) implementation.
AES_SBOX_REFERENCE: Tuple[int, ...] = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

#: Every irreducible polynomial of degree 8 over GF(2), as 9-bit integers.
#: There are 30 of them; the field is a *choice*, and a different choice gives a
#: different table.  Computed by :func:`irreducible_polynomials`.
_IRREDUCIBLE_CACHE: Optional[Tuple[int, ...]] = None


def _clmul(a: int, b: int) -> int:
    """Carry-less multiplication of two polynomials over GF(2)."""
    out = 0
    while b:
        if b & 1:
            out ^= a
        a <<= 1
        b >>= 1
    return out


def _polydiv_remainder(a: int, m: int) -> int:
    """Remainder of ``a`` modulo the polynomial ``m``, both over GF(2)."""
    deg_m = m.bit_length() - 1
    while a.bit_length() - 1 >= deg_m and a:
        a ^= m << (a.bit_length() - 1 - deg_m)
    return a


def is_irreducible(poly: int) -> bool:
    """Trial division by every polynomial of degree at most ``deg/2``."""
    deg = poly.bit_length() - 1
    if deg < 1 or not poly & 1:      # divisible by x
        return False
    for d in range(1, deg // 2 + 1):
        for cand in range(1 << d, 1 << (d + 1)):
            if _polydiv_remainder(poly, cand) == 0:
                return False
    return True


def irreducible_polynomials(degree: int = 8) -> Tuple[int, ...]:
    """All irreducible polynomials of the given degree over GF(2)."""
    global _IRREDUCIBLE_CACHE
    if degree == 8 and _IRREDUCIBLE_CACHE is not None:
        return _IRREDUCIBLE_CACHE
    out = tuple(p for p in range(1 << degree, 1 << (degree + 1))
                if is_irreducible(p))
    if degree == 8:
        _IRREDUCIBLE_CACHE = out
    return out


class GF256:
    """Arithmetic in GF(2^8) for one explicit reduction polynomial.

    The exponential and logarithm tables are built from a generator of the
    multiplicative group, which is found by search rather than assumed: not
    every field element is primitive, and not every irreducible polynomial has
    ``x`` itself as a primitive root.
    """

    def __init__(self, poly: int = AES_POLY, generator: Optional[int] = None) -> None:
        if not is_irreducible(poly):
            raise ValueError(f"0x{poly:X} is not irreducible over GF(2); "
                             f"the quotient would not be a field")
        self.poly = int(poly)
        self.generator = int(generator) if generator else self._find_generator()
        self.exp = np.zeros(512, dtype=np.int64)
        self.log = np.zeros(256, dtype=np.int64)
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x = self.mul(x, self.generator)
        self.exp[255:510] = self.exp[0:255]

    # -- the primitive operations ----------------------------------------
    def mul(self, a: int, b: int) -> int:
        """Multiply in the field: carry-less product reduced by the polynomial."""
        return _polydiv_remainder(_clmul(int(a), int(b)), self.poly)

    def order(self, a: int) -> int:
        """Multiplicative order of ``a``; 255 means ``a`` is primitive."""
        if a == 0:
            return 0
        n, x = 1, a
        while x != 1:
            x = self.mul(x, a)
            n += 1
            if n > 255:
                return -1
        return n

    def is_primitive(self, a: int) -> bool:
        return self.order(a) == 255

    def _find_generator(self) -> int:
        for cand in range(2, 256):
            if self.order(cand) == 255:
                return cand
        raise ValueError("no primitive element found; this cannot happen "
                         "for an irreducible polynomial")

    def inv(self, a: int) -> int:
        """Multiplicative inverse, with ``0`` mapped to ``0`` by convention.

        Zero has no inverse in a field.  Every S-box of this shape defines
        ``S(0)`` by convention instead, and the convention is part of the
        specification rather than an accident.
        """
        a = int(a)
        if a == 0:
            return 0
        return int(self.exp[255 - self.log[a]])

    def inverse_table(self) -> np.ndarray:
        """The full inverse map as a 256-entry table."""
        return np.array([self.inv(v) for v in range(256)], dtype=np.uint8)

    def describe(self) -> Dict[str, object]:
        deg = self.poly.bit_length() - 1
        terms = [f"x^{i}" if i > 1 else ("x" if i == 1 else "1")
                 for i in range(deg, -1, -1) if (self.poly >> i) & 1]
        return {
            "field": f"GF(2^{deg})",
            "polynomial_hex": f"0x{self.poly:X}",
            "polynomial": " + ".join(terms),
            "irreducible": True,
            "generator": f"0x{self.generator:02X}",
            "generator_is_primitive": self.is_primitive(self.generator),
            "n_irreducible_of_this_degree": len(irreducible_polynomials(deg)),
        }


# ------------------------------------------------------------ affine layer
def rotl8(x: int, n: int) -> int:
    """Rotate a byte left by ``n`` bits."""
    n &= 7
    return ((x << n) | (x >> (8 - n))) & 0xFF


def affine(x: int, rotations: Tuple[int, ...] = AES_AFFINE_ROTATIONS,
           constant: int = AES_AFFINE_CONST) -> int:
    """The circulant affine map over GF(2), as a XOR of rotations."""
    out = 0
    for r in rotations:
        out ^= rotl8(x, r)
    return out ^ constant


def algebraic_sbox(gf: Optional[GF256] = None,
                   rotations: Tuple[int, ...] = AES_AFFINE_ROTATIONS,
                   constant: int = AES_AFFINE_CONST) -> np.ndarray:
    """``S(v) = affine(v^-1)`` - the table computed, not stored.

    With the default field and affine map this reproduces the AES S-box byte
    for byte, which is how :mod:`tests.test_galois` checks the arithmetic.
    """
    gf = gf or GF256()
    return np.array([affine(gf.inv(v), rotations, constant) for v in range(256)],
                    dtype=np.uint8)


def inverse_algebraic_sbox(gf: Optional[GF256] = None,
                           rotations: Tuple[int, ...] = AES_INV_AFFINE_ROTATIONS,
                           constant: int = AES_INV_AFFINE_CONST) -> np.ndarray:
    """``S^-1(c) = (inverse_affine(c))^-1``, computed the same way."""
    gf = gf or GF256()
    return np.array([gf.inv(affine(c, rotations, constant)) for c in range(256)],
                    dtype=np.uint8)


# ------------------------------------------------- putting the key back in
class KeyedAlgebraicSbox:
    """``S_k(v) = S(v XOR k_in) XOR k_out`` over the computed table.

    The algebraic table is public, so on its own it hides nothing.  Adding key
    material before and after is how a block cipher uses a published S-box, and
    it is an affine-equivalent change: the differential and linear properties of
    ``S`` are preserved exactly.  :mod:`avsec.sbox_analysis` measures that
    instead of taking it on trust.

    Only the table is public; ``k_in`` and ``k_out`` come from the session key,
    so the resulting map is still key-only.
    """

    def __init__(self, key: bytes, gf: Optional[GF256] = None) -> None:
        self.gf = gf or GF256()
        self.key = key
        self.base = algebraic_sbox(self.gf)
        self._cache: Dict[bytes, np.ndarray] = {}

    def whitening(self, context: bytes) -> Tuple[int, int]:
        """The two key bytes for this context, derived like every other key."""
        import hashlib

        d = hashlib.sha256(b"avsec/gf-sbox|" + self.key + b"|" + context).digest()
        return int(d[0]), int(d[1])

    def table(self, context: bytes) -> np.ndarray:
        if context not in self._cache:
            k_in, k_out = self.whitening(context)
            idx = np.arange(256, dtype=np.int64) ^ k_in
            self._cache[context] = (self.base[idx] ^ k_out).astype(np.uint8)
        return self._cache[context]

    def inverse_table(self, context: bytes) -> np.ndarray:
        t = self.table(context)
        inv = np.empty(256, dtype=np.uint8)
        inv[t.astype(np.int64)] = np.arange(256, dtype=np.uint8)
        return inv

    def describe(self) -> Dict[str, object]:
        return {
            "construction": "S_k(v) = S(v XOR k_in) XOR k_out",
            "base_table": "algebraic, public, computed from the field",
            "secret_part": "two key bytes per context",
            "field": self.gf.describe(),
            "note": ("the algebraic table alone is public and hides nothing; "
                     "the key enters exactly where a block cipher puts it"),
        }


__all__ = [
    "AES_POLY", "AES_AFFINE_CONST", "AES_AFFINE_ROTATIONS",
    "AES_INV_AFFINE_ROTATIONS", "AES_INV_AFFINE_CONST", "AES_SBOX_REFERENCE",
    "GF256", "is_irreducible", "irreducible_polynomials", "rotl8", "affine",
    "algebraic_sbox", "inverse_algebraic_sbox", "KeyedAlgebraicSbox",
]
