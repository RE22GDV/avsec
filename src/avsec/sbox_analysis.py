"""Cryptographic properties of a substitution table, measured.

Two ways to build an S-box meet here.  A keyed random shuffle
(:mod:`avsec.substitution`) is secret but its resistance to differential and
linear analysis is only whatever the draw happened to give.  The algebraic
construction (:mod:`avsec.galois`) is public but its resistance is a theorem:
the inverse map in GF(2^8) has differential uniformity exactly 4 and
nonlinearity exactly 112 for eight bits.

This module computes the properties for any table, so the difference between
"on average" and "exactly" becomes a number in a table rather than a claim in
a sentence.

The four quantities
-------------------
``differential_uniformity``
    The largest number of inputs that share one input difference and one output
    difference.  Lower is better; 4 is the proven optimum reachable by the
    inverse map on eight bits, and the theoretical floor for any 8-bit
    permutation is 2, which is not attainable.

``nonlinearity``
    How far the table is from every affine function, via the Walsh-Hadamard
    transform.  Higher is better; 112 is what the inverse map achieves.

``algebraic_degree``
    The degree of the algebraic normal form of the worst component.  7 is the
    maximum for a permutation of eight bits.

``avalanche``
    Mean number of output bits that change when one input bit changes.  The
    ideal is half of eight, that is 4.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

ALPHABET = 256
BITS = 8


def _as_table(sbox) -> np.ndarray:
    t = np.asarray(sbox, dtype=np.int64).ravel()
    if t.size != ALPHABET:
        raise ValueError(f"an S-box must have {ALPHABET} entries, got {t.size}")
    return t


def is_bijective(sbox) -> bool:
    return bool(np.array_equal(np.sort(_as_table(sbox)), np.arange(ALPHABET)))


def differential_uniformity(sbox) -> int:
    """Largest cell of the difference distribution table, excluding ``a = 0``."""
    t = _as_table(sbox)
    x = np.arange(ALPHABET, dtype=np.int64)
    best = 0
    for a in range(1, ALPHABET):
        diffs = t[x ^ a] ^ t[x]
        best = max(best, int(np.bincount(diffs, minlength=ALPHABET).max()))
    return best


def _parity_matrix() -> np.ndarray:
    """``(-1)^popcount(a & x)`` for every pair, the Walsh kernel."""
    a = np.arange(ALPHABET, dtype=np.int64)[:, None]
    x = np.arange(ALPHABET, dtype=np.int64)[None, :]
    bits = np.bitwise_and(a, x)
    par = np.zeros_like(bits)
    for k in range(BITS):
        par ^= (bits >> k) & 1
    return np.where(par == 0, 1, -1).astype(np.int64)


def linearity(sbox) -> int:
    """Largest absolute Walsh coefficient over all non-trivial masks."""
    t = _as_table(sbox)
    kernel = _parity_matrix()
    best = 0
    for b in range(1, ALPHABET):
        comp = np.zeros(ALPHABET, dtype=np.int64)
        masked = t & b
        for k in range(BITS):
            comp ^= (masked >> k) & 1
        signs = np.where(comp == 0, 1, -1).astype(np.int64)
        w = kernel @ signs
        best = max(best, int(np.abs(w).max()))
    return best


def nonlinearity(sbox) -> int:
    """``2^(n-1) - max|W|/2``; 112 for the inverse map on eight bits."""
    return (ALPHABET // 2) - linearity(sbox) // 2


def algebraic_degree(sbox) -> int:
    """Highest algebraic normal form degree over the component functions."""
    t = _as_table(sbox)
    popcount = np.array([bin(i).count("1") for i in range(ALPHABET)],
                        dtype=np.int64)
    best = 0
    for b in range(1, ALPHABET):
        comp = np.zeros(ALPHABET, dtype=np.int64)
        masked = t & b
        for k in range(BITS):
            comp ^= (masked >> k) & 1
        anf = comp.copy()                      # Moebius transform in place
        step = 1
        while step < ALPHABET:
            for start in range(0, ALPHABET, step * 2):
                anf[start + step:start + 2 * step] ^= anf[start:start + step]
            step *= 2
        present = np.flatnonzero(anf)
        if present.size:
            best = max(best, int(popcount[present].max()))
    return best


def fixed_points(sbox) -> int:
    t = _as_table(sbox)
    return int((t == np.arange(ALPHABET)).sum())


def opposite_fixed_points(sbox) -> int:
    """Inputs with ``S(v) = v XOR 0xFF``; the AES affine map removes these too."""
    t = _as_table(sbox)
    return int((t == (np.arange(ALPHABET) ^ 0xFF)).sum())


def avalanche(sbox) -> float:
    """Mean output bits changed by a one-bit input change; the ideal is 4."""
    t = _as_table(sbox)
    x = np.arange(ALPHABET, dtype=np.int64)
    total = 0.0
    for k in range(BITS):
        d = t[x ^ (1 << k)] ^ t[x]
        for j in range(BITS):
            total += float(((d >> j) & 1).sum())
    return total / (BITS * ALPHABET)


def analyse(sbox, label: str = "") -> Dict[str, object]:
    """Every property above for one table, as one row."""
    return {
        "table": label,
        "bijective": is_bijective(sbox),
        "differential_uniformity": differential_uniformity(sbox),
        "nonlinearity": nonlinearity(sbox),
        "linearity": linearity(sbox),
        "algebraic_degree": algebraic_degree(sbox),
        "fixed_points": fixed_points(sbox),
        "opposite_fixed_points": opposite_fixed_points(sbox),
        "avalanche_bits": round(avalanche(sbox), 4),
    }


#: What the algebraic construction attains, for putting a measurement next to.
ALGEBRAIC_OPTIMUM: Dict[str, object] = {
    "differential_uniformity": 4,
    "nonlinearity": 112,
    "algebraic_degree": 7,
    "fixed_points": 0,
    "opposite_fixed_points": 0,
    "avalanche_bits": 4.0,
}


def random_table_reference(n_samples: int = 64, seed: int = 20240909
                           ) -> Dict[str, object]:
    """What a *random* bijective table gives, as a distribution rather than one draw.

    A single keyed table says nothing about the construction: the next key gives
    another draw.  Sampling many of them is what turns "the random table is
    worse" into a measured spread with a best case.
    """
    rng = np.random.default_rng(seed)
    du, nl = [], []
    for _ in range(n_samples):
        t = rng.permutation(ALPHABET)
        du.append(differential_uniformity(t))
        nl.append(nonlinearity(t))
    return {
        "n_samples": n_samples,
        "differential_uniformity_min": int(np.min(du)),
        "differential_uniformity_median": float(np.median(du)),
        "differential_uniformity_max": int(np.max(du)),
        "nonlinearity_min": int(np.min(nl)),
        "nonlinearity_median": float(np.median(nl)),
        "nonlinearity_max": int(np.max(nl)),
        "note": ("випадкова таблиця дає розподіл, а не значення: наступний "
                 "ключ дає інший розіграш. Алгебраїчна конструкція дає те "
                 "саме число завжди"),
    }


__all__ = [
    "ALPHABET", "BITS", "ALGEBRAIC_OPTIMUM", "is_bijective",
    "differential_uniformity", "linearity", "nonlinearity", "algebraic_degree",
    "fixed_points", "opposite_fixed_points", "avalanche", "analyse",
    "random_table_reference",
]
