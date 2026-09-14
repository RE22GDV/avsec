"""Bench for luminance-balanced encryption: the two paths, measured separately.

The runner behind ``docs/luma_balance.md``.  It keeps the monochrome and the
colour path apart throughout, because they are not two settings of one scheme:
counting the available colours shows that one fits and the other does not, and
every later number follows from that.

Sections, in the order the document uses them:

1. **Capacity.**  How many 8-bit colours share one luminance level, swept over
   all levels.  This is the number that decides whether a path is lossless.
2. **Monochrome path.**  Exact recovery, and what is left in the luminance
   plane of the ciphertext.
3. **Colour path.**  The bits that do not fit, and the quality cost of losing
   them, separated from any cost of the cipher itself.
4. **The luminance-only observer.**  What the analog path modelled in this
   project - which carries luma and nothing else - would deliver.
5. **Attacks.**  The key-free reassembly attack pointed at the luminance plane
   and at the chroma plane of the same ciphertext.
6. **Noise tolerance.**  What a perturbed ciphertext decodes to.

Usage::

    avsec luma-lab                  # -> results/luma/
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from avsec.config import ExperimentConfig
from avsec.luma_balance import (
    BEST_LEVEL,
    LumaBalancedColour,
    LumaBalancedMono,
    chroma_planes,
    constant_luma_palette,
    luma,
    luma_statistics,
    palette_capacity,
)
from avsec.utils import ensure_dir, environment_record, write_csv, write_json

BT601_R, BT601_G, BT601_B = 0.299, 0.587, 0.114

#: Amplitude perturbations used for the noise-tolerance section, in levels.
NOISE_SIGMAS: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)

#: Finer grid for the deeper analysis.
FINE_SIGMAS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
                                  4.0, 6.0, 8.0, 12.0, 16.0)

#: Perturbations an analog path actually applies, beyond additive noise.
IMPAIRMENTS: Tuple[str, ...] = ("gauss", "uniform", "chroma_lowpass",
                                "gain_offset", "burst")

#: Noise levels used for the picture strip.
STRIP_SIGMAS: Tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = float(((np.asarray(a, dtype=np.float64)
                - np.asarray(b, dtype=np.float64)) ** 2).mean())
    return 99.0 if m <= 1e-12 else float(10 * np.log10(255.0 ** 2 / m))


def _entropy(x: np.ndarray) -> float:
    h = np.bincount(np.rint(np.asarray(x)).astype(np.int64).ravel() & 0xFF,
                    minlength=256).astype(np.float64)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


# ------------------------------------------------------------ 1. capacity
def capacity_sweep(step: int = 8) -> List[Dict[str, Any]]:
    """Colours per luminance level, and what each level can carry."""
    r = np.arange(256, dtype=np.float64)
    y = (0.299 * r[:, None, None] + 0.587 * r[None, :, None]
         + 0.114 * r[None, None, :])
    counts = np.bincount(np.rint(y).astype(np.int64).ravel(), minlength=256)
    out: List[Dict[str, Any]] = []
    for level in list(range(0, 256, step)) + [int(counts.argmax())]:
        n = int(counts[level])
        out.append({
            "luma_level": int(level),
            "colours": n,
            "capacity_bits": round(float(np.log2(n)), 3) if n else 0.0,
            "fits_monochrome_8_bit": bool(n >= 256),
            "fits_colour_24_bit": bool(n >= 2 ** 24),
            "usable_bits": int(np.floor(np.log2(n))) if n else 0,
        })
    out.sort(key=lambda d: d["luma_level"])
    return out


# ---------------------------------------------------------- 2. monochrome
def monochrome_path(key: bytes, sid: bytes, frames: List[np.ndarray],
                    level: int = BEST_LEVEL) -> Tuple[List[Dict[str, Any]],
                                                      Dict[str, np.ndarray]]:
    """Exactness of the lossless path, and the luma left in the ciphertext."""
    mono = LumaBalancedMono(key, sid, level)
    rows: List[Dict[str, Any]] = []
    images: Dict[str, np.ndarray] = {}
    for i, img in enumerate(frames):
        ct = mono.encrypt(img, i)
        back = mono.decrypt(ct, i)
        st = luma_statistics(ct)
        rows.append({
            "frame": i,
            "bit_exact_recovery": bool(np.array_equal(back, img)),
            "source_luma_entropy_bits": round(_entropy(img), 4),
            "cipher_luma_entropy_bits": st["luma_entropy_bits"],
            "cipher_luma_levels_used": st["rounded_levels_used"],
            "cipher_luma_std": st["luma_std"],
            "cipher_luma_min": st["luma_min"],
            "cipher_luma_max": st["luma_max"],
        })
        if i == 0:
            pl = chroma_planes(ct)
            images["mono_1_source"] = img
            images["mono_2_cipher_rgb"] = ct
            images["mono_3_cipher_as_luma"] = np.rint(pl["Y"]).astype(np.uint8)
            images["mono_4_cipher_cb"] = _to_grey(pl["Cb"])
            images["mono_5_cipher_cr"] = _to_grey(pl["Cr"])
            images["mono_6_restored"] = back
    return rows, images


def _to_grey(plane: np.ndarray) -> np.ndarray:
    """A signed plane rendered for viewing, with its own range stated by use."""
    a = np.asarray(plane, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.full(a.shape, 128, dtype=np.uint8)
    return np.rint((a - lo) / (hi - lo) * 255).astype(np.uint8)


# -------------------------------------------------------------- 3. colour
def colour_path(key: bytes, sid: bytes, rgb: np.ndarray,
                level: int = BEST_LEVEL) -> Tuple[List[Dict[str, Any]],
                                                  Dict[str, np.ndarray]]:
    """The bits that do not fit, and the cost of losing them."""
    rows: List[Dict[str, Any]] = []
    images: Dict[str, np.ndarray] = {}
    for bits in ((5, 6, 5), (5, 5, 5), (4, 4, 4)):
        col = LumaBalancedColour(key, sid, level, bits)
        ct = col.encrypt(rgb, 0)
        back = col.decrypt(ct, 0)
        q = col.quantise(rgb)
        st = luma_statistics(ct)
        rows.append({
            "bits_kept": "-".join(str(b) for b in bits),
            "bits_kept_total": sum(bits),
            "bits_lost": 24 - sum(bits),
            "codewords": col.n_codewords,
            "cipher_equals_quantised_source": bool(np.array_equal(back, q)),
            "psnr_recovered_db": round(_psnr(back, rgb), 2),
            "psnr_quantisation_only_db": round(_psnr(q, rgb), 2),
            "cipher_luma_entropy_bits": st["luma_entropy_bits"],
            "cipher_luma_levels_used": st["rounded_levels_used"],
        })
        if bits == (5, 6, 5):
            pl = chroma_planes(ct)
            images["colour_1_source"] = rgb
            images["colour_2_cipher_rgb"] = ct
            images["colour_3_cipher_as_luma"] = np.rint(pl["Y"]).astype(np.uint8)
            images["colour_4_cipher_cb"] = _to_grey(pl["Cb"])
            images["colour_5_restored"] = back
    return rows, images


# ------------------------------------------- 4. the luminance-only observer
def luma_only_observer(key: bytes, sid: bytes, img: np.ndarray,
                       level: int = BEST_LEVEL) -> Dict[str, Any]:
    """What survives if only the luminance plane reaches the receiver.

    The analog path modelled in this project is monochrome by construction -
    no colour subcarrier, no burst, no PAL phase alternation.  A ciphertext of
    constant luminance therefore arrives as a flat field.  This measures how
    much information that field carries, rather than asserting that it carries
    none.
    """
    mono = LumaBalancedMono(key, sid, level)
    ct = mono.encrypt(img, 0)
    y = np.rint(luma(ct)).astype(np.uint8)
    wrong = mono.decrypt(np.stack([y, y, y], axis=2), 0)
    return {
        "source_entropy_bits": round(_entropy(img), 4),
        "cipher_luma_entropy_bits": round(_entropy(y), 6),
        "cipher_luma_distinct_values": int(np.unique(y).size),
        "psnr_of_luma_only_decode_db": round(_psnr(wrong, img), 2),
        "correlation_luma_vs_source": round(
            float(np.corrcoef(y.ravel().astype(np.float64),
                              img.ravel().astype(np.float64))[0, 1])
            if y.std() > 0 else 0.0, 6),
        "note": ("монохромний тракт цього проєкту переносить лише яскравість; "
                 "за нульової ентропії яскравості він переносить нуль бітів"),
    }


# ------------------------------------------------------------- 5. attacks
def plane_attacks(cfg: ExperimentConfig, key: bytes, sid: bytes,
                  img: np.ndarray, level: int = BEST_LEVEL
                  ) -> List[Dict[str, Any]]:
    """The key-free reassembly attack on each plane of the same ciphertext.

    Luminance balancing is applied *after* the block permutation, so the
    comparison is between three views of one ciphertext: the permuted
    greyscale that ``B2`` would transmit, the luminance plane of the balanced
    ciphertext, and its chroma plane.
    """
    from avsec.attacks import attack_boundary_reassembly
    from avsec.baselines import CryptoPermutationScrambler

    rows, cols = cfg.b2_grid
    perm = CryptoPermutationScrambler(rows, cols, key, sid)
    fitted = perm._join(perm._tiles(img))
    scrambled = perm.scramble(img, 0)
    keyed = chroma_planes(LumaBalancedMono(key, sid, level).encrypt(scrambled, 0))
    # control: the same luminance constraint with a structure-preserving
    # assignment, so the contribution of the keying can be separated from the
    # contribution of the constant-luminance constraint itself
    ordered = chroma_planes(
        LumaBalancedMono(key, sid, level, keyed=False).encrypt(scrambled, 0))

    views = [
        ("B2: перемішана яскравість", scrambled, ""),
        ("B2l ключова: яскравість", np.rint(keyed["Y"]).astype(np.uint8), "ключова"),
        ("B2l ключова: Cb", _to_grey(keyed["Cb"]), "ключова"),
        ("B2l ключова: Cr", _to_grey(keyed["Cr"]), "ключова"),
        ("B2l впорядкована: яскравість", np.rint(ordered["Y"]).astype(np.uint8),
         "впорядкована"),
        ("B2l впорядкована: Cb", _to_grey(ordered["Cb"]), "впорядкована"),
        ("B2l впорядкована: Cr", _to_grey(ordered["Cr"]), "впорядкована"),
    ]
    out: List[Dict[str, Any]] = []
    src = np.asarray(fitted, dtype=np.float64).ravel()
    for label, view, codebook in views:
        ba = attack_boundary_reassembly(view, rows, cols, perm.permutation(0),
                                        fitted)
        v = np.asarray(view, dtype=np.float64).ravel()
        corr = (0.0 if v.std() == 0 else
                float(np.corrcoef(v, src)[0, 1]))
        out.append({
            "view": label,
            "codebook": codebook,
            "correlation_with_source": round(corr, 4),
            "neighbour_accuracy_pct": round(ba.metrics["neighbour_accuracy"] * 100, 2),
            "direct_accuracy_pct": round(ba.metrics["direct_accuracy"] * 100, 2),
            "reconstruction_psnr_db": round(ba.metrics["reconstruction_psnr_db"], 2),
            "plane_std": round(float(np.asarray(view, dtype=np.float64).std()), 4),
            "seconds": round(ba.seconds, 4),
        })
    return out


# ------------------------------------------------------- 6. noise tolerance
def noise_tolerance(key: bytes, sid: bytes, img: np.ndarray,
                    level: int = BEST_LEVEL,
                    sigmas: Tuple[float, ...] = NOISE_SIGMAS
                    ) -> List[Dict[str, Any]]:
    """Decoding a perturbed ciphertext by nearest codeword.

    A codebook has no order structure, so a small amplitude error does not give
    a slightly wrong value - it gives an unrelated one, exactly as measured for
    the substitution table of ``B2s``.
    """
    mono = LumaBalancedMono(key, sid, level)
    ct = mono.encrypt(img, 0).astype(np.float64)
    rng = np.random.default_rng(20240909)
    out: List[Dict[str, Any]] = []
    for s in sigmas:
        noisy = np.clip(ct + (rng.normal(0, s, ct.shape) if s > 0 else 0),
                        0, 255).astype(np.uint8)
        got = mono.decrypt(noisy, 0, nearest=True)
        out.append({
            "sigma_levels": s,
            "exact_pixels_pct": round(float((got == img).mean() * 100), 2),
            "psnr_db": round(_psnr(got, img), 2),
            "usable_at_20db": bool(_psnr(got, img) >= 20.0),
        })
    return out


# ------------------------------------------------- 7. deeper noise analysis
def codeword_spacing(n_codewords: int, level: int = BEST_LEVEL) -> Dict[str, Any]:
    """Distance between neighbouring codewords - what decides tolerance.

    A codebook drawn from a palette of fixed size gets sparser as fewer
    codewords are taken.  The nearest-neighbour distance is therefore the
    quantity that predicts how much amplitude error a decoder can undo, and it
    differs between the monochrome and the colour path by construction.
    """
    from scipy.spatial import cKDTree

    pal = constant_luma_palette(level)
    picked = pal[np.linspace(0, pal.shape[0] - 1, n_codewords).astype(np.int64)]
    d, _ = cKDTree(picked.astype(np.float64)).query(picked.astype(np.float64), k=2)
    nn = d[:, 1]
    return {
        "codewords": int(n_codewords),
        "palette": int(pal.shape[0]),
        "min_distance": round(float(nn.min()), 3),
        "median_distance": round(float(np.median(nn)), 3),
        "mean_distance": round(float(nn.mean()), 3),
        "half_min_distance": round(float(nn.min()) / 2, 3),
    }


def _impair(ct: np.ndarray, kind: str, strength: float,
            rng: np.random.Generator) -> np.ndarray:
    """One perturbation of the ciphertext, in the units its name implies."""
    a = np.asarray(ct, dtype=np.float64)
    if strength <= 0:
        return np.clip(a, 0, 255).astype(np.uint8)
    if kind == "gauss":
        a = a + rng.normal(0, strength, a.shape)
    elif kind == "uniform":
        a = a + rng.uniform(-strength, strength, a.shape)
    elif kind == "chroma_lowpass":
        # a composite path carries chroma at a fraction of the luma bandwidth;
        # the message of this scheme lives entirely in chroma
        y = luma(a)
        cb, cr = a[..., 2] - y, a[..., 0] - y
        k = max(1, int(round(strength)))
        ker = np.ones(k) / k
        for plane in (cb, cr):
            for r in range(plane.shape[0]):
                plane[r] = np.convolve(plane[r], ker, mode="same")
        r_ = y + cr
        b_ = y + cb
        g_ = (y - BT601_R * r_ - BT601_B * b_) / BT601_G
        a = np.stack([r_, g_, b_], axis=2)
    elif kind == "gain_offset":
        gain = 1.0 + rng.normal(0, strength / 100.0, 3)
        off = rng.normal(0, strength, 3)
        a = a * gain[None, None, :] + off[None, None, :]
    elif kind == "burst":
        rows = int(round(strength))
        h = a.shape[0]
        for _ in range(max(1, h // 48)):
            r0 = int(rng.integers(0, max(1, h - rows)))
            a[r0:r0 + rows] = rng.integers(0, 256, (min(rows, h - r0),
                                                    a.shape[1], 3))
    else:
        raise ValueError(f"unknown impairment {kind!r}")
    return np.clip(a, 0, 255).astype(np.uint8)


def noise_deep(key: bytes, sid: bytes, img: np.ndarray,
               level: int = BEST_LEVEL) -> List[Dict[str, Any]]:
    """Every impairment against both decoders, with the error profile.

    ``mean_abs_error`` separates a degraded picture from a destroyed one: a
    codebook has no order, so a pixel that decodes to the wrong codeword is
    wrong by an arbitrary amount rather than by a little.
    """
    mono = LumaBalancedMono(key, sid, level)
    ct = mono.encrypt(img, 0)
    rng = np.random.default_rng(20240909)
    out: List[Dict[str, Any]] = []
    grid = {"gauss": FINE_SIGMAS, "uniform": FINE_SIGMAS,
            "chroma_lowpass": (0, 2, 3, 4, 6, 8, 12, 16),
            "gain_offset": (0.0, 0.5, 1.0, 2.0, 4.0, 8.0),
            "burst": (0, 2, 4, 8, 16, 24)}
    for kind in IMPAIRMENTS:
        for strength in grid[kind]:
            bad = _impair(ct, kind, float(strength), rng)
            st = luma_statistics(bad)
            row: Dict[str, Any] = {
                "impairment": kind,
                "strength": float(strength),
                "cipher_luma_std_after": st["luma_std"],
                "cipher_luma_levels_after": st["rounded_levels_used"],
            }
            for tag, nearest in (("exact", False), ("nearest", True)):
                got = mono.decrypt(bad, 0, nearest=nearest)
                err = np.abs(got.astype(np.int64) - img.astype(np.int64))
                row[f"{tag}_pixels_pct"] = round(float((err == 0).mean() * 100), 2)
                row[f"{tag}_psnr_db"] = round(_psnr(got, img), 2)
                row[f"{tag}_mean_abs_error"] = round(float(err.mean()), 2)
            out.append(row)
    return out


def failure_geometry(key: bytes, sid: bytes, img: np.ndarray,
                     level: int = BEST_LEVEL) -> List[Dict[str, Any]]:
    """Three mechanisms behind the numbers of the impairment sections.

    The impairment tables say how much each perturbation costs.  They do not
    say why the costs differ so much in *character*, and that question has a
    geometric answer that can be measured rather than asserted.

    ``напрям``
        The constant-luminance constraint is one linear equation, so the
        palette lies in a plane of RGB space.  Decoding compares distances to
        points of that plane, which means the component of the error along the
        plane normal is projected away and only the in-plane component can move
        a point to a different codeword.  Both components are applied on their
        own, rescaled to the same RMS, so the comparison is between directions
        and not between amounts.
    ``змішування``
        A smoothing kernel replaces a pixel by the mean of its window.  A mean
        of points of the plane is again a point of the plane, so the result is
        a legal ciphertext value belonging to an unrelated message value.  The
        prediction is that only pixels whose window carried a single value can
        survive, and both groups are counted separately here.
    ``величина``
        A wrong codeword carries an unrelated value, so the size of a single
        failure should match what a uniformly drawn value would give.  The
        reference is computed from the frame itself rather than assumed.
    """
    mono = LumaBalancedMono(key, sid, level)
    ct = mono.encrypt(img, 0)
    src = np.asarray(img, dtype=np.int64)
    rng = np.random.default_rng(90210)
    normal = np.array([BT601_R, BT601_G, BT601_B], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    out: List[Dict[str, Any]] = []

    def _decode(bad: np.ndarray) -> np.ndarray:
        return mono.decrypt(bad, 0, nearest=True).astype(np.int64)

    # -- 1. direction: the same amount of error, aimed two ways ------------
    for sigma in (0.5, 1.0, 2.0, 4.0, 8.0):
        e = rng.normal(0.0, sigma, ct.shape)
        along = (e * normal[None, None, :]).sum(axis=2)
        e_normal = along[..., None] * normal[None, None, :]
        e_plane = e - e_normal
        target = float(np.sqrt((e ** 2).mean()))
        parts = {"ізотропний": e,
                 "у площині палітри": e_plane,
                 "поперек площини палітри": e_normal}
        for label, vec in parts.items():
            rms = float(np.sqrt((vec ** 2).mean()))
            scaled = vec if rms <= 1e-12 else vec * (target / rms)
            bad = np.clip(np.asarray(ct, dtype=np.float64) + scaled,
                          0, 255).astype(np.uint8)
            got = _decode(bad)
            out.append({
                "mechanism": "напрям",
                "case": label,
                "strength": float(sigma),
                "applied_rms_levels": round(float(np.sqrt((scaled ** 2).mean())), 3),
                "exact_pixels_pct": round(float((got == src).mean() * 100), 2),
                "psnr_db": round(_psnr(got.astype(np.uint8), img), 2),
            })

    # -- 2. mixing: only a window of one value can survive -----------------
    for k in (2, 3, 4):
        bad = _impair(ct, "chroma_lowpass", float(k), rng)
        got = _decode(bad)
        ok = got == src
        # np.convolve(..., mode="same") reads x[i+s-k+1 .. i+s] with
        # s = (k - 1) // 2; a window is flat when the source value is the same
        # across all of it, and then the mean changes nothing
        sft = (k - 1) // 2
        flat = np.ones_like(src, dtype=bool)
        for off in range(sft - k + 1, sft + 1):
            shifted = np.roll(src, -off, axis=1)
            if off < 0:
                shifted[:, :(-off)] = -1
            elif off > 0:
                shifted[:, -off:] = -1
            flat &= shifted == src
        n_flat = int(flat.sum())
        n_diff = int((~flat).sum())
        out.append({
            "mechanism": "змішування",
            "case": f"ядро {k} відліки",
            "strength": float(k),
            "flat_window_pct": round(n_flat / src.size * 100, 2),
            "exact_pixels_pct": round(float(ok.mean() * 100), 2),
            "exact_within_flat_pct": round(
                float(ok[flat].mean() * 100) if n_flat else 0.0, 2),
            "exact_within_varying_pct": round(
                float(ok[~flat].mean() * 100) if n_diff else 0.0, 2),
            "psnr_db": round(_psnr(got.astype(np.uint8), img), 2),
        })

    # -- 3. magnitude: what one failure costs ------------------------------
    values = np.arange(256, dtype=np.float64)
    # expected |v - u| for u drawn uniformly over the 256 message values
    per_value = np.abs(values[:, None] - values[None, :]).mean(axis=1)
    hist = np.bincount(src.ravel(), minlength=256).astype(np.float64)
    random_ref = float((hist / hist.sum() * per_value).sum())
    for sigma in (1.0, 2.0, 4.0, 8.0, 16.0):
        bad = _impair(ct, "gauss", float(sigma), rng)
        got = _decode(bad)
        wrong = got != src
        n_wrong = int(wrong.sum())
        err = np.abs(got - src)
        out.append({
            "mechanism": "величина",
            "case": f"σ = {sigma:g}",
            "strength": float(sigma),
            "wrong_pixels_pct": round(n_wrong / src.size * 100, 2),
            "mean_abs_error_all": round(float(err.mean()), 2),
            "mean_abs_error_when_wrong": round(
                float(err[wrong].mean()) if n_wrong else 0.0, 2),
            "random_value_reference": round(random_ref, 2),
            # the same reference restricted to the pixels that actually failed:
            # which values fail is not uniform, so the whole-frame reference is
            # not the right yardstick at low sigma
            "random_value_reference_when_wrong": round(
                float(per_value[src[wrong]].mean()) if n_wrong else 0.0, 2),
            # how many distinct value pairs the failures actually consist of:
            # at a small sigma only geometrically close codewords are
            # reachable, so the average runs over few pairs and need not match
            # the full-range expectation
            "distinct_confusions": int(np.unique(
                (src[wrong] << 8) | got[wrong]).size) if n_wrong else 0,
        })
    return out


def noise_attacks(cfg: ExperimentConfig, key: bytes, sid: bytes,
                  img: np.ndarray, level: int = BEST_LEVEL
                  ) -> List[Dict[str, Any]]:
    """Does a perturbed ciphertext leak more than a clean one?

    Two questions at once.  Noise destroys the constant-luminance property, so
    the luminance plane is no longer empty - but the noise is independent of
    the picture, so whether it *leaks* is a separate matter and is measured
    here.  And the key-free reassembly attack is re-run on the perturbed
    planes, because a solver could in principle exploit the added structure.
    """
    from avsec.attacks import attack_boundary_reassembly
    from avsec.baselines import CryptoPermutationScrambler

    rows, cols = cfg.b2_grid
    perm = CryptoPermutationScrambler(rows, cols, key, sid)
    fitted = perm._join(perm._tiles(img))
    scrambled = perm.scramble(img, 0)
    mono = LumaBalancedMono(key, sid, level)
    ct = mono.encrypt(scrambled, 0)
    rng = np.random.default_rng(4242)
    src = fitted.astype(np.float64).ravel()

    out: List[Dict[str, Any]] = []
    for sigma in STRIP_SIGMAS:
        bad = _impair(ct, "gauss", float(sigma), rng)
        pl = chroma_planes(bad)
        y_plane = np.clip(np.rint(pl["Y"]), 0, 255).astype(np.uint8)
        for name, view in (("яскравість", y_plane), ("Cb", _to_grey(pl["Cb"]))):
            ba = attack_boundary_reassembly(view, rows, cols,
                                            perm.permutation(0), fitted)
            v = view.astype(np.float64).ravel()
            out.append({
                "sigma_levels": float(sigma),
                "plane": name,
                "plane_std": round(float(v.std()), 4),
                "correlation_with_source": round(
                    0.0 if v.std() == 0 else float(np.corrcoef(v, src)[0, 1]), 4),
                "neighbour_accuracy_pct":
                    round(ba.metrics["neighbour_accuracy"] * 100, 2),
                "direct_accuracy_pct":
                    round(ba.metrics["direct_accuracy"] * 100, 2),
            })
    return out


def noise_strip(key: bytes, sid: bytes, img: np.ndarray,
                level: int = BEST_LEVEL) -> Tuple[Dict[str, np.ndarray],
                                                  List[Dict[str, Any]]]:
    """Ciphertext and recovery at each noise level, for the picture strip."""
    mono = LumaBalancedMono(key, sid, level)
    ct = mono.encrypt(img, 0)
    rng = np.random.default_rng(20240909)
    images: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    for sigma in STRIP_SIGMAS:
        bad = _impair(ct, "gauss", float(sigma), rng)
        got = mono.decrypt(bad, 0, nearest=True)
        tag = f"{sigma:g}".replace(".", "p")
        images[f"noise_{tag}_cipher"] = bad
        images[f"noise_{tag}_recovered"] = got
        rows.append({"sigma_levels": float(sigma),
                     "psnr_db": round(_psnr(got, img), 2),
                     "exact_pixels_pct": round(float((got == img).mean() * 100), 2)})
    return images, rows


# ------------------------------------------ 8. the project's own analog path
#: Source depths swept over the tract: fewer bits buy codeword spacing.
ANALOG_BITS: Tuple[int, ...] = (8, 7, 6, 5, 4, 3)

#: The channel profiles every other method in this project is measured on.
ANALOG_CHANNELS: Tuple[str, ...] = ("clean", "mild", "moderate", "bursty",
                                    "harsh")


def tract_residual_error(cfg: ExperimentConfig, frames, scene: str
                         ) -> List[Dict[str, Any]]:
    """Amplitude error the tract leaves on one plain luma plane.

    Measured with the unprotected ``B0a``, so it is a property of the channel
    and the carrier, not of any cipher.  It is the quantity the codeword
    spacing has to beat.
    """
    from avsec.baselines import make_b0_analog
    from avsec.channel import ChannelTrace, preset

    H, W = cfg.frame_height, cfg.frame_width
    transport = cfg.profile("B4").transport_config(cfg.budget)
    out: List[Dict[str, Any]] = []
    for ch in ANALOG_CHANNELS:
        m = make_b0_analog(transport, preset(ch), H, W)
        m.reset()
        tr = ChannelTrace(seed=cfg.channel_seed_value, scene=scene,
                          repetition=0, profile=ch)
        errs = []
        for i, img in enumerate(frames):
            r = m.process(img, i, tr)
            errs.append(np.abs(r.reconstructed.astype(np.float64)
                               - img.astype(np.float64)))
        e = np.concatenate([x.ravel() for x in errs])
        out.append({
            "channel": ch,
            "rms_error_levels": round(float(np.sqrt((e ** 2).mean())), 2),
            "median_error_levels": round(float(np.median(e)), 2),
            "p95_error_levels": round(float(np.percentile(e, 95)), 2),
            "max_error_levels": round(float(e.max()), 2),
        })
    return out


def tract_example(cfg: ExperimentConfig, master, img: np.ndarray, scene: str,
                  level: int = BEST_LEVEL, source_bits: int = 8
                  ) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """The four stages of one frame, for every channel profile.

    Source, the ciphertext as it enters the tract, the same ciphertext as the
    receiver got it, and the frame the codebook produced from that.
    """
    from avsec.channel import ChannelTrace, preset
    from avsec.luma_channel import make_b2l

    H, W = cfg.frame_height, cfg.frame_width
    transport = cfg.profile("B4").transport_config(cfg.budget)
    images: Dict[str, np.ndarray] = {"example_source": img}
    rows: List[Dict[str, Any]] = []
    for ch in ANALOG_CHANNELS:
        m = make_b2l(transport, preset(ch), H, W, master,
                     session_id=b"B2LBNCH0", level=level,
                     source_bits=source_bits)
        m.reset()
        tr = ChannelTrace(seed=cfg.channel_seed_value, scene=scene,
                          repetition=0, profile=ch)
        res = m.process(img, 0, tr)
        images[f"example_{ch}_tx"] = np.asarray(m.last_tx_rgb, dtype=np.uint8)
        images[f"example_{ch}_rx"] = np.asarray(m.last_rx_rgb, dtype=np.uint8)
        images[f"example_{ch}_out"] = np.asarray(res.reconstructed, dtype=np.uint8)
        diff = np.abs(np.asarray(m.last_rx_rgb, dtype=np.float64)
                      - np.asarray(m.last_tx_rgb, dtype=np.float64)).mean(axis=2)
        images[f"example_{ch}_damage"] = _to_grey(diff)
        rows.append({
            "channel": ch,
            "psnr_db": round(_psnr(res.reconstructed, img), 2),
            "exact_pixels_pct": round(float((res.reconstructed == img).mean() * 100), 2),
            "mean_channel_damage_levels": round(float(diff.mean()), 2),
        })
    return images, rows


def tract_ablation(cfg: ExperimentConfig, master, img: np.ndarray,
                   scene: str, level: int = BEST_LEVEL) -> List[Dict[str, Any]]:
    """One impairment at a time, at the strength of the ``mild`` profile.

    ``B0a`` is carried along as the reference: it shows how much each
    impairment costs a scheme that does not depend on exact pixel values, so
    the difference isolates what the codebook is sensitive to.
    """
    from avsec.baselines import make_b0_analog
    from avsec.channel import ChannelTrace, preset
    from avsec.luma_channel import make_b2l

    H, W = cfg.frame_height, cfg.frame_width
    transport = cfg.profile("B4").transport_config(cfg.budget)
    mild = preset("mild")
    steps: List[Tuple[str, Optional[Dict[str, Any]]]] = [
        ("без спотворень", {}),
        ("підсилення й зміщення", {"gain": mild.gain, "offset": mild.offset}),
        ("шум", {"noise_sigma": mild.noise_sigma}),
        ("фільтр нижніх частот", {"lowpass_taps": mild.lowpass_taps}),
        ("зсув растру", {"shift_x": mild.shift_x}),
        ("джитер рядків", {"line_jitter_sigma": mild.line_jitter_sigma}),
        ("усе разом (профіль mild)", None),
    ]
    out: List[Dict[str, Any]] = []
    for label, over in steps:
        chan = preset("mild") if over is None else preset("clean")
        for k, v in (over or {}).items():
            setattr(chan, k, v)
        tr = ChannelTrace(seed=cfg.channel_seed_value, scene=scene,
                          repetition=0, profile="ablation")
        row: Dict[str, Any] = {"impairment_added": label}
        m = make_b0_analog(transport, chan, H, W)
        m.reset()
        row["B0a_psnr_db"] = round(_psnr(m.process(img, 0, tr).reconstructed, img), 2)
        for bits in (8, 5, 3):
            m = make_b2l(transport, chan, H, W, master, session_id=b"B2LBNCH0",
                         level=level, source_bits=bits)
            m.reset()
            shift = 8 - bits
            ref = (img if bits == 8 else
                   (((img.astype(np.int64) >> shift) << shift)
                    | ((img.astype(np.int64) >> shift)
                       >> max(0, 2 * bits - 8))).astype(np.uint8))
            row[f"B2l_{bits}bit_psnr_db"] = round(
                _psnr(m.process(img, 0, tr).reconstructed, ref), 2)
        out.append(row)
    return out


def analog_sweep(cfg: ExperimentConfig, master, frames, scene: str,
                 level: int = BEST_LEVEL, progress=None) -> List[Dict[str, Any]]:
    """``B2l`` over every profile and every source depth, plus the baselines.

    Every row uses the same trace as every other method would in that slot, so
    the numbers sit next to ``results/main`` rather than beside it.
    """
    from avsec.baselines import (make_b0_analog, make_b0a_repetition,
                                 make_b1_lfsr, make_b2_cryptoperm)
    from avsec.channel import ChannelTrace, preset
    from avsec.luma_channel import make_b2l

    H, W = cfg.frame_height, cfg.frame_width
    transport = cfg.profile("B4").transport_config(cfg.budget)
    rows, cols = cfg.b2_grid
    out: List[Dict[str, Any]] = []

    for ci, ch in enumerate(ANALOG_CHANNELS):
        if progress:
            progress(f"тракт: {ch}", 0.60 + 0.22 * ci / len(ANALOG_CHANNELS), {})
        chan = preset(ch)
        tr = ChannelTrace(seed=cfg.channel_seed_value, scene=scene,
                          repetition=0, profile=ch)
        builders: List[Tuple[str, int, Any]] = [
            ("B0a", 8, lambda: make_b0_analog(transport, chan, H, W)),
            ("B0a-R", 8, lambda: make_b0a_repetition(transport, chan, H, W,
                                                     cfg.budget.rasters_per_frame,
                                                     cfg.b0ar_combine)),
            ("B1", 8, lambda: make_b1_lfsr(transport, chan, H, W, cfg.lfsr)),
            ("B2", 8, lambda: make_b2_cryptoperm(transport, chan, H, W, rows,
                                                 cols, master,
                                                 session_id=b"B2-LUMA0")),
        ]
        for bits in ANALOG_BITS:
            builders.append((
                f"B2l-{bits}bit", bits,
                (lambda b=bits: make_b2l(transport, chan, H, W, master,
                                         session_id=b"B2LBNCH0", level=level,
                                         source_bits=b))))
        for name, bits, build in builders:
            method = build()
            method.reset()
            psnrs, exact = [], []
            for i, img in enumerate(frames):
                res = method.process(img, i, tr)
                ref = (img if bits == 8 else
                       ((img.astype(np.int64) >> (8 - bits)) << (8 - bits)
                        | (img.astype(np.int64) >> (8 - bits))
                        >> max(0, 2 * bits - 8)).astype(np.uint8))
                psnrs.append(_psnr(res.reconstructed, ref))
                exact.append(float((res.reconstructed == ref).mean() * 100))
            sp = codeword_spacing(1 << bits) if name.startswith("B2l") else None
            out.append({
                "channel": ch,
                "method": name,
                "source_bits": bits,
                "psnr_db": round(float(np.mean(psnrs)), 2),
                "exact_pixels_pct": round(float(np.mean(exact)), 2),
                "median_codeword_distance": sp["median_distance"] if sp else None,
                "tolerance_levels": round(sp["median_distance"] / 2, 2) if sp else None,
            })
    return out


# ------------------------------------------------------------------ runner
def run_luma_lab(cfg: ExperimentConfig, output_dir: str = "results/luma",
                 n_frames: int = 3, progress=None) -> Dict[str, Any]:
    """Every section above, written to ``output_dir``."""
    from avsec.crypto import derive_session_keys
    from avsec.sources.drone import DRONE_SCENES, drone_suite
    from avsec.subst_lab import colour_frame

    env = environment_record()
    out = ensure_dir(output_dir)
    img_dir = ensure_dir(os.path.join(out, "images"))
    H, W = cfg.frame_height, cfg.frame_width

    def say(stage: str, frac: float) -> None:
        if progress:
            try:
                progress(stage, frac, {})
            except Exception:
                pass

    sid = b"LUMALAB0"
    key = derive_session_keys(cfg.master_secret(), sid).key

    say("місткість палітри", 0.05)
    cap = palette_capacity(BEST_LEVEL)
    sweep = capacity_sweep()

    say("монохромний шлях", 0.2)
    picked = [d for d in DRONE_SCENES if d[0] == "village"]
    frames = drone_suite(H, W, n_frames=max(2, n_frames), scenes=picked)[0].frames
    mono_rows, mono_images = monochrome_path(key, sid, list(frames[:n_frames]))

    say("кольоровий шлях", 0.4)
    rgb = colour_frame(H, W)
    colour_rows, colour_images = colour_path(key, sid, rgb)

    say("спостерігач із яскравістю", 0.55)
    observer = luma_only_observer(key, sid, frames[0])

    say("атаки за площинами", 0.7)
    attacks = plane_attacks(cfg, key, sid, frames[0])

    say("стійкість до похибки", 0.78)
    noise = noise_tolerance(key, sid, frames[0])

    say("глибокий аналіз спотворень", 0.82)
    spacing = [codeword_spacing(n) for n in (256, 4096, 32768, 65536)]
    deep = noise_deep(key, sid, frames[0])

    say("геометрія відмови", 0.84)
    geometry = failure_geometry(key, sid, frames[0])

    say("атаки на спотворений шифротекст", 0.86)
    natt = noise_attacks(cfg, key, sid, frames[0])
    strip_images, strip_rows = noise_strip(key, sid, frames[0])

    say("залишкова похибка тракту", 0.58)
    residual = tract_residual_error(cfg, list(frames[:n_frames]), "village")

    say("прогін через аналоговий тракт", 0.60)
    tract = analog_sweep(cfg, cfg.master_secret(), list(frames[:n_frames]),
                         "village", progress=progress)

    say("наскрізний приклад за профілями", 0.87)
    ex_images, ex_rows = tract_example(cfg, cfg.master_secret(), frames[0],
                                       "village")

    say("розклад профілю за спотвореннями", 0.88)
    ablation = tract_ablation(cfg, cfg.master_secret(), frames[0], "village")

    say("зображення й рисунки", 0.92)
    images = dict(mono_images)
    images.update(colour_images)
    images.update(strip_images)
    images.update(ex_images)
    _save(img_dir, images)
    figures = _figures(out, images, sweep, mono_rows, colour_rows, attacks, noise)
    figures["noise_strip"] = _noise_strip_figure(out, images, strip_rows)
    figures["noise_deep"] = _noise_deep_figure(out, deep, natt, spacing)
    figures["analog_tract"] = _tract_figure(out, tract, residual)
    figures["tract_example"] = _example_figure(out, images, ex_rows)
    figures["failure_geometry"] = _geometry_figure(out, geometry, spacing)

    summary = {
        "kind": "luma_balance",
        "title": "Шифрування зі сталою яскравістю: монохромний і кольоровий шляхи",
        "frame": f"{W}x{H}",
        "capacity": cap,
        "capacity_sweep": sweep,
        "monochrome": mono_rows,
        "colour": colour_rows,
        "luma_only_observer": observer,
        "plane_attacks": attacks,
        "noise_tolerance": noise,
        "codeword_spacing": spacing,
        "noise_deep": deep,
        "failure_geometry": geometry,
        "noise_attacks": natt,
        "noise_strip": strip_rows,
        "tract_residual_error": residual,
        "analog_tract": tract,
        "tract_ablation": ablation,
        "tract_example": ex_rows,
        "figures": figures,
        "run_id": cfg.run_identity(),
        "commit": env.get("git_commit"),
        "git_worktree": env.get("git_worktree"),
        "environment": env.get("environment", env),
    }
    write_json(os.path.join(out, "luma.json"), summary)
    write_csv(os.path.join(out, "capacity_sweep.csv"), sweep)
    write_csv(os.path.join(out, "monochrome.csv"), mono_rows)
    write_csv(os.path.join(out, "colour.csv"), colour_rows)
    write_csv(os.path.join(out, "plane_attacks.csv"), attacks)
    write_csv(os.path.join(out, "noise_tolerance.csv"), noise)
    write_csv(os.path.join(out, "codeword_spacing.csv"), spacing)
    write_csv(os.path.join(out, "noise_deep.csv"), deep)
    write_csv(os.path.join(out, "noise_attacks.csv"), natt)
    write_csv(os.path.join(out, "noise_strip.csv"), strip_rows)
    write_csv(os.path.join(out, "tract_residual_error.csv"), residual)
    write_csv(os.path.join(out, "analog_tract.csv"), tract)
    write_csv(os.path.join(out, "tract_ablation.csv"), ablation)
    write_csv(os.path.join(out, "tract_example.csv"), ex_rows)
    write_csv(os.path.join(out, "failure_geometry.csv"), geometry)
    say("готово", 1.0)
    return summary


def _save(img_dir: str, images: Dict[str, np.ndarray]) -> None:
    from PIL import Image

    for name, arr in images.items():
        a = np.asarray(arr, dtype=np.uint8)
        Image.fromarray(a, mode="RGB" if a.ndim == 3 else "L").save(
            os.path.join(img_dir, f"{name}.png"))


def _figures(out_dir: str, images: Dict[str, np.ndarray],
             sweep: List[Dict[str, Any]], mono: List[Dict[str, Any]],
             colour: List[Dict[str, Any]], attacks: List[Dict[str, Any]],
             noise: List[Dict[str, Any]]) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: Dict[str, str] = {}

    # -- figure 1: the two paths, side by side --------------------------
    fig, axes = plt.subplots(2, 5, figsize=(15.5, 6.6))
    mono_order = [("mono_1_source", "1. Монохромний\nоригінал"),
                  ("mono_2_cipher_rgb", "2. Шифротекст RGB"),
                  ("mono_3_cipher_as_luma", "3. Його яскравість\n(що бачить\nмонохромний тракт)"),
                  ("mono_4_cipher_cb", "4. Площина Cb\n(тут дані)"),
                  ("mono_6_restored", "5. Відновлено\nточно до біта")]
    col_order = [("colour_1_source", "1. Кольоровий\nоригінал"),
                 ("colour_2_cipher_rgb", "2. Шифротекст RGB"),
                 ("colour_3_cipher_as_luma", "3. Його яскравість"),
                 ("colour_4_cipher_cb", "4. Площина Cb"),
                 ("colour_5_restored", "5. Відновлено\n16 з 24 бітів")]
    for row, order, tag in ((0, mono_order, "МОНОХРОМНИЙ ШЛЯХ"),
                            (1, col_order, "КОЛЬОРОВИЙ ШЛЯХ")):
        for k, (name, title) in enumerate(order):
            ax = axes[row, k]
            a = np.asarray(images[name], dtype=np.uint8)
            ax.imshow(a if a.ndim == 3 else a, cmap=None if a.ndim == 3 else "gray",
                      vmin=None if a.ndim == 3 else 0,
                      vmax=None if a.ndim == 3 else 255)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=8.5)
        axes[row, 0].set_ylabel(tag, fontsize=9.5, labelpad=8)
    fig.suptitle("Шифрування зі сталою яскравістю: два шляхи", fontsize=13)
    fig.text(0.012, 0.012,
             "Панель 3 обох рядків — це весь сигнал, який дістається "
             "монохромного тракту: рівне сіре поле з нульовою ентропією. "
             "Повідомлення живе в панелі 4.",
             fontsize=8, color="#37474f", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    p = os.path.join(out_dir, "paths.png")
    fig.savefig(p, dpi=140)
    fig.savefig(p.replace(".png", ".svg"))
    plt.close(fig)
    paths["paths"] = p

    # -- figure 2: capacity, attacks, noise ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    a0 = axes[0]
    lv = [r["luma_level"] for r in sweep]
    cnt = [r["colours"] for r in sweep]
    a0.plot(lv, cnt, marker="o", ms=3, color="#1565c0")
    a0.axhline(256, color="#2e7d32", ls="--", lw=1.2)
    a0.axhline(2 ** 24, color="#b71c1c", ls="--", lw=1.2)
    a0.set_yscale("log")
    a0.set_xlabel("рівень яскравості", fontsize=9)
    a0.set_ylabel("кольорів із цією яскравістю", fontsize=9)
    a0.set_title("Місткість: скільки кольорів\nмають задану яскравість", fontsize=10)
    a0.annotate("потрібно для 8 біт (моно)", (5, 300), fontsize=7.5, color="#2e7d32")
    a0.annotate("потрібно для 24 біт (колір)", (5, 2 ** 24 * 1.4), fontsize=7.5,
                color="#b71c1c")
    a0.grid(alpha=0.25)

    a1 = axes[1]
    labels = [r["view"].replace(": ", ":\n") for r in attacks]
    vals = [r["neighbour_accuracy_pct"] for r in attacks]
    colours = ["#ef5350", "#26a69a", "#ffa726", "#ffa726"]
    a1.bar(range(len(vals)), vals, color=colours)
    for i, v in enumerate(vals):
        a1.text(i, v + 1.2, f"{v:.2f}", ha="center", fontsize=8)
    a1.set_xticks(range(len(labels)))
    a1.set_xticklabels(labels, fontsize=7.5)
    a1.set_ylabel("правильних сусідств, %", fontsize=9)
    a1.set_title("Безключова атака за площинами\nодного шифротексту", fontsize=10)
    a1.grid(axis="y", alpha=0.25)

    a2 = axes[2]
    s = [r["sigma_levels"] for r in noise]
    ps = [r["psnr_db"] for r in noise]
    a2.plot(s, ps, marker="o", color="#6a1b9a")
    a2.axhline(20.0, color="#b71c1c", ls=":", lw=1.2)
    a2.annotate("робочий поріг 20 дБ", (0.05, 21), fontsize=7.5, color="#b71c1c")
    a2.set_xlabel("шум на шифротексті, рівнів (σ)", fontsize=9)
    a2.set_ylabel("PSNR відновлення, дБ", fontsize=9)
    a2.set_title("Стійкість до похибки амплітуди", fontsize=10)
    a2.grid(alpha=0.25)

    fig.suptitle("Місткість, витік за площинами і стійкість до похибки",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    p = os.path.join(out_dir, "capacity_attacks_noise.png")
    fig.savefig(p, dpi=140)
    fig.savefig(p.replace(".png", ".svg"))
    plt.close(fig)
    paths["capacity_attacks_noise"] = p
    return paths


def _noise_strip_figure(out_dir, images, rows):
    """Ciphertext and recovery at each noise level, one column per level."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.9))
    for k, r in enumerate(rows):
        tag = f"{r['sigma_levels']:g}".replace(".", "p")
        axes[0, k].imshow(np.asarray(images[f"noise_{tag}_cipher"], dtype=np.uint8))
        axes[0, k].set_title(f"σ = {r['sigma_levels']:g}", fontsize=10)
        axes[1, k].imshow(np.asarray(images[f"noise_{tag}_recovered"],
                                     dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
        axes[1, k].set_xlabel(f"{r['psnr_db']:.2f} дБ\n{r['exact_pixels_pct']:.2f} % точно",
                              fontsize=8.5)
        for row in (0, 1):
            axes[row, k].set_xticks([])
            axes[row, k].set_yticks([])
    axes[0, 0].set_ylabel("спотворений\nшифротекст", fontsize=9.5)
    axes[1, 0].set_ylabel("відновлено за\nнайближчим\nкодовим словом", fontsize=9.5)
    fig.suptitle("Зображення за різних рівнів шуму на шифротексті", fontsize=12.5)
    fig.text(0.012, 0.012,
             "Декодування за найближчим кодовим словом. Помилка в кодовому "
             "слові дає довільне значення, тому дефекти виглядають як окремі "
             "різкі точки, а не як розмиття.",
             fontsize=8, color="#37474f", wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    path = os.path.join(out_dir, "noise_strip.png")
    fig.savefig(path, dpi=140)
    fig.savefig(path.replace(".png", ".svg"))
    plt.close(fig)
    return path


def _noise_deep_figure(out_dir, deep, natt, spacing):
    """Impairment curves, the error profile, codeword spacing and attacks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.0))

    ax = axes[0, 0]
    for kind, colour, label in (
            ("gauss", "#1565c0", "гауссів шум, σ рівнів"),
            ("uniform", "#00897b", "рівномірний, ± рівнів"),
            ("gain_offset", "#8d6e63", "дрейф підсилення й зміщення")):
        rs = [r for r in deep if r["impairment"] == kind]
        ax.plot([r["strength"] for r in rs], [r["nearest_psnr_db"] for r in rs],
                marker="o", ms=4, color=colour, label=label)
    ax.axhline(20.0, color="#b71c1c", ls=":", lw=1.2)
    ax.annotate("робочий поріг 20 дБ", (0.2, 21), fontsize=8, color="#b71c1c")
    ax.set_xlabel("сила спотворення, рівнів", fontsize=9)
    ax.set_ylabel("PSNR відновлення, дБ", fontsize=9)
    ax.set_title("Амплітудні спотворення", fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    for kind, colour, label, xl in (
            ("chroma_lowpass", "#e65100", "смуга кольоровості, відліків", "ядро"),
            ("burst", "#6a1b9a", "пакетне пошкодження, рядків", "рядків")):
        rs = [r for r in deep if r["impairment"] == kind]
        ax.plot([r["strength"] for r in rs], [r["nearest_psnr_db"] for r in rs],
                marker="s", ms=4, color=colour, label=label)
    ax.axhline(20.0, color="#b71c1c", ls=":", lw=1.2)
    ax.set_xlabel("ширина ядра або висота пакета", fontsize=9)
    ax.set_ylabel("PSNR відновлення, дБ", fontsize=9)
    ax.set_title("Спотворення, властиві аналоговому тракту", fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    rs = [r for r in deep if r["impairment"] == "gauss"]
    x = [r["strength"] for r in rs]
    ax.plot(x, [r["exact_pixels_pct"] for r in rs], marker="o", ms=4,
            color="#ef5350", label="точний пошук у таблиці")
    ax.plot(x, [r["nearest_pixels_pct"] for r in rs], marker="o", ms=4,
            color="#2e7d32", label="за найближчим кодовим словом")
    ax2 = ax.twinx()
    ax2.plot(x, [r["nearest_mean_abs_error"] for r in rs], marker="^", ms=4,
             color="#5c6bc0", ls="--", label="середня похибка")
    ax2.set_ylabel("середня похибка значення, рівнів", fontsize=9,
                   color="#5c6bc0")
    ax.set_xlabel("гауссів шум, σ рівнів", fontsize=9)
    ax.set_ylabel("пікселів відновлено точно, %", fontsize=9)
    ax.set_title("Два декодери й профіль похибки", fontsize=10.5)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    sig = sorted({r["sigma_levels"] for r in natt})
    for plane, colour in (("яскравість", "#26a69a"), ("Cb", "#ffa726")):
        rs = [r for r in natt if r["plane"] == plane]
        rs.sort(key=lambda r: r["sigma_levels"])
        ax.plot([r["sigma_levels"] for r in rs],
                [r["neighbour_accuracy_pct"] for r in rs],
                marker="o", ms=4, color=colour, label=f"площина {plane}")
    ax.axhline(0.52, color="#607d8b", ls=":", lw=1.2)
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, 0.0), max(hi, 1.2))
    ax.annotate("рівень випадкового розкладання 0,52 %", (0.2, 0.56),
                fontsize=7.5, color="#607d8b")
    ax.set_xlabel("гауссів шум на шифротексті, σ рівнів", fontsize=9)
    ax.set_ylabel("правильних сусідств, %", fontsize=9)
    ax.set_title("Безключова атака на спотворений шифротекст", fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    txt = "  ".join(f"{d['codewords']} слів: мін. відстань {d['min_distance']:g}"
                    for d in spacing)
    fig.suptitle("Глибокий аналіз стійкості до спотворень", fontsize=13)
    fig.text(0.012, 0.012, "Відстань між кодовими словами у просторі RGB — "
                           + txt + ". Половина мінімальної відстані є межею, "
                           "до якої декодування за найближчим ще безпомилкове.",
             fontsize=8, color="#37474f", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path = os.path.join(out_dir, "noise_deep.png")
    fig.savefig(path, dpi=140)
    fig.savefig(path.replace(".png", ".svg"))
    plt.close(fig)
    return path




def _example_figure(out_dir, images, rows):
    """Source, what enters the tract, the damage, and what came out."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(10.4, 2.35 * n))
    titles = ("1. Оригінал", "2. Що йде в тракт",
              "3. Що прийнято з тракту", "4. Що отримали")
    for r, row in enumerate(rows):
        ch = row["channel"]
        panels = [images["example_source"], images[f"example_{ch}_tx"],
                  images[f"example_{ch}_rx"], images[f"example_{ch}_out"]]
        for c, arr in enumerate(panels):
            ax = axes[r, c]
            a = np.asarray(arr, dtype=np.uint8)
            if a.ndim == 3:
                ax.imshow(a)
            else:
                ax.imshow(a, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
        axes[r, 0].set_ylabel(ch, fontsize=10)
        note = ("точно" if row["psnr_db"] >= 98 else f"{row['psnr_db']:.2f} дБ")
        axes[r, 3].set_xlabel(f"{note}, {row['exact_pixels_pct']:.1f} % точно",
                              fontsize=8.5)
        axes[r, 2].set_xlabel(
            f"середнє пошкодження {row['mean_channel_damage_levels']:.1f} рівня",
            fontsize=8.5)
    fig.suptitle("Один кадр крізь тракт: що надіслано, що прийнято, що вийшло",
                 fontsize=12.5)
    fig.text(0.012, 0.008,
             "Стовпець 2 — шифротекст зі сталою яскравістю, що передається "
             "трьома площинами у трьох слотах растру. Стовпець 3 — ті самі "
             "площини після тракту. Стовпець 4 — результат декодування за "
             "найближчим кодовим словом.",
             fontsize=8, color="#37474f", wrap=True)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    path = os.path.join(out_dir, "tract_example.png")
    fig.savefig(path, dpi=140)
    fig.savefig(path.replace(".png", ".svg"))
    plt.close(fig)
    return path


def _geometry_figure(out_dir, geometry, spacing, level=BEST_LEVEL):
    """The plane, the two error directions, and what mixing does."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pal = constant_luma_palette(level).astype(np.float64)
    picked = pal[np.linspace(0, pal.shape[0] - 1, 256).astype(np.int64)]
    normal = np.array([BT601_R, BT601_G, BT601_B], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    u = np.array([1.0, 0.0, 0.0]) - normal * normal[0]
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    centre = pal.mean(axis=0)

    def flat(pts):
        d = pts - centre
        return d @ u, d @ v

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    ax = axes[0]
    # the palette holds 111 749 points; a subsample draws the same region and
    # keeps the vector version of the figure a reasonable size
    shown = pal[::max(1, pal.shape[0] // 6000)]
    px, py = flat(shown)
    ax.scatter(px, py, s=1.6, c=np.clip(shown / 255.0, 0, 1), alpha=0.55)
    kx, ky = flat(picked)
    ax.scatter(kx, ky, s=9, facecolors="none", edgecolors="#263238",
               linewidths=0.6, label="256 кодових слів")
    ax.set_title("Палітра лежить у площині сталої яскравості", fontsize=11)
    ax.set_xlabel("перша вісь у площині, рівнів")
    ax.set_ylabel("друга вісь у площині, рівнів")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)

    ax = axes[1]
    rows = [r for r in geometry if r["mechanism"] == "напрям"]
    styles = {"ізотропний": ("#455a64", "o", "-"),
              "у площині палітри": ("#c62828", "s", "-"),
              "поперек площини палітри": ("#2e7d32", "^", "--")}
    for label, (c, mk, ls) in styles.items():
        sel = [r for r in rows if r["case"] == label]
        ax.plot([r["strength"] for r in sel],
                [r["exact_pixels_pct"] for r in sel],
                color=c, marker=mk, linestyle=ls, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("RMS похибки, рівнів")
    ax.set_ylabel("пікселів відновлено точно, %")
    ax.set_title("Та сама похибка, спрямована по-різному", fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    ax = axes[2]
    mix = [r for r in geometry if r["mechanism"] == "змішування"]
    x = np.arange(len(mix))
    w = 0.27
    ax.bar(x - w, [r["flat_window_pct"] for r in mix], w,
           color="#90a4ae", label="вікно з одного значення")
    ax.bar(x, [r["exact_within_flat_pct"] for r in mix], w,
           color="#2e7d32", label="відновлено в таких вікнах")
    ax.bar(x + w, [r["exact_within_varying_pct"] for r in mix], w,
           color="#c62828", label="відновлено в решті")
    ax.set_xticks(x)
    ax.set_xticklabels([r["case"] for r in mix], fontsize=9)
    ax.set_ylabel("%")
    ax.set_title("Згладжування: виживає лише однорідне вікно", fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Чому спотворення діють саме так: геометрія кодової книги",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out_dir, "failure_geometry.png")
    fig.savefig(path, dpi=140)
    fig.savefig(path.replace(".png", ".svg"))
    plt.close(fig)
    return path


def _tract_figure(out_dir, tract, residual):
    """Residual error against codeword spacing, and the resulting quality."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))

    ax = axes[0]
    chans = [r["channel"] for r in residual]
    rms = [r["rms_error_levels"] for r in residual]
    ax.bar(range(len(chans)), rms, color="#ef5350", label="залишкова похибка тракту")
    for bits, colour in ((8, "#1565c0"), (6, "#00897b"), (4, "#f9a825")):
        tol = [r["tolerance_levels"] for r in tract
               if r["method"] == f"B2l-{bits}bit"][0]
        ax.axhline(tol, color=colour, ls="--", lw=1.4,
                   label=f"межа для {bits} біт: {tol:g}")
    ax.set_xticks(range(len(chans)))
    ax.set_xticklabels(chans, fontsize=8.5)
    ax.set_ylabel("рівнів яскравості", fontsize=9)
    ax.set_title("Прогноз: похибка тракту\nпроти межі декодування", fontsize=10.5)
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    for bits, colour in ((8, "#1565c0"), (7, "#5c6bc0"), (6, "#00897b"),
                         (5, "#7cb342"), (4, "#f9a825"), (3, "#e65100")):
        ys = [r["psnr_db"] for ch in ANALOG_CHANNELS
              for r in tract if r["channel"] == ch
              and r["method"] == f"B2l-{bits}bit"]
        ys = [min(y, 60.0) for y in ys]
        ax.plot(range(len(ANALOG_CHANNELS)), ys, marker="o", ms=4,
                color=colour, label=f"{bits} біт")
    ax.axhline(20.0, color="#b71c1c", ls=":", lw=1.2)
    ax.annotate("робочий поріг 20 дБ", (0.05, 21), fontsize=7.5, color="#b71c1c")
    ax.set_xticks(range(len(ANALOG_CHANNELS)))
    ax.set_xticklabels(ANALOG_CHANNELS, fontsize=8.5)
    ax.set_ylabel("PSNR, дБ (обрізано на 60)", fontsize=9)
    ax.set_title("Вимірювання: B2l за глибиною джерела", fontsize=10.5)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.25)

    ax = axes[2]
    base = ("B0a", "B0a-R", "B1", "B2", "B2l-8bit", "B2l-4bit")
    width = 0.14
    x = np.arange(len(ANALOG_CHANNELS))
    for k, name in enumerate(base):
        ys = [min([r["psnr_db"] for r in tract if r["channel"] == ch
                   and r["method"] == name][0], 60.0)
              for ch in ANALOG_CHANNELS]
        ax.bar(x + (k - 2.5) * width, ys, width, label=name)
    ax.axhline(20.0, color="#b71c1c", ls=":", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(ANALOG_CHANNELS, fontsize=8.5)
    ax.set_ylabel("PSNR, дБ (обрізано на 60)", fontsize=9)
    ax.set_title("Поруч зі схемами того самого бюджету", fontsize=10.5)
    ax.legend(fontsize=7.5, ncol=3)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Схема зі сталою яскравістю через аналоговий тракт проєкту",
                 fontsize=12.5)
    fig.text(0.012, 0.012,
             "Ті самі пʼять профілів каналу, ті самі траси й ті самі сцени, "
             "що й для решти схем. B2l займає три слоти растру — стільки ж, "
             "скільки B0a-R і цифрові схеми.",
             fontsize=8, color="#37474f", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    path = os.path.join(out_dir, "analog_tract.png")
    fig.savefig(path, dpi=140)
    fig.savefig(path.replace(".png", ".svg"))
    plt.close(fig)
    return path


__all__ = ["NOISE_SIGMAS", "FINE_SIGMAS", "IMPAIRMENTS", "STRIP_SIGMAS",
           "ANALOG_BITS", "ANALOG_CHANNELS", "tract_residual_error",
           "analog_sweep", "tract_ablation", "tract_example",
           "failure_geometry",
           "capacity_sweep", "monochrome_path", "colour_path",
           "luma_only_observer", "plane_attacks", "noise_tolerance",
           "codeword_spacing", "noise_deep", "noise_attacks", "noise_strip",
           "run_luma_lab"]
