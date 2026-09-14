"""Luminance-balanced encryption: the capacity argument and both paths.

The tests are written around the claims of ``docs/luma_balance.md``, including
the inconvenient ones: that the colour path cannot be lossless, and that the
concealment is of luminance only.
"""
from __future__ import annotations

import numpy as np
import pytest

from avsec import sources as S
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

KEY = bytes(range(32))
SID = b"TESTSID0"


def _frame(name="texture", h=96, w=128):
    return S.PATTERNS[name](h, w)


def _colour(h=96, w=128):
    rng = np.random.default_rng(7)
    base = _frame("texture", h, w).astype(np.int64)
    return np.clip(np.stack([base, np.roll(base, 3, 1), np.roll(base, 7, 0)],
                            axis=2) + rng.integers(-20, 20, (h, w, 3)),
                   0, 255).astype(np.uint8)


# --------------------------------------------------------------- capacity
def test_the_palette_holds_only_colours_of_that_luminance():
    pal = constant_luma_palette(BEST_LEVEL)
    assert pal.shape[1] == 3
    assert np.all(np.rint(luma(pal)).astype(np.int64) == BEST_LEVEL)


def test_the_palette_has_no_duplicates():
    pal = constant_luma_palette(BEST_LEVEL).astype(np.int64)
    packed = (pal[:, 0] << 16) | (pal[:, 1] << 8) | pal[:, 2]
    assert np.unique(packed).size == packed.size


def test_capacity_is_enough_for_eight_bits_and_not_for_twenty_four():
    """The counting argument the whole design rests on."""
    cap = palette_capacity(BEST_LEVEL)
    assert cap["enough_for_monochrome_8_bit"] is True
    assert cap["enough_for_colour_24_bit"] is False
    assert cap["usable_bits"] == 16
    assert cap["colour_bits_lost"] == 8


def test_the_best_level_really_is_the_best():
    n = palette_capacity(BEST_LEVEL)["colours_with_this_luma"]
    for other in (32, 64, 96, 160, 192, 224):
        assert palette_capacity(other)["colours_with_this_luma"] <= n


# ------------------------------------------------------------- monochrome
@pytest.mark.parametrize("name", ("smooth", "edges", "texture"))
def test_monochrome_round_trip_is_bit_exact(name):
    mono = LumaBalancedMono(KEY, SID)
    img = _frame(name)
    assert np.array_equal(mono.decrypt(mono.encrypt(img, 0), 0), img)


def test_monochrome_ciphertext_has_one_luminance_level():
    """The property the whole scheme is named after."""
    mono = LumaBalancedMono(KEY, SID)
    st = luma_statistics(mono.encrypt(_frame(), 0))
    assert st["rounded_levels_used"] == 1
    assert st["luma_entropy_bits"] == 0.0


def test_a_flat_source_and_a_detailed_source_give_the_same_luminance():
    """A luminance-only observer cannot even tell the two frames apart."""
    mono = LumaBalancedMono(KEY, SID)
    flat = np.full((96, 128), 17, dtype=np.uint8)
    a = np.rint(luma(mono.encrypt(flat, 0))).astype(np.uint8)
    b = np.rint(luma(mono.encrypt(_frame(), 0))).astype(np.uint8)
    assert np.array_equal(a, b)


def test_a_wrong_key_does_not_recover_the_frame():
    img = _frame()
    good = LumaBalancedMono(KEY, SID)
    bad = LumaBalancedMono(bytes(range(1, 33)), SID)
    out = bad.decrypt(good.encrypt(img, 0), 0)
    assert float((out == img).mean()) < 0.05


def test_the_codebook_changes_between_frames():
    mono = LumaBalancedMono(KEY, SID, per_frame=True)
    img = np.full((8, 8), 200, dtype=np.uint8)
    assert not np.array_equal(mono.encrypt(img, 0), mono.encrypt(img, 1))


# ----------------------------------------------------------------- colour
def test_colour_path_is_exact_on_the_reduced_representation():
    """The cipher adds no loss of its own; all of it is the capacity deficit."""
    col = LumaBalancedColour(KEY, SID)
    rgb = _colour()
    assert np.array_equal(col.decrypt(col.encrypt(rgb, 0), 0), col.quantise(rgb))


def test_colour_path_is_not_lossless_and_says_so():
    col = LumaBalancedColour(KEY, SID)
    rgb = _colour()
    assert col.describe()["lossless"] is False
    assert col.describe()["bits_lost"] == 8
    assert not np.array_equal(col.decrypt(col.encrypt(rgb, 0), 0), rgb)


def test_colour_ciphertext_also_has_one_luminance_level():
    col = LumaBalancedColour(KEY, SID)
    st = luma_statistics(col.encrypt(_colour(), 0))
    assert st["rounded_levels_used"] == 1


def test_asking_for_more_bits_than_the_palette_holds_is_refused():
    with pytest.raises(ValueError):
        LumaBalancedColour(KEY, SID, BEST_LEVEL, bits=(7, 7, 7))


# ------------------------------------------------------------- what leaks
def test_the_luminance_plane_carries_no_information():
    mono = LumaBalancedMono(KEY, SID)
    y = luma(mono.encrypt(_frame(), 0))
    assert float(np.rint(y).std()) == 0.0


def test_the_chroma_planes_do_carry_the_message():
    """Concealment is of luminance only; the data is in the chroma planes."""
    mono = LumaBalancedMono(KEY, SID)
    pl = chroma_planes(mono.encrypt(_frame(), 0))
    assert float(pl["Cb"].std()) > 10.0
    assert float(pl["Cr"].std()) > 10.0


def test_a_structure_preserving_codebook_leaks_more_than_a_keyed_one():
    """Separates the constraint from the keying, which is the control.

    An ordered codebook maps neighbouring values to neighbouring colours, so
    the chroma planes keep some of the original structure.  The keyed codebook
    does not.  Both keep the luminance plane empty.
    """
    img = _frame("smooth")
    src = img.ravel().astype(np.float64)

    def chroma_corr(keyed):
        cb = chroma_planes(
            LumaBalancedMono(KEY, SID, keyed=keyed).encrypt(img, 0))["Cb"]
        return abs(float(np.corrcoef(cb.ravel(), src)[0, 1]))

    assert chroma_corr(False) > chroma_corr(True)


# ------------------------------------------------------ noise, as measured
def test_a_perturbed_ciphertext_still_decodes_at_small_sigma():
    """Unlike a byte substitution table, the codebook lives in three dimensions.

    Nearest-codeword decoding therefore tolerates a small amplitude error; the
    bench measures where that stops.
    """
    mono = LumaBalancedMono(KEY, SID)
    img = _frame()
    ct = mono.encrypt(img, 0).astype(np.float64)
    rng = np.random.default_rng(1)
    noisy = np.clip(ct + rng.normal(0, 1.0, ct.shape), 0, 255).astype(np.uint8)
    got = mono.decrypt(noisy, 0, nearest=True)
    assert float((got == img).mean()) > 0.90
