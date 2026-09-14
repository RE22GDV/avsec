"""B2s: the keyed substitution and what it does and does not change.

The tests are written around the claims the documentation makes, so a change
that quietly weakens one of them fails here rather than in a reader's hands.
"""
from __future__ import annotations

import numpy as np
import pytest

from avsec import sources as S
from avsec.attacks import (
    attack_boundary_reassembly,
    attack_known_pair,
    attack_substitution_known_pair,
)
from avsec.baselines import CryptoPermutationScrambler
from avsec.substitution import (
    SUBSTITUTION_MODES,
    SubstitutionPermutationScrambler,
    crypto_sbox,
    invert_sbox,
)

KEY = bytes(range(32))
SID = b"TESTSID0"
ROWS, COLS = 12, 16


def _frame(name="texture", h=192, w=256):
    return S.PATTERNS[name](h, w)


# --------------------------------------------------------------- the table
def test_sbox_is_a_bijection_and_its_inverse_is_exact():
    for ctx in (b"a", b"b", b"|SID|session"):
        box = crypto_sbox(KEY, ctx)
        assert sorted(box.tolist()) == list(range(256))
        inv = invert_sbox(box)
        assert np.array_equal(inv[box.astype(np.int64)], np.arange(256))


def test_a_different_key_gives_a_different_table():
    a = crypto_sbox(KEY, b"ctx")
    b = crypto_sbox(bytes(range(1, 33)), b"ctx")
    assert not np.array_equal(a, b)


def test_the_table_is_deterministic_for_the_same_key_and_context():
    assert np.array_equal(crypto_sbox(KEY, b"ctx"), crypto_sbox(KEY, b"ctx"))


# ------------------------------------------------------------- correctness
@pytest.mark.parametrize("mode", SUBSTITUTION_MODES)
def test_round_trip_is_bit_exact(mode):
    sc = SubstitutionPermutationScrambler(ROWS, COLS, KEY, SID, mode=mode)
    img = _frame()
    for fid in (0, 1, 7):
        back = sc.descramble(sc.scramble(img, fid), fid)
        assert np.array_equal(back, sc._fit(img))


@pytest.mark.parametrize("mode", SUBSTITUTION_MODES)
def test_round_trip_survives_a_size_that_is_not_a_multiple_of_the_grid(mode):
    sc = SubstitutionPermutationScrambler(ROWS, COLS, KEY, SID, mode=mode)
    img = _frame(h=190, w=250)
    back = sc.descramble(sc.scramble(img, 0), 0)
    assert np.array_equal(back, sc._fit(img))


def test_a_wrong_key_does_not_recover_the_frame():
    img = _frame()
    good = SubstitutionPermutationScrambler(ROWS, COLS, KEY, SID, mode="block")
    bad = SubstitutionPermutationScrambler(ROWS, COLS, bytes(range(1, 33)), SID,
                                           mode="block")
    out = bad.descramble(good.scramble(img, 0), 0)
    mse = float(((out.astype(float) - good._fit(img).astype(float)) ** 2).mean())
    assert mse > 1000.0, "a wrong key must not come close to the original"


def test_block_mode_encrypts_identical_blocks_differently():
    """The point of a per-block table: equal plaintext blocks must differ."""
    img = np.full((192, 256), 77, dtype=np.uint8)
    sc = SubstitutionPermutationScrambler(ROWS, COLS, KEY, SID, mode="block")
    tiles = sc._tiles(sc.substitute(img, 0))
    values = {int(t.flat[0]) for t in tiles}
    assert len(values) > 1, "a flat frame must not encrypt to a flat frame"

    flat = SubstitutionPermutationScrambler(ROWS, COLS, KEY, SID, mode="session")
    tiles = flat._tiles(flat.substitute(img, 0))
    assert len({int(t.flat[0]) for t in tiles}) == 1, "one table maps 77 to one value"


def test_the_number_of_tables_matches_the_mode():
    for mode, expected in (("session", 1), ("frame", 1), ("block", ROWS * COLS)):
        sc = SubstitutionPermutationScrambler(ROWS, COLS, KEY, SID, mode=mode)
        assert sc.subst.n_tables() == expected


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        SubstitutionPermutationScrambler(ROWS, COLS, KEY, SID, mode="whatever")


# --------------------------------------------------------- what is hidden
def test_permutation_preserves_the_histogram_and_substitution_does_not():
    img = _frame()
    perm = CryptoPermutationScrambler(ROWS, COLS, KEY, SID)
    sc = SubstitutionPermutationScrambler(ROWS, COLS, KEY, SID, mode="session")

    def hist(x):
        return np.bincount(x.ravel().astype(np.int64), minlength=256)

    h0 = hist(sc._fit(img))
    assert np.array_equal(h0, hist(perm.scramble(img, 0)))
    assert not np.array_equal(h0, hist(sc.substitute(img, 0)))


def test_one_global_table_leaves_the_sorted_histogram_invariant():
    """This invariant is what the known-pair attack matches tiles on."""
    img = _frame()
    sc = SubstitutionPermutationScrambler(ROWS, COLS, KEY, SID, mode="session")

    def hist(x):
        return np.sort(np.bincount(x.ravel().astype(np.int64), minlength=256))

    assert np.array_equal(hist(sc._fit(img)), hist(sc.substitute(img, 0)))


def test_per_block_tables_destroy_that_invariant_too():
    img = _frame()
    sc = SubstitutionPermutationScrambler(ROWS, COLS, KEY, SID, mode="block")

    def hist(x):
        return np.sort(np.bincount(x.ravel().astype(np.int64), minlength=256))

    assert not np.array_equal(hist(sc._fit(img)), hist(sc.substitute(img, 0)))


# ------------------------------------------------------------------ attacks
@pytest.mark.parametrize("mode", SUBSTITUTION_MODES)
def test_plain_tile_matching_no_longer_works(mode):
    """The attack that breaks B2 in one frame must fail against B2s."""
    img = _frame()
    sc = SubstitutionPermutationScrambler(ROWS, COLS, KEY, SID, mode=mode)
    ciph = sc.scramble(img, 0)
    got = attack_known_pair(sc._fit(img), ciph, ROWS, COLS)
    acc = float((got.recovered_permutation == sc.permutation(0)).mean())
    assert acc < 0.2


@pytest.mark.parametrize("mode", SUBSTITUTION_MODES)
def test_the_substitution_aware_attack_still_recovers_the_permutation(mode):
    """Honesty test: substitution does not hide the block permutation.

    Matching on the sorted per-tile histogram is invariant under any per-block
    bijection on values, so the permutation falls to one known pair whatever
    the substitution does.  The documentation says so; this keeps it true.
    """
    img = _frame()
    sc = SubstitutionPermutationScrambler(ROWS, COLS, KEY, SID, mode=mode)
    ciph = sc.scramble(img, 0)
    got = attack_substitution_known_pair(sc._fit(img), ciph, ROWS, COLS,
                                         sc.permutation(0))
    assert got.metrics["permutation_accuracy"] > 0.9


def test_the_attack_detects_whether_one_table_or_many_are_in_use():
    img = _frame()
    for mode, expected in (("session", 1.0), ("frame", 1.0), ("block", 0.0)):
        sc = SubstitutionPermutationScrambler(ROWS, COLS, KEY, SID, mode=mode)
        got = attack_substitution_known_pair(sc._fit(img), sc.scramble(img, 0),
                                             ROWS, COLS, sc.permutation(0))
        assert got.metrics["substitution_is_global"] == expected


def test_substitution_collapses_the_key_free_reassembly_attack():
    """The gain that justifies the primitive, kept as a regression."""
    img = S.PATTERNS["smooth"](192, 256)
    perm = CryptoPermutationScrambler(ROWS, COLS, KEY, SID)
    without = attack_boundary_reassembly(perm.scramble(img, 0), ROWS, COLS,
                                         perm.permutation(0), perm._fit(img)
                                         if hasattr(perm, "_fit") else img)
    sc = SubstitutionPermutationScrambler(ROWS, COLS, KEY, SID, mode="block")
    with_sub = attack_boundary_reassembly(sc.scramble(img, 0), ROWS, COLS,
                                          sc.permutation(0), sc._fit(img))
    assert with_sub.metrics["neighbour_accuracy"] < \
        without.metrics["neighbour_accuracy"] / 2


# ------------------------------------------------------------ channel cost
def test_a_one_level_error_becomes_an_arbitrary_value():
    """The mechanism behind the channel result, as a number.

    A permutation carries values unchanged, so an error of one level stays one
    level.  A substitution table has no order structure, so the same error maps
    to an essentially uniform draw: mean absolute error near 255/3 = 85.
    """
    from avsec.subst_lab import error_amplification

    box = crypto_sbox(KEY, b"|SID|session")
    rows = {r["wire_error_levels"]: r for r in error_amplification(box, (1,))}
    assert rows[1]["permutation_error_levels"] == 1
    assert rows[1]["substitution_mean_abs_error"] > 50.0


# ------------------------------------------------------------------ colour
def _rgb(h=192, w=256):
    from avsec.subst_lab import colour_frame

    return colour_frame(h, w)


@pytest.mark.parametrize("source", ("random", "algebraic"))
@pytest.mark.parametrize("mode", SUBSTITUTION_MODES)
def test_colour_round_trip_is_bit_exact(mode, source):
    from avsec.substitution import ColourSubstitutionPermutation

    img = _rgb()
    sc = ColourSubstitutionPermutation(ROWS, COLS, KEY, SID, mode=mode,
                                       source=source)
    assert np.array_equal(sc.descramble(sc.scramble(img, 0), 0), sc._fit(img))


def test_each_channel_gets_its_own_table():
    from avsec.substitution import ColourSubstitutionPermutation

    sc = ColourSubstitutionPermutation(ROWS, COLS, KEY, SID, mode="session")
    tabs = sc.tables(0)
    assert set(tabs) == {"R", "G", "B"}
    assert not np.array_equal(tabs["R"][0], tabs["G"][0])
    assert not np.array_equal(tabs["R"][0], tabs["B"][0])
    assert not np.array_equal(tabs["G"][0], tabs["B"][0])


def test_a_table_covers_every_byte_value():
    """16 x 16 = 256: the whole alphabet of one byte, and nothing else."""
    from avsec.substitution import ColourSubstitutionPermutation

    sc = ColourSubstitutionPermutation(ROWS, COLS, KEY, SID, mode="session")
    for ch, t in sc.tables(0).items():
        assert t.shape[1] == 256, ch
        assert sorted(t[0].tolist()) == list(range(256)), ch


def test_one_shared_table_preserves_channel_equality_and_separate_ones_do_not():
    """The measurement that justifies a table per channel.

    Correlation does not separate the two designs - any substitution destroys
    it.  Equality does: one table maps equal values to equal values exactly.
    """
    from avsec.subst_lab import _shared_table_control
    from avsec.substitution import (ColourSubstitutionPermutation,
                                    channel_equality)

    img = _rgb()
    sc = ColourSubstitutionPermutation(ROWS, COLS, KEY, SID, mode="block")
    fitted = sc._fit(img)
    before = channel_equality(fitted)

    shared = _shared_table_control(ROWS, COLS, KEY, SID, fitted, "block", "random")
    assert channel_equality(shared) == before, "one table preserves equality exactly"

    per_channel = channel_equality(sc.substitute(img, 0))
    for pair, value in per_channel.items():
        assert value < before[pair] / 2, pair


def test_an_unsupported_channel_count_is_refused():
    from avsec.substitution import ColourSubstitutionPermutation

    with pytest.raises(ValueError):
        ColourSubstitutionPermutation(ROWS, COLS, KEY, SID, n_channels=2)
