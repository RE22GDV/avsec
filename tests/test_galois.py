"""GF(2^8) arithmetic, the computed S-box, and the properties it is built for.

The decisive test is the first one: an implementation of the field is right if
and only if it reproduces a table somebody else published.
"""
from __future__ import annotations

import numpy as np
import pytest

from avsec.galois import (
    AES_POLY,
    AES_SBOX_REFERENCE,
    GF256,
    KeyedAlgebraicSbox,
    affine,
    algebraic_sbox,
    inverse_algebraic_sbox,
    irreducible_polynomials,
    is_irreducible,
    rotl8,
)
from avsec.sbox_analysis import (
    algebraic_degree,
    analyse,
    avalanche,
    differential_uniformity,
    is_bijective,
    nonlinearity,
)

KEY = bytes(range(32))


# --------------------------------------------------------------- the field
def test_the_aes_polynomial_is_irreducible():
    assert is_irreducible(AES_POLY)


def test_a_reducible_polynomial_is_refused():
    # x^8 + x^4 = x^4 (x^4 + 1): divisible by x, so not a field
    with pytest.raises(ValueError):
        GF256(0x110)


def test_there_are_thirty_irreducible_polynomials_of_degree_eight():
    """A standard count, and a cheap check that the sieve is right."""
    assert len(irreducible_polynomials(8)) == 30


def test_the_generator_is_primitive():
    gf = GF256()
    assert gf.is_primitive(gf.generator)
    assert gf.order(gf.generator) == 255


def test_multiplication_has_the_field_properties():
    gf = GF256()
    assert gf.mul(0, 123) == 0
    assert gf.mul(1, 123) == 123
    for a in (1, 2, 3, 87, 255):
        for b in (1, 5, 200, 254):
            assert gf.mul(a, b) == gf.mul(b, a)


def test_every_non_zero_element_has_an_inverse():
    gf = GF256()
    for a in range(1, 256):
        assert gf.mul(a, gf.inv(a)) == 1
    assert gf.inv(0) == 0, "zero has no inverse; the convention must be 0"


# -------------------------------------------------------------- the table
def test_the_computed_table_is_the_published_aes_sbox():
    """The strongest available test of a GF(2^8) implementation."""
    got = algebraic_sbox(GF256())
    assert np.array_equal(got, np.array(AES_SBOX_REFERENCE, dtype=np.uint8))


def test_the_computed_inverse_table_undoes_it():
    gf = GF256()
    box, inv = algebraic_sbox(gf), inverse_algebraic_sbox(gf)
    assert np.array_equal(inv[box.astype(np.int64)], np.arange(256))
    assert np.array_equal(box[inv.astype(np.int64)], np.arange(256))


def test_the_affine_map_alone_matches_its_definition():
    # S(0) = affine(0) = 0x63, because inv(0) = 0
    assert affine(0) == 0x63
    assert rotl8(0x80, 1) == 0x01, "rotation must wrap, not shift"


@pytest.mark.parametrize("poly", irreducible_polynomials(8)[:6])
def test_every_field_representation_gives_a_valid_table(poly):
    box = algebraic_sbox(GF256(poly))
    assert is_bijective(box)


# ---------------------------------------------------------- the properties
def test_the_algebraic_table_attains_the_known_optimum():
    """Differential uniformity 4 and nonlinearity 112 are theorems, not draws."""
    box = algebraic_sbox(GF256())
    assert differential_uniformity(box) == 4
    assert nonlinearity(box) == 112
    assert algebraic_degree(box) == 7
    assert abs(avalanche(box) - 4.0) < 0.1


def test_the_field_choice_changes_the_table_but_not_its_resistance():
    """Which of the 30 polynomials is used is a representation, not a strength."""
    ref = np.array(AES_SBOX_REFERENCE, dtype=np.uint8)
    for poly in irreducible_polynomials(8)[:5]:
        box = algebraic_sbox(GF256(poly))
        assert differential_uniformity(box) == 4
        assert nonlinearity(box) == 112
        if poly != AES_POLY:
            assert not np.array_equal(box, ref)


def test_a_random_keyed_table_is_measurably_worse():
    """The comparison the documentation is built on, as a regression."""
    from avsec.substitution import crypto_sbox

    alg = algebraic_sbox(GF256())
    rnd = crypto_sbox(KEY, b"ctx")
    assert differential_uniformity(rnd) > differential_uniformity(alg)
    assert nonlinearity(rnd) < nonlinearity(alg)


# ------------------------------------------------------- putting the key in
def test_key_whitening_preserves_the_properties_exactly():
    """``S(v XOR a) XOR b`` is affine-equivalent, so DU and NL cannot change."""
    gf = GF256()
    base = analyse(algebraic_sbox(gf))
    keyed = KeyedAlgebraicSbox(KEY, gf)
    for ctx in (b"a", b"b", b"block-7"):
        got = analyse(keyed.table(ctx))
        assert got["differential_uniformity"] == base["differential_uniformity"]
        assert got["nonlinearity"] == base["nonlinearity"]
        assert got["algebraic_degree"] == base["algebraic_degree"]


def test_the_keyed_table_is_bijective_and_its_inverse_is_exact():
    keyed = KeyedAlgebraicSbox(KEY)
    t, inv = keyed.table(b"ctx"), keyed.inverse_table(b"ctx")
    assert is_bijective(t)
    assert np.array_equal(inv[t.astype(np.int64)], np.arange(256))


def test_a_different_key_gives_a_different_keyed_table():
    a = KeyedAlgebraicSbox(KEY).table(b"ctx")
    b = KeyedAlgebraicSbox(bytes(range(1, 33))).table(b"ctx")
    assert not np.array_equal(a, b)


def test_different_contexts_give_different_tables():
    keyed = KeyedAlgebraicSbox(KEY)
    assert not np.array_equal(keyed.table(b"a"), keyed.table(b"b"))


# ------------------------------------------------------ inside the scheme
@pytest.mark.parametrize("mode", ("session", "frame", "block"))
def test_b2s_round_trips_on_the_algebraic_tables(mode):
    from avsec import sources as S
    from avsec.substitution import SubstitutionPermutationScrambler

    img = S.PATTERNS["texture"](192, 256)
    sc = SubstitutionPermutationScrambler(12, 16, KEY, b"TESTSID0", mode=mode,
                                          source="algebraic")
    assert np.array_equal(sc.descramble(sc.scramble(img, 0), 0), sc._fit(img))


def test_an_unknown_table_source_is_refused():
    from avsec.substitution import SubstitutionPermutationScrambler

    with pytest.raises(ValueError):
        SubstitutionPermutationScrambler(12, 16, KEY, b"TESTSID0",
                                         source="whatever")
