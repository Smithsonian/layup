"""The g(r) sublimation laws, checked against their own definitions.

A non-gravitational fit takes five numbers describing how the acceleration falls
off with heliocentric distance:

    g(r) = alpha * (r/r0)^-nm * (1 + (r/r0)^nn)^-nk

Before these presets existed a caller had to supply all five by hand, which is
how a wrong law reaches a fit without anyone noticing -- the fit still converges,
it just answers a different question. The tests below check the constants against
the properties that define them rather than against themselves.
"""

import numpy as np
import pytest

from layup.constants import ASTEROIDAL_GOFR, MARSDEN_1973_GOFR
from layup.orbitfit import _gofr_arg


def g_of_r(r, law):
    """The law as ASSIST evaluates it, written out from the definition."""
    alpha, nm, nn, nk, r0 = law
    return alpha * (r / r0) ** (-nm) * (1.0 + (r / r0) ** nn) ** (-nk)


def test_both_laws_are_normalized_at_one_au():
    """alpha exists to make g(1 au) = 1. That is what makes A1/A2/A3 accelerations
    at 1 au rather than arbitrary scalings, so it is the property to check.

    The Marsden law misses by 2.4e-9, which is not an error in the constants: it
    is where alpha's ten published significant figures run out. Exact
    normalization would want 0.111262042338 against the tabulated 0.1112620426.
    The bound is set from that, so a genuine transcription error in any of the
    five numbers would still be caught by orders of magnitude.
    """
    for name, law in (("Marsden", MARSDEN_1973_GOFR), ("asteroidal", ASTEROIDAL_GOFR)):
        assert g_of_r(1.0, law) == pytest.approx(1.0, abs=1e-8), name


def test_the_asteroidal_law_is_the_inverse_square():
    """JPL reports ALN=1, NM=2, NK=0, R0=1 for asteroids with a measured Yarkovsky
    drift, which is (1 au / r)^2. ASSIST's own paper says the same."""
    for r in (0.5, 1.0, 2.0, 5.0, 30.0):
        assert g_of_r(r, ASTEROIDAL_GOFR) == pytest.approx(r**-2.0, rel=1e-12)


def test_the_marsden_law_falls_off_faster_than_inverse_square():
    """Water ice stops sublimating past a few au, so the cometary law must drop
    away from the asteroidal one beyond r0 -- and must not be confused with it."""
    assert g_of_r(2.0, MARSDEN_1973_GOFR) < g_of_r(2.0, ASTEROIDAL_GOFR)
    assert g_of_r(5.0, MARSDEN_1973_GOFR) < 1e-3 * g_of_r(5.0, ASTEROIDAL_GOFR)
    # inside 1 au the ice is more active, not less
    assert g_of_r(0.5, MARSDEN_1973_GOFR) > g_of_r(1.0, MARSDEN_1973_GOFR)


def test_the_marsden_law_is_monotonically_decreasing():
    r = np.geomspace(0.1, 50.0, 400)
    g = np.array([g_of_r(x, MARSDEN_1973_GOFR) for x in r])
    assert np.all(np.diff(g) < 0.0)


def test_r0_is_where_the_bracket_turns_over():
    """r0 is the sublimation scale, so at r = r0 the bracket contributes a factor
    2^-nk exactly. This pins r0 to its meaning rather than to a number."""
    alpha, nm, nn, nk, r0 = MARSDEN_1973_GOFR
    assert g_of_r(r0, MARSDEN_1973_GOFR) == pytest.approx(alpha * 2.0**-nk, rel=1e-12)


def test_the_presets_are_accepted_by_the_fitting_interface():
    """They exist to be passed to orbitfit's nongrav_gr, so check they survive it
    in order and unchanged."""
    for law in (MARSDEN_1973_GOFR, ASTEROIDAL_GOFR):
        assert _gofr_arg(law) == [float(v) for v in law]
        assert len(_gofr_arg(law)) == 5
    assert _gofr_arg(None) == []  # the default asteroidal path
    with pytest.raises(ValueError):
        _gofr_arg((1.0, 2.0, 3.0))


def test_nn_is_inert_for_the_asteroidal_law():
    """nk = 0 makes the bracket unity whatever nn is. ASSIST defaults nn to
    Marsden's 5.093, so a caller who sets nk without setting nn gets their own k
    with Marsden's n -- this records that the preset does not rely on the default."""
    alpha, nm, nn, nk, r0 = ASTEROIDAL_GOFR
    assert nk == 0.0
    for other_nn in (0.0, 5.093, 1.0):
        law = (alpha, nm, other_nn, nk, r0)
        assert g_of_r(2.0, law) == pytest.approx(g_of_r(2.0, ASTEROIDAL_GOFR), rel=1e-12)
