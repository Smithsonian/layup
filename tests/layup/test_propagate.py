"""Moving a fitted orbit to a different reference epoch (issue #578).

``convert`` changes the parameterization at a fixed epoch; nothing changed the
epoch. Layup already emits two conventions -- a cold-started fit is referenced
to the middle observation of its IOD triplet, a warm-started fit inherits its
initial guess's epoch -- and could not reconcile them.

The central test is COMPOSITION: propagating A to B to C must agree with A to C
directly, in the state, in the covariance, and in the state-transition matrix.
That is a self-check requiring no reference orbit, which makes it the right
regression test here -- it would catch a wrong sign, a dropped perturber or a
mis-indexed Phi without anything to compare against.
"""

import numpy as np
import pytest

from layup.routines import FitResult, get_ephem, propagate_state

from _bk_guards import EPHEM_CACHE, requires_ephem

pytestmark = requires_ephem

CACHE = str(EPHEM_CACHE)

# A main-belt-like state, and three epochs well inside ephemeris coverage.
STATE = [1.7, 0.9, 0.2, -5.5e-3, 9.1e-3, 1.2e-4]
EPOCH_A, EPOCH_B, EPOCH_C = 2460000.5, 2460400.5, 2460900.5

# Coverage is 1550-2650 (see #563); these sit outside it at either end.
BEYOND_UPPER, BEFORE_LOWER = 2708000.0, 2270000.0


def _fit(epoch=EPOCH_A, diag=(1e-8, 1e-8, 1e-8, 1e-12, 1e-12, 1e-12)):
    f = FitResult()
    f.state = list(STATE)
    f.epoch = epoch
    f.flag = 0
    cov = np.zeros((6, 6))
    np.fill_diagonal(cov, diag)
    f.cov = list(cov.flatten())
    return f


@pytest.fixture(scope="module")
def ephem():
    return get_ephem(CACHE)


@pytest.fixture(scope="module")
def legs(ephem):
    """A->B, B->C and the direct A->C, computed once."""
    g, p1 = propagate_state(ephem, _fit(), EPOCH_B)
    h, p2 = propagate_state(ephem, g, EPOCH_C)
    d, pd = propagate_state(ephem, _fit(), EPOCH_C)
    m = lambda p: np.asarray(list(p), dtype=float).reshape(6, 6)
    return {"two_step": h, "direct": d, "p1": m(p1), "p2": m(p2), "pd": m(pd)}


def test_state_composes(legs):
    a = np.asarray(list(legs["two_step"].state))
    b = np.asarray(list(legs["direct"].state))
    assert np.linalg.norm(a - b) < 1e-10, "A->B->C must reach the same state as A->C"


def test_covariance_composes(legs):
    a = np.asarray(list(legs["two_step"].cov)).reshape(6, 6)
    b = np.asarray(list(legs["direct"].cov)).reshape(6, 6)
    assert np.abs(a - b).max() / np.abs(b).max() < 1e-9


def test_the_stm_composes(legs):
    """Phi(B->C) Phi(A->B) == Phi(A->C). The matrix is a group element, and a
    mis-indexed one would still look plausible on its own."""
    assert np.abs(legs["p2"] @ legs["p1"] - legs["pd"]).max() < 1e-6


def test_the_stm_is_not_the_identity(legs):
    """🔴 The failure this guards against is silent.

    ``reb_simulation_add_variation_1st_order`` defaults to ``testparticle=-1``,
    and ASSIST fills variational accelerations only where ``testparticle == j``.
    With the default, Phi comes back as exactly the identity with no error
    raised -- a covariance propagated that way is unchanged, which looks
    entirely reasonable in output.
    """
    assert np.abs(legs["pd"] - np.eye(6)).max() > 1.0


def test_the_propagated_covariance_is_a_covariance(legs):
    c = np.asarray(list(legs["direct"].cov)).reshape(6, 6)
    assert np.abs(c - c.T).max() < 1e-18, "symmetric"
    assert np.linalg.eigvalsh(c).min() >= 0.0, "positive semi-definite"


def test_round_trip(ephem):
    out, _ = propagate_state(ephem, _fit(), EPOCH_C)
    back, _ = propagate_state(ephem, out, EPOCH_A)
    assert np.linalg.norm(np.asarray(list(back.state)) - np.asarray(STATE)) < 1e-10


def test_a_zero_interval_is_the_identity_and_costs_nothing(ephem):
    out, stm = propagate_state(ephem, _fit(), EPOCH_A)
    assert out.epoch == EPOCH_A
    assert np.allclose(np.asarray(list(stm)).reshape(6, 6), np.eye(6))
    assert list(out.state) == STATE


def test_a_zero_covariance_stays_zero(ephem):
    """An unconverged fit carries a zeroed covariance (#547); Phi 0 Phi^T = 0 is
    the right answer for it, not an error."""
    f = _fit(diag=(0, 0, 0, 0, 0, 0))
    out, _ = propagate_state(ephem, f, EPOCH_B)
    assert out.flag == 0
    assert np.allclose(np.asarray(list(out.cov)), 0.0)


@pytest.mark.parametrize("target", [BEYOND_UPPER, BEFORE_LOWER])
def test_outside_ephemeris_coverage_fails_without_moving_the_orbit(ephem, target):
    """The orbit must come back UNCHANGED, not plausibly wrong: a caller that
    ignores the flag then has the original rather than a fabricated state."""
    out, _ = propagate_state(ephem, _fit(), target)
    assert out.flag != 0
    assert out.epoch == EPOCH_A
    assert list(out.state) == STATE


# --- the Python wrapper, over rows of orbitfit output --- #


def _row(epoch_mjd=EPOCH_A - 2400000.5):
    """One row shaped like orbitfit output, with the covariance columns."""
    from layup.orbitfit import _get_result_dtypes

    r = np.zeros(1, dtype=_get_result_dtypes("provID"))
    r["provID"] = "test"
    r["FORMAT"] = "BCART_EQ"
    r["flag"] = 0
    for c, v in zip(("x", "y", "z", "xdot", "ydot", "zdot"), STATE):
        r[c] = v
    r["epochMJD_TDB"] = epoch_mjd
    for i, v in enumerate((1e-8, 1e-8, 1e-8, 1e-12, 1e-12, 1e-12)):
        r[f"cov_{i}_{i}"] = v
    return r


def test_wrapper_moves_the_epoch_column():
    from layup.propagate import propagate_row

    target = EPOCH_C - 2400000.5
    out = propagate_row(_row(), target, cache_dir=CACHE)
    assert out["flag"][0] == 0
    assert out["epochMJD_TDB"][0] == pytest.approx(target)
    assert out["x"][0] != pytest.approx(STATE[0]), "the state must actually move"


def test_wrapper_agrees_with_the_binding():
    """The wrapper must not introduce a unit error of its own -- MJD vs JD is
    exactly the sort of 2400000.5 mistake this class of code invites."""
    from layup.propagate import propagate_row

    out = propagate_row(_row(), EPOCH_C - 2400000.5, cache_dir=CACHE)
    ref, _ = propagate_state(get_ephem(CACHE), _fit(), EPOCH_C)
    got = np.asarray([out[c][0] for c in ("x", "y", "z", "xdot", "ydot", "zdot")])
    assert np.linalg.norm(got - np.asarray(list(ref.state))) < 1e-12


def test_wrapper_brings_two_epochs_to_a_common_one():
    """The use this exists for: two orbits referenced differently cannot be
    compared until one of them is moved."""
    from layup.propagate import propagate

    from layup.propagate import propagate_row

    # The SAME orbit expressed at two epochs -- the second derived from the
    # first, not merely the same state vector relabelled, which would be a
    # different trajectory.
    first = _row(EPOCH_A - 2400000.5)
    second = propagate_row(first, EPOCH_B - 2400000.5, cache_dir=CACHE)
    rows = np.concatenate([first, second])
    assert rows["epochMJD_TDB"][0] != rows["epochMJD_TDB"][1]
    assert rows["x"][0] != rows["x"][1]
    out = propagate(rows, EPOCH_C - 2400000.5, cache_dir=CACHE)
    assert len(out) == 2
    assert out["epochMJD_TDB"][0] == pytest.approx(out["epochMJD_TDB"][1])
    # Same orbit at two epochs, so brought to a common one they must agree.
    for c in ("x", "y", "z"):
        assert out[c][0] == pytest.approx(out[c][1], abs=1e-10)


def test_wrapper_keeps_failed_rows_rather_than_dropping_them():
    from layup.propagate import propagate

    rows = np.concatenate([_row(), _row()])
    out = propagate(rows, BEYOND_UPPER - 2400000.5, cache_dir=CACHE)
    assert len(out) == len(rows), "length and order must be preserved"
    assert (out["flag"] != 0).all()
    assert out["epochMJD_TDB"][0] == pytest.approx(EPOCH_A - 2400000.5), "unchanged"
