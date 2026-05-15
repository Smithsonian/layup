"""Layer 3 engine-sweep tests for the universal BK fitter.

Drives both engine='cartesian' and engine='bk_native' against the
diagnostic/scan dataset (outside the repo, at
``~/Dropbox/claude_layup/diagnostic/scan/truth/``) so the design
memory's prediction -- ``bk_native`` matches Cartesian across regimes
and shines on distant short arcs -- can be validated against real
ASSIST-integrated truth.

These tests skip cleanly when either the ASSIST ephemeris or the
diagnostic scan data is unavailable, so machines without either
setup are unaffected.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from layup.orbitfit import _MU_SUN, _run_fit
from layup.routines import (
    FitResult,
    Observation,
    get_ephem,
    run_bk_native_fit,
    run_from_vector_with_initial_guess,
)

# ---------------------------------------------------------------------------
# Environment guards
# ---------------------------------------------------------------------------

CACHE = os.path.expanduser("~/Library/Caches/layup")
EPHEM_PLANETS = os.path.join(CACHE, "linux_p1550p2650.440")
EPHEM_SMALLBODIES = os.path.join(CACHE, "sb441-n16.bsp")
EPHEM_AVAILABLE = os.path.exists(EPHEM_PLANETS) and os.path.exists(EPHEM_SMALLBODIES)

DIAGNOSTIC_SCAN = Path("~/Dropbox/claude_layup/diagnostic/scan/truth").expanduser()
DIAGNOSTIC_AVAILABLE = DIAGNOSTIC_SCAN.is_dir()

pytestmark = pytest.mark.skipif(
    not (EPHEM_AVAILABLE and DIAGNOSTIC_AVAILABLE),
    reason=(
        f"Skipping Layer 3 BK-everywhere tests: "
        f"ephem at {CACHE} = {EPHEM_AVAILABLE}, "
        f"diagnostic scan at {DIAGNOSTIC_SCAN} = {DIAGNOSTIC_AVAILABLE}."
    ),
)


# ---------------------------------------------------------------------------
# Helpers for loading and converting diagnostic-scan cases
# ---------------------------------------------------------------------------


def _load_case(name: str) -> dict:
    """Load a diagnostic-scan case by stem (e.g., 'classical_42AU_arc_007.00d')."""
    with open(DIAGNOSTIC_SCAN / f"{name}.json") as f:
        return json.load(f)


def _build_observations(case: dict) -> list:
    """Convert a case's observation list into layup Observation objects."""
    obs_list = []
    sigma_arcsec = float(case["sigma_arcsec"])
    sigma_rad = sigma_arcsec * np.pi / (180.0 * 3600.0)
    for o in case["observations"]:
        # observer_state_AU is position-only; we fill velocity with zeros.
        # Velocity is only used for second-order corrections (parallax, light-
        # time second derivative) and the design here doesn't depend on it.
        pos = list(o["observer_state_AU"])
        vel = [0.0, 0.0, 0.0]
        obs_list.append(
            Observation.from_astrometry(
                ra=np.deg2rad(o["ra"]),
                dec=np.deg2rad(o["dec"]),
                epoch=float(o["jd_tdb"]),
                observer_position=pos,
                observer_velocity=vel,
            )
        )
        # Per-observation astrometric uncertainty (in radians, matching sigma_arcsec).
        obs_list[-1].ra_unc = sigma_rad
        obs_list[-1].dec_unc = sigma_rad
    return obs_list


def _truth_seed(case: dict) -> FitResult:
    """Return a FitResult populated with the case's truth state at epoch."""
    fit = FitResult()
    fit.state = list(map(float, case["truth_state_at_epoch"]))
    fit.epoch = float(case["epoch_jd_tdb"])
    return fit


def _r_helio_AU(state) -> float:
    return float(np.linalg.norm(state[:3]))


# ---------------------------------------------------------------------------
# Engine-sweep tests
# ---------------------------------------------------------------------------


# Well-arced cases.  With ~30-60 day arcs of mainbelt or TNO objects, the
# data alone is enough to constrain the orbit; both engines should reach a
# state within sub-AU of truth and agree with each other.
WELL_ARCED_CASES = [
    "mainbelt_2.5AU_arc_030.00d",
    "mainbelt_3.5AU_arc_060.00d",
    "classical_42AU_arc_060.00d",
]


@pytest.mark.parametrize("case_name", WELL_ARCED_CASES)
def test_engine_sweep_well_arced_cases(case_name):
    """With the truth state as the LM seed on a well-constrained arc, both
    engines should converge near truth and agree with each other."""
    ephem = get_ephem(CACHE)
    case = _load_case(case_name)
    obs = _build_observations(case)
    seed = _truth_seed(case)
    truth = np.asarray(case["truth_state_at_epoch"])
    r_helio = _r_helio_AU(truth)
    tol = 0.01 * r_helio  # ~1% of heliocentric distance

    cart_res = _run_fit(ephem, seed, obs, "cartesian")
    bk_res = _run_fit(ephem, seed, obs, "bk_native")

    assert cart_res.flag == 0, f"[{case_name}] Cartesian flag={cart_res.flag}"
    assert bk_res.flag == 0, f"[{case_name}] BK flag={bk_res.flag}"

    cart_drift = np.linalg.norm(np.asarray(cart_res.state)[:3] - truth[:3])
    bk_drift = np.linalg.norm(np.asarray(bk_res.state)[:3] - truth[:3])
    assert cart_drift < tol, f"[{case_name}] Cartesian drift {cart_drift:.4f} > tol {tol:.4f}"
    assert bk_drift < tol, f"[{case_name}] BK drift {bk_drift:.4f} > tol {tol:.4f}"

    # Engine agreement: BK and Cartesian should converge to nearly the same point.
    state_disagreement = np.linalg.norm(np.asarray(bk_res.state)[:3] - np.asarray(cart_res.state)[:3])
    assert state_disagreement < tol, f"[{case_name}] BK and Cartesian disagree by {state_disagreement:.4f} AU"


# Short-arc / distant cases.  These are the cases that motivated BK in the
# first place: the line-of-sight direction is poorly constrained, so the
# Cartesian fit's LM step can walk significantly along that direction.  We
# test that BK does at least as well as Cartesian in this regime, AND that
# BK uses substantially fewer LM iterations (the BK basis is better
# conditioned, so the Marquardt damping doesn't need to fight as hard).
SHORT_ARC_DISTANT_CASES = [
    "scattered_70AU_arc_014.00d",
    "classical_42AU_arc_010.00d",
]


@pytest.mark.parametrize("case_name", SHORT_ARC_DISTANT_CASES)
def test_bk_beats_cartesian_on_short_arc_distant(case_name):
    """In the distant short-arc regime where the line-of-sight is poorly
    constrained, BK should drift no more than Cartesian from truth and use
    substantially fewer LM iterations."""
    ephem = get_ephem(CACHE)
    case = _load_case(case_name)
    obs = _build_observations(case)
    seed = _truth_seed(case)
    truth = np.asarray(case["truth_state_at_epoch"])

    cart_res = _run_fit(ephem, seed, obs, "cartesian")
    bk_res = _run_fit(ephem, seed, obs, "bk_native")

    assert cart_res.flag == 0, f"[{case_name}] Cartesian flag={cart_res.flag}"
    assert bk_res.flag == 0, f"[{case_name}] BK flag={bk_res.flag}"

    cart_drift = np.linalg.norm(np.asarray(cart_res.state)[:3] - truth[:3])
    bk_drift = np.linalg.norm(np.asarray(bk_res.state)[:3] - truth[:3])

    # BK should drift no more than Cartesian.  (In practice it's often
    # *much* less -- e.g. on scattered_70AU_arc_014.00d BK stays ~0.02 AU
    # from truth while Cartesian wanders ~4.5 AU.)
    assert (
        bk_drift <= cart_drift + 1e-6
    ), f"[{case_name}] BK drift {bk_drift:.4f} > Cartesian drift {cart_drift:.4f}"

    # BK should use fewer LM iterations than Cartesian: the BK basis is
    # naturally better-conditioned than the Cartesian state at epoch for
    # short-arc distant targets, so the LM step direction is healthier.
    assert (
        bk_res.niter < cart_res.niter
    ), f"[{case_name}] BK niter={bk_res.niter} not < Cartesian niter={cart_res.niter}"


def test_engine_sweep_produces_method_strings():
    """Sanity: each engine populates FitResult.method with its tag, so a
    downstream sweep harness can tell which engine produced each fit."""
    ephem = get_ephem(CACHE)
    case = _load_case("classical_42AU_arc_060.00d")
    obs = _build_observations(case)
    seed = _truth_seed(case)

    cart_res = _run_fit(ephem, seed, obs, "cartesian")
    bk_res = _run_fit(ephem, seed, obs, "bk_native")
    assert cart_res.method == "orbit_fit"
    assert bk_res.method == "bk_native"


# ---------------------------------------------------------------------------
# Diagnostic helper (not a test) -- used by sweep harness scripts.
# ---------------------------------------------------------------------------


def sweep_cases_from_diagnostic(case_names=None) -> list:
    """Return a list of (case_name, cartesian FitResult, bk_native FitResult)
    tuples for the requested case names (or all 98 cases if None).

    Intended for ad-hoc use from a sweep script that produces tables; not
    invoked by pytest collection.
    """
    if case_names is None:
        case_names = sorted(p.stem for p in DIAGNOSTIC_SCAN.glob("*.json"))
    ephem = get_ephem(CACHE)
    rows = []
    for name in case_names:
        case = _load_case(name)
        obs = _build_observations(case)
        seed = _truth_seed(case)
        rows.append(
            (
                name,
                _run_fit(ephem, seed, obs, "cartesian"),
                _run_fit(ephem, seed, obs, "bk_native"),
            )
        )
    return rows
