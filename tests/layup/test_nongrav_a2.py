"""Fitting the non-gravitational A2 (transverse Yarkovsky term), issue #351.

Generates a self-consistent, noise-free astrometric arc from a known orbit *with*
a known A2 (forward-integrated through ASSIST's Marsden non-grav force, 1/r^2
g(r)), then fits it two ways through the Cartesian engine:

* ``fit_a2=False`` -- the standard 6-parameter fit. It cannot absorb the A2
  along-track drift, so it leaves a visibly nonzero chi-square.
* ``fit_a2=True``  -- the joint 7-parameter (state + A2) fit. The A2 variational
  partial comes from ASSIST (a zero-seeded variational particle whose parameter
  direction dA2=1 is set via particle_params); the fit must drive chi-square to
  ~0 and recover both the state and the true A2.

That the 7-parameter fit converges to the true A2 from a perturbed seed is the
end-to-end check on the A2 partial: a wrong partial would not converge to it.

A2 is only weakly constrained on short arcs, so the arc spans several years.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from layup.orbitfit import orbitfit
from layup.routines import FitResult, Observation, get_ephem, run_from_vector_with_initial_guess

CACHE = os.path.expanduser("~/Library/Caches/layup")
_EPHEM = ("linux_p1550p2650.440", "sb441-n16.bsp")
_EPHEM_OK = all(os.path.exists(os.path.join(CACHE, f)) for f in _EPHEM)
pytestmark = pytest.mark.skipif(
    not _EPHEM_OK, reason="ASSIST ephemeris not in layup cache; run `layup bootstrap`"
)

# Apophis-like barycentric ICRF state (JPL Horizons, DE441) at the epoch, AU/day.
_EPOCH = 2456323.5
_STATE = np.array(
    [
        -6.885941010606326e-01,
        7.853011708041518e-01,
        2.744012411423299e-01,
        -1.236352140633316e-02,
        -7.926849409608376e-03,
        -3.263604704470501e-03,
    ]
)
_TRUE_A2 = -5.0e-14  # au/day^2, transverse
# g(r) = (r/r0)^-2: the standard small-body 1/r^2 non-grav model. Must match the
# C++ fitter's setup (orbit_fit.cpp).
_GR = dict(alpha=1.0, nm=2.0, nk=0.0, nn=1.0, r0=1.0)
_C = 2.99792458e8 * 86400.0 / 1.495978707e11  # au/day


def _build_arc(arc_days=4 * 365.25, n=40):
    import assist
    import rebound

    ephem = assist.Ephem(os.path.join(CACHE, _EPHEM[0]), os.path.join(CACHE, _EPHEM[1]))
    jr = ephem.jd_ref

    def ast_at(t_jd):
        sim = rebound.Simulation()
        sim.t = _EPOCH - jr
        sim.add(x=_STATE[0], y=_STATE[1], z=_STATE[2], vx=_STATE[3], vy=_STATE[4], vz=_STATE[5])
        ax = assist.Extras(sim, ephem)
        ax.forces = ["SUN", "PLANETS", "ASTEROIDS", "NON_GRAVITATIONAL", "GR_SIMPLE"]
        for k, v in _GR.items():
            setattr(ax, k, v)
        ax.particle_params = np.array([0.0, _TRUE_A2, 0.0])
        sim.integrate(t_jd - jr)
        p = sim.particles[0]
        return np.array([p.x, p.y, p.z])

    obs = []
    for dt in np.linspace(0.0, arc_days, n):
        t_jd = _EPOCH + dt
        e = ephem.get_particle("Earth", t_jd - jr)
        r_obs = np.array([e.x, e.y, e.z])
        v_obs = np.array([e.vx, e.vy, e.vz])
        lt = 0.0
        for _ in range(3):  # geocentric light-time to the asteroid
            rho = ast_at(t_jd - lt) - r_obs
            lt = np.linalg.norm(rho) / _C
        rho /= np.linalg.norm(rho)
        ra = np.arctan2(rho[1], rho[0])
        dec = np.arcsin(rho[2])
        o = Observation.from_astrometry_with_id("apophis_like", ra, dec, t_jd, list(r_obs), list(v_obs))
        o.ra_unc = o.dec_unc = 0.1 / 206265.0  # 0.1 arcsec
        obs.append(o)
    return obs


def _seed(state, a2=0.0):
    g = FitResult()
    g.state = list(state)
    g.epoch = _EPOCH
    g.flag = 0
    g.a2 = a2
    return g


def test_a2_joint_fit_recovers_state_and_a2():
    obs = _build_arc()
    ephem = get_ephem(CACHE)
    # Perturbed seed so the fit must converge via the partials (incl. A2).
    pert = _STATE.copy()
    pert[:3] += 1.0e-7
    pert[3:] += 1.0e-9

    fit = run_from_vector_with_initial_guess(ephem, _seed(pert, 0.0), obs, 100, True)
    assert fit.flag == 0
    assert fit.fit_a2 is True
    assert fit.ndof == 2 * len(obs) - 7

    state = np.array(fit.state)
    pos_rel = np.linalg.norm(state[:3] - _STATE[:3]) / np.linalg.norm(_STATE[:3])
    vel_rel = np.linalg.norm(state[3:] - _STATE[3:]) / np.linalg.norm(_STATE[3:])
    assert pos_rel < 1e-8, f"state position drift {pos_rel:.2e}"
    assert vel_rel < 1e-7, f"state velocity drift {vel_rel:.2e}"
    # The A2 partial is correct iff the fit converges to the true A2.
    assert abs(fit.a2 - _TRUE_A2) / abs(_TRUE_A2) < 1e-2, f"recovered A2 {fit.a2:.4e} vs {_TRUE_A2:.4e}"
    assert fit.csq < 1e-5, f"unexpected chi-square {fit.csq:.3e}"
    assert np.isfinite(fit.a2_unc) and fit.a2_unc > 0.0


def test_six_param_fit_unaffected_and_biased_by_a2():
    """fit_a2=False is the unchanged 6-parameter path; on an arc with a real A2
    drift it cannot reach ~0 chi-square (so the 7-parameter fit is a clear
    improvement), and it reports no fitted A2."""
    obs = _build_arc()
    ephem = get_ephem(CACHE)
    pert = _STATE.copy()
    pert[:3] += 1.0e-7
    pert[3:] += 1.0e-9

    fit6 = run_from_vector_with_initial_guess(ephem, _seed(pert, 0.0), obs, 100, False)
    fit7 = run_from_vector_with_initial_guess(ephem, _seed(pert, 0.0), obs, 100, True)
    assert fit6.flag == 0
    assert fit6.fit_a2 is False
    assert fit6.ndof == 2 * len(obs) - 6
    # The unmodeled A2 drift leaves the 6-parameter fit with a much larger
    # chi-square than the 7-parameter fit that absorbs it.
    assert fit6.csq > 1e3 * fit7.csq


def test_a2_weak_constraint_guard_on_short_arc():
    """On a short arc the A2 column is nearly collinear with the state, so the
    joint fit is rank-deficient. The fitter must flag this (flag=6) rather than
    returning a contaminated, over-confident solution."""
    obs = _build_arc(arc_days=10.0, n=10)  # ~10 days: A2 not separable from state
    ephem = get_ephem(CACHE)
    fit = run_from_vector_with_initial_guess(ephem, _seed(_STATE, 0.0), obs, 100, True)
    assert fit.flag == 6, f"expected weak-constraint flag 6, got {fit.flag}"


def _build_arc_array(arc_days=4 * 365.0, n=36):
    """The same A2 arc as a structured ra/dec array (obsTime/stn) for the
    orbitfit() driver. Truth ra/dec are generated against the driver's *own*
    obscodes_to_barycentric observer states (geocenter, code 500) so the
    noise-free joint fit reaches ~0 chi-square; plus a perturbed initial guess.
    """
    from datetime import datetime, timedelta

    import assist
    import numpy.lib.recfunctions as rfn
    import rebound
    import spiceypy as spice

    from layup.orbitfit import _get_result_dtypes
    from layup.utilities.data_processing_utilities import LayupObservatory, get_cov_columns
    from layup.utilities.datetime_conversions import convert_tdb_date_to_julian_date

    ephem = assist.Ephem(os.path.join(CACHE, _EPHEM[0]), os.path.join(CACHE, _EPHEM[1]))
    jr = ephem.jd_ref

    def ast_at(t_jd):
        sim = rebound.Simulation()
        sim.t = _EPOCH - jr
        sim.add(x=_STATE[0], y=_STATE[1], z=_STATE[2], vx=_STATE[3], vy=_STATE[4], vz=_STATE[5])
        ax = assist.Extras(sim, ephem)
        ax.forces = ["SUN", "PLANETS", "ASTEROIDS", "NON_GRAVITATIONAL", "GR_SIMPLE"]
        for k, v in _GR.items():
            setattr(ax, k, v)
        ax.particle_params = np.array([0.0, _TRUE_A2, 0.0])
        sim.integrate(t_jd - jr)
        p = sim.particles[0]
        return np.array([p.x, p.y, p.z])

    obstimes = [
        (datetime(2013, 1, 31) + timedelta(days=float(d))).strftime("%Y-%m-%dT%H:%M:%S")
        for d in np.linspace(0.0, arc_days, n)
    ]
    lo = LayupObservatory(cache_dir=CACHE)
    base = np.array(
        [("t", t, "500") for t in obstimes],
        dtype=[("provID", "U4"), ("obsTime", "U32"), ("stn", "U4")],
    )
    et = np.array([spice.str2et(r["obsTime"]) for r in base], dtype="<f8")
    base = rfn.append_fields(base, "et", et, usemask=False, asrecarray=True)
    pv = np.atleast_1d(lo.obscodes_to_barycentric(base))

    data = np.empty(
        len(obstimes),
        dtype=[("provID", "U4"), ("obsTime", "U32"), ("stn", "U4"), ("ra", "f8"), ("dec", "f8")],
    )
    for i, t in enumerate(obstimes):
        jd = convert_tdb_date_to_julian_date(t, CACHE)
        r_obs = np.array([pv[i]["x"], pv[i]["y"], pv[i]["z"]])
        lt = 0.0
        for _ in range(3):
            rho = ast_at(jd - lt) - r_obs
            lt = np.linalg.norm(rho) / _C
        rho /= np.linalg.norm(rho)
        data[i] = (
            "t",
            t,
            "500",
            np.degrees(np.arctan2(rho[1], rho[0])) % 360.0,
            np.degrees(np.arcsin(rho[2])),
        )

    guess = np.zeros(1, dtype=_get_result_dtypes("provID"))
    guess["provID"] = "t"
    pert = _STATE.copy()
    pert[:3] += 1.0e-7
    pert[3:] += 1.0e-9
    for k, v in zip(("x", "y", "z", "xdot", "ydot", "zdot"), pert):
        guess[k] = v
    guess["epochMJD_TDB"] = _EPOCH - 2400000.5
    guess["flag"] = 0
    guess["FORMAT"] = "BCART_EQ"
    guess["method"] = "seed"
    for c in get_cov_columns():
        guess[c] = 0.0
    return data, guess


def test_orbitfit_driver_fits_a2():
    """The orbitfit() driver with fit_nongrav=True adds a2/a2_unc columns and
    recovers the true A2 from a ground-based ra/dec arc."""
    data, guess = _build_arc_array()
    fit = orbitfit(data, cache_dir=CACHE, initial_guess=guess, fit_nongrav=True)
    assert "a2" in fit.dtype.names and "a2_unc" in fit.dtype.names
    row = fit[0]
    assert row["flag"] == 0
    assert row["ndof"] == 2 * len(data) - 7
    assert abs(row["a2"] - _TRUE_A2) / abs(_TRUE_A2) < 1e-2, f"recovered A2 {row['a2']:.4e}"
    assert np.isfinite(row["a2_unc"]) and row["a2_unc"] > 0.0
    assert row["csq"] < 1e-5


def test_orbitfit_default_schema_unchanged():
    """Without fit_nongrav the result schema has no a2 columns and the 6-parameter
    fit still succeeds (no regression)."""
    data, guess = _build_arc_array()
    fit = orbitfit(data, cache_dir=CACHE, initial_guess=guess)
    assert "a2" not in fit.dtype.names
    assert "a2_unc" not in fit.dtype.names
    assert fit[0]["flag"] == 0


def test_orbitfit_driver_short_arc_reports_a2_nan():
    """Through the driver, a short arc that cannot constrain A2 falls back to the
    6-parameter solution: flag is success, the state is still recovered, and a2 is
    reported as NaN (not a contaminated value)."""
    data, guess = _build_arc_array(arc_days=10.0, n=10)
    fit = orbitfit(data, cache_dir=CACHE, initial_guess=guess, fit_nongrav=True)
    row = fit[0]
    assert row["flag"] == 0  # 6-parameter fallback succeeded
    assert np.isnan(row["a2"]) and np.isnan(row["a2_unc"])
    # The fallback state is the clean 6-parameter orbit, still near truth.
    state = np.array([row["x"], row["y"], row["z"], row["xdot"], row["ydot"], row["zdot"]])
    pos_rel = np.linalg.norm(state[:3] - _STATE[:3]) / np.linalg.norm(_STATE[:3])
    assert pos_rel < 1e-6
