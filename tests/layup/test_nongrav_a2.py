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


def _build_arc():
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
    for dt in np.linspace(0.0, 4 * 365.25, 40):
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
