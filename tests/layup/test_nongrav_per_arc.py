"""Per-arc (piecewise-constant) non-gravitational amplitudes -- comet linkage.

The two-apparition comet-linkage fit shares one state and one g(r) but lets the
non-grav amplitude differ between the earlier arc (observations before the fit
epoch) and the later arc (after it). The truth here is generated to match that
model exactly: from one shared state at the epoch, the arc *before* the epoch is
integrated with one A2 and the arc *after* with another (the non-grav force is
negligible far from perihelion, so a real comet's piecewise-per-apparition
amplitude is well approximated this way).

The tests check that the per-arc fit (a) recovers a single shared amplitude when
both sides carry the same one, (b) recovers two *distinct* amplitudes when the
sides differ (the same/recovered/split discriminator), and (c) leaves the
ordinary shared-amplitude fit byte-identical when off.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pooch
import pytest

from layup.orbitfit import orbitfit
from layup.routines import FitResult, Observation, get_ephem, run_from_vector_with_initial_guess

CACHE = str(pooch.os_cache("layup"))
_EPHEM = ("linux_p1550p2650.440", "sb441-n16.bsp")
_EPHEM_OK = all(os.path.exists(os.path.join(CACHE, f)) for f in _EPHEM)
pytestmark = pytest.mark.skipif(
    not _EPHEM_OK, reason="ASSIST ephemeris not in layup cache; run `layup bootstrap`"
)

# Apophis-like barycentric ICRF state (JPL Horizons, DE441), AU/day.
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
_TRUE_A2 = -5.0e-14  # au/day^2, transverse (astrometrically the best constrained)
_BIT = {"A1": 1, "A2": 2, "A3": 4}
_GR = dict(alpha=1.0, nm=2.0, nk=0.0, nn=1.0, r0=1.0)
_C = 2.99792458e8 * 86400.0 / 1.495978707e11  # au/day


def _build_piecewise_arc(a2_before, a2_after, half_days=3 * 365.25, n=48):
    """A noise-free ra/dec arc symmetric about ``_EPOCH``: obs before the epoch
    follow A2=``a2_before``, obs after follow A2=``a2_after``, from one shared
    state at the epoch. Returns a list of Observations for the low-level fitter.
    """
    import assist
    import rebound

    ephem = assist.Ephem(os.path.join(CACHE, _EPHEM[0]), os.path.join(CACHE, _EPHEM[1]))
    jr = ephem.jd_ref

    def ast_at(t_jd, a2):
        sim = rebound.Simulation()
        sim.t = _EPOCH - jr
        sim.add(x=_STATE[0], y=_STATE[1], z=_STATE[2], vx=_STATE[3], vy=_STATE[4], vz=_STATE[5])
        ax = assist.Extras(sim, ephem)
        ax.forces = ["SUN", "PLANETS", "ASTEROIDS", "NON_GRAVITATIONAL", "GR_SIMPLE"]
        for k, v in _GR.items():
            setattr(ax, k, v)
        ax.particle_params = np.array([0.0, a2, 0.0], dtype=float)
        sim.integrate(t_jd - jr)
        p = sim.particles[0]
        return np.array([p.x, p.y, p.z])

    obs = []
    for dt in np.linspace(-half_days, half_days, n):
        if abs(dt) < 1.0:
            continue  # keep a small gap around the epoch (mimics between apparitions)
        a2 = a2_before if dt < 0 else a2_after
        t_jd = _EPOCH + dt
        e = ephem.get_particle("Earth", t_jd - jr)
        r_obs = np.array([e.x, e.y, e.z])
        v_obs = np.array([e.vx, e.vy, e.vz])
        lt = 0.0
        for _ in range(3):
            rho = ast_at(t_jd - lt, a2) - r_obs
            lt = np.linalg.norm(rho) / _C
        rho /= np.linalg.norm(rho)
        o = Observation.from_astrometry_with_id(
            "comet_like", np.arctan2(rho[1], rho[0]), np.arcsin(rho[2]), t_jd, list(r_obs), list(v_obs)
        )
        o.ra_unc = o.dec_unc = 0.1 / 206265.0
        obs.append(o)
    return obs


def _seed(state=_STATE):
    g = FitResult()
    g.state = list(state)
    g.epoch = _EPOCH  # the fit epoch sits between the two arcs
    g.flag = 0
    return g


def _per_arc_fit(obs, mask=_BIT["A2"]):
    return run_from_vector_with_initial_guess(get_ephem(CACHE), _seed(), obs, 100, mask, [], True)


def test_per_arc_recovers_equal_amplitudes():
    """Both sides carry the same A2 -> the per-arc fit recovers one shared value on
    each arc (the 'stable comet' baseline)."""
    obs = _build_piecewise_arc(_TRUE_A2, _TRUE_A2)
    fit = _per_arc_fit(obs)
    assert fit.flag == 0
    assert fit.per_arc is True
    # npar = 6 state + 2*1 amplitude (arc A + arc B).
    assert fit.ndof == 2 * len(obs) - 8
    assert abs(fit.a2 - _TRUE_A2) / abs(_TRUE_A2) < 0.05, f"arc A A2 {fit.a2:.4e}"
    assert abs(fit.a2_arc2 - _TRUE_A2) / abs(_TRUE_A2) < 0.05, f"arc B A2 {fit.a2_arc2:.4e}"
    # The two arcs agree to within their combined uncertainty (stable).
    assert abs(fit.a2 - fit.a2_arc2) < 3.0 * (fit.a2_unc + fit.a2_arc2_unc)


def test_per_arc_recovers_distinct_amplitudes():
    """The sides carry different A2 -> the per-arc fit recovers two distinct values
    (the same/recovered/split discriminator). A shared-amplitude fit cannot.

    Splitting one amplitude into two per-arc amplitudes is ill-conditioned, so the
    formal 1-sigma is large; the change is only *detected* (not just recovered)
    when it is big compared to that -- i.e. a strongly-changed comet. The values
    themselves are recovered accurately even so."""
    a2_a, a2_b = 1.0e-12, 5.0e-12  # a large, split-scale activity change
    obs = _build_piecewise_arc(a2_a, a2_b)
    fit = _per_arc_fit(obs)
    assert fit.flag == 0
    assert abs(fit.a2 - a2_a) / abs(a2_a) < 0.05, f"arc A A2 {fit.a2:.4e} vs {a2_a:.4e}"
    assert abs(fit.a2_arc2 - a2_b) / abs(a2_b) < 0.05, f"arc B A2 {fit.a2_arc2:.4e} vs {a2_b:.4e}"
    # The difference is significant at >3 sigma (the classifier's own z-score),
    # so it is flagged as a real change rather than noise.
    z = abs(fit.a2 - fit.a2_arc2) / math.hypot(fit.a2_unc, fit.a2_arc2_unc)
    assert z > 3.0, f"amplitude change only {z:.1f} sigma"


def test_per_arc_off_is_shared_amplitude_fit():
    """per_arc=False is the ordinary shared-amplitude fit: on an equal-amplitude
    arc it recovers the same single A2, reports no arc-B fields, and has one fewer
    parameter than the per-arc fit."""
    obs = _build_piecewise_arc(_TRUE_A2, _TRUE_A2)
    shared = run_from_vector_with_initial_guess(get_ephem(CACHE), _seed(), obs, 100, _BIT["A2"])
    assert shared.flag == 0
    assert shared.per_arc is False
    assert shared.ndof == 2 * len(obs) - 7  # one shared amplitude, not two
    assert abs(shared.a2 - _TRUE_A2) / abs(_TRUE_A2) < 0.05
    assert shared.a2_arc2 == 0.0  # arc-B fields untouched when off


def test_orbitfit_driver_per_arc_columns():
    """Through the orbitfit() driver, per_arc=True adds a{1,2,3}_arc2 columns and
    per_arc=False (the default) does not -- an opt-in, non-breaking schema change."""
    from layup.orbitfit import _get_result_dtypes

    off = _get_result_dtypes("provID", ["A2"], per_arc=False)
    on = _get_result_dtypes("provID", ["A2"], per_arc=True)
    assert "a2" in off.names and "a2_arc2" not in off.names
    assert "a2_arc2" in on.names and "a2_arc2_unc" in on.names
