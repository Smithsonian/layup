"""Direct numeric test of the Gauss angles-only IOD method (``gauss``).

The Gauss method is purely two-body geometry: given three lines of sight and
the observer positions, it returns candidate heliocentric/barycentric states at
the middle epoch. It needs no ASSIST/REBOUND ephemeris, so this test runs
everywhere (including CI without the ephemeris cache).

Strategy: build a known two-body orbit, propagate it exactly with a
universal-variable Kepler solver to three epochs, synthesize the exact lines of
sight from an analytic Earth-proxy observer, feed them to ``gauss``, and check
that the recovered middle state matches truth. Gauss uses a truncated f/g series,
so the agreement is approximate and tightens for shorter arcs -- we use a short
(few-day) arc and assert accordingly.

Because root selection near opposition is a known angles-only degeneracy (and is
the job of the picker, not of ``gauss`` itself), this test validates the
*math* by selecting the returned root closest to truth.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from layup.routines import Observation, gauss

# GM_sun in AU^3 / day^2 (Gaussian gravitational constant squared).
MU_SUN = 0.00029591220828559104
# Speed of light in AU/day: c = 299792.458 km/s * 86400 s/day / (AU in km).
SPEED_OF_LIGHT = 299792.458 * 86400.0 / 149597870.7


# ---------------------------------------------------------------------------
# Universal-variable two-body propagator (numpy-only; exact, no truncation)
# ---------------------------------------------------------------------------


def _stumpff_c(z: float) -> float:
    if z > 1e-8:
        sz = math.sqrt(z)
        return (1.0 - math.cos(sz)) / z
    if z < -1e-8:
        sz = math.sqrt(-z)
        return (math.cosh(sz) - 1.0) / (-z)
    return 0.5 - z / 24.0  # series limit at z -> 0


def _stumpff_s(z: float) -> float:
    if z > 1e-8:
        sz = math.sqrt(z)
        return (sz - math.sin(sz)) / sz**3
    if z < -1e-8:
        sz = math.sqrt(-z)
        return (math.sinh(sz) - sz) / sz**3
    return 1.0 / 6.0 - z / 120.0  # series limit at z -> 0


def propagate_kepler(r0, v0, dt, mu=MU_SUN):
    """Exact two-body propagation of (r0, v0) by dt via universal variables."""
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    sqrt_mu = math.sqrt(mu)
    r0n = np.linalg.norm(r0)
    vr0 = np.dot(r0, v0) / r0n
    alpha = 2.0 / r0n - np.dot(v0, v0) / mu  # 1/a

    chi = sqrt_mu * abs(alpha) * dt  # ellipse initial guess
    for _ in range(100):
        psi = chi * chi * alpha
        c = _stumpff_c(psi)
        s = _stumpff_s(psi)
        r = chi * chi * c + (vr0 / sqrt_mu) * chi * (1.0 - psi * s) + r0n * (1.0 - psi * c)
        f_func = (
            (r0n * vr0 / sqrt_mu) * chi * chi * c
            + (1.0 - alpha * r0n) * chi**3 * s
            + r0n * chi
            - sqrt_mu * dt
        )
        dchi = f_func / r
        chi -= dchi
        if abs(dchi) < 1e-12:
            break

    psi = chi * chi * alpha
    c = _stumpff_c(psi)
    s = _stumpff_s(psi)
    f = 1.0 - (chi * chi / r0n) * c
    g = dt - (1.0 / sqrt_mu) * chi**3 * s
    rvec = f * r0 + g * v0
    rn = np.linalg.norm(rvec)
    fdot = (sqrt_mu / (r0n * rn)) * (alpha * chi**3 * s - chi)
    gdot = 1.0 - (chi * chi / rn) * c
    vvec = fdot * r0 + gdot * v0
    return rvec, vvec


def _earth_proxy(t, t0=0.0):
    """Analytic 1-AU circular Earth-proxy observer position at time t (days)."""
    n = 2.0 * math.pi / 365.25  # mean motion, rad/day
    ang = n * (t - t0)
    return np.array([math.cos(ang), math.sin(ang), 0.0])


def _ra_dec_from_los(los):
    """RA, Dec (radians) of a line-of-sight vector."""
    x, y, z = los
    ra = math.atan2(y, x)
    dec = math.asin(z / np.linalg.norm(los))
    return ra, dec


def _make_observation(r_obj, t):
    """Build an astrometry Observation: exact line of sight from Earth-proxy."""
    obs_pos = _earth_proxy(t)
    los = np.asarray(r_obj) - obs_pos
    ra, dec = _ra_dec_from_los(los)
    return Observation.from_astrometry(ra, dec, t, list(obs_pos), [0.0, 0.0, 0.0])


# A bound main-belt-ish orbit (a ~ 2.8 AU, e ~ moderate), state at the middle epoch.
_R2_TRUTH = np.array([1.8, 1.2, 0.3])
_V2_TRUTH = np.array([-0.008, 0.010, 0.001])


@pytest.mark.parametrize("half_arc_days", [2.0, 3.0, 5.0])
def test_gauss_recovers_two_body_state(half_arc_days):
    """gauss returns a candidate state matching the true middle state.

    Tolerance scales with arc length because the f/g series is truncated.
    """
    t2 = 0.0
    t1 = t2 - half_arc_days
    t3 = t2 + half_arc_days

    r1, _ = propagate_kepler(_R2_TRUTH, _V2_TRUTH, t1 - t2)
    r3, _ = propagate_kepler(_R2_TRUTH, _V2_TRUTH, t3 - t2)

    o1 = _make_observation(r1, t1)
    o2 = _make_observation(_R2_TRUTH, t2)
    o3 = _make_observation(r3, t3)

    results = gauss(MU_SUN, o1, o2, o3, 0.0, SPEED_OF_LIGHT)
    assert results is not None and len(results) > 0, "gauss returned no roots"

    # Pick the physically correct root: the candidate closest to truth.
    states = [np.asarray(r.state) for r in results]
    best = min(states, key=lambda s: np.linalg.norm(s[:3] - _R2_TRUTH))

    pos_err = np.linalg.norm(best[:3] - _R2_TRUTH) / np.linalg.norm(_R2_TRUTH)
    vel_err = np.linalg.norm(best[3:] - _V2_TRUTH) / np.linalg.norm(_V2_TRUTH)

    # f/g truncation error is ~O((mu/r^3) * tau^2); generous but meaningful.
    assert pos_err < 5e-3, f"position rel err {pos_err:.2e} (arc {2*half_arc_days}d)"
    assert vel_err < 5e-2, f"velocity rel err {vel_err:.2e} (arc {2*half_arc_days}d)"


def test_gauss_returns_none_for_degenerate_collinear_lines():
    """Coplanar/identical lines of sight have an indeterminate range.

    The scalar triple product d0 = rho1 . (rho2 x rho3) vanishes, so the range
    cannot be solved; gauss must report no solution rather than emit a
    non-finite state (the d0 floor guard in gauss.cpp).
    """
    obs = [_make_observation(_R2_TRUTH, ti) for ti in (-2.0, 0.0, 2.0)]
    # Force all three pointings identical -> exactly coplanar (d0 == 0).
    obs[1].rho_hat = obs[0].rho_hat
    obs[2].rho_hat = obs[0].rho_hat
    result = gauss(MU_SUN, obs[0], obs[1], obs[2], 0.0, SPEED_OF_LIGHT)
    assert result is None or len(result) == 0


def test_gauss_returns_none_when_no_root_above_min_distance():
    """When no candidate range exceeds min_distance, gauss reports no solution.

    Exercises the ``roots.empty() -> std::nullopt`` branch deterministically:
    a min_distance far larger than any physical root filters out every root.
    """
    t1, t2, t3 = -3.0, 0.0, 3.0
    r1, _ = propagate_kepler(_R2_TRUTH, _V2_TRUTH, t1 - t2)
    r3, _ = propagate_kepler(_R2_TRUTH, _V2_TRUTH, t3 - t2)
    o1 = _make_observation(r1, t1)
    o2 = _make_observation(_R2_TRUTH, t2)
    o3 = _make_observation(r3, t3)

    # min_distance = 1e9 AU: no real root can exceed it -> no solution.
    result = gauss(MU_SUN, o1, o2, o3, 1.0e9, SPEED_OF_LIGHT)
    assert result is None or len(result) == 0
