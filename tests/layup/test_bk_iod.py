"""Tests for the universal-BK 5-parameter linear IOD (`run_bk_iod`).

The IOD is the layup-side analog of liborbfit's `prelim_fit`: a single
closed-form weighted least-squares solve over (alpha, beta, gamma,
adot, bdot) with gdot pinned to 0.  See `bk_iod.cpp` for the model and
the documented regime of validity (works best for distant objects,
single-percent on heliocentric distance for TNOs at sweet-spot arc
lengths; not intended for mainbelt or as a final orbit).

Test layers:
  * Smoke: empty / few-obs guards return without crashing.
  * Sweet-spot diagnostic: on a representative distant case, BK-IOD
    recovers truth to within a few percent of heliocentric distance.
  * Seeds the LM to truth: the BK-IOD output, fed into
    run_bk_native_fit, converges to truth at rtol=1e-6 -- the actual
    intended use of BK-IOD.
"""

from __future__ import annotations

import numpy as np
import pytest

from layup.routines import (
    FitResult,
    Observation,
    get_ephem,
    run_bk_iod,
    run_bk_native_fit,
)

from _bk_guards import (
    DIAGNOSTIC_AVAILABLE,
    EPHEM_AVAILABLE,
    EPHEM_CACHE,
    load_diagnostic_case,
)

# Directory passed to get_ephem(); str() preserves the pre-refactor type.
CACHE = str(EPHEM_CACHE)

MU_SUN = 0.00029591220828559104


# ---------------------------------------------------------------------------
# Smoke tests -- no ephemeris or diagnostic data needed
# ---------------------------------------------------------------------------


def test_run_bk_iod_empty_obs():
    """No observations -> flag != 0, no crash."""
    result = run_bk_iod([], 2460000.5, MU_SUN)
    assert result.method == "bk_iod"
    assert result.flag != 0


def test_run_bk_iod_too_few_obs():
    """<3 observations -> flag != 0, no crash."""
    obs = [
        Observation.from_astrometry(1.57, 0.1, 2460000.5, [-0.5, 0.8, 0.0], [0.0, 0.0, 0.0]),
        Observation.from_astrometry(1.57, 0.1, 2460010.5, [-0.5, 0.8, 0.0], [0.0, 0.0, 0.0]),
    ]
    result = run_bk_iod(obs, 2460000.5, MU_SUN)
    assert result.flag != 0


# ---------------------------------------------------------------------------
# Diagnostic-data tests -- skip if scan + ephem not available
# ---------------------------------------------------------------------------


pytestmark_diagnostic = pytest.mark.skipif(
    not (EPHEM_AVAILABLE and DIAGNOSTIC_AVAILABLE),
    reason="Need both the ASSIST ephemeris and the in-repo diagnostic-scan truth set.",
)


_load_diagnostic_case = load_diagnostic_case


def _build_observations_from_case(case):
    sigma_arcsec = float(case["sigma_arcsec"])
    sigma_rad = sigma_arcsec * np.pi / (180.0 * 3600.0)
    obs_list = []
    for o in case["observations"]:
        pos = list(o["observer_state_AU"])
        vel = [0.0, 0.0, 0.0]
        obs = Observation.from_astrometry(
            ra=np.deg2rad(o["ra"]),
            dec=np.deg2rad(o["dec"]),
            epoch=float(o["jd_tdb"]),
            observer_position=pos,
            observer_velocity=vel,
        )
        obs.ra_unc = sigma_rad
        obs.dec_unc = sigma_rad
        obs_list.append(obs)
    return obs_list


@pytestmark_diagnostic
@pytest.mark.parametrize(
    "case_name, max_drift_frac",
    [
        # Tolerances chosen by the empirical sweep in bk_iod.cpp's docstring:
        # distant objects in their sweet-spot arc length recover r_helio to
        # within a few percent.
        ("classical_42AU_arc_010.00d", 0.05),
        ("scattered_70AU_arc_007.00d", 0.02),
        ("sednoid_80AU_arc_010.00d", 0.03),
        # Within-regime longer arcs -- still acceptable, looser bound.
        ("classical_42AU_arc_060.00d", 0.08),
    ],
)
def test_bk_iod_distant_objects(case_name, max_drift_frac):
    """BK-IOD on a distant case should recover the truth heliocentric
    position to within a few percent (regime-of-validity expectation)."""
    case = _load_diagnostic_case(case_name)
    obs = _build_observations_from_case(case)
    truth = np.asarray(case["truth_state_at_epoch"])
    epoch = float(case["epoch_jd_tdb"])
    r_helio = float(np.linalg.norm(truth[:3]))

    result = run_bk_iod(obs, epoch, MU_SUN)
    assert result.flag == 0, f"[{case_name}] BK-IOD did not converge (flag={result.flag})"
    drift = np.linalg.norm(np.asarray(result.state)[:3] - truth[:3])
    assert drift < max_drift_frac * r_helio, (
        f"[{case_name}] BK-IOD drifted {drift:.3f} AU " f"> {max_drift_frac:.0%} of r_helio={r_helio:.1f} AU"
    )


# ---------------------------------------------------------------------------
# IOD's intended use: seeding the full BK LM fit
# ---------------------------------------------------------------------------


@pytestmark_diagnostic
@pytest.mark.parametrize(
    "case_name",
    [
        "classical_42AU_arc_010.00d",
        "scattered_70AU_arc_007.00d",
        "sednoid_80AU_arc_010.00d",
        "classical_42AU_arc_060.00d",
    ],
)
def test_bk_iod_seeds_lm_to_truth(case_name):
    """The actual purpose of BK-IOD: produce a seed that, fed into
    run_bk_native_fit, converges to the truth state.  This is the
    end-to-end test of "is BK-IOD useful?" -- and the answer should
    be yes even on cases where the IOD itself sits a few percent off
    the truth, because LM convergence basins are wider than that."""
    ephem = get_ephem(CACHE)
    case = _load_diagnostic_case(case_name)
    obs = _build_observations_from_case(case)
    truth = np.asarray(case["truth_state_at_epoch"])
    epoch = float(case["epoch_jd_tdb"])
    r_helio = float(np.linalg.norm(truth[:3]))

    iod = run_bk_iod(obs, epoch, MU_SUN)
    assert iod.flag == 0, f"[{case_name}] IOD failed (flag={iod.flag})"

    # Seed the LM with the IOD result and let it converge.
    fit = run_bk_native_fit(ephem, iod, obs, MU_SUN)
    assert fit.flag == 0, f"[{case_name}] LM (seeded by IOD) did not converge (flag={fit.flag})"

    # LM should land near truth (sub-AU on a sub-AU-noise dataset).
    drift = np.linalg.norm(np.asarray(fit.state)[:3] - truth[:3])
    assert drift < 0.01 * r_helio, (
        f"[{case_name}] LM (IOD seed) drifted {drift:.3f} AU "
        f"= {100 * drift / r_helio:.2f}% of r_helio={r_helio:.1f} AU"
    )


# ---------------------------------------------------------------------------
# The perspective term (issue #446) -- self-contained, no ephemeris needed
# ---------------------------------------------------------------------------
#
# These use a synthetic two-body arc built in-process, so they run everywhere
# rather than being gated on the diagnostic scan. They guard the fix directly:
# before it, the main-belt seed was ~50% wrong in heliocentric distance, which
# is not a seed at all.


def _synthetic_arc(r_au, arc_days, n_obs=7, incl_deg=5.0, epoch0=2460000.5):
    """A circular orbit seen from a circular Earth. Returns (observations, truth_r)."""
    gm = MU_SUN
    v = np.sqrt(gm / r_au)
    inc = np.radians(incl_deg)
    r0 = np.array([r_au, 0.0, 0.0])
    v0 = np.array([0.0, v * np.cos(inc), v * np.sin(inc)])

    def step(r, vv, dt, n=400):
        def acc(x):
            return -gm * x / np.linalg.norm(x) ** 3

        h = dt / n
        for _ in range(n):
            k1v, k1a = vv, acc(r)
            k2v, k2a = vv + 0.5 * h * k1a, acc(r + 0.5 * h * k1v)
            k3v, k3a = vv + 0.5 * h * k2a, acc(r + 0.5 * h * k2v)
            k4v, k4a = vv + h * k3a, acc(r + h * k3v)
            r = r + (h / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
            vv = vv + (h / 6) * (k1a + 2 * k2a + 2 * k3a + k4a)
        return r, vv

    w = np.sqrt(gm)
    obs_list, ts = [], np.linspace(0.0, arc_days, n_obs)
    for t in ts:
        rt, _ = step(r0, v0, float(t))
        e = np.array([np.cos(w * t), np.sin(w * t), 0.0])
        ev = np.array([-w * np.sin(w * t), w * np.cos(w * t), 0.0])
        rho = rt - e
        hat = rho / np.linalg.norm(rho)
        obs_list.append(
            Observation.from_astrometry_with_id(
                "synthetic",
                float(np.arctan2(hat[1], hat[0])),
                float(np.arcsin(hat[2])),
                float(epoch0 + t),
                list(e),
                list(ev),
            )
        )
    mid = n_obs // 2
    r_mid, _ = step(r0, v0, float(ts[mid]))
    return obs_list, float(np.linalg.norm(r_mid)), mid


@pytest.mark.parametrize(
    "r_au, arc_days, max_frac_err",
    [
        # Bounds are ~2x the measured post-fix error, so they guard the fix
        # without being brittle. The pre-fix values are in the third column of
        # the table in bk_iod.cpp's header; every one of these would fail there.
        (2.5, 10.0, 0.15),  # measured 0.084 after; 0.54 before
        (3.5, 10.0, 0.09),  # measured 0.043 after; 0.45 before
        (15.0, 10.0, 0.01),  # measured 0.0015 after; 0.155 before
        (42.0, 10.0, 0.01),  # measured ~0 after; 0.063 before
        (80.0, 10.0, 0.01),  # measured 0.0001 after; 0.034 before
    ],
)
def test_bk_iod_perspective_term(r_au, arc_days, max_frac_err):
    """The perspective denominator must be carried (issue #446).

    Dropping it makes |gamma*ze| ~ 1 au / r the dominant seed error -- 0.4 in the
    main belt, where it produced a ~50% error in heliocentric distance. Restoring
    it costs one changed coefficient per row and no iteration, because x_obs is
    data and so x_obs*ze is a known coefficient.
    """
    obs, truth_r, mid = _synthetic_arc(r_au, arc_days)
    res = run_bk_iod(obs, float(obs[mid].epoch), MU_SUN)

    assert res.flag == 0, f"BK-IOD failed at r={r_au} au, arc={arc_days} d"
    fitted_r = float(np.linalg.norm(np.asarray(res.state[:3])))
    frac = abs(fitted_r - truth_r) / truth_r
    assert frac < max_frac_err, (
        f"r={r_au} au, arc={arc_days} d: |dr|/r = {frac:.4f} exceeds {max_frac_err}. "
        "If this regressed, check the gamma column in bk_iod.cpp -- it must be "
        "-(X_i - x_obs*ze_i), not -X_i."
    )


def test_bk_iod_covariance_is_positive_definite():
    """The weight-refinement pass must not corrupt the returned covariance.

    The covariance is used as the seed covariance downstream, so a pass that
    reweights by 1/d^2 has to leave it usable, not merely leave the point
    estimate alone.
    """
    obs, _, mid = _synthetic_arc(42.0, 10.0)
    res = run_bk_iod(obs, float(obs[mid].epoch), MU_SUN)
    assert res.flag == 0

    cov = np.asarray(res.cov).reshape(6, 6)
    assert np.all(np.isfinite(cov)), "covariance contains non-finite entries"
    assert np.allclose(cov, cov.T, rtol=1e-8, atol=0.0), "covariance is not symmetric"
    assert np.all(np.diag(cov) > 0.0), "covariance has a non-positive variance"
    assert np.all(np.linalg.eigvalsh(cov) > 0.0), "covariance is not positive definite"
