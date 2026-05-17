"""Tests for the universal-BK 5-parameter linear IOD (`run_bk_iod`).

The IOD is the layup-side analog of liborbfit's `prelim_fit`: a single
closed-form weighted least-squares solve over (alpha, beta, gamma,
adot, bdot) with gdot pinned to 0.  Mathematically exact in the
small-angle / no-gravity limit, which is exactly the regime where
Gauss IOD struggles (short arc, distant target).

Test layers:
  * Smoke: empty/few-obs guards return without crashing.
  * Synthetic-orbit recovery: feed in noise-free observations from a
    known orbit, check the recovered Cartesian state matches.
  * Diagnostic-scan comparison: on representative cases from the
    BK-everywhere diagnostic dataset, BK-IOD should recover the truth
    state at sub-AU level for short-arc distant targets where Gauss
    typically struggles.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from layup.routines import (
    FitResult,
    Observation,
    get_ephem,
    predict_sequence,
    run_bk_iod,
)


CACHE = os.path.expanduser("~/Library/Caches/layup")
EPHEM_PLANETS = os.path.join(CACHE, "linux_p1550p2650.440")
EPHEM_SMALLBODIES = os.path.join(CACHE, "sb441-n16.bsp")
EPHEM_AVAILABLE = os.path.exists(EPHEM_PLANETS) and os.path.exists(EPHEM_SMALLBODIES)

DIAGNOSTIC_SCAN = Path("~/Dropbox/claude_layup/diagnostic/scan/truth").expanduser()
DIAGNOSTIC_AVAILABLE = DIAGNOSTIC_SCAN.is_dir()

MU_SUN = 0.00029591220828559104


# ---------------------------------------------------------------------------
# Smoke tests (no ephemeris needed)
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
# Synthetic-orbit recovery (needs ASSIST for the predict path)
# ---------------------------------------------------------------------------


pytestmark_synthetic = pytest.mark.skipif(
    not EPHEM_AVAILABLE,
    reason=f"ASSIST ephemeris missing at {CACHE}; skipping synthetic-orbit tests.",
)


def _generate_synthetic_observations(ephem, truth_state, truth_epoch, obs_times):
    """Same helper used by test_bk_fit.py.  Uses a fixed barycenter observer
    so the only dynamical content in the synthetic obs is the orbit."""
    observer_position = [0.0, 0.0, 0.0]
    observer_velocity = [0.0, 0.0, 0.0]
    template = [
        Observation.from_astrometry(
            ra=0.0, dec=0.0, epoch=float(t),
            observer_position=observer_position,
            observer_velocity=observer_velocity,
        )
        for t in obs_times
    ]
    truth_fit = FitResult()
    truth_fit.state = list(map(float, truth_state))
    truth_fit.epoch = float(truth_epoch)
    cov = np.zeros((6, 6))
    preds = predict_sequence(ephem, truth_fit, template, cov)
    synth = []
    for t, pr in zip(obs_times, preds):
        rho = np.asarray(pr.rho)
        rho /= np.linalg.norm(rho)
        ra = np.arctan2(rho[1], rho[0])
        dec = np.arcsin(np.clip(rho[2], -1.0, 1.0))
        synth.append(
            Observation.from_astrometry(
                ra=float(ra), dec=float(dec), epoch=float(t),
                observer_position=observer_position,
                observer_velocity=observer_velocity,
            )
        )
    return synth


@pytestmark_synthetic
@pytest.mark.parametrize(
    "name, state, arc_days, nobs, pos_tol_AU",
    [
        # Distant TNO -- the regime where BK linear-IOD shines.  With a
        # barycenter observer (no parallax) we expect tight recovery.
        ("tno_40au_arc_30d", [40.0, 0.0, 5.0, 0.0, 0.00125, 0.0], 30.0, 12, 0.5),
        # Centaur, longer arc.
        ("centaur_15au_arc_30d", [15.0, 0.0, 0.0, 0.0, 0.0042, 0.00038], 30.0, 12, 0.5),
    ],
)
def test_bk_iod_recovers_distant_orbit(name, state, arc_days, nobs, pos_tol_AU):
    """For a distant orbit observed across a moderate arc, the linear
    IOD should recover the heliocentric position component-wise at the
    sub-AU level."""
    ephem = get_ephem(CACHE)
    truth_epoch = 2460000.5
    obs_times = np.linspace(truth_epoch - 0.5 * arc_days, truth_epoch + 0.5 * arc_days, nobs)
    obs = _generate_synthetic_observations(ephem, state, truth_epoch, obs_times)

    result = run_bk_iod(obs, truth_epoch, MU_SUN)
    assert result.flag == 0, f"[{name}] BK-IOD flag={result.flag}"
    np.testing.assert_allclose(
        np.asarray(result.state)[:3],
        np.asarray(state)[:3],
        atol=pos_tol_AU,
        err_msg=f"[{name}] BK-IOD position recovery failed",
    )


# ---------------------------------------------------------------------------
# Diagnostic-scan comparison
# ---------------------------------------------------------------------------


pytestmark_diagnostic = pytest.mark.skipif(
    not (EPHEM_AVAILABLE and DIAGNOSTIC_AVAILABLE),
    reason="Need both ephemeris and diagnostic scan data.",
)


def _load_diagnostic_case(name):
    with open(DIAGNOSTIC_SCAN / f"{name}.json") as f:
        return json.load(f)


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
        # Well-arced TNO: should recover at sub-1% of heliocentric distance.
        ("classical_42AU_arc_060.00d", 0.01),
        # Distant short-arc: looser tolerance, but BK-IOD should still
        # land in the right ballpark (within ~10% of helio distance).
        ("scattered_70AU_arc_014.00d", 0.10),
    ],
)
def test_bk_iod_on_diagnostic_scan(case_name, max_drift_frac):
    """Run BK-IOD on a diagnostic-scan case and check the recovered
    Cartesian position lands within `max_drift_frac` * r_helio of truth."""
    case = _load_diagnostic_case(case_name)
    obs = _build_observations_from_case(case)
    truth = np.asarray(case["truth_state_at_epoch"])
    epoch = float(case["epoch_jd_tdb"])
    r_helio = float(np.linalg.norm(truth[:3]))

    result = run_bk_iod(obs, epoch, MU_SUN)
    assert result.flag == 0, f"[{case_name}] BK-IOD did not converge (flag={result.flag})"
    drift = np.linalg.norm(np.asarray(result.state)[:3] - truth[:3])
    assert drift < max_drift_frac * r_helio, (
        f"[{case_name}] BK-IOD drifted {drift:.3f} AU "
        f"> {max_drift_frac:.0%} of r_helio={r_helio:.1f} AU"
    )
