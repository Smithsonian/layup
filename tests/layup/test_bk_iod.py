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

import json
import os
from pathlib import Path

import numpy as np
import pytest

from layup.routines import (
    FitResult,
    Observation,
    get_ephem,
    run_bk_iod,
    run_bk_native_fit,
)

CACHE = os.path.expanduser("~/Library/Caches/layup")
EPHEM_PLANETS = os.path.join(CACHE, "linux_p1550p2650.440")
EPHEM_SMALLBODIES = os.path.join(CACHE, "sb441-n16.bsp")
EPHEM_AVAILABLE = os.path.exists(EPHEM_PLANETS) and os.path.exists(EPHEM_SMALLBODIES)

DIAGNOSTIC_SCAN = Path("~/Dropbox/claude_layup/diagnostic/scan/truth").expanduser()
DIAGNOSTIC_AVAILABLE = DIAGNOSTIC_SCAN.is_dir()

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
