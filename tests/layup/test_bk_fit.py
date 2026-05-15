"""Layer 2 tests for the universal BK fitter (`run_bk_native_fit`).

These tests cover the LM driver itself.  They reuse the same Gauss IOD +
observation setup as the existing Cartesian fit so the only difference
between the two engines is the parameterization + the energy prior on
gdot, isolating any disagreement to the BK-specific code path.

Tests skip when the ASSIST ephemeris files aren't available, so CI on
machines without `~/Library/Caches/layup/{linux_p1550p2650.440,
sb441-n16.bsp}` is unaffected.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from layup.routines import (
    FitResult,
    Observation,
    get_ephem,
    run_bk_native_fit,
)

CACHE = os.path.expanduser("~/Library/Caches/layup")
EPHEM_PLANETS = os.path.join(CACHE, "linux_p1550p2650.440")
EPHEM_SMALLBODIES = os.path.join(CACHE, "sb441-n16.bsp")
EPHEM_AVAILABLE = os.path.exists(EPHEM_PLANETS) and os.path.exists(EPHEM_SMALLBODIES)

# GM_sun in AU^3 / day^2.
MU_SUN = 0.00029591220828559104

pytestmark = pytest.mark.skipif(
    not EPHEM_AVAILABLE,
    reason=f"ASSIST ephemeris missing at {CACHE}; skipping BK-fit Layer 2 tests.",
)


# ---------------------------------------------------------------------------
# Tests that don't need real observations -- exercise the API + early-exit
# guards.
# ---------------------------------------------------------------------------


def test_run_bk_native_fit_returns_fitresult_for_empty_obs():
    """With zero observations, the fit returns a FitResult with flag != 0 and
    does not crash."""
    ephem = get_ephem(CACHE)
    ig = FitResult()
    ig.state = [40.0, 10.0, 5.0, -8e-4, 9e-4, 1e-4]
    ig.epoch = 2460000.5
    result = run_bk_native_fit(ephem, ig, [], MU_SUN)
    assert result.method == "bk_native"
    assert result.flag != 0


def test_run_bk_native_fit_returns_fitresult_for_too_few_obs():
    """With <3 observations the early-exit guard fires; no crash, flag != 0."""
    ephem = get_ephem(CACHE)
    ig = FitResult()
    ig.state = [40.0, 10.0, 5.0, -8e-4, 9e-4, 1e-4]
    ig.epoch = 2460000.5
    obs = [
        Observation.from_astrometry(
            ra=1.57,
            dec=0.1,
            epoch=2459995.5,
            observer_position=[-0.5, 0.8, 0.0],
            observer_velocity=[-0.018, -0.009, 0.0],
        ),
        Observation.from_astrometry(
            ra=1.57,
            dec=0.1,
            epoch=2460005.5,
            observer_position=[-0.5, 0.8, 0.0],
            observer_velocity=[-0.018, -0.009, 0.0],
        ),
    ]
    result = run_bk_native_fit(ephem, ig, obs, MU_SUN)
    assert result.method == "bk_native"
    assert result.flag != 0
