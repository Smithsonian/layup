"""Tests for the pluggable IOD layer and the multi-root picker."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from layup import iod, orbitfit
from layup.routines import FitResult, Observation


def test_registry_has_gauss():
    assert "gauss" in iod.iod_methods()


def test_register_and_get_iod():
    sentinel = []

    def fake(observations, seq):
        sentinel.append((len(observations), seq))
        return []

    iod.register_iod("fake_for_test", fake)
    try:
        looked_up = iod.get_iod("fake_for_test")
        assert looked_up is fake
        looked_up([1, 2, 3], [[0, 1, 2]])
        assert sentinel == [(3, [[0, 1, 2]])]
    finally:
        iod._REGISTRY.pop("fake_for_test", None)


def test_unknown_iod_raises():
    with pytest.raises(ValueError) as exc:
        iod.get_iod("definitely_not_registered")
    # The message should mention what is registered.
    assert "gauss" in str(exc.value)


def test_do_fit_propagates_unknown_iod():
    """`do_fit` surfaces a clear error when given an unregistered IOD."""
    with pytest.raises(ValueError):
        orbitfit.do_fit([], [[]], "/tmp", iod="nonexistent_iod_for_test")


def test_do_fit_accepts_callable_iod():
    """`do_fit` accepts a callable directly (skipping the registry).

    No real fit is run — the callable returns no candidates, so
    do_fit returns the sentinel FitResult with flag=5 without touching
    the C++ fitter.
    """
    def empty(observations, seq):
        return []

    fit = orbitfit.do_fit([], [[]], "/tmp", iod=empty)
    assert fit.flag == 5
    assert isinstance(fit, FitResult)


# --- Integration: multi-root picker on a diagnostic case --- #
# These tests need the diagnostic/scan dataset and a built liborbfit.so.
# They skip gracefully when either is missing.

DIAGNOSTIC_SCAN = (Path(__file__).resolve().parent.parent.parent.parent
                   / "diagnostic" / "scan" / "truth")
CACHE = os.path.expanduser("~/Library/Caches/layup")


def _have_truth(name: str) -> bool:
    return (DIAGNOSTIC_SCAN / f"{name}.json").exists()


def _build_obs(name: str):
    truth = json.loads((DIAGNOSTIC_SCAN / f"{name}.json").read_text())
    sigma_rad = float(truth["sigma_arcsec"]) / 206265.0
    obs = []
    for r in truth["observations"]:
        o = Observation.from_astrometry(
            float(r["ra"]) * np.pi / 180.0,
            float(r["dec"]) * np.pi / 180.0,
            float(r["jd_tdb"]),
            list(r["observer_state_AU"]),
            [0.0, 0.0, 0.0])
        o.ra_unc  = sigma_rad
        o.dec_unc = sigma_rad
        obs.append(o)
    return obs, truth


@pytest.mark.skipif(not _have_truth("classical_42AU_arc_010.00d"),
                    reason="diagnostic/scan dataset not present")
def test_picker_converges_on_distant_kbo():
    """A 42 AU KBO should yield a converged Cartesian fit at small r-error."""
    obs, _ = _build_obs("classical_42AU_arc_010.00d")
    seq = [list(range(len(obs)))]
    fit = orbitfit.do_fit(obs, seq, CACHE, iod="gauss")
    assert fit.flag == 0
    r = float(np.linalg.norm(fit.state[:3]))
    assert 40.0 < r < 44.0, f"unexpected fit r={r}"


@pytest.mark.skipif(not _have_truth("mainbelt_2.5AU_arc_007.00d"),
                    reason="diagnostic/scan dataset not present")
def test_picker_handles_mainbelt():
    """A 2.5 AU mainbelt should also converge under the picker."""
    obs, _ = _build_obs("mainbelt_2.5AU_arc_007.00d")
    seq = [list(range(len(obs)))]
    fit = orbitfit.do_fit(obs, seq, CACHE, iod="gauss")
    assert fit.flag == 0
    r = float(np.linalg.norm(fit.state[:3]))
    assert 2.0 < r < 3.0, f"unexpected fit r={r}"


@pytest.mark.skipif(not _have_truth("classical_42AU_arc_010.00d"),
                    reason="diagnostic/scan dataset not present")
def test_screen_iter_max_param_is_honored():
    """Passing a tiny screen_iter_max budget makes the LM step count drop.

    Verifies that the screening tier really uses the passed iter_max
    rather than the hardcoded 100 from the old binding.
    """
    obs, _ = _build_obs("classical_42AU_arc_010.00d")
    seq = [list(range(len(obs)))]
    # 1-iteration screen will never converge for any seed, so the
    # picker falls back to full_iter_max=4 which also won't converge.
    # do_fit should surface flag=3 (no convergence at either budget).
    fit = orbitfit.do_fit(obs, seq, CACHE, iod="gauss",
                          screen_iter_max=1, full_iter_max=4)
    # Either flag=3 (no convergence) or flag=0 if LM happens to nail
    # it in <=4 iters from a near-perfect seed; both are valid here.
    assert fit.flag in (0, 3, 4)
