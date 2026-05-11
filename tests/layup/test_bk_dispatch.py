"""Tests for orbitfit.do_fit's BK engine dispatch.

These tests are guarded by an importability check for `liborbfit.so`:
if the library isn't available, they skip rather than fail, so CI on
machines without the BK build is unaffected.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from layup import orbitfit
from layup.routines import Observation


# liborbfit.so may live in the repo root via create_lib_links.sh, or be
# pointed at by LIBORBFIT_PATH. If neither is present, skip these tests.
def _liborbfit_available() -> bool:
    candidates = []
    if "LIBORBFIT_PATH" in os.environ:
        candidates.append(Path(os.environ["LIBORBFIT_PATH"]))
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    candidates.append(repo_root / "liborbfit.so")
    candidates.append(repo_root / "src" / "liborbfit.so")
    return any(c.exists() for c in candidates)


pytestmark = pytest.mark.skipif(
    not _liborbfit_available(),
    reason="liborbfit.so not available (run create_lib_links.sh)")


# Truth-data fixtures live in claude_layup/diagnostic/scan/. The
# pre-bundled tests/data tree doesn't carry them. Tests skip gracefully
# if the diagnostic harness isn't present.
DIAGNOSTIC_SCAN = (Path(__file__).resolve().parent.parent.parent.parent
                   / "diagnostic" / "scan" / "truth")


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
    return obs


CACHE = os.path.expanduser("~/Library/Caches/layup")


@pytest.mark.skipif(not _have_truth("classical_42AU_arc_010.00d"),
                    reason="diagnostic/scan dataset not present")
def test_auto_routes_to_bk_for_distant():
    """A 42 AU classical KBO should auto-dispatch to BK."""
    obs = _build_obs("classical_42AU_arc_010.00d")
    seq = [list(range(len(obs)))]
    fit = orbitfit.do_fit(obs, seq, CACHE, iod="gauss", engine="auto")
    assert fit.flag == 0
    assert fit.method == "BK"


@pytest.mark.skipif(not _have_truth("mainbelt_2.5AU_arc_007.00d"),
                    reason="diagnostic/scan dataset not present")
def test_auto_routes_to_cartesian_for_mainbelt():
    """A 2.5 AU mainbelt object should auto-dispatch to Cartesian-LM."""
    obs = _build_obs("mainbelt_2.5AU_arc_007.00d")
    seq = [list(range(len(obs)))]
    fit = orbitfit.do_fit(obs, seq, CACHE, iod="gauss", engine="auto")
    assert fit.flag == 0
    assert fit.method != "BK"


@pytest.mark.skipif(not _have_truth("classical_42AU_arc_010.00d"),
                    reason="diagnostic/scan dataset not present")
def test_explicit_engine_bk():
    """Forcing engine='bk' produces a BK result regardless of r."""
    obs = _build_obs("classical_42AU_arc_010.00d")
    seq = [list(range(len(obs)))]
    fit = orbitfit.do_fit(obs, seq, CACHE, iod="gauss", engine="bk")
    assert fit.flag == 0
    assert fit.method == "BK"


@pytest.mark.skipif(not _have_truth("classical_42AU_arc_010.00d"),
                    reason="diagnostic/scan dataset not present")
def test_explicit_engine_cartesian():
    """engine='cartesian' bypasses the dispatch and uses the original LM path."""
    obs = _build_obs("classical_42AU_arc_010.00d")
    seq = [list(range(len(obs)))]
    fit = orbitfit.do_fit(obs, seq, CACHE, iod="gauss", engine="cartesian")
    assert fit.flag == 0
    assert fit.method != "BK"


@pytest.mark.skipif(not _have_truth("classical_42AU_arc_010.00d"),
                    reason="diagnostic/scan dataset not present")
def test_threshold_pushes_to_cartesian():
    """A very high threshold forces auto to pick Cartesian even for distant targets."""
    obs = _build_obs("classical_42AU_arc_010.00d")
    seq = [list(range(len(obs)))]
    fit = orbitfit.do_fit(obs, seq, CACHE, iod="gauss",
                          engine="auto", bk_threshold_AU=1000.0)
    assert fit.flag == 0
    assert fit.method != "BK"
