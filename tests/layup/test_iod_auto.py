"""Tests for `do_fit(iod='auto')`: Gauss first, falling back to the BK
5-parameter linear IOD if every Gauss root fails to seed the LM.

The empirical motivation is the sweep in `tools/bk_iod_sweep.py`,
which showed Gauss + BK-IOD fallback covers 90/98 cases on the
diagnostic-scan dataset vs Gauss alone (82) or BK-IOD alone (72).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from layup.orbitfit import do_fit
from layup.routines import Observation

CACHE = os.path.expanduser("~/Library/Caches/layup")
EPHEM_PLANETS = os.path.join(CACHE, "linux_p1550p2650.440")
EPHEM_SMALLBODIES = os.path.join(CACHE, "sb441-n16.bsp")
EPHEM_AVAILABLE = os.path.exists(EPHEM_PLANETS) and os.path.exists(EPHEM_SMALLBODIES)

DIAGNOSTIC_SCAN = Path("~/Dropbox/claude_layup/diagnostic/scan/truth").expanduser()
DIAGNOSTIC_AVAILABLE = DIAGNOSTIC_SCAN.is_dir()


def _build_observations(case_dict):
    sigma_arcsec = float(case_dict["sigma_arcsec"])
    sigma_rad = sigma_arcsec * np.pi / (180.0 * 3600.0)
    obs_list = []
    for o in case_dict["observations"]:
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


def _load(name):
    return json.load(open(DIAGNOSTIC_SCAN / f"{name}.json"))


# ---------------------------------------------------------------------------
# Dispatch validation -- no ephemeris needed
# ---------------------------------------------------------------------------


def test_do_fit_unknown_iod_raises():
    """A typo'd iod value raises ValueError before any fitting starts."""
    with pytest.raises(ValueError, match="not supported"):
        do_fit(observations=[], seq=[[]], cache_dir=CACHE, iod="not_a_real_iod")


# ---------------------------------------------------------------------------
# Fallback behavior -- needs ephemeris + diagnostic data
# ---------------------------------------------------------------------------


pytestmark_diagnostic = pytest.mark.skipif(
    not (EPHEM_AVAILABLE and DIAGNOSTIC_AVAILABLE),
    reason="Need both ephemeris and diagnostic scan data.",
)


@pytestmark_diagnostic
def test_iod_auto_matches_gauss_on_easy_case():
    """On a well-conditioned mainbelt case, iod='auto' should produce the
    same result as iod='gauss' (the fallback path never fires)."""
    case = _load("mainbelt_2.5AU_arc_030.00d")
    obs = _build_observations(case)
    seq = [list(range(len(obs)))]

    res_gauss = do_fit(obs, seq, CACHE, iod="gauss")
    res_auto = do_fit(obs, seq, CACHE, iod="auto")
    assert res_gauss.flag == 0
    assert res_auto.flag == 0
    np.testing.assert_allclose(np.asarray(res_auto.state), np.asarray(res_gauss.state), rtol=1e-9, atol=1e-12)


@pytestmark_diagnostic
def test_iod_auto_recovers_when_gauss_fails():
    """On a case where every Gauss root fails to seed the LM, iod='auto'
    falls back to BK-IOD and produces a converged fit.

    The case `sednoid_80AU_arc_001.00d` is one we know `do_fit` with
    iod='gauss' fails on (flag=3, 'primary interval failed') under its
    default Cartesian LM -- the 80 AU object with a 1-day arc has a
    degenerate Gauss geometry.  iod='auto' picks up the BK-IOD fallback
    and recovers (flag=0)."""
    case = _load("sednoid_80AU_arc_001.00d")
    obs = _build_observations(case)
    seq = [list(range(len(obs)))]

    res_gauss = do_fit(obs, seq, CACHE, iod="gauss")
    res_auto = do_fit(obs, seq, CACHE, iod="auto")

    # The gauss path fails (flag != 0).
    assert res_gauss.flag != 0, (
        f"Test premise broken: this case was supposed to be 'gauss fails'."
        f"  Got gauss flag={res_gauss.flag}.  Maybe pick a different case."
    )
    # The auto path recovers via BK-IOD fallback.
    assert res_auto.flag == 0, (
        f"iod='auto' did not recover: flag={res_auto.flag}, " f"method={res_auto.method}"
    )


@pytestmark_diagnostic
def test_iod_auto_engine_choice_propagates():
    """The engine parameter is independent of iod choice -- iod='auto' with
    engine='bk_native' should work just like the Cartesian engine."""
    case = _load("classical_42AU_arc_010.00d")
    obs = _build_observations(case)
    seq = [list(range(len(obs)))]

    res_cart = do_fit(obs, seq, CACHE, iod="auto", engine="cartesian")
    res_bk = do_fit(obs, seq, CACHE, iod="auto", engine="bk_native")
    assert res_cart.flag == 0
    assert res_bk.flag == 0
