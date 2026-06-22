"""Tests for `do_fit(iod='auto')`: Gauss first, falling back to the BK
5-parameter linear IOD if every Gauss root fails to seed the LM.

The empirical motivation is the sweep in `tools/bk_iod_sweep.py`,
which showed Gauss + BK-IOD fallback covers 90/98 cases on the
diagnostic-scan dataset vs Gauss alone (82) or BK-IOD alone (72).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import layup.orbitfit as of
from layup.orbitfit import do_fit
from layup.routines import Observation

from _bk_guards import (
    DIAGNOSTIC_AVAILABLE,
    EPHEM_AVAILABLE,
    EPHEM_CACHE,
    load_diagnostic_case,
)

# Directory passed to do_fit(cache_dir=...); str() preserves the pre-refactor type.
CACHE = str(EPHEM_CACHE)


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


_load = load_diagnostic_case


# ---------------------------------------------------------------------------
# Dispatch validation -- no ephemeris needed
# ---------------------------------------------------------------------------


def test_do_fit_unknown_iod_raises():
    """A typo'd iod value raises ValueError (via the IOD registry) before any
    fitting starts."""
    with pytest.raises(ValueError, match="register_iod"):
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
def test_iod_auto_handles_no_gauss_roots(monkeypatch):
    """Regression: when the Gauss IOD returns None (no real roots) rather than
    [], iod='auto' must normalize it and fall through to the BK fallback rather
    than raising 'NoneType is not iterable'/len(None) in the prefilter+picker.

    Reproduces a Linux-only crash (Gauss found roots on macOS but not Linux for
    the same input). We force the registered IOD to return None to make it
    deterministic on any platform."""
    case = _load("mainbelt_2.5AU_arc_030.00d")
    obs = _build_observations(case)
    seq = [list(range(len(obs)))]
    # do_fit('auto') pulls the Gauss IOD via the registry; force it to yield
    # nothing so the BK fallback path runs.
    monkeypatch.setattr(of, "get_iod", lambda name: (lambda *a, **k: None))

    # Must return a FitResult without raising, regardless of whether the BK
    # fallback ultimately converges on this (well-conditioned) geometry.
    res = do_fit(obs, seq, CACHE, iod="auto")
    assert hasattr(res, "flag")


# A degenerate distant short-arc geometry (80 AU, 1-day arc) where Gauss IOD
# fails but BK-IOD produces a good seed, so iod='auto' recovers.
_RECOVERY_CASE = "sednoid_80AU_arc_001.00d"


@pytestmark_diagnostic
@pytest.mark.skipif(
    not os.environ.get("LAYUP_SLOW_TESTS"),
    reason=(
        "The iod='auto' recovery fit runs a Levenberg-Marquardt fit on a "
        "degenerate distant orbit; a poor BK-IOD seed can drive the ASSIST "
        "integrator into a very long adaptive-step grind that is nondeterministic "
        "on Linux (observed: an ubuntu CI leg at ~80 min while its twin finished "
        "in ~12 min). Run it on demand with LAYUP_SLOW_TESTS=1; the underlying "
        "deep-integration grind is tracked for an ASSIST-side fix."
    ),
)
def test_iod_auto_recovers_when_gauss_fails():
    """The BK-IOD fallback's reason to exist: on a degenerate distant short arc
    where iod='gauss' fails to seed the LM, iod='auto' recovers via the BK-IOD
    seed.

    Deliberately bounded to a *single* well-characterized case rather than a
    sweep over many degenerate geometries. Because exactly which borderline
    geometry tips Gauss over flag!=0 is platform-dependent (libm/BLAS), we
    *skip* rather than fail if Gauss happens not to fail here."""
    case = _load(_RECOVERY_CASE)
    obs = _build_observations(case)
    seq = [list(range(len(obs)))]

    # Gauss-only fit is cheap (~0.1 s) and is the precondition for exercising
    # the fallback at all.
    if do_fit(obs, seq, CACHE, iod="gauss").flag == 0:
        pytest.skip(
            f"Gauss IOD did not fail on {_RECOVERY_CASE} on this platform; "
            "cannot exercise the BK-IOD fallback here."
        )

    # The one bounded expensive fit: iod='auto' must fall back to BK-IOD and
    # converge.
    res_auto = do_fit(obs, seq, CACHE, iod="auto")
    assert res_auto.flag == 0, (
        f"iod='auto' did not recover on {_RECOVERY_CASE}: " f"flag={res_auto.flag}, method={res_auto.method}"
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
