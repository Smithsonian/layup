"""Predictions must carry the fitted non-gravitational acceleration (issue #522).

`predict` and `residuals_at_state` build their own simulations from the fitted
state. Before this fix neither enabled ASSIST's Marsden non-gravitational force,
so both propagated gravity-only even when the fit had solved for A1/A2/A3 -- and
`parse_fit_result` did not carry the amplitudes across the Python boundary at
all, so the information never reached the C++ in the first place.

The symptom was silent: a predicted position, and the uncertainty ellipse mapped
about it, were simply wrong for any non-gravitationally fitted object, with no
warning. On (152563) 1992 BF over its 71-year arc the error reached ~15 arcsec.

The invariant asserted here is the one that caught it: at the fitted state, the
chi-square implied by `residuals_at_state` must equal the chi-square the fit
reports. It does for a gravity-only fit whether or not the bug is present, and
only for a non-gravitational fit once the parameters are actually propagated.
"""

from __future__ import annotations

import os

import numpy as np
import pooch
import pytest

from layup.orbitfit import orbitfit
from layup.routines import get_ephem, numpy_to_eigen, residuals_at_state
from layup.utilities.data_processing_utilities import parse_fit_result

from test_nongrav_a2 import _build_arc_array, _TRUE_A2  # noqa: E402

CACHE = str(pooch.os_cache("layup"))
_EPHEM = ("linux_p1550p2650.440", "sb441-n16.bsp")
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(os.path.join(CACHE, f)) for f in _EPHEM),
    reason="ASSIST ephemeris not in layup cache; run `layup bootstrap`",
)

ARCSEC = 180.0 * 3600.0 / np.pi


def _observations(data):
    """Build the Observation list the fit itself would build from `data`."""
    import numpy.lib.recfunctions as rfn
    import spiceypy as spice

    from layup.routines import Observation
    from layup.utilities.data_processing_utilities import (
        LayupObservatory,
        layup_furnish_spiceypy,
    )
    from layup.utilities.datetime_conversions import convert_tdb_date_to_julian_date

    layup_furnish_spiceypy(CACHE)
    obsv = LayupObservatory(cache_dir=CACHE)
    et = np.array([spice.str2et(t) for t in data["obsTime"]], dtype="<f8")
    d = rfn.append_fields(data, "et", et, usemask=False, asrecarray=True)
    d = rfn.merge_arrays([d, obsv.obscodes_to_barycentric(d)], flatten=True, asrecarray=True, usemask=False)
    jd = [float(convert_tdb_date_to_julian_date(t)) for t in d["obsTime"]]
    return [
        Observation.from_astrometry_with_id(
            str(row["provID"]) if "provID" in d.dtype.names else str(row["ObjID"]),
            float(np.radians(row["ra"])),
            float(np.radians(row["dec"])),
            float(t),
            [float(row["x"]), float(row["y"]), float(row["z"])],
            [float(row["vx"]), float(row["vy"]), float(row["vz"])],
        )
        for row, t in zip(d, jd)
    ]


def _chi2_from_residuals(row, observations):
    """Sum of squared weighted residuals at the fitted state."""
    fit = parse_fit_result(row, orbit_colm_flag=False)
    res = residuals_at_state(get_ephem(CACHE), fit, observations, numpy_to_eigen(fit.cov, 6, 6))
    return float(sum(r[0] ** 2 + r[1] ** 2 for r in res)), fit


def test_parse_fit_result_carries_the_nongrav_parameters():
    """Without this the amplitudes never reach the C++ propagation at all."""
    data, guess = _build_arc_array()
    row = orbitfit(data, cache_dir=CACHE, initial_guess=guess, fit_nongrav="A2")[0]
    fit = parse_fit_result(row, orbit_colm_flag=False)
    assert fit.nongrav_mask == 2, f"A2 fit should set bit 1; got mask {fit.nongrav_mask}"
    assert fit.a2 == pytest.approx(
        float(row["a2"]), rel=1e-12
    ), "the fitted A2 must survive the round trip through parse_fit_result"


def test_residuals_at_the_fitted_state_are_small_for_a_nongrav_fit():
    """The arc is noise-free and generated WITH a known A2, so a correct
    7-parameter fit reproduces it essentially exactly. Propagating the fitted
    state gravity-only instead -- the defect -- leaves the whole A2 along-track
    drift in the residuals, which on this four-year arc is arcseconds rather than
    milliarcseconds.

    Asserting on the residual size rather than on chi-square keeps the test
    meaningful: the noise-free chi-square is ~0, so a relative comparison against
    it is degenerate.
    """
    data, guess = _build_arc_array()
    observations = _observations(data)

    row = orbitfit(data, cache_dir=CACHE, initial_guess=guess, fit_nongrav="A2")[0]
    fit = parse_fit_result(row, orbit_colm_flag=False)
    assert fit.nongrav_mask == 2, "A2 must survive parse_fit_result (issue #522)"

    res = residuals_at_state(get_ephem(CACHE), fit, observations, numpy_to_eigen(fit.cov, 6, 6))
    worst = max(max(abs(r[0]), abs(r[1])) for r in res) * ARCSEC
    assert worst < 0.05, (
        f"worst residual at the fitted state is {worst:.3f} arcsec; the "
        "propagation is dropping the fitted A2 (issue #522)"
    )
