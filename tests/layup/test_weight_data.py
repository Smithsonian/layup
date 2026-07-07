"""Regression tests for `orbitfit(weight_data=True)`.

Guards two bugs that previously broke the `weight_data=True` code path:

  1. `g_column_present` was defined but `astcat_column_present` was
     referenced inside _orbitfit, raising NameError at the first
     astrometric_uncertainty_Veres2017 call -- failure mode: hard crash before
     any fitting happened.

  2. `astrometric_uncertainty_Veres2017` returns astrometric uncertainty in
     ARCSECONDS (per its docstring), but the call site assigned the
     return value directly to Observation.ra_unc / dec_unc which are
     stored in RADIANS -- failure mode: weights were ~206000x too
     loose, fit covariance inflated to absurd values (sigma_pos
     ~25 million km on a 3 AU object), chi-square -> 0.

The fix is a one-line rename plus an explicit arcsec->radian
conversion at the call site; this file guards against either kind
of regression.
"""

from __future__ import annotations

import os
import pooch

import numpy as np
import pytest

from layup.orbitfit import orbitfit
from layup.utilities.data_processing_utilities import parse_cov
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader

CACHE = str(pooch.os_cache("layup"))
EPHEM_PLANETS = os.path.join(CACHE, "linux_p1550p2650.440")
EPHEM_SMALLBODIES = os.path.join(CACHE, "sb441-n16.bsp")
EPHEM_AVAILABLE = os.path.exists(EPHEM_PLANETS) and os.path.exists(EPHEM_SMALLBODIES)
AU_KM = 149597870.7

pytestmark = pytest.mark.skipif(
    not EPHEM_AVAILABLE,
    reason=f"ASSIST ephemeris missing at {CACHE}; skipping weight_data test.",
)


def _fit_one(prov_id: str, weight_data: bool):
    reader = CSVDataReader(
        get_test_filepath("4_random_mpc_ADES_provIDs_no_sats.csv"),
        "csv",
        primary_id_column_name="provID",
    )
    data = reader.read_rows()
    data = data[data["provID"] == prov_id]
    assert len(data) > 0, f"test fixture missing provID {prov_id}"
    return orbitfit(data, cache_dir=CACHE, weight_data=weight_data)[0]


def test_orbitfit_weight_data_true_runs_without_crashing():
    """orbitfit(weight_data=True) on real MPC data without an astCat column
    must run to completion -- previously raised NameError on the missing
    astcat_column_present variable."""
    row = _fit_one("119839", weight_data=True)
    assert "flag" in row.dtype.names


def _fit_supplied(prov_id, sigma_arcsec):
    """Fit with weight_data='supplied' and an explicit per-obs sigma (arcsec)."""
    import numpy.lib.recfunctions as rfn

    reader = CSVDataReader(
        get_test_filepath("4_random_mpc_ADES_provIDs_no_sats.csv"),
        "csv",
        primary_id_column_name="provID",
    )
    data = reader.read_rows()
    data = data[data["provID"] == prov_id]
    cols = data.dtype.names
    sig = np.full(len(data), float(sigma_arcsec))
    if "rmsRA" in cols:
        data = data.copy()
        data["rmsRA"] = sig
        data["rmsDec"] = sig
    else:
        data = rfn.append_fields(data, ["rmsRA", "rmsDec"], [sig, sig.copy()], usemask=False)
    return orbitfit(data, cache_dir=CACHE, weight_data="supplied")[0]


def test_orbitfit_weight_data_supplied_uses_the_columns():
    """weight_data='supplied' reads the per-obs rmsRA/rmsDec columns: a looser
    supplied sigma yields a proportionally looser position covariance (the weights
    are actually applied), and the fit still converges."""
    tight = _fit_supplied("119839", 0.2)
    loose = _fit_supplied("119839", 2.0)
    assert tight["flag"] == 0 and loose["flag"] == 0
    s_tight = float(np.sqrt(sum(parse_cov(tight)[i, i] for i in range(3))))
    s_loose = float(np.sqrt(sum(parse_cov(loose)[i, i] for i in range(3))))
    # 10x looser astrometry -> ~10x looser position sigma (linear least squares).
    assert 5.0 < s_loose / s_tight < 20.0, f"ratio {s_loose / s_tight:.1f} (weights not applied?)"


def test_orbitfit_weight_data_supplied_requires_columns():
    """weight_data='supplied' without rmsRA/rmsDec columns raises a clear error."""
    reader = CSVDataReader(
        get_test_filepath("4_random_mpc_ADES_provIDs_no_sats.csv"),
        "csv",
        primary_id_column_name="provID",
    )
    data = reader.read_rows()
    data = data[data["provID"] == "119839"]
    if "rmsRA" in data.dtype.names:
        import numpy.lib.recfunctions as rfn

        data = rfn.drop_fields(data, ["rmsRA", "rmsDec"])
    with pytest.raises(ValueError, match="rmsRA"):
        orbitfit(data, cache_dir=CACHE, weight_data="supplied")


def test_orbitfit_weight_data_true_gives_sensible_uncertainty():
    """orbitfit(weight_data=True) must apply weights in the right units.
    Previously the returned arcsecond value was assigned directly to
    Observation.ra_unc (which is in radians), inflating sigma_pos by
    ~5 orders of magnitude.  An explicit cap of 10000 km on the position
    uncertainty for our well-constrained 27-year mainbelt arc catches
    that regression: actual sigma_pos with the fix is O(100 km)."""
    row = _fit_one("119839", weight_data=True)
    assert row["flag"] == 0, f"weight_data=True fit did not converge (flag={row['flag']})"
    cov = parse_cov(row)
    sigma_pos_km = float(np.sqrt(cov[0, 0] + cov[1, 1] + cov[2, 2])) * AU_KM
    assert sigma_pos_km < 10000.0, (
        f"sigma_pos={sigma_pos_km:.1f} km is unreasonably large for a "
        f"587-obs 27-year arc; likely a units regression in the "
        f"weight assignment (arcsec vs radian)."
    )
