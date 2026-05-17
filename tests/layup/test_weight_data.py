"""Regression tests for `orbitfit(weight_data=True)`.

Guards two bugs that previously broke the `weight_data=True` code path:

  1. `g_column_present` was defined but `astcat_column_present` was
     referenced inside _orbitfit, raising NameError at the first
     data_weight_Veres2017 call -- failure mode: hard crash before
     any fitting happened.

  2. `data_weight_Veres2017` returns astrometric uncertainty in
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

import numpy as np
import pytest

from layup.orbitfit import orbitfit
from layup.utilities.data_processing_utilities import parse_cov
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader

CACHE = os.path.expanduser("~/Library/Caches/layup")
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
