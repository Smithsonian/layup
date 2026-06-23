"""Accuracy validation of an interstellar (hyperbolic) orbit fit against JPL.

Fits 3I/ATLAS (designation A11pl3Z, JPL C/2025 N1) from a curated subset of
its real MPC discovery-arc astrometry (`tests/data/3I_ATLAS_ades.csv`) and
checks the recovered barycentric state against a JPL Horizons reference state
captured at the same epoch (`tests/data/jpl_reference_3I_ATLAS.json`).

This is the interstellar/hyperbolic counterpart of the main-belt real-data
validation: 3I/ATLAS has e ~ 6.5, so it exercises orbit recovery in a regime
very different from bound asteroids. The discovery arc is short (~19 days), so
the absolute agreement with JPL's refined orbit is at the ~1% level -- but the
disagreement is consistent with the fit's *own* covariance (the 6-parameter
Mahalanobis distance is ~sqrt(6)), i.e. layup recovers the orbit to within the
uncertainty it reports.

The Horizons reference is stored, so the test is offline and deterministic;
see the `_comment` field in the JSON for the regeneration recipe.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from layup.orbitfit import orbitfit
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader

CACHE = os.path.expanduser("~/Library/Caches/layup")
EPHEM_PLANETS = os.path.join(CACHE, "linux_p1550p2650.440")
EPHEM_SMALLBODIES = os.path.join(CACHE, "sb441-n16.bsp")
EPHEM_AVAILABLE = os.path.exists(EPHEM_PLANETS) and os.path.exists(EPHEM_SMALLBODIES)

pytestmark = pytest.mark.skipif(
    not EPHEM_AVAILABLE,
    reason=f"ASSIST ephemeris missing at {CACHE}; skipping JPL validation.",
)

_REFERENCE_PATH = Path(__file__).parent.parent / "data" / "jpl_reference_3I_ATLAS.json"


def test_orbitfit_3I_ATLAS_matches_jpl_horizons():
    """Fit 3I/ATLAS from its real discovery-arc astrometry and confirm the
    recovered barycentric state matches the JPL Horizons reference, both in
    absolute relative agreement and relative to the fit's own covariance."""
    reference = json.loads(_REFERENCE_PATH.read_text())
    ref_epoch = reference["epoch_jd_tdb"]
    ref_state = np.asarray(reference["state_au_au_per_day"])

    obs = CSVDataReader(
        get_test_filepath("3I_ATLAS_ades.csv"),
        "csv",
        primary_id_column_name="provID",
    ).read_rows()

    fit = orbitfit(obs, cache_dir=CACHE)
    assert len(fit) == 1, f"expected one fitted orbit, got {len(fit)}"
    row = fit[0]
    assert row["flag"] == 0, f"orbitfit did not converge (flag={row['flag']})"

    # Layup output is BCART_EQ -- the same barycentric equatorial frame as the
    # stored Horizons reference. The epoch must match for a direct state
    # comparison (regenerate the reference if it drifts).
    fit_epoch = row["epochMJD_TDB"] + 2400000.5
    np.testing.assert_allclose(
        fit_epoch,
        ref_epoch,
        atol=1e-6,
        err_msg=(
            f"layup chose epoch {fit_epoch} but the JPL reference is at {ref_epoch}; "
            "the fixture or layup's epoch-selection logic changed -- regenerate the reference."
        ),
    )

    fit_state = np.array([row["x"], row["y"], row["z"], row["xdot"], row["ydot"], row["zdot"]])

    # Absolute accuracy: from a ~19-day hyperbolic discovery arc, layup recovers
    # JPL's refined orbit to ~1% in position and ~1.5% in velocity. The bounds
    # sit a couple of times above the measured agreement.
    pos_rel = np.linalg.norm(fit_state[:3] - ref_state[:3]) / np.linalg.norm(ref_state[:3])
    vel_rel = np.linalg.norm(fit_state[3:] - ref_state[3:]) / np.linalg.norm(ref_state[3:])
    assert pos_rel < 2e-2, f"relative position drift {pos_rel:.2e} exceeds 2e-2"
    assert vel_rel < 3e-2, f"relative velocity drift {vel_rel:.2e} exceeds 3e-2"

    # Stronger statement: the disagreement is consistent with the fit's reported
    # uncertainty. The 6-parameter Mahalanobis distance between the fit and the
    # JPL state should be ~sqrt(6) ~ 2.4 for a well-calibrated covariance.
    cov = np.array([[row[f"cov_{i}_{j}"] for j in range(6)] for i in range(6)])
    delta = fit_state - ref_state
    mahalanobis = float(np.sqrt(delta @ np.linalg.inv(cov) @ delta))
    assert mahalanobis < 4.0, (
        f"6D Mahalanobis distance {mahalanobis:.2f} -- the fit disagrees with JPL "
        "by more than its covariance allows"
    )
