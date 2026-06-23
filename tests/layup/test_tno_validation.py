"""Accuracy validation of a TNO orbit fit (incl. a space-based observation).

Fits the classical trans-Neptunian object 2000 FV53 from a subset of the
Bernstein et al. (2004) HST survey astrometry
(`tests/data/bernstein_2004_kbos_data_with_sats_occ.csv`) and checks the
recovered barycentric state against a stored JPL Horizons reference at the
same epoch (`tests/data/jpl_reference_2000_FV53.json`).

This is the distant-orbit counterpart of the main-belt and interstellar
real-data validations. The fixture's 28 observations span ~19 years and
include one HST (space-based, obscode 250) observation, so the test also
exercises the moving-observatory barycentric path -- which previously
mis-placed space-based observers by a factor of the Earth radius and made any
such fit diverge (see
test_data_processing_utilities.test_moving_observatory_barycentric_position).

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

_REFERENCE_PATH = Path(__file__).parent.parent / "data" / "jpl_reference_2000_FV53.json"


def test_orbitfit_2000_FV53_matches_jpl_horizons():
    """Fit the TNO 2000 FV53 (ground + HST astrometry) and confirm the recovered
    barycentric state matches the JPL Horizons reference, both in absolute
    relative agreement and relative to the fit's own covariance."""
    reference = json.loads(_REFERENCE_PATH.read_text())
    ref_epoch = reference["epoch_jd_tdb"]
    ref_state = np.asarray(reference["state_au_au_per_day"])

    data = CSVDataReader(
        get_test_filepath("bernstein_2004_kbos_data_with_sats_occ.csv"),
        "csv",
        primary_id_column_name="provID",
    ).read_rows()
    data = data[data["provID"] == "2000 FV53"]
    assert len(data) == 28, f"expected 28 observations for 2000 FV53, got {len(data)}"
    # Sanity: the fixture really does include the space-based (HST) observation.
    assert "250" in set(data["stn"]), "expected an HST (obscode 250) observation in the fixture"

    fit = orbitfit(data, cache_dir=CACHE)
    assert len(fit) == 1, f"expected one fitted orbit, got {len(fit)}"
    row = fit[0]
    assert row["flag"] == 0, f"orbitfit did not converge (flag={row['flag']})"

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

    # Absolute accuracy: with a ~19-year arc the recovered state matches JPL to
    # ~2e-5 in both position and velocity; the bounds sit a few times above that.
    pos_rel = np.linalg.norm(fit_state[:3] - ref_state[:3]) / np.linalg.norm(ref_state[:3])
    vel_rel = np.linalg.norm(fit_state[3:] - ref_state[3:]) / np.linalg.norm(ref_state[3:])
    assert pos_rel < 1e-4, f"relative position drift {pos_rel:.2e} exceeds 1e-4"
    assert vel_rel < 1e-4, f"relative velocity drift {vel_rel:.2e} exceeds 1e-4"

    # And the disagreement is consistent with the fit's own covariance: the
    # 6-parameter Mahalanobis distance should be ~sqrt(6) ~ 2.4.
    cov = np.array([[row[f"cov_{i}_{j}"] for j in range(6)] for i in range(6)])
    delta = fit_state - ref_state
    mahalanobis = float(np.sqrt(delta @ np.linalg.inv(cov) @ delta))
    assert mahalanobis < 4.0, (
        f"6D Mahalanobis distance {mahalanobis:.2f} -- the fit disagrees with JPL "
        "by more than its covariance allows"
    )
