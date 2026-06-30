"""End-to-end validation against real MPC astrometry + JPL ground truth.

These tests run `orbitfit` on the real-MPC-data fixture
`tests/data/4_random_mpc_ADES_provIDs_no_sats.csv` and assert that
the recovered Cartesian state matches the corresponding JPL Horizons
reference state for each object.  This is the most realistic check
in the suite -- everything else uses synthetic observations.

The JPL reference states live in `tests/data/jpl_reference_states.json`
and were captured once on 2026-05-17 (see the _comment_ field there for
regeneration instructions).  As long as the input MPC observations and
layup's epoch-selection logic don't change, the recovered states should
remain stable.

Empirically, layup recovers each tested object's barycentric position
to a relative agreement of ~4e-7 (and velocity to ~6e-7) against JPL --
i.e. ~20-200 km out of an ~3 x 10^8 km distance.  The tests assert a
*relative* tolerance (drift normalised by the reference magnitude), set
a few times above the observed agreement so a benign numerical drift
doesn't flag a regression.
"""

from __future__ import annotations

import json
import os
import pooch
from pathlib import Path

import numpy as np
import pytest

from layup.orbitfit import orbitfit
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader

CACHE = str(pooch.os_cache("layup"))
EPHEM_PLANETS = os.path.join(CACHE, "linux_p1550p2650.440")
EPHEM_SMALLBODIES = os.path.join(CACHE, "sb441-n16.bsp")
EPHEM_AVAILABLE = os.path.exists(EPHEM_PLANETS) and os.path.exists(EPHEM_SMALLBODIES)

pytestmark = pytest.mark.skipif(
    not EPHEM_AVAILABLE,
    reason=f"ASSIST ephemeris missing at {CACHE}; skipping real-data validation.",
)


_REFERENCE_PATH = Path(__file__).parent.parent / "data" / "jpl_reference_states.json"


def _load_jpl_references():
    """Read tests/data/jpl_reference_states.json."""
    with open(_REFERENCE_PATH) as f:
        return json.load(f)["objects"]


def _orbitfit_one(prov_id: str):
    """Run orbitfit on a single provID from the 4-object real-MPC fixture."""
    reader = CSVDataReader(
        get_test_filepath("4_random_mpc_ADES_provIDs_no_sats.csv"),
        "csv",
        primary_id_column_name="provID",
    )
    data = reader.read_rows()
    data = data[data["provID"] == prov_id]
    if len(data) == 0:
        pytest.skip(f"provID {prov_id} not found in test fixture")
    return orbitfit(data, cache_dir=CACHE)


# Per-object relative tolerances (drift / |reference|).  All tested objects
# are mainbelt at 2-3 AU; layup recovers each to ~4e-7 in relative position
# and ~6e-7 in relative velocity against JPL.  Tolerances are a few times
# above the observed agreement so a benign numerical drift doesn't flag a
# regression, while staying ~5x tighter than the previous absolute bound.
@pytest.mark.parametrize(
    "prov_id, pos_rtol, vel_rtol",
    [
        # 27-year arc, 587 obs -- best constrained.
        ("119839", 1e-6, 2e-6),
        # 117 obs, shorter arc.
        ("742428", 1e-6, 2e-6),
        # 109 obs, shorter arc.
        ("609631", 1e-6, 2e-6),
    ],
)
def test_orbitfit_matches_jpl_on_real_mpc_data(prov_id, pos_rtol, vel_rtol):
    """Orbitfit on real MPC astrometry recovers the JPL reference state
    at the same epoch within tight tolerances."""
    references = _load_jpl_references()
    assert prov_id in references, f"missing JPL reference for {prov_id}"
    ref = references[prov_id]

    fit = _orbitfit_one(prov_id)
    assert len(fit) == 1, f"expected one row for {prov_id}, got {len(fit)}"
    row = fit[0]

    assert row["flag"] == 0, f"[{prov_id}] orbitfit did not converge (flag={row['flag']})"

    # Layup output: BCART_EQ (default).  Compare in the same frame as JPL.
    fit_epoch_jd = row["epochMJD_TDB"] + 2400000.5
    fit_state = np.array([row["x"], row["y"], row["z"], row["xdot"], row["ydot"], row["zdot"]])

    # Epoch sanity: the fixture's epoch should match what layup picks for
    # this object's observation set, otherwise we can't directly compare
    # state vectors.
    np.testing.assert_allclose(
        fit_epoch_jd,
        ref["epoch_jd_tdb"],
        atol=1e-9,
        err_msg=(
            f"[{prov_id}] layup chose epoch {fit_epoch_jd} but the JPL fixture "
            f"is at {ref['epoch_jd_tdb']}.  Has the input data changed, or "
            f"layup's epoch-selection logic?  The fixture needs regeneration."
        ),
    )

    ref_state = np.asarray(ref["state_au_au_per_day"])
    # Relative drift: position/velocity error normalised by the reference
    # magnitude (cf. Hanno's PR review -- the meaningful quantity is the
    # fractional agreement, not an absolute AU bound).
    pos_rel = np.linalg.norm(fit_state[:3] - ref_state[:3]) / np.linalg.norm(ref_state[:3])
    vel_rel = np.linalg.norm(fit_state[3:] - ref_state[3:]) / np.linalg.norm(ref_state[3:])
    assert pos_rel < pos_rtol, f"[{prov_id}] relative position drift {pos_rel:.2e} > tolerance {pos_rtol:.0e}"
    assert vel_rel < vel_rtol, f"[{prov_id}] relative velocity drift {vel_rel:.2e} > tolerance {vel_rtol:.0e}"
