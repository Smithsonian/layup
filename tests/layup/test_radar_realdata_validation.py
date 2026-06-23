"""Real-data validation of the radar (delay/Doppler) fit on (99942) Apophis (#146).

Fits the 2013 apparition of real JPL radar astrometry for Apophis (monostatic
delay + Doppler from Goldstone DSS-14 and Arecibo,
``tests/data/apophis_2013_radar.json``) and checks the recovered barycentric
state against a stored JPL Horizons reference at the fit epoch. This is the radar
counterpart of the optical real-data validations (test_tno_validation etc.): the
reference is stored, so the test is offline and deterministic; see the fixture's
``_comment`` for the regeneration recipe.

It exercises the full driver on real measurements -- ``orbitfit()`` computes the
station barycentric states (and the observer acceleration the two-leg radar model
needs) from the ISO ``obsTime``/``stn``, converts JPL units, and runs the
variable-row Cartesian fit. Radar over a short single-station arc is fit by
refining a prior orbit, so a (perturbed) Horizons state is supplied as the initial
guess, which bypasses IOD.

The recovered orbit matches Horizons to < 1e-6 (the chi-square sits well above 1
because the geometric model omits the ~2 us Shapiro relativistic delay -- the
documented next refinement -- but that bias barely moves the orbit). A regression
to the old single-position monostatic model would blow the agreement up by orders
of magnitude (real radar needs the two-leg round-trip station displacement).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from layup.orbitfit import _get_result_dtypes, orbitfit
from layup.utilities.data_processing_utilities import get_cov_columns

CACHE = os.path.expanduser("~/Library/Caches/layup")
_EPHEM_OK = all(os.path.exists(os.path.join(CACHE, f)) for f in ("linux_p1550p2650.440", "sb441-n16.bsp"))
pytestmark = pytest.mark.skipif(
    not _EPHEM_OK, reason="ASSIST ephemeris not in layup cache; run `layup bootstrap`"
)

_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "apophis_2013_radar.json"


def _build_inputs():
    fx = json.loads(_FIXTURE.read_text())
    ref_epoch = fx["reference"]["epoch_jd_tdb"]
    ref_state = np.asarray(fx["reference"]["state_au_au_per_day"], dtype=float)
    obs = fx["observations"]

    data = np.empty(
        len(obs),
        dtype=[
            ("provID", "U8"),
            ("obsTime", "U32"),
            ("stn", "U4"),
            ("delay", "f8"),
            ("rmsDelay", "f8"),
            ("doppler", "f8"),
            ("rmsDoppler", "f8"),
            ("freqTx", "f8"),
        ],
    )
    for i, o in enumerate(obs):
        data[i] = ("apophis", o["obsTime"], o["stn"], np.nan, np.nan, np.nan, np.nan, o["freqTx_hz"])
        if o["units"] == "us":
            data[i]["delay"] = o["value"]
            data[i]["rmsDelay"] = o["sigma"]
        else:
            data[i]["doppler"] = o["value"]
            data[i]["rmsDoppler"] = o["sigma"]

    # Prior orbit: Horizons reference, perturbed so the fit must converge using the
    # partials rather than starting on the answer. flag==0 -> orbitfit uses it
    # directly (bypassing IOD).
    guess = np.zeros(1, dtype=_get_result_dtypes("provID"))
    guess["provID"] = "apophis"
    pert = ref_state.copy()
    pert[:3] += 1.0e-4  # AU
    pert[3:] += 1.0e-6  # AU/day
    for k, v in zip(("x", "y", "z", "xdot", "ydot", "zdot"), pert):
        guess[k] = v
    guess["epochMJD_TDB"] = ref_epoch - 2400000.5
    guess["flag"] = 0
    guess["FORMAT"] = "BCART_EQ"
    guess["method"] = "seed"
    for c in get_cov_columns():
        guess[c] = 0.0

    return data, guess, ref_epoch, ref_state


def test_apophis_2013_radar_matches_jpl_horizons():
    data, guess, ref_epoch, ref_state = _build_inputs()
    assert len(data) == 36
    assert {"253", "251"} <= set(data["stn"]), "expected both Goldstone (253) and Arecibo (251)"

    fit = orbitfit(data, cache_dir=CACHE, initial_guess=guess, engine="cartesian")
    assert len(fit) == 1
    row = fit[0]
    assert row["flag"] == 0, f"radar fit did not converge (flag={row['flag']})"
    assert row["ndof"] == 36 - 6

    fit_epoch = row["epochMJD_TDB"] + 2400000.5
    np.testing.assert_allclose(fit_epoch, ref_epoch, atol=1e-6)

    fit_state = np.array([row["x"], row["y"], row["z"], row["xdot"], row["ydot"], row["zdot"]])
    pos_rel = np.linalg.norm(fit_state[:3] - ref_state[:3]) / np.linalg.norm(ref_state[:3])
    vel_rel = np.linalg.norm(fit_state[3:] - ref_state[3:]) / np.linalg.norm(ref_state[3:])
    # Two-leg model: recovered orbit agrees with JPL to ~1e-7. Bounds sit a few x
    # above that. The single-position model would miss by orders of magnitude.
    assert pos_rel < 1e-6, f"radar fit position drift {pos_rel:.2e} vs JPL Horizons"
    assert vel_rel < 1e-5, f"radar fit velocity drift {vel_rel:.2e} vs JPL Horizons"
