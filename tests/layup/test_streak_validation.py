"""End-to-end validation of the streak (sky-motion-rate) fit.

Fits a synthetic, noise-free streak arc generated from a known orbit
(``tests/data/streak_synthetic.json``: a ~2.6 AU MBA near opposition, positions
and great-circle rates produced by a REBOUND/ASSIST propagation with light-time)
and checks that

* the rate residuals are correct -- the fit reaches ~0 chi-squared and recovers
  the truth from a perturbed seed, and
* the rate rows are actually live -- corrupting one observed rate raises
  chi-squared.

The fit propagates with ASSIST, so the suite skips when the ephemeris is not in
the layup cache (same dependency as the other orbit-fit tests).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pooch
import pytest

from layup.routines import FitResult, Observation, get_ephem, run_from_vector_with_initial_guess

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "streak_synthetic.json"
CACHE = str(pooch.os_cache("layup"))
_EPHEM_OK = all(
    os.path.exists(os.path.join(CACHE, f)) for f in ("linux_p1550p2650.440", "sb441-n16.bsp")
)
pytestmark = pytest.mark.skipif(
    not _EPHEM_OK, reason="ASSIST ephemeris not in layup cache; run `layup bootstrap`"
)


def _load():
    d = json.loads(FIXTURE.read_text())
    obs = []
    for o in d["observations"]:
        ob = Observation.from_streak_with_id(
            "synth", o["ra"], o["dec"], o["ra_rate"], o["dec_rate"], o["epoch"],
            o["observer_position"], o["observer_velocity"],
            d["rate_unc_radday"], d["rate_unc_radday"],
        )
        ob.ra_unc = d["ra_unc_rad"]
        ob.dec_unc = d["ra_unc_rad"]
        obs.append(ob)
    return d, obs


def _seed(state, epoch):
    g = FitResult()
    g.state = list(state)
    g.epoch = epoch
    g.flag = 0
    return g


def test_streak_fit_recovers_synthetic_orbit():
    """From a seed offset from truth, the streak fit converges back to the true
    orbit at ~0 chi-squared -- validating both the rate residual (csq -> 0 on
    noise-free data) and the rate partials (they drive LM to the truth)."""
    d, obs = _load()
    truth = np.array(d["true_state"])
    # Seed 0.1% off in velocity so LM must converge using the partials.
    seed = truth.copy()
    seed[3:] *= 1.001
    res = run_from_vector_with_initial_guess(get_ephem(CACHE), _seed(seed, d["epoch"]), obs, 50)

    assert res.flag == 0
    assert res.ndof == 4 * len(obs) - 6  # 4 residual rows per streak obs (2 pos + 2 rate)
    assert res.csq < 1e-6  # noise-free data => essentially zero chi-squared
    st = np.array([res.state[i] for i in range(6)])
    assert np.linalg.norm(st[:3] - truth[:3]) < 1e-6
    assert np.linalg.norm(st[3:] - truth[3:]) < 1e-7


def test_streak_rate_rows_are_live():
    """Corrupting one observed rate must raise chi-squared -- proof the rate
    residual rows actually enter the normal equations (regression against the
    rates being silently ignored, the original #140 bug)."""
    d, obs = _load()
    truth = np.array(d["true_state"])
    good = run_from_vector_with_initial_guess(get_ephem(CACHE), _seed(truth, d["epoch"]), obs, 50)
    assert good.flag == 0 and good.csq < 1e-6

    o = d["observations"][0]
    bad_obs = list(obs)
    bad0 = Observation.from_streak_with_id(
        "synth", o["ra"], o["dec"],
        o["ra_rate"] + 50.0 * d["rate_unc_radday"],  # 50-sigma corruption of one ra rate
        o["dec_rate"], o["epoch"], o["observer_position"], o["observer_velocity"],
        d["rate_unc_radday"], d["rate_unc_radday"],
    )
    bad0.ra_unc = d["ra_unc_rad"]
    bad0.dec_unc = d["ra_unc_rad"]
    bad_obs[0] = bad0
    bad = run_from_vector_with_initial_guess(get_ephem(CACHE), _seed(truth, d["epoch"]), bad_obs, 50)
    assert bad.csq > good.csq + 10.0  # the corrupted rate row dominates chi-squared
