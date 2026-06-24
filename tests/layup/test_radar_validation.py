"""End-to-end validation of the radar (delay/Doppler) fit (issue #146).

Fits a synthetic, noise-free radar arc generated from a known orbit
(``tests/data/radar_synthetic.json``: the same ~2.6 AU MBA near opposition as the
streak fixture, round-trip delay and Doppler produced by a REBOUND/ASSIST
propagation with the C++ light-time convention) and checks that

* the radar residuals are correct -- the fit reaches ~0 chi-squared and recovers
  the truth from a perturbed seed (delay = 2 rho/c, doppler = 2 rho_hat.v_rel),
* the variable-row packing is right -- ndof = (#delay rows + #doppler rows) - 6,
* the radar rows are actually live -- corrupting one observed value raises
  chi-squared, and
* delay-only / Doppler-only observations contribute exactly one row each.

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

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "radar_synthetic.json"
CACHE = str(pooch.os_cache("layup"))
_EPHEM_OK = all(os.path.exists(os.path.join(CACHE, f)) for f in ("linux_p1550p2650.440", "sb441-n16.bsp"))
pytestmark = pytest.mark.skipif(
    not _EPHEM_OK, reason="ASSIST ephemeris not in layup cache; run `layup bootstrap`"
)


def _load(has_delay=True, has_doppler=True):
    d = json.loads(FIXTURE.read_text())
    obs = []
    for o in d["observations"]:
        ob = Observation.from_radar_with_id(
            "synth",
            o["delay"],
            o["doppler"],
            has_delay,
            has_doppler,
            o["epoch"],
            o["observer_position"],
            o["observer_velocity"],
            d["delay_unc_days"],
            d["doppler_unc_audy"],
        )
        obs.append(ob)
    return d, obs


def _seed(state, epoch):
    g = FitResult()
    g.state = list(state)
    g.epoch = epoch
    g.flag = 0
    return g


def test_radar_fit_recovers_synthetic_orbit():
    """From a seed offset from truth, the radar fit converges back to the true
    orbit at ~0 chi-squared -- validating both the delay/Doppler residuals
    (csq -> 0 on noise-free data) and their partials (they drive LM to truth)."""
    d, obs = _load()
    truth = np.array(d["true_state"])
    # Seed 0.1% off in velocity so LM must converge using the partials.
    seed = truth.copy()
    seed[3:] *= 1.001
    res = run_from_vector_with_initial_guess(get_ephem(CACHE), _seed(seed, d["epoch"]), obs, 50)

    assert res.flag == 0
    assert res.ndof == 2 * len(obs) - 6  # delay + doppler row per obs
    assert res.csq < 1e-6  # noise-free data => essentially zero chi-squared
    st = np.array([res.state[i] for i in range(6)])
    assert np.linalg.norm(st[:3] - truth[:3]) < 1e-6
    assert np.linalg.norm(st[3:] - truth[3:]) < 1e-7


def test_radar_rows_are_live():
    """Corrupting one observed delay must raise chi-squared -- proof the radar
    residual rows actually enter the normal equations."""
    d, obs = _load()
    truth = np.array(d["true_state"])
    good = run_from_vector_with_initial_guess(get_ephem(CACHE), _seed(truth, d["epoch"]), obs, 50)
    assert good.flag == 0 and good.csq < 1e-6

    o = d["observations"][0]
    bad_obs = list(obs)
    bad0 = Observation.from_radar_with_id(
        "synth",
        o["delay"] + 50.0 * d["delay_unc_days"],  # 50-sigma corruption of one delay
        o["doppler"],
        True,
        True,
        o["epoch"],
        o["observer_position"],
        o["observer_velocity"],
        d["delay_unc_days"],
        d["doppler_unc_audy"],
    )
    bad_obs[0] = bad0
    bad = run_from_vector_with_initial_guess(get_ephem(CACHE), _seed(truth, d["epoch"]), bad_obs, 50)
    assert bad.csq > good.csq + 10.0  # the corrupted delay row dominates chi-squared


def test_delay_only_and_doppler_only_row_counts():
    """A delay-only or Doppler-only observation contributes exactly one row, so
    ndof reflects the variable-row packing."""
    d, delay_only = _load(has_delay=True, has_doppler=False)
    truth = np.array(d["true_state"])
    res = run_from_vector_with_initial_guess(get_ephem(CACHE), _seed(truth, d["epoch"]), delay_only, 50)
    assert res.ndof == 1 * len(delay_only) - 6  # one delay row each
    assert res.csq < 1e-6

    _, doppler_only = _load(has_delay=False, has_doppler=True)
    res2 = run_from_vector_with_initial_guess(get_ephem(CACHE), _seed(truth, d["epoch"]), doppler_only, 50)
    assert res2.ndof == 1 * len(doppler_only) - 6  # one doppler row each
