"""End-to-end radar (delay/Doppler) fit through the ``orbitfit()`` driver (issue #146, P3).

Where ``test_radar_validation.py`` calls ``run_from_vector`` directly with observer
states baked into a fixture, this test exercises the *full* Python driver path:

    structured array of radar rows (delay [us], Doppler [Hz], freqTx [Hz], stn,
    ISO obsTime)
      -> orbitfit(): spice.str2et + obscodes_to_barycentric station states
      -> _radar_observation(): us->days, Hz->au/day unit conversion + dispatch
      -> variable-row Cartesian fit (run_from_vector_with_initial_guess)
      -> result structured array (state, csq, ndof, flag).

The truth delay/Doppler are generated in-test against the driver's *own*
``obscodes_to_barycentric`` station states and ``convert_tdb_date_to_julian_date``
epochs, so the observation can never drift from the observer model the fitter
actually uses -- exactly the consistency gap noted when the C++ and Python cores
were validated separately. The orbit is the same ~2.6 AU main-belt object near
opposition as the streak fixture (``tests/data/streak_synthetic.json``); the truth
observables are an independent ASSIST propagation at the C++ light-time convention
(``delay = 2 rho/c``; ``doppler = 2 rho_hat . v_rel``).

Radar over a short single-station arc weakly constrains the plane-of-sky
position, so -- as in real radar astrometry -- the fit *refines a prior orbit*:
``orbitfit()`` is given a perturbed initial guess, which bypasses IOD. With
noise-free observables the truth is a zero-residual fixed point, so a fit seeded
near it must return to it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as rfn
import pooch
import pytest
import spiceypy as spice

from layup.orbitfit import SPEED_OF_LIGHT, US_TO_DAYS, _get_result_dtypes, orbitfit
from layup.utilities.data_processing_utilities import LayupObservatory, get_cov_columns
from layup.utilities.datetime_conversions import convert_tdb_date_to_julian_date

CACHE = str(pooch.os_cache("layup"))
_EPHEM_OK = all(os.path.exists(os.path.join(CACHE, f)) for f in ("linux_p1550p2650.440", "sb441-n16.bsp"))
pytestmark = pytest.mark.skipif(
    not _EPHEM_OK, reason="ASSIST ephemeris not in layup cache; run `layup bootstrap`"
)

STREAK_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "streak_synthetic.json"

# Monostatic Arecibo (251), S-band transmit frequency.
_STN = "251"
_FREQ_TX = 2.38e9  # Hz
# Seven epochs spanning ~20 days, bracketing the true_state epoch (2021-12-06).
_OBS_TIMES = [
    "2021-12-03T00:00:00",
    "2021-12-06T08:00:00",
    "2021-12-09T16:00:00",
    "2021-12-13T00:00:00",
    "2021-12-16T08:00:00",
    "2021-12-19T16:00:00",
    "2021-12-23T00:00:00",
]


def _radar_observables(ephem, true_state, epoch, obs_jd_tdb, r_obs, v_obs, a_obs):
    """Two-leg round-trip delay (days) and Doppler (au/day).

    Mirrors the C++ orbit_fit.cpp radar model exactly so the noise-free fit
    reaches ~0 chi-squared: down leg to the retarded bounce time with the station
    at receive, up leg with the station Taylor-extrapolated (pos+vel) to the
    transmit time t - tau using the observer acceleration ``a_obs``.
    """
    import rebound

    r_obs = np.asarray(r_obs, dtype=float)
    v_obs = np.asarray(v_obs, dtype=float)
    a_obs = np.asarray(a_obs, dtype=float)
    jd_ref = ephem.jd_ref

    def state_at(t_jd):
        sim = rebound.Simulation()
        sim.t = epoch - jd_ref
        sim.add(
            x=true_state[0],
            y=true_state[1],
            z=true_state[2],
            vx=true_state[3],
            vy=true_state[4],
            vz=true_state[5],
        )
        import assist

        ax = assist.Extras(sim, ephem)
        sim.integrate(t_jd - jd_ref)
        p = sim.particles[0]
        r = np.array([p.x, p.y, p.z])
        v = np.array([p.vx, p.vy, p.vz])
        ax.detach(sim)
        return r, v

    tau_d = 0.0
    for _ in range(4):  # down leg, station at receive
        r_ast, v_ast = state_at(obs_jd_tdb - tau_d)
        rho_d_vec = r_ast - r_obs
        rho_d = float(np.linalg.norm(rho_d_vec))
        tau_d = rho_d / SPEED_OF_LIGHT
    rho_hat_d = rho_d_vec / rho_d
    tau_u = tau_d
    for _ in range(5):  # up leg, station Taylor-extrapolated to transmit
        tau = tau_d + tau_u
        r_tx = r_obs - v_obs * tau - 0.5 * a_obs * tau * tau
        rho_u_vec = r_ast - r_tx
        rho_u = float(np.linalg.norm(rho_u_vec))
        tau_u = rho_u / SPEED_OF_LIGHT
    rho_hat_u = rho_u_vec / rho_u
    v_tx = v_obs - a_obs * (tau_d + tau_u)
    delay = tau_d + tau_u
    doppler = float(rho_hat_d @ (v_ast - v_obs)) + float(rho_hat_u @ (v_ast - v_tx))
    return delay, doppler


def _build_radar_data(has_delay=True, has_doppler=True):
    """Construct a radar observation array (driver-units) whose truth is consistent
    with the driver's own observer states, plus a perturbed initial guess.

    Returns (data, initial_guess, true_state).
    """
    import assist

    d = json.loads(STREAK_FIXTURE.read_text())
    true_state = np.asarray(d["true_state"], dtype=float)
    epoch = d["epoch"]

    ephem = assist.Ephem(
        os.path.join(CACHE, "linux_p1550p2650.440"),
        os.path.join(CACHE, "sb441-n16.bsp"),
    )

    n = len(_OBS_TIMES)
    base = np.array(
        [("synth", t, _STN) for t in _OBS_TIMES],
        dtype=[("provID", "U8"), ("obsTime", "U32"), ("stn", "U4")],
    )

    # Replicate orbitfit()'s observer-state preamble so the truth is generated at
    # exactly the station states and epochs the fitter will later compute.
    lo = LayupObservatory(cache_dir=CACHE)
    et = np.array([spice.str2et(r["obsTime"]) for r in base], dtype="<f8")
    base_et = rfn.append_fields(base, "et", et, usemask=False, asrecarray=True)
    pos_vel = np.atleast_1d(lo.obscodes_to_barycentric(base_et))

    # Observer acceleration via the same finite-difference the driver uses, so the
    # in-test two-leg truth matches the C++ model and the noise-free fit -> csq 0.
    def _vel(shift_sec):
        b = base_et.copy()
        b["et"] = et + shift_sec
        pv = np.atleast_1d(lo.obscodes_to_barycentric(b))
        return np.stack([pv["vx"], pv["vy"], pv["vz"]], axis=-1)

    acc = (_vel(2.0) - _vel(-2.0)) * (86400.0 / 4.0)

    delay_us = np.full(n, np.nan)
    rms_delay = np.full(n, np.nan)
    doppler_hz = np.full(n, np.nan)
    rms_doppler = np.full(n, np.nan)
    for i in range(n):
        jd_tdb = convert_tdb_date_to_julian_date(base[i]["obsTime"], CACHE)
        r_obs = [pos_vel[i]["x"], pos_vel[i]["y"], pos_vel[i]["z"]]
        v_obs = [pos_vel[i]["vx"], pos_vel[i]["vy"], pos_vel[i]["vz"]]
        delay_days, doppler_audy = _radar_observables(ephem, true_state, epoch, jd_tdb, r_obs, v_obs, acc[i])
        if has_delay:
            delay_us[i] = delay_days / US_TO_DAYS
            rms_delay[i] = 1.0  # us
        if has_doppler:
            # Inverse of _radar_observation's doppler[au/day] = -c * Hz / freqTx.
            doppler_hz[i] = -doppler_audy * _FREQ_TX / SPEED_OF_LIGHT
            rms_doppler[i] = 0.1  # Hz

    data = np.empty(
        n,
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
    data["provID"] = "synth"
    data["obsTime"] = _OBS_TIMES
    data["stn"] = _STN
    data["delay"] = delay_us
    data["rmsDelay"] = rms_delay
    data["doppler"] = doppler_hz
    data["rmsDoppler"] = rms_doppler
    data["freqTx"] = _FREQ_TX

    # A perturbed prior orbit (flag==0 so orbitfit() uses it directly, bypassing
    # IOD). Small offset: well inside the basin of a zero-residual problem.
    guess = np.zeros(1, dtype=_get_result_dtypes("provID"))
    guess["provID"] = "synth"
    pert = true_state.copy()
    pert[:3] += 1.0e-5  # AU
    pert[3:] += 1.0e-7  # AU/day
    for k, v in zip(("x", "y", "z", "xdot", "ydot", "zdot"), pert):
        guess[k] = v
    guess["epochMJD_TDB"] = epoch - 2400000.5
    guess["flag"] = 0
    guess["FORMAT"] = "BCART_EQ"
    guess["method"] = "seed"
    for c in get_cov_columns():
        guess[c] = 0.0

    return data, guess, true_state


def test_radar_end_to_end_recovers_orbit():
    """delay + Doppler through orbitfit() recovers the truth at ~0 chi-squared."""
    data, guess, true_state = _build_radar_data(has_delay=True, has_doppler=True)

    fit = orbitfit(data, cache_dir=CACHE, initial_guess=guess, engine="cartesian")
    assert len(fit) == 1
    row = fit[0]
    assert row["flag"] == 0, f"radar fit did not converge (flag={row['flag']})"

    # 7 epochs x (delay + Doppler) = 14 rows, 6 free parameters.
    assert row["ndof"] == 2 * len(_OBS_TIMES) - 6

    fit_state = np.array([row["x"], row["y"], row["z"], row["xdot"], row["ydot"], row["zdot"]])
    pos_rel = np.linalg.norm(fit_state[:3] - true_state[:3]) / np.linalg.norm(true_state[:3])
    vel_rel = np.linalg.norm(fit_state[3:] - true_state[3:]) / np.linalg.norm(true_state[3:])
    assert pos_rel < 1e-6, f"position drift {pos_rel:.2e}"
    assert vel_rel < 1e-6, f"velocity drift {vel_rel:.2e}"

    # Noise-free observables -> the recovered orbit reproduces them to ~0 csq.
    assert row["csq"] < 1e-6, f"unexpected chi-square {row['csq']:.3e}"


def test_radar_end_to_end_delay_only_row_count_and_recovery():
    """A delay-only arc dispatches one row per observation and still recovers truth.

    Exercises the has_delay/has_doppler dispatch and the variable-row packing
    through the full driver: 7 delay rows, 6 parameters -> ndof = 1.
    """
    data, guess, true_state = _build_radar_data(has_delay=True, has_doppler=False)

    fit = orbitfit(data, cache_dir=CACHE, initial_guess=guess, engine="cartesian")
    assert len(fit) == 1
    row = fit[0]
    assert row["flag"] == 0, f"delay-only fit did not converge (flag={row['flag']})"
    assert row["ndof"] == len(_OBS_TIMES) - 6

    fit_state = np.array([row["x"], row["y"], row["z"], row["xdot"], row["ydot"], row["zdot"]])
    pos_rel = np.linalg.norm(fit_state[:3] - true_state[:3]) / np.linalg.norm(true_state[:3])
    assert pos_rel < 1e-5, f"delay-only position drift {pos_rel:.2e}"
