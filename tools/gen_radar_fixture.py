"""Generate tests/data/radar_synthetic.json for the radar (delay/Doppler) fit.

Reuses the streak fixture's true orbit, epoch, and per-observation observer
states, and computes the round-trip radar observables with the SAME light-time
convention as the C++ model (predict.cpp::integrate_light_time iterates to the
retarded time t_obs - rho/c):

    delay   = 2 * rho / c + Shapiro    round-trip light time, days
    doppler = 2 * (rho_hat . v_rel)     round-trip range rate, au/day

rho/v are evaluated at the retarded emission time; v_rel = v_ast - v_obs.

The Shapiro (relativistic) term is included, matching orbit_fit.cpp. It has to
be: this fixture is truth for a fit that models it, so omitting it would make the
test measure the difference between two models rather than the fitter's ability
to recover an orbit. It is 1-2 us here against a stated 1 us uncertainty, on
seven observations constraining six parameters, so the difference is not subtle.
"""
import json
from pathlib import Path

import assist
import numpy as np
import pooch
import rebound

AU_M = 149597870700.0
C_AU_DAY = 2.99792458e8 * 86400.0 / AU_M  # matches predict.cpp SPEED_OF_LIGHT

CACHE = pooch.os_cache("layup")
STREAK = Path("tests/data/streak_synthetic.json")
OUT = Path("tests/data/radar_synthetic.json")

ephem = assist.Ephem(
    str(CACHE / "linux_p1550p2650.440"),
    str(CACHE / "sb441-n16.bsp"),
)
JD_REF = ephem.jd_ref


def state_at(true_state, epoch, t_target_jd):
    """Asteroid barycentric (r, v) at t_target_jd, integrating true_state@epoch."""
    sim = rebound.Simulation()
    sim.t = epoch - JD_REF
    sim.add(
        x=true_state[0], y=true_state[1], z=true_state[2],
        vx=true_state[3], vy=true_state[4], vz=true_state[5],
    )
    ax = assist.Extras(sim, ephem)
    sim.integrate(t_target_jd - JD_REF)
    p = sim.particles[0]
    r = np.array([p.x, p.y, p.z])
    v = np.array([p.vx, p.vy, p.vz])
    ax.detach(sim)
    return r, v


def radar_observables(true_state, epoch, obs_epoch, r_obs, v_obs):
    """Two-leg round-trip delay (days) and Doppler (au/day), observer accel = 0.

    Mirrors the C++ orbit_fit.cpp radar model with observer_acceleration left at
    its default of zero (this fixture is fed to run_from_vector directly without
    acceleration). Down leg: station at the receive epoch. Up leg: station linearly
    extrapolated to the transmit time t - tau by v_obs (no accel term).
    """
    r_obs = np.asarray(r_obs)
    v_obs = np.asarray(v_obs)
    # Down leg: retarded bounce time using the station at receive.
    tau_d = 0.0
    for _ in range(4):
        r_ast, v_ast = state_at(true_state, epoch, obs_epoch - tau_d)
        rho_d_vec = r_ast - r_obs
        rho_d = np.linalg.norm(rho_d_vec)
        tau_d = rho_d / C_AU_DAY
    rho_hat_d = rho_d_vec / rho_d
    # Up leg: station at the transmit time t - (tau_d + tau_u), linear in v_obs.
    tau_u = tau_d
    for _ in range(5):
        r_tx = r_obs - v_obs * (tau_d + tau_u)
        rho_u_vec = r_ast - r_tx
        rho_u = np.linalg.norm(rho_u_vec)
        tau_u = rho_u / C_AU_DAY
    rho_hat_u = rho_u_vec / rho_u

    # Shapiro delay on both legs -- the same formula and constant as
    # orbit_fit.cpp::compute_radar_residuals. The Sun moves ~1e-5 au over a round
    # trip, so a single evaluation at the receive epoch is ample.
    # At the C++ residual, integrate_light_time has left the simulation at the
    # emission (bounce) time, so the Sun is evaluated there and not at receive.
    sun = ephem.get_particle(0, (obs_epoch - tau_d) - JD_REF)   # ASSIST_BODY_SUN
    S = np.array([sun.x, sun.y, sun.z])
    GM_SUN = 2.9591220828559115e-4                    # au^3/day^2
    k = 2.0 * GM_SUN / C_AU_DAY**3
    r_b = np.linalg.norm(r_ast - S)
    r_r = np.linalg.norm(r_obs - S)
    r_t = np.linalg.norm(r_tx - S)
    shapiro = k * (np.log((r_t + r_b + rho_u) / (r_t + r_b - rho_u))
                   + np.log((r_b + r_r + rho_d) / (r_b + r_r - rho_d)))

    delay = tau_d + tau_u + shapiro
    doppler = float(rho_hat_d @ (v_ast - v_obs)) + float(rho_hat_u @ (v_ast - v_obs))
    return delay, doppler


def main():
    d = json.loads(STREAK.read_text())
    true_state = d["true_state"]
    epoch = d["epoch"]

    # 1-sigma uncertainties: realistic radar quality.
    # JPL delay ~ a few us round-trip; Doppler ~ sub-Hz. Convert to internal units.
    delay_unc_days = 1.0e-6 / 86400.0       # 1 us in days
    doppler_unc_audy = 1.0e-9                # ~ mm/s-level range-rate, au/day

    out = {
        "description": (
            "Synthetic noise-free radar (delay/Doppler) arc from the same MBA "
            "orbit as streak_synthetic.json. delay = round-trip light time (days), "
            "doppler = round-trip range rate (au/day); generated with ASSIST using "
            "the C++ light-time convention including the Shapiro delay (gen_radar_fixture.py)."
        ),
        "jd_ref": JD_REF,
        "epoch": epoch,
        "true_state": true_state,
        "delay_unc_days": delay_unc_days,
        "doppler_unc_audy": doppler_unc_audy,
        "observations": [],
    }
    for o in d["observations"]:
        delay, doppler = radar_observables(
            true_state, epoch, o["epoch"], o["observer_position"], o["observer_velocity"]
        )
        out["observations"].append(
            {
                "epoch": o["epoch"],
                "observer_position": o["observer_position"],
                "observer_velocity": o["observer_velocity"],
                "delay": delay,
                "doppler": doppler,
            }
        )
        print(f"  t={o['epoch']:.1f}  delay={delay:.10e} d  doppler={doppler:.10e} au/d")

    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
