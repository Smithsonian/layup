# Given an initial and final position, refine the velocity using
# Herget method as a way of creating a first guess for the inital orbit (IOD)
# import modules
import numpy as np
from scipy.integrate import RK45
import spiceypy as spice
import matplotlib.pyplot as plt
from sorcha.ephemeris.simulation_setup import furnish_spiceypy, create_assist_ephemeris
import assist
import rebound
from _layup_cpp._core import FitResult
from layup.utilities.universal_kepler import universal_step, KeplerConvergenceError
import copy

SPEED_OF_LIGHT_AU_DAY = 173.145


def herget_with_assist(observations, seq, tolerance, args, aux, max_iterations=100):
    seq_lengths = [len(i) for i in seq]
    longest_i = np.argmax(seq_lengths)  # finds the sequence index with the most observations contained in it
    obs = np.array(observations)[seq[longest_i]]

    # Define our values

    obs_1 = obs[0]
    r_e_1 = obs_1.observer_position
    rho_hat_1 = np.array(obs_1.rho_hat)
    rho_1 = 40  # this is the magnitude of rho, direction given by rho_hat, initial guess is 40au
    t_1 = obs_1.epoch
    r_1 = r_e_1 + rho_1 * rho_hat_1

    obs_n = obs[-1]
    r_e_n = obs_n.observer_position
    rho_hat_n = np.array(obs_n.rho_hat)
    rho_n = 40  # this is the magnitude of rho, direction given by rho_hat, initial guess is 40au
    t_n = obs_n.epoch
    r_n = r_e_n + rho_n * rho_hat_n

    iteration = 0
    delta_rho1 = tolerance + 1
    delta_rhon = tolerance + 1

    # Get original epochs so we can light-time correct them each iteration
    epochs = np.zeros(len(obs))
    for i, observation in enumerate(obs):
        epochs[i] = observation.epoch

    while (abs(delta_rho1) + abs(delta_rhon)) / 2 > tolerance and iteration < max_iterations:

        # Light-time correct the observation times
        for i, observation in enumerate(obs):
            # print(observation.epoch)
            observation.epoch = epochs[i] - ((rho_1) + (rho_n)) / (2 * SPEED_OF_LIGHT_AU_DAY)
            # print(observation.epoch)

        delta_rho1, delta_rhon, x_1, y_1, z_1, vx1, vy1, vz1 = find_drho(
            obs, t_1, t_n, r_1, r_n, tolerance, args, aux, rho_hat_1, rho_hat_n
        )

        # Update rho values
        rho_1 -= delta_rho1
        r_1 = r_e_1 + rho_1 * np.array(rho_hat_1)
        rho_n -= delta_rhon
        r_n = r_e_n + rho_n * np.array(rho_hat_n)
        print(delta_rho1, delta_rhon)
        # print(rho_1, rho_n)

        iteration += 1

    state = [x_1, y_1, z_1, vx1, vy1, vz1]
    solution = FitResult()
    solution.state = state
    solution.epoch = epochs[0]
    solution.method = "herget"
    solution.niter = iteration
    solution.flag = 0  # Success flag
    solution.ndof = len(observations)
    solution.csq = 0.0
    solution.cov = [0.01] * 36

    return [solution]


def find_drho(observations, t_1, t_n, r_1, r_n, tolerance, args, aux, rho_hat_1, rho_hat_n):

    # Find velocities at rho_1 and rho_n
    vx1, vy1, vz1, vxn, vyn, vzn = find_velocity(t_1, t_n, r_1, r_n, tolerance, args, aux)
    var_vx1, var_vy1, var_vz1, _, _, _ = find_velocity(t_1, t_n, r_1 + rho_hat_1, r_n, tolerance, args, aux)

    # Simulation setup
    ephem, _, _ = create_assist_ephemeris(args, aux)
    sim = rebound.Simulation()

    sim.add(x=r_1[0], y=r_1[1], z=r_1[2], vx=vx1, vy=vy1, vz=vz1)
    var = sim.add_variation(testparticle=0)
    var.particles[0].xyz = rho_hat_1
    var.particles[0].vxyz = np.array([var_vx1 - vx1, var_vy1 - vy1, var_vz1 - vz1])

    ex = assist.Extras(sim, ephem)
    sim.t = t_1 - ephem.jd_ref
    a1, a2, b = np.zeros((3, 2 * len(observations)))

    # For each observation, integrate to that time and find the residuals
    for i, observation in enumerate(observations):

        # For this observation, get A and D
        A = observation.a_vec
        D = observation.d_vec

        t = observation.epoch
        sim.integrate(t - ephem.jd_ref)

        r_e = np.array(observation.observer_position)
        r = sim.particles[0].xyz
        r_var = var.particles[0].xyz
        rho = r - r_e

        # Add these to the arrays
        b[2 * i] = np.dot(rho / np.linalg.norm(rho), A)
        b[2 * i + 1] = np.dot(rho / np.linalg.norm(rho), D)
        a1[2 * i] = b[2 * i] - np.dot((rho + r_var) / np.linalg.norm(rho + r_var), A)
        a1[2 * i + 1] = b[2 * i + 1] - np.dot((rho + r_var) / np.linalg.norm(rho + r_var), D)

    _, _, _, var_vxn, var_vyn, var_vzn = find_velocity(t_1, t_n, r_1, r_n + rho_hat_n, tolerance, args, aux)

    # Do the same for rho_n, set up simulation again
    vxn, vyn, vzn = sim.particles[0].vxyz
    sim = rebound.Simulation()
    sim.add(x=r_n[0], y=r_n[1], z=r_n[2], vx=vxn, vy=vyn, vz=vzn)
    var = sim.add_variation(testparticle=0)
    var.particles[0].xyz = rho_hat_n
    var.particles[0].vxyz = np.array([var_vxn - vxn, var_vyn - vyn, var_vzn - vzn])

    ex = assist.Extras(sim, ephem)
    sim.t = t_n - ephem.jd_ref

    # Find residuals for each observation
    for i, observation in enumerate(observations):
        A = observation.a_vec
        D = observation.d_vec

        t = observation.epoch

        sim.integrate(t - ephem.jd_ref)
        r_e = np.array(observation.observer_position)
        r = sim.particles[0].xyz
        r_var = var.particles[0].xyz
        rho = r - r_e

        # Add to array
        a2[2 * i] = b[2 * i] - np.dot((rho + r_var) / np.linalg.norm(rho + r_var), A)
        a2[2 * i + 1] = b[2 * i + 1] - np.dot((rho + r_var) / np.linalg.norm(rho + r_var), D)

    sigma_a1b = sum(a1 * b)
    sigma_a2b = sum(a2 * b)
    sigma_a1squared = sum(a1**2)
    sigma_a2squared = sum(a2**2)
    sigma_a1a2 = sum(a1 * a2)

    delta_rho1 = (sigma_a1b * sigma_a2squared - sigma_a2b * sigma_a1a2) / (
        sigma_a1a2**2 - sigma_a1squared * sigma_a2squared
    )
    delta_rhon = (-delta_rho1 * sigma_a1squared - sigma_a1b) / sigma_a1a2

    # Check this is the solution, should equal zero
    # print(sigma_a1b + delta_rho1*sigma_a1squared + delta_rhon*sigma_a1a2)
    # print(sigma_a2b + delta_rho1*sigma_a1a2 + delta_rhon*sigma_a2squared)
    # print(sum(a1*(b + delta_rho1*a1 + delta_rhon*a2)))

    return delta_rho1, delta_rhon, r_1[0], r_1[1], r_1[2], vx1, vy1, vz1


def find_velocity(t1, tn, r_1, r_n, tolerance, args, aux):

    # Initialising data so I can work with it
    delta_t = tn - t1
    x1 = r_1[0]
    y1 = r_1[1]
    z1 = r_1[2]
    xn = r_n[0]
    yn = r_n[1]
    zn = r_n[2]
    vx1 = (xn - x1) / delta_t
    vy1 = (yn - y1) / delta_t
    vz1 = (zn - z1) / delta_t

    pos = r_n + abs(tolerance) + 100

    # Find new values for vx, vy and vz in turn
    while np.linalg.norm(pos - r_n) > tolerance:
        [vx1, vy1, vz1], [*pos, vxn, vyn, vzn] = find_new_vel_with_universal_kepler(
            t1, tn, x1, y1, z1, vx1, vy1, vz1, xn, yn, zn
        )
        pos = np.array(pos)

        # vx1, vy1, vz1, vxn, vyn, vzn, pos = find_new_vel(ephem, t1, tn, x1, y1, z1, vx1, vy1, vz1, xn, yn, zn, change = 'x')
        # vx1, vy1, vz1, vxn, vyn, vzn, pos = find_new_vel(ephem, t1, tn, x1, y1, z1, vx1, vy1, vz1, xn, yn, zn, change = 'y')
        # vx1, vy1, vz1, vxn, vyn, vzn, pos = find_new_vel(ephem, t1, tn, x1, y1, z1, vx1, vy1, vz1, xn, yn, zn, change = 'z')

        # print(pos, r_n)
    # print(vx1, vy1, vz1)
    return (
        vx1,
        vy1,
        vz1,
        vxn,
        vyn,
        vzn,
    )


def find_new_vel(ephem, t1, tn, x1, y1, z1, vx1, vy1, vz1, xn, yn, zn, change):

    # Starting a new simulation
    sim = rebound.Simulation()

    sim.add(x=x1, y=y1, z=z1, vx=vx1, vy=vy1, vz=vz1)
    var = sim.add_variation(testparticle=0)

    # Depending on the direction of the variational particle,
    # see how the final position will change by varying in that direction

    # Vary the velocity in that direction by a factor that will get it as close to
    # the desired position as possible

    if change == "x":
        var.particles[0].vx = 1
        ex = assist.Extras(sim, ephem)
        sim.t = t1 - ephem.jd_ref
        sim.integrate(tn - ephem.jd_ref)
        diff = find_mag_to_adjust(
            np.array([xn, yn, zn]),
            np.array(sim.particles[0].xyz),
            np.array(sim.particles[0].xyz) + np.array(var.particles[0].xyz),
        )
        vx1 += diff
    elif change == "y":
        var.particles[0].vy = 1
        ex = assist.Extras(sim, ephem)
        sim.t = t1 - ephem.jd_ref
        sim.integrate(tn - ephem.jd_ref)
        diff = find_mag_to_adjust(
            np.array([xn, yn, zn]),
            np.array(sim.particles[0].xyz),
            np.array(sim.particles[0].xyz) + np.array(var.particles[0].xyz),
        )
        vy1 += diff
    elif change == "z":
        var.particles[0].vz = 1
        ex = assist.Extras(sim, ephem)
        sim.t = t1 - ephem.jd_ref
        sim.integrate(tn - ephem.jd_ref)
        diff = find_mag_to_adjust(
            np.array([xn, yn, zn]),
            np.array(sim.particles[0].xyz),
            np.array(sim.particles[0].xyz) + np.array(var.particles[0].xyz),
        )
        vz1 += diff
    [vxn, vyn, vzn] = sim.particles[0].vxyz
    return vx1, vy1, vz1, vxn, vyn, vzn, np.array(sim.particles[0].xyz)


def find_mag_to_adjust(P, Q, R):
    # This is the formula for the point on a line (defined by Q and R)
    # that is closest to a point outside the line, P
    # For our purpose, this is the scale factor to vary the velocity by so that it
    # will be closest to rho_n next time
    mag = np.dot(R - Q, P - Q) / np.dot(R - Q, R - Q)
    return mag


def find_new_vel_with_universal_kepler(t1, tn, x1, y1, z1, vx1, vy1, vz1, xn, yn, zn):
    # Initialise state
    dt = tn - t1
    state = np.array([x1, y1, z1, vx1, vy1, vz1])
    GMtotal = 0.0002963092748799319
    variation_vx = np.array([0, 0, 0, 1, 0, 0])
    variation_vy = np.array([0, 0, 0, 0, 1, 0])
    variation_vz = np.array([0, 0, 0, 0, 0, 1])

    var_vx = universal_step(GMtotal, dt, state, variation=variation_vx)
    diff = find_mag_to_adjust(
        np.array([xn, yn, zn]),
        np.array(var_vx.state[:3]),
        np.array(var_vx.state[:3]) + np.array(var_vx.variation[:3]),
    )
    state[3] += diff
    var_vy = universal_step(GMtotal, dt, state, variation=variation_vy)
    diff = find_mag_to_adjust(
        np.array([xn, yn, zn]),
        np.array(var_vy.state[:3]),
        np.array(var_vy.state[:3]) + np.array(var_vy.variation[:3]),
    )
    state[4] += diff
    var_vz = universal_step(GMtotal, dt, state, variation=variation_vz)
    diff = find_mag_to_adjust(
        np.array([xn, yn, zn]),
        np.array(var_vz.state[:3]),
        np.array(var_vz.state[:3]) + np.array(var_vz.variation[:3]),
    )
    state[5] += diff

    return state[3:], np.array(var_vz.state)
