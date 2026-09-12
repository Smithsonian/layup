# Given an initial and final position, refine the velocity using
# Herget method as a way of creating a first guess for the inital orbit (IOD)
# import modules
import numpy as np
from sorcha.ephemeris.simulation_setup import create_assist_ephemeris
import assist
import rebound
from layup.routines import FitResult
from layup.utilities.universal_kepler import universal_step, KeplerConvergenceError, state_transition_matrix
from layup.constants import MU_SUN, SPEED_OF_LIGHT


def herget_with_assist(observations, seq, ephem, tolerance=0.003, max_iterations=100, initial_rho=2):
    """Runs the Herget method on a set of observations.

    Parameters
    ----------
    observations : list
        List of all the observations of the object
    seq : list[list]
        list of lists containing the indices of observations that are closely spaced in time
    ephem : assist.ephem.Ephem object
        the ephemeris data for running assist
    tolerance : float
        the maximum delta_rho residuals allowed; will continue to converge until the residuals are below this value
    max_iterations : int (optional, default: 100)
        the maximum number of iterations before the fitting stops
    initial_rho : float (optional, default: 2.0)
        The initial guess for the range (for both the first and nth observation)"""
    seq_lengths = [len(i) for i in seq]
    longest_i = np.argmax(seq_lengths)  # finds the sequence index with the most observations contained in it
    obs = np.array(observations)[seq[longest_i]]

    # Define our values

    obs_1 = obs[0]
    r_e_1 = obs_1.observer_position
    rho_hat_1 = np.array(obs_1.rho_hat)
    rho_1 = initial_rho  # this is the magnitude of rho, direction given by rho_hat, initial guess is 2au
    t1 = obs_1.epoch
    r1 = r_e_1 + rho_1 * rho_hat_1

    obs_n = obs[-1]
    r_e_n = obs_n.observer_position
    rho_hat_n = np.array(obs_n.rho_hat)
    rho_n = initial_rho  # this is the magnitude of rho, direction given by rho_hat, initial guess is 2au
    tn = obs_n.epoch
    rn = r_e_n + rho_n * rho_hat_n

    iteration = 0
    delta_rho1 = tolerance + 1
    delta_rhon = tolerance + 1
    dets = [1] # determinant of the 2x2 matrix, will stop the loop if this is too small

    # Get original epochs so we can light-time correct them each iteration
    epochs = np.zeros(len(obs))
    for i, observation in enumerate(obs):
        epochs[i] = observation.epoch

    while (abs(delta_rho1) + abs(delta_rhon)) / 2 > tolerance and iteration < max_iterations and np.median(np.array(dets)) > 0.01:

        # Light-time correct the observation times
        for i, observation in enumerate(obs):
            observation.epoch = epochs[i] - ((rho_1) + (rho_n)) / (2 * SPEED_OF_LIGHT)

        delta_rho1, delta_rhon, state_1, det = find_drho(
            obs, t1, tn, r1, rn, tolerance, ephem, rho_1, rho_hat_1, rho_n, rho_hat_n, max_iterations
        )
        if abs(delta_rho1) > rho_1 / 2:
            delta_rho1 = (
                (abs(delta_rho1) / delta_rho1) * rho_1 / 2
            )  # to prevent a runaway effect, cap delta_rho to half of rho
        if abs(delta_rhon) > rho_n / 2:
            delta_rhon = (abs(delta_rhon) / delta_rhon) * rho_n / 2
        # print(delta_rho1, delta_rhon, state_1)
        
        dets.append(abs(det))
        print(det)

        # Update rho values
        rho_1 -= delta_rho1
        r1 = r_e_1 + rho_1 * np.array(rho_hat_1)
        rho_n -= delta_rhon
        rn = r_e_n + rho_n * np.array(rho_hat_n)

        iteration += 1
        
    # restore observation epochs
    for i, observation in enumerate(obs):
        observation.epoch = epochs[i]
    
    if iteration >= max_iterations or det <= 0.01:
        return (
            []
        )  # if max_iterations is reached consider the IOD a failure, return empty list (will trigger flag 5)

    

    state = state_1
    solution = FitResult()
    solution.state = state
    solution.epoch = epochs[0]
    solution.method = "herget"
    solution.niter = iteration
    solution.flag = 0  # success flag
    solution.ndof = len(observations)
    solution.csq = 0.0
    solution.cov = [0.01] * 36

    return [solution]


def find_drho(
    observations, t1, tn, r1, rn, tolerance, ephem, rho_1, rho_hat_1, rho_n, rho_hat_n, max_iterations=100
):
    """Find the adjustment to make to rho_1 and rho_n to make in order to reduce the residuals of the observations

    Parameters
    ----------
    observations : list
        list of observation objects
    t1 : float
        light-time corrected time for position r1, in TDB MJD
    tn : float
        light-time corrected time for position rn, in TDB MJD
    r1 : numpy array
        position vector at time t1
    rn : numpy array
        position vector at time tn
    tolerance : float
        the average value of delta_rho1 and delta_rhon at which the orbit is considered to have converged at
    ephem : assist.ephem.Ephem object
        the ephemeris data for running assist
    rho_1 : float
        the magnitude of rho at time t1
    rho_hat_1 : numpy array
        unit vector of rho at time t1
    rho_n : float
        the magnitude of rho at time tn
    rho_hat_n : numpy array
        unit vector of rho at time tn
    max_iterations : int (optional, default: 100)
        the maximum number of iterations before the velocity fitter stops

    Returns
    -------
    delta_rho1 : float
        the amount to adjust rho_1 by to return a more accurate orbit
    delta_rhon : float
        the amount to adjust rho_n by to return a more accurate orbit
    state_1[x, y, z, vx, vy, vz]
        the new guess for the state vector at t1
    det : float
        determinant of the matrix of coefficients to the normal equations
    """

    # Find velocities at rho_1 and rho_n
    [vx1, vy1, vz1], [vxn, vyn, vzn] = find_velocity(t1, tn, r1, rn, tolerance * rho_1 / 100, max_iterations)
    
    # finding vel using state transition matrix; uses the state_transition_matrix from universal_kepler.py to analytically
    # determine the change in velocity 
    Phi = state_transition_matrix(MU_SUN, tn - t1, [*r1, vx1, vy1, vz1])
    
    if rho_1 > 2:
        drho1 = rho_hat_1
        drhon = rho_hat_n
    else:
        drho1 = rho_hat_1 * 0.01
        drhon = rho_hat_n * 0.01
        
    var_vxyz_1 = -np.matmul(np.matmul(np.linalg.inv(Phi[0:3, 3:6]), Phi[0:3, 0:3]), np.transpose(drho1))
    var_vxyz_n = np.matmul(np.matmul(Phi[3:6, 3:6], np.linalg.inv(Phi[0:3, 3:6])), np.transpose(drhon))

    # Simulation setup
    sim = rebound.Simulation()

    sim.add(x=r1[0], y=r1[1], z=r1[2], vx=vx1, vy=vy1, vz=vz1)
    var = sim.add_variation(testparticle=0)
    var.particles[0].xyz = drho1
    var.particles[0].vxyz = var_vxyz_1

    ex = assist.Extras(sim, ephem)
    sim.t = t1 - ephem.jd_ref
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


    # Do the same for rho_n, set up simulation again
    sim = rebound.Simulation()
    sim.add(x=rn[0], y=rn[1], z=rn[2], vx=vxn, vy=vyn, vz=vzn)
    var = sim.add_variation(testparticle=0)
    var.particles[0].xyz = drhon
    var.particles[0].vxyz = var_vxyz_n

    ex = assist.Extras(sim, ephem)
    sim.t = tn - ephem.jd_ref

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

    sigma_a1b = sum((a1 * b)/np.linalg.norm(drho1))
    sigma_a2b = sum((a2 * b)/np.linalg.norm(drhon))
    sigma_a1squared = sum((a1**2)/sum(drho1**2))
    sigma_a2squared = sum(a2**2/sum(drhon**2)) 
    sigma_a1a2 = sum(a1 * a2/ sum(drho1*drhon))

    delta_rho1 = (sigma_a1b * sigma_a2squared - sigma_a2b * sigma_a1a2) / (
        sigma_a1a2**2 - sigma_a1squared * sigma_a2squared
    )
    delta_rhon = (-delta_rho1 * sigma_a1squared - sigma_a1b) / sigma_a1a2

    # Check this is the solution, should equal zero
    # print(sigma_a1b + delta_rho1*sigma_a1squared + delta_rhon*sigma_a1a2)
    # print(sigma_a2b + delta_rho1*sigma_a1a2 + delta_rhon*sigma_a2squared)
    # print(sum(a1*(b + delta_rho1*a1 + delta_rhon*a2)))
    det = sigma_a1squared * sigma_a2squared - sigma_a1a2**2 # determinant of the 2x2 matrix; good check for degeneracy

    return delta_rho1, delta_rhon, [*r1, vx1, vy1, vz1], det


def find_velocity(t1, tn, r1, rn, tolerance=0.0001, max_iterations=100):
    """Converge on a velocity which takes position r1 at time t1 to position rn at time tn.
    Uses the universal kepler stepper to integrate over time.

    Parameters
    ----------
    t1 : float
        Light-time corrected time of first state, in TDB MJD
    tn : float
        Light-time corrected time of nth state, in TDB MJD
    r1 : numpy array
        Position vector at time t1
    rn : numpy array
        Position vector at time tn
    tolerance : float
        how closely the calculated rn value must lie within the correct value
    max_iterations : int (optional, default: 100)
        the maximum number of iterations before the velocity fitter stops


    Returns
    -------
    state_1[vx, vy, vz]
        The velocity components of state vector at t1
    state_n[vx, vy, vz]
        The velocity components of state vector at tn
    """

    # Initialising data
    delta_t = tn - t1
    state_1 = np.array([*r1, 0, 0, 0])

    for i in range(3):
        state_1[i + 3] = (rn[i] - r1[i]) / delta_t

    state_n = [*rn + abs(tolerance) + 1, 0, 0, 0]

    # Find new values for vx, vy and vz in turn
    iter = 0
    while np.linalg.norm(state_n[:3] - rn) > tolerance and iter < max_iterations:
        state_1[3:], state_n = find_new_vel_with_universal_kepler(t1, tn, state_1, rn)
        iter += 1

    return state_1[3:], state_n[3:]


def find_mag_to_adjust(P, Q, R):
    # This is the formula for the point on a line (defined by Q and R)
    # that is closest to a point outside the line, P
    # For our purpose, this is the scale factor to vary the velocity by so that it
    # will be closest to rho_n next time
    mag = np.dot(R - Q, P - Q) / np.dot(R - Q, R - Q)
    return mag


def find_new_vel_with_universal_kepler(t1, tn, state_1, state_n):
    """Adjust the velocity components of an input position-velocity state to land closer to a desired position

    Parameters
    ----------
    t1 : float
        Light-time corrected time of first state, in TDB MJD
    tn : float
        Light-time corrected time of nth state, in TDB MJD
    state_1 : numpy array
        the state vector at time t1; in AU and AU/day
    state_n : numpy array
        the state vector at time tn; in AU and AU/day

    Returns
    -------
    state_1[vx, vy, vz]
        Adjusted velocity components of state_1
    state_n[x, y, z, vx, vy, vz]
        The new state vector at time tn when using state_1 as the input"""

    # Initialise variables
    dt = tn - t1

    for i in range(3):
        variation = np.zeros(6)
        variation[i + 3] = 1  # we are varying vx, vy and vz by 1, once at a time
        var_results = universal_step(MU_SUN, dt, state_1, variation=variation)

        diff = find_mag_to_adjust(
            np.array(state_n[:3]),
            np.array(var_results.state[:3]),
            np.array(var_results.state[:3]) + np.array(var_results.variation[:3]),
        )  # find the velocity in this cartesian direction that will get us closest to the desired position
        state_1[i + 3] += diff

    return state_1[3:], np.array(var_results.state)
