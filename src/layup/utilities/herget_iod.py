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

def herget_with_assist(observations, seq, tolerance, args, aux, max_iterations=100):

    # Define our values

    obs_1 = observations[0]
    r_e_1 = obs_1.observer_position
    rho_hat_1 = np.array(obs_1.get_rho_hat()) 
    rho_1 = 30 # this is the magnitude of rho, direction given by rho_hat, initial guess is 30au
    t_1 = obs_1.epoch
    r_1 = r_e_1 + rho_1*rho_hat_1


    obs_n = observations[-1]
    r_e_n = obs_n.observer_position
    rho_hat_n = np.array(obs_n.get_rho_hat())
    rho_n = 30 # this is the magnitude of rho, direction given by rho_hat, initial guess is 30au
    t_n = obs_n.epoch
    r_n = r_e_n + rho_n*rho_hat_n
    iteration = 0
    delta_rho1 = tolerance + 1
    delta_rhon = tolerance + 1
    while (delta_rho1 + delta_rhon) / 2 > tolerance and iteration < max_iterations:
        
        delta_rho1, delta_rhon, x_1, y_1, z_1, vx1, vy1, vz1 = find_drho(observations, t_1, t_n, r_1, r_n, tolerance, args, aux, rho_hat_1, rho_hat_n)
        
        # Update rho values
        rho_1 -= delta_rho1
        r_1 = r_e_1 + rho_1*np.array(rho_hat_1)
        rho_n -= delta_rhon 
        r_n = r_e_n + rho_n*np.array(rho_hat_n)

        iteration += 1



    state = [x_1, y_1, z_1, vx1, vy1, vz1]
    solution = FitResult()
    solution.state = state
    solution.epoch = obs_1.epoch
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

    # Simulation setup
    ephem, _, _ = create_assist_ephemeris(args, aux)
    sim = rebound.Simulation()
    
    sim.add(x = r_1[0], y = r_1[1], z = r_1[2], vx = vx1, vy =vy1, vz = vz1)
    var = sim.add_variation(testparticle=0)
    var.particles[0].xyz = rho_hat_1
    ex = assist.Extras(sim, ephem)
    sim.t = t_1 - ephem.jd_ref
    a1, a2, b = np.zeros((3, 2*len(observations)))


    print('going forward')
    for i, observation in enumerate(observations):

        # For this observation, get A and D
        A = observation.get_a_vec()
        D = observation.get_d_vec()
        
        t = observation.epoch
        sim.integrate(t - ephem.jd_ref)

        r_e = np.array(observation.observer_position)
        r = sim.particles[0].xyz
        r_var = var.particles[0].xyz
        rho = r - r_e
        rho_var = r_var


        # Add these to the arrays
        b[2*i] = np.dot(rho, A)
        b[2*i + 1] = np.dot(rho, D)
        a1[2*i] = (b[2*i] - np.dot(rho + rho_var, A))
        a1[2*i + 1] = (b[2*i + 1] - np.dot(rho + rho_var, D))

    

    # Do the same for rho_n, set up simulation again
    vxn, vyn, vzn = sim.particles[0].vxyz
    sim = rebound.Simulation()
    sim.add(x = r_n[0], y = r_n[1], z = r_n[2], vx = vxn, vy =vyn, vz = vzn)
    var = sim.add_variation(testparticle=0)
    var.particles[0].xyz = rho_hat_n
    ex = assist.Extras(sim, ephem)
    sim.t = t_n - ephem.jd_ref

    
    for i, observation in enumerate(observations):
        A = observation.get_a_vec()
        D = observation.get_d_vec()

        
        t = observation.epoch
        
        sim.integrate(t - ephem.jd_ref)
        r_e = np.array(observation.observer_position)
        r = sim.particles[0].xyz
        r_var = var.particles[0].xyz
        rho = r - r_e
        
        # Add to array
        a2[2*i] = (b[2*i] - np.dot(rho + r_var, A))
        a2[2*i + 1] = (b[2*i + 1] - np.dot(rho + r_var, D))

        

    sigma_a1b = sum(a1*b)
    sigma_a2b = sum(a2*b)
    sigma_a1squared = sum(a1**2)
    sigma_a2squared = sum(a2**2)
    sigma_a1a2 = sum(a1*a2)

    delta_rho1 = (sigma_a1b*sigma_a2squared - sigma_a2b*sigma_a1a2)/(sigma_a1a2**2 - sigma_a1squared*sigma_a2squared)
    delta_rhon = (-delta_rho1*sigma_a1squared - sigma_a1b)/sigma_a1a2

    # Check this is the solution, should equal zero
    #print(sigma_a1b + delta_rho1*sigma_a1squared + delta_rhon*sigma_a1a2)
    #print(sigma_a2b + delta_rho1*sigma_a1a2 + delta_rhon*sigma_a2squared)
    #print(sum(a1*(b + delta_rho1*a1 + delta_rhon*a2)))

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

    furnish_spiceypy(args, aux)
    ephem, _, _ = create_assist_ephemeris(args, aux)
    pos = r_n + abs(tolerance) + 100
    
    while abs(np.sqrt(sum(r_n**2)) - np.sqrt(sum(pos**2))) > tolerance:
        vx1, vy1, vz1, vxn, vyn, vzn, pos = find_new_vel(ephem, t1, tn, x1, y1, z1, vx1, vy1, vz1, xn, yn, zn, change = 'x')
        vx1, vy1, vz1, vxn, vyn, vzn, pos = find_new_vel(ephem, t1, tn, x1, y1, z1, vx1, vy1, vz1, xn, yn, zn, change = 'y')
        vx1, vy1, vz1, vxn, vyn, vzn, pos = find_new_vel(ephem, t1, tn, x1, y1, z1, vx1, vy1, vz1, xn, yn, zn, change = 'z')
    spice.kclear()
    return vx1, vy1, vz1, vxn, vyn, vzn,

def find_new_vel(ephem, t1, tn, x1, y1, z1, vx1, vy1, vz1, xn, yn, zn, change):

    # Starting a new simulation
    sim = rebound.Simulation()
    
    sim.add(x = x1, y = y1, z = z1, vx = vx1, vy =vy1, vz = vz1)
    var = sim.add_variation(testparticle=0)

    # Depending on the direction of the variational particle, 
    # see how the final position will change by varying in that direction

    # Vary the velocity in that direction by a factor that will get it as close to 
    # the desired position as possible

    if change == 'x':
        var.particles[0].vx = 1
        ex = assist.Extras(sim, ephem)
        sim.t = t1 - ephem.jd_ref
        sim.integrate(tn - ephem.jd_ref)
        diff = find_mag_to_adjust(np.array([xn, yn, zn]), np.array(sim.particles[0].xyz), np.array(sim.particles[0].xyz) + np.array(var.particles[0].xyz))
        vx1 -= diff 
    elif change == 'y':
        var.particles[0].vy = 1
        ex = assist.Extras(sim, ephem)
        sim.t = t1 - ephem.jd_ref
        sim.integrate(tn - ephem.jd_ref)
        diff = find_mag_to_adjust(np.array([xn, yn, zn]), np.array(sim.particles[0].xyz), np.array(sim.particles[0].xyz) + np.array(var.particles[0].xyz))
        vy1 -= diff
    elif change == 'z':
        var.particles[0].vz = 1
        ex = assist.Extras(sim, ephem)
        sim.t = t1 - ephem.jd_ref
        sim.integrate(tn - ephem.jd_ref)
        diff = find_mag_to_adjust(np.array([xn, yn, zn]), np.array(sim.particles[0].xyz), np.array(sim.particles[0].xyz) + np.array(var.particles[0].xyz))
        vz1 -= diff 
    [vxn, vyn, vzn] = sim.particles[0].vxyz
    return vx1, vy1, vz1, vxn, vyn, vzn, np.array(sim.particles[0].xyz)

def find_mag_to_adjust(P, Q, R):
    # This is the formula for the point on a line (defined by Q and R)
    # that is closest to a point outside the line, P
    # For our purpose, this is the scale factor to vary the velocity by so that it
    # will be closest to rho_n next time
    mag = np.dot(R-Q, Q-P)/np.dot(R-Q, R-Q)
    return mag
