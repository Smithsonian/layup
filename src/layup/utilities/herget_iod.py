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
    #print(r_n)
    difference = abs(rho_1 - rho_n) 
    old_difference = abs(rho_1 - rho_n)
    iteration = 0
    while iteration < 10:
        
        delta_rho1, delta_rhon, x_1, y_1, z_1, vx1, vy1, vz1 = find_drho(observations, t_1, t_n, r_1, r_n, tolerance, args, aux, rho_hat_1, rho_hat_n)
        
        # Update rho values
        rho_1 += delta_rho1
        r_1 = r_e_1 + rho_1*np.array(rho_hat_1)
        rho_n += delta_rhon 
        r_n = r_e_n + rho_n*np.array(rho_hat_n)

        print(delta_rho1, delta_rhon)
        iteration += 1

    print('success: rho_1 =', rho_1,'rho_n =', rho_n)



    state = [x_1, y_1, z_1, vx1, vy1, vz1]
    solution = FitResult()
    solution.state = state
    solution.epoch = obs_1.epoch
    solution.method = "herget"
    solution.niter = 0
    solution.flag = 0  # Success flag
    solution.ndof = 0
    solution.csq = 0.0
    solution.cov = [0.01] * 36

    return [solution]
        

    

def find_drho(observations, t_1, t_n, r_1, r_n, tolerance, args, aux, rho_hat_1, rho_hat_n): 
    # Find velocities at rho_1 and rho_n
    vx1, vy1, vz1, vxn, vyn, vzn = find_velocity(t_1, t_n, r_1, r_n, tolerance, args, aux)

    # Simulation setup
    ephem, _, _ = create_assist_ephemeris(args, aux)
    sim = rebound.Simulation()
    sim.t = t_1 - ephem.jd_ref
    sim.add(x = r_1[0], y = r_1[1], z = r_1[2], vx = vx1, vy =vy1, vz = vz1)
    var = sim.add_variation(testparticle=0)
    sim.add(x = r_1[0], y = r_1[1], z = r_1[2], vx = vx1, vy =vy1, vz = vz1)
    var2 = sim.add_variation(testparticle=1)
    var.particles[0].xyz = rho_hat_1
    var2.particles[0].xyz = -rho_hat_1
    ex = assist.Extras(sim, ephem)
    times = []
    rhos = []
    a1, a2, b = np.zeros((3, 2*len(observations)))


    print('going forward')
    for i, observation in enumerate(observations):

        # For this observation, get A and D
        A = observation.get_a_vec()
        D = observation.get_d_vec()
        
        r_e = np.array(observation.observer_position)
        t = observation.epoch

        ex.integrate_or_interpolate(t - ephem.jd_ref)

        r = sim.particles[0].xyz
        r_var = var.particles[0].xyz
        r_var2 = var2.particles[0].xyz
        rho = r - r_e
        rho_var = r_var - r_e
        rho_var2 = r_var2 - r_e

        # Add these to the arrays
        b[2*i] = np.dot(rho, A)
        b[2*i + 1] = np.dot(rho, D)
        a1[2*i] = (np.dot(rho_var, A) - np.dot(rho_var2, A))/2
        a1[2*i + 1] = (np.dot(rho_var, D) - np.dot(rho_var2, D))/2

        times.append(t)
        rhos.append(np.sqrt(sum(np.array(rho)**2)))
    
    plt.plot(times, a1[1::2], '.')
    plt.savefig('rho_over_time.png')

    # Do the same for rho_n, set up simulation again
    vxn, vyn, vzn = sim.particles[0].vxyz
    sim = rebound.Simulation()
    sim.t = t_n - ephem.jd_ref
    sim.add(x = r_n[0], y = r_n[1], z = r_n[2], vx = vxn, vy =vyn, vz = vzn)
    sim.add(x = r_n[0], y = r_n[1], z = r_n[2], vx = vxn, vy =vyn, vz = vzn)
    var = sim.add_variation(testparticle=0)
    var2 = sim.add_variation(testparticle=1)
    var.particles[0].xyz = rho_hat_n
    var2.particles[0].xyz = -rho_hat_n
    ex = assist.Extras(sim, ephem)

    times = []
    rhos = []
    print('going backwards')
    for i, observation in enumerate(observations):
        A = observation.get_a_vec()
        D = observation.get_d_vec()

        r_e = np.array(observation.observer_position)
        t = observation.epoch
        
        ex.integrate_or_interpolate(t - ephem.jd_ref)

        r = sim.particles[0].xyz
        r_var = var.particles[0].xyz
        r_var2 = var2.particles[0].xyz
        rho = r - r_e
        rho_var = r_var - r_e
        rho_var2 = r_var2 - r_e
        times.append(t)
        rhos.append(np.sqrt(sum(np.array(rho_var)**2)))
        
        # Add to array
        a2[2*i] = (np.dot(rho_var, A) - np.dot(rho_var2, A)) / 2
        a2[2*i + 1] = (np.dot(rho_var, D) - np.dot(rho_var2, D)) / 2

        
    #print(a1, a2, b)
    plt.plot(times, a2[1::2], '.')
    plt.savefig('rho_over_time_backwards.png')
    print(a1[1::2] - a2[1::2])

    sigma_a1b = sum(a1*b)
    sigma_a2b = sum(a2*b)
    sigma_a1squared = sum(a1**2)
    sigma_a2squared = sum(a2**2)
    sigma_a1a2 = sum(a1*a2)
    #print(b, sigma_a1b, sigma_a2b, sigma_a1squared, sigma_a2squared, sigma_a1a2)

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
    sim.t = t1 - ephem.jd_ref
    sim.add(x = x1, y = y1, z = z1, vx = vx1, vy =vy1, vz = vz1)
    var = sim.add_variation(testparticle=0)

    # Depending on the direction of the variational particle, 
    # see how the final position will change by varying in that direction

    # Vary the velocity in that direction by a factor that will get it as close to 
    # the desired position as possible

    if change == 'x':
        var.particles[0].vx = 1
        ex = assist.Extras(sim, ephem)
        ex.integrate_or_interpolate(tn - ephem.jd_ref)
        diff = find_mag_to_adjust(np.array([xn, yn, zn]), np.array(sim.particles[0].xyz), np.array(sim.particles[0].xyz) + np.array(var.particles[0].xyz))
        vx1 -= diff 
    elif change == 'y':
        var.particles[0].vy = 1
        ex = assist.Extras(sim, ephem)
        ex.integrate_or_interpolate(tn - ephem.jd_ref)
        diff = find_mag_to_adjust(np.array([xn, yn, zn]), np.array(sim.particles[0].xyz), np.array(sim.particles[0].xyz) + np.array(var.particles[0].xyz))
        vy1 -= diff
    elif change == 'z':
        var.particles[0].vz = 1
        ex = assist.Extras(sim, ephem)
        ex.integrate_or_interpolate(tn - ephem.jd_ref)
        diff = find_mag_to_adjust(np.array([xn, yn, zn]), np.array(sim.particles[0].xyz), np.array(sim.particles[0].xyz) + np.array(var.particles[0].xyz))
        vz1 -= diff 
    [vxn, vyn, vzn] = sim.particles[0].vxyz
    #print('new_pos should be' ,np.array(sim.particles[0].xyz) - diff*np.array(var.particles[0].xyz))
    return vx1, vy1, vz1, vxn, vyn, vzn, np.array(sim.particles[0].xyz)

def find_mag_to_adjust(P, Q, R):
    # This is the formula for the point on a line (defined by Q and R)
    # that is closest to a point outside the line, P
    # For our purpose, this is the scale factor to vary the velocity by so that it
    # will be closest to rho_n next time
    mag = np.dot(R-Q, Q-P)/np.dot(R-Q, R-Q)
    return mag
