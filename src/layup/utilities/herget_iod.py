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
    #obs = [observations[i] for i in seq[0]]
    obs_1 = observations[0]
    r_e_1 = obs_1.observer_position
    rho_hat_1 = np.array(obs_1.get_rho_hat()) 
    rho_1 = 30 # this is the magnitude of rho, direction given by rho_hat, initial guess is 30au
    t_1 = obs_1.epoch
    r_1 = r_e_1 + rho_1*rho_hat_1
    #print('initial r_1 = ',r_1)

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
    while iteration < 40:#old_difference >= difference:
        
        delta_rho1, delta_rhon, x_1, y_1, z_1, vx1, vy1, vz1 = find_rho(observations, t_1, t_n, r_1, r_n, tolerance, args, aux, rho_hat_1, rho_hat_n)
        rho_1 += delta_rho1
        r_1 = r_e_1 + rho_1*np.array(rho_hat_1)
        rho_n += delta_rhon
        old_difference = difference
        difference = abs(rho_1 - rho_n) 
        #print(rho_1 - rho_n)
        print(delta_rho1, delta_rhon)
        r_n = r_e_n + rho_n*np.array(rho_hat_n)
        iteration += 1
        #print(r_n)
        #input('continue?')
    print('success!', rho_1, rho_n)



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
        

    

def find_rho(observations, t_1, t_n, r_1, r_n, tolerance, args, aux, rho_hat_1, rho_hat_n): 
    #print('r_n before = ', r_1)
    vx1, vy1, vz1, vxn, vyn, vzn = find_velocity(t_1, t_n, r_1, r_n, tolerance, args, aux)
    #print('r_n after = ', r_1)
    ephem, _, _ = create_assist_ephemeris(args, aux)
    sim = rebound.Simulation()
    sim.t = t_1 - ephem.jd_ref
    #print(r_1)
    sim.add(x = r_1[0], y = r_1[1], z = r_1[2], vx = vx1, vy =vy1, vz = vz1)
    var = sim.add_variation(testparticle=0)
    var.particles[0].xyz = rho_hat_1
    #print(var.particles[0])
    ex = assist.Extras(sim, ephem)
    times = []
    rhos = []
    a1, a2, b = np.zeros((3, 2*len(observations)))
    for i, observation in enumerate(observations):
        A = observation.get_a_vec()
        D = observation.get_d_vec()
        

        r_e = np.array(observation.observer_position)
        #print('r_e =', r_e)
        t = observation.epoch
        ex.integrate_or_interpolate(t - ephem.jd_ref)
        r = sim.particles[0].xyz
        #print('r = ',r)
        r_var = var.particles[0].xyz
        rho = r - r_e
        #print('rho = ', rho)
        times.append(t)
        rhos.append(np.sqrt(sum(rho**2)))
        rho_var = r_var - r_e
        b[2*i] = np.dot(rho, A)
        b[2*i + 1] = np.dot(rho, D)
        a1[2*i] = np.dot(rho_var, A)
        a1[2*i+1] = np.dot(rho_var, D)
        #print(dPdrho1, dQdrho1)
    plt.plot(times, rhos)
    plt.savefig('rho_over_time.png')
    # Do the same for rho_n
    #vxn, vyn, vzn = sim.particles[0].vxyz
    sim = rebound.Simulation()
    sim.t = t_n - ephem.jd_ref
    sim.add(x = r_n[0], y = r_n[1], z = r_n[2], vx = vxn, vy =vyn, vz = vzn)
    var = sim.add_variation(testparticle=0)
    var.particles[0].xyz = rho_hat_n
    #print(var.particles[0])
    ex = assist.Extras(sim, ephem)
    for i, observation in enumerate(observations):
        A = observation.get_a_vec()
        D = observation.get_d_vec()

        r_e = np.array(observation.observer_position)
        t = observation.epoch
        ex.integrate_or_interpolate(t - ephem.jd_ref)
        r = var.particles[0].xyz
        rho = r - r_e
        a2[2*i] = np.dot(rho, A)
        a2[2*i + 1] = np.dot(rho, D)
        #print(dPdrhon, dQdrhon)
    #print(a1, a2, b)
    k = 2*len(observations)
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

    #print(delta_rho1, delta_rhon)
    return delta_rho1, delta_rhon, r_1[0], r_1[1], r_1[2], vx1, vy1, vz1


def find_velocity(t1, tn, r_1, r_n, tolerance, args, aux):
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
    sim = rebound.Simulation()
    sim.t = t1 - ephem.jd_ref
    sim.add(x = x1, y = y1, z = z1, vx = vx1, vy =vy1, vz = vz1)
    var = sim.add_variation(testparticle=0)

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
    mag = np.dot(R-Q, Q-P)/np.dot(R-Q, R-Q)
    return mag
