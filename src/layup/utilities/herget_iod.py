# Given an initial and final position, refine the velocity using 
# Herget method as a way of creating a first guess for the inital orbit (IOD)
# import modules
import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from sorcha.ephemeris.simulation_setup import furnish_spiceypy, create_assist_ephemeris
import assist
import rebound
from _layup_cpp._core import FitResult

def cart2sphere(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.acos(z/r)
    phi = np.acos(x/np.sqrt(x**2 + y**2))
    return r, theta, phi

def sphere2cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def dstate_dt(t, state, t_initial):
    """
    Differential equation for two-body problem.
    state = [x, y, z, vx, vy, vz]
    """
    x, y, z, vx, vy, vz = state

    dt = t - t_initial
    r, theta, phi = cart2sphere(x, y, z)

    # Gravitational parameter for the Sun in AU^3/day^2
    k = 0.01720209895
    mu = k**2  # = 2.959122082855911e-4 AU^3/day^2
    
    # acceleration in radial coords
    ar = -mu / r**2
    ax, ay, az = sphere2cart(ar, theta, phi)

    # accelerations in each direction ie. dv/dt
    vx += ax * dt
    vy += ay * dt
    vz += az * dt


    return np.array([vx, vy, vz, ax, ay, az])

# this is very thrown together so youll have to check and mess with everything. 
def herget(start_obs, end_obs, tolerance, max_iterations=1000):
    """
    random attempt  at herget's method for initial orbit determination. not right
    """
    x_1, y_1, z_1 = start_obs.observer_position
    x_2, y_2, z_2 = end_obs.observer_position
    print(start_obs.observer_position)
    print(end_obs.observer_position)

    # Initial guess and velocities
    xr2, yr2, zr2 = x_2 - 1, y_2 - 1, z_2 - 1
    t_1 = start_obs.epoch
    t_2 = end_obs.epoch
    delta_t = end_obs.epoch - start_obs.epoch
    vx1 = (x_2 - x_1) / delta_t
    vy1 = (y_2 - y_1) / delta_t
    vz1 = (z_2 - z_1) / delta_t

    iteration = 0


    while abs(xr2 - x_2) > tolerance or abs(yr2 - y_2) > tolerance or abs(zr2 - z_2) > tolerance:
        #print(abs(xr2 - x2), abs(yr2 - y2), abs(zr2 - z2))
        if iteration >= max_iterations:  # failed sol
            state = [x_1, y_1, z_1, vx1, vy1, vz1]
            solution = FitResult()
            solution.state = state
            solution.epoch = start_obs.epoch
            solution.method = "herget"
            solution.niter = iteration
            solution.flag = 1  # fail flag?
            solution.ndof = 0
            solution.csq = 0.0
            solution.cov = [0.0] * 36
            return [solution]

        state0 = np.array([x_1, y_1, z_1, vx1, vy1, vz1])
        T = [0]
        X = [x_1]
        Y = [y_1]
        Z = [z_1]
        
        rk45 = RK45(lambda t, y : dstate_dt(t, y, T[-1]), t0=t_1, y0=state0, t_bound=t_2)

        # using RKF
        while rk45.status not in ['finished', 'failed']:
            rk45.step()
            
            X.append(rk45.y[0])
            Y.append(rk45.y[1])
            Z.append(rk45.y[2])
            T.append(rk45.t)

        xr2, yr2, zr2 = X[-1], Y[-1], Z[-1]

        # updated velocity estimate
        vx1 = vx1 + (x_2 - xr2) / delta_t
        vy1 = vy1 + (y_2 - yr2) / delta_t
        vz1 = vz1 + (z_2 - zr2) / delta_t
        
        iteration += 1

    state = [x_1, y_1, z_1, vx1, vy1, vz1]
    solution = FitResult()
    solution.state = state
    solution.epoch = start_obs.epoch
    solution.method = "herget"
    solution.niter = iteration
    solution.flag = 0  # Success flag
    solution.ndof = 0
    solution.csq = 0.0
    solution.cov = [0.01] * 36

    return [solution]

def herget_with_assist(observations, seq, tolerance, args, aux, max_iterations=100):
    obs = [observations[i] for i in seq[0]]
    obs_1 = obs[0]
    r_e_1 = obs_1.observer_position
    rho_hat_1 = obs_1.get_rho_hat() 
    A_1 = obs_1.get_a_vec()
    D_1 = obs_1.get_d_vec()
    print(D_1)
    print(A_1)
    print(rho_hat_1)


    rho_1 = 1 # this is the magnitude of rho, direction given by rho_hat, initial guess is 1au
    t_1 = obs_1.epoch
    r_1 = r_e_1 + rho_1*rho_hat_1
    obs_n = observations[-1]
    r_e_n = obs_n.observer_position
    rho_hat_n = obs_n.get_rho_hat()
    rho_n = 1 # this is the magnitude of rho, direction given by rho_hat, initial guess is 1au
    t_n = obs_n.epoch
    r_n = r_e_n + rho_n*rho_hat_n

    # Initial guess for velocity
    delta_t = t_n - t_1
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
    for i in range(10):
        vx1, vy1, vz1, pos = find_new_vel(ephem, t_1, t_n, x1, y1, z1, vx1, vy1, vz1, xn, yn, zn, change = 'x')
        vx1, vy1, vz1, pos = find_new_vel(ephem, t_1, t_n, x1, y1, z1, vx1, vy1, vz1, xn, yn, zn, change = 'y')
        vx1, vy1, vz1, pos = find_new_vel(ephem, t_1, t_n, x1, y1, z1, vx1, vy1, vz1, xn, yn, zn, change = 'z')
        print(pos)

    print([xn, yn, zn])
    print('velocity is now:', vx1, vy1, vz1)
    
    

def find_new_vel(ephem, t1, tn, x1, y1, z1, vx1, vy1, vz1, xn, yn, zn, change):
    sim = rebound.Simulation()
    sim.t = t1 - ephem.jd_ref
    sim.add(x = x1, y = y1, z = z1, vx = vx1, vy =vy1, vz = vz1)
    var = sim.add_variation(testparticle=0)
    #print(var.particles[0].jacobi_com)
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
    #print('this pos was', sim.particles[0].xyz)
    #print('new_pos should be' ,np.array(sim.particles[0].xyz) - diff*np.array(var.particles[0].xyz))
    return vx1, vy1, vz1, sim.particles[0].xyz

def find_mag_to_adjust(P, Q, R):
    mag = np.dot(R-Q, Q-P)/np.dot(R-Q, R-Q)
    return mag