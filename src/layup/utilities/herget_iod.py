import numpy as np
from _layup_cpp._core import FitResult



    # the differential equation for orbital motion probaily, probaily wrong 
def f(state, t):
    """
    Differential equation for two-body problem.
    state = [x, y, z, vx, vy, vz]
    """
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)
    # Gravitational parameter for the Sun in AU^3/day^2
    k = 0.01720209895
    mu = k**2  # = 2.959122082855911e-4 AU^3/day^2

    ax = -mu * x / r**3
    ay = -mu * y / r**3
    az = -mu * z / r**3

    return np.array([vx, vy, vz, ax, ay, az])


# this is very thrown together so youll have to check and mess with everything. 
def herget(start_obs, end_obs, tolerance, max_iterations=1000):
    """
    random attempt  at herget's method for initial orbit determination. not right
    """
    x1, y1, z1 = start_obs.observer_position
    x2, y2, z2 = end_obs.observer_position

    # Initial guess and velocities
    xr2, yr2, zr2 = x2 + 100, y2 + 100, z2 + 100
    delta_t = end_obs.epoch - start_obs.epoch
    vx1 = (x2 - x1) / delta_t
    vy1 = (y2 - y1) / delta_t
    vz1 = (z2 - z1) / delta_t

    iteration = 0



    while abs(xr2 - x2) > tolerance or abs(yr2 - y2) > tolerance or abs(zr2 - z2) > tolerance:

        if iteration >= max_iterations:  # failed sol
            state = [x1, y1, z1, vx1, vy1, vz1]
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

        # initial state vector
        state0 = np.array([x1, y1, z1, vx1, vy1, vz1])

        # using RKF
        T, X = rkf(f, 0, delta_t, state0, tol=1e-6, hmax=delta_t / 10, hmin=1e-10)

        # final position
        final_state = X[-1]
        xr2, yr2, zr2 = final_state[0:3]

        # updated velocity estimate
        vx1 = vx1 - (xr2 - x2) / delta_t
        vy1 = vy1 - (yr2 - y2) / delta_t
        vz1 = vz1 - (zr2 - z2) / delta_t

        iteration += 1

    state = [x1, y1, z1, vx1, vy1, vz1]
    solution = FitResult()
    solution.state = state
    solution.epoch = start_obs.epoch
    solution.method = "herget"
    solution.niter = iteration
    solution.flag = 0  # Success flag
    solution.ndof = 0
    solution.csq = 0.0
    solution.cov = [0.0] * 36

    return [solution]


def rkf(f, a, b, x0, tol, hmax, hmin): # tried to do this, again not very confident i have it working properly 
    """
    Runge-Kutta-Fehlberg method (RKF45). thank you stack overflow for most of this code.

    Parameters:
    -----------
    f : function, derivative function f(x, t)
    a : float, initial time
    b : float, final time
    x0 : array, initial state
    tol : float, tolerance
    hmax : float, maximum step size
    hmin : float, minimum step size

    Returns:
    --------
    T : array, time values
    X : array, state values
    """
    # RKF45 coefficients- there is definately a better way than this but im lazy 
    a2 = 2.500000000000000e-01
    a3 = 3.750000000000000e-01
    a4 = 9.230769230769231e-01
    a5 = 1.000000000000000e00
    a6 = 5.000000000000000e-01

    b21 = 2.500000000000000e-01
    b31 = 9.375000000000000e-02
    b32 = 2.812500000000000e-01
    b41 = 8.793809740555303e-01
    b42 = -3.277196176604461e00
    b43 = 3.320892125625853e00
    b51 = 2.032407407407407e00
    b52 = -8.000000000000000e00
    b53 = 7.173489278752436e00
    b54 = -2.058966861598441e-01
    b61 = -2.962962962962963e-01
    b62 = 2.000000000000000e00
    b63 = -1.381676413255361e00
    b64 = 4.529727095516569e-01
    b65 = -2.750000000000000e-01

    r1 = 2.777777777777778e-03
    r3 = -2.994152046783626e-02
    r4 = -2.919989367357789e-02
    r5 = 2.000000000000000e-02
    r6 = 3.636363636363636e-02

    c1 = 1.157407407407407e-01
    c3 = 5.489278752436647e-01
    c4 = 5.353313840155945e-01
    c5 = -2.000000000000000e-01

    t = a
    x = np.array(x0, dtype=float)
    h = hmax

    T = [t]
    X = [x.copy()]

    while t < b:
        if t + h > b:
            h = b - t

        k1 = h * f(x, t)
        k2 = h * f(x + b21 * k1, t + a2 * h)
        k3 = h * f(x + b31 * k1 + b32 * k2, t + a3 * h)
        k4 = h * f(x + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * h)
        k5 = h * f(x + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * h)
        k6 = h * f(x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, t + a6 * h)

        r = abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6) / h

        if len(np.shape(r)) > 0:
            r = max(r)

        if r <= tol:
            t = t + h
            x = x + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
            T.append(t)
            X.append(x.copy())

        h = h * min(max(0.84 * (tol / r) ** 0.25, 0.1), 4.0)

        if h > hmax:
            h = hmax
        elif h < hmin:
            raise RuntimeError(
                f"Error: Could not converge to the required tolerance {tol} " f"with minimum stepsize {hmin}."
            )

    return np.array(T), np.array(X)




