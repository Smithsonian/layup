import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from layup.utilities.layup_configs import LayupConfigs
from layup.utilities.bootstrap_utilties.download_utilities import make_retriever

from sorcha.ephemeris.simulation_geometry import equatorial_to_ecliptic
from sorcha.ephemeris.simulation_constants import ECL_TO_EQ_ROTATION_MATRIX, EQ_TO_ECL_ROTATION_MATRIX
from sorcha.ephemeris.orbit_conversion_utilities import universal_cartesian

from assist import Ephem

logger = logging.getLogger(__name__)

# --- orbit types and their columns ---
ORBIT_FORMAT = Literal["BCART", "BCOM", "BKEP", "CART", "COM", "KEP"]
REQUIRED_COLUMN_NAMES: dict[str, list[str]] = {
    "BCART": ["ObjID", "FORMAT", "x", "y", "z", "xdot", "ydot", "zdot", "epochMJD_TDB"],
    "BCOM": ["ObjID", "FORMAT", "q", "e", "inc", "node", "argPeri", "t_p_MJD_TDB", "epochMJD_TDB"],
    "BKEP": ["ObjID", "FORMAT", "a", "e", "inc", "node", "argPeri", "ma", "epochMJD_TDB"],
    "CART": ["ObjID", "FORMAT", "x", "y", "z", "xdot", "ydot", "zdot", "epochMJD_TDB"],
    "COM": ["ObjID", "FORMAT", "q", "e", "inc", "node", "argPeri", "t_p_MJD_TDB", "epochMJD_TDB"],
    "KEP": ["ObjID", "FORMAT", "a", "e", "inc", "node", "argPeri", "ma", "epochMJD_TDB"],
}

PLANET_PERIOD_DAYS = {
    "Mercury": 87.969,
    "Venus": 224.701,
    "Earth": 365.256,
    "Mars": 686.980,
    "Jupiter": 4332.589,
    "Saturn": 10759.22,
    "Uranus": 30688.5,
    "Neptune": 60182.0
}

@dataclass(frozen=True)
class ClassicalConic:
    "Data class for each object conic section"

    obj_id: np.ndarray
    "object identifer tag"
    L: np.ndarray
    "object semilatus rectum (au)"
    e: np.ndarray
    "object eccentricity"
    inc: np.ndarray
    "object inclination (radians)"
    node: np.ndarray
    "object longitude of ascending node (radians)"
    argp: np.ndarray
    "object argument of perihelion (radians)"
    epochMJD_TDB: np.ndarray
    "epoch of object observation (MJD TDB)"


# --- converter ---
def convert_cart_to_classical_conic(
        rows: np.ndarray,
        mu: float
) -> ClassicalConic:
    """
    Convert cartesian elements into classical conic elements (L, e, i, Omega, omega)

    Parameters
    -----------
    rows : numpy structured array
        Array with all of the orbits (shape = (N,), i.e. one orbit per row/record)

    mu : float
        Standard gravitional parameter (au^3 / day^2)

    Returns
    --------
    ClassicalConic : object
        Object instance containing N orbits and their classical elements
    """
    # read each orbit id tag
    obj_id = rows["ObjID"].astype(str)

    # set up our position and velocity vectors
    r = np.vstack([rows["x"], rows["y"], rows["z"]]).T.astype(float)
    v = np.vstack([rows["xdot"], rows["ydot"], rows["zdot"]]).T.astype(float)

    # calculate specific angular momentum:
    # hvec = r x v
    # hnorm = |hvcec|
    hvec = np.cross(r, v)
    hnorm = np.linalg.norm(hvec, axis=1)

    # calculate inclination:
    # i = cos^-1(h_z / |h|)
    # notes: hvec[:, 2] = h_z, we use np.maximum and np.clip to avoid 
    # division by zero and floating point arithmetic giving answers that 
    # aren't in the range -1 <= h_z / |h| <= 1
    incl = np.arccos(np.clip(hvec[:, 2] / np.maximum(hnorm, 1e-30), -1.0, 1.0))

    # calculate the node vector:
    # nvec = zhat x hvec
    # nnorm = |nvec|
    zhat = np.array([0.0, 0.0, 1.0])
    nvec = np.cross(np.tile(zhat, (r.shape[0], 1)), hvec)
    nnorm = np.linalg.norm(nvec, axis=1)

    # calculate longitude of ascending node:
    # Omega = atan2(n_y, n_x)
    # notes: doing arctan2 rather than arccos just to skip having
    # to check quadrants when nhat dot jhat > or < 0, then wrapping
    # from what atan2 returns (-pi, pi] to [0, 2pi)
    node = np.arctan2(nvec[:, 1], nvec[:, 0]) % (2 * np.pi)

    # calculate eccentricity vector:
    # evec = (v x hvec)/mu - r/|r|
    # e = |e| <- this is eccentricity
    rnorm = np.linalg.norm(r, axis=1)
    evec = (np.cross(v, hvec) / mu) - (r / rnorm[:, None])
    e = np.linalg.norm(evec, axis=1)

    # calculate semilatus rectum:
    # L = |h|^2 / mu
    L = hnorm**2 / mu

    # np.maxmimum prevents us doing division by 0 for circular
    # orbits (e=0) or equatorial orbits (nnorm = 0)
    e = np.maximum(e, 1e-30)
    nnorm = np.maximum(nnorm, 1e-30)

    # calculate the argument of perihelion:
    # omega = atan2(sin(omega), cos(omega))
    # cos(omega) = (nvec dot evec) / (|n||e|)
    # sin(omega) = ((nvec x evec) dot hvec) / (|n||e||h|)
    # note: we're clipping again for the same reasons as above
    # (floating point arithmetic + division by zero), and we do 
    # np.sum as we have an (N, 3) list of orbits in e.g. nvec, so
    # doing np.dot would result in matrix multiplication (if pedro
    # reads this in future, yes np.einsum does this but i hate the 
    # notation :p). finally arctan2 once again solves quadrant issues
    cosw = np.sum(nvec * evec, axis=1) / (nnorm * e)
    cosw = np.clip(cosw, -1.0, 1.0)
    sinw = np.sum(np.cross(nvec, evec) * hvec, axis=1) / (nnorm * e * np.maximum(hnorm, 1e-30))
    argp = (np.arctan2(sinw, cosw) % (2 * np.pi))

    # finally check if orbit is sat in reference plane, if so we 
    # overwrrite Omega to be = 0, and even though omega is strictly
    # undefined in this case, perihelion direction isn't due to evec,
    # which we need to draw the orbit later, therefore we just recalc
    # it as the angle from the ref x axis to perihelion again via atan2
    equatorial = nnorm < 1e-12
    if np.any(equatorial):
        node[equatorial] = 0.0
        argp[equatorial] = (np.arctan2(evec[equatorial, 1], evec[equatorial, 0]) % (2 * np.pi))

    return ClassicalConic(obj_id=obj_id, L=L, e=e, inc=incl, node=node, argp=argp, epochMJD_TDB=rows["epochMJD_TDB"])


# --- frame swapping ---
def rv_to_cart(
        obj_id: np.ndarray,
        r: np.ndarray,
        v: np.ndarray,
        epochMJD_TDB: np.ndarray
) -> np.ndarray:
    """
    Wrapper function to create structured array of cartesian elements from position+velocity state vectors

    Parameters
    -----------
    obj_id : numpy string array 
        Object identifier tags

    r : numpy float array
        Object position state vector with shape (N, 3) (au)

    v : numpy float array
        Object velocity state vector with shape (N, 3) (au/day)

    epochMJD_TDB: numpy float array
        Object epoch with shape (N, 3) (MJD TDB)

    Returns
    --------
    out : numpy structured array
        Array containing the cartesian elements (x,y,z,vx,vy,vz) with shape (N,)
    """
    out = np.empty(r.shape[0], dtype=[("ObjID","U64"), ("x","f8"), ("y","f8"), ("z","f8"), ("xdot","f8"), ("ydot","f8"), ("zdot","f8"), ("epochMJD_TDB","f8")])
    out["ObjID"] = obj_id.astype("U64")    
    out["x"], out["y"], out["z"] = r[:,0], r[:,1], r[:,2]
    out["xdot"], out["ydot"], out["zdot"] = v[:,0], v[:,1], v[:,2]
    out["epochMJD_TDB"] = epochMJD_TDB.astype("f8")
    return out

def to_rv(
        rows: np.ndarray,
        fmt: ORBIT_FORMAT,
        mu_sun: float,
        mu_total: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert orbits into cartesian state vectors regardless of format utilising Sorcha functions.
    Cometary/keplerian formats are converted using universal_cartesian()

    Parameters
    -----------
    rows : numpy structured array
        Array with all of the orbits (shape = (N,), i.e. one orbit per row/record)

    fmt : str
        Format of the orbit. Must be one of "BCART", "BCOM", "BKEP", "CART", "COM", "KEP" 

    mu_sun : float
        Standard (heliocentric) gravitional parameter (au^3 / day^2)

    mu_total : float
        Standard (barycentric) gravitional parameter (au^3 / day^2)

    Returns
    --------
    r : numpy float array
        Object position state vector with shape (N, 3) (au)

    v : numpy float array
        Object velocity state vector with shape (N, 3) (au/day)
    """
    epochJD = rows["epochMJD_TDB"].astype(float) + 2400000.5

    # if it's cartesian we can just pass it straight through
    if fmt in ("CART", "BCART"):
        r = np.vstack([rows["x"], rows["y"], rows["z"]]).T.astype(float)
        v = np.vstack([rows["xdot"], rows["ydot"], rows["zdot"]]).T.astype(float)
        return r, v
    # if it's cometary however, we convert to cartesian for easy origin shifting
    if fmt in ("COM", "BCOM"):
        mu = mu_sun if fmt == "COM" else mu_total

        q = rows["q"].astype(float)
        e = rows["e"].astype(float)
        incl = np.deg2rad(rows["inc"].astype(float))
        node = np.deg2rad(rows["node"].astype(float))
        argp = np.deg2rad(rows["argPeri"].astype(float))
        tpJD = rows["t_p_MJD_TDB"].astype(float) + 2400000.5

        r = np.empty((rows.size, 3), dtype=float)
        v = np.empty((rows.size, 3), dtype=float)

        for i in range(rows.size):
            x, y, z, vx, vy, vz = universal_cartesian(
                mu, q[i], e[i], incl[i], node[i], argp[i], tpJD[i], epochJD[i]
            )
            r[i] = (x, y, z)
            v[i] = (vx, vy, vz)
        
        return r, v
    # same for keplerian orbits
    if fmt in ("KEP", "BKEP"):
        mu = mu_sun if fmt == "KEP" else mu_total

        a = rows["a"].astype(float)
        e = rows["e"].astype(float)
        incl = np.deg2rad(rows["inc"].astype(float))
        node = np.deg2rad(rows["node"].astype(float))
        argp = np.deg2rad(rows["argPeri"].astype(float))
        M = np.deg2rad(rows["ma"].astype(float))

        M_wrap = M.copy()
        idx = M_wrap > np.pi
        M_wrap[idx] -= 2*np.pi

        tpJD = epochJD - M_wrap * np.sqrt(a**3/mu)
        q = a * (1.0 - e)

        r = np.empty((rows.size, 3), dtype=float)
        v = np.empty((rows.size, 3), dtype=float)

        for i in range(rows.size):
            x, y, z, vx, vy, vz = universal_cartesian(
                mu, q[i], e[i], incl[i], node[i], argp[i], tpJD[i], epochJD[i]
            )
            r[i] = (x, y, z)
            v[i] = (vx, vy, vz)
        
        return r, v
    
    logger.error(f"Unsupported format: {fmt}")
    raise ValueError(f"Unsupported format: {fmt}")

def build_ephem_and_mus(cache_dir: Optional[str] = None) -> tuple[Ephem, float, float]:
    """
    Create Assist instance utilising Layup functions in order to find standard gravitationl parameters 
    
    Parameters
    -----------
    cache_dir : str, optional
        Cache directory containing Assist+Rebound files if used

    Returns
    --------
    ephem : Assist object
        Assist instance containing the Sun, planets, and massive perturbers

    mu_sun : float
        Standard (heliocentric) gravitional parameter (au^3 / day^2)

    mu_total : float
        Standard (barycentric) gravitional parameter (au^3 / day^2)
    """
    # yoink layup's auxiliary config class to locate the user's cached files 
    configs = LayupConfigs()
    aux = configs.auxiliary

    retriever = make_retriever(aux, cache_dir)
    ephem = Ephem(planets_path=retriever.fetch(aux.jpl_planets), asteroids_path=retriever.fetch(aux.jpl_small_bodies))
    
    # calculate mu same way as in sorcha
    mu_sun = ephem.get_particle("Sun", 0).m
    mu_total = sum(sorted([ephem.get_particle(i,0).m for i in range(27)]))

    return ephem, mu_sun, mu_total

def convert_sun_to_baryecliptic(
        ephem: Ephem,
        epochJD: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to create translation vectors to go from barycentric to heliocentric origins
    
    Parameters
    -----------
    ephem : assist Ephem object
        Assist built ephemeris of the Sun, planets, and massive perturbers

    epochJD : numpy float array
        Array of input object epochs in Julian dates of shape (N,)

    Returns
    --------
    S_r : numpy float array
        Translation barycentric -> heliocentric position vector of shape (N,3)

    S_v : numpy float array
        Translation barycentric -> heliocentric velocity vector of shape (N,3)
    """
    # we'll build a cache of the sun's state vectors in all four combinations here
    cache: dict[float, object] = {}
    S_r = np.empty((epochJD.size, 3), dtype=float)
    S_v = np.empty((epochJD.size, 3), dtype=float)

    # we're just using assist here to get the sun's position at each epoch of our objects
    for i, t in enumerate(epochJD):
        t_key = float(t)
        if t_key not in cache:
            cache[t_key] = ephem.get_particle("Sun", t_key - ephem.jd_ref)
        sun = cache[t_key]

        # quick convert to ecliptic as assist is in equatorial
        S_r[i] = -equatorial_to_ecliptic([sun.x, sun.y, sun.z])
        S_v[i] = -equatorial_to_ecliptic([sun.vx, sun.vy, sun.vz])

    return S_r, S_v

# --- build all variants ---
def prepopulate_orbit_variants(
        rows: np.ndarray,
        orbit_format: ORBIT_FORMAT,
        input_plane: Literal["equatorial", "ecliptic"],
        input_origin: Literal["heliocentric", "barycentric"]
) -> tuple[
    dict[tuple[str, str], ClassicalConic],
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], np.ndarray]
]:
    """
    Create an output cache of input object class instances, an empty placeholder dict for their 
    associated orbit lines, the Sun's positions, and the object's positions in all four combinations 
    of plane+origin

    Parameters
    -----------
    rows : numpy structured array
        Array with all of the orbits (shape = (N,), i.e. one orbit per row/record)

    orbit_format : str
        String detailing the input orbit format. Must be one of "BCART", "BCOM", "BKEP", "CART", "COM", "KEP" 

    input_plane : str
        Input reference plane of the orbits. Must be one of "equatorial" or "ecliptic"

    input_origin : str
        Input origin of the orbits. Must be one of "heliocentric" or "barycentric"
    
    Returns
    --------
    canon_cache : dict of objects
        Dictionary with the conic section class instances of each object and their properties
    
    lines_cache : dict of numpy array
        Dictionary of arrays with the orbit lines for each object in each plane+origin combination

    sunpos_cache : dict of numpy array
        Dictionary of arrays with the Sun positions in each plane+origin combination

    pos_cache : dict of numpy array
        Dictionary of arrays with object positions in each plane+origin combination
    """
    # grab our planets+major perturbers and solar gravitational parameters
    # # (mu_sun = heliocentric, mu_total = barycentric) 
    ephem, mu_sun, mu_total = build_ephem_and_mus()
    
    obj_id = rows["ObjID"].astype(str)
    epochJD = rows["epochMJD_TDB"].astype(float) + 2400000.5

    # grab state vectors
    r_raw, v_raw = to_rv(rows, orbit_format, mu_sun, mu_total)
    S_r, S_v = convert_sun_to_baryecliptic(ephem, epochJD)

    # everything is easier if we start from one frame+origin combo, so let's choose
    # heliocentric ecliptic and convert equa -> ecl first to this if needs be
    if input_plane == "equatorial":
        r_raw = np.dot(r_raw, EQ_TO_ECL_ROTATION_MATRIX)
        v_raw = np.dot(v_raw, EQ_TO_ECL_ROTATION_MATRIX)
    
    # again, easier if it's heliocentric ecliptic, so convert bary -> helio
    r_helio_eclipt = r_raw.copy()
    v_helio_eclipt = v_raw.copy()
    if input_origin == "barycentric":
        r_helio_eclipt = r_helio_eclipt + S_r
        v_helio_eclipt = v_helio_eclipt + S_v

    # now we can find heliocentric equatorial via obliquity rotation matrix
    r_helio_equa = np.dot(r_helio_eclipt, ECL_TO_EQ_ROTATION_MATRIX)
    v_helio_equa = np.dot(v_helio_eclipt, ECL_TO_EQ_ROTATION_MATRIX)

    # barycentric ecliptic is just a subtraction of the sun's barycentre state vector
    r_bary_eclipt = r_helio_eclipt - S_r
    v_bary_eclipt = v_helio_eclipt - S_v

    # and barycentric equatorial is another obliquity rotation matrix
    r_bary_equa = np.dot(r_bary_eclipt, ECL_TO_EQ_ROTATION_MATRIX)
    v_bary_equa = np.dot(v_bary_eclipt, ECL_TO_EQ_ROTATION_MATRIX)

    # now we build up our cache of different orbit lines
    canon_cache: dict[tuple[str, str], ClassicalConic] = {}
    lines_cache: dict[tuple[str, str], np.ndarray] = {}

    variants = {
        ("helio", "ecl"): (r_helio_eclipt, v_helio_eclipt, mu_sun),
        ("helio", "equ"): (r_helio_equa, v_helio_equa, mu_sun),
        ("bary", "ecl"): (r_bary_eclipt, v_bary_eclipt, mu_total),
        ("bary", "equ"): (r_bary_equa, v_bary_equa, mu_total),
    }

    # these are for the scatter marker positions of the sun and the objects themselves
    sunpos_cache = {
        ("helio", "ecl"): np.array([0.0, 0.0, 0.0]),
        ("helio", "equ"): np.array([0.0, 0.0, 0.0]),
        ("bary", "ecl"): (-S_r[0]),
        ("bary", "equ"): (np.dot(-S_r[0], ECL_TO_EQ_ROTATION_MATRIX)),
    }

    pos_cache: dict[tuple[str, str], np.ndarray] = {}

    for key, (r_i, v_i, mu_i) in variants.items():
        pos_cache[key] = r_i
        cart_rows = rv_to_cart(obj_id, r_i, v_i, rows["epochMJD_TDB"].astype(float))
        canon_cache[key] = convert_cart_to_classical_conic(cart_rows, mu_i)

    return canon_cache, lines_cache, sunpos_cache, pos_cache


# --- orbit line generation ---
def conic_lines_from_classical_conic(
        canon: ClassicalConic,
        n_points: int = 200,
        r_max: float = 50.0
) -> np.ndarray:
    """
    Given a set of classical conic elements (L, e, i, Omega, omega), draw the line of the orbit via
    conic section (see page 27 onwards in "Solar System Dynamics" by Murray & Dermott)

    Parameters
    -----------
    canon : object
        Object instance containing N orbits and their classical elements

    n_points : int, optional (default=200)
        Number of points to use to construct the line

    r_max : float, optional (default=50.0 au)
        Maximum distance to render out to for hyperbolic orbits

    Returns
    --------
    r : numpy float array
        Object position state vector with shape (N, n_points, 3) (au)
    """
    # unpack classical elements for readability
    e = canon.e
    L = canon.L

    # set up true anomaly array
    nu = np.empty((e.size, n_points))

    # which orbits are elliptical or hyperbolic (and near-parabolic)
    ell = e < 1.0
    hyp = ~ell

    # if elliptical, we just draw it from -pi to pi so it's centered on 
    # perihelion rather than 0 to 2pi
    if np.any(ell):
        nu[ell] = np.linspace(-np.pi, np.pi, n_points)

    # if hyperbolic, we want to draw the orbit out to some max position
    # r_max, so our maximum angular extent is calculated as:
    # nu_max = arccos( (L/r_max - 1)/e )
    # then we make our nu array by just making it as [-nu_max, +nu_max]
    if np.any(hyp):
        c = (L[hyp] / r_max - 1.0) / np.maximum(e[hyp], 1e-15)
        c = np.clip(c, -1.0, 1.0)
        nu_max = np.arccos(c)
        t = np.linspace(-1.0, 1.0, n_points)
        nu[hyp] = nu_max[:, None] * t

    # now we calculate radial distance of the conic section via the polar
    # coordinates equation:
    # r = L / (1 + e cos(nu))
    # note: as always we are preventing division by zero via np.maximum
    rdist = L[:, None] / np.maximum(1.0 + e[:, None] * np.cos(nu), 1e-12)

    # now in the perifocal frame (ie origin at focus, q1 towards perihelion),
    # the orbit will be in the q1q2 plane:
    # qvec = (q1, q2, q3) = (r cos(nu), r sin(nu), 0)
    q1 = rdist * np.cos(nu)
    q2 = rdist * np.sin(nu)
    q3 = np.zeros_like(q1)

    # and now we rotate back into inertial frame via rotation matrix:
    # R = R_z(Omega)R_x(i)R_z(omega)
    c0, s0 = np.cos(canon.node), np.sin(canon.node)
    co, so = np.cos(canon.argp), np.sin(canon.argp)
    ci, si = np.cos(canon.inc), np.sin(canon.inc)

    # we now find the position vector rvec by:
    # rvec = (x, y, z) = R qvec
    x = (c0*co - s0*so*ci)[:, None]*q1 + (-c0*so - s0*co*ci)[:, None]*q2
    y = (s0*co + c0*so*ci)[:, None]*q1 + (-s0*so + c0*co*ci)[:, None]*q2
    z = (so*si)[:, None]*q1 + (co*si)[:, None]*q2
    r = np.stack([x, y, z], axis=-1)

    return r

def build_planet_lines_cache(
        ephem: Ephem,
        epochJD_center: float,
        planet_names: list[str],
        n_points: int = 800
) -> tuple[dict[tuple[str, str], np.ndarray], np.ndarray]:
    """
    
    Parameters
    -----------
    ephem : Assist object
        Assist instance containing the Sun, planets, and massive perturbers

    epochJD_center : float
        Reference epoch to use to sample ellipse symmetrically over one orbtial period (JD)
    
    planet_names : list
        List of planet names as strings

    n_points : int, optional (default=800)
        Number of points to use to construct the line

    Returns 
    --------
    planet_lines_cache : dict of arrays
        Dictiionary of arrays containing planet orbit lines for each planet in each plane+origin 
        combination of shape (n_planets, n_points, 3)
    
    planet_id : numpy string array
        Array containing ID tags for each planet of shape (n_planets,)
    """
    # much like the object orbits, it will be easier if we start from some frame+origin and
    # convert to the other 3 from there. we choose barycentric equatorial here 
    planet_id = np.array(planet_names, dtype="U32")
    lines_bary_equ = np.empty((len(planet_names), n_points, 3), dtype=float)

    for p, name in enumerate(planet_names):
        P = PLANET_PERIOD_DAYS[name]
        if P is None:
            logger.error(f"Missing period for planet: '{name}' in PLANET_PERIOD_DAYS")
            raise KeyError(f"Missing period for planet: '{name}' in PLANET_PERIOD_DAYS")
        
        # basically build an ellipse of times to get the planet positions at where the epochJD_center
        # is a reference epoch, which should just be the epoch of the first input object (no scatter
        # points so really just need the shape of orbit) 
        epochJD = np.linspace(epochJD_center - float(P)/2.0, epochJD_center + float(P)/2.0, n_points)

        for i, jd in enumerate(epochJD):
            part = ephem.get_particle(name, float(jd) - ephem.jd_ref)
            lines_bary_equ[p, i, :] = (part.x, part.y, part.z)

    # this looks weird but it's just being fancy and instead of looping over it all we reshape 
    # lines_bary_equ from (N_planets, N_points, 3) -> (N_planets*N_points, 3) (ie one xyz per row)
    # so we can apply rotation matrix quickly and easily. then we do the inverse reshape back 
    lines_bary_ecl = np.dot(lines_bary_equ.reshape(-1, 3), EQ_TO_ECL_ROTATION_MATRIX).reshape(lines_bary_equ.shape)

    # because the sun has different positions at each planet's epochs, we need to redo S_r here
    S_r_ecl = np.empty_like(lines_bary_ecl)
    S_r_equ = np.empty_like(lines_bary_ecl)

    for p, name in enumerate(planet_names):
        P = PLANET_PERIOD_DAYS[name]
        epochJD = np.linspace(epochJD_center - float(P)/2.0, epochJD_center + float(P)/2.0, n_points)

        S_r, _ = convert_sun_to_baryecliptic(ephem, epochJD)
        S_r_ecl[p, :, :] = S_r
        S_r_equ[p, : ,:] = np.dot(S_r, ECL_TO_EQ_ROTATION_MATRIX)

    # now we can simply build our cache of planets
    planet_lines_cache: dict[tuple[str, str], np.ndarray] = {}

    planet_lines_cache[("bary", "equ")] = lines_bary_equ
    planet_lines_cache[("bary", "ecl")] = lines_bary_ecl

    planet_lines_cache[("helio", "equ")] = lines_bary_equ + S_r_equ
    planet_lines_cache[("helio", "ecl")] = lines_bary_ecl + S_r_ecl

    return planet_lines_cache, planet_id
