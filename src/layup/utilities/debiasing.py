from itertools import product
from pathlib import Path

import numpy as np
import healpy as hp
import pandas as pd
import spiceypy as spice

# From Siegfried Eggl's code
mpc_catalogs = {
    "a": "USNOA1",
    "b": "USNOSA1",
    "c": "USNOA2",
    "d": "USNOSA2",
    "e": "UCAC1",
    "g": "Tyc2",
    "i": "GSC1.1",
    "j": "GSC1.2",
    "l": "ACT",
    "m": "GSCACT",
    "n": "SDSS8",
    "o": "USNOB1",
    "p": "PPM",
    "q": "UCAC4",
    "r": "UCAC2",
    "t": "PPMXL",
    "u": "UCAC3",
    "v": "NOMAD",
    "w": "CMC14",
    "L": "2MASS",
    "N": "SDSS7",
    "Q": "CMC15",
    "R": "SSTRC4",
    "S": "URAT1",
    "U": "Gaia1",
    "W": "Gaia3",
}
coords = ["ra", "dec", "pm_ra", "pm_dec"]

columns = ["_".join(pair) for pair in product(mpc_catalogs.keys(), coords)]


def generate_bias_dict(cache_dir=None):

    #! We need to include this in the bootstrap files
    #! curl ftp://ssd.jpl.nasa.gov/pub/ssd/debias/debias_hires2018.tgz --output debias_hires2018.tgz
    #! tar -xvzf debias_hires2018.tgz
    biasdf = pd.read_csv(Path(cache_dir) / "bias.dat", sep="\\s+", skiprows=23, names=columns)

    # Turn this into a dictionary for speed
    bias_dict = {}
    for catalog in mpc_catalogs:
        colnames = [f"{catalog}_ra", f"{catalog}_dec", f"{catalog}_pm_ra", f"{catalog}_pm_dec"]
        bias_dict[catalog] = {}
        for col in colnames:
            short_col = col[2:]
            bias_dict[catalog][short_col] = biasdf[col].values
    return bias_dict


def debias(ra, dec, epoch, catalog, bias_dict, nside=256):

    if catalog not in bias_dict:
        return ra, dec

    # find pixel from RADEC
    idx = hp.ang2pix(nside, ra, dec, nest=False, lonlat=True)

    ra_off = bias_dict[catalog]["ra"][idx]
    pm_ra = bias_dict[catalog]["pm_ra"][idx]
    dec_off = bias_dict[catalog]["dec"][idx]
    pm_dec = bias_dict[catalog]["pm_dec"][idx]

    # time from epoch in Julian years
    dt_jy = (epoch - spice.j2000()) / 365.25

    # bias correction
    ddec = dec_off + dt_jy * pm_dec / 1000
    dec_deb = dec - ddec / 3600.0
    dra = (ra_off + dt_jy * pm_ra / 1000) / np.cos(np.deg2rad(dec))
    ra_deb = ra - dra / 3600.0

    # Quadrant correction
    xyz = radec2icrf(ra_deb, dec_deb, deg=True)
    ra_deb, dec_deb = icrf2radec(xyz[0], xyz[1], xyz[2], deg=True)

    return ra_deb, dec_deb


# From Siegfried Eggl's code
def radec2icrf(ra, dec, deg=True):
    """Convert Right Ascension and Declination to ICRF xyz unit vector.
    Geometric states on unit sphere, no light travel time/aberration correction.
    Parameters:
    -----------
    ra ... Right Ascension [deg]
    dec ... Declination [deg]
    deg ... True: angles in degrees, False: angles in radians
    Returns:
    --------
    x,y,z ... 3D vector of unit length (ICRF)
    """

    if deg:
        a = np.deg2rad(ra)
        d = np.deg2rad(dec)
    else:
        a = np.array(ra)
        d = np.array(dec)

    cosd = np.cos(d)
    x = cosd * np.cos(a)
    y = cosd * np.sin(a)
    z = np.sin(d)

    return np.array([x, y, z])


# From Siegfried Eggl's code
def icrf2radec(x, y, z, deg=True):
    """Convert ICRF xyz to Right Ascension and Declination.
    Geometric states on unit sphere, no light travel time/aberration correction.
    Parameters:
    -----------
    x,y,z ... 3D vector of unit length (ICRF)
    deg ... True: angles in degrees, False: angles in radians
    Returns:
    --------
    ra ... Right Ascension [deg]
    dec ... Declination [deg]
    """

    pos = np.array([x, y, z])
    r = np.linalg.norm(pos, axis=0) if (pos.ndim > 1) else np.linalg.norm(pos)
    xu = x / r
    yu = y / r
    zu = z / r
    phi = np.arctan2(yu, xu)
    delta = np.arcsin(zu)
    if deg:
        ra = np.mod(np.rad2deg(phi) + 360, 360)
        dec = np.rad2deg(delta)
    else:
        ra = np.mod(phi + 2 * np.pi, 2 * np.pi)
        dec = delta
    return ra, dec
