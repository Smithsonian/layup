from itertools import product
from pathlib import Path

import numpy as np
import healpy as hp
import pandas as pd
import pooch

# From Siegfried Eggl's code
MPC_CATALOGS = {
    "USNOA1": "a",
    "USNOSA1": "b",
    "USNOA2": "c",
    "USNOSA2": "d",
    "UCAC1": "e",
    "Tyc2": "g",
    "GSC1.1": "i",
    "GSC1.2": "j",
    "ACT": "l",
    "GSCACT": "m",
    "SDSS8": "n",
    "USNOB1": "o",
    "PPM": "p",
    "UCAC4": "q",
    "UCAC2": "r",
    "PPMXL": "t",
    "UCAC3": "u",
    "NOMAD": "v",
    "CMC14": "w",
    "2MASS": "L",
    "SDSS7": "N",
    "CMC15": "Q",
    "SSTRC4": "R",
    "URAT1": "S",
    "Gaia1": "U",
    "Gaia3": "W",
}
COORDS = ["ra", "dec", "pm_ra", "pm_dec"]

COLUMNS = ["_".join(pair) for pair in product(MPC_CATALOGS.values(), COORDS)]


def generate_bias_dict(cache_dir=None):
    """Convert the bias.dat file into a dictionary for fast access.

    Parameters
    ----------
    cache_dir : str, optional
        Directory containing cache files for layup, by default None

    Returns
    -------
    dict
        Dictionary representation of bias.dat file indexed by catalog keys.
    """
    if cache_dir is None:
        cache_dir = pooch.os_cache("layup")

    bias_file_path = Path(cache_dir) / "bias.dat"
    if not bias_file_path.exists():
        raise FileNotFoundError(f"The bias.dat file was not found in the cache directory: {cache_dir}")

    # Read in the bias.dat file as a pandas DataFrame
    biasdf = pd.read_csv(Path(cache_dir) / "bias.dat", sep="\\s+", skiprows=23, names=COLUMNS)

    # Turn this into a dictionary for speed
    bias_dict = {}
    for catalog in MPC_CATALOGS.values():
        colnames = [f"{catalog}_ra", f"{catalog}_dec", f"{catalog}_pm_ra", f"{catalog}_pm_dec"]
        bias_dict[catalog] = {}
        for col in colnames:
            short_col = col[2:]
            bias_dict[catalog][short_col] = biasdf[col].values
    return bias_dict


def debias(ra, dec, epoch_jd_tdb, catalog, bias_dict, nside=256):

    catalog_key = MPC_CATALOGS[catalog]
    if catalog_key not in bias_dict.keys():
        return ra, dec

    # find pixel from RADEC
    idx = hp.ang2pix(nside, ra, dec, nest=False, lonlat=True)

    # Retrieve the offsets and proper motions from the bias dictionary
    ra_off = bias_dict[catalog_key]["ra"][idx]
    pm_ra = bias_dict[catalog_key]["pm_ra"][idx]
    dec_off = bias_dict[catalog_key]["dec"][idx]
    pm_dec = bias_dict[catalog_key]["pm_dec"][idx]

    # time from epoch in Julian years
    dt_jy = epoch_jd_tdb / 365.25

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
