import os
import numpy as np
import healpy as hp


__all__ = ['lowres_path','hires_path','debiasRADec','icrf2radec','radec2icrf']

cwd = os.path.dirname(os.path.realpath(__file__))
lowres_path = os.path.join(cwd,'lowres_data')
hires_path = os.path.join(cwd,'hires_data')

def debiasRADec(ra,dec,epoch,catalog,biasdf,J2000=2451545.0,nside=256):

    """Astrometric catalog bias correction following Eggl et al. (2020).
    
    Parameters:
    -----------
    ra      ... [rad] Right Ascension
    dec     ... [rad] Declination
    epoch   ... [JD]  epoch of observation (Julian Date)
    catalog ... [string] MPC catalog identifier (one letter)
    biasdf  ... [pandas] pandas DataFrame containing debiasing information
    J2000   ... [JD] J2k epoch
    nside   ... [int] nside resolution for healpix tesselation (nside = 2^N)
    
    Dependencies:
    -------------
    numpy, healpy, pandas
    
    Returns:
    --------
    ra_deb  ... [rad] debiased Right Ascension at epoch
    dec_deb ... [rad] debiased declination at epoch
    """
    # arcseconds to radians
    as2rad = 1/3600*np.pi/180
    # find pixel from RADEC
    idx = hp.ang2pix(nside, np.rad2deg(ra), np.rad2deg(dec), nest=False, lonlat=True)
    # find catalog data in pandas Data Frame
    colnames = [f'{catalog}_ra', f'{catalog}_dec', f'{catalog}_pm_ra', f'{catalog}_pm_dec']
    bias = biasdf[colnames].iloc[idx]
    # time from epoch in Julian years
    dt_jy = (epoch-J2000)/365.25
    # bias correction
    ddec = (bias[colnames[1]]+dt_jy*bias[colnames[3]]/1000)*as2rad
    dec_deb = dec-ddec
    dra = (bias[colnames[0]]+dt_jy*bias[colnames[2]]/1000)*as2rad / np.cos(dec)
    ra_deb = ra-dra
    # Quadrant correction
    xyz = radec2icrf(ra_deb, dec_deb, deg=False)
    ra_deb, dec_deb = icrf2radec(xyz[0], xyz[1], xyz[2], deg=False)
    return ra_deb, dec_deb


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

    pos = np.array([x,y,z])
    r = np.linalg.norm(pos,axis=0) if (pos.ndim>1) else np.linalg.norm(pos)
    xu = x/r
    yu = y/r
    zu = z/r
    phi = np.arctan2(yu,xu)
    delta = np.arcsin(zu)
    if(deg):
        ra = np.mod(np.rad2deg(phi)+360,360)
        dec = np.rad2deg(delta)
    else:
        ra = np.mod(phi+2*np.pi,2*np.pi)
        dec = delta
    return ra, dec


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

    if(deg):
        a = np.deg2rad(ra)
        d = np.deg2rad(dec)
    else:
        a = np.array(ra)
        d = np.array(dec)

    cosd = np.cos(d)
    x = cosd*np.cos(a)
    y = cosd*np.sin(a)
    z = np.sin(d)

    return np.array([x, y, z])
