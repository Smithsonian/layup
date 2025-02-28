#!/usr/bin/env python

"""Read and format obs80 data for orbit fitting.
   Add astrometric uncertainties based on Veres et al 2017
   Do the catalog debiasing according to Eggl et al ...
"""

# Import standard packages
import os
import sys

# Import third-party packages
import numpy as np
import spiceypy as spice
import pandas as pd
import healpy as hp

import MPC_library # for a small number of routines
import tracklets as tr

first=lambda x: x[0]
second=lambda x: x[1]
third=lambda x: x[2]
fourth=lambda x: x[3]

au2m = 149597870700
au_km = au2m/1000.

early_703 = 2456658.5
early_691 = 2452640.5
early_644 = 2452883.5

cat_codes = {
    'a' : 'USNO-A1.0',
    'b' : 'USNO-SA1.0',
    'c' : 'USNO-A2.0',
    'd' : 'USNO-SA2.0',
    'e' : 'UCAC-1',
    'f' : 'Tycho-1',
    'g' : 'Tycho-2',
    'h' : 'GSC-1.0',
    'i' : 'GSC-1.1',
    'j' : 'GSC-1.2',
    'k' : 'GSC-2.2',
    'l' : 'ACT',
    'm' : 'GSC-ACT',
    'n' : 'SDSS-DR8',
    'o' : 'USNO-B1.0',
    'p' : 'PPM',
    'q' : 'UCAC-4',
    'r' : 'UCAC-2',
    's' : 'USNO-B2.0',
    't' : 'PPMXL',
    'u' : 'UCAC-3',
    'v' : 'NOMAD',
    'w' : 'CMC-14',
    'x' : 'Hipparcos 2',
    'y' : 'Hipparcos',
    'z' : 'GSC (version unspecified)',
    'A' : 'AC',
    'B' : 'SAO 1984',
    'C' : 'SAO',
    'D' : 'AGK 3',
    'E' : 'FK4',
    'F' : 'ACRS',
    'G' : 'Lick Gaspra Catalogue',
    'H' : 'Ida93 Catalogue',
    'I' : 'Perth 70',
    'J' : 'COSMOS/UKST Southern Sky Catalogue',
    'K' : 'Yale',
    'L' : '2MASS',
    'M' : 'GSC-2.3',
    'N' : 'SDSS-DR7',
    'O' : 'SST-RC1',
    'P' : 'MPOSC3',
    'Q' : 'CMC-15',
    'R' : 'SST-RC4',
    'S' : 'URAT-1',
    'T' : 'URAT-2',
    'U' : 'Gaia-DR1',
    'V' : 'Gaia-DR2',
    'W' : 'Gaia-DR3',
    'X' : 'Gaia-EDR3',
    'Y' : 'UCAC-5',
    'Z' : 'ATLAS-2',
    '0' : 'IHW',
    '1' : 'PS1-DR1',
    '2' : 'PS1-DR2',
    '3' : 'Gaia_Int  ',
    '4' : 'GZ',
    '5' : 'USNO-UBAD',
    '6' : 'Gaia2016'
    }
    

def data_weight_Veres2017(obsCode, jd_tdb, catalog=None):

    # Need to put in the photographic and other observation
    # types

    dw = 1.5
    if obsCode == '703':
        if jd_tdb<=early_703:
            dw = 1.0
        else:
            dw = 0.8
    elif obsCode == '691':
        if jd_tdb<=early_691:
            dw = 0.6
        else:
            dw = 0.5
    elif obsCode == '644':
        if jd_tdb<=early_644:
            dw = 0.6
        else:
            dw = 0.4
    elif obsCode == '704':
        dw = 1.0
    elif obsCode == 'G96':
        dw = 0.5
    elif obsCode == 'F51':
        dw = 0.2
    elif obsCode == 'G45':
        dw = 0.6        
    elif obsCode == '699':
        dw = 0.8        
    elif obsCode == 'D29':
        dw = 0.75        
    elif obsCode == 'C51':
        dw = 1.0
    elif obsCode == 'E12':
        dw = 0.75
    elif obsCode == '608':
        dw = 0.6       
    elif obsCode == 'J75':
        dw = 1.0        
    elif obsCode == '645':
        dw = 0.3
    elif obsCode == '673':
        dw = 0.3
    elif obsCode == '689':
        dw = 0.5
    elif obsCode == '950':
        dw = 0.5
    elif obsCode == 'H01':
        dw = 0.3
    elif obsCode == 'J04':
        dw = 0.4
    elif obsCode == 'W84':
        dw = 0.5
    elif obsCode == '645':
        dw = 0.3
    elif obsCode == 'G83' and prg == '2':
        if catalog in cat_codes:
            if cat_codes[catalog] in ['UCAC-4', 'PPMXL']:
                dw = 0.3
            elif cat_codes[catalog] in ['Gaia-DR1', 'Gaia-DR2',  'Gaia-DR3', 'Gaia-EDR3']:
                dw = 0.2
            else:
                dw = 0.3
        else:
            dw = 1.0
    elif obsCode in ['K92', 'K93', 'Q63', 'Q64', 'V37', 'W84', 'W85', 'W86', 'W87', 'K91', 'E10', 'F65']:
        dw = 0.4 # Careful with W84
    elif obsCode == 'Y28':
        if catalog in cat_codes:
            if cat_codes[catalog] in ['PPMXL', 'Gaia-DR1']:
                dw = 0.3
            else:
                dw = 1.5
        else:
            dw = 1.5
    elif obsCode == '568':
        if catalog in cat_codes:        
            if cat_codes[catalog] in ['USNO-B1.0', 'USNO-B2.0']:
                dw = 0.5
            elif cat_codes[catalog] in ['Gaia-DR1', 'Gaia-DR2',  'Gaia-DR3', 'Gaia-EDR3']:
                dw = 0.1
            elif cat_codes[catalog] in ['PPMXL']:
                dw = 0.2            
            else:
                dw = 1.5
        else:
            dw = 1.5
    elif obsCode in ['T09', 'T12', 'T14']:
        if catalog in cat_codes:
            if cat_codes[catalog] in ['Gaia-DR1', 'Gaia-DR2',  'Gaia-DR3', 'Gaia-EDR3']:
                dw = 0.1
            else:
                dw = 1.5
        else:
            dw = 1.5            
    elif obsCode == '309' and prg == '&': # Micheli
        if cat_codes[catalog] in ['UCAC-4', 'PPMXL']:
            dw = 0.3
        elif cat_codes[catalog] in ['Gaia-DR1', 'Gaia-DR2',  'Gaia-DR3', 'Gaia-EDR3']:
            dw = 0.2
        else:
            dw = 1.5
    elif catalog:
        dw = 1.0
    else:
        dw = 1.5

    return dw

# From Siegfried Eggl's code
mpc_catalogs = ['a', 'b', 'c', 'd', 'e', 'g', 'i', 'j', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 't', 'u', 'v', 'w', 'L', 'N', 'Q', 'R', 'S', 'U', 'W']
columns = []

for cat in mpc_catalogs:
    columns.extend([cat+"_ra",cat+"_dec",cat+"_pm_ra",cat+'_pm_dec'])

biasdf=pd.read_csv('/Users/mholman/astrocat_debiasing/debias/hires_data/bias.dat',sep='\s+',skiprows=23,names=columns)


# Turn this into a dictionary for speed
bias_dict = {}
for catalog in mpc_catalogs:
    colnames = [f'{catalog}_ra', f'{catalog}_dec', f'{catalog}_pm_ra', f'{catalog}_pm_dec']
    bias_dict[catalog] = {}
    for col in colnames:
        short_col = col[2:]
        bias_dict[catalog][short_col] = biasdf[col].values


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
        

def debias(ra, dec, epoch, catalog, bias_dict, J2000=2451545.0, nside=256):        

    # arcseconds to radians
    as2rad = 1/3600*np.pi/180
    # find pixel from RADEC
    idx = hp.ang2pix(nside, ra, dec, nest=False, lonlat=True)

    ra_off = bias_dict[catalog]['ra'][idx]
    pm_ra = bias_dict[catalog]['pm_ra'][idx]
    dec_off = bias_dict[catalog]['dec'][idx]
    pm_dec = bias_dict[catalog]['pm_dec'][idx]

    # time from epoch in Julian years
    dt_jy = (epoch-J2000)/365.25
    # bias correction
    
    ddec = (dec_off + dt_jy*pm_dec/1000)
    dec_deb = dec-ddec/3600.
    dra = (ra_off + dt_jy*pm_ra/1000) / np.cos(np.deg2rad(dec))
    ra_deb = ra-dra/3600.

    # Quadrant correction
    xyz = radec2icrf(ra_deb, dec_deb, deg=True)
    ra_deb, dec_deb = icrf2radec(xyz[0], xyz[1], xyz[2], deg=True)

    return ra_deb, dec_deb
        

def format_astrometry_line(line, readfunc=tr.convertObs80, ecliptic=False, jd_tdb_min=-1e9, baryhelio='bary'):

    if line[14] in ['R', 'x', 'X', 'v', 'V']:
        return None

    objName, jd_tdb, raDeg, decDeg, mag, filt, RA_sig, Dec_sig, mag_sig, obsCode, prob = readfunc(line[0:80])

    cat = line[71]
    if cat == ' ':
        cat = None

    prg = line[13]
    if prg == ' ':
        prg = None

    dw = data_weight_Veres2017(obsCode, jd_tdb, cat)
    '''
    if cat in cat_codes:
        print(cat_codes[cat], obsCode, dw)
    else:
        print('no catalog', obsCode, 'blah', dw)
    '''

    if jd_tdb<jd_tdb_min:
        print('Too early')
        return None

    if cat and cat in bias_dict:
        ra_deb, dec_deb = debias(raDeg, decDeg, jd_tdb, cat, bias_dict)
        raDeg, decDeg = ra_deb, dec_deb

    xt = np.cos(decDeg*np.pi/180.)*np.cos(raDeg*np.pi/180.)
    yt = np.cos(decDeg*np.pi/180.)*np.sin(raDeg*np.pi/180.)  
    zt = np.sin(decDeg*np.pi/180.)

    if ecliptic:
        xt, yt, zt = equatorial_to_ecliptic(np.array((xt, yt, zt)))

    et = (jd_tdb-spice.j2000())*24*60*60

    if(len(line.strip())>80):
        obsCode_test, geoc_pos = tr.satellite_pos(line[80:])
        if obsCode_test != obsCode:
            print('obs codes not the same', 'x', obsCode_test, 'x', obsCode, 'x')
    else:
        geoc_pos = tr.geocentricObservatory(et, obsCode)

    # Get the barycentric position of Earth
    if baryhelio=='bary':
        pos, _= spice.spkpos('EARTH', et, 'J2000', 'NONE', 'SSB')
    elif baryhelio=='helio':
        pos, _= spice.spkpos('EARTH', et, 'J2000', 'NONE', 'SUN')

    observatory_pos = pos + geoc_pos
        
    observatory_pos /= au_km
                        
    if filt.isspace():
        filt = '-'
    if mag.isspace():
        mag = '----'

    if ecliptic:
        observatory_obs = equatorial_to_ecliptic(observatory_pos)

    xo, yo, zo = observatory_pos

    outstring = "%11s %4s %6s %s %13.6lf %13.10lf %13.10lf %13.10lf %15.11lf %15.11lf %15.11lf %6.3lf"% \
        (objName, obsCode, mag, filt, jd_tdb, xt, yt, zt, xo, yo, zo, dw)

    return outstring

                
def main(in_filename):


    merge_filename = in_filename.replace('.txt', '_merge.txt')
    tr.merge_MPC_file(in_filename, merge_filename)

    out_filename = in_filename.replace('.txt', '_out.txt')    

    with open(merge_filename) as file, open(out_filename, 'w') as out_file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            outstring = format_astrometry_line(line)
            if outstring != None:
                out_file.write(outstring+'\n')

    return

if __name__ == "__main__":

    if len(sys.argv)==2:    
        in_filename  = sys.argv[1]
    else:
        exit(-1)

    main(in_filename)

