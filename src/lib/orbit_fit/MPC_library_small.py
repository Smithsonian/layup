# -*- coding: utf-8 -*-
import math
import numpy as np
import scipy
from scipy import interpolate

class Constants:
    GMsun = 2.9591220828559115e-04
    GMEarth = 8.8862436e-10
    Rearth_km = 6378.1363
    au_km = 149597870.700 # This is now a definition 
    Rearth_AU = Rearth_km/au_km
    ecl = (84381.448*(1./3600)*np.pi/180.) # Obliquity of ecliptic at J2000 (From Horizons 12 April 2019)
    speed_of_light = 2.99792458e5 * 86400./au_km

# This rotates from equatorial to ecliptic
def rotate_matrix(ecl):
    ce = np.cos(ecl)
    se = np.sin(-ecl)
    rotmat = np.array([[1.0, 0.0, 0.0],
                       [0.0,  ce,  se],
                       [0.0, -se,  ce]])
    return rotmat

class Observatory:

    def __init__(self, oc_file='ObsCodes.txt'):

        self.observatoryPositionCache = {} # previously calculated positions to speed up the process

        # Convert ObsCodes.txt lines to geocentric x,y,z positions and
        # store them in a dictionary.  The keys are the observatory
        # code strings, and the values are (x,y,z) tuples.
        # Spacecraft and other moving observatories have (None,None,None)
        # as position.
        ObservatoryXYZ = {}
        with open(oc_file, 'r') as f:
            next(f)
            for line in f:
                code, longitude, rhocos, rhosin, Obsname = self.parseObsCode(line)
                if longitude and rhocos and rhosin:
                    rhocos, rhosin, longitude = float(rhocos), float(rhosin), float(longitude)
                    longitude *= np.pi/180.
                    x = rhocos*np.cos(longitude)
                    y = rhocos*np.sin(longitude)
                    z = rhosin
                    ObservatoryXYZ[code]=(x,y,z)
                else:
                    ObservatoryXYZ[code]=(None,None,None)
        self.ObservatoryXYZ = ObservatoryXYZ

    # Parses a line from the MPC's ObsCode.txt file
    def parseObsCode(self, line):
        code, longitude, rhocos, rhosin, ObsName = line[0:3], line[4:13], line[13:21], line[21:30], line[30:].rstrip('\n')
        if longitude.isspace():
            longitude = None
        if rhocos.isspace():
            rhocos = None
        if rhosin.isspace():
            rhosin = None
        return code, longitude, rhocos, rhosin, ObsName

    # This routine parses the section of the second line that encodes the geocentric satellite position.
    def parseXYZ(xyz):
        try:
            xs = xyz[0]
            x = float(xyz[1:11])
            if xs=='-':
                x = -x
            ys = xyz[12]
            y = float(xyz[13:23])
            if ys=='-':
                y = -y
            zs = xyz[24]
            z = float(xyz[25:])
            if zs=='-':
                z = -z
        except:
            print('parseXYZ')
            print(xyz)
        return x, y, z    


# Parses the date string from the 80-character record
def parseDate(dateObs):
    yr = dateObs[0:4]
    mn = dateObs[5:7]
    dy = dateObs[8:]
    return yr, mn, dy

# These routines convert the RA and Dec strings to floats.
def RA2degRA(RA):
    hr = RA[0:2]
    if hr.strip() == '':
        hr = 0.0
    else:
        hr = float(hr)
    mn = RA[3:5]
    if mn.strip == '':
        mn = 0.0
    else:
        mn = float(mn)
    sc = RA[6:]
    if sc.strip() == '':
        sc = 0.0
    else:
        sc = float(sc)
    degRA = 15.0*(hr + 1./60. * (mn + 1./60. * sc))
    return degRA

def Dec2degDec(Dec):
    s = Dec[0]
    dg = Dec[1:3]
    if dg.strip()=='':
        dg = 0.0
    else:
        dg = float(dg)
    mn = Dec[4:6]
    if mn.strip()=='':
        mn = 0.0
    else:
        mn = float(mn)
    sc = Dec[7:]
    if sc.strip() == '':
        sc = 0.0
    else:
        sc = float(sc)
    degDec = dg + 1./60. * (mn + 1./60. * sc)
    if s == '-':
        degDec = -degDec
    return degDec

def convertEpoch(Epoch):
    yr0 = Epoch[0]
    yr1 = Epoch[1:3]
    mn  = Epoch[3]
    dy  = Epoch[4]
    if yr0=='I':
        yr0 = 1800
    elif yr0=='J':
        yr0 = 1900
    elif yr0=='K':
        yr0 = 2000
    else:
        print("Year error in convertEpoch")
        yr0 = 2000
    yr = yr0+int(yr1)
    if mn.isdigit():
        mn = int(mn)
    elif mn=='A':
        mn = 10
    elif mn=='B':
        mn = 11
    elif mn=='C':
        mn = 12
    else:
        print("Month error in convertEpoch")
        mn = 0
    if not dy.isdigit():
        dy = 10 + ord(dy) - ord('A')
    return yr, mn, int(dy)

'''
def yrmndy2JD(yrmndy):
    yr, mn, dy = yrmndy
    hr, dy = math.modf(float(dy))
    jd = novas.julian_date(int(yr), int(mn), int(dy), 24.*hr)
    return jd
'''

def deg2dms(v):
    minus_flag = (v<0.0)
    v = abs(v)
    v_deg = int(math.floor(v))
    v_min = int(math.floor((v-v_deg)*60.0))
    v_sec = (v-v_deg-v_min/60.)*3600
    if minus_flag:
        v_sgn = '-'
    else:
        v_sgn = "+"
    return v_sgn, v_deg, v_min, v_sec


def H_alpha(H, G, alpha):
    # H is the absolute magnitude 
    # alpha is the solar phase angle in degrees
    A_1 = 3.332
    B_1 = 0.631
    C_1 = 0.986
    A_2 = 1.862
    B_2 = 1.218
    C_2 = 0.238

    alp = alpha*np.pi/180.0
    ta = np.tan(0.5*alp)
    sa = np.sin(alp)

    W = np.exp(-90.56*ta*ta)

    Phi_1_s = 1.0 - C_1 * np.sin(alp)/(0.119 + 1.341*sa - 0.754*sa*sa)
    Phi_1_l = np.exp(-A_1*np.power(ta, B_1))

    Phi_2_s = 1.0 - C_2 * np.sin(alp)/(0.119 + 1.341*sa - 0.754*sa*sa)
    Phi_2_l = np.exp(-A_2*np.power(ta, B_2))

    Phi_1 = W*Phi_1_s + (1.0-W)*Phi_1_l
    Phi_2 = W*Phi_2_s + (1.0-W)*Phi_2_l

    Ha = H - 2.5*np.log10((1.0-G)*Phi_1 + G*Phi_2)
  
    return Ha

