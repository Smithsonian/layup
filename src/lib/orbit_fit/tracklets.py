dir_path = '/Users/mholman/Dropbox/support/'

import os
import bisect
import pickle

from collections import defaultdict
from collections import Counter

import scipy
import numpy as np
import healpy as hp
import spiceypy as spice

import MPC_library
#import kepcart as kc

# Load a few spice kernels
spice.furnsh(dir_path+'/kernels/MetaK_new.txt')

Observatories = MPC_library.Observatory(dir_path+'ObsCodes.txt')
ObservatoryXYZ = Observatories.ObservatoryXYZ

au2m = 149597870700
au_km = au2m/1000.

# This routine checks the 80-character input line to see if it contains a special character (S, R, or V) that indicates a 2-line 
# record.
def is_two_line(line):
    note2 = line[14]
    obsCode   = line[77:80]
    return note2=='S' or note2=='R' or note2=='V'

def satellite_pos(second_line):
    obsCode = second_line[77:81].rstrip()
    flag = second_line[32:34]
    if flag == '1 ' or flag == '2 ':
        pos = [float(second_line[34]+second_line[35:45].strip()),
               float(second_line[46]+second_line[47:57].strip()), 
               float(second_line[58]+second_line[59:69].strip())]
        pos = np.array(pos)
    else:
        pos = None
    return obsCode, pos

# This routine opens and reads filename, separating the records into those in the 1-line and 2-line formats.
# The 2-line format lines are merged into single 160-character records for processing line-by-line.
def split_MPC_file(filename):
    filename_1_line = filename.rstrip('.txt')+"_1_line.txt"
    filename_2_line = filename.rstrip('.txt')+"_2_line.txt"
    with open(filename_1_line, 'w') as f1_out, open(filename_2_line, 'w') as f2_out:
        line1=None
        with open(filename, 'r') as f:
            for line in f:
                if is_two_line(line):
                    line1=line
                    continue
                if line1 != None:
                    merged_lines = line1.rstrip('\n') + line
                    f2_out.write(merged_lines)
                    line1 = None
                else:
                    f1_out.write(line)
                    line1 = None

# Grab a line of ATLAS data and convert it to values
def convertATLAS(line):
    line = line.split()    
    objName, jd_utc, raDeg, decDeg, mag   = line[0:5]
    filt, raSig, decSig, magSig, obsCode, prob = line[5:]
    jd_utc  = 2400000.5 + float(jd_utc)
    timeStr = 'JD %.6lf' % (jd_utc)
    jd_tdb = spice.j2000() + spice.str2et(timeStr)/(24.*60.*60.)
    raDeg   = float(raDeg)
    decDeg  = float(decDeg)
    mag     = float(mag)
    filt    = filt.strip()
    raSig   = float(raSig)
    decSig  = float(decSig)
    magSig  = float(magSig)
    obsCode = obsCode.strip()
    prob    = float(prob)
    return objName, jd_tdb, raDeg, decDeg, mag, filt, raSig, decSig, magSig, obsCode, prob

def mpctime2isotime(mpctimeStr, digits=4):
    yr, mn, dy = MPC_library.parseDate(mpctimeStr)
    dy = float(dy)
    frac_day, day = np.modf(dy)
    frac_hrs, hrs = np.modf(frac_day*24)
    frac_mins, mins = np.modf(frac_hrs*60)
    secs = frac_mins*60
    if np.round(secs, digits)>=60.0:
        secs = np.round(secs, digits) - 60
        if secs<0.0:
            secs = 0.0
        mins += 1
    if mins>=60:
        mins -= 60
        hrs += 1
    if hrs>=24:
        hrs -= 24
        day += 1 # Could mess up the number of days in the month
    formatStr = '%4s-%2s-%02dT%02d:%02d:%02.' + str(digits)+'f'
    isoStr = formatStr % (yr, mn, day, hrs, mins, secs)
    return isoStr

def mpctime2et(mpctimeStr, digits=4):
    isoStr = mpctime2isotime(mpctimeStr, digits=digits)
    return spice.str2et(isoStr)

# Grab a line of obs80 data and convert it to values
# this assumes the object is numbered.
def convertObs80(line):
    objName   = line[0:5]
    provDesig = line[5:12]
    disAst    = line[12:13]
    note1     = line[13:14]
    note2     = line[14:15]
    dateObs   = line[15:32]
    RA        = line[32:44]
    Dec       = line[44:56]
    mag       = line[65:70]
    filt      = line[70:71]
    obsCode   = line[77:80]

    if objName.strip()!='':
        objID = objName
    elif provDesig.strip()!='':
        objID = provDesig
    else:
        raise Exception('No object identifier' + objName + provDesig)
    t = mpctime2et(dateObs)
    jd_tdb = spice.j2000() + t/(24*60*60)
    raDeg, decDeg = MPC_library.RA2degRA(RA), MPC_library.Dec2degDec(Dec)
    return objID, jd_tdb, raDeg, decDeg, mag, filt, 0.0, 0.0, 0.0, obsCode, 0.0

# Grab a line of obs80 data and convert it to values
# this assumes the object is numbered.
def convertJWST(line, desig='', provID='K15G56K', obsCode='274'):
    fields = line.split()
    objID   = provID
    mjd_mid = float(fields[2])
    geopos = np.array(fields[5:8], dtype=float)
    RA        = fields[8]
    Dec       = fields[9]

    jd_utc = "JD %.7lf" % (2400000.5 + mjd_mid)
    t = spice.str2et(jd_utc)
    jd_tdb = spice.j2000() + t/(24*60*60)
    raDeg, decDeg = MPC_library.RA2degRA(RA), MPC_library.Dec2degDec(Dec)
    return objID, jd_tdb, raDeg, decDeg, obsCode, geopos

def barycentricObservatory(et, obsCode):
    # et is JPL internal time
    
    # Get the barycentric position of Earth
    pos, _= spice.spkpos('EARTH', et, 'J2000', 'NONE', 'SSB')

    # Get the matrix that rotates from the Earth's equatorial body fixed frame to the J2000 equatorial frame.
    m=spice.pxform('ITRF93', 'J2000', et)
    
    # Get the MPC's unit vector from the geocenter to
    # the observatory
    obsVec = Observatories.ObservatoryXYZ[obsCode]
    obsVec = np.array(obsVec)
    
    # Carry out the rotation and scale
    mVec = np.dot(m, obsVec)*6378.137 # This JPL's quoted Earth radius.
    
    return pos+mVec

from functools import cache
@cache
def geocentricObservatory(et, obsCode):
    # et is JPL's internal time

    # Get the matrix that rotates from the Earth's equatorial
    # body fixed frame to the J2000 equatorial frame.
    #
    # For dates before 1972-01-1 use the older model
    # otherwise the more accurate model
    current = spice.str2et('1972-01-01') 
    if et < current:
        m=spice.pxform('IAU_EARTH', 'J2000', et)
    else:
        m=spice.pxform('ITRF93', 'J2000', et)
    
    # Get the MPC's unit vector from the geocenter to
    # the observatory
    obsVec = Observatories.ObservatoryXYZ[obsCode]
    obsVec = np.array(obsVec)
    
    # Carry out the rotation and scale
    mVec = np.dot(m, obsVec)* 6378.137
    
    return mVec

# Grab a line of obs80 data and convert it to values
# this assumes the object is numbered.
def convertObs80_trksub(line):
    trksub    = line[0:12]
    disAst    = line[12:13]
    note1     = line[13:14]
    note2     = line[14:15]
    dateObs   = line[15:32]
    RA        = line[32:44]
    Dec       = line[44:56]
    mag       = line[65:70]
    filt      = line[70:71]
    obsCode   = line[77:80]
    jd_utc = MPC_library.date2JD(dateObs)
    raDeg, decDeg = MPC_library.RA2degRA(RA), MPC_library.Dec2degDec(Dec)
    return trksub, jd_utc, raDeg, decDeg, mag, filt, 0.0, 0.0, 0.0, obsCode, 0.0

# Grab a line of obs80 data and convert it to values
# this assumes the object has a provID or a trkSub
def convertITF80(line):
    objName   = line[0:5]
    provDesig = line[5:12]
    disAst    = line[12:13]
    note1     = line[13:14]
    note2     = line[14:15]
    dateObs   = line[15:32]
    RA        = line[32:44]
    Dec       = line[44:56]
    mag       = line[65:70]
    filt      = line[70:71]
    obsCode   = line[77:80]

    t = mpctime2et(dateObs)
    jd_tdb = spice.j2000() + t/(24*60*60)
    raDeg, decDeg = MPC_library.RA2degRA(RA), MPC_library.Dec2degDec(Dec)
    return provDesig, jd_tdb, raDeg, decDeg, mag, filt, 0.0, 0.0, 0.0, obsCode, 0.0

def get_date(line):
    return line[15:32]

# This rotation is taking things from equatorial to ecliptic
rot_mat = MPC_library.rotate_matrix(-MPC_library.Constants.ecl)
def equatorial_to_ecliptic(v, rot_mat=rot_mat):
    return np.dot(v, rot_mat.T)

def ecliptic_to_equatorial(v, rot_mat=rot_mat.T):
    return np.dot(v, rot_mat.T)

# This could be streamlined to avoid repeated calculation of the same quantities.
# This assumes that v1 is a unit vector.
def get_residuals(v1, v2):
    x, y, z = v1
    #x = v1[:,0]
    #y = v1[:,1]
    #z = v1[:,2]
    delta = np.arcsin(z)
    alpha = np.arctan2(y, x)
    sina = np.sin(alpha)
    cosa = np.cos(alpha)
    sind = np.sin(delta)
    cosd = np.cos(delta)
    A = np.array((-sina, cosa, 0.0))
    D = np.array((-sind*cosa, -sind*sina, cosd))
    return np.array([np.dot(v2, A), np.dot(v2, D)])

# This routine opens and reads filename, separating the records into those in the 1-line and 2-line formats.
# The 2-line format lines are merged into single 160-character records for processing line-by-line.
def merge_MPC_file(filename, new_filename, comment_char='#'):
    with open(new_filename, 'w') as f1_out:
        line1=None
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith(comment_char):
                    continue
                if is_two_line(line):
                    line1=line
                    continue
                if line1 != None:
                    merged_lines = line1.rstrip('\n') + line
                    f1_out.write(merged_lines)
                    line1 = None
                else:
                    f1_out.write(line)
                    line1 = None




def format_astrometry_orbfit(filename, h_filename, readfunc=convertObs80, ecliptic=False):
    with open(h_filename, 'w') as outfile:
        with open(filename, 'r') as f:
            # skip the header line.  This should be generalized.
            #f.readline()
            outstring = "#trackletID jd_tdb                ra          ra_unc    dec       dec_unc       x_obs           y_obs           z_obs   obsCode  \n"
            outfile.write(outstring)
            for i, line in enumerate(f):
                if line.strip().startswith('/'):
                    continue

                if line[14]=='R':
                    continue

                objName, jd_tdb, raDeg, decDeg, mag, filt, RA_sig, Dec_sig, mag_sig, obsCode, prob = readfunc(line[0:80])
                xt = np.cos(decDeg*np.pi/180.)*np.cos(raDeg*np.pi/180.)
                yt = np.cos(decDeg*np.pi/180.)*np.sin(raDeg*np.pi/180.)  
                zt = np.sin(decDeg*np.pi/180.)

                RA_sig = 0.2
                Dec_sig = 0.2

                if ecliptic:
                    xt, yt, zt = equatorial_to_ecliptic(np.array((xt, yt, zt)))

                et = (jd_tdb-spice.j2000())*24*60*60

                if(len(line.strip())>80):
                    obsCode_test, geoc_pos = satellite_pos(line[80:])
                    if obsCode_test != obsCode:
                        print('obs codes not the same', 'x', obsCode_test, 'x', obsCode, 'x')
                        print(line)
                else:
                    geoc_pos = geocentricObservatory(et, obsCode)

                # Get the barycentric position of Earth
                pos, _= spice.spkpos('EARTH', et, 'J2000', 'NONE', 'SSB')

                bary_obs = pos + geoc_pos

                bary_obs /= au_km
                        
                
                if filt.isspace():
                    filt = '-'
                if mag.isspace():
                    mag = '----'

                if ecliptic:
                    bary_obs = equatorial_to_ecliptic(bary_obs)

                xo, yo, zo = bary_obs
                
                outstring = "%11s %13.10lf %13.7lf %5.2lf %13.7lf %5.2lf %15.11lf %15.11lf %15.11lf %4s\n"% \
                            (objName, jd_tdb, raDeg, RA_sig, decDeg, Dec_sig, xo, yo, zo, obsCode)
                outfile.write(outstring)

def format_astrometry_JWST(filename, h_filename, readfunc=convertJWST, ecliptic=False):
    with open(h_filename, 'w') as outfile:
        with open(filename, 'r') as f:
            f.readline()
            f.readline()
            f.readline()
            # skip the header line.  This should be generalized.
            #f.readline()
            outstring = "#trackletID jd_tdb                ra          ra_unc    dec       dec_unc       x_obs           y_obs           z_obs   obsCode  \n"
            outfile.write(outstring)
            for i, line in enumerate(f):
                if line.strip().startswith('/'):
                    continue

                if line[14]=='R':
                    continue

                objName, jd_tdb, raDeg, decDeg, obsCode, geoc_pos = readfunc(line)
                xt = np.cos(decDeg*np.pi/180.)*np.cos(raDeg*np.pi/180.)
                yt = np.cos(decDeg*np.pi/180.)*np.sin(raDeg*np.pi/180.)  
                zt = np.sin(decDeg*np.pi/180.)

                RA_sig = 0.002
                Dec_sig = 0.002

                if ecliptic:
                    xt, yt, zt = equatorial_to_ecliptic(np.array((xt, yt, zt)))

                et = (jd_tdb-spice.j2000())*24*60*60

                # Get the barycentric position of Earth
                pos, _= spice.spkpos('EARTH', et, 'J2000', 'NONE', 'SSB')

                bary_obs = pos + geoc_pos

                bary_obs /= au_km
                        
                if ecliptic:
                    bary_obs = equatorial_to_ecliptic(bary_obs)

                xo, yo, zo = bary_obs
                
                outstring = "%11s %13.10lf %13.7lf %6.3lf %13.7lf %6.3lf %15.11lf %15.11lf %15.11lf %4s\n"% \
                            (objName, jd_tdb, raDeg, RA_sig, decDeg, Dec_sig, xo, yo, zo, obsCode)
                outfile.write(outstring)

                
# Look for time gaps > dt in a sorted list of times
def segment_times(times, dt):
    result =[]
    t0 = times[0]
    for i, t in enumerate(times):
        if np.abs(t-t0)>dt:
            result.append(i)
        t0 = t
    return result

# This is used to separate a sorted array of values
# into chunks wherever a gap of dt or larger is encountered.
def segment(times, dt):
    times = np.array(times)
    idxs = list(np.where((times[1:] - times[:-1])>dt)[0]+1)
    idxs0 = [0]+idxs
    idxs = idxs + [len(times)]
    return list(zip(idxs0, idxs))

# Returns the jd of new moon, to the nearest half day
def lunation_center(n, tref=2457722.0125, p=29.53055):
    t = tref + p*n
    tp = np.floor(t) + 0.5
    return tp

def full_moon_center(n, tref=2457722.0125, p=29.53055):
    t = tref + p*(n+0.5)
    tp = np.floor(t) + 0.5
    return tp

# Return the number of sequential comment lines that were skipped at the
# beginning of a file.
def count_skip_lines(comment_char, filename):
    with open(filename) as infile:
        i = 0
        line = infile.readline()
        while line.startswith(comment_char):
            line = infile.readline()
            i += 1
        return i




# This routine separates tracklets into time windows (centered on a list of times within a time range) 
# and saves the results in pickle files.
def separate_time_windows_v5(tracklets, sortedTracklets, tracklets_jd_dict, time_centers, file_stem='data/itf_new_1_line_ec.mpc', dt=15., suff='.mpc'):

    files = {}

    header='#trackletID yr   mn dy      obsCode mag filter  jd_tdb       x_target     y_target     z_target      x_obs       y_obs        z_obs     '

    jds = [tracklets_jd_dict[k] for k in sortedTracklets]
    
    for t in time_centers:
                
        i = bisect.bisect(jds, t-dt, 0, len(jds))
        j = bisect.bisect(jds, t+dt, 0, len(jds))
        
        if j>i:
            
            tmp_tracklets={}
            tmp_sortedTracklets=sortedTracklets[i:j]
            tmp_tracklets_jd_dict={}
            outfile = file_stem.replace('.mpc', '')+'_'+str(t)+'_pm'+str(dt)+suff

            for key in tmp_sortedTracklets:
                tmp_tracklets[key] = tracklets[key]
                tmp_tracklets_jd_dict[key] = tracklets_jd_dict[key]
                
            with open(outfile, 'wb') as handle:
                pickle.dump((tmp_tracklets, tmp_tracklets_jd_dict, tmp_sortedTracklets), handle, protocol=pickle.HIGHEST_PROTOCOL)


def tracklet_time_windows(tracklets, sortedTracklets, tracklets_jd_dict, time_centers, dt=15.):

    jds = [tracklets_jd_dict[k] for k in sortedTracklets]
    
    results =[]
    
    for t in time_centers:
                
        i = bisect.bisect(jds, t-dt, 0, len(jds))
        j = bisect.bisect(jds, t+dt, 0, len(jds))
        results.append((i, j))
        
    return results


def parse_line(line):
    fields = line.split()
    trackletID = fields[0].strip()
    obsCode = fields[1]
    jd_tdb = float(fields[4])

    return trackletID, obsCode, jd_tdb
    
# This routine is fundamental.  It takes both the name of the
# file that contains the data in the original MPC format and
# same file in our format, then it goes through and identifies
# tracklets according to a set of criteria.
#
# The criteria for a tracklet are:
# 1) The same obsCode
# 2) The same night (within +/- 0.5 day)
# 3) The same trkSub (observer-assigned trackletID)
#
# Make the time span flexible
#
# It returns a dictionary of tracklets for which the keys
# are a 3-part tuple composed of a trackletID, a day number,
# and an obsCode; a dictionary of the start times of those tracklets,
# using the same keys; and a list of the 3-part keys in the order
# of those start times.
def get_sorted_tracklets(itf_filename_orig, itf_filename, parse_func=parse_line):

    # Skip the header lines
    skip_count = count_skip_lines('#', itf_filename_orig)
    infile_orig = open(itf_filename_orig)
    for _ in range(skip_count):
        infile_orig.readline()
        
    skip_count = count_skip_lines('#', itf_filename)
    infile = open(itf_filename)
    for _ in range(skip_count):
        infile.readline()

    temp_tracklets = defaultdict(list)
    
    for line, line_orig in zip(infile, infile_orig):

        # The next lines should be replaced with
        # more general methods
        try:
            trackletID, obsCode, jd_tdb = parse_func(line)
            #fields = line.split()
            #trackletID = fields[0].strip()
            #obsCode = fields[1]
            #jd_tdb = float(fields[4])
        except:
            print(line)

        # As an intermediate step, we are gathering
        # tracklets that might span multiple nights.
        #print(trackletID.strip(), obsCode)
        
        temp_key = (trackletID.strip(), obsCode)
        temp_tracklets[temp_key].append((line, line_orig))
  
    # Now iterate over the sets of gathered lines, converting them to sets
    # and back to lists to eliminate exact duplicates
    for key in temp_tracklets.keys():
        temp_tracklets[key]= list(set(temp_tracklets[key]))
        
    #return temp_tracklets

    # Now iterate over the sets of gathered lines, sorting the lines, and
    # splitting them up into tracklets that meet the criteria.
    tracklets = defaultdict(list)
    tracklets_jd_dict = {}
    
    for temp_key, v in temp_tracklets.items():
        
        trackletID = temp_key[0]
        obsCode = temp_key[1]

        # This is not general
        sortedLines = sorted(v, key=lambda k: float(k[0].split()[4]))
        sortedTimes = sorted([float(k[0].split()[4]) for k in sortedLines])
                                        
        #dt = sortedTimes[-1] - sortedTimes[0]
        idxs = segment(sortedTimes, 0.5)
        for s in idxs:
            jd_tdb = sortedTimes[s[0]]
            mjdp = int(jd_tdb-2400000.5)
            key = (trackletID, mjdp, obsCode)
            if key not in tracklets_jd_dict:
                tracklets_jd_dict[key] = jd_tdb
            tracklets[key].extend(sortedLines[s[0]:s[1]])
        

    # We changed this on 20 Mar 2018 to exclude singleton tracklets.  
    # The line below was active before, including the singleton tracklets.
    trackletKeys = [k for k in tracklets.keys() if len(tracklets[k])>1]
    
    sortedTrackletKeys = sorted(trackletKeys, key=lambda k: tracklets_jd_dict[k]) 
    return tracklets, tracklets_jd_dict, sortedTrackletKeys


# Get the time and vectors from a line, where the line is
# in our format.
def get_observation_data(line):
    
    fields = line.split()
    trackletID = fields[0].strip()
    jd_tdb = float(fields[4])
        
    x_target, y_target, z_target = fields[-6:-3]
    r_target = np.array([float(x_target), float(y_target), float(z_target)])

    x_obs, y_obs, z_obs = fields[-3:]
    r_obs = np.array([float(x_obs), float(y_obs), float(z_obs)])
    
    return jd_tdb, r_target, r_obs

def fg_series(mu, r0, r0dot, t):
    sigma = mu/(r0*r0*r0)
    tau = r0dot/r0
    f = 1.0 - 0.5*sigma*t*t + 0.5*sigma*tau*t*t*t
    g = t - sigma*t*t*t/6.0
    return f, g

def initialize_observations(tracklet):
    jd_tdb, r_target, r_E = [], [], []
    for obs in tracklet:
        jd, rt, re = get_observation_data(obs[0])
        jd_tdb.append(jd)
        r_target.append(rt)
        r_E.append(re)

    jd_tdb = np.array(jd_tdb)
    r_target = np.array(r_target)
    r_E = np.array(r_E)
    
    return(jd_tdb, r_target, r_E)

def apply_constraints(r0, r0dot, r, v):
    r0hat = r/np.linalg.norm(r)
    vpara = np.dot(v, r0hat)*r0hat
    vperp = v - vpara

    v_new = vperp + r0dot*r0hat
    r_new = r0*r0hat
    
    return(r_new, v_new)

def initial_solution(t0, r0, r0dot, jd_tdb, r_target, r_E, 
                     rho = None,
                     c = MPC_library.Constants.speed_of_light,
                     GMsun = MPC_library.Constants.GMsun,
                     constraints = True,):
    if rho==None:
        rho = np.ones(len(jd_tdb))*r0
    
    # The equations could be normalized by r0.
    # Not sure if that would be helpful.
    # rhs = r_target.T*rho/r0 + r_E.T/r0
    rhs = r_target.T*rho + r_E.T
    rhs = rhs.T
    dts = rho/c
    tp = jd_tdb-dts
    A = np.array(fg_series(GMsun, r0, r0dot, tp-t0)).T
    res = np.linalg.lstsq(A, rhs, rcond=None)[0]
    r = res[0]
    v = res[1]
    
    if constraints:
        r_new, v_new = apply_constraints(r0, r0dot, r, v)
    else:
        r_new, v_new = r, v
        
    return dts, r_new, v_new

def get_solution(t0, r0, r0dot, tracklet):
    jd_tdb, r_target, r_E = initialize_observations(tracklet)
    dts, r_new, v_new = initial_solution(t0, r0, r0dot, jd_tdb, r_target, r_E)
    dts, r_new, v_new, delta_t, states = iterative_solution(t0, r0, r0dot, jd_tdb, r_target, r_E, dts, r_new, v_new)
    return r_new, v_new, dts, delta_t, states

def transform_tracklets(t0, r0, r0dot, tracklets_dict):
    solutions = {}
    # It woud be good to get rid of this explicit loop.
    for k, tracklet in tracklets_dict.items(): 
        if len(tracklet)>1:
            try:
                r_new, v_new, dts, delta_t, _ = get_solution(t0, r0, r0dot, tracklet)
                solutions[k] = r_new, v_new, dts, delta_t, t0
            except:
                print("Error", k)
    return solutions

def transform_tracklets(t0, r0, r0dot, tracklets_dict):
    solutions = {}
    for k, tracklet in tracklets_dict.items(): 
        if len(tracklet)>1:
            r_new, v_new, dts, delta_t, _ = get_solution(t0, r0, r0dot, tracklet)
            solutions[k] = r_new, v_new, dts, delta_t, t0
    return solutions


def DFS(v, visited, matches):
    visited.add(v)
    for neighbour in matches[v]:
        if neighbour not in visited:
            DFS(neighbour, visited, matches)
    return visited

def group_clusters(matches, nm=1):
    clusters = []
    remainders = set([i for i, m in enumerate(matches) if len(m)>=nm])
    while len(remainders)>0:
        s = remainders.pop()
        cluster = DFS(s, set(), matches)
        clusters.append(cluster)
        remainders = remainders.difference(cluster)
    return clusters
def find_clusters(solutions, r0, r0dot, scale, rad, nm=2):
    # There is no guarantee that solutions.items() will
    # behave the same in the two calls.
    solns = np.array([np.append(v[0], scale*v[1]) for (k, v) in solutions.items()])/r0
    #solns = np.array([np.append(v[0]/r0, scale*v[1]/r0dot) for (k, v) in solutions.items()])
    #solns = np.array([v[0]/r0 for (k, v) in solutions.items()])
    keys = [k for (k, v) in solutions.items()]
    points=np.array(solns)
    tree = scipy.spatial.cKDTree(points)
    matches = tree.query_ball_tree(tree, rad)
    clusters = group_clusters(matches, nm=nm)
    return clusters, keys

def find_matches(solutions, r0, r0dot, scale, rad, nm=2):
    # There is no guarantee that solutions.items() will
    # behave the same in the two calls.
    solns = np.array([np.append(v[0], scale*v[1]) for (k, v) in solutions.items()])/r0
    #solns = np.array([np.append(v[0]/r0, scale*v[1]/r0dot) for (k, v) in solutions.items()])
    #solns = np.array([v[0]/r0 for (k, v) in solutions.items()])
    keys = [k for (k, v) in solutions.items()]
    points=np.array(solns)
    tree = scipy.spatial.cKDTree(points)
    matches = tree.query_ball_tree(tree, rad)
    #clusters = group_clusters(matches, nm=nm)
    return matches, keys

# This returns the topocentric distances and new heliocentric
# position vectors to the target, given the assumed distance
# r and the position vector of the observatory re.
def adjust_position(r, rho_target, re):
    rho_x, rho_y, rho_z = rho_target
    xe, ye, ze = re
    Robs = np.sqrt(xe * xe + ye * ye + ze * ze)
    cos_phi = -(rho_x * xe + rho_y * ye + rho_z * ze) / Robs
    phi = np.arccos(cos_phi)
    sin_phi = np.sin(phi)

    xx2 = r*r - Robs*sin_phi * Robs*sin_phi
    
    if xx2 < 0:
        None, None

    xx = np.sqrt(xx2)
    yy = Robs * cos_phi
    
    rho_p = yy + xx

    # This could be done with numpy arrays
    x_p = xe + rho_p*rho_x
    y_p = ye + rho_p*rho_y
    z_p = ze + rho_p*rho_z
    
    rho_m = yy - xx
    
    # This could be done with numpy arrays    
    x_m = xe + rho_m*rho_x
    y_m = ye + rho_m*rho_y
    z_m = ze + rho_m*rho_z
        
    return (rho_p, (x_p, y_p, z_p)), (rho_m, (x_m, y_m, z_m))

# This returns the topocentric distances and new heliocentric
# position vectors to the target, given the assumed distance
# r and the position vector of the observatory re.
# This version assumes that only the positive solution is valid.


# This should be converted to numpy arrays.
# This should go in a library.

def adjust_position_p(r, rho_target, re):
    rho_x, rho_y, rho_z = rho_target
    xe, ye, ze = re
    Robs = np.sqrt(xe * xe + ye * ye + ze * ze)
    cos_phi = -(rho_x * xe + rho_y * ye + rho_z * ze) / Robs
    phi = np.arccos(cos_phi)
    sin_phi = np.sin(phi)

    xx2 = r*r - Robs*sin_phi * Robs*sin_phi
    
    xx = np.sqrt(xx2)
    yy = Robs * cos_phi
    
    rho_p = yy + xx

    # This could be done with numpy arrays
    x_p = xe + rho_p*rho_x
    y_p = ye + rho_p*rho_y
    z_p = ze + rho_p*rho_z
    
    return (rho_p, (x_p, y_p, z_p))


# Check again
# This routine returns the 3-D rotation matrix for the 
# given reference vector.
def xyz_to_proj_matrix(r_ref):
    x_ref, y_ref, z_ref = r_ref
    r = np.sqrt(x_ref*x_ref + y_ref*y_ref + z_ref*z_ref)
    lon0 = np.arctan2(y_ref, x_ref)
    lat0 = np.arcsin(z_ref/r)
    slon0 = np.sin(lon0)
    clon0 = np.cos(lon0)
    slat0 = np.sin(lat0)
    clat0 = np.cos(lat0)

    mat = np.array([[-slon0, clon0, 0], 
                    [-clon0*slat0, -slon0*slat0, clat0], 
                    [clon0*clat0, slon0*clat0, slat0 ]])
    
    return mat

# Check again
# This routine returns the 3-D rotation matrix for the 
# given reference vector.
def proj_to_xyz_matrix(r_ref):
    x_ref, y_ref, z_ref = r_ref
    r = np.sqrt(x_ref*x_ref + y_ref*y_ref + z_ref*z_ref)
    lon0 = np.arctan2(y_ref, x_ref)
    lat0 = np.arcsin(z_ref/r)
    slon0 = np.sin(lon0)
    clon0 = np.cos(lon0)
    slat0 = np.sin(lat0)
    clat0 = np.cos(lat0)

    mat = np.array([[-slon0, -clon0*slat0, clon0*clat0], 
                    [clon0,  -slon0*slat0, slon0*clat0], 
                    [0, clat0,  slat0 ]])
    
    return mat

def residuals(p, v, t_ref, gdot, GM=MPC_library.Constants.GMsun, speed_of_light=MPC_library.Constants.speed_of_light):

    a, adot, b, bdot, g = p

    t_emit = np.array([(obs.t-t_ref - obs.ze/speed_of_light) for obs in v])
    acc_z = -GM*g*g
    fac = np.array([(1.0 - gdot*t - 0.5*g*acc_z*t*t + g*obs.ze) for obs, t in zip(v, t_emit)])
    #fac = np.array([(1.0 - gdot*t) for obs, t in zip(v, t_emit)])    

    theta_x = np.array([obs.theta_x for obs in v])
    theta_y = np.array([obs.theta_y for obs in v])

    xe = np.array([obs.xe for obs in v])
    ye = np.array([obs.ye for obs in v])    

    mod_x = a + adot*t_emit - g*xe - gdot*t_emit*(adot*t_emit - g*xe)
    mod_y = b + bdot*t_emit - g*ye - gdot*t_emit*(bdot*t_emit - g*ye)

    #mod_x = (a + adot*t_emit - g*xe)/fac # - gdot*t_emit*(adot*t_emit - g*xe)
    #mod_y = (b + bdot*t_emit - g*ye)/fac # - gdot*t_emit*(bdot*t_emit - g*ye)
    
    res_x = theta_x - mod_x
    res_y = theta_y - mod_y

    res = np.empty((res_x.size + res_y.size,), dtype=res_x.dtype)
    res[0::2] = res_x
    res[1::2] = res_y
    
    return res

def partials(p, v, t_ref, gdot, GM=MPC_library.Constants.GMsun, speed_of_light=MPC_library.Constants.speed_of_light):

    a, adot, b, bdot, g = p

    #a, adot, b, bdot, g, gdot = p

    t_emit = np.array([(obs.t-t_ref - obs.ze/speed_of_light) for obs in v])

    xe = np.array([obs.xe for obs in v])
    ye = np.array([obs.ye for obs in v])    

    mod_x = a + adot*t_emit - g*xe - gdot * (adot*t_emit*t_emit - g*xe*t_emit)
    mod_y = b + bdot*t_emit - g*ye - gdot * (bdot*t_emit*t_emit - g*ye*t_emit)
    
    dres_x_da = -np.ones(len(t_emit))
    dres_x_dadot = -t_emit + gdot*t_emit*t_emit
    dres_x_db = np.zeros(len(t_emit))
    dres_x_dbdot = np.zeros(len(t_emit))
    dres_x_dg = xe - gdot*xe*t_emit
    dres_x_dgdot = adot*t_emit*t_emit - g*xe*t_emit

    dres_x = np.column_stack((dres_x_da, dres_x_dadot,
                              dres_x_db, dres_x_dbdot,
                              dres_x_dg))#, dres_x_dgdot))

    dres_y_da = np.zeros(len(t_emit))
    dres_y_dadot = np.zeros(len(t_emit))
    dres_y_db = -np.ones(len(t_emit))
    dres_y_dbdot = -t_emit + gdot*t_emit*t_emit
    dres_y_dg = ye - gdot*ye*t_emit
    dres_y_dgdot = bdot*t_emit*t_emit - g*ye*t_emit

    dres_y = np.column_stack((dres_y_da, dres_y_dadot,
                              dres_y_db, dres_y_dbdot,
                              dres_y_dg))#, dres_y_dgdot))

    dres = np.empty((dres_x.shape[0]*2, dres_x.shape[1]), dtype=dres_x.dtype)
    dres[0::2] = dres_x
    dres[1::2] = dres_y
    
    return dres

def residuals_rates(p, v, t_ref, gdot, GM=MPC_library.Constants.GMsun, speed_of_light=MPC_library.Constants.speed_of_light):
    a, adot, b, bdot, g = p

    t_emit = np.array([(obs.t-t_ref - obs.ze/speed_of_light) for obs in v])

    theta_x  = np.array([obs.theta_x for obs in v])
    theta_y  = np.array([obs.theta_y for obs in v])

    dtheta_x_dt = np.array([obs.dtheta_x_dt for obs in v])
    dtheta_y_dt = np.array([obs.dtheta_y_dt for obs in v])    

    xe = np.array([obs.xe for obs in v])
    ye = np.array([obs.ye for obs in v])
    vxe = np.array([obs.vxe for obs in v])
    vye = np.array([obs.vye for obs in v])  

    mod_x = a + adot*t_emit - g*xe - gdot*t_emit*(adot*t_emit - g*xe)
    mod_y = b + bdot*t_emit - g*ye - gdot*t_emit*(bdot*t_emit - g*ye)

    mod_dx_dt = adot - g*vxe - gdot*(adot*t_emit - g*xe) - gdot*t_emit*(adot - g*vxe)
    mod_dx_dt = bdot - g*vye - gdot*(bdot*t_emit - g*vxe) - gdot*t_emit*(bdot - g*vye)    
    
    res_x = theta_x - mod_x
    res_y = theta_y - mod_y

    res_dx_dt = dtheta_x_dt - mod_dx_dt
    res_dy_dt = dtheta_y_dt - mod_dy_dt
    
    res = np.empty((res_x.size + res_y.size + res_dx_dt.size + res_dy_dt.size,), dtype=res_x.dtype)

    res[0::4] = res_x
    res[1::4] = res_y
    res[2::4] = res_dx_dt
    res[3::4] = res_dy_dt
    
    return res

def partials_rates(p, v, t_ref, gdot, GM=MPC_library.Constants.GMsun, speed_of_light=MPC_library.Constants.speed_of_light):
    a, adot, b, bdot, g = p

    t_emit = np.array([(obs.t-t_ref - obs.ze/speed_of_light) for obs in v])

    xe = np.array([obs.xe for obs in v])
    ye = np.array([obs.ye for obs in v])
    vxe = np.array([obs.vxe for obs in v])
    vye = np.array([obs.vye for obs in v])  

    mod_x = a + adot*t_emit - g*xe - gdot * (adot*t_emit*t_emit - g*xe*t_emit)
    mod_y = b + bdot*t_emit - g*ye - gdot * (bdot*t_emit*t_emit - g*ye*t_emit)

    dres_x_da = -np.ones(len(t_emit))
    dres_x_dadot = -t_emit + gdot*t_emit*t_emit
    dres_x_db = np.zeros(len(t_emit))
    dres_x_dbdot = np.zeros(len(t_emit))
    dres_x_dg = xe - gdot*xe*t_emit
    #dres_x_dgdot = adot*t_emit*t_emit - g*xe*t_emit

    dres_x = np.column_stack((dres_x_da, dres_x_dadot,
                              dres_x_db, dres_x_dbdot,
                              dres_x_dg))

    dres_y_da = np.zeros(len(t_emit))
    dres_y_dadot = np.zeros(len(t_emit))
    dres_y_db = -np.ones(len(t_emit))
    dres_y_dbdot = -t_emit + gdot*t_emit*t_emit
    dres_y_dg = ye - gdot*ye*t_emit
    #dres_y_dgdot = bdot*t_emit*t_emit - g*ye*t_emit

    dres_y = np.column_stack((dres_y_da, dres_y_dadot,
                              dres_y_db, dres_y_dbdot,
                              dres_y_dg))

    mod_dx_dt = adot - g*vxe - gdot * (2*adot*t_emit - g*xe - g*vxe*t_emit)
    mod_dy_dt = bdot - g*vye - gdot * (2*bdot*t_emit - g*ye - g*vye*t_emit)

    dres_dx_dt_da = np.zeros(len(t_emit))
    dres_dx_dt_dadot = np.ones(len(t_emit)) - 2*gdot*t_emit
    dres_dx_dt_db = np.zeros(len(t_emit))
    dres_dx_dt_dbdot = np.zeros(len(t_emit))
    dres_dx_dt_dg = -vxe + gdot * (xe + vxe*t_emit)
    #dres_dx_dt_dgdot = -(2*adot*t_emit - g*xe - g*vxe*t_emit)

    dres_dx_dt = np.column_stack((dres_dx_dt_da, dres_dx_dt_dadot,
                              dres_dx_dt_db, dres_dx_dt_dbdot,
                              dres_dx_dt_dg))
    
    dres_dy_dt_da = np.zeros(len(t_emit))
    dres_dy_dt_dadot = np.zeros(len(t_emit))
    dres_dy_dt_db = np.zeros(len(t_emit))
    dres_dy_dt_dbdot = np.ones(len(t_emit)) - 2*gdot*t_emit    
    dres_dy_dt_dg = -vye + gdot * (ye + vye*t_emit)
    #dres_dy_dt_dgdot = -(2*bdot*t_emit - g*ye - g*vye*t_emit)

    dres_dy_dt = np.column_stack((dres_dy_dt_da, dres_dy_dt_dadot,
                              dres_dy_dt_db, dres_dy_dt_dbdot,
                              dres_dy_dt_dg))
    
    dres = np.empty((dres_x.shape[0]*4, dres_x.shape[1]), dtype=dres_x.dtype)
    dres[0::4] = dres_x
    dres[1::4] = dres_y
    dres[2::4] = dres_dx_dt
    dres[3::4] = dres_dy_dt
    
    return dres


def kbo2d_linear(p, v, t_ref, gdot, GM=MPC_library.Constants.GMsun, speed_of_light=MPC_library.Constants.speed_of_light):
    a, adot, b, bdot, g = p

    #a, adot, b, bdot, g = p

    #gdot = 0.0

    t_emit = np.array([(obs.t-t_ref - obs.ze/speed_of_light) for obs in v])
    #acc_z = -GM*g*g
    #fac = np.array([(1.0 - gdot*t - 0.5*g*acc_z*t*t + g*obs.ze) for obs, t in zip(v, t_emit)])
    fac = np.array([(1.0 - gdot*t) for obs, t in zip(v, t_emit)])    

    theta_x = np.array([obs.theta_x for obs in v])
    theta_y = np.array([obs.theta_y for obs in v])

    xe = np.array([obs.xe for obs in v])
    ye = np.array([obs.ye for obs in v])    

    mod_x = a + adot*t_emit - g*xe - gdot * (adot*t_emit*t_emit - g*xe*t_emit)
    mod_y = b + bdot*t_emit - g*ye - gdot * (bdot*t_emit*t_emit - g*ye*t_emit)
    
    res_x = theta_x - mod_x
    res_y = theta_y - mod_y

    mod = np.empty((mod_x.size + mod_y.size,), dtype=res_x.dtype)
    mod[0::2] = mod_x
    mod[1::2] = mod_y
    
    return mod


def fit_tracklet_old(t_ref, g, gdot, v, GM=MPC_library.Constants.GMsun):
    # Here's a version that incorporates radial gravitational
    # acceleration
    
    t_emit = [(obs[0]-obs[1]-t_ref) for obs in v]
    acc_z = -GM*g*g
    fac =[(1.0 + gdot*t + 0.5*g*acc_z*t*t - g*obs[7]) for obs, t in zip(v, t_emit)]
                
    A = np.vstack([t_emit, np.ones(len(t_emit))]).T 

    # Added division by theta_z, MJH 1 Feb 2024
    x = [(obs[2]/obs[4])*f + obs[5]*g for obs, f in zip(v, fac)]                 
    mx, cx = np.linalg.lstsq(A, x, rcond=None)[0]
    #res_x = np.sqrt(res_x[0]/len(v))

    # Added division by theta_z, MJH 1 Feb 2024    
    y = [(obs[3]/obs[4])*f + obs[6]*g for obs, f in zip(v, fac)]                 
    my, cy = np.linalg.lstsq(A, y, rcond=None)[0]
    #res_y = np.sqrt(res_y[0]/len(v))
    
    return (cx, mx, cy, my, t_emit[0])

def fit_tracklet_rms(t_ref, g, gdot, v, GM=MPC_library.Constants.GMsun, speed_of_light=MPC_library.Constants.speed_of_light):
    # Here's a version that incorporates radial gravitational
    # acceleration
    
    t_emit = [(obs.t-t_ref - obs.ze/speed_of_light) for obs in v]
    acc_z = -GM*g*g
    fac =[(1.0 + gdot*t + 0.5*g*acc_z*t*t - g*obs.ze) for obs, t in zip(v, t_emit)]
                
    A = np.vstack([t_emit, np.ones(len(t_emit))]).T 
    
    x = [obs.theta_x*f + g*obs.xe for obs, f in zip(v, fac)]                 
    (mx, cx), res_x = np.linalg.lstsq(A, x, rcond=None)[0:2]
    res_x = np.sqrt(res_x[0]/len(v))
            
    y = [obs.theta_y*f + g*obs.ye for obs, f in zip(v, fac)]                 
    (my, cy), res_y = np.linalg.lstsq(A, y, rcond=None)[0:2]
    res_y = np.sqrt(res_y[0]/len(v))
    
    return (cx, mx, cy, my, res_x, res_y, t_emit[0])

def fit_tracklet(t_ref, g, gdot, v, GM=MPC_library.Constants.GMsun, speed_of_light=MPC_library.Constants.speed_of_light):
    # Here's a version that incorporates radial gravitational
    # acceleration
    
    t_emit = [(obs.t-t_ref - obs.ze/speed_of_light) for obs in v]
    acc_z = -GM*g*g
    fac =[(1.0 + gdot*t + 0.5*g*acc_z*t*t - g*obs.ze) for obs, t in zip(v, t_emit)]
                
    A = np.vstack([t_emit, np.ones(len(t_emit))]).T 
    
    x = [obs.theta_x*f + g*obs.xe for obs, f in zip(v, fac)]                 
    mx, cx = np.linalg.lstsq(A, x, rcond=None)[0]
            
    y = [obs.theta_y*f + g*obs.ye for obs, f in zip(v, fac)]                 
    my, cy = np.linalg.lstsq(A, y, rcond=None)[0]
    
    return (cx, mx, cy, my, t_emit[0])

def fit_tracklet_tuple(t_ref, g, gdot, v, GM=MPC_library.Constants.GMsun, speed_of_light=MPC_library.Constants.speed_of_light):
    # Here's a version that incorporates radial gravitational
    # acceleration
    
    # It also uses a namedtuple to clarify the tracklet components.
    
    # obs[0] is jd_tdb 
    # obs[1] is theta_x
    # obs[2] is theta_y
    # obs[3] is xe
    # obs[4] is ye
    # obs[5] is ze
    
    # Careful with the parsing here
    t_emit = [(obs[0]-t_ref - obs[5]/speed_of_light) for obs in v]
    acc_z = -GM*g*g
    fac =[(1.0 + gdot*t + 0.5*g*acc_z*t*t - g*obs[5]) for obs, t in zip(v, t_emit)]
                
    A = np.vstack([t_emit, np.ones(len(t_emit))]).T 
    
    x = [obs[1]*f + obs[3]*g for obs, f in zip(v, fac)]                 
    mx, cx = np.linalg.lstsq(A, x, rcond=None)[0]
            
    y = [obs[2]*f + obs[4]*g for obs, f in zip(v, fac)]                 
    my, cy = np.linalg.lstsq(A, y, rcond=None)[0]
    
    return (cx, mx, cy, my, t_emit[0])

# This routine uses a specified value of gamma (g), 
# determines the HEALPix index of each tracklet, based
# on the heliocentric transformation of the first position
# of the tracklet for that value of g, and builds and
# returns a dictionary keyed on HEALPix indices with values
# being trackletIDs.
# This routine will be used later for selecting which
# tracklets to process.
def index_tracklets_gamma(g, tracklets_dict, nside=8):
    pix_dict=defaultdict(list)
    r = 1.0/g
    for key, value in tracklets_dict.items():
        line, line_orig = value[0]
        jd_tdb, r_target, r_obs = get_observation_data(line)
        rho_r_p, rho_r_m = adjust_position(r, r_target, r_obs)
        xp, yp, zp = rho_r_p[1]
                
        # Calculate HEALPix index
        pix = hp.vec2pix(nside, xp, yp, zp, nest=True)
        pix_dict[pix].append(key)

    return pix_dict

from collections import namedtuple
Detection = namedtuple("Detection", ("t", "theta_x", "theta_y", "xe", "ye", "ze"))

# This takes a reference time (t_ref), a set of g, gdot pairs (g_gdot_pairs), 
# a reference direction vector (vec), and a set of observation lines that 
# have been selected for a region of sky and time slice (lines)
#
# It returns a dictionary of results that have z, zdot pairs as keys and
# sets of fitted tracklets as results.  Each result has the form:
# 
# trackletID alpha alpha_dot beta beta_dot t_emit,
# where t_emit is the light time-corrected time relative to the reference
# time.  The coordinates are now in tangent plane projection.
#
def transform_to_arrows(t_ref, g_gdot_pairs, vec, trackletKeys, tracklets_dict, fit_tracklet_func=fit_tracklet, get_obs_func=get_observation_data):
        
    #GM = MPC_library.Constants.GMsun

    #rot_mat = MPC_library.rotate_matrix(MPC_library.Constants.ecl)

    results_dict = defaultdict(list)
    
    vec = np.array(vec)
    vec = vec/np.linalg.norm(vec)
    mat = xyz_to_proj_matrix(vec)

    # Try to get rid of this loop
    for trackletKey in trackletKeys:
        for line, line_orig in tracklets_dict[trackletKey]:
            jd_tdb, r_target, r_obs = get_obs_func(line)

            # Rotate to projection coordinates
            x, y, z = np.dot(mat, r_target)
            theta_x = x/z
            theta_y = y/z
            # z should be very nearly 1.

            # Rotate to projection coordinates            
            xe, ye, ze = np.dot(mat, r_obs)

            #dlt = ze/MPC_library.Constants.speed_of_light

            d = Detection(jd_tdb, theta_x, theta_y, xe, ye, ze)
            results_dict[trackletKey].append(d)
            #(jd_tdb, dlt, theta_x, theta_y, theta_z, xe, ye, ze))
        
    # All the work done above is independent of the g and gdot values
    #return results_dict

    master_results = {}
    for g_gdot in g_gdot_pairs:
        g, gdot = g_gdot

        results = []
        for k, v in results_dict.items():  
            
            # This condition is new
            if len(v)>1:
                cx, mx, cy, my, t0 = fit_tracklet_func(t_ref, g, gdot, v)
                result = (k, cx, mx, cy, my, t0)
                results.append(result)

        master_results[g_gdot] = results
    
    return master_results, results_dict

# Can probably pass in the clustering function, to reduce code duplication
def transform_to_arrows_for_regions_gamma(t_ref, g_gdot_pairs, pixels, pix_dict, tracklets_dict, nside=8, angDeg=5.5, transform_to_arrows_function=transform_to_arrows):

    tracklet_visits = Counter()
    pixel_results = {}
    for pixel in pixels:
        vec = hp.pix2vec(nside, pixel, nest=True)
        neighbors = hp.query_disc(nside, vec, angDeg*np.pi/180., inclusive=True, nest=True)
        trackletKeys = []
        for pix in neighbors:
            trackletKeys.extend(pix_dict[pix])
        print(pixel, len(trackletKeys))
        tracklet_visits.update(set(trackletKeys))
        if len(trackletKeys) > 0:
            pixel_results[pixel] = transform_to_arrows_function(t_ref, g_gdot_pairs, vec, trackletKeys, tracklets_dict)
            
    return pixel_results, tracklet_visits


def do_sky(dt, rad, pixel_results, mincount=3):
    cluster_counter = Counter()
    cluster_arrow_dict = defaultdict()
    for pix, d in pixel_results.items():
        # Probably want to keep the pixel regions separate.
        for g_gdot, arrows in d.items():

            g, gdot = g_gdot
            
            # The bit from here
            i = 0
            label_dict={}
            arrow_dict={}
            # Combined is not a very clear name
            combined=[]
            for k, cx, mx, cy, my, t in arrows:
                label_dict[i] = k
                arrow_dict[k] = [pix, cx, mx, cy, my, t, g, gdot]
                combined.append([cx, mx*dt, cy, my*dt])
                i +=1
            
            # Skip if there are no arrows
            if len(combined)==0:
                continue
            
            points=np.array(combined)
            # to here can be a function,
            # that takes arrows and dt and
            # returns label_dict, points array, and arrow_dict

            
            # Below is just one way of doing the clustering.
            # There could be different versions of this
            # that use the same input.
            #
            # The bit from here
            tree = scipy.spatial.cKDTree(points)
            matches = tree.query_ball_tree(tree, rad)
            # to here can be a function, that takes
            # points ands rad and returns tree and
            # matches

            for j, match in enumerate(matches):
                # There needs to be a function that
                # accepts a candidate match (cluster)
                # and returns a cluster key if the 
                # candidate match satisifies the criteria
                # and None otherwise.
                # The criteria can be the number of
                # tracklets on different nights, something
                # about RMS.
                # The function even might distill contaminated
                # clusters.
                if len(match)>=mincount:
                    cluster_list =[]
                    for idx in match:
                        cluster_list.append(label_dict[idx])
                        # At this point, collect the arrow information too,
                        # and append it to another list.
                    cluster_key = tuple(sorted(cluster_list))
                    # If the same cluster is found with different parameters
                    # we are only getting the most recent one.
                    cluster_arrow_dict[cluster_key] = [arrow_dict[k] for k in cluster_key]
                    cluster_counter.update({cluster_key: 1})

    return list(cluster_counter.keys()), cluster_arrow_dict

def do_run(pixels, tracklets_dict, t_ref, 
           g_select, dt, rad,
           transform_to_arrows_for_regions_function, g_gdots, mincount=3, nside=8): 

    pix_dict = index_tracklets_gamma(g_select, tracklets_dict, nside=nside)
    
    print("here1")
    
    pixel_results, tracklet_visits = transform_to_arrows_for_regions_function(t_ref, g_gdots, pixels, pix_dict, tracklets_dict)
    
    print("here2")

    results, cluster_arrow_dict = do_sky(dt, rad, pixel_results, mincount=mincount)
    
    return results, cluster_arrow_dict, pixel_results, tracklet_visits

def do_master(pixels, tracklets_dict, t_ref, 
              g_select, dt, rad,
              transform_to_arrows_for_regions_function, g_gdots, mincount=3, nside=8): 

    pix_dict = index_tracklets_gamma(g_select, tracklets_dict, nside=nside)
    
    print("here1")
    
    master, tracklet_visits = transform_to_arrows_for_regions_function(t_ref, g_gdots, pixels, pix_dict, tracklets_dict)
    
    return master

def do_clustering(pixels, tracklets_dict, t_ref, 
           g_select, dt, rad,
           master, g_gdots, mincount=3, nside=8): 

    results, cluster_arrow_dict = do_sky_v3(dt, rad, master, mincount=mincount)
    
    return results, cluster_arrow_dict


def get_date(line):
    return line[15:32]

def principal_value_0(theta):
    tc = theta.copy()
    tc -= 360.*np.floor(tc/360.)
    wrap = tc>180.
    tc[wrap]=tc[wrap]-360.
    return tc
