dir_path = "/Users/mholman/Dropbox/support/"

import numpy as np
import spiceypy as spice

import MPC_library_small as MPC_library

# Load a few spice kernels
spice.furnsh(dir_path + "/kernels/MetaK_new.txt")

Observatories = MPC_library.Observatory(dir_path + "ObsCodes.txt")
ObservatoryXYZ = Observatories.ObservatoryXYZ

au2m = 149597870700
au_km = au2m / 1000.0


# This routine checks the 80-character input line to see if it contains a special character (S, R, or V) that indicates a 2-line
# record.
def is_two_line(line):
    note2 = line[14]
    obsCode = line[77:80]
    return note2 == "S" or note2 == "R" or note2 == "V"


def satellite_pos(second_line):
    obsCode = second_line[77:81].rstrip()
    flag = second_line[32:34]
    if flag == "1 " or flag == "2 ":
        pos = [
            float(second_line[34] + second_line[35:45].strip()),
            float(second_line[46] + second_line[47:57].strip()),
            float(second_line[58] + second_line[59:69].strip()),
        ]
        pos = np.array(pos)
    else:
        pos = None
    return obsCode, pos


# This routine opens and reads filename, separating the records into those in the 1-line and 2-line formats.
# The 2-line format lines are merged into single 160-character records for processing line-by-line.
def split_MPC_file(filename):
    filename_1_line = filename.rstrip(".txt") + "_1_line.txt"
    filename_2_line = filename.rstrip(".txt") + "_2_line.txt"
    with open(filename_1_line, "w") as f1_out, open(filename_2_line, "w") as f2_out:
        line1 = None
        with open(filename, "r") as f:
            for line in f:
                if is_two_line(line):
                    line1 = line
                    continue
                if line1 != None:
                    merged_lines = line1.rstrip("\n") + line
                    f2_out.write(merged_lines)
                    line1 = None
                else:
                    f1_out.write(line)
                    line1 = None


def mpctime2isotime(mpctimeStr, digits=4):
    yr, mn, dy = MPC_library.parseDate(mpctimeStr)
    dy = float(dy)
    frac_day, day = np.modf(dy)
    frac_hrs, hrs = np.modf(frac_day * 24)
    frac_mins, mins = np.modf(frac_hrs * 60)
    secs = frac_mins * 60
    if np.round(secs, digits) >= 60.0:
        secs = np.round(secs, digits) - 60
        if secs < 0.0:
            secs = 0.0
        mins += 1
    if mins >= 60:
        mins -= 60
        hrs += 1
    if hrs >= 24:
        hrs -= 24
        day += 1  # Could mess up the number of days in the month
    formatStr = "%4s-%2s-%02dT%02d:%02d:%02." + str(digits) + "f"
    isoStr = formatStr % (yr, mn, day, hrs, mins, secs)
    return isoStr


def mpctime2et(mpctimeStr, digits=4):
    isoStr = mpctime2isotime(mpctimeStr, digits=digits)
    return spice.str2et(isoStr)


# Grab a line of obs80 data and convert it to values
# this assumes the object is numbered.
def convertObs80(line):
    objName = line[0:5]
    provDesig = line[5:12]
    disAst = line[12:13]
    note1 = line[13:14]
    note2 = line[14:15]
    dateObs = line[15:32]
    RA = line[32:44]
    Dec = line[44:56]
    mag = line[65:70]
    filt = line[70:71]
    obsCode = line[77:80]

    if objName.strip() != "":
        objID = objName
    elif provDesig.strip() != "":
        objID = provDesig
    else:
        raise Exception("No object identifier" + objName + provDesig)
    t = mpctime2et(dateObs)
    jd_tdb = spice.j2000() + t / (24 * 60 * 60)
    raDeg, decDeg = MPC_library.RA2degRA(RA), MPC_library.Dec2degDec(Dec)
    return objID, jd_tdb, raDeg, decDeg, mag, filt, 0.0, 0.0, 0.0, obsCode, 0.0


from functools import cache


@cache
def geocentricObservatory(et, obsCode):
    # et is JPL's internal time

    # Get the matrix that rotates from the Earth's equatorial
    # body fixed frame to the J2000 equatorial frame.
    #
    # For dates before 1972-01-1 use the older model
    # otherwise the more accurate model
    current = spice.str2et("1972-01-01")
    if et < current:
        m = spice.pxform("IAU_EARTH", "J2000", et)
    else:
        m = spice.pxform("ITRF93", "J2000", et)

    # Get the MPC's unit vector from the geocenter to
    # the observatory
    obsVec = Observatories.ObservatoryXYZ[obsCode]
    obsVec = np.array(obsVec)

    # Carry out the rotation and scale
    mVec = np.dot(m, obsVec) * 6378.137

    return mVec


# This rotation is taking things from equatorial to ecliptic
rot_mat = MPC_library.rotate_matrix(-MPC_library.Constants.ecl)


def equatorial_to_ecliptic(v, rot_mat=rot_mat):
    return np.dot(v, rot_mat.T)


def ecliptic_to_equatorial(v, rot_mat=rot_mat.T):
    return np.dot(v, rot_mat.T)


# This routine opens and reads filename, separating the records into those in the 1-line and 2-line formats.
# The 2-line format lines are merged into single 160-character records for processing line-by-line.
def merge_MPC_file(filename, new_filename, comment_char="#"):
    with open(new_filename, "w") as f1_out:
        line1 = None
        with open(filename, "r") as f:
            for line in f:
                if line.startswith(comment_char):
                    continue
                if is_two_line(line):
                    line1 = line
                    continue
                if line1 != None:
                    merged_lines = line1.rstrip("\n") + line
                    f1_out.write(merged_lines)
                    line1 = None
                else:
                    f1_out.write(line)
                    line1 = None


def principal_value_0(theta):
    tc = theta.copy()
    tc -= 360.0 * np.floor(tc / 360.0)
    wrap = tc > 180.0
    tc[wrap] = tc[wrap] - 360.0
    return tc
