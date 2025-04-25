import numpy as np
import spiceypy as spice

from layup.utilities.file_io.ObjectDataReader import ObjectDataReader

# Characters we remove from column names.
_INVALID_COL_CHARS = "!#$%&‘()*+, ./:;<=>?@[\\]^{|}~"

_OUTPUT_DTYPE = [
    ("ObjID", "U10"),
    ("isotime", "U25"),
    ("raDeg", "f8"),
    ("decDeg", "f8"),
    ("mag", "f4"),
    ("filt", "U1"),
    ("obsCode", "U3"),
    ("cat", "U1"),
    ("prg", "U1"),
]


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

'''
    et = (jd_tdb - spice.j2000()) * 24 * 60 * 60

    if len(line.strip()) > 80:
        obsCode_test, geoc_pos = satellite_pos(line[80:])
        if obsCode_test != obsCode:
            print("obs codes not the same", "x", obsCode_test, "x", obsCode, "x")
    else:
        geoc_pos = tr.geocentricObservatory(et, obsCode)

    # Get the barycentric position of Earth
    if baryhelio == "bary":
        pos, _ = spice.spkpos("EARTH", et, "J2000", "NONE", "SSB")
    elif baryhelio == "helio":
        pos, _ = spice.spkpos("EARTH", et, "J2000", "NONE", "SUN")

    observatory_pos = pos + geoc_pos

    observatory_pos /= au_km
'''


# This routine opens and reads filename, separating the records into those in the 1-line and 2-line formats.
# The 2-line format lines are merged into single 160-character records for processing line-by-line.
def merge_MPC_file(filename, new_filename, comment_char="#"):
    with open(new_filename, "w") as f1_out:
        line1 = None
        with open(filename, "r") as f:
            print("opened ", filename)
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


# These routines convert the RA and Dec strings to floats.
def RA2degRA(RA):
    hr = RA[0:2]
    if hr.strip() == "":
        hr = 0.0
    else:
        hr = float(hr)
    mn = RA[3:5]
    if mn.strip == "":
        mn = 0.0
    else:
        mn = float(mn)
    sc = RA[6:]
    if sc.strip() == "":
        sc = 0.0
    else:
        sc = float(sc)
    degRA = 15.0 * (hr + 1.0 / 60.0 * (mn + 1.0 / 60.0 * sc))
    return degRA


def Dec2degDec(Dec):
    s = Dec[0]
    dg = Dec[1:3]
    if dg.strip() == "":
        dg = 0.0
    else:
        dg = float(dg)
    mn = Dec[4:6]
    if mn.strip() == "":
        mn = 0.0
    else:
        mn = float(mn)
    sc = Dec[7:]
    if sc.strip() == "":
        sc = 0.0
    else:
        sc = float(sc)
    degDec = dg + 1.0 / 60.0 * (mn + 1.0 / 60.0 * sc)
    if s == "-":
        degDec = -degDec
    return degDec


# Parses the date string from the 80-character record
def parseDate(dateObs):
    yr = dateObs[0:4]
    mn = dateObs[5:7]
    dy = dateObs[8:]
    return yr, mn, dy


def mpctime2isotime(mpctimeStr, digits=4):
    yr, mn, dy = parseDate(mpctimeStr)
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


# Grab a line of obs80 data and return the designations.
def get_obs80_id(line):
    objName = line[0:5]
    provDesig = line[5:12]
    if objName.strip() != "":
        objID = objName
    elif provDesig.strip() != "":
        objID = provDesig
    else:
        raise Exception("No object identifier" + objName + provDesig)
    return objID


# Grab a line of obs80 data and convert it to values
# this assumes the object is numbered.
def convertObs80(line, digits=4):
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

    cat = line[71].strip()

    prg = line[13].strip()

    if objName.strip() != "":
        objID = objName
    elif provDesig.strip() != "":
        objID = provDesig
    else:
        raise Exception("No object identifier" + objName + provDesig)
    iso_time = mpctime2isotime(dateObs, digits=digits)
    mag = float(mag) if mag.strip() != "" else 0.0
    raDeg, decDeg = RA2degRA(RA), Dec2degDec(Dec)
    return objID, iso_time, raDeg, decDeg, mag, filt, obsCode, cat, prg


# From google
from pathlib import Path


def append_to_filename(filepath, text_to_append):
    path = Path(filepath)
    new_filename = f"{path.stem}{path.suffix}{text_to_append}"
    new_filepath = path.with_name(new_filename)
    return new_filepath


class Obs80DataReader(ObjectDataReader):
    """A class to read in object data files stored in the MPC's obs80
    format.

    Note that we will ignore the header lines that might accompany the
    file.
    """

    def __init__(self, filename, **kwargs):
        """A class for reading observational data from an MPC obs80 file.

        Parameters
        ----------
        filename : string
            Location/name of the data file.

        sep : string, optional
            Format of input file ("whitespace"/"comma"/"csv").
            Default = csv

        **kwargs: dictionary, optional
            Extra arguments
        """
        super().__init__(**kwargs)
        self.filename = filename
        print(filename)

        self.filename_merge = append_to_filename(filename, "_merge")

        merge_MPC_file(filename, self.filename_merge)

        # Header lines for obs80 data are about observational
        # circumstances.  We identify and skip those lines.

        # To pre-validation the header information.
        # self._validate_header_line()
        self.header_row = 0  # The header row is always the first row

        # A table holding just the object ID for each row. Only populated
        # if we try to read data for specific object IDs.
        self.obj_id_table = None

        # A dictionary to hold the number of rows for each object ID. Only populated
        # if we try to read data for specific object IDs.
        self.obj_id_counts = {}

    def get_reader_info(self):
        """Return a string identifying the current reader name
        and input information (for logging and output).

        Returns
        --------
        name : string
            The reader information.
        """
        return f"Obs80DataReader:{self.filename}"

    def get_row_count(self):
        # Should this be the count of observation lines?
        """Return the total number of rows in the file.

        Returns
        -------
        int
            Total rows in the file.
        """
        with open(self.filename_merge, "r") as file:
            data = file.readlines()
        return len(data)

    def _read_rows_internal(self, block_start=0, block_size=None, **kwargs):
        """Reads in a set number of rows from the input.

        Parameters
        -----------
        block_start : integer, optional
            The 0-indexed row number from which
            to start reading the data. For example in a CSV file
            block_start=2 would skip the first two lines after the header
            and return data starting on row=2. Default =0

        block_size: integer, optional, default=None
            The number of rows to read in.
            Use block_size=None to read in all available data.
            default =None

        **kwargs : dictionary, optional
            Extra arguments

        Returns
        -----------
        res : numpy structured array
            The data read in from the file.
        """
        lines, records = [], []
        with open(self.filename_merge, "r") as file:
            lines = file.readlines()[block_start : block_start + block_size]
        for line in lines:
            objID, iso_time, raDeg, decDeg, mag, filt, obsCode, cat, prg = convertObs80(line[0:80])
            records.append(
                (
                    objID,
                    iso_time,
                    raDeg,
                    decDeg,
                    mag,
                    filt,
                    obsCode,
                    cat,
                    prg,
                )
            )

        for i in records[0]:
            print(i, type(i))

        return np.array(records, dtype=_OUTPUT_DTYPE)

    def _build_id_map(self):
        """Builds a table of just the object IDs"""

        if self.obj_id_table is not None:
            return

        with open(self.filename_merge, "r") as file:
            lines = file.readlines()
            print(len(lines))
            objIDs = [get_obs80_id(line) for line in lines]
            data_type = np.dtype([("provID", "U10")])
            objIDs = np.array(objIDs, dtype=data_type)
            self.obj_id_table = objIDs
            self.obj_id_table = self._validate_object_id_column(self.obj_id_table)

        # Create a dictionary of the object ID counts.
        for i in self.obj_id_table[self._primary_id_column_name]:
            self.obj_id_counts[i] = self.obj_id_counts.get(str(i), 0) + 1

    def _read_objects_internal(self, obj_ids, **kwargs):
        """Read in a chunk of data for given object IDs.

        Parameters
        -----------
        obj_ids : list
            A list of object IDs to use.

        **kwargs : dictionary, optional
            Extra arguments

        Returns
        -----------
        res : numpy structured array
            The data read in from the file.
        """
        self._build_id_map()
        print(self.obj_id_table)

        skipped_row = ~np.isin(self.obj_id_table[self._primary_id_column_name], obj_ids)
        print(skipped_row)

        records = []
        with open(self.filename_merge, "r") as file:
            lines = file.readlines()
            for line, sr in zip(lines, skipped_row, strict=False):
                if not sr:
                    objID, iso_time, raDeg, decDeg, mag, filt, obsCode, cat, prg = convertObs80(line[0:80])
                    records.append(
                        (
                            objID,
                            iso_time,
                            raDeg,
                            decDeg,
                            mag,
                            filt,
                            obsCode,
                            cat,
                            prg,
                        )
                    )
        return np.array(records, dtype=_OUTPUT_DTYPE)

    def _process_and_validate_input_table(self, input_table, **kwargs):
        """Perform any input-specific processing and validation on the input table.
        Modifies the input table in place.

        Notes
        -----
        The base implementation includes filtering that is common to most
        input types. Subclasses should call super.process_and_validate()
        to ensure that the ancestor’s validation is also applied.

        Parameters
        -----------
        input_table : numpy structured array
            A loaded table.

        **kwargs : dictionary, optional
            Extra arguments

        Returns
        -----------
        input_table: numpy structured array
            Returns the input table modified in-place.
        """
        # Perform the parent class's validation (checking object ID column).
        input_table = super()._process_and_validate_input_table(input_table, **kwargs)

        # Strip out the whitespace from the column names.
        input_table.dtype.names = [name.strip() for name in input_table.dtype.names]

        return input_table
