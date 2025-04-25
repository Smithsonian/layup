import numpy as np

from layup.utilities.file_io.ObjectDataReader import ObjectDataReader

# The output data type for the structured array of the obs80 reader
_OUTPUT_DTYPE = [
    ("ObjID", "U10"),
    ("obstime", "U25"),
    ("raDeg", "f8"),
    ("decDeg", "f8"),
    ("mag", "f4"),
    ("filt", "U1"),
    ("obsCode", "U3"),
    ("cat", "U1"),
    ("prg", "U1"),
    ("obs_geo_x", "f8"),
    ("obs_geo_y", "f8"),
    ("obs_geo_z", "f8"),
]


def is_two_line_row(line):
    """Checks if the line is the first line of a two-line row format."""
    note2 = line[14]
    return note2 == "S" or note2 == "R" or note2 == "V"


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


# Grab a line of obs80 data and return the designations.
def get_obs80_id(line):
    # TODO revisit if we should cast as string
    # TODO should we strip directly when reading in
    objName = line[0:5]
    provDesig = line[5:12]
    if objName.strip() != "":
        objID = objName
    elif provDesig.strip() != "":
        objID = provDesig
    else:
        raise Exception("No object identifier" + objName + provDesig)
    return str(objID)


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

    # The geo position of the observatory is not always provided in the obs80 file.
    obs_geo_x, obs_geo_y, obs_geo_z = np.nan, np.nan, np.nan
    if len(line.rstrip()) > 80:
        # We process the observatory positions which were mereged in from a second line
        # TODO revisit
        second_line = line[80:]
        if len(second_line) < 80:
            raise ValueError(
                f"Observatory position line is too short for {objID} and line of lenght {len(second_line)} {second_line}"
            )
        if second_line[77:80].rstrip() != obsCode:
            raise ValueError(
                f"Observatory codes do not match in the seond line provided for the observatory position. {obsCode} and {second_line[77:80].rstrip()}"
            )
        flag = second_line[32:34]
        if flag == "1 " or flag == "2 ":
            obs_geo_x = float(second_line[34] + second_line[35:45].strip())
            obs_geo_y = float(second_line[46] + second_line[47:57].strip())
            obs_geo_z = float(second_line[58] + second_line[59:69].strip())

    return objID, iso_time, raDeg, decDeg, mag, filt, obsCode, cat, prg, obs_geo_x, obs_geo_y, obs_geo_z


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
        """Return the total number of rows in the file.

        Returns
        -------
        int
            Total rows in the file.
        """
        row_cnt = 0
        with open(self.filename, "r") as f:
            for line in f:
                if line.strip() != "" and not is_two_line_row(line):
                    row_cnt += 1

        return row_cnt

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
        records = []
        with open(self.filename, "r") as f:
            curr_block = block_start
            block_end = block_start + block_size if block_size is not None else None
            prev_line = None
            for curr_line in f:
                if block_end is not None and curr_block >= block_end:
                    # We have read enough rows from the file.
                    break
                if is_two_line_row(curr_line):
                    # We have a two-line row. We will save our current line
                    # and wait for the next line to merge them as a single row to process.
                    prev_line = curr_line
                    continue
                row_to_process = curr_line
                if prev_line is not None:
                    row_to_process = prev_line.rstrip("\n") + curr_line
                    prev_line = None

                records.append(convertObs80(row_to_process))
                curr_block += 1

        return np.array(records, dtype=_OUTPUT_DTYPE)

    def _build_id_map(self):
        """Builds a table of just the object IDs"""

        if self.obj_id_table is not None:
            return

        objIDs = []
        with open(self.filename, "r") as f:
            for curr_line in f:
                if is_two_line_row(curr_line):
                    # We have a two-line row, so skip it and only
                    # add the object ID from the final row.
                    continue
                objID = get_obs80_id(curr_line)
                objIDs.append(objID)
                self.obj_id_counts[objID] = self.obj_id_counts.get(objID, 0) + 1

        self.obj_id_table = np.array(objIDs, dtype=np.dtype([("ObjID", "U10")]))
        self.obj_id_table = self._validate_object_id_column(self.obj_id_table)

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
        curr_row_idx = -1
        prev_line = None
        with open(self.filename, "r") as f:
            for curr_line in f:
                if is_two_line_row(curr_line):
                    # We have a two-line row. We will save our current line
                    # and wait for the next line to merge them as a single row to process.
                    prev_line = curr_line
                    continue
                curr_row_idx += 1
                if skipped_row[curr_row_idx]:
                    continue
                row_to_process = curr_line
                if prev_line is not None:
                    row_to_process = prev_line.rstrip("\n") + curr_line
                    prev_line = None

                records.append(convertObs80(row_to_process))
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
