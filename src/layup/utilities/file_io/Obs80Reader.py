import numpy as np

from layup.utilities.file_io.ObjectDataReader import ObjectDataReader

# The output data type for the structured array of the obs80 reader
_OUTPUT_DTYPE = [
    ("provID", "U10"),
    ("obstime", "U25"),
    ("ra", "f8"),
    ("dec", "f8"),
    ("mag", "f4"),
    ("filt", "U1"),
    ("stn", "U3"),
    ("cat", "U1"),
    ("prg", "U1"),
    ("obs_geo_x", "f8"),
    ("obs_geo_y", "f8"),
    ("obs_geo_z", "f8"),
]


def two_line_row_start(line):
    """Checks if the MPC Obs80 line is the first line of a two-line row format."""
    note2 = line[14]
    return note2 == "S" or note2 == "R" or note2 == "V"


def ra_to_deg_ra(ra):
    """Converts the Obs80 RA string to degrees."""
    hr = ra[0:2].strip()
    hr = float(hr) if hr != "" else 0.0

    mn = ra[3:5].strip()
    mn = float(mn) if mn != "" else 0.0

    sc = ra[6:].strip()
    sc = float(sc) if sc != "" else 0.0

    deg_ra = 15.0 * (hr + 1.0 / 60.0 * (mn + 1.0 / 60.0 * sc))
    return deg_ra


def dec_to_deg_dec(dec):
    """Converts the Obs80 Dec string to degrees."""
    dg = dec[1:3].strip()
    dg = float(dg) if dg != "" else 0.0

    mn = dec[4:6].strip()
    mn = float(mn) if mn != "" else 0.0

    sc = dec[7:].strip()
    sc = float(sc) if sc != "" else 0.0

    deg_dec = dg + 1.0 / 60.0 * (mn + 1.0 / 60.0 * sc)

    sign = dec[0]
    if sign == "-":
        deg_dec = -deg_dec
    return deg_dec


def mpctime_to_isotime(mpc_time):
    """Converts from the MPC time formatted string to ISO time format."""
    yr = mpc_time[0:4]
    mn = mpc_time[5:7]
    dy = float(mpc_time[8:])

    frac_day, day = np.modf(dy)
    frac_hrs, hrs = np.modf(frac_day * 24)
    frac_mins, mins = np.modf(frac_hrs * 60)
    secs = frac_mins * 60

    if np.round(secs, 4) >= 60.0:
        secs = np.round(secs, 4) - 60
        if secs < 0.0:
            secs = 0.0
        mins += 1

    if mins >= 60:
        mins -= 60
        hrs += 1

    if hrs >= 24:
        hrs -= 24
        day += 1  # Could mess up the number of days in the month

    format_string = "%4s-%2s-%02dT%02d:%02d:%02." + str(4) + "f"
    return format_string % (yr, mn, day, hrs, mins, secs)


def get_obs80_id(line):
    """Get the object ID from the Obs80 line. Using the object name if provided,
    otherwise using the provisional ID."""
    obj_name = line[0:5].strip()
    prov_id = line[5:12].strip()
    if obj_name != "":
        obj_id = obj_name
    elif prov_id != "":
        obj_id = prov_id
    else:
        raise Exception(f"No object identifier: Name was {obj_name} and provId was {prov_id}")
    return obj_id


def convert_obs80(line, second_line=None):
    """
    Converts a row of obs80 data to a tuple of values.
    The second line is optional and may contain the observatory position.

    Parameters
    ----------
    line : str
        The line of obs80 data to convert.
    second_line : str, optional
        The optional second line of obs80 data to convert. Default is None.

    Returns
    -------
    tuple
        A tuple of values containing the object ID, ISO time, RA in degrees,
        Dec in degrees, magnitude, filter, observatory code, catalog, program,
        and observatory position (x, y, z).
    """
    # Extract the relevant fields from the first mpc obs80 line.
    obj_name = line[0:5].strip()
    prov_id = line[5:12].strip()
    prg = line[13].strip()
    obstime = line[15:32].strip()
    ra = line[32:44].strip()
    dec = line[44:56].strip()
    mag = line[65:70].strip()
    filt = line[70:71].strip()
    cat = line[71].strip()
    obs_code = line[77:80].strip()

    # Use the object name as object ID if provided, otherwise use the provisional ID.
    if prov_id != "":
        obj_id = prov_id
    elif obj_name != "":
        obj_id = obj_name
    else:
        raise Exception(f"No object identifier: Name was {obj_name} and provId was {prov_id}")

    # Do any unit and type conversions. Note that various layup verbs will likely
    # convert these values again to their internal formats.
    iso_time = mpctime_to_isotime(obstime)
    ra_deg, dec_deg = ra_to_deg_ra(ra), dec_to_deg_dec(dec)
    mag = float(mag) if mag != "" else 0.0

    # Process the observatory position if provided.
    obs_geo_x, obs_geo_y, obs_geo_z = np.nan, np.nan, np.nan
    if second_line is not None:
        # Check that the second line is long enough to contain the observatory position.
        if len(second_line) < 80:
            raise ValueError(
                f"Observatory position line is too short for {obj_id} and line (with length {len(second_line)}: {second_line}"
            )
        if second_line[77:80].strip() != obs_code:
            raise ValueError(
                f"Observatory codes do not match in the second line provided for the observatory position. {obsCode} and {second_line[77:80].rstrip()}"
            )
        flag = second_line[32:34].strip()
        if flag in ["1", "2"]:
            # For each coordinate, the first character is a sign (+/-) and the next 10 characters are the value.
            obs_geo_x = float(second_line[34] + second_line[35:45].strip())
            obs_geo_y = float(second_line[46] + second_line[47:57].strip())
            obs_geo_z = float(second_line[58] + second_line[59:69].strip())

    return obj_id, iso_time, ra_deg, dec_deg, mag, filt, obs_code, cat, prg, obs_geo_x, obs_geo_y, obs_geo_z


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

        **kwargs: dictionary, optional
            Extra arguments
        """
        super().__init__(**kwargs)
        self.filename = filename

        self._primary_id_column_name = "provID"

        # A table holding just the object ID for each row. Only populated
        # if we try to read data for specific object IDs.
        self.obj_id_table = None

        # A dictionary to hold the number of rows for each object ID. Only populated
        # if we try to read data for specific object IDs.
        self.obj_id_counts = {}

    def _is_header_row(self, line):
        """Check if the line is a header row.

        Parameters
        ----------
        line : str
            The line to check.

        Returns
        -------
        bool
            True if the line is a header row, False otherwise.
        """
        # We know a line is a header row if it starts with an all-caps 3 letter code
        # followed by a space.
        return line[0:3].isupper() and line[3] == " "

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

        Note that the obs 80 format allows for two-line rows, so the number of
        lines used to store the data is not the same as the number of rows.

        Returns
        -------
        int
            Total rows in the file.
        """
        row_cnt = 0
        with open(self.filename, "r") as f:
            for line in f:
                # Skip empty lines, header rows, and the starting line of two-line rows.
                if line.strip() != "" and not self._is_header_row(line) and not two_line_row_start(line):
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
            curr_block = 0
            block_end = block_start + block_size if block_size is not None else None
            prev_line = None
            check_header = True
            for curr_line in f:
                if check_header and self._is_header_row(curr_line):
                    continue
                else:
                    check_header = False
                if block_end is not None and curr_block >= block_end:
                    # We have read enough rows from the file.
                    break
                if two_line_row_start(curr_line):
                    # We have a two-line row. We will save our current line
                    # and wait for the next line to merge them as a single row to process.
                    prev_line = curr_line
                    continue

                # Process our current MPC Obs80 row.
                if curr_block >= block_start:
                    if prev_line is not None:
                        # We have a two-line row to process.
                        records.append(convert_obs80(prev_line, second_line=curr_line))
                        # Remove the previous line so we don't process it again.
                        prev_line = None
                    else:
                        # We have a single line to process.
                        records.append(convert_obs80(curr_line))
                curr_block += 1

        return np.array(records, dtype=_OUTPUT_DTYPE)

    def _build_id_map(self):
        """Builds a table of just the object IDs"""

        if self.obj_id_table is not None:
            return

        obj_ids = []
        with open(self.filename, "r") as f:
            check_header = True
            for curr_line in f:
                if check_header and self._is_header_row(curr_line):
                    continue
                else:
                    check_header = False
                if two_line_row_start(curr_line):
                    # We have a two-line row, so skip it and only
                    # add the object ID from the final row.
                    continue
                obj_id = get_obs80_id(curr_line)
                obj_ids.append(obj_id)
                # Count the number of times we see this object ID.
                self.obj_id_counts[obj_id] = self.obj_id_counts.get(obj_id, 0) + 1

        self.obj_id_table = np.array(obj_ids, dtype=np.dtype([("provID", "U10")]))
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

        # Find which rows map to the requested object ID list
        skipped_rows = ~np.isin(self.obj_id_table[self._primary_id_column_name], obj_ids)

        records = []
        # The index of the current row we are processing to check against skipped_rows.
        # We start at -1 because we will increment it before processing the first row.
        curr_row_idx = -1
        prev_line = None
        check_header = True
        with open(self.filename, "r") as f:
            for curr_line in f:
                if check_header and self._is_header_row(curr_line):
                    continue
                else:
                    check_header = False
                if two_line_row_start(curr_line):
                    # We have a two-line row. We will save our current line
                    # and wait for the next line to merge them as a single row to process.
                    prev_line = curr_line
                    continue

                # We're at a potentially processable row, so increment our index.
                curr_row_idx += 1
                if skipped_rows[curr_row_idx]:
                    continue

                # Process our current MPC Obs80 row.
                if prev_line is not None:
                    records.append(convert_obs80(prev_line, curr_line))
                    # Remove the previous line so we don't process it again.
                    prev_line = None
                else:
                    # Our row is a single line to process.
                    records.append(convert_obs80(curr_line))
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
