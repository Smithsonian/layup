import numpy as np

from layup.utilities.file_io.ObjectDataReader import ObjectDataReader

# Column 15 (0-indexed 14) is the MPC "note 2" / observation-type code. The
# codes S (satellite), R (radar) and V (roving observer) each emit a SECOND
# line carrying the observer position/data; that continuation line repeats the
# designation and carries the same code in lower case (s / r / v).
_TWO_LINE_FIRST_NOTES = ("S", "R", "V")
_TWO_LINE_CONT_NOTES = ("s", "r", "v")
# Note 2 codes X / x flag an observation the MPC has deleted or replaced. Such a
# line is not a usable observation and must not be read as one -- in particular a
# deleted satellite (C-code) astrometry line carries no observer position, so
# emitting it produces a positionless record that would fall back to a per-row
# JPL Horizons lookup during fitting.
_DELETED_NOTES = ("X", "x")


def deleted_observation(line):
    """Checks if the MPC Obs80 line is a deleted/replaced observation (note 2
    code X or x), which should be skipped entirely."""
    return len(line) > 14 and line[14] in _DELETED_NOTES


def two_line_row_start(line):
    """Checks if the MPC Obs80 line is the first line of a two-line row format.

    A line bearing an upper-case S, R or V in note 2 (column 15) is the first
    of a two-line record.
    """
    return len(line) > 14 and line[14] in _TWO_LINE_FIRST_NOTES


def two_line_row_continuation(line):
    """Checks if the MPC Obs80 line is the second (continuation) line of a
    two-line record.

    The continuation line repeats note 2 in lower case (s / r / v) and carries
    the observer position rather than an astrometric measurement. It must never
    be emitted as a standalone observation: on its own its position columns
    would be misread as RA/Dec and it would carry no observatory position.
    """
    return len(line) > 14 and line[14] in _TWO_LINE_CONT_NOTES


def two_line_rows_match(first_line, second_line):
    """Checks that ``second_line`` is the continuation line belonging to
    ``first_line``: it is a continuation line, repeats the same designation
    (columns 1-14) and reports the same observatory code (columns 78-80).

    This guards against a desynchronised file (an orphan continuation line, or
    a first line whose continuation is missing) silently mis-pairing.
    """
    if not two_line_row_continuation(second_line):
        return False
    if first_line[0:14] != second_line[0:14]:
        return False
    return first_line[77:80] == second_line[77:80]


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

        # The output data type for the structured array of the obs80 reader
        self.output_dtype = [
            (self._primary_id_column_name, "U10"),
            ("obsTime", "U25"),
            ("ra", "f8"),
            ("dec", "f8"),
            ("mag", "f4"),
            ("filt", "U1"),
            ("stn", "U3"),
            ("cat", "U1"),
            ("prg", "U1"),
            ("sys", "U7"),
            ("ctr", "i4"),
            ("pos1", "f8"),
            ("pos2", "f8"),
            ("pos3", "f8"),
        ]

        # define the column slices for the obs80 format
        self.col_names = {
            "ObjID": slice(0, 5),
            "provID": slice(5, 12),
            "prg": slice(13, 14),
            "obsTime": slice(15, 32),
            "ra": slice(32, 44),
            "dec": slice(44, 56),
            "mag": slice(65, 70),
            "filt": slice(70, 71),
            "cat": slice(71, 72),
            "obs_code": slice(77, 80),
        }

        if self._primary_id_column_name not in self.col_names:
            raise ValueError(
                f"Primary ID column name '{self._primary_id_column_name}' not found in Obs80 column definitions. "
                "Valid column names are: " + ", ".join(self.col_names.keys())
            )

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
        # followed by a space. The length guard tolerates a blank/short leading
        # line (issue #407).
        return len(line) >= 4 and line[0:3].isupper() and line[3] == " "

    def get_reader_info(self):
        """Return a string identifying the current reader name
        and input information (for logging and output).

        Returns
        --------
        name : string
            The reader information.
        """
        return f"Obs80DataReader:{self.filename}"

    def _iter_records(self, f):
        """Yield one ``(main_line, second_line)`` tuple per logical obs80 record.

        This is the single source of truth for how the raw lines of the file
        group into records; every read path (``get_row_count``, ``read_rows``,
        ``read_objects``) walks it so their record counts and ordering always
        agree. It pairs each two-line record (S/R/V first line + its lower-case
        s/r/v continuation line) and, crucially, refuses to emit a malformed
        observation:

        * a deleted/replaced observation (note 2 code X / x) is skipped;
        * an orphan continuation line (one with no matching preceding first
          line) is skipped -- it carries no astrometry of its own and would
          otherwise be emitted as a positionless observation whose position
          columns are misread as RA/Dec;
        * a first line whose continuation is missing is dropped rather than
          paired with the next unrelated line.

        ``main_line`` is the astrometry line; ``second_line`` is the
        observer-position line, or ``None`` for a single-line observation.
        """
        prev_first = None
        check_header = True
        for line in f:
            if check_header and self._is_header_row(line):
                continue
            check_header = False
            # Skip blank / truncated lines (issue #407): note 2 is at column 15,
            # so anything shorter cannot be an obs80 record.
            if len(line.rstrip("\n")) < 15:
                continue
            if deleted_observation(line):
                # A deleted/replaced observation. If it was the first line of a
                # two-line record its continuation is now orphaned and will be
                # skipped by the branch below.
                prev_first = None
                continue
            if two_line_row_start(line):
                # Start of a two-line record. Any unconsumed previous first line
                # had no continuation and is dropped.
                prev_first = line
                continue
            if two_line_row_continuation(line):
                if prev_first is not None and two_line_rows_match(prev_first, line):
                    yield prev_first, line
                # else: orphan / mismatched continuation -> skip (emit nothing).
                prev_first = None
                continue
            # A normal single-line observation. Any unconsumed previous first
            # line lacked its continuation and is dropped.
            prev_first = None
            yield line, None

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
            for _ in self._iter_records(f):
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
        block_end = block_start + block_size if block_size is not None else None
        with open(self.filename, "r") as f:
            for curr_block, (main_line, second_line) in enumerate(self._iter_records(f)):
                if block_end is not None and curr_block >= block_end:
                    # We have read enough rows from the file.
                    break
                if curr_block >= block_start:
                    records.append(self.convert_obs80(main_line, second_line=second_line))

        return np.array(records, dtype=self.output_dtype)

    def _build_id_map(self):
        """Builds a table of just the object IDs"""

        if self.obj_id_table is not None:
            return

        obj_ids = []
        with open(self.filename, "r") as f:
            for main_line, _second_line in self._iter_records(f):
                obj_id = self.get_obs80_id(main_line)
                obj_ids.append(obj_id)
                # Count the number of times we see this object ID.
                self.obj_id_counts[obj_id] = self.obj_id_counts.get(obj_id, 0) + 1

        self.obj_id_table = np.array(obj_ids, dtype=np.dtype([(self._primary_id_column_name, "U10")]))
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
        with open(self.filename, "r") as f:
            # _iter_records enumerates records in the same order as _build_id_map,
            # so curr_row_idx lines up with skipped_rows.
            for curr_row_idx, (main_line, second_line) in enumerate(self._iter_records(f)):
                if skipped_rows[curr_row_idx]:
                    continue
                records.append(self.convert_obs80(main_line, second_line=second_line))
        return np.array(records, dtype=self.output_dtype)

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

    def get_obs80_id(self, line):
        """Get the object ID from the Obs80 line. Note that we have already confirmed
        that `self.primary_id_column_name` is in `self.col_names`.
        Parameters
        ----------
        line : str
            The line of obs80 data to extract the object ID from.

        Returns
        -------
        str
            The object ID extracted from the line.
        """
        # Use the object name as object ID if provided, otherwise use the provisional ID.
        return line[self.col_names[self._primary_id_column_name]].strip()

    def convert_obs80(self, line, second_line=None):
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
        # obj_name = line[self.col_names["obj_name"]].strip()
        # prov_id = line[self.col_names["prov_id"]].strip()
        prg = line[self.col_names["prg"]].strip()
        obstime = line[self.col_names["obsTime"]].strip()
        ra = line[self.col_names["ra"]].strip()
        dec = line[self.col_names["dec"]].strip()
        mag = line[self.col_names["mag"]].strip()
        filt = line[self.col_names["filt"]].strip()
        cat = line[self.col_names["cat"]].strip()
        obs_code = line[self.col_names["obs_code"]].strip()

        # Use the object name as object ID if provided, otherwise use the provisional ID.
        obj_id = line[self.col_names[self._primary_id_column_name]].strip()

        # Do any unit and type conversions. Note that various layup verbs will likely
        # convert these values again to their internal formats.
        iso_time = mpctime_to_isotime(obstime)
        ra_deg, dec_deg = ra_to_deg_ra(ra), dec_to_deg_dec(dec)
        mag = float(mag) if mag != "" else 0.0

        # Process the observatory position if provided.
        obs_geo_x, obs_geo_y, obs_geo_z = np.nan, np.nan, np.nan
        ades_sys = ""
        if second_line is not None:
            # Check that the second line is long enough to contain the observatory position.
            if len(second_line) < 80:
                raise ValueError(
                    f"Observatory position line is too short for {obj_id} and line (with length {len(second_line)}: {second_line}"
                )
            if second_line[77:80].strip() != obs_code:
                raise ValueError(
                    f"Observatory codes do not match in the second line provided for the observatory position. {obs_code} and {second_line[77:80].rstrip()}"
                )
            # The three two-line record types share the S/R/V mechanism but carry
            # different second-line payloads, so dispatch on the first line's note2
            # rather than assuming a satellite geocentric position (issue #402).
            record_type = line[14]
            if record_type == "S":
                # Satellite: geocentric equatorial (ICRF) position, km or AU.
                unit_flag = second_line[32:34].strip()
                if unit_flag in ["1", "2"]:
                    ades_sys = "ICRF_KM" if unit_flag == "1" else "ICRF_AU"
                    # For each coordinate, the first character is a sign (+/-) and the next 10 characters are the value.
                    obs_geo_x = float(second_line[34] + second_line[35:45].strip())
                    obs_geo_y = float(second_line[46] + second_line[47:57].strip())
                    obs_geo_z = float(second_line[58] + second_line[59:69].strip())
                else:
                    raise ValueError(
                        f"Unknown observatory position unit flag '{unit_flag}' in the second line of obs80 data. Should be '1' (km) or '2' (AU)."
                    )
            elif record_type == "V":
                # Roving observer: geodetic East longitude / latitude (degrees) and
                # altitude (metres) on the WGS84 ellipsoid. We capture the position
                # and its frame here; the geodetic -> geocentric-ICRF conversion is
                # the observatory's job (issue #282).
                ades_sys = "WGS84"
                obs_geo_x = float(second_line[33:45])  # East longitude (deg)
                obs_geo_y = float(second_line[45:56])  # latitude (deg)
                obs_geo_z = float(second_line[56:67])  # altitude (m)
            elif record_type == "R":
                # Radar: the second line carries range/Doppler, not an observer
                # position. Radar is ingested via ADES delay/doppler, not here.
                raise ValueError(
                    f"Radar (R/r) obs80 two-line records are not supported by Obs80DataReader "
                    f"(object {obj_id}); ingest radar via ADES delay/doppler columns."
                )
            else:
                raise ValueError(f"Unexpected two-line record type '{record_type}' for object {obj_id}.")

        return (
            obj_id,
            iso_time,
            ra_deg,
            dec_deg,
            mag,
            filt,
            obs_code,
            cat,
            prg,
            ades_sys,
            399,  # The 'ctr' code for the Earth
            obs_geo_x,
            obs_geo_y,
            obs_geo_z,
        )
