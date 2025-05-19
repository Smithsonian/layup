import logging
import sys

import numpy as np
import pandas as pd

from layup.utilities.file_io.ObjectDataReader import ObjectDataReader

# Characters we remove from column names.
_INVALID_COL_CHARS = "!#$%&‘()*+, ./:;<=>?@[\\]^{|}~"

# Note that the separators (aside from whitespace) are all single characters. This
# is important as it means that in all cases of calling `read_csv` the C parser
# will be used under the hood. If the separators are multiple characters, other
# than `/s+` the slower Python parser will be used. See the pandas.read_csv doc:
# https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
VALID_FILE_FORMATS = {
    "csv": ",",
    "comma": ",",
    ",": ",",
    "whitespace": "\\s+",
    "psv": "|",
    "pipe": "|",
    "|": "|",
}

# Any pre-header line that starts with one of these strings will be ignored.
# CAUTION! - Avoid adding a character that would exclude the column header line.
PRE_HEADER_COMMENT_AND_EXCLUDE_STRINGS = ("#", "!")


class CSVDataReader(ObjectDataReader):
    """A class to read in object data files stored as CSV or whitespace
    separated values.

    Note that we require the header line to be the first line of the file
    """

    def __init__(self, filename, sep="csv", **kwargs):
        """A class for reading the object data from a CSV file.

        Parameters
        ----------
        filename : string
            Location/name of the data file.

        sep : string, optional
            Format of input file. The following are supported:
            - "whitespace" for text files
            - "comma", "csv" for CSV files
            - "pipe", "psv", "|" for pipe-separated values (PSV) files
            Default = csv

        **kwargs: dictionary, optional
            Extra arguments
        """
        super().__init__(**kwargs)
        self.filename = filename

        if sep not in VALID_FILE_FORMATS.keys():
            logger = logging.getLogger(__name__)
            logger.error(f"ERROR: Unrecognized delimiter ({sep})")
            sys.exit(f"ERROR: Unrecognized delimiter ({sep})")
        self.sep = sep

        self.data_separator = VALID_FILE_FORMATS[self.sep]

        # Number of lines of comments before the header line.
        self.num_pre_header_lines = 0
        # The header row is always the first row after the pre-header lines.
        self.header_row_index = 0

        # To pre-validation the header information.
        self._validate_header_line()

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
        return f"CSVDataReader:{self.filename}"

    def get_row_count(self):
        """Return the total number of rows in the [C|P|W]SV file.

        Returns
        -------
        int
            Total rows in the first key of the input [C|P|W]SV file.
        """

        if self.sep == "csv" or self.sep == "comma":
            delimiter = ","
        elif self.sep == "psv" or self.sep == "pipe" or self.sep == "|":
            delimiter = "|"
        elif self.sep == "whitespace":
            delimiter = None

        data = np.genfromtxt(
            self.filename,
            delimiter=delimiter,
            names=True,
            dtype=None,
            encoding="utf8",
            deletechars=_INVALID_COL_CHARS,
            ndmin=1,  # Ensure we always get a structured array even with a single result
            usecols=(0,),  # Only read in the first column, self._primary_id_column_name
            skip_header=self.num_pre_header_lines,
        )

        return len(data)

    def _validate_header_line(self):
        """Read and validate the header line (first line of the file)"""
        logger = logging.getLogger(__name__)
        with open(self.filename) as fh:
            for i, line in enumerate(fh):
                # If the line starts with a comment character, increment the pre-header line count
                if line.startswith(PRE_HEADER_COMMENT_AND_EXCLUDE_STRINGS):
                    # Skip comment lines
                    self.num_pre_header_lines += 1
                else:
                    logger.info(f"Reading the first line of {self.filename} as header:\n{line}")
                    self._check_header_line(line)
                    # Note - header row INDEX is 0-indexed.
                    self.header_row_index = self.num_pre_header_lines
                    return

                if i >= 100:
                    # If we have read 100 lines and not found a valid header line, exit.
                    error_str = (
                        f"ERROR: CSVReader: column headings not found in the first 100 lines of {self.filename}. "
                        f"Ensure column headings exist in input files and first column is {self._primary_id_column_name}."
                    )
                    logger.error(error_str)
                    sys.exit(error_str)

        # If we reach here, we did not find a valid header line.
        error_str = (
            f"ERROR: CSVReader: column headings not found in the first lines of {self.filename}. "
            f"Ensure column headings exist in input files and first column is {self._primary_id_column_name}."
        )
        logger.error(error_str)
        sys.exit(error_str)

    def _check_header_line(self, header_line):
        """Check that a given header line is valid and exit if it is invalid.

        Parameters
        ----------
        header_line : str
            The proposed header line.
        """
        logger = logging.getLogger(__name__)

        # This is a bit ugly, but splitting the header in this way, means that we
        # can generally define the value separators at the top of the file, _and_
        # use pandas C parser when we call `read_csv`, which is significantly
        # faster than the alternatively Python parser.
        if self.sep == "whitespace":
            column_names = header_line.split()
        else:
            column_names = [col.strip() for col in header_line.split(self.data_separator)]

        if len(column_names) < 2:
            error_str = (
                f"ERROR: {self.filename} header has {len(column_names)} column(s) but requires >= 2. "
                "Confirm that you using the correct delimiter."
            )
            logger.error(error_str)
            sys.exit(error_str)

        if self._primary_id_column_name not in column_names:
            error_str = (
                f"ERROR: {self.filename} header does not have '{self._primary_id_column_name}' column. Instead it has {column_names}. "
                "Confirm that you using the correct delimiter."
            )
            logger.error(error_str)
            sys.exit(error_str)

        self._primary_id_col_index = column_names.index(self._primary_id_column_name)

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
        # Skip the rows before the header and then begin_loc rows after the header.
        skip_rows = []
        if self.header_row_index > 0:
            skip_rows = [i for i in range(0, self.header_row_index)]
        if block_start > 0:
            skip_rows.extend(
                [i for i in range(self.header_row_index + 1, self.header_row_index + 1 + block_start)]
            )

        # Read in the data from self.filename, extracting the header row, reading
        # in `block_size` rows, and skipping the `skip_rows`.
        res_df = pd.read_csv(
            self.filename,
            sep=self.data_separator,
            skiprows=skip_rows,
            nrows=block_size,
            dtype={self._primary_id_column_name: str},
        )

        res_df.columns = [col.strip() for col in res_df.columns]
        records = res_df.to_records(index=False)
        return np.array(records, dtype=records.dtype.descr)

    def _build_id_map(self):
        """Builds a table of just the object IDs"""
        if self.obj_id_table is not None:
            return

        self.obj_id_table = pd.read_csv(
            self.filename,
            sep=self.data_separator,
            usecols=[self._primary_id_col_index],
            header=self.header_row_index,
            dtype={self._primary_id_column_name: str},
        )

        self.obj_id_table.columns = [col.strip() for col in self.obj_id_table.columns]
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

        # Create list of only the matching rows for these object IDs and the header row.
        skipped_row = [True] * self.header_row_index  # skip the pre-header
        skipped_row.extend([False])  # Keep the the column header
        skipped_row.extend(~np.isin(self.obj_id_table[self._primary_id_column_name], obj_ids))

        # Read in the data from self.filename, extracting the header row, reading
        # in `block_size` rows, and skipping all the `skip_rows`.
        res_df = pd.read_csv(
            self.filename,
            sep=self.data_separator,
            skiprows=(lambda x: skipped_row[x]),
            dtype={self._primary_id_column_name: str},
        )

        res_df.columns = [col.strip() for col in res_df.columns]
        records = res_df.to_records(index=False)
        return np.array(records, dtype=records.dtype.descr)

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
