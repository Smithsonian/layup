import numpy as np
import logging
import sys
import pandas as pd
from layup.utilities.file_io.ObjectDataReader import ObjectDataReader

# Characters we remove from column names.
_INVALID_COL_CHARS = "!#$%&‘()*+, ./:;<=>?@[\\]^{|}~"

# This routine checks the 80-character input line to see if it contains a special character (S, R, or V) that indicates a 2-line
# record.
def is_two_line(line):
    note2 = line[14]
    obsCode = line[77:80]
    return note2 == "S" or note2 == "R" or note2 == "V"

# This routine opens and reads filename, separating the records into those in the 1-line and 2-line formats.
# The 2-line format lines are merged into single 160-character records for processing line-by-line.
def merge_MPC_file(filename, new_filename, comment_char="#"):
    with open(new_filename, "w") as f1_out:
        line1 = None
        with open(filename, "r") as f:
            print('opened ', filename)
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


# From google
from pathlib import Path
import os
def append_to_filename(filepath, text_to_append):
    path = Path(filepath)
    
    new_filename = f"{path.stem}{text_to_append}{path.suffix}"
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

        self.filename_merge = append_to_filename(filename, "._merge")
        
        merge_MPC_file(filename, self.filename_merge)        

        # Header lines for obs80 data are about observational
        # circumstances.  We identify and skip those lines.

        # To pre-validation the header information.
        #self._validate_header_line()
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
        with open(self.filename_merge, 'r') as file:
            data = file.readlines()
        return len(data)

    def _validate_header_line(self):
        """Read and validate the header line (first line of the file)"""
        logger = logging.getLogger(__name__)
        with open(self.filename) as fh:
            line = fh.readline()
            logger.info(f"Reading the first line of {self.filename} as header:\n{line}")
            self._check_header_line(line)
            return

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

        if self.sep == "csv" or self.sep == "comma":
            column_names = header_line.split(",")
        elif self.sep == "whitespace":
            column_names = header_line.split()
        else:
            logger.error(f"ERROR: Unrecognized delimiter ({self.sep})")
            sys.exit(f"ERROR: Unrecognized delimiter ({self.sep})")

        if len(column_names) < 2:
            error_str = (
                f"ERROR: {self.filename} header has {len(column_names)} column(s) but requires >= 2. "
                "Confirm that you using the correct delimiter."
            )
            logger.error(error_str)
            sys.exit(error_str)

        if self._primary_id_column_name not in column_names:
            error_str = (
                f"ERROR: {self.filename} header does not have '{self._primary_id_column_name}' column. "
                "Confirm that you using the correct delimiter."
            )
            logger.error(error_str)
            sys.exit(error_str)

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
        if block_start > 0:
            skip_rows.extend([i for i in range(self.header_row + 1, self.header_row + 1 + block_start)])

        # Read in the data from self.filename, extracting the header row, and skipping in all of
        # block_size rows, skipping all of the skip_rows.
        if self.sep == "whitespace":
            res_df = pd.read_csv(
                self.filename,
                sep="\\s+",
                skiprows=skip_rows,
                nrows=block_size,
            )
        else:
            res_df = pd.read_csv(
                self.filename,
                delimiter=",",
                skiprows=skip_rows,
                nrows=block_size,
            )
        records = res_df.to_records(index=False)
        return np.array(records, dtype=records.dtype.descr)

    def _build_id_map(self):
        """Builds a table of just the object IDs"""
        if self.obj_id_table is not None:
            return

        if self.sep == "whitespace":
            self.obj_id_table = pd.read_csv(
                self.filename,
                sep="\\s+",
                usecols=[self._primary_id_column_name],
                header=self.header_row,
            )
        else:
            self.obj_id_table = pd.read_csv(
                self.filename,
                delimiter=",",
                usecols=[self._primary_id_column_name],
                header=self.header_row,
            )

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
        skipped_row = [True] * self.header_row  # skip the pre-header
        skipped_row.extend([False])  # Keep the the column header
        skipped_row.extend(~np.isin(self.obj_id_table[self._primary_id_column_name], obj_ids))

        # Read in the data from self.filename, extracting the header row, and skipping in all of
        # block_size rows, skipping all of the skip_rows.
        if self.sep == "whitespace":
            res_df = pd.read_csv(
                self.filename,
                sep="\\s+",
                skiprows=(lambda x: skipped_row[x]),
            )
        else:
            res_df = pd.read_csv(
                self.filename,
                delimiter=",",
                skiprows=(lambda x: skipped_row[x]),
            )

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
