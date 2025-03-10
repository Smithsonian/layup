"""Base class for reading object-related data from a variety of sources
and returning a numpy structured array.

Each subclass of ObjectDataReader must implement at least the functions
_read_rows_internal and _read_objects_internal, both of which return a
numpy structured array. Each data source needs to have a column ObjID that
identifies the object and can be used for joining and filtering.

Caching is implemented in the base class. This will lazy load the full
table into memory from the chosen data source, so it should only be
used with smaller data sets. Both ``read_rows`` and ``read_objects``
will check for a cached table before reading the files, allowing them
to perform direct numpy operations if the data is already in memory.
"""

import abc
import logging
import sys

import numpy as np


class ObjectDataReader(abc.ABC):
    """The base class for reading in the object data."""

    def __init__(self, cache_table=False, **kwargs):
        """Set up the reader.

        Note
        ----
        Does not load cached table into memory until the first read.

        Parameters
        ----------
        cache_table : bool, optional
            Indicates whether to keep the entire table in memory.
        """
        self._cache_table = cache_table
        self._table = None

    @abc.abstractmethod
    def get_reader_info(self):
        """Return a string identifying the current reader name
        and input information (for logging and output).

        Returns
        --------
        name : str
            The reader information.
        """
        pass  # pragma: no cover

    def read_rows(self, block_start=0, block_size=None, **kwargs):
        """Reads in a set number of rows from the input, performs
        post-processing and validation, and returns a numpy structured array.

        Parameters
        -----------
        block_start : int (optional)
            The 0-indexed row number from which
            to start reading the data. For example in a CSV file
            block_start=2 would skip the first two lines after the header
            and return data starting on row=2. Default=0

        block_size : int (optional)
            the number of rows to read in.
            Use block_size=None to read in all available data.
            Default = None

        **kwargs : dictionary, optional
            Extra arguments

        Returns
        -----------
        res : numpy structured array
            The data read in from the file.

        """
        if self._cache_table:
            # Load the entire table the first time.
            if self._table is None:
                self._table = self._read_rows_internal()
                self._table = self._process_and_validate_input_table(self._table, **kwargs)

            # Read from the cached table.
            if block_size is None:
                block_end = len(self._table)
            else:
                block_end = block_start + block_size
            return self._table[block_start:block_end]

        # Load the table using the subclass and perform any needed validation.
        res = self._read_rows_internal(block_start, block_size, **kwargs)
        res = self._process_and_validate_input_table(res, **kwargs)
        return res

    @abc.abstractmethod
    def _read_rows_internal(self, block_start=0, block_size=None, **kwargs):
        """Function to do the actual source-specific reading."""
        pass  # pragma: no cover

    def read_objects(self, obj_ids, **kwargs):
        """Read in a chunk of data corresponding to all rows for
        a given set of object IDs.

        Parameters
        -----------
        obj_ids : list
            A list of object IDs to use.

        **kwargs : dictionary, optional
            Extra arguments

        Returns
        -----------
        res : Numpy structured array
            The data read in from the file.
        """
        if self._cache_table:
            # Load the entire table the first time.
            if self._table is None:
                self._table = self._read_rows_internal()
                self._table = self._process_and_validate_input_table(self._table, **kwargs)
            return self._table[np.isin(self._table["ObjID"], obj_ids)]

        # Load the table using the subclass and perform any needed validation.
        res = self._read_objects_internal(obj_ids, **kwargs)
        res = self._process_and_validate_input_table(res, **kwargs)
        return res

    @abc.abstractmethod
    def _read_objects_internal(self, obj_ids, **kwargs):
        """Function to do the actual source-specific reading."""
        pass  # pragma: no cover

    def _validate_object_id_column(self, input_table):
        """Checks that the object ID column exists and converts it to a string.
        This is the common validity check for all object data tables.

        Parameters
        -----------
        input_table : structured array
            A loaded table.

        Returns
        -----------
        input_table : structured array
            Returns the input dataframe modified in-place.
        """
        # Check that the ObjID column exists and convert it to a string.
        try:
            input_table["ObjID"] = input_table["ObjID"].astype(str)
        except KeyError:
            logger = logging.getLogger(__name__)
            err_str = f"ERROR: Unable to find ObjID column headings ({self.get_reader_info()})."
            logger.error(err_str)
            sys.exit(err_str)

        return input_table

    def _process_and_validate_input_table(self, input_table, **kwargs):
        """Perform any input-specific processing and validation on the input table.
        Modifies the input dataframe in place.

        Parameters
        -----------
        input_table : Pandas dataframe
            A loaded table.

        **kwargs : dictionary, optional
                Extra arguments

        Returns
        -----------
        input_table : Numpy structured array
            Returns the input table modified in-place.

        Notes
        --------
        The base implementation includes filtering that is common to most
        input types. Subclasses should call super.process_and_validate()
        to ensure that the ancestor’s validation is also applied.

        Additional arguments to use:

        disallow_nan : boolean
            if True then checks the data for  NaNs or nulls.

        """
        logger = logging.getLogger(__name__)

        # Check that the table has more than one column.
        if len(input_table.dtype.names) <= 1:
            outstr = (
                f"ERROR: While reading table {self.filename}. Only one column found. "
                "Check that you specified the correct format."
            )
            logger.error(outstr)
            sys.exit(outstr)

        # Check that "ObjID" is a column and is a string.
        input_table = self._validate_object_id_column(input_table)

        # Check that the table has a "FORMAT" column
        if "FORMAT" not in input_table.dtype.names:
            outstr = (
                f"ERROR: While reading table {self.filename}. FORMAT column not found."
                "Check that you specified the correct format."
            )
            logger.error(outstr)
            sys.exit(outstr)
        
        # Check that the table has a single "FORMAT" value.
        if len(np.unique(input_table["FORMAT"])) > 1:
            outstr = (
                f"ERROR: While reading table {self.filename}. FORMAT column has multiple values. "
                "Check that you specified the correct format."
            )
            logger.error(outstr)
            sys.exit(outstr)

        # Check for NaNs or nulls.
        if kwargs.get("disallow_nan", False):  # pragma: no cover
            if np.isnan(input_table).any():
                inds = input_table["ObjID"][np.isnan(input_table).any(axis=1)]
                outstr = f"ERROR: While reading table {self.filename} found uninitialised values ObjID: {str(inds)}."
                logger.error(outstr)
                sys.exit(outstr)
        return input_table
