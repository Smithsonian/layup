import numpy as np
from numpy.testing import assert_equal
import tempfile

from layup.utilities.file_io.CSVReader import CSVDataReader
from layup.utilities.file_io.HDF5Reader import HDF5DataReader

from layup.utilities.data_utilities_for_tests import get_test_filepath

from layup.utilities.file_io.file_output import *


def test_write_empty_csv():
    # Write an empty numpy structured array to a temporary csv file.
    data = np.array([], dtype=[("ObjID", "<U7"), ("FORMAT", "<U4")])
    with tempfile.NamedTemporaryFile(suffix=".csv") as temp_file:
        temp_filepath = temp_file.name
        write_csv(data, temp_filepath)

        # Read the data back in and check that it's empty.
        csv_reader = CSVDataReader(temp_filepath, "csv")
        data2 = csv_reader.read_rows()
        assert_equal(data, data2)


def test_write_csv():
    # Read a test CSV file into a numpy structured array.
    csv_reader = CSVDataReader(get_test_filepath("CART.csv"))
    data = csv_reader.read_rows()

    # Write the data to a temporary csv file.
    with tempfile.NamedTemporaryFile(suffix=".csv") as temp_file:
        temp_filepath = temp_file.name
        write_csv(data, temp_filepath)

        # Read the data back in and compare it to the original data.
        csv_reader2 = CSVDataReader(temp_filepath)
        data2 = csv_reader2.read_rows()
        assert_equal(data, data2)

        # Read the data for only a few objects
        csv_reader3 = CSVDataReader(temp_filepath)
        obj_data = csv_reader3.read_objects(["S000015", "S00002b"])
        assert_equal(len(obj_data), 2)


def test_write_empty_hdf5():
    # Write an empty numpy structured array to a temporary HDF5 file.
    data = np.array([], dtype=[("ObjID", "<U7"), ("FORMAT", "<U4")])
    with tempfile.NamedTemporaryFile(suffix=".h5") as temp_file:
        temp_filepath = temp_file.name
        write_hdf5(data, temp_filepath)


def test_write_hdf5():
    # Read a test HDF5 file into a numpy structured array.
    hdf5_reader = HDF5DataReader(get_test_filepath("BCOM.h5"))
    data = hdf5_reader.read_rows()

    # Write the data to a temporary HDF5 file.
    with tempfile.NamedTemporaryFile(suffix=".h5") as temp_file:
        temp_filepath = temp_file.name
        write_hdf5(data, temp_filepath)

        # Read the data back in and compare it to the original data.
        hdf5_reader2 = HDF5DataReader(temp_filepath)
        data2 = hdf5_reader2.read_rows()
        assert_equal(data, data2)

        # Read the data back in for just a few objects
        hdf5_reader3 = HDF5DataReader(temp_filepath)
        obj_data = hdf5_reader3.read_objects(["2003 QX111", "2014 SR373"])
        assert_equal(len(obj_data), 2)
