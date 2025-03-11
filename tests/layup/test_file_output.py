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
    with tempfile.TemporaryDirectory() as temp_out_dir:
        temp_filepath = os.path.join(temp_out_dir, "test_output.csv")
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
    with tempfile.TemporaryDirectory() as temp_out_dir:
        temp_filepath = os.path.join(temp_out_dir, "test_output.csv")
        write_csv(data, temp_filepath)

        # Read the data back in and compare it to the original data.
        csv_reader2 = CSVDataReader(temp_filepath)
        data2 = csv_reader2.read_rows()
        assert_equal(data, data2)

        # Read the data for only a few objects
        csv_reader3 = CSVDataReader(temp_filepath)
        obj_data1 = csv_reader3.read_objects(["S000015"])
        assert_equal(len(obj_data1), 1)
        obj_data2 = csv_reader3.read_objects(["S00002b"])
        assert_equal(len(obj_data2), 1)

        # Create a new CSV file and write the data in append mode.
        temp_append_filepath = os.path.join(temp_out_dir, "test_output_append.csv")
        write_csv(obj_data1, temp_append_filepath)
        write_csv(obj_data2, temp_append_filepath)

        # Read the data back in and compare it to the original data.
        csv_reader4 = CSVDataReader(temp_append_filepath)
        appended_data = csv_reader4.read_rows()
        assert_equal(len(appended_data), 2)
        assert_equal(appended_data[0], obj_data1[0])
        assert_equal(appended_data[1], obj_data2[0])


def test_write_empty_hdf5():
    # Write an empty numpy structured array to a temporary HDF5 file.
    data = np.array([], dtype=[("ObjID", "<U7"), ("FORMAT", "<U4")])
    with tempfile.TemporaryDirectory() as temp_out_dir:
        temp_filepath = os.path.join(temp_out_dir, "test_output.h5")
        write_hdf5(data, temp_filepath)


def test_write_hdf5():
    # Read a test HDF5 file into a numpy structured array.
    hdf5_reader = HDF5DataReader(get_test_filepath("BCOM.h5"))
    data = hdf5_reader.read_rows()

    # Write the data to a temporary HDF5 file.
    with tempfile.TemporaryDirectory() as temp_out_dir:
        temp_filepath = os.path.join(temp_out_dir, "test_output.h5")
        write_hdf5(data, temp_filepath)

        # Read the data back in and compare it to the original data.
        hdf5_reader2 = HDF5DataReader(temp_filepath)
        data2 = hdf5_reader2.read_rows()
        assert_equal(data, data2)

        # Read the data back in for just a few objects
        hdf5_reader3 = HDF5DataReader(temp_filepath)
        obj_data1 = hdf5_reader3.read_objects(["2003 QX111"])
        obj_data2 = hdf5_reader3.read_objects(["2014 SR373"])
        assert_equal(len(obj_data1), 1)
        assert_equal(len(obj_data2), 1)

        # Create a new HDF5 file and write the data in append mode.
        temp_append_filepath = os.path.join(temp_out_dir, "test_output_append.h5")
        write_hdf5(obj_data1, temp_append_filepath, key="data")
        write_hdf5(obj_data2, temp_append_filepath, key="data")

        # Read the data back in and compare it to the original data.
        hdf5_reader4 = HDF5DataReader(temp_append_filepath)
        appended_data = hdf5_reader4.read_rows()
        assert_equal(len(appended_data), 2)
        assert_equal(appended_data[0], obj_data1[0])
        assert_equal(appended_data[1], obj_data2[0])
