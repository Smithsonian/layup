from layup.convert import convert_cli
from layup.utilities.data_utilities_for_tests import get_test_filepath

from layup.utilities.file_io.CSVReader import CSVDataReader
from layup.utilities.file_io.HDF5Reader import HDF5DataReader

from numpy.testing import assert_equal
import os
import pytest


@pytest.mark.parametrize(
    "chunk_size, num_workers",
    [
        (10_000, 8),
        (10_000, 4),
        (10_0000, 1),
        (10_0000, -1),
        (500, 8),
        (500, 4),
        (500, 1),
        (500, -1),
        (10, 8),
        (10, 4),
        (10, 1),
        (10, -1),
        (2, 4),
        (1, 3),
        (1, 1),
        (1, -1),
    ],
)
def test_convert_round_trip_csv(tmpdir, chunk_size, num_workers):
    """Test that the convert function works for a small CSV file."""
    input_csv_reader = CSVDataReader(get_test_filepath("CART.csv"), "csv")
    input_data = input_csv_reader.read_rows()

    # Since the convert CLI outputs to the current working directory, we need to change to our temp directory
    output_file_stem = "test_output"
    os.chdir(tmpdir)
    temp_out_file = os.path.join(tmpdir, f"{output_file_stem}.csv")
    convert_cli(
        get_test_filepath("CART.csv"),
        output_file_stem,
        "BCOM",
        "csv",
        chunk_size=chunk_size,
        num_workers=num_workers,
    )

    assert os.path.exists(temp_out_file)

    output_csv_reader = CSVDataReader(temp_out_file, "csv")
    output_data = output_csv_reader.read_rows()
    # TODO we can round trip to test but right now convert simply copies and reappends data
    assert_equal(input_data, output_data)


@pytest.mark.parametrize(
    "chunk_size, num_workers",
    [
        (10_000, 8),
        (10_000, 4),
        (10_0000, 1),
        (10_0000, -1),
        (500, 8),
        (500, 4),
        (500, 1),
        (500, -1),
        (10, 8),
        (10, 4),
        (10, 1),
        (10, -1),
        (2, 4),
        (1, 3),
        (1, 1),
        (1, -1),
    ],
)
def test_convert_one_chunk_one_worker_hdf5(tmpdir, chunk_size, num_workers):
    """Test that the convert function works for a small HDF5 file."""
    input_hdf5_reader = HDF5DataReader(get_test_filepath("CART.h5"))
    input_data = input_hdf5_reader.read_rows()

    # Since the convert CLI outputs to the current working directory, we need to change to our temp directory
    output_file_stem = "test_output"
    os.chdir(tmpdir)
    temp_out_file = os.path.join(tmpdir, f"{output_file_stem}.h5")
    convert_cli(
        get_test_filepath("CART.h5"),
        temp_out_file,
        "BCOM",
        "hdf5",
        chunk_size=chunk_size,
        num_workers=num_workers,
    )

    assert os.path.exists(temp_out_file)

    output_hdf5_reader = HDF5DataReader(temp_out_file)
    output_data = output_hdf5_reader.read_rows()
    # TODO we can round trip to test but right now convert simply copies and reappends data
    assert_equal(input_data, output_data)
