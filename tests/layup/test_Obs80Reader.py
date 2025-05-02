import numpy as np
import pytest

from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.Obs80Reader import Obs80DataReader


def test_row_count():
    """Test reading in an MPC Obs80 data filer and reading in the correct number of rows."""
    reader = Obs80DataReader(get_test_filepath("03666.txt"))
    row_count = reader.get_row_count()
    assert row_count == 4313

    reader = Obs80DataReader(get_test_filepath("newy6.txt"))
    row_count = reader.get_row_count()
    assert row_count == 6585


@pytest.mark.parametrize("filename", ["03666.txt", "newy6_tiny.txt"])
def test_read_rows(filename):
    """Test reading in a block of rows from an MPC Obs80 data file."""
    # Read in a block of 10 rows from the file.
    reader = Obs80DataReader(get_test_filepath(filename))
    data = reader.read_rows(block_start=0, block_size=10)
    assert len(data) == 10

    overlapping_data = reader.read_rows(block_start=5, block_size=10)
    assert len(overlapping_data) == 10

    # Check that the 5 overlapping rows are the same values
    for i in range(5):
        for col in data.dtype.names:
            val1 = data[col][i + 5]
            val2 = overlapping_data[col][i]

            # For numeric values, special handling for NaN
            if np.issubdtype(data[col].dtype, np.number):
                assert (
                    np.isnan(val1) and np.isnan(val2) or val1 == val2
                ), f"Values don't match for {col}: {val1} != {val2}"
            else:
                # For non-numeric values, regular comparison
                assert val1 == val2, f"Values don't match for {col}: {val1} != {val2}"

    # Read in all rows from the file.
    data = reader.read_rows()
    assert len(data) == reader.get_row_count()


def test_read_objects_with_multiple_ids():
    """Test reading in a group of objects from an MPC Obs80 data file."""
    expected_object_ids = ["DES0028", "DES0007", "DES0075"]

    reader = Obs80DataReader(get_test_filepath("newy6_tiny.txt"), primary_id_column_name="provID")
    full_data = reader.read_rows()

    # Because test file has multiple object than we read in we expect less than the full data.
    data = reader.read_objects(expected_object_ids)
    assert len(data) < len(full_data)
    assert set(data["provID"]) == set(expected_object_ids)

    # Now request all objects in the file.
    all_object_ids = list(set(full_data["provID"]))
    data = reader.read_objects(all_object_ids)
    assert len(data) == len(full_data)


def test_unsupported_pid_raises():
    """Test reading in an MPC Obs80 data file with an unsupported primary ID column."""
    with pytest.raises(ValueError) as e:
        _ = Obs80DataReader(get_test_filepath("03666.txt"), primary_id_column_name="unsupported_id")
        assert (
            e.value.args[0]
            == "The primary_id_column_name 'unsupported_id' is not supported for Obs80DataReader."
        )
