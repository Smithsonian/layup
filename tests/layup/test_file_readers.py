import pytest
from numpy.testing import assert_equal, assert_allclose

from layup.utilities.file_io.CSVReader import CSVDataReader
from layup.utilities.file_io.HDF5Reader import HDF5DataReader
from layup.utilities.data_utilities_for_tests import get_test_filepath


@pytest.mark.parametrize("orbit_format", ["BCOM", "CART", "KEP"])
def test_file_readers_produce_identical_values(orbit_format):
    """Test that the HDF5 and CSV readers produce identical values."""

    hdf5_reader = HDF5DataReader(get_test_filepath(f"{orbit_format}.h5"))
    hdf5_data = hdf5_reader.read_rows()

    csv_reader = CSVDataReader(get_test_filepath(f"{orbit_format}.csv"))
    csv_data = csv_reader.read_rows()

    for column_name in hdf5_data.dtype.names:
        # For non-numeric columns, we can't use assert_allclose, so we use assert_equal.
        if (
            hdf5_data[column_name].dtype.kind == "S"
            or hdf5_data[column_name].dtype.kind == "U"
            or hdf5_data[column_name].dtype.kind == "O"
        ):
            assert_equal(
                hdf5_data[column_name],
                csv_data[column_name],
                err_msg=f"Column {column_name} not equal with dtype {hdf5_data[column_name].dtype}",
            )
        else:
            # Test that we convert back to our original numeric values within a small tolerance of lost precision.
            assert_allclose(
                hdf5_data[column_name],
                csv_data[column_name],
                err_msg=f"Column {column_name} not equal with dtype {hdf5_data[column_name].dtype}",
            )


@pytest.mark.parametrize("orbit_format", ["BCOM", "CART", "KEP"])
def test_file_readers_produce_identical_values_with_cache(orbit_format):
    """Test that the HDF5 and CSV readers produce identical values when using the cache"""

    hdf5_reader = HDF5DataReader(get_test_filepath(f"{orbit_format}.h5"), cache_table=True)
    hdf5_data = hdf5_reader.read_rows()

    csv_reader = CSVDataReader(get_test_filepath(f"{orbit_format}.csv"), cache_table=True)
    csv_data = csv_reader.read_rows()

    for column_name in hdf5_data.dtype.names:
        # For non-numeric columns, we can't use assert_allclose, so we use assert_equal.
        if (
            hdf5_data[column_name].dtype.kind == "S"
            or hdf5_data[column_name].dtype.kind == "U"
            or hdf5_data[column_name].dtype.kind == "O"
        ):
            assert_equal(
                hdf5_data[column_name],
                csv_data[column_name],
                err_msg=f"Column {column_name} not equal with dtype {hdf5_data[column_name].dtype}",
            )
        else:
            # Test that we convert back to our original numeric values within a small tolerance of lost precision.
            assert_allclose(
                hdf5_data[column_name],
                csv_data[column_name],
                err_msg=f"Column {column_name} not equal with dtype {hdf5_data[column_name].dtype}",
            )


def test_required_columns():
    """Ensure that ObjectReader will accept required columns."""

    required_columns = [
        "ObjID",
        "FORMAT",
        "x",
        "y",
        "z",
        "xdot",
        "ydot",
        "zdot",
        "epochMJD_TDB",
    ]

    csv_reader = CSVDataReader(
        get_test_filepath("CART.csv"),
        required_columns=required_columns,
    )

    rows = csv_reader.read_rows()
    assert len(rows) == 5
    assert set(required_columns) == set(rows.dtype.names)


def test_required_columns_with_options():
    """When we pass a tuple of possible values, ensure that the expected ones
    are found."""
    required_columns = [
        "ObjID",
        (
            set(["foo", "bar"]),  # This will not be found
            set(["x", "y"]),  # This set of columns should be found
        ),
    ]

    csv_reader = CSVDataReader(
        get_test_filepath("CART.csv"),
        required_columns=required_columns,
    )

    rows = csv_reader.read_rows()
    assert len(rows) == 5


def test_data_reader_raises_when_missing_columns():
    """Ensure that ObjectReader will raise an error defined required columns are
    missing."""

    csv_reader = CSVDataReader(
        get_test_filepath("CART.csv"),
        required_columns=[
            "ObjID",
            "FORMAT",
            "DOES_NOT_EXIST",
        ],
    )
    with pytest.raises(SystemExit):
        _ = csv_reader.read_rows()


def test_data_reader_raises_when_missing_column_options():
    """Ensure that ObjectReader will raise an error defined required columns (with
    multiple options are missing."""

    csv_reader = CSVDataReader(
        get_test_filepath("CART.csv"),
        required_columns=[
            "ObjID",
            "FORMAT",
            (
                set(["DOES_NOT_EXIST", "ALSO_DOES_NOT_EXIST"]),
                set(["NOPE", "ALSO_NOPE"]),
            ),
        ],
    )
    with pytest.raises(SystemExit):
        _ = csv_reader.read_rows()
