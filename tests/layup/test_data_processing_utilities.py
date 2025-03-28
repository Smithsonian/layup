import pytest
from numpy.testing import assert_equal, assert_allclose

import numpy as np
from numpy.lib import recfunctions as rfn
import spiceypy as spice

from layup.utilities.file_io.CSVReader import CSVDataReader

from layup.utilities.data_processing_utilities import process_data, LayupObservatory
from layup.utilities.data_utilities_for_tests import get_test_filepath


def no_op(data):
    # Returns the data unchanged
    return data


def increment_q(data):
    # Slightly modifies the data by incrementing our q value by 1
    data["q"] += 1
    return data


def count(data):
    # Returns a structured array with a column for the length of the input data
    return np.array([(len(data),)], dtype=[("cnt", "<i8")])


def test_invalid_workers():
    """Test that we fail if we try to use an invalid number of workers."""
    data = np.array([(1, 2), (3, 4), (5, 6)], dtype=[("a", "O"), ("b", "O")])
    with pytest.raises(ValueError):
        _ = process_data(data, 0, no_op)
    with pytest.raises(ValueError):
        _ = process_data(data, -1, no_op)


@pytest.mark.parametrize("n_workers", [1, 2, 4, 8])
def test_empty(n_workers):
    """Test that we can handle an empty file."""
    data = np.array([], dtype=[("a", "O"), ("b", "O")])
    processed_data = process_data(data, n_workers, no_op)
    assert len(processed_data) == 0


@pytest.mark.parametrize("n_workers", [1, 2, 4, 5, 8])
@pytest.mark.parametrize("n_rows", [1, 10000, 1000000])
def test_no_op(n_rows, n_workers):
    """Test that we can apply simple functions on datasets of various sizes and numbers of workers."""
    dtypes = [
        ("ObjID", "O"),
        ("q", "<f8"),
    ]

    # Generate a structured array with random values for nrows
    data = np.empty(n_rows, dtype=dtypes)
    data["ObjID"] = np.random.choice(["red", "fox", "hype"], n_rows)
    data["q"] = np.random.rand(n_rows)

    # Apply the no_op function to the data
    processed_data = process_data(data, n_workers, no_op)
    assert len(processed_data) == n_rows

    # Check that the data is unchanged
    assert_equal(processed_data, data)
    assert_equal(processed_data.dtype, data.dtype)


@pytest.mark.parametrize("n_workers", [1, 2, 4, 5, 8])
@pytest.mark.parametrize("n_rows", [1, 10000, 1000000])
def test_data_modify(n_rows, n_workers):
    """Test that we can apply simple functions on datasets of various sizes and numbers of workers."""
    dtypes = [
        ("ObjID", "O"),
        ("q", "<f8"),
    ]

    # Generate a structured array with random values for nrows
    data = np.empty(n_rows, dtype=dtypes)
    data["ObjID"] = np.random.choice(["red", "fox", "hype"], n_rows)
    data["q"] = np.random.rand(n_rows)

    # Apply the slightly_modify function to the data
    processed_data = process_data(data, n_workers, increment_q)
    assert len(processed_data) == n_rows
    assert_equal(processed_data.dtype, data.dtype)
    for i in range(n_rows):
        assert processed_data["q"][i] == data["q"][i] + 1


@pytest.mark.parametrize("n_workers", [1, 2, 4, 5, 8])
@pytest.mark.parametrize("n_rows", [1, 10000, 1000000])
def test_parallelization(n_rows, n_workers):
    """Test that we can apply simple functions on datasets of various sizes and numbers of workers."""
    dtypes = [
        ("ObjID", "O"),
        ("q", "<f8"),
    ]

    # Generate a structured array with random values for nrows
    data = np.empty(n_rows, dtype=dtypes)
    data["ObjID"] = np.random.choice(["red", "fox", "hype"], n_rows)
    data["q"] = np.random.rand(n_rows)

    # Apply the count function to the data, since this only returns a
    # single row, we can use the length of the returned result to test how
    # the data was sharded across n_workers
    processed_data = process_data(data, n_workers, count)
    if n_workers == 1:
        # All rows were reduced on a single worker
        assert_equal(len(processed_data), 1)
    elif n_rows < n_workers:
        # Since we have less rows to shard than workers, n_rows workers
        # each received a single row to process
        assert_equal(len(processed_data), n_rows)
    else:
        # Each worker reduced its share of data down to a single row
        assert_equal(len(processed_data), n_workers)

    # Check that the sum of the "cnt" column is equal to the number of rows in the original data
    assert_equal(sum(processed_data["cnt"]), len(data))


def test_layup_observatory_obscodes_to_barycentric():
    """Test that we can process the obscodes data."""
    csv_reader = CSVDataReader(get_test_filepath("100_random_mpc_ADES.csv"), "csv")
    data = csv_reader.read_rows()

    observatory = LayupObservatory()
    # Check that the cache is empty
    assert_equal(len(observatory.cached_obs), 0)

    # Add an et column to the data representing the ephemeris time in tdb
    et_col = np.array([spice.str2et(row["obstime"]) for row in data], dtype="<f8")
    data = rfn.append_fields(data, "et", et_col, usemask=False)

    processed_data = observatory.obscodes_to_barycentric(data)
    assert len(processed_data) == len(data)
    assert processed_data.dtype == [
        ("x", "<f8"),
        ("y", "<f8"),
        ("z", "<f8"),
        ("vx", "<f8"),
        ("vy", "<f8"),
        ("vz", "<f8"),
    ]

    # The test file has observatories with no set barycentric positions
    # so we should expect NaNs for all of these

    # Check that fail_on_missing=True raises an error
    with pytest.raises(ValueError):
        _ = observatory.obscodes_to_barycentric(data, fail_on_missing=True)

    # Drop nans from the results
    nan_free_data = processed_data[~np.isnan(processed_data["x"])]
    assert len(nan_free_data) > 0
    assert len(nan_free_data) != len(data)

    # Check that the cache is populated
    assert len(observatory.cached_obs) > 0
    for key in observatory.cached_obs:
        assert len(observatory.cached_obs[key]) > 0
