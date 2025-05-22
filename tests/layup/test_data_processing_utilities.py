import numpy as np
import pytest
import spiceypy as spice
from numpy.lib import recfunctions as rfn
from numpy.testing import assert_array_equal, assert_equal

from layup.utilities.data_processing_utilities import (
    LayupObservatory,
    get_cov_columns,
    get_format,
    has_cov_columns,
    parse_fit_result,
    process_data,
)
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader
from layup.utilities.file_io.Obs80Reader import Obs80DataReader


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


@pytest.mark.parametrize("n_workers", [1, 5, 8])
@pytest.mark.parametrize("n_rows", [1, 413])
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


@pytest.mark.parametrize("n_workers", [1, 5, 8])
@pytest.mark.parametrize("n_rows", [1, 413])
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


def test_get_cov_columns():
    """Test that we can get the covariance columns."""
    # Check that the function returns the expected number of columns
    cov_columns = get_cov_columns()
    assert len(cov_columns) == 36
    assert cov_columns == [
        "cov_0_0",
        "cov_0_1",
        "cov_0_2",
        "cov_0_3",
        "cov_0_4",
        "cov_0_5",
        "cov_1_0",
        "cov_1_1",
        "cov_1_2",
        "cov_1_3",
        "cov_1_4",
        "cov_1_5",
        "cov_2_0",
        "cov_2_1",
        "cov_2_2",
        "cov_2_3",
        "cov_2_4",
        "cov_2_5",
        "cov_3_0",
        "cov_3_1",
        "cov_3_2",
        "cov_3_3",
        "cov_3_4",
        "cov_3_5",
        "cov_4_0",
        "cov_4_1",
        "cov_4_2",
        "cov_4_3",
        "cov_4_4",
        "cov_4_5",
        "cov_5_0",
        "cov_5_1",
        "cov_5_2",
        "cov_5_3",
        "cov_5_4",
        "cov_5_5",
    ]


@pytest.mark.parametrize("n_workers", [1, 5, 8])
@pytest.mark.parametrize("n_rows", [1, 413])
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
    reader = Obs80DataReader(get_test_filepath("03666_no_bad_dates.txt"))
    data = reader.read_rows()

    observatory = LayupObservatory()
    # Check that the cache is empty
    assert_equal(len(observatory.cached_obs), 0)

    # Add an et column to the data representing the ephemeris time in tdb
    et_col = np.array([spice.str2et(row["obsTime"]) for row in data], dtype="<f8")
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

    # Check that the cache is populated
    assert len(observatory.cached_obs) > 0
    for key in observatory.cached_obs:
        assert len(observatory.cached_obs[key]) > 0


def test_moving_observatory_coordinate_cache():
    """Test that populate_observatory correctly populates the cache, is consistent across calls, and raises errors when required fields are missing."""
    observatory = LayupObservatory()

    # Define test data
    obscode = "C51"  # WISE (space observatory)
    ets = np.array([2451545.0, 2451546.0, 2451547.0])  # Example ephemeris times
    row_dtype = [
        ("obs_geo_x", "<f8"),
        ("obs_geo_y", "<f8"),
        ("obs_geo_z", "<f8"),
    ]
    data = np.array(
        [
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
            (7.0, 8.0, 9.0),
        ],
        dtype=row_dtype,
    )
    for i in range(len(data)):
        row = data[i]
        et = ets[i]

        # Test errors when required fields are missing
        row_missing_x = np.array((np.nan, 2.0, 3.0), dtype=row_dtype)
        with pytest.raises(ValueError):
            observatory.populate_observatory(obscode, et, row_missing_x)

        row_missing_y = np.array((1.0, np.nan, 3.0), dtype=row_dtype)
        with pytest.raises(ValueError):
            observatory.populate_observatory(obscode, et, row_missing_y)

        row_missing_z = np.array((1.0, 2.0, np.nan), dtype=row_dtype)
        with pytest.raises(ValueError):
            observatory.populate_observatory(obscode, et, row_missing_z)

        # Test errors when the observatory position columns are missing
        for i in range(3):
            missing_col_dtype = row_dtype.copy()
            # Replace the ith observatory column with an unexpected column
            missing_col_dtype[i] = ("bad", "<f8")
            row_missing_col = np.array((1.0, 2.0, 3.0), dtype=missing_col_dtype)
            with pytest.raises(ValueError):
                observatory.populate_observatory(obscode, et, row_missing_col)

        # Test observatory coordinate caches are empty for the obscode and epoch
        expected_cache_key = f"{obscode}_{et}"
        assert None in observatory.ObservatoryXYZ[obscode]
        assert expected_cache_key not in observatory.ObservatoryXYZ

        # Test cache population
        cache_key = observatory.populate_observatory(obscode, et, row)
        assert cache_key == expected_cache_key
        assert expected_cache_key in observatory.ObservatoryXYZ
        assert np.allclose(observatory.ObservatoryXYZ[cache_key], list(row))

        # Test consistency across multiple calls
        cache_key_2 = observatory.populate_observatory(obscode, et, row)
        assert expected_cache_key == cache_key_2
        assert np.allclose(observatory.ObservatoryXYZ[cache_key_2], list(row))

        # Test error when coordinates are inconsistent across epochs
        row_inconsistent = np.array((10.0, 12.0, 13.0), dtype=row_dtype)
        with pytest.raises(ValueError):
            observatory.populate_observatory(obscode, et, row_inconsistent)


def test_get_format():
    """Test that the get_format function works for a small CSV file."""
    input_file = get_test_filepath("BCOM.csv")
    input_csv_reader = CSVDataReader(input_file)
    input_data = input_csv_reader.read_rows()
    input_format = input_data[0]["FORMAT"]
    assert get_format(input_data) == input_format


def test_get_format_without_first_row():
    """Test that the get_format function works for a small CSV file that doesn't
    have a valid FORMAT in the first row."""

    input_file = get_test_filepath("BCOM.csv")
    input_csv_reader = CSVDataReader(input_file)
    input_data = input_csv_reader.read_rows()

    input_data["FORMAT"][0] = None

    input_format = input_data[1]["FORMAT"]
    assert get_format(input_data) == input_format


def test_get_format_raises_with_no_data():
    """Test that the get_format function raises error when data is empty."""

    input_file = get_test_filepath("BCOM.csv")
    input_csv_reader = CSVDataReader(input_file)
    input_data = input_csv_reader.read_rows(block_size=0)

    with pytest.raises(ValueError) as e:
        _ = get_format(input_data)
    assert "Data is empty" in str(e.value)


def test_get_format_raises_with_unknown_format_values():
    """Test that the get_format function raises error when FORMAT column is all
    None."""

    input_file = get_test_filepath("BCOM.csv")
    input_csv_reader = CSVDataReader(input_file)
    input_data = input_csv_reader.read_rows(block_size=3)
    input_data["FORMAT"][0] = "Poop"
    input_data["FORMAT"][1] = "Slap"
    input_data["FORMAT"][2] = "Fish"

    with pytest.raises(ValueError) as e:
        _ = get_format(input_data)
    assert "Data does not contain valid orbit format" in str(e.value)


def test_get_format_raises_with_no_format_column():
    """Test that the get_format function raises error when FORMAT column is not
    present."""

    input_file = get_test_filepath("BCOM.csv")
    input_csv_reader = CSVDataReader(input_file)
    input_data = input_csv_reader.read_rows(block_size=3)
    input_data = rfn.drop_fields(input_data, "FORMAT")

    with pytest.raises(ValueError) as e:
        _ = get_format(input_data)
    assert "Data does not contain 'FORMAT' column" in str(e.value)


def test_has_cov_columns():
    """Test that `has_cov_columns` returns True when all covariance columns are present.
    And includes some extra columns as well."""

    # Create a structured array with the covariance columns cov_0_0 through cov_5_5
    dtypes = [
        ("other", "<f8"),
        ("cov_0_0", "<f8"),
        ("cov_0_1", "<f8"),
        ("cov_0_2", "<f8"),
        ("cov_0_3", "<f8"),
        ("cov_0_4", "<f8"),
        ("cov_0_5", "<f8"),
        ("cov_1_0", "<f8"),
        ("cov_1_1", "<f8"),
        ("cov_1_2", "<f8"),
        ("cov_1_3", "<f8"),
        ("cov_1_4", "<f8"),
        ("cov_1_5", "<f8"),
        ("cov_2_0", "<f8"),
        ("cov_2_1", "<f8"),
        ("cov_2_2", "<f8"),
        ("cov_2_3", "<f8"),
        ("cov_2_4", "<f8"),
        ("cov_2_5", "<f8"),
        ("cov_3_0", "<f8"),
        ("cov_3_1", "<f8"),
        ("cov_3_2", "<f8"),
        ("cov_3_3", "<f8"),
        ("cov_3_4", "<f8"),
        ("cov_3_5", "<f8"),
        ("cov_4_0", "<f8"),
        ("cov_4_1", "<f8"),
        ("cov_4_2", "<f8"),
        ("cov_4_3", "<f8"),
        ("cov_4_4", "<f8"),
        ("cov_4_5", "<f8"),
        ("cov_5_0", "<f8"),
        ("cov_5_1", "<f8"),
        ("cov_5_2", "<f8"),
        ("cov_5_3", "<f8"),
        ("cov_5_4", "<f8"),
        ("cov_5_5", "<f8"),  # This should be ignored
    ]
    data = np.empty(1, dtype=dtypes)

    assert has_cov_columns(data) is True


def test_has_cov_columns_not_all_columns():
    """Ensure that `has_cov_columns` returns False when not all covariance
    columns are present."""

    # Create a structured array with the covariance columns cov_0_0 through cov_5_5
    dtypes = [
        ("cov_0_0", "<f8"),
        ("cov_0_1", "<f8"),
        ("cov_0_2", "<f8"),
        ("cov_0_3", "<f8"),
        ("cov_0_4", "<f8"),
        ("cov_0_5", "<f8"),
        ("cov_1_0", "<f8"),
        ("cov_1_1", "<f8"),
        ("cov_1_2", "<f8"),
        ("cov_1_3", "<f8"),
        ("cov_1_4", "<f8"),
        ("cov_1_5", "<f8"),
        ("cov_2_0", "<f8"),
        ("cov_2_1", "<f8"),
        ("cov_2_2", "<f8"),
        ("cov_2_3", "<f8"),
        ("cov_2_4", "<f8"),
        ("cov_2_5", "<f8"),
        ("cov_3_0", "<f8"),
        # Intentionally missing 2 columns
        ("cov_3_3", "<f8"),
        ("cov_3_4", "<f8"),
        ("cov_3_5", "<f8"),
        ("cov_4_0", "<f8"),
        ("cov_4_1", "<f8"),
        ("cov_4_2", "<f8"),
        ("cov_4_3", "<f8"),
        ("cov_4_4", "<f8"),
        ("cov_4_5", "<f8"),
        ("cov_5_0", "<f8"),
        ("cov_5_1", "<f8"),
        ("cov_5_2", "<f8"),
        ("cov_5_3", "<f8"),
        ("cov_5_4", "<f8"),
        ("cov_5_5", "<f8"),  # This should be ignored
    ]
    data = np.empty(1, dtype=dtypes)

    assert has_cov_columns(data) is False


def test_constant_results_with_repeated_calls():
    """Test that we can call obscodes_to_barycentric multiple times from the cache
    and get identical answers each time."""
    csv_reader = CSVDataReader(
        filename=get_test_filepath("4_random_mpc_ADES_provIDs_no_sats.csv"),
        sep="csv",
        primary_id_column_name="provID",
    )
    data = csv_reader.read_rows()

    observatory = LayupObservatory()
    # Check that the cache is empty
    assert_equal(len(observatory.cached_obs), 0)

    # Add an et column to the data representing the ephemeris time in tdb
    et_col = np.array([spice.str2et(row["obsTime"]) for row in data], dtype="<f8")
    data = rfn.append_fields(data, "et", et_col, usemask=False)

    processed_data = observatory.obscodes_to_barycentric(data)
    processed_data_2 = observatory.obscodes_to_barycentric(data)

    for i, j in zip(processed_data, processed_data_2, strict=False):
        if not np.isnan(i["x"]):
            assert_array_equal(i, j)


def test_parse_fit_result():
    """Base path test of parse_fit_results"""
    input_csv_reader = CSVDataReader(
        get_test_filepath("predict_chunk_BCART_EQ.csv"),
        "csv",
        primary_id_column_name="provID",
    )

    input_data = input_csv_reader.read_rows()

    for d in input_data:
        fit_result = parse_fit_result(d)

        assert fit_result.csq != 0
        assert fit_result.ndof != 0
        assert fit_result.niter != 0
        assert len(fit_result.cov) == 36
        assert np.all(np.array(fit_result.cov) != 0.0)


def test_parse_fit_result_no_orbit_column_in_result():
    """Request parse_fit_results does not populate orbit prediction columns in output"""
    input_csv_reader = CSVDataReader(
        get_test_filepath("predict_chunk_BCART_EQ.csv"),
        "csv",
        primary_id_column_name="provID",
    )

    input_data = input_csv_reader.read_rows()

    for d in input_data:
        fit_result = parse_fit_result(d, orbit_colm_flag=False)

        assert fit_result.csq == 0
        assert fit_result.ndof == 0
        assert fit_result.niter == 0
        assert len(fit_result.cov) == 36
        assert np.all(np.array(fit_result.cov) != 0.0)


def test_parse_fit_result_no_covariance_in_input():
    """Test parse_fit_results produces 0s for output of covariance if data is missing"""
    input_csv_reader = CSVDataReader(
        get_test_filepath("predict_chunk_BCART_EQ_no_covariance.csv"),
        "csv",
        primary_id_column_name="provID",
    )

    input_data = input_csv_reader.read_rows()

    for d in input_data:
        fit_result = parse_fit_result(d)

        assert len(fit_result.cov) == 36
        assert np.all(np.array(fit_result.cov) == 0.0)


def test_parse_fit_result_missing_some_cov_columns():
    """Test parse_fit_results produces 0s for output of covariance if the columns
    are missing."""
    input_csv_reader = CSVDataReader(
        get_test_filepath("predict_chunk_BCART_EQ_missing_cov_column.csv"),
        "csv",
        primary_id_column_name="provID",
    )

    input_data = input_csv_reader.read_rows()

    for d in input_data:
        fit_result = parse_fit_result(d)

        assert len(fit_result.cov) == 36
        assert np.all(np.array(fit_result.cov[:-1]) != 0.0)
        assert fit_result.cov[-1] == 0.0
