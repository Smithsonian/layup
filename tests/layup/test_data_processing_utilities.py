import numpy as np
import pytest
import spiceypy as spice
from numpy.lib import recfunctions as rfn
from numpy.testing import assert_array_equal, assert_equal

from layup.utilities.data_processing_utilities import (
    AU_KM,
    LayupObservatory,
    get_cov_columns,
    get_format,
    has_cov_columns,
    parse_fit_result,
    process_data,
    skyplane_cov_to_radec_cov,
    write_fallback_obscodes,
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


@pytest.mark.parametrize("test_filename", ["holman_data_with_sats_occ.csv", "03666_no_bad_dates.txt"])
def test_layup_observatory_obscodes_to_barycentric(test_filename):
    """Test that we can process the obscodes data."""
    if test_filename.endswith(".txt"):
        reader = Obs80DataReader(get_test_filepath(test_filename))
    elif test_filename.endswith(".csv"):
        reader = CSVDataReader(get_test_filepath(test_filename), primary_id_column_name="provid")
    else:
        raise ValueError("Unsupported file format")
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

    # Define test data. Use a roving observer (247): a moving observatory that
    # must carry its own ADES position, and -- unlike a spacecraft code such as
    # C51 -- is NOT resolved via JPL Horizons (issue #55), so a missing/NaN
    # position is the error this test exercises.
    obscode = "247"  # Roving Observer (ADES position required)
    ets = np.array([2451545.0, 2451546.0, 2451547.0])  # Example ephemeris times
    row_dtype = [
        ("sys", "U7"),
        ("ctr", "i4"),
        ("pos1", "<f8"),
        ("pos2", "<f8"),
        ("pos3", "<f8"),
    ]

    # Test errors when required observatory position columns have invalid values.
    # The only valid system is "ICRF_KM" and "ICRF_AU" which are tested below
    row_invalid_sys = np.array(("BAD", 399, 1.0, 2.0, 3.0), dtype=row_dtype)
    with pytest.raises(ValueError):
        observatory.populate_observatory(obscode, ets[0], row_invalid_sys)

    # The only valid center is 399 (Earth)
    row_invalid_ctr = np.array(("ICRF_KM", 398, 1.0, 2.0, 3.0), dtype=row_dtype)
    with pytest.raises(ValueError):
        observatory.populate_observatory(obscode, ets[0], row_invalid_ctr)

    # Test that the observatory position columns must not be NaN
    row_missing_x = np.array(("ICRF_KM", 399, np.nan, 2.0, 3.0), dtype=row_dtype)
    with pytest.raises(ValueError):
        observatory.populate_observatory(obscode, ets[0], row_missing_x)
    row_missing_y = np.array(("ICRF_KM", 399, 1.0, np.nan, 3.0), dtype=row_dtype)
    with pytest.raises(ValueError):
        observatory.populate_observatory(obscode, ets[0], row_missing_y)
    row_missing_z = np.array(("ICRF_KM", 399, 1.0, 2.0, np.nan), dtype=row_dtype)
    with pytest.raises(ValueError):
        observatory.populate_observatory(obscode, ets[0], row_missing_z)

    # Test errors when the observatory position column names are missing
    for i in range(5):
        missing_col_dtype = row_dtype.copy()
        # Replace the ith observatory column with an unexpected column
        missing_col_dtype[i] = ("bad", "O")
        row_missing_col = np.array(("ICRF_KM", 399, 1.0, 2.0, 3.0), dtype=missing_col_dtype)
        with pytest.raises(ValueError):
            observatory.populate_observatory(obscode, ets[0], row_missing_col)

    data = np.array(
        [
            ("ICRF_KM", 399, 1.0, 2.0, 3.0),
            ("ICRF_KM", 399, 4.0, 5.0, 6.0),
            ("ICRF_KM", 399, 7.0, 8.0, 9.0),
        ],
        dtype=row_dtype,
    )
    for i in range(len(data)):
        row = data[i]
        et = ets[i]
        expected_coords = list(row)[2:]

        # Test observatory coordinate caches are empty for the obscode and epoch
        expected_cache_key = f"{obscode}_{et}"
        assert None in observatory.ObservatoryXYZ[obscode]
        assert expected_cache_key not in observatory.ObservatoryXYZ

        # Test cache population
        cache_key = observatory.populate_observatory(obscode, et, row)
        assert cache_key == expected_cache_key
        assert expected_cache_key in observatory.ObservatoryXYZ
        assert np.allclose(observatory.ObservatoryXYZ[cache_key], expected_coords)

        # Test consistency across multiple calls
        cache_key_2 = observatory.populate_observatory(obscode, et, row)
        assert expected_cache_key == cache_key_2
        assert np.allclose(observatory.ObservatoryXYZ[cache_key_2], expected_coords)

        # Test that we can convert AU inputs to km
        row_au = row.copy()
        row_au[0] = "ICRF_AU"  # Specify that the system is in AU
        au_et = et + 0.1  # Different epoch to avoid a cache conflict
        cache_key_au = observatory.populate_observatory(obscode, au_et, row_au)
        assert cache_key_au != expected_cache_key
        expected_cooreds_au = [x * 149597870.7 for x in expected_coords]
        assert np.allclose(observatory.ObservatoryXYZ[cache_key_au], expected_cooreds_au)

        # Test error when coordinates are inconsistent across epochs
        row_inconsistent = np.array(("ICRF_KM", 399, 10.0, 12.0, 13.0), dtype=row_dtype)
        with pytest.raises(ValueError):
            observatory.populate_observatory(obscode, et, row_inconsistent)


def test_fixed_observatory_with_zero_parallax_constant():
    """Regression test for issue #286.

    Observatories whose parallax constants are legitimately 0.0 must still be
    treated as fixed-position observatories. The previous truthiness check
    treated a 0.0 Longitude/cos/sin as "no position", which routed these codes
    to the moving-observatory path and raised "invalid coordinates" for plain
    MPC input that carries no per-observation position.
    """
    observatory = LayupObservatory()

    # Codes that have parallax-constant keys (possibly 0.0) must resolve to a
    # finite fixed position. The geocenter codes resolve to (0, 0, 0).
    fixed_with_zero = {
        "000": None,  # Greenwich: Longitude == 0.0
        "782": None,  # Quito (equatorial): sin == 0.0
        "500": (0.0, 0.0, 0.0),  # Geocentric: Longitude == cos == sin == 0.0
        "244": (0.0, 0.0, 0.0),  # Geocentric Occultation Observation
        "248": (0.0, 0.0, 0.0),  # Hipparcos
    }
    for obscode, expected in fixed_with_zero.items():
        coords = observatory.ObservatoryXYZ.get(obscode)
        assert coords is not None
        assert None not in coords, f"{obscode} wrongly treated as having no fixed position"
        assert not np.isnan(np.asarray(coords, dtype=float)).any()
        if expected is not None:
            assert np.allclose(coords, expected)

    # Codes with no parallax-constant keys must still report no fixed position,
    # so they take a per-observation path: roving/other codes use the ADES
    # position; spacecraft codes (C51, 250, ...) are resolved via JPL Horizons
    # (issue #55).
    for obscode in ("247", "C51", "C57", "250"):
        coords = observatory.ObservatoryXYZ.get(obscode)
        assert coords == (None, None, None)


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


def test_moving_observatory_barycentric_position():
    """A moving (space-based) observatory's barycentric position must be Earth's
    barycentric position plus the geocentric offset supplied in the data.

    Regression test for a bug that routed a moving observatory's position --
    stored in km in the J2000/ICRF frame -- through the fixed-station transform
    (rotate Earth-fixed->J2000, then scale by the Earth radius). That misplaced
    the observatory by a factor of the Earth radius (~6378x), i.e. tens of
    millions of km, which made any fit including a space-based observation
    diverge.
    """
    observatory = LayupObservatory()
    et = spice.str2et("2003-01-26T00:24:24.480Z")
    # An HST-like low-Earth-orbit position (km, J2000 geocentric), |r| ~ 6948 km.
    geocentric_km = np.array([-6905.9, -673.9, -353.1])
    row = np.array(
        [("250", et, "ICRF_KM", 399, geocentric_km[0], geocentric_km[1], geocentric_km[2])],
        dtype=[
            ("stn", "U4"),
            ("et", "<f8"),
            ("sys", "U7"),
            ("ctr", "i4"),
            ("pos1", "<f8"),
            ("pos2", "<f8"),
            ("pos3", "<f8"),
        ],
    )

    bary = np.atleast_1d(observatory.obscodes_to_barycentric(row))
    obs_pos_km = np.array([bary["x"][0], bary["y"][0], bary["z"][0]]) * AU_KM

    earth_state, _ = spice.spkezr("EARTH", et, "J2000", "NONE", "SSB")
    earth_pos_km = np.array(earth_state[:3])

    # The observatory must sit at exactly its given geocentric offset from Earth
    # (~6948 km), not ~6378x further out.
    offset_km = obs_pos_km - earth_pos_km
    np.testing.assert_allclose(offset_km, geocentric_km, atol=1.0)


def test_obs_vel_to_km_s_units():
    """Pin the ADES OBS_VEL unit convention used by issue #147: ICRF_KM is km/s
    (unchanged) and ICRF_AU is au/day -> km/s."""
    vel = np.array([1.0, -2.0, 0.5])
    assert_array_equal(LayupObservatory._obs_vel_to_km_s(vel, "ICRF_KM"), vel)
    np.testing.assert_allclose(
        LayupObservatory._obs_vel_to_km_s(vel, "ICRF_AU"), vel * AU_KM / (24 * 60 * 60)
    )


def test_moving_observatory_velocity_uses_user_value():
    """When ADES vel1/vel2/vel3 are supplied for a moving observer, the barycentric
    velocity is Earth's barycentric velocity plus the geocentric observer velocity
    (issue #147), not Earth's velocity alone."""
    observatory = LayupObservatory()
    et = spice.str2et("2003-01-26T00:24:24.480Z")
    geocentric_km = np.array([-6905.9, -673.9, -353.1])
    # An LEO-like geocentric velocity (km/s, ICRF), |v| ~ 7.6 km/s.
    geocentric_vel_km_s = np.array([0.73, -5.41, -5.30])
    row = np.array(
        [("250", et, "ICRF_KM", 399, *geocentric_km, *geocentric_vel_km_s)],
        dtype=[
            ("stn", "U4"),
            ("et", "<f8"),
            ("sys", "U7"),
            ("ctr", "i4"),
            ("pos1", "<f8"),
            ("pos2", "<f8"),
            ("pos3", "<f8"),
            ("vel1", "<f8"),
            ("vel2", "<f8"),
            ("vel3", "<f8"),
        ],
    )

    bary = np.atleast_1d(observatory.obscodes_to_barycentric(row))
    # obscodes_to_barycentric returns velocity in AU/day; convert back to km/s.
    obs_vel_km_s = np.array([bary["vx"][0], bary["vy"][0], bary["vz"][0]]) * AU_KM / (24 * 60 * 60)

    earth_state, _ = spice.spkezr("EARTH", et, "J2000", "NONE", "SSB")
    earth_vel_km_s = np.array(earth_state[3:6])

    np.testing.assert_allclose(obs_vel_km_s - earth_vel_km_s, geocentric_vel_km_s, atol=1e-6)


def test_moving_observatory_velocity_icrf_au_units():
    """An ICRF_AU observer velocity (au/day) yields the same barycentric velocity
    as the equivalent ICRF_KM (km/s) input -- guards the #147 unit conversion
    end-to-end."""
    observatory = LayupObservatory()
    et = spice.str2et("2003-01-26T00:24:24.480Z")
    geocentric_km = np.array([-6905.9, -673.9, -353.1])
    geocentric_vel_km_s = np.array([0.73, -5.41, -5.30])
    # Same physical state expressed in AU and AU/day.
    pos_au = geocentric_km / AU_KM
    vel_au_day = geocentric_vel_km_s * (24 * 60 * 60) / AU_KM
    row = np.array(
        [("250", et, "ICRF_AU", 399, *pos_au, *vel_au_day)],
        dtype=[
            ("stn", "U4"),
            ("et", "<f8"),
            ("sys", "U7"),
            ("ctr", "i4"),
            ("pos1", "<f8"),
            ("pos2", "<f8"),
            ("pos3", "<f8"),
            ("vel1", "<f8"),
            ("vel2", "<f8"),
            ("vel3", "<f8"),
        ],
    )

    bary = np.atleast_1d(observatory.obscodes_to_barycentric(row))
    obs_vel_km_s = np.array([bary["vx"][0], bary["vy"][0], bary["vz"][0]]) * AU_KM / (24 * 60 * 60)

    earth_state, _ = spice.spkezr("EARTH", et, "J2000", "NONE", "SSB")
    np.testing.assert_allclose(obs_vel_km_s - np.array(earth_state[3:6]), geocentric_vel_km_s, atol=1e-6)


def test_moving_observatory_velocity_absent_falls_back_to_earth():
    """Without ADES velocity columns, a moving observer's barycentric velocity
    falls back to Earth's barycentric velocity (the pre-#147 behavior)."""
    observatory = LayupObservatory()
    et = spice.str2et("2003-01-26T00:24:24.480Z")
    geocentric_km = np.array([-6905.9, -673.9, -353.1])
    row = np.array(
        [("250", et, "ICRF_KM", 399, *geocentric_km)],
        dtype=[
            ("stn", "U4"),
            ("et", "<f8"),
            ("sys", "U7"),
            ("ctr", "i4"),
            ("pos1", "<f8"),
            ("pos2", "<f8"),
            ("pos3", "<f8"),
        ],
    )

    bary = np.atleast_1d(observatory.obscodes_to_barycentric(row))
    obs_vel_km_s = np.array([bary["vx"][0], bary["vy"][0], bary["vz"][0]]) * AU_KM / (24 * 60 * 60)

    earth_state, _ = spice.spkezr("EARTH", et, "J2000", "NONE", "SSB")
    np.testing.assert_allclose(obs_vel_km_s, np.array(earth_state[3:6]), atol=1e-6)


def test_observatory_velocity_inconsistent_at_same_epoch_raises():
    """Two different velocities for the same moving observer at the same epoch is
    an error, mirroring the existing position-consistency check."""
    observatory = LayupObservatory()
    et = spice.str2et("2003-01-26T00:24:24.480Z")
    dtype = [
        ("stn", "U4"),
        ("et", "<f8"),
        ("sys", "U7"),
        ("ctr", "i4"),
        ("pos1", "<f8"),
        ("pos2", "<f8"),
        ("pos3", "<f8"),
        ("vel1", "<f8"),
        ("vel2", "<f8"),
        ("vel3", "<f8"),
    ]
    row1 = np.array([("250", et, "ICRF_KM", 399, 1000.0, 0.0, 0.0, 1.0, 0.0, 0.0)], dtype=dtype)[0]
    row2 = np.array([("250", et, "ICRF_KM", 399, 1000.0, 0.0, 0.0, 2.0, 0.0, 0.0)], dtype=dtype)[0]

    observatory.populate_observatory("250", et, row1)
    with pytest.raises(ValueError):
        observatory.populate_observatory("250", et, row2)


def test_skyplane_cov_to_radec_cov():
    """The sky-plane covariance is already an on-sky (great-circle) covariance,
    so the error ellipse is the eigen-decomposition of the 2x2 matrix with no
    cos(dec) scaling. Regression test for the previous version, which scaled by
    cos(dec) with dec in degrees and was called with obs_cov_yy and obs_cov_xy
    swapped.
    """
    rad_to_arcsec = (180.0 / np.pi) * 3600.0

    # Axis-aligned covariance, major axis along Dec (yy > xx): PA points North (0 deg).
    a, b, pa = skyplane_cov_to_radec_cov(np.array([1e-12]), np.array([0.0]), np.array([4e-12]))
    assert np.isclose(a[0], np.sqrt(4e-12) * rad_to_arcsec)
    assert np.isclose(b[0], np.sqrt(1e-12) * rad_to_arcsec)
    assert np.isclose(pa[0] % 180.0, 0.0)

    # Axis-aligned covariance, major axis along RA (xx > yy): PA points East (90 deg).
    a, b, pa = skyplane_cov_to_radec_cov(np.array([4e-12]), np.array([0.0]), np.array([1e-12]))
    assert np.isclose(a[0], np.sqrt(4e-12) * rad_to_arcsec)
    assert np.isclose(pa[0] % 180.0, 90.0)

    # Isotropic covariance: a circle, a == b.
    a, b, _ = skyplane_cov_to_radec_cov(np.array([2e-12]), np.array([0.0]), np.array([2e-12]))
    assert np.isclose(a[0], b[0])

    # General positive-definite case must match the eigenvalues of the 2x2,
    # independent of any RA/Dec (no cos(dec) dependence in the signature).
    rng = np.random.default_rng(0)
    for _ in range(10):
        m = rng.normal(size=(2, 2))
        cov = m @ m.T * 1e-12  # symmetric positive-definite, radians^2
        a, b, _ = skyplane_cov_to_radec_cov(
            np.array([cov[0, 0]]), np.array([cov[0, 1]]), np.array([cov[1, 1]])
        )
        eigvals = np.linalg.eigvalsh(cov)
        assert np.isclose(a[0], np.sqrt(eigvals[1]) * rad_to_arcsec)
        assert np.isclose(b[0], np.sqrt(eigvals[0]) * rad_to_arcsec)
        assert a[0] >= b[0] > 0.0


def test_write_fallback_obscodes():
    """The obscodes copy bundled with layup decompresses to valid JSON that
    sorcha's Observatory can read directly (network-outage fallback)."""
    import json

    path = write_fallback_obscodes()
    with open(path) as f:
        obs = json.load(f)
    assert len(obs) > 2000
    assert "X05" in obs and "500" in obs  # Rubin + geocenter


def test_layup_observatory_falls_back_on_mpc_failure(monkeypatch):
    """If the MPC obscodes download fails (e.g. server unreachable), the
    LayupObservatory should fall back to the bundled copy instead of raising."""
    import requests

    import layup.utilities.data_processing_utilities as dpu

    orig_init = dpu.SorchaObservatory.__init__
    calls = []

    def fake_init(self, args, auxconfigs, oc_file=None):
        calls.append(oc_file)
        if oc_file is None:
            # Simulate the MPC download failing.
            raise requests.exceptions.ConnectTimeout("simulated MPC outage")
        # The fallback path supplies a local oc_file; let the real init read it.
        orig_init(self, args, auxconfigs, oc_file=oc_file)

    monkeypatch.setattr(dpu.SorchaObservatory, "__init__", fake_init)

    observatory = LayupObservatory()

    # First attempt (download) raised; the second used the bundled fallback.
    assert calls[0] is None and calls[1] is not None
    assert "X05" in observatory.ObservatoryXYZ
    assert len(observatory.ObservatoryXYZ) > 2000


def test_layup_observatory_falls_back_on_corrupt_obscodes(monkeypatch):
    """If the MPC obscodes download "succeeds" but yields an empty/corrupt file
    that fails to parse as JSON, the LayupObservatory should still fall back to
    the bundled copy instead of raising (the download error is a
    json.JSONDecodeError, not a network exception)."""
    import json

    import layup.utilities.data_processing_utilities as dpu

    orig_init = dpu.SorchaObservatory.__init__
    calls = []

    def fake_init(self, args, auxconfigs, oc_file=None):
        calls.append(oc_file)
        if oc_file is None:
            # Simulate sorcha decompressing an empty obscodes file and json.load
            # choking on the empty string -- the failure observed on CI.
            raise json.JSONDecodeError("Expecting value", "", 0)
        # The fallback path supplies a local oc_file; let the real init read it.
        orig_init(self, args, auxconfigs, oc_file=oc_file)

    monkeypatch.setattr(dpu.SorchaObservatory, "__init__", fake_init)

    observatory = LayupObservatory()

    # First attempt (corrupt download) raised; the second used the bundled fallback.
    assert calls[0] is None and calls[1] is not None
    assert "X05" in observatory.ObservatoryXYZ
    assert len(observatory.ObservatoryXYZ) > 2000
