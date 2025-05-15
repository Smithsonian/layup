import os

import numpy as np
import pooch
import pytest
from numpy.testing import assert_equal

from layup.orbitfit import orbitfit, orbitfit_cli
from layup.routines import Observation, get_ephem, run_from_vector
from layup.utilities.data_processing_utilities import get_cov_columns, parse_cov, parse_fit_result
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader

OUTPUT_COL_PER_ORBIT_TYPE = {
    "BCART_EQ": ["x", "y", "z", "xdot", "ydot", "zdot"],
    "BCART": ["x", "y", "z", "xdot", "ydot", "zdot"],
    "CART": ["x", "y", "z", "xdot", "ydot", "zdot"],
    "COM": ["q", "inc", "node", "argPeri", "t_p_MJD_TDB"],
    "BCOM": ["q", "inc", "node", "argPeri", "t_p_MJD_TDB"],
    "KEP": ["a", "e", "inc", "node", "argPeri", "ma"],
    "BKEP": ["a", "e", "inc", "node", "argPeri", "ma"],
}


@pytest.mark.parametrize(
    "chunk_size, num_workers, output_orbit_format",
    [
        (100_000, 1, "BCART_EQ"),
        (100_000, 1, "COM"),
        (100_000, 2, "KEP"),
    ],
)
def test_orbit_fit_cli(tmpdir, chunk_size, num_workers, output_orbit_format):
    """Test that the orbit_fit cli works for a small CSV file."""
    print(
        f"test orbit_fit_cli: chunk_size={chunk_size}, num_workers={num_workers}, output_orbit_format={output_orbit_format}"
    )
    # Since the orbit_fit CLI outputs to the current working directory, we need to change to our temp directory
    os.chdir(tmpdir)
    guess_file_stem = "test_guess"
    temp_guess_file = os.path.join(tmpdir, f"{guess_file_stem}.csv")
    temp_out_file = "test_output"

    test_input_filepath = get_test_filepath("4_random_mpc_ADES_provIDs_no_sats.csv")

    class FakeCliArgs:
        def __init__(self, g=None):
            self.ar_data_file_path = None
            self.primary_id_column_name = "provID"
            self.separate_flagged = False
            self.force = False
            self.debias = False
            self.weight_data = False
            self.g = g  # Command line argument for initial guesses file
            self.output_orbit_format = output_orbit_format

    # Now run the orbit_fit cli with overwrite set to True
    orbitfit_cli(
        input=test_input_filepath,
        input_file_format="ADES_csv",
        output_file_stem=guess_file_stem,  # Our first run will create our initial guess file
        output_file_format="csv",
        chunk_size=chunk_size,
        num_workers=num_workers,
        cli_args=FakeCliArgs(),
    )

    # Verify the orbit fit produced an output file
    assert os.path.exists(temp_guess_file)

    # Get output of guess file
    # Create a new CSV reader to read in our output file
    guess_csv_reader = CSVDataReader(temp_guess_file, "csv", primary_id_column_name="provID")
    guess_data = guess_csv_reader.read_rows()
    # Verify that the the appropriate orbit fit columns are in the output data
    for col in ["flag", "csq", "ndof", "niter", "method"]:
        assert col in guess_data.dtype.names

    # Use the output of our first orbit fit as the initial guesses for our
    # final orbit fit run
    orbitfit_cli(
        input=test_input_filepath,
        input_file_format="ADES_csv",
        output_file_stem=temp_out_file,
        output_file_format="csv",
        chunk_size=chunk_size,
        num_workers=num_workers,
        cli_args=FakeCliArgs(
            g=temp_guess_file,  # Use our first run for the initial guesses
        ),
    )

    # Verify the orbit fit produced an output file
    assert os.path.exists(temp_out_file + ".csv")
    # Create a new CSV reader to read in our output file
    output_csv_reader = CSVDataReader(temp_out_file + ".csv", "csv", primary_id_column_name="provID")
    output_data = output_csv_reader.read_rows()

    # Read the input data and get the provID column
    input_csv_reader = CSVDataReader(
        test_input_filepath,
        "csv",
        primary_id_column_name="provID",
    )
    input_data = input_csv_reader.read_rows()

    # Verify that the number of rows outputted orbit fit is the same as the number of unique provIDs in the input file
    # Note that the input file includes some rows without our provID column, so exclude the nans
    n_uniq_ids = sum([1 if id else 0 for id in set(input_data["provID"])])
    assert_equal(len(output_data), n_uniq_ids)

    # Verify the columns in the output data
    expected_cols = [
        "provID",
        "csq",
        "ndof",
        "epochMJD_TDB",
        "niter",
        "method",
        "flag",
        "FORMAT",
    ] + get_cov_columns()
    assert_equal(0, len(set(expected_cols) - set(output_data.dtype.names)))

    # 222222 only has one data point and 333333 has a datapoint before 1801, both should output flag = -1
    assert all(np.isin(output_data["provID"][output_data["flag"] == -1], ["222222", "333333"]))
    # Verify that all of the output data is in the requested output format for flag == 0 and is nan for flag !=0
    assert np.all(output_data["FORMAT"][output_data["flag"] == 0] == output_orbit_format)
    assert np.all(output_data["FORMAT"][output_data["flag"] != 0] == "NONE")

    # For each row in the output data, check that there is a non-NaN covariance matrix
    # and orbital parameters if there was a successful fit
    for row in output_data:
        # Check that the covariance matrix has populated values.
        cov_matrix = parse_cov(row)
        # Check if the cov_matrix has any NaN values indicating a failed fit
        nan_mask = np.isnan(cov_matrix)
        if nan_mask.any():
            # If any values are NaN, all should be NaN
            assert np.all(nan_mask)
            # Since the fit failed, check that the flag is set to 1 or -1
            assert row["flag"] == 1 or row["flag"] == -1
            for col in OUTPUT_COL_PER_ORBIT_TYPE[output_orbit_format]:
                # Check that the expected orbit format elements are not populated
                assert np.isnan(row[col])
            assert np.isnan(row["epochMJD_TDB"])
        else:
            # Since no values are NaN, check that the flag is set to 0
            assert row["flag"] == 0
            # Check that the covariance matrix is non-zero
            assert np.count_nonzero(cov_matrix) > 0
            # Check that the expected orbit format elements are populated
            for col in OUTPUT_COL_PER_ORBIT_TYPE[output_orbit_format]:
                assert not np.isnan(row[col])
            assert not np.isnan(row["epochMJD_TDB"])


def test_orbit_fit_mixed_inputs():
    """Test that the orbit_fit cli works for a mixed input file."""
    print("test orbit_fit_mixed_inputs")
    # Since the orbit_fit CLI outputs to the current working directory, we need to change to our temp directory
    observations = []
    for i in range(20):
        if i % 2 == 0:
            obs = Observation.from_astrometry(
                ra=0.0 + (i / 100.0),
                dec=0.0,
                epoch=2460000.0 + i,
                observer_position=[0.0, 0.0, 0.0],
                observer_velocity=[0.0, 0.0, 0.0],
            )
        else:
            obs = Observation.from_streak(
                ra=0.0 + (i / 100.0),
                dec=0.0,
                ra_rate=1 / 100.0,
                dec_rate=0.0,
                epoch=2460000.0 + i,
                observer_position=[0.0, 0.0, 0.0],
                observer_velocity=[0.0, 0.0, 0.0],
            )
        observations.append(obs)

    result = run_from_vector(get_ephem(str(pooch.os_cache("layup"))), observations)

    assert result is not None


def test_orbitfit_result_parsing():
    """Perform a simple orbit fit and check that we can parse the results back correctly."""

    input_data = CSVDataReader(
        get_test_filepath("1_random_mpc_ADES_provIDs_no_sats_micro.csv"),
        "csv",
        primary_id_column_name="provID",
    ).read_rows()

    fitted_orbits = orbitfit(
        input_data,
        cache_dir=None,
    )

    for row in fitted_orbits:
        fit_res = parse_fit_result(row)
        if row["flag"] == 0:
            # Test that our parsed rows has the correct values.
            assert fit_res.csq == row["csq"]
            assert fit_res.ndof == row["ndof"]
            assert fit_res.state == [
                row["x"],
                row["y"],
                row["z"],
                row["xdot"],
                row["ydot"],
                row["zdot"],
            ]
            # Note that the fit result is in JD_TDB
            assert fit_res.epoch == row["epochMJD_TDB"] + 2400000.5
            assert fit_res.niter == row["niter"]

            # Check our flattened covariance matrix against each covariance matrix column in the results.
            for i, col in enumerate(get_cov_columns()):
                assert fit_res.cov[i] == row[col]
