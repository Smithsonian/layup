import math
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


@pytest.mark.parametrize(
    "chunk_size, num_workers",
    [
        (100_000, 1),
        (20_000, 2),
    ],
)
def test_orbit_fit_cli(tmpdir, chunk_size, num_workers):
    """Test that the orbit_fit cli works for a small CSV file."""
    # Since the orbit_fit CLI outputs to the current working directory, we need to change to our temp directory
    os.chdir(tmpdir)
    guess_file_stem = "test_guess"
    temp_guess_file = os.path.join(tmpdir, f"{guess_file_stem}.csv")
    temp_out_file = "test_output"

    class FakeCliArgs:
        def __init__(self, g=None):
            self.ar_data_file_path = None
            self.primary_id_column_name = "provID"
            self.separate_flagged = False
            self.force = False
            self.debias = False
            self.g = g  # Command line argument for initial guesses file

    # Now run the orbit_fit cli with overwrite set to True
    orbitfit_cli(
        input=get_test_filepath("4_random_mpc_ADES_provIDs_no_sats.csv"),
        input_file_format="ADES_csv",
        output_file_stem=guess_file_stem,  # Our first run will create our initial guess file
        output_file_format="csv",
        chunk_size=chunk_size,
        num_workers=num_workers,
        cli_args=FakeCliArgs(),
    )

    # Verify the orbit fit produced an output file
    assert os.path.exists(temp_guess_file)

    # Use the output of our first orbit fit as the initial guesses for our
    # final orbit fit run
    orbitfit_cli(
        input=get_test_filepath("4_random_mpc_ADES_provIDs_no_sats.csv"),
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
        get_test_filepath("4_random_mpc_ADES_provIDs_no_sats.csv"), "csv", primary_id_column_name="provID"
    )
    input_data = input_csv_reader.read_rows()

    # Verify that the number of rows outputted orbit fit is the same as the number of unique provIDs in the input file
    # Note that the input file includes some rows without our provID column, so exclude the nans
    n_uniq_ids = sum([0 if np.isnan(id) else 1 for id in set(input_data["provID"])])
    assert_equal(len(output_data), n_uniq_ids)

    # Verify the columns in the output data
    expected_cols = [
        "provID",
        "csq",
        "ndof",
        "x",
        "y",
        "z",
        "xdot",
        "ydot",
        "zdot",
        "epochMJD_TDB",
        "niter",
        "method",
        "flag",
        "FORMAT",
    ] + get_cov_columns()
    assert set(output_data.dtype.names) == set(expected_cols)

    # Verify that all of the output data is in the default BCART format for flag == 0 and is nan for flag !=0
    assert np.all(output_data["FORMAT"][output_data["flag"] == 0] == "BCART")
    for i in np.arange(len(output_data["FORMAT"][output_data["flag"] != 0])):
        assert math.isnan(output_data["FORMAT"][output_data["flag"] != 0][i])

    # For each row in the output data, check that there is a non-zero covariance matrix
    # if there was a successful fit
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
        else:
            # Since no values are NaN, check that the flag is set to 0
            assert row["flag"] == 0
            # Check that the covariance matrix is non-zero
            assert np.count_nonzero(cov_matrix) > 0


def test_orbit_fit_mixed_inputs():
    """Test that the orbit_fit cli works for a mixed input file."""
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
        get_test_filepath("4_random_mpc_ADES_provIDs_no_sats.csv"), "csv", primary_id_column_name="provID"
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
