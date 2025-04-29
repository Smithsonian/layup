import os

import numpy as np
import pooch
import pytest
from numpy.testing import assert_equal

from layup.orbitfit import orbitfit_cli
from layup.routines import Observation, get_ephem, run_from_vector
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
    output_file_stem = "test_output"
    temp_out_file = os.path.join(tmpdir, f"{output_file_stem}.csv")

    orbitfit_cli(
        input=get_test_filepath("100_random_mpc_ADES_provIDs_no_sats.csv"),
        input_file_format="ADES_csv",
        output_file_stem=output_file_stem,
        output_file_format="csv",
        chunk_size=chunk_size,
        num_workers=num_workers,
    )

    # Verify the orbit fit produced an output file
    assert os.path.exists(temp_out_file)
    # Create a new CSV reader to read in our output file
    output_csv_reader = CSVDataReader(temp_out_file, "csv", primary_id_column_name="provID")
    output_data = output_csv_reader.read_rows()

    # Read the input data and get the provID column
    input_csv_reader = CSVDataReader(
        get_test_filepath("100_random_mpc_ADES_provIDs.csv"), "csv", primary_id_column_name="provID"
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
        "vx",
        "vy",
        "vz",
        "epoch",
        "niter",
        "method",
        "flag",
        "FORMAT",
        "cov_00",
        "cov_01",
        "cov_02",
        "cov_03",
        "cov_04",
        "cov_05",
        "cov_06",
        "cov_07",
        "cov_08",
        "cov_09",
        "cov_10",
        "cov_11",
        "cov_12",
        "cov_13",
        "cov_14",
        "cov_15",
        "cov_16",
        "cov_17",
        "cov_18",
        "cov_19",
        "cov_20",
        "cov_21",
        "cov_22",
        "cov_23",
        "cov_24",
        "cov_25",
        "cov_26",
        "cov_27",
        "cov_28",
        "cov_29",
        "cov_30",
        "cov_31",
        "cov_32",
        "cov_33",
        "cov_34",
        "cov_35",
    ]
    assert set(output_data.dtype.names) == set(expected_cols)

    # Verify that all of the output data is in the default BCART format
    assert np.all(output_data["FORMAT"] == "BCART")


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
