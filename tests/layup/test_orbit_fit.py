import argparse
import os

import numpy as np
import pytest
from numpy.testing import assert_equal

from layup.orbitfit import orbitfit_cli
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader


@pytest.mark.parametrize(
    "chunk_size, num_workers",
    [
        (630, 1),
        (630, 2),
    ],
)
def test_orbit_fit_cli(tmpdir, chunk_size, num_workers):
    """Test that the orbit_fit cli works for a small CSV file."""
    # Since the orbit_fit CLI outputs to the current working directory, we need to change to our temp directory
    os.chdir(tmpdir)
    output_file_stem = "test_output"
    temp_out_file = os.path.join(tmpdir, f"{output_file_stem}.csv")

    orbitfit_cli(
        input=get_test_filepath("1_random_mpc_ADES_provIDs_micro.csv"),
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
        get_test_filepath("1_random_mpc_ADES_provIDs_micro.csv"), "csv", primary_id_column_name="provID"
    )
    input_data = input_csv_reader.read_rows()

    # Verify that the number of rows outputted orbit fit is the same as the number of unique provIDs in the input file
    # Note that the input file includes some rows without our provID column, so exclude the nans
    n_uniq_ids = sum([0 if np.isnan(id) else 1 for id in set(input_data["provID"])])
    assert_equal(len(output_data), n_uniq_ids)
