import math
import os

import numpy as np
import pooch
import pytest
from numpy.testing import assert_equal

from layup.unpack import unpack, unpack_cli
from layup.utilities.data_processing_utilities import get_cov_columns, parse_cov, parse_fit_result
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader


def test_orbit_fit_cli(tmpdir):
    """Test that the orbit_fit cli works for a small CSV file."""
    # Since the orbit_fit CLI outputs to the current working directory, we need to change to our temp directory
    os.chdir(tmpdir)

    temp_out_file = "test_output"

    # Now run the orbit_fit cli with overwrite set to True
    unpack_cli(
        input=get_test_filepath("unpack_test_orbitfit_output.csv"),
        file_format="csv",
        output_file_stem=temp_out_file,
    )

    # Verify the orbit fit produced an output file
    assert os.path.exists(temp_out_file + ".csv")

    # Create a new CSV reader to read in our output file
    output_csv_reader = CSVDataReader(temp_out_file + ".csv", "csv", primary_id_column_name="provID")
    output_data = output_csv_reader.read_rows()

    # Read the input data and get the provID column
    input_csv_reader = CSVDataReader(
        get_test_filepath("unpack_test_orbitfit_output.csv"), "csv", primary_id_column_name="provID"
    )
    input_data = input_csv_reader.read_rows()

    # Verify that the number of rows outputted is the same as the number of unique provIDs in the input file
    n_uniq_ids = sum([1 if id else 0 for id in set(input_data["provID"])])
    assert_equal(len(output_data), n_uniq_ids)

    # Verify the columns in the output data
    expected_cols = [
        "provID",
        "csq",
        "ndof",
        "x",
        "sigma_x",
        "y",
        "sigma_y",
        "z",
        "sigma_z",
        "xdot",
        "sigma_xdot",
        "ydot",
        "sigma_ydot",
        "zdot",
        "sigma_zdot",
        "epochMJD_TDB",
        "niter",
        "method",
        "flag",
        "FORMAT",
    ]
    assert set(output_data.dtype.names) == set(expected_cols)

    orbit_para = ["x", "y", "z", "xdot", "ydot", "zdot"]
    for i, para in enumerate(orbit_para):
        # checks that all values are close in value
        assert np.allclose(output_data["sigma_" + para], np.sqrt(input_data["cov_" + str(i) + "_" + str(i)]))
