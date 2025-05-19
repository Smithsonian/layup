import os

import numpy as np
import pytest
from numpy.testing import assert_equal

from layup.predict import predict, predict_cli
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader


@pytest.mark.parametrize(
    "chunk_size, time_step, input_format",
    [
        (5, 0.5, "BCART_EQ"),
        (5, 1, "BCART_EQ"),
        (5, 2, "BCART_EQ"),
        (3, 0.5, "BCART_EQ"),
        (3, 1, "BCART_EQ"),
        (3, 2, "BCART_EQ"),
        (5, 0.5, "COM"),
        (5, 1, "COM"),
        (5, 2, "COM"),
        (3, 0.5, "COM"),
        (3, 1, "COM"),
        (3, 2, "COM"),
    ],
)
def test_predict_cli(tmpdir, chunk_size, time_step, input_format):
    """Test that the predict cli works for a small CSV file."""
    os.chdir(tmpdir)

    start = 2461091.50080075
    end = 2461101.50080075

    class FakeCliArgs:
        def __init__(self, g=None):
            self.primary_id_column_name = "provID"
            self.n = 1
            self.chunk = chunk_size
            self.station = "X05"

    # The naming scheme for the test files indicates its orbit format
    test_filename = f"predict_chunk_{input_format}.csv"
    temp_out_file = f"test_output_{os.path.basename(test_filename)}"
    predict_cli(
        cli_args=FakeCliArgs(),
        input_file=get_test_filepath(test_filename),
        start_date=start,
        end_date=end,
        timestep_day=time_step,
        output_file=temp_out_file,
        cache_dir=None,
    )

    # Verify predict produced an output file
    assert os.path.exists(temp_out_file)
    # Create a new CSV reader to read in our output file
    output_csv_reader = CSVDataReader(temp_out_file, "csv", primary_id_column_name="provID")
    output_data = output_csv_reader.read_rows()

    # Read the input data and get the provID column
    input_csv_reader = CSVDataReader(get_test_filepath(test_filename), "csv", primary_id_column_name="provID")
    input_data = input_csv_reader.read_rows()

    n_uniq_ids = sum([1 if id else 0 for id in set(input_data["provID"])])
    number_of_predictions_per = len(np.arange(start, end + time_step, time_step))

    # ensure that have a prediction for each object at every time step
    assert_equal(len(output_data), n_uniq_ids * number_of_predictions_per)

    assert np.all(output_data["ra_deg"] <= 360.0) and np.all(output_data["ra_deg"] >= 0.0)
    assert np.all(output_data["dec_deg"] <= 90.0) and np.all(output_data["dec_deg"] >= -90.0)

    # Ensure that the epoch_utc column is present and in the correct format
    assert all(isinstance(epoch, str) for epoch in output_data["epoch_UTC"])
    # Validate the first epoch_UTC value has the expectd time
    assert output_data["epoch_UTC"][0] == "2026 FEB 20 00:00:00"
    assert all(len(epoch) == 20 for epoch in output_data["epoch_UTC"])
    # All of our start and end dates for our predictions are in the year 2026
    assert all(epoch.startswith("2026") for epoch in output_data["epoch_UTC"])

    assert all(isinstance(epoch, float) for epoch in output_data["epoch_JD_TDB"])


def test_external_predict(tmpdir):
    """Ensure that we can run predict with data that doesn't have our csq and ndof columns."""
    # this file contains some rows with csq and ndof columns and some without
    # so this should test that all functionality remains the same.
    data = CSVDataReader(
        get_test_filepath("fit_result_file_example.csv"), "csv", primary_id_column_name="provID"
    ).read_rows()

    times = np.arange(2461091.50080075, 2461101.50080075 + 0.5, step=0.5)
    predictions = predict(
        data,
        obscode="X05",
        times=times,
        num_workers=1,
        cache_dir=None,
        primary_id_column_name="provID",
    )

    # make sure we generated a prediction for each object at every time step
    n_uniq_ids = sum([1 if id else 0 for id in set(data["provID"])])
    assert len(predictions) == n_uniq_ids * len(times)
