import os

import numpy as np
import pytest
from numpy.testing import assert_equal

from layup.predict import predict_cli
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader


@pytest.mark.parametrize(
    "chunk_size, time_step",
    [
        (5, 0.5),
        (5, 1),
        (5, 2),
        (3, 0.5),
        (3, 1),
        (3, 2),
    ],
)
def test_predict_cli(tmpdir, chunk_size, time_step):
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

    for test_file_name in ["predict_chunk_BCART_EQ.csv", "predict_chunk_COM.csv"]:
        temp_out_file = f"test_output_{os.path.basename(test_file_name)}"
        predict_cli(
            cli_args=FakeCliArgs(),
            input_file=get_test_filepath(test_file_name),
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

        # # Read the input data and get the provID column
        input_csv_reader = CSVDataReader(
            get_test_filepath(test_file_name), "csv", primary_id_column_name="provID"
        )
        input_data = input_csv_reader.read_rows()

        n_uniq_ids = sum([1 if id else 0 for id in set(input_data["provID"])])
        number_of_predictions_per = len(np.arange(start, end + time_step, time_step))

        # ensure that have a prediction for each object at every time step
        assert_equal(len(output_data), n_uniq_ids * number_of_predictions_per)
