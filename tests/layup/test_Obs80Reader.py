from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.Obs80Reader import Obs80DataReader


def test_row_count():
    reader = Obs80DataReader(get_test_filepath("03666.txt"), sep="csv")
    row_count = reader.get_row_count()
    assert row_count == 4313
