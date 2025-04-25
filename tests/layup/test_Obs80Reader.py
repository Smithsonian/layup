from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.Obs80Reader import Obs80DataReader


def test_row_count():
    reader = Obs80DataReader(get_test_filepath("03666.txt"), sep="csv")
    row_count = reader.get_row_count()
    assert row_count == 4313


def test_read_rows():
    reader = Obs80DataReader(get_test_filepath("03666.txt"), sep="csv")
    data = reader.read_rows(block_start=0, block_size=10)
    assert len(data) == 10

    data = reader.read_rows()
    assert len(data) == reader.get_row_count()
