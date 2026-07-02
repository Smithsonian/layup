import numpy as np
import pytest

from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.Obs80Reader import Obs80DataReader


def test_row_count():
    """Test reading in an MPC Obs80 data filer and reading in the correct number of rows."""
    reader = Obs80DataReader(get_test_filepath("03666.txt"))
    row_count = reader.get_row_count()
    assert row_count == 4313

    reader = Obs80DataReader(get_test_filepath("newy6.txt"))
    row_count = reader.get_row_count()
    assert row_count == 6585


@pytest.mark.parametrize("filename", ["03666.txt", "newy6_tiny.txt"])
def test_read_rows(filename):
    """Test reading in a block of rows from an MPC Obs80 data file."""
    # Read in a block of 10 rows from the file.
    reader = Obs80DataReader(get_test_filepath(filename))
    data = reader.read_rows(block_start=0, block_size=10)
    assert len(data) == 10

    overlapping_data = reader.read_rows(block_start=5, block_size=10)
    assert len(overlapping_data) == 10

    # Check that the 5 overlapping rows are the same values
    for i in range(5):
        for col in data.dtype.names:
            val1 = data[col][i + 5]
            val2 = overlapping_data[col][i]

            # For numeric values, special handling for NaN
            if np.issubdtype(data[col].dtype, np.number):
                assert (
                    np.isnan(val1) and np.isnan(val2) or val1 == val2
                ), f"Values don't match for {col}: {val1} != {val2}"
            else:
                # For non-numeric values, regular comparison
                assert val1 == val2, f"Values don't match for {col}: {val1} != {val2}"

    # Read in all rows from the file.
    data = reader.read_rows()
    assert len(data) == reader.get_row_count()


def test_read_objects_with_multiple_ids():
    """Test reading in a group of objects from an MPC Obs80 data file."""
    expected_object_ids = ["DES0028", "DES0007", "DES0075"]

    reader = Obs80DataReader(get_test_filepath("newy6_tiny.txt"), primary_id_column_name="provID")
    full_data = reader.read_rows()

    # Because test file has multiple object than we read in we expect less than the full data.
    data = reader.read_objects(expected_object_ids)
    assert len(data) < len(full_data)
    assert set(data["provID"]) == set(expected_object_ids)

    # Now request all objects in the file.
    all_object_ids = list(set(full_data["provID"]))
    data = reader.read_objects(all_object_ids)
    assert len(data) == len(full_data)


def test_unsupported_pid_raises():
    """Test reading in an MPC Obs80 data file with an unsupported primary ID column."""
    with pytest.raises(ValueError) as e:
        _ = Obs80DataReader(get_test_filepath("03666.txt"), primary_id_column_name="unsupported_id")
        assert (
            e.value.args[0]
            == "The primary_id_column_name 'unsupported_id' is not supported for Obs80DataReader."
        )


def test_read_primary_id_as_str():
    """Test reading in an MPC Obs80 data file with primary ID as string."""
    reader = Obs80DataReader(
        get_test_filepath("newy6_10_row_numeric_ids.txt"), primary_id_column_name="provID"
    )

    data = reader.read_rows()
    assert len(data) == 10
    assert all(isinstance(i, str) for i in data["provID"])
    assert data["provID"][0] == "0000024"


def test_read_obs_pos_units():
    """Test that we read in observatory position units correctly."""
    # Create a reader and read in the lines
    reader = Obs80DataReader(get_test_filepath("03666.txt"))

    # Two lines pulled from 03666.txt
    first_line = "03666         S2015 08 23.89823 03 55 05.36 +17 52 12.2                L~1WXlC51"
    second_line = "03666         s2015 08 23.89823 1 + 1788.6473 + 6184.6473 + 2386.0656   ~1WXlC51"

    data = reader.convert_obs80(first_line, second_line)

    obs_pos_x = 1788.6473
    obs_pos_y = 6184.6473
    obs_pos_z = 2386.0656

    # Check that the observatory position is in km
    assert data[-5] == "ICRF_KM"
    assert data[-4] == 399
    assert data[-3] == pytest.approx(obs_pos_x, rel=1e-5)
    assert data[-2] == pytest.approx(obs_pos_y, rel=1e-5)
    assert data[-1] == pytest.approx(obs_pos_z, rel=1e-5)

    # Now try again with a unit flag of 2. This should read the input as AU which is then converted to km.
    au_second_line = second_line[:32] + "2" + second_line[33:]
    au_data = reader.convert_obs80(first_line, au_second_line)
    assert au_data[-5] == "ICRF_AU"
    assert au_data[-4] == 399
    assert au_data[-3] == pytest.approx(obs_pos_x, rel=1e-5)
    assert au_data[-2] == pytest.approx(obs_pos_y, rel=1e-5)
    assert au_data[-1] == pytest.approx(obs_pos_z, rel=1e-5)

    bad_second_line = second_line[:32] + "3" + second_line[33:]
    with pytest.raises(ValueError):
        reader.convert_obs80(first_line, bad_second_line)


def test_read_roving_observer_position():
    """A roving-observer (V/v) two-line record carries geodetic longitude/latitude
    (deg) and altitude (m) on WGS84 -- not a geocentric satellite position. The
    reader must dispatch on the record type and parse it as such rather than
    mis-reading it as a satellite position (issue #402)."""
    reader = Obs80DataReader(get_test_filepath("03666.txt"))

    first_line = "00433         V2023 08 26.19193220 55 41.10 -08 18 29.6          15.1 VV~7811270"
    second_line = "00433         v2023 08 26.1919321 237.76096  +38.11385      0           ~7811270"

    data = reader.convert_obs80(first_line, second_line)
    assert data[-5] == "WGS84"
    assert data[-4] == 399
    assert data[-3] == pytest.approx(237.76096, rel=1e-6)  # East longitude (deg)
    assert data[-2] == pytest.approx(38.11385, rel=1e-6)  # latitude (deg)
    assert data[-1] == pytest.approx(0.0, abs=1e-9)  # altitude (m)


def test_radar_two_line_record_raises():
    """Radar (R/r) obs80 two-line records are not observer-position lines; the
    reader should raise a clear error rather than mis-parsing them (issue #402)."""
    reader = Obs80DataReader(get_test_filepath("03666.txt"))
    r_first = "00433         R2011 10 23.34124006 53 03.495+46 43 06.69               X~7lwF275"
    r_second = "00433         r2011 10 23.3412401 + 4353.0030 -  481.6100 + 1382.3400   ~7lwF275"
    with pytest.raises(ValueError, match="[Rr]adar"):
        reader.convert_obs80(r_first, r_second)


def test_reader_skips_blank_lines_and_parses_two_line_records():
    """Blank/truncated lines must be skipped, not crash the reader (issue #407),
    and a mixed file of single-line, satellite, and roving records reads cleanly."""
    reader = Obs80DataReader(get_test_filepath("obs80_two_line_records.txt"), primary_id_column_name="provID")
    data = reader.read_rows()
    # single optical + satellite + roving = 3 records; the blank lines are skipped.
    assert len(data) == 3
    assert list(data["sys"]) == ["", "ICRF_KM", "WGS84"]
    # the roving record kept its geodetic position
    roving = data[data["sys"] == "WGS84"][0]
    assert roving["pos1"] == pytest.approx(237.76096, rel=1e-6)
    assert roving["pos3"] == pytest.approx(0.0, abs=1e-9)
