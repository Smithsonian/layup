import numpy as np
import pytest

from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.Obs80Reader import Obs80DataReader

# A well-formed satellite two-line record (S astrometry line + its lower-case s
# observer-position line) and a second distinct one, plus a normal ground-based
# single-line observation. Column layout matches the real C51/WISE records in
# tests/data/03666.txt.
_SAT1_S = "03666         S2015 08 23.89823 03 55 05.36 +17 52 12.2                L~1WXlC51"
_SAT1_s = "03666         s2015 08 23.89823 1 + 1788.6473 + 6184.6473 + 2386.0656   ~1WXlC51"
_SAT2_S = "03666         S2015 08 24.10000 03 55 10.00 +17 53 00.0                L~1WXlC51"
_SAT2_s = "03666         s2015 08 24.10000 1 + 1700.0000 + 6200.0000 + 2400.0000   ~1WXlC51"
_GROUND = "     DES0024* C2016 10 02.18440 00 35 08.563+01 31 50.69         23.27i      W84"
# A deleted/replaced observation (note 2 code 'x'), a real j4767 WISE/C51 line:
# same exposure as a kept satellite obs but superseded, and carrying no observer
# position of its own.
_DELETED = "j4767K10F61M* x2010 03 26.92800 06 54 29.05 +42 58 36.4                L~0FofC51"


def _write(tmp_path, lines):
    fp = tmp_path / "obs80.txt"
    fp.write_text("\n".join(lines) + "\n")
    return str(fp)


def test_row_count():
    """Test reading in an MPC Obs80 data filer and reading in the correct number of rows."""
    reader = Obs80DataReader(get_test_filepath("03666.txt"))
    row_count = reader.get_row_count()
    # 03666.txt contains one deleted observation (a 1938 photographic plate
    # flagged with note 2 'X'), which is skipped, so the record count is one
    # fewer than the number of astrometry lines.
    assert row_count == 4312

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


def test_clean_satellite_pairs_have_positions(tmp_path):
    """Baseline: two well-formed satellite records parse to two rows, each
    carrying its ICRF_KM observatory position."""
    reader = Obs80DataReader(_write(tmp_path, [_SAT1_S, _SAT1_s, _SAT2_S, _SAT2_s]))
    data = reader.read_rows()
    assert len(data) == 2 == reader.get_row_count()
    assert list(data["sys"]) == ["ICRF_KM", "ICRF_KM"]
    assert not np.isnan(data["pos1"]).any()


def test_duplicate_continuation_line_is_dropped(tmp_path):
    """Regression for the WISE/C51 orphan (Rubin catalog run, object j4767): a
    duplicated ``s`` continuation line must not be emitted as a standalone,
    positionless observation whose position columns get misread as RA/Dec and
    which then falls back to a JPL Horizons lookup. The good pair is kept; the
    stray duplicate is skipped."""
    lines = [_SAT1_S, _SAT1_s, _SAT1_s, _SAT2_S, _SAT2_s]  # note: _SAT1_s appears twice
    reader = Obs80DataReader(_write(tmp_path, lines))
    data = reader.read_rows()
    # Only the two real records survive -- no positionless orphan.
    assert len(data) == 2 == reader.get_row_count()
    assert list(data["sys"]) == ["ICRF_KM", "ICRF_KM"]
    assert not np.isnan(data["pos1"]).any()
    # The orphan would have duplicated _SAT1_s's timestamp; confirm it is gone.
    assert len(set(data["obsTime"])) == 2


def test_leading_orphan_continuation_is_skipped(tmp_path):
    """A continuation line with no preceding first line (e.g. a file sliced in
    the middle of a two-line record) is skipped rather than mis-read."""
    reader = Obs80DataReader(_write(tmp_path, [_SAT1_s, _SAT2_S, _SAT2_s]))
    data = reader.read_rows()
    assert len(data) == 1 == reader.get_row_count()
    assert data["sys"][0] == "ICRF_KM"
    assert not np.isnan(data["pos1"]).any()


def test_first_line_missing_continuation_is_dropped(tmp_path):
    """An S first line whose s continuation is missing must not be paired with
    the next unrelated observation; it is dropped and the following record is
    read on its own."""
    reader = Obs80DataReader(_write(tmp_path, [_SAT1_S, _GROUND, _SAT2_S, _SAT2_s]))
    data = reader.read_rows()
    assert len(data) == 2 == reader.get_row_count()
    # The surviving records: the ground-based single obs (no observer position)
    # and the intact satellite pair.
    assert sorted(data["stn"]) == ["C51", "W84"]
    sat = data[data["stn"] == "C51"][0]
    assert sat["sys"] == "ICRF_KM" and not np.isnan(sat["pos1"])


def test_deleted_observation_is_skipped(tmp_path):
    """A note-2 'x' (deleted/replaced) observation must be skipped entirely, not
    emitted as a positionless record. This is the WISE/C51 orphan that surfaced
    in the full MPC-catalog fit (object j4767): a deleted satellite astrometry
    line has no observer position, so reading it produced a sys='' row that fell
    back to a per-row JPL Horizons lookup."""
    reader = Obs80DataReader(_write(tmp_path, [_SAT1_S, _SAT1_s, _DELETED, _SAT2_S, _SAT2_s]))
    data = reader.read_rows()
    assert len(data) == 2 == reader.get_row_count()
    assert list(data["sys"]) == ["ICRF_KM", "ICRF_KM"]
    assert not np.isnan(data["pos1"]).any()


def test_j4767_wise_excerpt_no_positionless_satellite():
    """Real-data regression: the WISE/C51 block of MPC object j4767 has 12
    satellite S/s pairs and one deleted 'x' line at the same timestamp as a kept
    obs. Every emitted C51 record must carry its ICRF_KM observer position; none
    may be positionless (which is what triggered the Horizons fallback)."""
    reader = Obs80DataReader(get_test_filepath("j4767_wise_excerpt.txt"))
    data = reader.read_rows()
    assert len(data) == 12 == reader.get_row_count()
    assert set(data["stn"]) == {"C51"}
    assert list(data["sys"]) == ["ICRF_KM"] * 12
    assert not np.isnan(data["pos1"]).any()


def test_desync_count_matches_read_and_objects(tmp_path):
    """get_row_count, read_rows and read_objects must all agree on the record
    set even when the file contains a desyncing orphan continuation line."""
    lines = [_SAT1_S, _SAT1_s, _SAT1_s, _GROUND, _SAT2_S, _SAT2_s]
    reader = Obs80DataReader(_write(tmp_path, lines), primary_id_column_name="provID")
    data = reader.read_rows()
    assert reader.get_row_count() == len(data) == 3
    # read_objects over every id present returns exactly the same rows.
    all_ids = list(set(data["provID"]))
    assert len(reader.read_objects(all_ids)) == len(data)
