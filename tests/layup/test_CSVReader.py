import numpy as np
import os
import pandas as pd
import pytest
from numpy.testing import assert_equal
from pandas.testing import assert_frame_equal
import tempfile
from pathlib import Path

from layup.utilities.file_readers.CSVReader import CSVDataReader


# TODO fix to actually use the import
# from layup.utilities.data_utilities_for_tests import get_test_filepath
def get_test_filepath(filename):
    # This file's path: `<base_directory>/src/layup/utilities/dataUtilitiesForTests.py
    # THIS_DIR = `<base_directory>/`
    THIS_DIR = Path(__file__).parent.parent

    # Returned path: `<base_directory>/src/layup/config_setups
    return os.path.join(THIS_DIR, "data", filename)


def check_equal_rows(a, b):
    """Check that two arrays are equal, ignoring the data types."""
    # Assert that each array has the same length (not shape because one is a structured array)
    assert len(a) == len(b)

    # Assert that each element is the same
    for i in range(len(a)):
        assert a[i] == b[i]


@pytest.mark.parametrize("use_cache", [True, False])
def test_CSVDataReader_ephem(use_cache):
    """Test that we can read in the ephemeris data from a CSV.

    This test does not perform any transformations, filtering, or validation of the data.
    It just loads it directly from a CSV.
    """
    csv_reader = CSVDataReader(get_test_filepath("CART.csv"), "csv", cache_table=use_cache)
    assert csv_reader.header_row == 0
    assert csv_reader.get_reader_info() == "CSVDataReader:" + get_test_filepath("CART.csv")

    # Read in all 9 rows.
    ephem_data = csv_reader.read_rows()
    assert len(ephem_data) == 5

    expected_first_row = np.array(
        [
            "S00000t",
            "CART",
            0.952105479028,
            0.504888475701,
            4.899098347472,
            148.881068605772,
            39.949789586436,
            54486.32292808,
            54466.0,
        ],
        dtype="object",
    )
    check_equal_rows(expected_first_row, ephem_data[0])

    column_headings = np.array(
        [
            "ObjID",
            "FORMAT",
            "x",
            "y",
            "z",
            "xdot",
            "ydot",
            "zdot",
            "epochMJD_TDB",
        ],
        dtype=object,
    )
    assert_equal(column_headings, ephem_data.dtype.names)

    # Read in rows 3, 4 + the header
    ephem_data = csv_reader.read_rows(3, 4)
    assert len(ephem_data) == 2
    assert_equal(column_headings, ephem_data.dtype.names)
    assert_equal("S00002b", ephem_data[0][0])
    assert_equal("S000044", ephem_data[1][0])


@pytest.mark.parametrize("use_cache", [True, False])
def test_CSVDataReader_specific_ephem(use_cache):
    # Test that we can read in the ephemeris data for specific object IDs only.
    csv_reader = CSVDataReader(get_test_filepath("CART.csv"), "csv", cache_table=use_cache)
    ephem_data = csv_reader.read_objects(["S000015", "S000044"])
    assert len(ephem_data) == 2

    # Check that we correctly loaded the header information.
    column_headings = np.array(
        [
            "ObjID",
            "FORMAT",
            "x",
            "y",
            "z",
            "xdot",
            "ydot",
            "zdot",
            "epochMJD_TDB",
        ],
        dtype=object,
    )
    assert_equal(column_headings, ephem_data.dtype.names)

    # Check that the first row matches.
    expected_first_row = np.array(
        [
            "S000015",
            "CART",
            0.154159694141,
            0.938877338769,
            48.223407545506,
            105.219186748093,
            38.658234184755,
            54736.8815041081,
            54466.0,
        ],
        dtype="object",
    )
    check_equal_rows(expected_first_row, ephem_data[0])

    # Check that the remaining rows have the correct IDs.
    assert_equal(ephem_data[1][0], "S000044")

    # Read different object IDs.
    ephem_data2 = csv_reader.read_objects(["S000021"])
    assert len(ephem_data2) == 1
    assert_equal(ephem_data2[0][0], "S000021")


def test_CSVDataReader_orbits():
    """Test that we can read in the orbit data.

    This test does not perform any transformations, filtering, or validation of the orbit data.
    It just loads it directly from a CSV.
    """
    orbit_des_reader = CSVDataReader(get_test_filepath("testorb.des"), "whitespace")
    assert orbit_des_reader.header_row == 0
    orbit_des = orbit_des_reader.read_rows()

    orbit_csv_reader = CSVDataReader(get_test_filepath("testorb.csv"), "csv")
    assert orbit_csv_reader.header_row == 0
    orbit_csv = orbit_des_reader.read_rows()

    # Check that the two files are the same.
    assert_equal(orbit_csv, orbit_des)

    # Check that the column names and first row match expectations.
    expected_first_row = np.array(
        [
            "S00000t",
            "COM",
            0.952105479028,
            0.504888475701,
            4.899098347472,
            148.881068605772,
            39.949789586436,
            54486.32292808,
            54466.0,
        ],
        dtype=object,
    )

    expected_columns = np.array(
        [
            "ObjID",
            "FORMAT",
            "q",
            "e",
            "inc",
            "node",
            "argPeri",
            "t_p_MJD_TDB",
            "epochMJD_TDB",
        ],
        dtype=object,
    )
    check_equal_rows(expected_first_row, orbit_des[0])
    check_equal_rows(expected_columns, orbit_des.dtype.names)
    assert len(orbit_des) == 5

    with pytest.raises(SystemExit) as e2:
        bad_reader = CSVDataReader(get_test_filepath("testorb.csv"), "whitespace")
        _ = bad_reader.read_rows()
    assert e2.type == SystemExit


def test_CSVDataReader_parameters():
    """Test that we can read in the parameters data.

    This test does not perform any transformations, filtering, or validation of the parameters data.
    It just loads it directly from a CSV.
    """
    # Only read in the first two lines.
    txt_reader = CSVDataReader(get_test_filepath("testcolour.txt"), "whitespace")
    assert txt_reader.header_row == 0
    params_txt = txt_reader.read_rows(0, 2)
    assert len(params_txt) == 2

    csv_reader = CSVDataReader(get_test_filepath("testcolour.csv"), "csv")
    assert csv_reader.header_row == 0
    params_csv = csv_reader.read_rows(0, 2)
    assert len(params_txt) == 2

    expected_first_line = np.array(["S00000t", 17.615, 0.3, 0.0, 0.1, 0.15], dtype=object)
    expected_columns = np.array(["ObjID", "H_r", "g-r", "i-r", "z-r", "GS"], dtype=object)
    assert_equal(params_txt, params_csv)

    check_equal_rows(params_txt[0], expected_first_line)
    assert_equal(params_txt.dtype.names, expected_columns)

    # Check a bad read.
    with pytest.raises(SystemExit) as e1:
        bad_reader = CSVDataReader(get_test_filepath("testcolour.txt"), "csv")
        _ = bad_reader.read_rows()
    assert e1.type == SystemExit

    # Test reading the full text file.
    params_txt2 = txt_reader.read_rows()
    assert len(params_txt2) == 5


def test_CSVDataReader_parameters_objects():
    """Test that we can read in the parameters data by object ID."""
    # Only read in the first two lines.
    txt_reader = CSVDataReader(get_test_filepath("testcolour.txt"), "whitespace")
    params_txt = txt_reader.read_objects(["S000015", "NonsenseID"])
    assert len(params_txt) == 1

    expected_first_line = np.array(["S000015", 22.08, 0.3, 0.0, 0.1, 0.15], dtype=object)
    expected_columns = np.array(["ObjID", "H_r", "g-r", "i-r", "z-r", "GS"], dtype=object)
    check_equal_rows(params_txt[0], expected_first_line)
    assert_equal(params_txt.dtype.names, expected_columns)


def test_CSVDataReader_comets():
    reader = CSVDataReader(get_test_filepath("testcomet.txt"), "whitespace")
    observations = reader.read_rows(0, 1)

    expected = pd.DataFrame({"ObjID": ["67P/Churyumov-Gerasimenko"], "afrho1": [1552], "k": [-3.35]})
    check_equal_rows(observations[0], expected.iloc[0].values)

    # Check reading with a bad format specification.
    with pytest.raises(SystemExit) as e1:
        reader = CSVDataReader(get_test_filepath("testcomet.txt"), "csv")
        _ = reader.read_rows(0, 1)
    assert e1.type == SystemExit


def test_CSVDataReader_delims():
    """Test that we check and match the delimiter during reader creation."""
    _ = CSVDataReader(get_test_filepath("testcolour.txt"), "whitespace")

    # Wrong delim type.
    with pytest.raises(SystemExit) as e1:
        _ = CSVDataReader(get_test_filepath("testcolour.txt"), "csv")
    assert e1.type == SystemExit

    # Invalid delim type.
    with pytest.raises(SystemExit) as e1:
        _ = CSVDataReader(get_test_filepath("testcolour.txt"), "many_commas")
    assert e1.type == SystemExit

    # Empty delim type.
    with pytest.raises(SystemExit) as e2:
        _ = CSVDataReader(get_test_filepath("testcolour.txt"), "")
    assert e2.type == SystemExit
