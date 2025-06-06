import numpy as np
import pytest
from numpy.testing import assert_equal

from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader


@pytest.mark.parametrize("use_cache", [True, False])
def test_CSVDataReader_ephem(use_cache):
    """Test that we can read in the ephemeris data from a CSV.

    This test does not perform any transformations, filtering, or validation of the data.
    It just loads it directly from a CSV.
    """
    csv_reader = CSVDataReader(get_test_filepath("CART.csv"), "csv", cache_table=use_cache)
    assert csv_reader.header_row_index == 0
    assert csv_reader.get_reader_info() == "CSVDataReader:" + get_test_filepath("CART.csv")

    # Read in all 9 rows.
    ephem_data = csv_reader.read_rows()
    assert len(ephem_data) == 5

    expected_first_row = np.array(
        [
            (
                "S00000t",
                "CART",
                0.952105479028,
                0.504888475701,
                4.899098347472,
                148.881068605772,
                39.949789586436,
                54486.32292808,
                54466.0,
            )
        ],
        dtype=[
            ("ObjID", "<U7"),
            ("FORMAT", "<U4"),
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("xdot", "<f8"),
            ("ydot", "<f8"),
            ("zdot", "<f8"),
            ("epochMJD_TDB", "<f8"),
        ],
    )
    assert_equal(expected_first_row, ephem_data[0])

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
def test_CSVDataReader_read_by_row(use_cache):
    """Test that reading in all rows produces the same row as reading in one row at a time."""
    csv_reader = CSVDataReader(get_test_filepath("CART.csv"), "csv", cache_table=use_cache)
    assert csv_reader.header_row_index == 0
    assert csv_reader.get_reader_info() == "CSVDataReader:" + get_test_filepath("CART.csv")

    # Read in all 9 rows.
    all_data = csv_reader.read_rows()
    assert len(all_data) == 5

    # Read in the rows one at a time.
    row_data = []
    for i in range(5):
        single_row = csv_reader.read_rows(i, 1)
        assert len(single_row) == 1
        row_data.append(single_row)
        assert np.all(single_row == all_data[i])

    assert len(row_data) == len(all_data)


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
        dtype="object",
    )
    assert_equal(column_headings, ephem_data.dtype.names)

    # Check that the first row matches.
    expected_first_row = np.array(
        [
            (
                "S000015",
                "CART",
                0.154159694141,
                0.938877338769,
                48.223407545506,
                105.219186748093,
                38.658234184755,
                54736.8815041081,
                54466.0,
            )
        ],
        dtype=[
            ("ObjID", "<U7"),
            ("FORMAT", "<U4"),
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("xdot", "<f8"),
            ("ydot", "<f8"),
            ("zdot", "<f8"),
            ("epochMJD_TDB", "<f8"),
        ],
    )
    assert_equal(expected_first_row, ephem_data[0])

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
    assert orbit_des_reader.header_row_index == 0
    orbit_des = orbit_des_reader.read_rows()

    orbit_csv_reader = CSVDataReader(get_test_filepath("testorb.csv"), "csv")
    assert orbit_csv_reader.header_row_index == 0
    orbit_csv = orbit_des_reader.read_rows()

    # Check that the two files are the same.
    assert_equal(orbit_csv, orbit_des)

    # Check that the column names and first row match expectations.
    expected_first_row = np.array(
        [
            (
                "S00000t",
                "COM",
                0.952105479028,
                0.504888475701,
                4.899098347472,
                148.881068605772,
                39.949789586436,
                54486.32292808,
                54466.0,
            )
        ],
        dtype=[
            ("ObjID", "<U7"),
            ("FORMAT", "<U3"),
            ("q", "<f8"),
            ("e", "<f8"),
            ("inc", "<f8"),
            ("node", "<f8"),
            ("argPeri", "<f8"),
            ("t_p_MJD_TDB", "<f8"),
            ("epochMJD_TDB", "<f8"),
        ],
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
    assert_equal(expected_first_row, orbit_des[0])
    assert_equal(expected_columns, orbit_des.dtype.names)
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
    txt_reader = CSVDataReader(get_test_filepath("CART.txt"), "whitespace")
    assert txt_reader.header_row_index == 0
    params_txt = txt_reader.read_rows(0, 2)
    assert len(params_txt) == 2

    csv_reader = CSVDataReader(get_test_filepath("CART.csv"), "csv")
    assert csv_reader.header_row_index == 0
    params_csv = csv_reader.read_rows(0, 2)
    assert len(params_txt) == 2

    expected_first_line = np.array(
        [
            (
                "S00000t",
                "CART",
                0.952105479028,
                0.504888475701,
                4.899098347472,
                148.881068605772,
                39.949789586436,
                54486.32292808,
                54466.0,
            )
        ],
        dtype=[
            ("ObjID", "<U7"),
            ("FORMAT", "<U4"),
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("xdot", "<f8"),
            ("ydot", "<f8"),
            ("zdot", "<f8"),
            ("epochMJD_TDB", "<f8"),
        ],
    )
    expected_columns = np.array(
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

    assert_equal(params_txt, params_csv)

    assert_equal(params_txt[0], expected_first_line)
    assert_equal(params_txt.dtype.names, expected_columns)

    # Check a bad read.
    with pytest.raises(SystemExit) as e1:
        bad_reader = CSVDataReader(get_test_filepath("CART.txt"), "csv")
        _ = bad_reader.read_rows()
    assert e1.type == SystemExit

    # Test reading the full text file.
    params_txt2 = txt_reader.read_rows()
    assert len(params_txt2) == 5


def test_CSVDataReader_parameters_objects():
    """Test that we can read in the parameters data by object ID."""
    # Only read in the first two lines.
    txt_reader = CSVDataReader(get_test_filepath("CART.txt"), "whitespace")
    params_txt = txt_reader.read_objects(["S000015", "NonsenseID"])
    assert len(params_txt) == 1

    expected_first_line = np.array(
        [
            (
                "S000015",
                "CART",
                0.154159694141,
                0.938877338769,
                48.223407545506,
                105.219186748093,
                38.658234184755,
                54736.8815041081,
                54466.0,
            )
        ],
        dtype=[
            ("ObjID", "<U7"),
            ("FORMAT", "<U4"),
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("xdot", "<f8"),
            ("ydot", "<f8"),
            ("zdot", "<f8"),
            ("epochMJD_TDB", "<f8"),
        ],
    )
    expected_columns = np.array(
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
    assert_equal(params_txt[0], expected_first_line)
    assert_equal(params_txt.dtype.names, expected_columns)


def test_CSVDataReader_delims():
    """Test that we check and match the delimiter during reader creation."""
    _ = CSVDataReader(get_test_filepath("CART.txt"), "whitespace")

    # Wrong delim type.
    with pytest.raises(SystemExit) as e1:
        _ = CSVDataReader(get_test_filepath("CART.txt"), "csv")
    assert e1.type == SystemExit

    # Invalid delim type.
    with pytest.raises(SystemExit) as e1:
        _ = CSVDataReader(get_test_filepath("CART.txt"), "many_commas")
    assert e1.type == SystemExit

    # Empty delim type.
    with pytest.raises(SystemExit) as e2:
        _ = CSVDataReader(get_test_filepath("CART.txt"), "")
    assert e2.type == SystemExit


def test_CSVDataReader_missing_format():
    """Test that we fail if the format column is missing."""
    with pytest.raises(SystemExit) as e1:
        csv_reader = CSVDataReader(
            get_test_filepath("CART_missing_format.csv"), "csv", format_column_name="FORMAT"
        )
        csv_reader.read_rows()
    assert e1.type == SystemExit
    assert "Format column FORMAT not found" in str(e1.value)


def test_CSVDataReader_mixed_formats():
    """Test that we fail if the format column has mixed formats."""
    with pytest.raises(SystemExit) as e1:
        csv_reader = CSVDataReader(
            get_test_filepath("CART_mixed_format.csv"), "csv", format_column_name="FORMAT"
        )
        csv_reader.read_rows()
    assert e1.type == SystemExit
    assert "Multiple formats found." in str(e1.value)


def test_file_count():
    reader = CSVDataReader(get_test_filepath("BCOM.csv"), sep="csv")
    row_count = reader.get_row_count()
    assert row_count == 814
    assert reader.num_pre_header_lines == 2


def test_CSVReader_csv_read_rows_for_object():
    """Test that we can read in rows for a specific object from a CSV file.
    We want to ensure that the number of expected rows from `read_objects` is correct.
    Additionally, we want to ensure data types match between read_rows and read_objects."""
    reader = CSVDataReader(get_test_filepath("BCOM.csv"), sep="csv")

    one_row = reader.read_rows(3, 1)
    data = reader.read_objects(one_row["ObjID"])
    assert len(data) == 1
    assert one_row.dtype.names == data.dtype.names


def test_file_count_too_many_pre_header_comments():
    with pytest.raises(SystemExit) as e1:
        reader = CSVDataReader(get_test_filepath("csv_pre_header_comments.csv"), sep="csv")
        _ = reader.get_row_count()
    assert e1.type == SystemExit
    assert "not found in the first 100" in str(e1.value)


def test_CSVDataReader_psv():
    """Test that we read in a pipe-separated values (PSV) file correctly."""
    reader = CSVDataReader(get_test_filepath("example.psv"), "pipe", primary_id_column_name="provID")
    row_count = reader.get_row_count()
    assert row_count == 27


def test_CSVReader_psv_read_rows():
    """Test that we can read in rows from a pipe-separated values (PSV) file."""
    reader = CSVDataReader(get_test_filepath("example.psv"), "pipe", primary_id_column_name="provID")
    # reader = CSVDataReader(get_test_filepath("deep7.psv"), "pipe", primary_id_column_name="ra")

    data = reader.read_rows(0, 5)
    assert len(data) == 5
    # TODO: Validate that the data read in is correct.


def test_CSVReader_psv_read_rows_for_object():
    """Test that we can read in rows for a specific object from a PV file."""
    reader = CSVDataReader(
        get_test_filepath("example_no_whitespace.psv"), "pipe", primary_id_column_name="provID"
    )

    one_row = reader.read_rows(0, 1)
    data = reader.read_objects(one_row["provID"])
    assert len(data) == 27
    assert one_row.dtype.names == data.dtype.names


def test_CSVDataReader_primary_col_is_string():
    """Test that primary id columns that are numeric are read in as strings."""
    csv_reader = CSVDataReader(get_test_filepath("CART_with_numeric_objid.csv"), "csv")

    expected_objids = [
        "12345",
        "00015",
        "00021",
        "00024",
        "00044",
    ]

    all_data = csv_reader.read_rows()
    assert all(isinstance(i, str) for i in all_data["ObjID"])
    assert all(all_data["ObjID"] == expected_objids)


def test_CSVReader_stn_as_string():
    """Tests that the station column is read in as a string."""
    csv_reader = CSVDataReader(
        get_test_filepath("bernstein_2004_kbos_data_with_sats_occ.csv"),
        "csv",
        primary_id_column_name="provID",
    )
    all_data = csv_reader.read_rows()
    assert all(isinstance(stn, str) for stn in all_data["stn"])
    expected_stations = {
        "568",
        "950",
        "304",
        "695",
        "705",
        "250",
    }
    assert set(all_data["stn"]) == expected_stations

    objs = ["2000 FV53", "2003 BG91", "2003 BF91", "2003 BH91"]
    obj_data = csv_reader.read_objects(objs)
    assert len(obj_data) == (len(all_data))
    assert all(isinstance(stn, str) for stn in obj_data["stn"])
    assert set(obj_data["stn"]) == expected_stations
