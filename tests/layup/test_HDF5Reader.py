import numpy as np
import pytest
from numpy.testing import assert_equal

from layup.utilities.file_io.HDF5Reader import HDF5DataReader
from layup.utilities.data_utilities_for_tests import get_test_filepath


def test_bad_format():
    """Test that we fail if we try to read a non-HDF5 file."""
    reader = HDF5DataReader(get_test_filepath("CART.txt"))
    with pytest.raises(RuntimeError):
        _ = reader.read_rows()
    reader = HDF5DataReader(get_test_filepath("CART.csv"))
    with pytest.raises(RuntimeError):
        _ = reader.read_rows()


@pytest.mark.parametrize("use_cache", [True, False])
def test_HDF5DataReader_read_rows(use_cache):
    """Test that we can read in the ephemeris data from an HDF5 file."""
    reader = HDF5DataReader(get_test_filepath("BCOM.h5"), cache_table=use_cache)
    data = reader.read_rows()
    assert len(data) == 814
    assert reader.get_reader_info() == "HDF5DataReader:" + get_test_filepath("BCOM.h5")

    expected_first_row = np.void(
        (
            "1997 RT5",
            40.243037707640035,
            0.02570883,
            12.7336892,
            163.75249874,
            18.62136958,
            104395.31292273,
            57388.00078916584,
            "BCOM",
        ),
        dtype=[
            ("ObjID", "O"),
            ("q", "<f8"),
            ("e", "<f8"),
            ("inc", "<f8"),
            ("node", "<f8"),
            ("argPeri", "<f8"),
            ("t_p_MJD_TDB", "<f8"),
            ("epochMJD_TDB", "<f8"),
            ("FORMAT", "O"),
        ],
    )
    assert_equal(expected_first_row, data[0])

    column_headings = np.array(
        ["ObjID", "q", "e", "inc", "node", "argPeri", "t_p_MJD_TDB", "epochMJD_TDB", "FORMAT"],
        dtype=object,
    )
    assert_equal(column_headings, data.dtype.names)

    # Read in rows 3, 4, 5, 6 + the header
    data = reader.read_rows(3, 4)
    assert len(data) == 4
    assert_equal(column_headings, data.dtype.names)
    assert_equal("1999 RK215", data[0][0])


@pytest.mark.parametrize("use_cache", [True, False])
def test_HDF5DataReader_read_objects(use_cache):
    """Test that we can read in the ephemeris data for specific object IDs only."""
    reader = HDF5DataReader(get_test_filepath("BCOM.h5"), cache_table=use_cache)
    ephem_data = reader.read_objects(["2003 QX111", "2014 SR373"])
    assert len(ephem_data) == 2

    # Check that we correctly loaded the header information.
    column_headings = np.array(
        ["ObjID", "q", "e", "inc", "node", "argPeri", "t_p_MJD_TDB", "epochMJD_TDB", "FORMAT"],
        dtype=object,
    )
    assert_equal(column_headings, ephem_data.dtype.names)

    # Check that the first row matches.
    expected_first_row = np.array(
        [
            (
                "2003 QX111",
                34.057964320101405,
                0.13352223,
                9.53134345,
                157.49537494,
                100.78289348,
                32309.58539116,
                57388.00078916584,
                "BCOM",
            )
        ],
        dtype=[
            ("ObjID", "O"),
            ("q", "<f8"),
            ("e", "<f8"),
            ("inc", "<f8"),
            ("node", "<f8"),
            ("argPeri", "<f8"),
            ("t_p_MJD_TDB", "<f8"),
            ("epochMJD_TDB", "<f8"),
            ("FORMAT", "O"),
        ],
    )
    assert_equal(expected_first_row, ephem_data[0])

    # Check that the remaining rows have the correct IDs.
    assert_equal(ephem_data[1][0], "2014 SR373")

    # Read different object IDs.
    ephem_data2 = reader.read_objects(["2010 TJ"])
    assert len(ephem_data2) == 1
    assert_equal(ephem_data2[0][0], "2010 TJ")


def test_HDFDataReader_missing_format():
    """Test that we fail if the format column is missing."""
    with pytest.raises(SystemExit) as e1:
        hdf_reader = HDF5DataReader(get_test_filepath("CART_missing_format.h5"), format_column_name="FORMAT")
        hdf_reader.read_rows()
    assert e1.type == SystemExit
    assert "Format column FORMAT not found" in str(e1.value)


def test_HDFDataReader_mixed_formats():
    """Test that we fail if the format column has mixed formats."""
    with pytest.raises(SystemExit) as e1:
        hdf_reader = HDF5DataReader(get_test_filepath("CART_mixed_format.h5"), format_column_name="FORMAT")
        hdf_reader.read_rows()
    assert e1.type == SystemExit
    assert "Multiple formats found." in str(e1.value)


def test_file_count():
    reader = HDF5DataReader(get_test_filepath("BCOM.h5"))
    row_count = reader.get_row_count()
    assert row_count == 814


def test_HDF5DataReader_primary_id_is_str():
    """Ensure that when the primary ID is a string, it is read in as a string."""
    reader = HDF5DataReader(get_test_filepath("CART_with_numeric_objid.h5"))

    expected_objids = [
        "12345",
        "00015",
        "00021",
        "00024",
        "00044",
    ]

    all_data = reader.read_rows()
    assert all(isinstance(i, str) for i in all_data["ObjID"])
    assert all(all_data["ObjID"] == expected_objids)
