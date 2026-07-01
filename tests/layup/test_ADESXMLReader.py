"""Tests for the ADES XML observation reader (issue #44)."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.ADESXMLReader import ADESXMLDataReader, parse_ades_xml
from layup.utilities.file_io.CSVReader import CSVDataReader

SAMPLE = "ades_sample.xml"


def _reader():
    return ADESXMLDataReader(get_test_filepath(SAMPLE), primary_id_column_name="provID")


# ---------------------------------------------------------------------------
# parse_ades_xml
# ---------------------------------------------------------------------------


def test_parse_ades_xml_records():
    records = parse_ades_xml(get_test_filepath(SAMPLE))
    assert len(records) == 3
    assert records[0]["provID"] == "3666"
    assert records[0]["stn"] == "024"
    assert records[0]["ra"] == "72.51275"
    # The space-based record carries the ADES satellite-position fields.
    assert records[2]["provID"] == "K14X00X"
    assert records[2]["sys"] == "ICRF_KM"
    assert records[2]["pos1"] == "-4588.997"


def test_parse_ades_xml_handles_namespaces(tmp_path):
    """A namespaced <ades> document still yields records (localname matching)."""
    xml = (
        "<ades xmlns='http://example.org/ades' version='2022'>"
        "<optical><provID>X1</provID><stn>500</stn>"
        "<ra>10.0</ra><dec>5.0</dec></optical></ades>"
    )
    path = tmp_path / "ns.xml"
    path.write_text(xml)
    records = parse_ades_xml(str(path))
    assert len(records) == 1
    assert records[0]["provID"] == "X1"
    assert records[0]["ra"] == "10.0"


def test_parse_ades_xml_no_records_raises(tmp_path):
    path = tmp_path / "empty.xml"
    path.write_text("<ades version='2022'></ades>")
    with pytest.raises(ValueError, match="no optical or radar"):
        parse_ades_xml(str(path))


# ---------------------------------------------------------------------------
# ADESXMLDataReader
# ---------------------------------------------------------------------------


def test_read_rows_columns_and_dtypes():
    data = _reader().read_rows()
    assert len(data) == 3

    names = set(data.dtype.names)
    for col in ("provID", "stn", "obsTime", "ra", "dec", "mag", "sys", "ctr", "pos1"):
        assert col in names, col

    # Identifier/station/time columns stay strings; numeric fields are numeric.
    assert data["provID"].dtype.kind in ("U", "O")
    assert data["stn"].dtype.kind in ("U", "O")
    assert data["obsTime"].dtype.kind in ("U", "O")
    assert np.issubdtype(data["ra"].dtype, np.floating)
    assert np.issubdtype(data["dec"].dtype, np.floating)

    # Station "024" must NOT be coerced to the integer 24.
    assert data["stn"][0] == "024"
    np.testing.assert_allclose(data["ra"][0], 72.51275)


def test_space_based_columns_present_and_filled():
    data = _reader().read_rows()
    # Ground-station rows have no satellite position; the space-based row does.
    sat = data[data["provID"] == "K14X00X"][0]
    assert sat["sys"] == "ICRF_KM"
    assert int(sat["ctr"]) == 399
    np.testing.assert_allclose(sat["pos1"], -4588.997)


def test_get_row_count():
    assert _reader().get_row_count() == 3


def test_read_objects_filters_by_id():
    data = _reader().read_objects(["3666"])
    assert len(data) == 2
    assert set(data["provID"]) == {"3666"}


def test_build_id_map_counts():
    reader = _reader()
    reader._build_id_map()
    assert reader.obj_id_counts == {"3666": 2, "K14X00X": 1}


def test_create_chunks_roundtrip():
    from layup.utilities.data_processing_utilities import create_chunks

    reader = _reader()
    chunks = create_chunks(reader, chunk_size=10)
    # Both objects fit in one chunk; every row is recoverable via read_objects.
    all_ids = [oid for chunk in chunks for oid in chunk]
    assert set(all_ids) == {"3666", "K14X00X"}
    total = sum(len(reader.read_objects(chunk)) for chunk in chunks)
    assert total == 3


# ---------------------------------------------------------------------------
# Equivalence with the CSV reader on real ADES data
# ---------------------------------------------------------------------------


def _array_to_ades_xml(arr, path):
    """Serialize a structured array to ADES XML (one <optical> per row).

    Empty/NaN cells are omitted, mirroring how a sparse ADES record is written.
    """
    root = ET.Element("ades", version="2022")
    for row in arr:
        opt = ET.SubElement(root, "optical")
        for name in arr.dtype.names:
            val = row[name]
            if isinstance(val, (float, np.floating)) and np.isnan(val):
                continue
            text = repr(float(val)) if isinstance(val, (float, np.floating)) else str(val)
            ET.SubElement(opt, name).text = text
    ET.ElementTree(root).write(str(path), encoding="utf-8", xml_declaration=True)


def test_orbitfit_dispatch_and_instantiation():
    """The reader is registered for ADES_xml and accepts orbitfit's exact kwargs."""
    from layup.orbitfit import (
        INPUT_FORMAT_READERS,
        REQUIRED_INPUT_OBSERVATIONS_COLUMN_NAMES,
    )

    reader_class, separator = INPUT_FORMAT_READERS["ADES_xml"]
    assert reader_class is ADESXMLDataReader

    # Instantiate exactly as orbitfit.orbitfit does (sep + required_columns).
    reader = reader_class(
        get_test_filepath(SAMPLE),
        primary_id_column_name="provID",
        sep=separator,
        required_columns=REQUIRED_INPUT_OBSERVATIONS_COLUMN_NAMES,
    )
    data = reader.read_objects(["3666"])
    assert len(data) == 2
    assert set(data["provID"]) == {"3666"}


def test_xml_matches_csv_reader(tmp_path):
    """An ADES XML built from a CSV fixture reads back identically (key columns)."""
    csv = CSVDataReader(
        get_test_filepath("1_random_mpc_ADES_provIDs_no_sats_micro.csv"),
        primary_id_column_name="provID",
    )
    csv_data = csv.read_rows()

    xml_path = tmp_path / "from_csv.xml"
    _array_to_ades_xml(csv_data, xml_path)
    xml_data = ADESXMLDataReader(str(xml_path), primary_id_column_name="provID").read_rows()

    assert len(xml_data) == len(csv_data)
    np.testing.assert_array_equal(xml_data["provID"], csv_data["provID"])
    np.testing.assert_array_equal(xml_data["stn"], csv_data["stn"])
    np.testing.assert_array_equal(xml_data["obsTime"], csv_data["obsTime"])
    np.testing.assert_allclose(xml_data["ra"], csv_data["ra"])
    np.testing.assert_allclose(xml_data["dec"], csv_data["dec"])
