"""End-to-end workflow-chain smoke test.

`orbitfit` writes a provID-keyed orbit in the BCART_EQ format (its default
`-of`). This checks that such a fit output flows into the downstream consumers
with the shared default primary-id column (provID) and *no extra flags* -- the
"validate the verbs as a chain" guard that would have caught #384 / #392.
"""

import numpy as np

from layup.convert import convert
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader

# A realistic orbit-fit output: provID-keyed, FORMAT=BCART_EQ, with covariance.
FIT_OUTPUT = "test_convert_BCART_EQ.csv"


def _read_fit_output():
    # The default primary-id column (provID) must read a fit output with no flag.
    reader = CSVDataReader(get_test_filepath(FIT_OUTPUT), "csv", primary_id_column_name="provID")
    return np.atleast_1d(reader.read_rows())


def test_fit_output_reads_with_default_provid():
    data = _read_fit_output()
    assert "provID" in data.dtype.names
    assert set(data["FORMAT"]) == {"BCART_EQ"}


def test_fit_output_converts_with_default_provid():
    """fit -> convert works with the shared default primary-id (provID)."""
    data = _read_fit_output()
    out = np.atleast_1d(convert(data, "KEP", num_workers=1, primary_id_column_name="provID"))
    assert "provID" in out.dtype.names
    assert set(out["FORMAT"]) == {"KEP"}
    assert len(out) == len(data)
