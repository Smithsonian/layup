"""Reader for MPC ADES observation files in XML form (issue #44).

ADES XML wraps each observation in an ``<optical>`` (or ``<radar>``) element
whose child tags are the ADES field names -- the same fields the CSV/PSV reader
consumes, e.g. ``provID``, ``stn``, ``obsTime``, ``ra``, ``dec``, ``mag``.
Both the "flat" form (``<ades><optical>...</optical></ades>``) and the
``<obsBlock>``-wrapped form are handled: we collect every ``<optical>``/
``<radar>`` element anywhere under the root.

Each record becomes one row; the union of all field tags becomes the columns
(missing fields are filled, exactly as a mixed CSV would be). Field text is
coerced to numeric where possible, while the primary-id and station columns are
kept as strings -- mirroring :class:`CSVDataReader` so downstream code sees an
identical structured array regardless of whether the input was CSV, PSV, or XML.
"""

import logging
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from layup.utilities.file_io.ObjectDataReader import ObjectDataReader

logger = logging.getLogger(__name__)

# ADES per-observation record elements we ingest as rows. ``offset`` (residual
# blocks) and the ``obsContext`` metadata are intentionally not treated as
# observations.
_RECORD_TAGS = ("optical", "radar")


def _localname(tag):
    """Strip any XML namespace from a tag: ``{ns}optical`` -> ``optical``."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def parse_ades_xml(filename, record_tags=_RECORD_TAGS):
    """Parse an ADES XML file into a list of per-observation dicts.

    Uses streaming ``iterparse`` so large files do not load the whole DOM at
    once. Each returned dict maps ADES field name -> stripped text value for one
    observation record.
    """
    wanted = set(record_tags)
    records = []
    for _, elem in ET.iterparse(filename, events=("end",)):
        if _localname(elem.tag) in wanted:
            record = {_localname(child.tag): (child.text or "").strip() for child in elem}
            if record:
                records.append(record)
            elem.clear()
    if not records:
        raise ValueError(
            f"ERROR: ADESXMLReader: no {' or '.join(record_tags)} observation records "
            f"found in {filename}. Confirm this is an ADES XML file."
        )
    return records


class ADESXMLDataReader(ObjectDataReader):
    """Read MPC ADES observation data stored as XML into a structured array."""

    def __init__(self, filename, sep=None, **kwargs):
        """Set up the reader.

        Parameters
        ----------
        filename : str
            Location/name of the ADES XML file.
        sep : str, optional
            Unused; accepted so the reader is interchangeable with the
            delimiter-based readers in ``orbitfit``'s format dispatch.
        **kwargs : dict, optional
            Passed to :class:`ObjectDataReader` (``primary_id_column_name`` etc.).
        """
        super().__init__(**kwargs)
        self.filename = filename

        # Parsed records are cached on first read; the whole file is held in
        # memory (consistent with the reader's cache-table design).
        self._records = None
        self._id_map_built = False
        self.obj_id_counts = {}

    def get_reader_info(self):
        """Return a string identifying the reader and its input file."""
        return f"ADESXMLDataReader:{self.filename}"

    def _parse(self):
        """Parse (once) and cache the observation records."""
        if self._records is None:
            self._records = parse_ades_xml(self.filename)
        return self._records

    def get_row_count(self):
        """Return the total number of observation records in the file."""
        return len(self._parse())

    def _get_fixed_dtypes(self):
        """Columns forced to ``str`` (identifiers, not numbers).

        Mirrors :meth:`CSVDataReader._get_fixed_dtypes` so the primary-id and
        station columns are never coerced to numeric (e.g. station ``"024"`` must
        stay a string, not become ``24``).
        """
        fixed_dtypes = {self._primary_id_column_name: str}
        if self._station_column_name is not None:
            fixed_dtypes[self._station_column_name] = str
        return fixed_dtypes

    def _records_to_array(self, records):
        """Convert a list of record dicts into a numpy structured array."""
        df = pd.DataFrame.from_records(records)
        fixed_dtypes = self._get_fixed_dtypes()
        for col in df.columns:
            if col in fixed_dtypes:
                df[col] = df[col].astype(fixed_dtypes[col])
            else:
                # Coerce ADES text fields to numbers where the whole column is
                # numeric; leave genuinely non-numeric fields (obsTime, astCat,
                # band, ...) as strings. pd.to_numeric raises on a non-numeric
                # value, which we treat as "keep this column as text".
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        records_arr = df.to_records(index=False)
        return np.array(records_arr, dtype=records_arr.dtype.descr)

    def _read_rows_internal(self, block_start=0, block_size=None, **kwargs):
        """Read a contiguous block of observation rows."""
        records = self._parse()
        if block_size is None:
            records = records[block_start:]
        else:
            records = records[block_start : block_start + block_size]
        return self._records_to_array(records)

    def _build_id_map(self):
        """Populate ``obj_id_counts`` (rows per object id) for ``create_chunks``."""
        if self._id_map_built:
            return
        for record in self._parse():
            oid = str(record.get(self._primary_id_column_name))
            self.obj_id_counts[oid] = self.obj_id_counts.get(oid, 0) + 1
        self._id_map_built = True

    def _read_objects_internal(self, obj_ids, **kwargs):
        """Read all rows belonging to the given object ids."""
        wanted = set(np.atleast_1d(obj_ids).tolist())
        records = [r for r in self._parse() if str(r.get(self._primary_id_column_name)) in wanted]
        return self._records_to_array(records)

    def _process_and_validate_input_table(self, input_table, **kwargs):
        """Run the shared validation and strip whitespace from column names."""
        input_table = super()._process_and_validate_input_table(input_table, **kwargs)
        input_table.dtype.names = [name.strip() for name in input_table.dtype.names]
        return input_table
