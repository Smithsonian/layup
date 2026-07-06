"""Utility functions for writing data from our internal representation of numpy structured arrays to output files."""

import logging
import os
import pandas as pd


def write_csv(data, filepath, move_columns=None):
    """Write a numpy structured array to a CSV file.

    Parameters
    ----------
    data : numpy structured array
        The data to write to the file.
    filepath : str
        The path to the file to write.
    move_columns : dict, optional
        Dict of any column names that need moved, paired with their new position.
    """
    df = pd.DataFrame(data)

    if move_columns != None:
        column_names = list(df.columns.values)
        for col in move_columns.keys():
            if abs(move_columns[col]) > len(column_names):
                raise IndexError(
                    f"Column position is outside of range. Must be between +-{len(column_names)}"
                )
            try:
                column_names.pop(column_names.index((col)))
                column_names.insert(move_columns[col], col)
            except:
                raise ValueError(f"column {col} not found in df.columns.values.")

        df = df.reindex(columns=column_names)

    if os.path.exists(filepath):
        df.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df.to_csv(filepath, index=False)

    logging.info(f"Data written to {filepath}")


def _store_hdf5(data, filepath, key, mode):
    """Write ``data`` to ``filepath`` as an appendable HDF5 table.

    ``mode="w"`` truncates any existing file first (a fresh write); ``mode="a"``
    appends to it (creating it if absent).
    """
    df = pd.DataFrame(data)

    store = pd.HDFStore(filepath, mode=mode)
    store.append(key, df, format="t", data_columns=True)
    store.close()

    logging.info(f"Data written to {filepath}")


def write_hdf5(data, filepath, key="data"):
    """Write a numpy structured array to an HDF5 file, overwriting any existing
    file so that repeated calls are idempotent.

    To accumulate data across several calls (e.g. writing successive chunks into
    one file), call this once for the first write and :func:`append_hdf5` for the
    rest; otherwise a second call to a pre-existing file would duplicate rows.

    Parameters
    ----------
    data : numpy structured array
        The data to write to the file.
    filepath : str
        The path to the file to write.
    key : str, optional
        The key to use in the HDF5 file.
    """
    _store_hdf5(data, filepath, key, mode="w")


def append_hdf5(data, filepath, key="data"):
    """Append a numpy structured array to an HDF5 table, creating the file if it
    does not yet exist.

    Use to accumulate chunked output after an initial :func:`write_hdf5`. On its
    own this grows whatever is already on disk, so it is not idempotent across
    re-runs -- start each run with :func:`write_hdf5`.

    Parameters
    ----------
    data : numpy structured array
        The data to append to the file.
    filepath : str
        The path to the file to write.
    key : str, optional
        The key to use in the HDF5 file.
    """
    _store_hdf5(data, filepath, key, mode="a")
