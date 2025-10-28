"""Utility functions for writing data from our internal representation of numpy structured arrays to output files."""

import logging
import os
import pandas as pd


def write_csv(data, filepath, move_columns={}):
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

    if move_columns != {}:
        column_names = list(df.columns.values)
        for col in move_columns.keys():
            column_names.pop(column_names.index((col)))
            column_names.insert(move_columns[col], col)

        df = df[column_names]

    if os.path.exists(filepath):
        df.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df.to_csv(filepath, index=False)

    logging.info(f"Data written to {filepath}")


def write_hdf5(data, filepath, key="data"):
    """Write a numpy structured array to an HDF5 file.

    Parameters
    ----------
    data : numpy structured array
        The data to write to the file.
    filepath : str
        The path to the file to write.
    key : str, optional
        The key to use in the HDF5 file.
    """
    df = pd.DataFrame(data)

    store = pd.HDFStore(filepath)
    store.append(key, df, format="t", data_columns=True)
    store.close()

    logging.info(f"Data written to {filepath}")
