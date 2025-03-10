"""Utility functions for writing data from our internal representation of numpy structured arrays to output files."""

import logging

import pandas as pd


def write_csv(data, filepath):
    """Write a numpy structured array to a CSV file.

    Parameters
    ----------
    data : numpy structured array
        The data to write to the file.
    filepath : str
        The path to the file to write.
    """
    df = pd.DataFrame(data)
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
    df.to_hdf(filepath, key, mode="w")
    logging.info(f"Data written to {filepath}")
