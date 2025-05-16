import numpy as np
import logging
from pathlib import Path

from layup.utilities.data_processing_utilities import parse_fit_result
from layup.utilities.file_io import CSVDataReader, HDF5DataReader
from layup.utilities.file_io.file_output import write_csv, write_hdf5

logger = logging.getLogger(__name__)


# Columns which may be added to the output data by the orbit fitting process
ORBIT_FIT_COLS = [
    ("csq", "f8"),  # Chi-square value
    ("ndof", "i4"),  # Number of degrees of freedom
    ("niter", "i4"),  # Number of iterations
    ("method", "O"),  # Method used for orbit fitting
    ("flag", "i4"),  # Single-character flag indicating success of the fit
]

INPUT_READERS = {
    "csv": CSVDataReader,
    "hdf5": HDF5DataReader,
}


def _get_result_dtypes(primary_id_column_name: str, state: list, sigma: list, orbit_col_flag):
    """
    Creates a structured numpy dtype for the output data based on the provided parameters.

    Parameters
    ----------
    primary_id_column_name : str
        The name of the primary ID column (e.g., object identifier).
    state : list
        A list of parameter names corresponding to the state vector elements (e.g., position or orbital parameters).
    sigma : list
        A list of parameter names corresponding to the uncertainties (sigma) of the state vector elements.
    orbit_col_flag : bool
        A flag indicating whether orbit fitting columns (e.g., chi-square, degrees of freedom) should be included.

    Returns
    -------
    np.dtype
        A structured numpy dtype defining the format of the output array.
    """


    if orbit_col_flag == True:
        return np.dtype(
            [
                (primary_id_column_name, "O"),  # Object ID
                ("csq", "f8"),  # Chi-square value
                ("ndof", "i4"),  # Number of degrees of freedom
                (state[0], "f8"),  # The first of 6 state vector elements
                (sigma[0], "f8"),  # The first of 6 sigma vector elements
                (state[1], "f8"),
                (sigma[1], "f8"),
                (state[2], "f8"),
                (sigma[2], "f8"),
                (state[3], "f8"),
                (sigma[3], "f8"),
                (state[4], "f8"),
                (sigma[4], "f8"),
                (state[5], "f8"),  # The last of 6 state vector elements
                (sigma[5], "f8"),  # The last of 6 state vector elements
                ("epochMJD_TDB", "f8"),  # Epoch
                ("niter", "i4"),  # Number of iterations
                ("method", "O"),  # Method used for orbit fitting
                ("flag", "i4"),  # Single-character flag indicating success of the fit
                ("FORMAT", "O"),  # Orbit format
            ]
        )

    else:
        return np.dtype(
            [
                (primary_id_column_name, "O"),  # Object ID
                (state[0], "f8"),  # The first of 6 state vector elements
                (sigma[0], "f8"),  # The first of 6 sigma vector elements
                (state[1], "f8"),
                (sigma[1], "f8"),
                (state[2], "f8"),
                (sigma[2], "f8"),
                (state[3], "f8"),
                (sigma[3], "f8"),
                (state[4], "f8"),
                (sigma[4], "f8"),
                (state[5], "f8"),  # The last of 6 state vector elements
                (sigma[5], "f8"),  # The last of 6 state vector elements
                ("epochMJD_TDB", "f8"),  # Epoch
                ("FORMAT", "O"),  # Orbit format
            ]
        )


def unpack(res, name, orbit_para, _RESULT_DTYPES, orbit_cols_flag, format):
    """
    Unpacks a file containing a covariance matrix into associated uncertainties.

    Parameters
    ----------
    res : FitResult
        The parsed fit result.
    name : str
        name of the object
    orbit_para : list
        A list of parameter names corresponding to the state vector elements
    _RESULT_DTYPES : np,dtype
        A structured numpy dtype defining the format of the output array.
    Returns
    -------
    output : numpy array
        the numpy array containing the unpacked data.
    orbit_cols_flag : bool
        flag if input file has orbit fit columns
    format : str
        orbital format of data
    """

    j = [0, 7, 14, 21, 28, 35]  # location of cov_ii [00,11,22,33,44,55]
    error_list = []
    for i, orbit_para in enumerate(orbit_para):
        error_list.append(np.sqrt(res.cov[j[i]]))
    state_error_tuple = tuple(item for pair in zip(res.state, error_list) for item in pair)
    if orbit_cols_flag == True:
        # Construct the output array
        output = np.array(
            [
                (
                    name,
                    res.csq,
                    res.ndof,
                )
                + state_error_tuple
                + (
                    res.epoch - 2400000.5,  # changing from jd back to mjd
                    res.niter,
                    res.method,
                    res.flag,
                    format,  # The base format returned by the C++ code
                )
            ],
            dtype=_RESULT_DTYPES,
        )
    else:
        # Construct the output array
        output = np.array(
            [
                (name,)
                + state_error_tuple
                + (
                    res.epoch - 2400000.5,  # changing from jd back to mjd
                    format,  # The base format returned by the C++ code
                )
            ],
            dtype=_RESULT_DTYPES,
        )

    return output


def unpack_cli(
    input,
    file_format,
    output_file_stem,
    chunk_size: int = 10_000,
    cli_args: dict = None,
):
    """
    unpacks an input files covarience into uncertainties

    Note that the output file will be written in the caller's current working directory.

    Parameters
    ----------
    input : str
        The path to the input file.
    output_file_stem : str
        The stem of the output file.
    file_format : str, optional (default="csv")
        The format of the output file. Must be one of: "csv", "hdf5"
    chunk_size : int, optional (default=10_000)
        The number of rows to read in at a time.

    cli_args : argparse, optional (default=None)
        The argparse object that was created when running from the CLI.
    """
    primary_id_column_name = cli_args.primary_id_column_name if cli_args else "provID"

    input_file = Path(input)
    if file_format == "csv":
        output_file = Path(f"{output_file_stem}.{file_format.lower()}")
    else:
        output_file = (
            Path(f"{output_file_stem}")
            if output_file_stem.endswith(".h5")
            else Path(f"{output_file_stem}.h5")
        )

    # Open the input file and read the first line
    reader_class = INPUT_READERS.get(file_format)

    sample_reader = reader_class(
        input_file,
        format_column_name="FORMAT",
        primary_id_column_name=primary_id_column_name,
    )

    sample_data = sample_reader.read_rows(block_start=0, block_size=1)

    orbit_fit_col_names = [col[0] for col in ORBIT_FIT_COLS]  # Extract column names from ORBIT_FIT_COLS
    if all(col in sample_data.dtype.names for col in orbit_fit_col_names):
        orbit_cols_flag = True
    else:
        orbit_cols_flag = False

    # Check orbit format in the file
    input_format = None
    if "FORMAT" in sample_data.dtype.names:
        input_format = sample_data["FORMAT"][0]
    else:
        logger.error("Input file does not contain 'FORMAT' column")

    format = input_format

    if format in ["CART", "BCART_EQ"]:
        orbit_para = ["x", "y", "z", "xdot", "ydot", "zdot"]
        orbit_para_sigma = ["sigma_x", "sigma_y", "sigma_z", "sigma_xdot", "sigma_ydot", "sigma_zdot"]
    elif format in ["KEP", "BKEP"]:
        orbit_para = ["a", "e", "inc", "node", "argPeri", "ma"]
        orbit_para_sigma = ["sigma_a", "sigma_e", "sigma_inc", "sigma_node", "sigma_argPeri", "sigma_ma"]
    elif format in ["COM", "BCOM"]:
        orbit_para = ["q", "e", "inc", "node", "argPeri", "t_p_MJD_TDB"]
        orbit_para_sigma = [
            "sigma_q",
            "sigma_e",
            "sigma_inc",
            "sigma_node",
            "sigma_argPeri",
            "sigma_t_p_MJD_TDB",
        ]

    _RESULT_DTYPES = _get_result_dtypes(
        primary_id_column_name, orbit_para, orbit_para_sigma, orbit_col_flag=orbit_cols_flag
    )

    # read data
    full_reader = reader_class(
        input_file,
        format_column_name="FORMAT",
        primary_id_column_name=primary_id_column_name,
    )

    # Calculate the start and end indices for each chunk, as a list of tuples
    # of the form (start, end) where start is the starting index of the chunk
    # and the last index of the chunk + 1.
    total_rows = full_reader.get_row_count()
    chunks = [(i, min(i + chunk_size, total_rows)) for i in range(0, total_rows, chunk_size)]

    for chunk_start, chunk_end in chunks:
        # Read the chunk of data
        chunk_data = full_reader.read_rows(block_start=chunk_start, block_size=chunk_end - chunk_start)

        # loop that parses each row/object and unpacks
        for row in chunk_data:

            res = parse_fit_result(
                row, orbit_cols_flag, orbit_para
            )  # res.epoch is converted to jd in this function. It is converted back to mjd in output
            res_unpacked = unpack(res, row[0], orbit_para, _RESULT_DTYPES, orbit_cols_flag, format)

            # All results go to a single output file
            if file_format == "hdf5":
                write_hdf5(res_unpacked, output_file, key="data")
            else:
                write_csv(res_unpacked, output_file)

    print(f"Data has been written to {output_file}")
