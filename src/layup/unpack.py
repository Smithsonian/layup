import numpy as np
import logging
from pathlib import Path

from layup.utilities.data_processing_utilities import parse_fit_result
from layup.utilities.file_io import CSVDataReader, HDF5DataReader
from layup.utilities.file_io.file_output import write_csv, write_hdf5

logger = logging.getLogger(__name__)


INPUT_READERS = {
    "csv": CSVDataReader,
    "hdf5": HDF5DataReader,
}


def _get_result_dtypes(primary_id_column_name: str, state: list, sigma: list):
    """Helper function to create the result dtype with the correct primary ID column name."""
    # Define a structured dtype for format of unpacked output file

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


def unpack(res, name, orbit_para, _RESULT_DTYPES):
    """
    unpacks a file containing a covarience matrix into assoicated uncertainties.
    e.g. name, values (x6), covariance (6x6) ---> name, value_i, sigma_value_i, ....

    """

    j = [0, 7, 14, 21, 28, 35]  # location of cov_ii [00,11,22,33,44,55]
    error_list = []
    for i, orbit_para in enumerate(orbit_para):
        error_list.append(np.sqrt(res.cov[j[i]]))

    output = np.array(
        [
            (
                name,
                res.csq,
                res.ndof,
            )
            + (
                res.state[0],
                error_list[0],
                res.state[1],
                error_list[1],
                res.state[2],
                error_list[2],
                res.state[3],
                error_list[3],
                res.state[4],
                error_list[4],
                res.state[5],
                error_list[5],
            )
            + (
                res.epoch - 2400000.5, # changing from jd back to mjd
                res.niter,
                res.method,
                res.flag,
                "BCART",  # The base format returned by the C++ code
            )
        ],
        dtype=_RESULT_DTYPES,
    )
    return output


def unpack_cli(
    input,
    file_format,
    output_file_stem,
):

    primary_id_column_name = "provID"

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

    # Check orbit format in the file
    input_format = None
    if "FORMAT" in sample_data.dtype.names:
        input_format = sample_data["FORMAT"][0]
    else:
        logger.error("Input file does not contain 'FORMAT' column")

    format = input_format

    if format in ["CART", "BCART"]:
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

    _RESULT_DTYPES = _get_result_dtypes(primary_id_column_name, orbit_para, orbit_para_sigma)

    # read data
    data = reader_class(
        input_file,
        format_column_name="FORMAT",
        primary_id_column_name=primary_id_column_name,
    ).read_rows()

    # loop that parses each row/object and unpacked
    for row in data:

        res = parse_fit_result(row) # res.epoch is converted to jd in this function. It is converted back to mjd in output
        res_unpacked = unpack(res, row[0], orbit_para, _RESULT_DTYPES)

        # All results go to a single output file
        if file_format == "hdf5":
            write_hdf5(res_unpacked, output_file, key="data")
        else:
            write_csv(res_unpacked, output_file)

    print(f"Data has been written to {output_file}")
