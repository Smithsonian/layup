from layup.convert import convert, convert_cli
from layup.utilities.data_utilities_for_tests import get_test_filepath

from layup.utilities.file_io.CSVReader import CSVDataReader
from layup.utilities.file_io.HDF5Reader import HDF5DataReader

from numpy.testing import assert_equal, assert_allclose
import os
import pytest


def test_convert_round_trip():
    # Convert BCOM into all 6 possible output formats
    hdf5_input_files = ["BCOM.h5", "CART.h5", "KEP.h5"]
    for hdf5_input_file in hdf5_input_files:
        input_hdf5_reader = HDF5DataReader(get_test_filepath(hdf5_input_file))
        input_data = input_hdf5_reader.read_rows()
        input_format = input_data[0]["FORMAT"]
        for output_format in ["BCOM", "BCART", "BKEP", "COM", "CART", "KEP"]:
            # TODO FIX formats that currently don't work
            # Column `q` is the column with differences
            if input_format == "BCOM":
                if output_format in ["KEP"]:
                    continue

            # Column `x` is all nan
            if input_format == "CART":
                if output_format in ["BKEP"]:
                    continue

            first_convert_data = convert(input_data, output_format, num_workers=1)
            # Convert back to the original format for round trip checking
            output_data = convert(first_convert_data, input_format, num_workers=1)

            assert_equal(len(input_data), len(output_data))

            for column_name in input_data.dtype.names:
                if (
                    input_data[column_name].dtype.kind == "S"
                    or input_data[column_name].dtype.kind == "U"
                    or input_data[column_name].dtype.kind == "O"
                ):
                    assert_equal(
                        input_data[column_name],
                        output_data[column_name],
                        err_msg=f"Column {column_name} not equal with dtype {input_data[column_name].dtype} after converting from {hdf5_input_file} to {output_format} and back",
                    )
                else:
                    # Assert equal with message if not equal printing the column name
                    assert_allclose(
                        input_data[column_name],
                        output_data[column_name],
                        rtol=1,  # TODO reevaluate
                        atol=1,  # TODO reevaluate
                        err_msg=f"Column {column_name} not equal with dtype {input_data[column_name].dtype} after converting from {hdf5_input_file} to {output_format} and back",
                    )


@pytest.mark.parametrize(
    "chunk_size, num_workers",
    [
        (10_0000, 1),
    ],
)
def test_convert_round_trip_csv(tmpdir, chunk_size, num_workers):
    """Test that the convert function works for a small CSV file."""
    input_file = get_test_filepath("BCOM.csv")
    input_csv_reader = CSVDataReader(input_file, "csv")
    input_data = input_csv_reader.read_rows()

    # Since the convert CLI outputs to the current working directory, we need to change to our temp directory
    output_file_stem_BCART = "test_output_BCART"
    os.chdir(tmpdir)
    temp_BCART_out_file = os.path.join(tmpdir, f"{output_file_stem_BCART}.csv")
    convert_cli(
        input_file,
        output_file_stem_BCART,
        "BCART",
        "csv",
        chunk_size=chunk_size,
        num_workers=num_workers,
    )

    assert os.path.exists(temp_BCART_out_file)

    output_csv_reader = CSVDataReader(temp_BCART_out_file, "csv")
    output_data_BCART = output_csv_reader.read_rows()
    # TODO we can round trip to test but right now convert simply copies and reappends data
    assert_equal(len(input_data), len(output_data_BCART))

    output_file_stem_BCOM = "test_output_BCOM"
    temp_BCOM_out_file = os.path.join(tmpdir, f"{output_file_stem_BCOM}.csv")
    convert_cli(
        temp_BCART_out_file,
        output_file_stem_BCOM,
        "BCOM",
        "csv",
        chunk_size=chunk_size,
        num_workers=num_workers,
    )

    assert os.path.exists(temp_BCOM_out_file)

    output_csv_reader = CSVDataReader(temp_BCOM_out_file, "csv")
    output_data_BCOM = output_csv_reader.read_rows()

    assert_equal(len(input_data), len(output_data_BCOM))
    # Test that the sets of column names are the same
    assert_equal(set(input_data.dtype.names), set(output_data_BCOM.dtype.names))

    for column_name in input_data.dtype.names:
        # TODO(wilsonbb): remove this once we correctly have these columns in radians rather than degrees.
        if column_name in set(["inc", "node", "argPeri"]):
            continue
        # check if the column has string data
        if (
            input_data[column_name].dtype.kind == "S"
            or input_data[column_name].dtype.kind == "U"
            or input_data[column_name].dtype.kind == "O"
        ):
            assert_equal(
                input_data[column_name],
                output_data_BCOM[column_name],
                err_msg=f"Column {column_name} not equal with dtype {input_data[column_name].dtype}",
            )
        else:
            # Assert equal with message if not equal printing the column name
            assert_allclose(
                input_data[column_name],
                output_data_BCOM[column_name],
                err_msg=f"Column {column_name} not equal with dtype {input_data[column_name].dtype}",
            )


@pytest.mark.parametrize(
    "chunk_size, num_workers",
    [
        (10_0000, 1),
    ],
)
def test_convert_round_trip_hdf5(tmpdir, chunk_size, num_workers):
    # Test that the convert function works for a small HDF5 file.
    input_file_BCOM = get_test_filepath("BCOM.h5")
    input_hdf5_reader = HDF5DataReader(input_file_BCOM)
    input_data_BCOM = input_hdf5_reader.read_rows()

    # Since the convert CLI outputs to the current working directory, we need to change to our temp directory
    output_file_stem_BCART = "test_output_BCART"
    os.chdir(tmpdir)
    temp_out_file_BCART = os.path.join(tmpdir, f"{output_file_stem_BCART}.h5")
    convert_cli(
        input_file_BCOM,
        output_file_stem_BCART,
        "BCART",
        "hdf5",
        chunk_size=chunk_size,
        num_workers=num_workers,
    )

    assert os.path.exists(temp_out_file_BCART)

    output_hdf5_reader = HDF5DataReader(temp_out_file_BCART)
    output_data_BCART = output_hdf5_reader.read_rows()

    assert_equal(len(input_data_BCOM), len(output_data_BCART))

    # Convert back to BCOM
    output_file_stem_BCOM = "test_output_BCOM"
    temp_BCOM_out_file = os.path.join(tmpdir, f"{output_file_stem_BCOM}.h5")
    convert_cli(
        temp_out_file_BCART,
        output_file_stem_BCOM,
        "BCOM",
        "hdf5",
        chunk_size=chunk_size,
        num_workers=num_workers,
    )

    assert os.path.exists(temp_BCOM_out_file)

    output_hdf5_reader = HDF5DataReader(temp_BCOM_out_file)
    output_data_BCOM = output_hdf5_reader.read_rows()

    assert_equal(len(input_data_BCOM), len(output_data_BCOM))
    # Test that the sets of column names are the same
    assert_equal(set(input_data_BCOM.dtype.names), set(output_data_BCOM.dtype.names))
    for column_name in input_data_BCOM.dtype.names:
        # TODO(wilsonbb): remove this once we correctly have these columns in radians rather than degrees.
        if column_name in set(["inc", "node", "argPeri"]):
            continue
        if (
            input_data_BCOM[column_name].dtype.kind == "S"
            or input_data_BCOM[column_name].dtype.kind == "U"
            or input_data_BCOM[column_name].dtype.kind == "O"
        ):
            assert_equal(
                input_data_BCOM[column_name],
                output_data_BCOM[column_name],
                err_msg=f"Column {column_name} not equal with dtype {input_data_BCOM[column_name].dtype}",
            )
        else:
            # Assert equal with message if not equal printing the column name
            assert_allclose(
                input_data_BCOM[column_name],
                output_data_BCOM[column_name],
                err_msg=f"Column {column_name} not equal with dtype {input_data_BCOM[column_name].dtype}",
            )
