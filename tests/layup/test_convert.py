import argparse
import os

import pytest
from numpy.testing import assert_allclose, assert_equal
from numpy.lib.recfunctions import drop_fields

from layup.convert import convert, convert_cli
from layup.utilities.data_processing_utilities import has_cov_columns
from layup.utilities.data_utilities_for_tests import get_test_filepath
from layup.utilities.file_io.CSVReader import CSVDataReader
from layup.utilities.file_io.HDF5Reader import HDF5DataReader


def create_argparse_object():
    parser = argparse.ArgumentParser(description="Convert orbital data formats.")
    parser.add_argument("--ar_data_file_path", type=str, required=False, help="cache directory")
    parser.add_argument(
        "--primary-id-column-name",
        type=str,
        default="ObjID",
        required=False,
        help="Column name for primary ID",
    )

    args = parser.parse_args([])

    return args


def test_convert_round_trip():
    """Convert into all 6 possible output formats and then conver the output back into its original format."""
    # TODO(wbeebe): Add additional test files to test more input formats.
    csv_input_files = ["BCOM.csv", "KEP.csv", "one_cent_orbs.csv", "two_cent_orbs.csv"]
    for csv_input_file in csv_input_files:
        input_csv_reader = CSVDataReader(get_test_filepath(csv_input_file))
        input_data = input_csv_reader.read_rows()
        input_format = input_data[0]["FORMAT"]
        if input_format == "BCOM":
            # TODO(wbeebe): The last row is a hyperbolic orbit in barycentric coordinates.
            # It is removed here to avoid a conversion failure, but we should handle this
            # case in the future, and add this row back into the test
            input_data = input_data[0 : len(input_data) - 1]
            assert len(input_data) == 813
        for output_format in ["BCOM", "BCART", "BCART_EQ", "BKEP", "COM", "CART", "KEP"]:
            # Convert to the output format.
            first_convert_data = convert(input_data, output_format, num_workers=1)
            # Convert back to the original format for round trip checking.
            output_data = convert(first_convert_data, input_format, num_workers=1)

            assert_equal(len(input_data), len(output_data))

            # Test that the columns are the same. Note that column order may not be preserved.
            for column_name in input_data.dtype.names:
                # For non-numeric columns, we can't use assert_allclose, so we use assert_equal.
                if (
                    input_data[column_name].dtype.kind == "S"
                    or input_data[column_name].dtype.kind == "U"
                    or input_data[column_name].dtype.kind == "O"
                ):
                    assert_equal(
                        input_data[column_name],
                        output_data[column_name],
                        err_msg=f"Column {column_name} not equal with dtype {input_data[column_name].dtype} after converting from {csv_input_file} to {output_format} and back",
                    )
                else:
                    # Test that we convert back to our original numeric values within a small tolerance of lost precision.
                    assert_allclose(
                        input_data[column_name],
                        output_data[column_name],
                        err_msg=f"Column {column_name} not equal with dtype {input_data[column_name].dtype} after converting from {csv_input_file} to {output_format} and back",
                    )


@pytest.mark.parametrize(
    "chunk_size, num_workers",
    [
        (10_0000, 1),
    ],
)
def test_convert_round_trip_csv(tmpdir, chunk_size, num_workers):
    """Test that the convert function works for a small CSV file."""
    cli_args = create_argparse_object()
    input_file = get_test_filepath("BCOM.csv")
    input_csv_reader = CSVDataReader(input_file, "csv")
    input_data = input_csv_reader.read_rows()

    # Since the convert CLI outputs to the current working directory, we need to change to our temp directory
    output_file_stem_BCART = "test_output_BCART"
    os.chdir(tmpdir)
    temp_BCART_out_file = os.path.join(tmpdir, f"{output_file_stem_BCART}.csv")
    # Convert our BCOM CV file to a BCART CSV file
    convert_cli(
        input_file,
        output_file_stem_BCART,
        "BCART",
        "csv",
        chunk_size=chunk_size,
        num_workers=num_workers,
        cli_args=cli_args,
    )

    # Verify the conversion produced an output file
    assert os.path.exists(temp_BCART_out_file)

    # Create a new CSV reader to read in our output BCART file
    output_csv_reader = CSVDataReader(temp_BCART_out_file, "csv")
    output_data_BCART = output_csv_reader.read_rows()
    # Verify that the number of rows in the input and output files are the same
    assert_equal(len(input_data), len(output_data_BCART))

    # Now convert back to BCOM so we can verify the round trip conversion.
    output_file_stem_BCOM = "test_output_BCOM"
    temp_BCOM_out_file = os.path.join(tmpdir, f"{output_file_stem_BCOM}.csv")
    convert_cli(
        temp_BCART_out_file,
        output_file_stem_BCOM,
        "BCOM",
        "csv",
        chunk_size=chunk_size,
        num_workers=num_workers,
        cli_args=cli_args,
    )

    # Verify the conversion produced an output file
    assert os.path.exists(temp_BCOM_out_file)
    output_csv_reader = CSVDataReader(temp_BCOM_out_file, "csv")
    output_data_BCOM = output_csv_reader.read_rows()

    # Test that the file has the same number of rows and columns as our input file
    assert_equal(len(input_data), len(output_data_BCOM))
    assert_equal(set(input_data.dtype.names), set(output_data_BCOM.dtype.names))

    # Test that the columns have equivalent values, note that column order may have changed.
    for column_name in input_data.dtype.names:
        # For non-numeric columns, we can't use assert_allclose, so we use assert_equal.
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
            # Test that we convert back to our original numeric values within a small tolerance of lost precision.
            assert_allclose(
                input_data[column_name],
                output_data_BCOM[column_name],
                err_msg=f"Column {column_name} not equal with dtype {input_data[column_name].dtype}",
            )


@pytest.mark.parametrize(
    "chunk_size, num_workers, output_format",
    [
        (10_0000, 1, "BKEP"),
        (10_0000, 1, "BCOM"),
        (10_0000, 1, "COM"),
        (10_0000, 1, "KEP"),
        (10_0000, 1, "BCART"),
        (10_0000, 1, "CART"),
    ],
)
def test_convert_BCART_EQ_csv_with_covariance(tmpdir, chunk_size, num_workers, output_format):
    """Test that the convert function works for a small CSV file."""
    cli_args = create_argparse_object()
    cli_args.primary_id_column_name = "provID"
    input_file = get_test_filepath("test_convert_BCART_EQ.csv")
    input_csv_reader = CSVDataReader(input_file, "csv", primary_id_column_name="provID")
    input_data = input_csv_reader.read_rows()

    # Since the convert CLI outputs to the current working directory, we need to change to our temp directory
    output_file_stem = f"test_output_{output_format}"
    os.chdir(tmpdir)
    temp_out_file = os.path.join(tmpdir, f"{output_file_stem}.csv")
    # Convert our BCART_EQ CSV file to a different format CSV file
    convert_cli(
        input_file,
        output_file_stem,
        output_format,
        "csv",
        chunk_size=chunk_size,
        num_workers=num_workers,
        cli_args=cli_args,
    )

    # Verify the conversion produced an output file
    assert os.path.exists(temp_out_file)

    # Create a new CSV reader to read in our output file
    output_csv_reader = CSVDataReader(temp_out_file, "csv", primary_id_column_name="provID")
    output_data = output_csv_reader.read_rows()
    # Verify that the number of rows in the input and output files are the same
    assert_equal(len(input_data), len(output_data))

    # Now convert that output file back to BCART so we can verify the round trip conversion.
    output_file_stem_BCART_EQ = "final_output_BCART_EQ"
    temp_BCART_EQ_out_file = os.path.join(tmpdir, f"{output_file_stem_BCART_EQ}.csv")
    convert_cli(
        temp_out_file,
        output_file_stem_BCART_EQ,
        "BCART_EQ",
        "csv",
        chunk_size=chunk_size,
        num_workers=num_workers,
        cli_args=cli_args,
    )

    # Verify the conversion produced an output file
    assert os.path.exists(temp_BCART_EQ_out_file)
    output_csv_reader = CSVDataReader(temp_BCART_EQ_out_file, "csv", primary_id_column_name="provID")
    output_data_BCART_EQ = output_csv_reader.read_rows()

    # Test that the file has the same number of rows and columns as our input file
    assert_equal(len(input_data), len(output_data_BCART_EQ))
    # assert_equal(set(input_data.dtype.names), set(output_data_BCOM.dtype.names))

    # Test that the columns have equivalent values, note that column order may have changed.
    assert has_cov_columns(input_data)
    assert has_cov_columns(output_data_BCART_EQ)
    for column_name in output_data_BCART_EQ.dtype.names:
        # For non-numeric columns, we can't use assert_allclose, so we use assert_equal.
        if (
            input_data[column_name].dtype.kind == "S"
            or input_data[column_name].dtype.kind == "U"
            or input_data[column_name].dtype.kind == "O"
        ):
            assert_equal(
                input_data[column_name],
                output_data_BCART_EQ[column_name],
                err_msg=f"Column {column_name} not equal with dtype {input_data[column_name].dtype}",
            )
        elif column_name:
            # Test that we convert back to our original numeric values within a small tolerance of lost precision.
            assert_allclose(
                input_data[column_name],
                output_data_BCART_EQ[column_name],
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
    cli_args = create_argparse_object()
    input_file_BCOM = get_test_filepath("BCOM.h5")
    input_hdf5_reader = HDF5DataReader(input_file_BCOM)
    input_data_BCOM = input_hdf5_reader.read_rows()

    # Since the convert CLI outputs to the current working directory, we need to change to our temp directory
    output_file_stem_BCART = "test_output_BCART"
    os.chdir(tmpdir)
    temp_out_file_BCART = os.path.join(tmpdir, f"{output_file_stem_BCART}.h5")

    # Convert our BCOM HDF5 file to a BCART HDF5 file
    convert_cli(
        input_file_BCOM,
        output_file_stem_BCART,
        "BCART",
        "hdf5",
        chunk_size=chunk_size,
        num_workers=num_workers,
        cli_args=cli_args,
    )

    # Verify the conversion produced an output file
    assert os.path.exists(temp_out_file_BCART)
    output_hdf5_reader = HDF5DataReader(temp_out_file_BCART)
    output_data_BCART = output_hdf5_reader.read_rows()
    assert_equal(len(input_data_BCOM), len(output_data_BCART))

    # Convert our output BCART file back to BCOM so we can verify the round trip conversion.
    output_file_stem_BCOM = "test_output_BCOM"
    temp_BCOM_out_file = os.path.join(tmpdir, f"{output_file_stem_BCOM}.h5")
    convert_cli(
        temp_out_file_BCART,
        output_file_stem_BCOM,
        "BCOM",
        "hdf5",
        chunk_size=chunk_size,
        num_workers=num_workers,
        cli_args=cli_args,
    )

    # Verify the conversion produced an output file
    assert os.path.exists(temp_BCOM_out_file)
    output_hdf5_reader = HDF5DataReader(temp_BCOM_out_file)
    output_data_BCOM = output_hdf5_reader.read_rows()

    # Test that the file has the same number of rows and columns as our input file
    assert_equal(len(input_data_BCOM), len(output_data_BCOM))
    assert_equal(set(input_data_BCOM.dtype.names), set(output_data_BCOM.dtype.names))

    # Test that the columns have equivalent values, note that column order may have changed.
    for column_name in input_data_BCOM.dtype.names:
        # For non-numeric columns, we can't use assert_allclose, so we use assert_equal.
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
            # Test that we convert back to our original numeric values within a small tolerance of lost precision.
            assert_allclose(
                input_data_BCOM[column_name],
                output_data_BCOM[column_name],
                err_msg=f"Column {column_name} not equal with dtype {input_data_BCOM[column_name].dtype}",
            )


def test_get_format():
    """Test that the get_format function works for a small CSV file."""
    # Test that the get_format function works for a small CSV file.
    input_file = get_test_filepath("BCOM.csv")
    input_csv_reader = CSVDataReader(input_file)
    input_data = input_csv_reader.read_rows()
    input_format = input_data[0]["FORMAT"]
    assert get_format(input_data) == input_format


def test_get_format_without_first_row():
    """Test that the get_format function works for a small CSV file that doesn't
    have a valid FORMAT in the first row."""

    input_file = get_test_filepath("BCOM.csv")
    input_csv_reader = CSVDataReader(input_file)
    input_data = input_csv_reader.read_rows()

    input_data["FORMAT"][0] = None

    input_format = input_data[1]["FORMAT"]
    assert get_format(input_data) == input_format


def test_get_format_raises_with_no_data():
    """Test that the get_format function raises error when data is empty."""

    input_file = get_test_filepath("BCOM.csv")
    input_csv_reader = CSVDataReader(input_file)
    input_data = input_csv_reader.read_rows(block_size=0)

    with pytest.raises(ValueError) as e:
        _ = get_format(input_data)
    assert "Data is empty" in str(e.value)


def test_get_format_raises_with_unknown_format_values():
    """Test that the get_format function raises error when FORMAT column is all
    None."""

    input_file = get_test_filepath("BCOM.csv")
    input_csv_reader = CSVDataReader(input_file)
    input_data = input_csv_reader.read_rows(block_size=3)
    input_data["FORMAT"][0] = "Poop"
    input_data["FORMAT"][1] = "Slap"
    input_data["FORMAT"][2] = "Fish"

    with pytest.raises(ValueError) as e:
        _ = get_format(input_data)
    assert "Data does not contain valid orbit format" in str(e.value)


def test_get_format_raises_with_no_format_column():
    """Test that the get_format function raises error when FORMAT column is not
    present."""

    input_file = get_test_filepath("BCOM.csv")
    input_csv_reader = CSVDataReader(input_file)
    input_data = input_csv_reader.read_rows(block_size=3)
    input_data = drop_fields(input_data, "FORMAT")

    with pytest.raises(ValueError) as e:
        _ = get_format(input_data)
    assert "Data does not contain 'FORMAT' column" in str(e.value)
