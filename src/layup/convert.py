import os
import math
from pathlib import Path
from typing import Literal

from concurrent.futures import ProcessPoolExecutor

from layup.file_io.CSVReader import CSVDataReader
from layup.file_io.HDF5Reader import HDF5DataReader
from layup.file_io.file_output import write_csv, write_hdf5


def process_chunk(data):
    pass


def convert(
    input: str,
    output_file_stem: str,
    convert_to: Literal["BCART", "BCOM", "BKEP", "CART", "COM", "KEP"],
    file_format: Literal["csv", "hdf5"] = "csv",
    chunk_size: int = 10_000,
    num_workers: int = -1,
):
    input_file = Path(input)
    output_file = Path(f"{output_file_stem}.{file_format.lower()}")
    output_directory = output_file.parent.resolve()

    if num_workers < 0:
        num_workers = os.cpu_count()

    # Check that input file exists
    if not input_file.exists():
        raise FileNotFoundError(f"Input file {input_file} does not exist")

    # Check that output directory exists
    if not output_directory.exists():
        raise FileNotFoundError(f"Output directory {output_directory} does not exist")

    # Check that chunk size is a positive integer
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer")

    # Check that the file format is valid
    if file_format.lower() not in ["csv", "hdf5"]:
        raise ValueError("File format must be 'csv' or 'hdf5'")

    # Check that the conversion type is valid
    if convert_to not in ["BCART", "BCOM", "BKEP", "CART", "COM", "KEP"]:
        raise ValueError("Conversion type must be 'BCART', 'BCOM', 'BKEP', 'CART', 'COM', or 'KEP'")

    # Open the input file and read the first line
    if file_format is "hdf5":
        reader = HDF5DataReader(input_file, format_column_name="FORMAT")
    else:
        reader = CSVDataReader(input_file, format_column_name="FORMAT")

    sample_data = reader.read_rows(block_start=0, block_size=1)

    # Check orbit format in the file
    input_format = None
    if "FORMAT" in sample_data[0]:
        input_format = sample_data[0]["FORMAT"]
    else:
        raise ValueError("Input file does not contain 'FORMAT' column")

    # Check that the input format is not already the desired format
    if convert_to == input_format:
        raise ValueError("Input file is already in the desired format")

    # TODO Need to implement a get_row_count function in the readers
    total_rows = reader.get_row_count()
    chunks = [(i, min(i + chunk_size, total_rows)) for i in range(0, total_rows, chunk_size)]

    for chunk_start, chunk_end in chunks:
        # Define the blocks of each chunk to send to the workersc
        block_size = max(1, math.ceil(int(chunk_size / num_workers)))
        blocks = [(i, min(i + block_size, chunk_end)) for i in range(chunk_start, chunk_end, block_size)]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_chunk, reader.read_rows(block_start=start, block_size=end - start))
                for start, end in blocks
            ]
            for future in futures:
                if file_format == "hdf5":
                    # TODO make the output key match the input key
                    write_hdf5(future.result(), output_file, key="data")
                else:
                    write_csv(future.result(), output_file)

    print(f"Data has been written to {output_file}")
