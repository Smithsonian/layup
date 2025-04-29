import argparse

PRE_HEADER_COMMENT_AND_EXCLUDE_STRINGS = ("#", "!")


def main():
    parser = argparse.ArgumentParser(
        prog="layup utility - remove psv spaces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This utility removes spaces from Rubin-formatted PSV files to ensure compatibility with other tools.",
    )

    req = parser.add_argument_group("Required arguments")

    req.add_argument(
        "-i",
        "--input-file",
        help="The file path to the input PSV file that needs spaces removed.",
        type=str,
        dest="input_file",
        required=True,
    )

    req.add_argument(
        "-o",
        "--output-file",
        help="The file path to the output PSV file with spaces removed.",
        type=str,
        dest="output_file",
        required=True,
    )

    optional = parser.add_argument_group("Optional arguments")

    optional.add_argument(
        "-f",
        "--force",
        help="Force overwrite of existing files without prompting.",
        type=bool,
        dest="force",
        default=False,
        required=False,
    )

    args = parser.parse_args()

    return execute(args)


def execute(args):
    from pathlib import Path
    import pandas as pd

    # Do a little input validation
    input_file_path = Path(args.input_file).resolve()
    if not input_file_path.exists():
        print(f"Error: Input file '{input_file_path}' does not exist.")
        return 1

    output_file_path = Path(args.output_file).resolve()
    if output_file_path.exists() and not args.force:
        print(f"Error: Output file '{output_file_path}' already exists.")
        print("Use --force to overwrite the existing file.")
        return 1

    # It's not necessary carry both of these variables, but it makes the logic a bit more clear.
    num_pre_header_lines = 0
    header_row_index = 0

    with open(input_file_path) as fh:
        for i, line in enumerate(fh):
            # If the line starts with a comment character, increment the pre-header line count
            if line.startswith(PRE_HEADER_COMMENT_AND_EXCLUDE_STRINGS):
                num_pre_header_lines += 1
            else:
                # Note - header row INDEX is 0-indexed.
                header_row = line
                header_row_index = num_pre_header_lines
                break

    # skip_rows is used to prevent pd.read_csv from trying to read the pre-header comments.
    skip_rows = []
    if header_row_index > 0:
        skip_rows = [i for i in range(0, header_row_index)]

    # Define the pd.read_csv "converters" functions to process each value in all columns.
    column_converters = {col_name: str.strip for col_name in header_row.strip().split("|")}

    # Read in the PSV file, removing leading and trailing spaces from all values in all columns.
    res_df = pd.read_csv(input_file_path, sep="|", skiprows=skip_rows, converters=column_converters)

    # Update the names of the columns to remove any leading or trailing spaces.
    res_df.columns = [col.strip() for col in res_df.columns]

    # Copy over the comments at the top of the input file
    with open(input_file_path) as input_file:
        with open(output_file_path, "w") as output_file:
            for _ in range(num_pre_header_lines):
                output_file.write(input_file.readline())

    # Write the header and data to the output file.
    res_df.to_csv(output_file_path, sep="|", mode="w", index=False)


if __name__ == "__main__":
    main()
