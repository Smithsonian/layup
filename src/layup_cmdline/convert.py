#
# The `layup convert` subcommand implementation
#
import argparse
from layup_cmdline.layupargumentparser import LayupArgumentParser
import logging
import sys

logger = logging.getLogger(__name__)


def main():
    parser = LayupArgumentParser(
        prog="layup convert",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This would convert orbits",
    )

    positionals = parser.add_argument_group("Positional arguments")
    positionals.add_argument(
        help="input orbit file",
        dest="input",
        type=str,
    )

    positionals.add_argument(
        help="orbit reference frame to transform to [COM, BCOM, KEP, BKEP, CART, BCART]",
        dest="orbit_type",
        type=str,
    )

    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "--ar",
        "--ar-data-path",
        help="Directory path where Assist+Rebound data files where stored when running bootstrap_sorcha_data_files from the command line.",
        type=str,
        dest="ar_data_file_path",
        required=False,
    )
    optional.add_argument(
        "-c",
        "--conf",
        help="optional configuration file",
        type=str,
        dest="c",
        required=False,
    )
    optional.add_argument(
        "-ch",
        "--chunksize",
        help="number of orbits to be processed at once",
        dest="chunk",
        type=int,
        default=10000,
        required=False,
    )
    optional.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output file",
    )
    optional.add_argument(
        "-i",
        "--input-type",
        help="input format type of file",
        dest="i",
        type=str,
        default="csv",
        required=False,
    )
    optional.add_argument(
        "-o",
        "--output",
        help="output file stem. default path is current working directory",
        dest="o",
        type=str,
        default="converted_output",
        required=False,
    )

    optional.add_argument(
        "-n",
        "--num-workers",
        help="Number of CPU workers to use for parallel processing each chunk. -1 uses all available CPUs.",
        dest="n",
        type=int,
        default=-1,
        required=False,
    )

    args = parser.parse_args()

    return execute(args)


def execute(args):
    from layup.convert import convert_cli
    from layup.utilities.cli_utilities import warn_or_remove_file
    from layup.utilities.file_access_utils import find_file_or_exit, find_directory_or_exit

    # check ar directory exists if specified
    if args.ar_data_file_path:
        find_directory_or_exit(args.ar_data_file_path, "-ar, --ar_data_path")

    # check input exists
    find_file_or_exit(args.input, "input")
    # check format of input file 
    if args.i.lower() == "csv":
        output_file = args.o + ".csv"
    elif args.i.lower() == "hdf5":
        output_file = args.o + ".h5"
    else:
        sys.exit("File format must be 'csv' or 'hdf5'")

    # check for overwriting output file
    warn_or_remove_file(str(output_file), args.force, logger)

    # Check that the conversion type is valid
    if args.orbit_type not in ["BCART", "BCOM", "BKEP", "CART", "COM", "KEP"]:
        logger.error("Conversion type must be 'BCART', 'BCOM', 'BKEP', 'CART', 'COM', or 'KEP'")


    # Check that chunk size is a positive integer
    if not isinstance(args.chunk, int) or args.chunk <= 0:
        logger.error("Chunk size must be a positive integer")
    convert_cli(
        input=args.input,
        output_file_stem=args.o,
        convert_to=args.orbit_type,
        file_format=args.i,
        chunk_size=args.chunk,
        num_workers=args.n,
        cli_args=args,
    )


if __name__ == "__main__":
    main()
