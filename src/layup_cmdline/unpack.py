#
# The `layup unpack` subcommand implementation
#
import argparse
from layup_cmdline.layupargumentparser import LayupArgumentParser
import logging
import sys

logger = logging.getLogger(__name__)


def main():
    parser = LayupArgumentParser(
        prog="layup unpack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This would unpack covariance matrix into the orbital uncertainties",
    )

    positionals = parser.add_argument_group("Positional arguments")
    positionals.add_argument(
        help="input orbit file",
        dest="input",
        type=str,
    )

    optional = parser.add_argument_group("Optional arguments")
    
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
        default="unpacked_output",
        required=False,
    )


    args = parser.parse_args()

    return execute(args)


def execute(args):
    from layup.utilities.cli_utilities import warn_or_remove_file
    from layup.utilities.file_access_utils import find_file_or_exit, find_directory_or_exit

    # check input exists
    find_file_or_exit(args.input, "input")

    # Check that output directory exists
    find_directory_or_exit(args.o, "-o, --")
    # check format of input file
    if args.i.lower() == "csv":
        output_file = args.o + ".csv"
    elif args.i.lower() == "hdf5":
        output_file = args.o + ".h5"
    else:
        sys.exit("ERROR: File format must be 'csv' or 'hdf5'")

    # check for overwriting output file
    warn_or_remove_file(str(output_file), args.force, logger)
    from layup.unpack import unpack_cli

    unpack_cli(
        input=args.input,
        input_type = args.i,
        output_file= output_file,
    )
   



if __name__ == "__main__":
    main()
