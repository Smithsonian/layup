#
# The `layup visualize` subcommand implementation
#
import argparse
from layup_cmdline.layupargumentparser import LayupArgumentParser
import logging
import sys

logger = logging.getLogger(__name__)


def main():
    parser = LayupArgumentParser(
        prog="layup visualize",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This would start visualize",
    )

    positionals = parser.add_argument_group("Positional arguments")
    positionals.add_argument(
        help="input orbit file",
        dest="input",
        type=str,
    )

    optional = parser.add_argument_group("Optional arguments")
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
        "-n",
        "--num",
        help="random number of orbits to take from input file",
        dest="n",
        type=str,
        default=1000,
        required=False,
    )
    optional.add_argument(
        "-d",
        "--dimensions",
        help="dimensions the plot will be in [2D, 3D]",
        dest="d",
        type=str,
        default="2D",
        required=False,
    )
    optional.add_argument(
        "-b",
        "--backend",
        help="backend used for plotting [matplot, plotly]",
        dest="b",
        type=str,
        default="matplot",
        required=False,
    )
    optional.add_argument(
        "-o",
        "--output",
        help="output file stem. default path is current working directory",
        dest="o",
        type=str,
        default="output",
        required=False,
    )

    args = parser.parse_args()

    return execute(args)


def execute(args):

    from layup.utilities.file_access_utils import find_file_or_exit, find_directory_or_exit

    # from layup.visualize import visualize_cli

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

    if args.d.upper() not in ["2D", "3D"]:
        sys.exit("ERROR: -d --dimensions must be '2D' or '3D'")

    if args.b not in ["matplot", "plotly"]:
        sys.exit("ERROR: -b --backend must be 'matplot' or 'plotly'")
    # visualize_cli(
    #     input = args.input,
    #     num = args.n,
    #     dimensions = args.d,
    #     backend = args.b,
    #     output = output_file,
    # )


if __name__ == "__main__":
    main()
