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
        type=str.lower,
        default="csv",
        required=False,
    )
    optional.add_argument(
        "--orbit_type",
        help="orbit reference frame of orbit from [COM, BCOM, KEP, BKEP, CART, BCART]",
        dest="orbit_type",
        type=str,
    )
    optional.add_argument(
        "-n",
        "--num",
        help="random number of orbits to take from input file",
        dest="n",
        type=int,
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
    optional.add_argument(
        "--fade", help="fade out the orbits of input objects", dest="fade", action="store_true"
    )
    optional.add_argument(
        "--planets",
        "-p",
        help="choose which planets to overplot. must be from ['Me', 'V', 'E', 'Ma', 'J', 'S', 'U', 'N']",
        dest="planets",
        nargs="+",
        required=False,
    )
    optional.add_argument(
        "--plot_planets",
        help="overplot the planets. default is True",
        dest="plot_planets",
        action="store_true",
    )
    optional.add_argument(
        "--plot_sun", help="overplot the sun. default is True", dest="plot_sun", action="store_true"
    )
    optional.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output file",
    )

    args = parser.parse_args()

    return execute(args)


def execute(args):
    from layup.visualize import visualize_cli
    from layup.utilities.cli_utilities import warn_or_remove_file
    from layup.utilities.file_access_utils import find_file_or_exit, find_directory_or_exit
    from layup.utilities.layup_logging import LayupLogger

    layup_logger = LayupLogger()
    logger = layup_logger.get_logger("layup.visualize_cmdline")

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
        logger.error("File format must be 'csv' or 'hdf5'")
        sys.exit("ERROR: File format must be 'csv' or 'hdf5'")

    # check for overwriting output file
    warn_or_remove_file(str(output_file), args.force, logger)

    if args.d.upper() not in ["2D", "3D"]:
        logger.error(f"Value for -d --dimensions must be '2D' or '3D', but is {args.d.upper()}")
        sys.exit("ERROR: -d --dimensions must be '2D' or '3D'")

    if args.b not in ["matplot", "plotly"]:
        logger.error(f"Value for -b --backend must be 'matplot' or 'plotly', but is {args.b}")
        sys.exit("ERROR: -b --backend must be 'matplot' or 'plotly'")

    if args.plot_planets and not args.planets:
        logger.warning("WARNING: --plot-planets given without --planets <list>. defaulting to all planets")
        args.planets = ["Me", "V", "E", "Ma", "J", "S", "U", "N"]

    visualize_cli(
        input=args.input,
        output_file_stem=args.o,
        planets=args.planets,
        input_format=args.orbit_type,
        backend=args.b,
        dimensions=args.d,
        num_orbs=args.n,
        plot_planets=args.plot_planets,
        plot_sun=args.plot_sun,
        fade=args.fade,
    )


if __name__ == "__main__":
    main()
