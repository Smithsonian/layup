#
# The `layup orbitfit` subcommand implementation
#
import argparse
from layup_cmdline.layupargumentparser import LayupArgumentParser
from layup.utilities.file_access_utils import find_file_or_exit


def main():
    parser = LayupArgumentParser(
        prog="layup orbitfit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This would start orbitfit",
    )
    positionals = parser.add_argument_group("Positional arguments")
    positionals.add_argument(
        help="astrometry input file",
        dest="input",
        type=str,
    )
    positionals.add_argument(
        help = "input file type [MPC80col, ADES_csv, ADES_psv, ADES_xml, ADES_hdf5]",
        dest="type",
        type=str,
    )

    optional = parser.add_argument_group("Optional arguments")
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
        dest="c",
        type=int,
        default=10000,
        required=False,
    )

    optional.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output file",
        required=False,
    )
    optional.add_argument(
        "-g",
        "--guess",
        help="initial guess file",
        dest="g",
        required=False
        )
    optional.add_argument(
        "-i",
        "--iod",
        help="IOD choice",
        dest="i",
        default="gauss",
        required=False,
    )
    optional.add_argument(
        "-o",
        "--output",
        help="output file name. default path is current working directory",
        dest="o",
        type=str,
        default="output",
        required=False,
    )
    optional.add_argument(
        "-of",
        "--output_format",
        help="output file format.",
        dest="of",
        type=str,
        default="csv",
        required=False,
    )
    
    args = parser.parse_args()
    if args.g:
        args.i = None

    return execute(args)


def execute(args):
    print("Hello world this would start orbitfit")

    from layup.utilities.layup_configs import LayupConfigs

    # Showing how Configs file is called and how parameters are used
    if args.c:
        find_file_or_exit(args.c, "-c, --config")
        configs = LayupConfigs(args.c)
        print("printing the config file filename of jpl_planets:", configs.auxiliary.jpl_planets)
    else:
        configs = LayupConfigs()
        print("printing the default filename of jpl_planets:", configs.auxiliary.jpl_planets)


if __name__ == "__main__":
    main()
