#
# The `layup orbitfit` subcommand implementation
#
import argparse
import sys
from layup_cmdline.layupargumentparser import LayupArgumentParser
from layup.utilities.file_access_utils import find_file_or_exit
from layup.utilities.file_access_utils import find_directory_or_exit


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
        help="input file type [MPC80col, ADES_csv, ADES_psv, ADES_xml, ADES_hdf5]",
        dest="type",
        type=str,
    )

    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "--ar",
        "--ar-data-path",
        help="Directory path where Assist+Rebound data files where stored when running bootstrap_layup_data_files from the command line.",
        type=str,
        dest="ar",
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
        dest="cs",  # Change dest to cs to avoid conflict with --conf
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
    optional.add_argument("-g", "--guess", help="initial guess file", dest="g", required=False)
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
        help="output file stem. default path is current working directory",
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

    return execute(args)


def execute(args):
    print("Starting orbitfit...")

    if args.g and args.i == "gauss":
        args.i = None
    elif args.g and args.i != None:
        sys.exit("ERROR: IOD and initial guess file cannot be called together")

    find_file_or_exit(arg_fn=args.input, argname="positional input")
    if args.ar:
        find_directory_or_exit(args.ar, argname="--ar --ar-data-path")
    if not ((args.type.lower()) in ["mpc80col", "ades_csv", "ades_psv", "ades_xml", "ades_hdf5"]):
        sys.exit("Not a supported file type [MPC80col, ADES_csv, ADES_psv, ADES_xml, ADES_hdf5]")

    from layup.utilities.layup_configs import LayupConfigs

    if args.g is not None:
        find_file_or_exit(args.g, "-g, --guess")

    if args.c:
        find_file_or_exit(args.c, "-c, --config")
        configs = LayupConfigs(args.c)
        print("printing the config file filename of jpl_planets:", configs.auxiliary.jpl_planets)
    else:
        configs = LayupConfigs()
        print("printing the default filename of jpl_planets:", configs.auxiliary.jpl_planets)

    import os

    output_file = f"{args.o}.{args.of}"
    if os.path.exists(output_file) and not args.force:
        sys.exit(f"ERROR: Output file {output_file} already exists. Use -f/--force to overwrite.")

    # Not handling chunk size for now
    print(f"Loading observations from {args.input} as {args.type}")
    try:
        if args.type.lower() == "mpc80col":
            print("read the 80 column mpc format")

        elif args.type.lower() == "ades_csv":
            from layup.utilities.file_io.CSVReader import CSVDataReader

            reader = CSVDataReader(args.input, sep="csv")
            observations = reader._read_rows_internal()

        elif args.type.lower() == "ades_psv":
            from layup.utilities.file_io.CSVReader import CSVDataReader

            reader = CSVDataReader(args.input, sep="|")
            observations = reader._read_rows_internal()

        elif args.type.lower() == "ades_xml":
            print("read the xml format")

        elif args.type.lower() == "ades_hdf5":
            from layup.utilities.file_io.HDF5Reader import HDF5DataReader

            reader = HDF5DataReader(args.input)
            observations = reader._read_rows_internal()

        row_count = len(observations)
        print(f"Successfully loaded {row_count} observation records")

    except ImportError as ie:
        sys.exit(f"ERROR: Failed to import required modules for reading observations: {ie}")
    except Exception as e:
        sys.exit(f"ERROR: Failed to load observations: {e}")


if __name__ == "__main__":
    main()
