#
# The `layup orbitfit` subcommand implementation
#
import argparse
import logging
import sys

from layup.utilities.cli_utilities import warn_or_remove_file
from layup.utilities.file_access_utils import find_directory_or_exit, find_file_or_exit
from layup.cmdline.layupargumentparser import LayupArgumentParser


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
        help="Directory path where Assist+Rebound data files were stored when running `layup bootstrap` from the command line.",
        type=str,
        dest="ar_data_file_path",
        required=False,
    )

    optional.add_argument(
        "-c",
        "--conf",
        help="optional configuration file",
        type=str,
        dest="config",
        required=False,
    )
    optional.add_argument(
        "--chunksize",
        help="number of orbits to be processed at once",
        dest="chunksize",
        type=int,
        default=10000,
        required=False,
    )
    optional.add_argument(
        "-d",
        "--debias",
        action="store_true",
        help="Perform debiasing of the input astrometry based on catalog and epoch.",
        required=False,
        dest="debias",
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
        dest="iod",
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
        "--output_format",
        help="output file format.",
        dest="output_format",
        type=str.lower,
        default="csv",
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

    optional.add_argument(
        "-pid",
        "--primary-id-column-name",
        help="Column name in input file that contains the primary ID of the object.",
        dest="primary_id_column_name",
        type=str,
        default="provID",
        required=False,
    )

    optional.add_argument(
        "-sf",
        "--separate-flagged",
        help="Split flagged results into separate output file. Flagged results file is called `output_file_stem` + '_flagged', i.e. 'output_flagged.csv'. Default is False.",
        dest="separate_flagged",
        action="store_true",
        required=False,
    )

    optional.add_argument(
        "-of",
        "--output-orbit-format",
        help="Orbit format for output file. [KEP, CART, COM, BKEP, BCART, BCART_EQ, BCOM]",
        default="BCART_EQ",
        dest="output_orbit_format",
        required=False,
    )

    optional.add_argument(
        "-wd",
        "--weight-data",
        action="store_true",
        help="Apply data weighting based on the observation code, date, catalog and program. ",
        required=False,
        dest="weight_data",
    )

    args = parser.parse_args()

    return execute(args)


def execute(args):
    from layup.orbitfit import orbitfit_cli
    from layup.utilities.bootstrap_utilties.download_utilities import download_files_if_missing

    logger = logging.getLogger(__name__)

    logger.info("Starting orbitfit...")

    if args.g and args.i == "gauss":
        args.i = None
    elif args.g and args.i != None:
        sys.exit("ERROR: IOD and initial guess file cannot be called together")

    find_file_or_exit(arg_fn=args.input, argname="positional input")
    if args.ar_data_file_path:
        find_directory_or_exit(args.ar_data_file_path, argname="--a --ar-data-path")
    if (args.type.lower()) not in ["mpc80col", "ades_csv", "ades_psv", "ades_xml", "ades_hdf5"]:
        sys.exit("Not a supported file type [MPC80col, ADES_csv, ADES_psv, ADES_xml, ADES_hdf5]")

    # check orbit format
    if args.output_orbit_format not in ["BCART", "BCART_EQ", "BCOM", "BKEP", "CART", "COM", "KEP"]:
        logger.error(
            "ERROR: output orbit format must be 'BCART', 'BCART_EQ', 'BCOM', 'BKEP', 'CART', 'COM', or 'KEP'"
        )
    # check format of input file
    if args.output_format.lower() == "csv":
        output_file = args.o + ".csv"
    elif args.output_format.lower() == "hdf5":
        output_file = args.o + ".h5"
    else:
        sys.exit("ERROR: File format must be 'csv' or 'hdf5'")

    # check for overwriting output file
    warn_or_remove_file(str(output_file), args.force, logger)
    from layup.utilities.layup_configs import LayupConfigs

    if args.g is not None:
        find_file_or_exit(args.g, "-g, --guess")

    configs = LayupConfigs()
    if args.config:
        find_file_or_exit(args.config, "-c, --config")
        configs = LayupConfigs(args.config)

    # check if bootstrap files are missing, and download if necessary
    download_files_if_missing(configs.auxiliary, args)

    orbitfit_cli(
        input=args.input,
        input_file_format=args.type,
        output_file_stem=args.o,
        output_file_format=args.output_format,
        chunk_size=args.chunksize,
        num_workers=args.n,
        cli_args=args,
    )


if __name__ == "__main__":
    main()
