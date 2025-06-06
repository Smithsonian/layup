#
# The `layup comet` subcommand implementation
#
import argparse
import logging
import sys
from layup_cmdline.layupargumentparser import LayupArgumentParser

logger = logging.getLogger(__name__)


def main():
    parser = LayupArgumentParser(
        prog="layup comet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Determines original orbit for comets",
    )

    positionals = parser.add_argument_group("Positional arguments")
    positionals.add_argument(
        help="input orbit file",
        dest="input",
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
        type=str.lower,
        default="csv",
        required=False,
    )
    optional.add_argument(
        "-o",
        "--output",
        help="output file stem. default path is current working directory",
        dest="o",
        type=str,
        default="cometed_output",
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
        default="ObjID",
        required=False,
    )
    args = parser.parse_args()

    return execute(args)


def execute(args):
    from layup.comet import comet_cli
    from layup.utilities.bootstrap_utilties.download_utilities import download_files_if_missing
    from layup.utilities.file_access_utils import find_file_or_exit
    from layup.utilities.layup_configs import LayupConfigs
    from layup.utilities.cli_utilities import warn_or_remove_file
    from layup.utilities.file_access_utils import find_directory_or_exit, find_file_or_exit
    from layup.utilities.layup_logging import LayupLogger

    layup_logger = LayupLogger()
    logger = layup_logger.get_logger("layup.comet_cmdline")

    # check ar directory exists if specified
    if args.ar_data_file_path:
        find_directory_or_exit(args.ar_data_file_path, "-ar, --ar_data_path")

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

    # Check that chunk size is a positive integer
    if not isinstance(args.chunk, int) or args.chunk <= 0:
        logger.error("Chunk size must be a positive integer")

    configs = LayupConfigs()
    if args.config:
        find_file_or_exit(args.config, "-c, --config")
        configs = LayupConfigs(args.config)

    # check if bootstrap files are missing, and download if necessary
    download_files_if_missing(configs.auxiliary, args)

    comet_cli(
        input=args.input,
        output_file_stem=args.o,
        file_format=args.i,
        chunk_size=args.chunk,
        num_workers=args.n,
        cli_args=args,
    )


if __name__ == "__main__":
    main()
