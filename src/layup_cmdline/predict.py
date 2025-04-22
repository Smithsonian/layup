#
# The `layup predict` subcommand implementation
#
import argparse
from layup_cmdline.layupargumentparser import LayupArgumentParser
from astropy.time import Time
from datetime import datetime, timezone
import logging


logger = logging.getLogger(__name__)


def main():
    parser = LayupArgumentParser(
        prog="layup predict",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This would start predict",
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
        "-d",
        "--days",
        help="number of days to predict on sky positions. Only used if the end date is not specified on the command line",
        dest="days",
        type=int,
        default=14,
        required=False,
    )
    optional.add_argument(
        "-e",
        "--end-date",
        help="end date, mjd. Only specified if not specifying the number of days",
        dest="e",
        type=str,
        default=None,
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
        help="input format type of file (csv or hdf5)",
        dest="i",
        type=str,
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
        "-o",
        "--output",
        help="output file stem. default path is current working directory",
        dest="o",
        type=str,
        default="predicted_output",
        required=False,
    )

    optional.add_argument(
        "-s",
        "--start-date",
        help="start date mjd_UTC. If not specified default wil be current date",
        dest="s",
        type=str,
        default=str(
            Time(datetime.now(timezone.utc), format="datetime", scale="utc").mjd
        ),  # gives current time in mjd_UTC
        required=False,
    )

    optional.add_argument(
        "-t",
        "--timestep",
        help="timestep for predict. must be string consisting of float followed by the unit (d for day, h for hour, m for minutes, s for seconds) e.g. 1h or 30m",
        dest="t",
        type=str,
        default="1h",  # we would want to parse the float and the unit, and convert into day unit
        required=False,
    )

    args = parser.parse_args()

    return execute(args)


def execute(args):
    import astropy.units as u
    import re
    from layup.utilities.cli_utilities import warn_or_remove_file
    from layup.utilities.file_access_utils import find_file_or_exit, find_directory_or_exit
    import sys

    # check input exists
    find_file_or_exit(args.input, "input")

    # Check that output directory exists
    find_directory_or_exit(args.o, "-o, --")

    # check for overwriting output file
    warn_or_remove_file(str(args.o), args.force, logger)

    # check that days is greater than zero if being used

    if (args.e == None) and args.days <= 0:
        sys.exit(f"Number of days must be greater than 0. Value entered was : {args.days}")

    # converting timestep argument args.t into a float in day units.
    timestep_str = args.t
    match = re.match(
        r"(?P<float>\d+(\.\d*)?)(?P<unit>\w+)", timestep_str.strip()
    )  # parses float/int and unit
    if not match:
        sys.exit(f"Could not parse timestep: {timestep_str}")
    value = float(match.group("float"))
    unit_str = match.group("unit").lower()

    unit_dict = {
        "second": u.s,
        "seconds": u.s,
        "sec": u.s,
        "s": u.s,
        "minute": u.min,
        "minutes": u.min,
        "min": u.min,
        "m": u.min,
        "hour": u.h,
        "hours": u.h,
        "hr": u.h,
        "h": u.h,
        "day": u.day,
        "days": u.day,
        "d": u.day,
    }  # used to convert unit_str to an astropy.unit

    if unit_str not in unit_dict:
        sys.exit(f"Unsupported unit: {unit_str}")

    timestep_day = (value * unit_dict[unit_str]).to(u.day).value  # converting value into day units

    print("Hello world this would start predict")


if __name__ == "__main__":
    main()
