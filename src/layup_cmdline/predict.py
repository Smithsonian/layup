#
# The `layup predict` subcommand implementation
#
import argparse
from layup_cmdline.layupargumentparser import LayupArgumentParser
from astropy.time import Time
import astropy.units as u
from datetime import datetime, timezone
import logging


logger = logging.getLogger(__name__)

UNIT_DICT = {
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
        help="Directory path where Assist+Rebound data files were stored when running `layup bootstrap` from the command line.",
        type=str,
        dest="ar_data_file_path",
        required=False,
    )

    optional.add_argument(
        "-c",
        "--conf",
        help="Optional configuration file",
        type=str,
        dest="c",
        required=False,
    )

    optional.add_argument(
        "-ch",
        "--chunksize",
        help="Number of orbits to be processed at once",
        dest="chunk",
        type=int,
        default=10000,
        required=False,
    )

    optional.add_argument(
        "-d",
        "--days",
        help="Number of days to predict on sky positions. Ignored if end-date is specified on the command line.",
        dest="days",
        type=int,
        default=14,
        required=False,
    )

    optional.add_argument(
        "-e",
        "--end-date",
        help="End date as JD_TDB float or date string YYYY-mm-dd that will be converted to JD_TDB. Only required if not specifying the number of days",
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
        help="Input format type of file (csv or hdf5)",
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
        help="Output file stem. Default path is the current working directory",
        dest="o",
        type=str,
        default="predicted_output",
        required=False,
    )

    optional.add_argument(
        "-s",
        "--start-date",
        help="Start date as JD_TDB float or date string YYYY-mm-dd that will be converted to JD_TDB. Defaults to current UTC date",
        dest="s",
        type=str,
        default=str(datetime.now(timezone.utc).date()),  # current UTC date as YYYY-mm-dd
        required=False,
    )

    optional.add_argument(
        "-t",
        "--timestep",
        help="Timestep for predict. Must be string consisting of float followed by the unit (d=day, h=hour, m=minute, s=second) e.g. 1.3h or 30m",
        dest="t",
        type=str,
        default="1h",  # we would want to parse the float and the unit, and convert into day unit
        required=False,
    )

    args = parser.parse_args()

    return execute(args)


def convert_input_to_JD_TDB(input_str: str) -> float:
    """
    Convert a string to a JD_TDB date. The string can be in the format of a
    Julian date TDB for or a date string in the format YYYY-mm-dd.

    Parameters
    ----------
    input_str : str
        The input string to convert.

    Raises
    ------
    ValueError
        If the input string is not in the expected format, YYYY-mm-dd

    Returns
    -------
    float
        The converted JD_TDB date.
    """
    try:
        # Assume that input is a JD_TDB float that was converted to string. Attempt
        # to convert it back to a float.
        date_JD_TDB = float(input_str)
    except ValueError:
        # If conversion to float fails, assume that the input was a date string
        # with a format like YYYY-mm-dd. Convert that to a datetime and then to
        # a float representation of a JD_TDB date.
        try:
            date_JD_TDB = Time(
                datetime.strptime(input_str, "%Y-%m-%d"), format="datetime", scale="tdb"
            ).tdb.jd
        except ValueError:
            # If the date string is not in the expected format, raise an error.
            raise ValueError(f"Invalid date string format: {input_str}. Expected format is YYYY-mm-dd.")

    return date_JD_TDB


def execute(args):
    import astropy.units as u
    import re
    from layup.utilities.cli_utilities import warn_or_remove_file
    from layup.utilities.file_access_utils import find_file_or_exit, find_directory_or_exit
    import sys

    start_date = convert_input_to_JD_TDB(args.s)

    end_date = convert_input_to_JD_TDB(args.e) if args.e else start_date + args.days

    # check input exists
    find_file_or_exit(args.input, "input")

    # Check that output directory exists
    find_directory_or_exit(args.o, "-o, --")

    # check for overwriting output file
    warn_or_remove_file(str(args.o), args.force, logger)

    # check ar directory exists if specified
    if args.ar_data_file_path:
        find_directory_or_exit(args.ar_data_file_path, "-ar, --ar_data_path")

    # check format of input file
    if args.i.lower() == "csv":
        output_file = args.o + ".csv"
    elif args.i.lower() == "hdf5":
        output_file = args.o + ".h5"
    else:
        sys.exit("ERROR: File format must be 'csv' or 'hdf5'")

    # check for overwriting output file
    warn_or_remove_file(str(output_file), args.force, logger)

    # check that start date is before end date
    if end_date <= start_date:
        sys.exit(f"Start date {start_date} is after than end date {end_date}")

    # converting timestep argument args.t into a float in day units.
    timestep_str = args.t
    match = re.match(
        r"(?P<float>\d+(\.\d*)?)(?P<unit>\w+)", timestep_str.strip()
    )  # parses float/int and unit
    if not match:
        sys.exit(f"Could not parse timestep: {timestep_str}")
    value = float(match.group("float"))
    unit_str = match.group("unit").lower()

    if unit_str not in UNIT_DICT:
        sys.exit(f"Unsupported unit, {unit_str}, for timestep: {timestep_str}.")

    timestep_day = (value * UNIT_DICT[unit_str]).to(u.day).value  # converting value into day units

    print("Hello world this would start predict")


if __name__ == "__main__":
    main()
