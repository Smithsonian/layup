#
# The `layup predict` subcommand implementation
#
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import astropy.units as u

from layup_cmdline.layupargumentparser import LayupArgumentParser

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

SEC_PER_DAY = 24 * 60 * 60


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
        "-ar",
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
        dest="config",
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
        help="End date as JD_TDB float or date string YYYY-mm-ddTDB that will be converted to JD_TDB. Only required if not specifying the number of days",
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
        help="Start date as JD_TDB float or date string YYYY-mm-ddTDB that will be converted to JD_TDB. Defaults to current UTC date.",
        dest="s",
        type=str,
        default=str(datetime.now(timezone.UTC).date()),  # current UTC date as YYYY-mm-dd
        required=False,
    )

    optional.add_argument(
        "-st",
        "--station",
        type=str,
        dest="station",
        default="X05",  # Rubin observatory
        help="Station code for the observatory. Default is X05 (Rubin observatory). See https://www.minorplanetcenter.net/iau/lists/ObsCodes.html",
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

    optional.add_argument(
        "-pid",
        "--primary-id-column-name",
        help="Column name in input file that contains the primary ID of the object.",
        dest="primary_id_column_name",
        type=str,
        default="provID",
        required=False,
    )

    args = parser.parse_args()

    return execute(args)


def convert_input_to_JD_TDB(input_str: str, cache_path: Path) -> float:
    """
    Convert a string to a JD_TDB date. The string can be in the format of a
    Julian date TDB for or a date string in the format YYYY-mm-dd.

    Parameters
    ----------
    input_str : str
        The input string to convert.
    cache_path : Path, optional
        The path to the cache directory. If not provided, the default cache
        directory will be used.

    Raises
    ------
    ValueError
        If the input string is not in the expected format, YYYY-mm-dd

    Returns
    -------
    float
        The converted JD_TDB date.
    """
    from layup.utilities.datetime_conversions import convert_tdb_date_to_julian_date

    try:
        # Assume that input is a JD_TDB float that was converted to string. Attempt
        # to convert it back to a float.
        date_JD_TDB = float(input_str)
    except ValueError:
        try:
            # If conversion to float fails, assume that the input was a date string
            # in the format YYYY-mm-ddTDB.
            date_JD_TDB = convert_tdb_date_to_julian_date(input_str, str(cache_path))
        except:
            # Several different exceptions can be raised here, but they all allude
            # to the fact that the input string is not in the expected format.
            raise ValueError(
                f"Could not parse input date string '{input_str}'. Try the format YYYY-mm-ddTDB."
            )

    return date_JD_TDB


def execute(args):
    import re
    import sys

    import astropy.units as u
    import pooch

    from layup.predict import predict_cli
    from layup.utilities.bootstrap_utilties.download_utilities import download_files_if_missing
    from layup.utilities.cli_utilities import warn_or_remove_file
    from layup.utilities.file_access_utils import find_directory_or_exit, find_file_or_exit
    from layup.utilities.layup_configs import LayupConfigs

    # check input exists
    find_file_or_exit(args.input, "input")

    # Check that output directory exists
    find_directory_or_exit(args.o, "-o, --output")

    # check for overwriting output file
    warn_or_remove_file(str(args.o), args.force, logger)

    # check ar directory exists if specified
    if args.ar_data_file_path:
        find_directory_or_exit(args.ar_data_file_path, "-ar, --ar_data_path")
        cache_dir = Path(args.ar_data_file_path)
    else:
        cache_dir = Path(pooch.os_cache("layup"))

    start_date = convert_input_to_JD_TDB(args.s, cache_dir)

    end_date = convert_input_to_JD_TDB(args.e, cache_dir) if args.e else start_date + args.days

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

    configs = LayupConfigs()
    if args.config:
        find_file_or_exit(args.config, "-c, --config")
        configs = LayupConfigs(args.config)

    # check if bootstrap files are missing, and download if necessary
    download_files_if_missing(configs.auxiliary, args)

    predict_cli(
        cli_args=args,
        input_file=args.input,
        start_date=start_date,
        end_date=end_date,
        timestep_day=timestep_day,
        output_file=output_file,
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    main()
