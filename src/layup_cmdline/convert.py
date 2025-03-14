#
# The `layup convert` subcommand implementation
#
import argparse
from layup_cmdline.layupargumentparser import LayupArgumentParser


def main():
    parser = LayupArgumentParser(
        prog="layup convert",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This would convert orbits",
    )

    positionals = parser.add_argument_group("Positional arguments")
    positionals.add_argument(
        help="input orbit file",
        dest="input",
        type=str,
    )

    positionals.add_argument(
        help="orbit reference frame to transform to [COM, BCOM, KEP, BKEP, CART, BCART]",
        dest="orbit_type",
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
        default="converted_output",
        required=False,
    )

    args = parser.parse_args()

    return execute(args)


def execute(args):
    print("Hello world this would start convert")

    from layup.convert import convert_cli

    convert_cli(
        input=args.input,
        output_file_stem=args.o,
        convert_to=args.orbit_type,
        file_format=args.i,
        chunk_size=args.chunk,
        cli_args=args,
    )


if __name__ == "__main__":
    main()
