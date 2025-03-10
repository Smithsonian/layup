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
    required = parser.add_argument_group("Required arguments")

    required.add_argument(
        "-i",
        "--input",
        help= "input orbit file",
        dest="i",
        type=str,
        required=True,
    )
    required.add_argument(
        "-f",
        "--format",
        help= "format of input file",
        dest="f",
        type=str,
        required=True,
    )
    required.add_argument(
        "-t",
        "--type",
        help="orbit type to convert to",
        dest="t",
        type=str,
        required=True,
    )
    required.add_argument(
        "-o",
        "--output",
        help= "output file name. default path is current working directory",
        dest='o',
        type=str,
        required=True,
    )
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "-c",
        "--chunksize",
        help="number of orbits to be processed at once",
        dest="c",
        type=int,
        default= 10000,
        required=False,
    )
    
    optional.add_argument(
        "-p",
        "--print",
        help="Prints statement to terminal.",
        dest="p",
        action="store_true",
        required=False,
    )

    args = parser.parse_args()

    return execute(args)


def execute(args):
    if args.p:
        print("print statement used for convert")
    else:
        print("Hello world this would start convert")


if __name__ == "__main__":
    main()
