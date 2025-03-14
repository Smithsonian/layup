#
# The `layup visualize` subcommand implementation
#
import argparse
from layup_cmdline.layupargumentparser import LayupArgumentParser


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
        type=str,
        default="csv",
        required=False,
    )
    optional.add_argument(
        "-n",
        "--num",
        help="random number of orbits to take from input file",
        dest="n",
        type=str,
        default=50,
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

    args = parser.parse_args()

    return execute(args)


def execute(args):
    print("Hello world this would start visualise")


if __name__ == "__main__":
    main()
