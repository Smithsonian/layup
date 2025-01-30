#
# The `layup bootstrap` subcommand implementation
#
import argparse
from layup_cmdline.layupargumentparser import LayupArgumentParser
import os


def main():
    parser = LayupArgumentParser(
        prog="layup bootstrap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This would start bootstrap",
    )
    optional = parser.add_argument_group("Optional arguments")
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
        print("print statement used for bootstrap")
    else:
        print("Hello world this would start bootstrap")


if __name__ == "__main__":
    main()
