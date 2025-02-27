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
        print("print statement used for convert")
    else:
        print("Hello world this would start convert")


if __name__ == "__main__":
    main()
