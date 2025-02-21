#
# The `layup bootstrap` subcommand implementation
#
import argparse
from layup_cmdline.layupargumentparser import LayupArgumentParser

from layup.utilities.fileAccessUtils import FindFileOrExit


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

    optional.add_argument(
        "-c",
        "--config",
        help="Input configuration file name",
        type=str,
        dest="c",
        required=False,
    )

    args = parser.parse_args()

    return execute(args)


def execute(args):
    if args.p:
        print("print statement used for bootstrap")
    else:
        print("Hello world this would start bootstrap")
    # Showing how Configs file is called and how parameters are used

    from layup.utilities.layupConfigs import layupConfigs

    if args.c:
        FindFileOrExit(args.c, "-c, --config")
        configs = layupConfigs(args.c)
        print("printing the config file filename of jpl_planets:", configs.auxiliary.jpl_planets)
    else:
        configs = layupConfigs()
        print("printing the default filename of jpl_planets:", configs.auxiliary.jpl_planets)


if __name__ == "__main__":
    main()
