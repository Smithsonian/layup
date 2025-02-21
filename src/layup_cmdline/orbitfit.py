#
# The `layup orbitfit` subcommand implementation
#
import argparse
from layup_cmdline.layupargumentparser import LayupArgumentParser
from layup.utilities.fileAccessUtils import FindFileOrExit


def main():
    parser = LayupArgumentParser(
        prog="layup orbitfit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This would start orbitfit",
    )
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "-c",
        "--config",
        help="Input configuration file name",
        type=str,
        dest="c",
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
        print("print statement used for orbitfit")
    else:
        print("Hello world this would start orbitfit")

    from layup.utilities.layupConfigs import layupConfigs

    # Showing how Configs file is called and how parameters are used
    if args.c:
        FindFileOrExit(args.c, "-c, --config")
        configs = layupConfigs(args.c)
        print("printing the config file filename of jpl_planets:", configs.auxiliary.jpl_planets)
    else:
        configs = layupConfigs()
        print("printing the default filename of jpl_planets:", configs.auxiliary.jpl_planets)


if __name__ == "__main__":
    main()
