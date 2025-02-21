#
# The `layup init` subcommand implementation
#
import argparse
from layup_cmdline.layupargumentparser import LayupArgumentParser


def execute(args):  # pragma: no cover
    #
    # NOTE: DO NOT MOVE THESE IMPORTS TO THE TOP LEVEL OF THE MODULE !!!
    #
    #       Importing layup from the function and not at the top-level of the module
    #       allows us to exit quickly and print the help/error message (in case there
    #       was a mistake on the command line). Importing layup can take 5 seconds or
    #       more, and making the user wait that long just to print out an erro message
    #       is poor user experience.
    #
    from layup.utilities.layupCopyConfigs import copy_demo_configs
    import os

    copy_location = os.path.abspath(args.path)
    which_configs = "Default"
    return copy_demo_configs(copy_location, which_configs, args.force)


def main():
    parser = LayupArgumentParser(
        prog="layup init",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Initializes layup.",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="./",
        help="Filepath where you want to copy the config files. Default is current working directory.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Force deletion/overwrite of existing config file(s). Default False.",
    )

    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    main()
