import argparse
import logging

from layup_cmdline.layupargumentparser import LayupArgumentParser

logger = logging.getLogger(__name__)


def main():
    parser = LayupArgumentParser(
        prog="layup visualize",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Launch the interactive orbit visualizer (Dash).",
    )

    positionals = parser.add_argument_group("Positional arguments")
    positionals.add_argument(
        help="input orbit file (csv or hdf5)",
        dest="input",
        type=str,
    )

    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "--block-size",
        help="number of rows to read from file (initial load)",
        dest="block_size",
        type=int,
        default=10000,
        required=False,
    )
    optional.add_argument(
        "-n",
        "--num-orbs",
        help="random number of orbits to plot (sampled without replacement)",
        dest="num_orbs",
        type=int,
        default=100,
        required=False,
    )
    optional.add_argument(
        "--n-points",
        help="number of points per orbit line",
        dest="n_points",
        type=int,
        default=500,
        required=False,
    )
    optional.add_argument(
        "--r-max",
        help="maximum radius (au) when drawing hyperbolic orbits",
        dest="r_max",
        type=float,
        default=50.0,
        required=False,
    )
    optional.add_argument(
        "--random",
        help="randomly display --num_orbs from input file",
        dest="random",
        type=bool,
        default=False,
        required=False,
    )
    optional.add_argument(
        "--special",
        help="optional second orbit file whose orbits are highlighted in a distinct colour (regular orbits are greyed out when this is supplied)",
        dest="special",
        type=str,
        default=None,
        required=False,
    )
    optional.add_argument(
        "--ar-data-file-path",
        dest="ar_data_file_path",
        type=str,
        default=None,
        required=False,
        help=argparse.SUPPRESS,  # hide from --help, i don't think a visualize user will be specifying this, but it is needed for bootstrapping fles if user doesn't have them
    )

    args = parser.parse_args()
    return execute(args)


def execute(args):
    from layup.visualize import visualize_cli
    from layup.utilities.file_access_utils import find_file_or_exit
    from layup.utilities.bootstrap_utilties.download_utilities import download_files_if_missing
    from layup.utilities.layup_configs import LayupConfigs

    configs = LayupConfigs()
    download_files_if_missing(configs.auxiliary, args)

    cache_dir = getattr(args, "ar_data_file_path", None)

    find_file_or_exit(args.input, "input")
    if args.special is not None:
        find_file_or_exit(args.special, "special")

    visualize_cli(
        input=args.input,
        num_orbs=args.num_orbs,
        block_size=args.block_size,
        n_points=args.n_points,
        r_max=args.r_max,
        random=args.random,
        cache_dir=cache_dir,
        special=args.special,
    )


if __name__ == "__main__":
    main()
