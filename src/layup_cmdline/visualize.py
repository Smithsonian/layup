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

    frame = parser.add_argument_group("Frame arguments")
    frame.add_argument(
        "--input-plane",
        help="reference plane of the input orbit elements",
        dest="input_plane",
        type=str,
        choices=["equatorial", "ecliptic"],
        required=False,
        default=None
    )
    frame.add_argument(
        "--input-origin",
        help="origin of the input orbit elements",
        dest="input_origin",
        type=str,
        choices=["heliocentric", "barycentric"],
        required=False,
        default=None
    )

    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "--block-size",
        help="number of rows to read from file (initial load)",
        dest="block_size",
        type=int,
        default=10000,
        required=False
    )
    optional.add_argument(
        "-n",
        "--num-orbs",
        help="random number of orbits to plot (sampled without replacement)",
        dest="num_orbs",
        type=int,
        default=100,
        required=False
    )
    optional.add_argument(
        "--n-points",
        help="number of points per orbit line",
        dest="n_points",
        type=int,
        default=500,
        required=False
    )
    optional.add_argument(
        "--r-max",
        help="maximum radius (au) when drawing hyperbolic orbits",
        dest="r_max",
        type=float,
        default=50.0,
        required=False
    )
    optional.add_argument(
        "--random",
        help="randomly display --num_orbs from input file",
        dest="random",
        type=bool,
        default=False,
        required=False
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

    visualize_cli(
        input=args.input,
        input_plane=args.input_plane,
        input_origin=args.input_origin,
        num_orbs=args.num_orbs,
        block_size=args.block_size,
        n_points=args.n_points,
        r_max=args.r_max,
        random=args.random,
        cache_dir=cache_dir
    )

if __name__ == "__main__":
    main()