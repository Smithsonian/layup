#
# The `layup bootstrap` subcommand implementation
#
import argparse
import pooch
from layup_cmdline.layupargumentparser import LayupArgumentParser

from layup.utilities.file_access_utils import find_file_or_exit


def main():
    parser = LayupArgumentParser(
        prog="layup bootstrap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This would start bootstrap",
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

    parser.add_argument(
        "--cache",
        type=str,
        default=pooch.os_cache("layup"),
        help="Local directory where downloaded files will be stored.",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Delete and re-download data files.",
    )

    args = parser.parse_args()

    return execute(args)


def execute(args):
    import concurrent.futures
    from functools import partial

    from layup.utilities.layup_configs import LayupConfigs
    from layup.utilities.bootstrap_utilties.create_meta_kernel import (
        build_meta_kernel_file,
    )
    from layup.utilities.bootstrap_utilties.download_utilities import (
        make_retriever,
        _remove_files,
        _check_for_existing_files,
        _decompress,
    )

    # Bootstrap will always take the default filenames and urls (stored in
    # layup.AuxiliaryConfigs) for the current version of layup. A user can
    # download new files by running layup and specifying in the config file
    # under the section [AUXILIARY] a new filename and url.
    if args.c:
        find_file_or_exit(args.c, "-c, --config")
        configs = LayupConfigs(args.c)
    else:
        configs = LayupConfigs()

    aux_config = configs.auxiliary

    # create the Pooch retriever that tracks and retrieves the requested files
    retriever = make_retriever(aux_config, args.cache)

    # determine if we should attempt to download or create any files.
    found_all_files = False
    if args.force:
        _remove_files(aux_config, retriever)
    else:
        print("Checking cache for existing files.")
        found_all_files = _check_for_existing_files(aux_config, retriever)

    if not found_all_files:
        # create a partial function of `Pooch.fetch` including the `_decompress` method
        fetch_partial = partial(retriever.fetch, processor=_decompress, progressbar=True)

        # download the data files in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(fetch_partial, aux_config.data_files_to_download)

        # build the meta_kernel.txt file
        build_meta_kernel_file(aux_config, retriever)

        print("Checking cache after attempting to download and create files.")
        _check_for_existing_files(aux_config, retriever)


if __name__ == "__main__":
    main()
