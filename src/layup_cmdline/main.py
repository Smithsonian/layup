import argparse
import sys
from importlib.metadata import distribution

#
# Generic verb dispatcher code
#


def find_layup_verbs():
    """The verbs this installation provides, as a dict of name -> entry point.

    Read from the installed package's own metadata. Do not go back to searching
    PATH for executables named layup-<verb>: that ran whichever layup came first
    on PATH, which on a machine with more than one installation was often not
    the one the user meant.
    """
    verbs = {}
    for ep in distribution("layup").entry_points:
        if ep.group == "console_scripts" and ep.name.startswith("layup-"):
            verbs[ep.name[len("layup-") :]] = ep
    return verbs


def main():
    # Discover available layup verbs
    available_verbs = find_layup_verbs()

    if not available_verbs:
        print("Error: No available 'layup-' utilities found.")
        sys.exit(1)

    # Set up the argument parser with epilog text
    description = "layup: orbit determination for solar system objects, at LSST scale."
    epilog_text = (
        "These are the layup verbs:\n\n"
        "   bootstrap      download the ephemeris and reference data layup needs\n"
        "   init           write a layup configuration file\n"
        "   orbitfit       fit orbits to observations\n"
        "   convert        convert orbits between element sets and frames\n"
        "   predict        predict on-sky positions and uncertainty ellipses\n"
        "   visualize      visualize orbits\n"
        "   comet          determine original orbits for comets\n"
        "   unpack         unpack a covariance matrix into uncertainties\n"
        "   demo           run demonstrations of the other verbs\n"
        "\n"
        "To get more information, run the verb with --help. For example:\n\n"
        "   layup orbitfit --help\n"
        " "
    )

    parser = argparse.ArgumentParser(
        description=description, epilog=epilog_text, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--version",
        help="Print version information",
        dest="version",
        action="store_true",
    )

    parser.add_argument("verb", nargs="?", choices=sorted(available_verbs), help="Verb to execute")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for the verb")

    args = parser.parse_args()

    # intercept global options (just version, for now)
    if args.version:
        import layup

        print(layup.__version__)
        return

    # Ensure a verb is provided if not just checking the version
    if not args.verb:
        parser.print_help()
        sys.exit(1)

    utility = f"layup-{args.verb}"
    entry = available_verbs.get(args.verb)
    if entry is None:
        print(f"Error: '{utility}' is not available.")
        sys.exit(1)

    # Run the verb in this process. Nothing is resolved by name, so the verb
    # that runs is always the one belonging to this installation.
    verb_main = entry.load()
    argv = sys.argv
    sys.argv = [utility, *args.args]
    try:
        code = verb_main()
    except SystemExit as exc:  # the verbs exit on their own error paths
        code = exc.code
    finally:
        sys.argv = argv
    if code not in (0, None):
        print(f"Error: Command '{utility}' failed with exit code {code}.")
        sys.exit(code)
    sys.exit(0)


if __name__ == "__main__":
    main()
