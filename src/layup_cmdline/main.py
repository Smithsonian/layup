import argparse
import subprocess
import sys
import shutil
import os

#
# Generic verb dispatcher code
#


def find_layup_verbs():
    """Find available layup commands in the system's PATH."""
    layup_verbs = []
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if os.path.isdir(directory):
            for item in os.listdir(directory):
                if item.startswith("layup-") and os.access(os.path.join(directory, item), os.X_OK):
                    layup_verbs.append(item[len("layup-") :])
    return sorted(set(layup_verbs))


def main():
    # Discover available layup verbs
    available_verbs = find_layup_verbs()

    if not available_verbs:
        print("Error: No available 'layup-' utilities found.")
        sys.exit(1)

    # Set up the argument parser with epilog text
    description = "layup survey simulator suite."
    epilog_text = (
        "These are the most common layup verbs:\n\n"
        "   bootstrap      Download datafiles required to run layup\n"
        "   init           Initialize layup.\n"
        "   orbitfit       fit orbits\n"
        "   convert        convert orbits to different formats\n"
        "   predict        predict orbits\n"
        "   visualize      visualize orbits\n"
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

    parser.add_argument("verb", nargs="?", choices=available_verbs, help="Verb to execute")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for the verb")

    args = parser.parse_args()

    # intercept global options (just version, for now)
    if args.version:
        import layup

        # print(layup.__version__)
        print("layup.__version__ has not be added yet")
        return

    # Ensure a verb is provided if not just checking the version
    if not args.verb:
        parser.print_help()
        sys.exit(1)

    # Construct the full command name
    utility = f"layup-{args.verb}"

    # Ensure the command is available
    if not shutil.which(utility):
        print(f"Error: '{utility}' is not available.")
        sys.exit(1)

    # Execute the command with the remaining arguments
    try:
        result = subprocess.run([utility] + args.args, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{utility}' failed with exit code {e.returncode}.")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
