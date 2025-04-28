def print_demo_command(verb, printall=True):
    """
    Prints the current working version of the layup orbitfit demo command to the terminal, with
    optional functionality to also tell the user how to copy the demo files.

    Parameters
    -----------
    printall : boolean
        When True, prints the demo command plus the instructions for copying the demo files.
        When False, prints the demo command only.

    Returns
    -----------
    None.

    """
    if verb == "orbitfit":

        current_demo_command = "layup orbitfit cmd (demo not created yet)"
    elif verb == "convert":

        current_demo_command = "layup convert cmd (demo not created yet)"
    elif verb == "predict":

        current_demo_command = "layup predict cmd (demo not created yet)"
    elif verb == "comet":

        current_demo_command = "layup comet cmd (demo not created yet)"
    elif verb == "visualize":

        current_demo_command = "layup visualize cmd (demo not created yet)"

    print("\nThe command to run the layup orbitfit demo in this version of layup is:\n")

    print("    \033[1;38;2;255;165;0m" + current_demo_command + "\033[0m\n")

    print("WARNING: This command assumes that the demo files are in your working directory.\n")

    if printall:
        print("You can copy the demo files into your working directory by running:\n")

        print("    \033[1;38;2;255;165;0mlayup demo prepare -v " + verb + "\033[0m\n")

        print("Or, to copy them into a directory of your choice, run:\n")

        print("    \033[1;38;2;255;165;0mlayup demo prepare -v " + verb + " -p /path/to/files \033[0m\n")

    print(
        "If copying into a directory of your choice, you will need to modify the demo command to path to your files.\n"
    )
