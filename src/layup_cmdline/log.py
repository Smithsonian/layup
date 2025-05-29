"""The `layup log` subcommand for testing multiprocess logging
This module is a minimal working example that shows how to setup logging in a
layup_cmdline verb.
"""


def main():
    return execute()


def execute():
    from layup.log import log_cli
    from layup.utilities.layup_logging import LayupLogger

    # Instantiate a LayupLogger object
    layup_logger = LayupLogger()

    # Create a logger under the root logger. NOTE - the string passed to `get_logger`
    # starts with "layup.". Failure to do this may result in lost log messages.
    logger = layup_logger.get_logger("layup.log_cmdline")

    # Use the logger to emit messages to the various handlers configured by LayupLogger.
    logger.info("About to call `log_cli`.")
    log_cli()
    logger.info("All done.")


if __name__ == "__main__":
    main()
