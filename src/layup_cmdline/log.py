#
# The `layup log` subcommand for testing multiprocess logging
#


def main():
    return execute()


def execute():
    from layup.log import log_cli
    from layup.utilities.layup_logging import LayupLogger

    layup_logger = LayupLogger()
    logger = layup_logger.get_logger("layup.log_cmdline")
    logger.info("About to call `log_cli`.")
    log_cli(layup_logger)
    logger.info("All done.")


if __name__ == "__main__":
    main()
