#
# The `layup log` subcommand for testing multiprocess logging
#
from layup.utilities.layup_logging import LayupLogger


def main():
    return execute()


def execute():
    from layup.log import log_cli

    with LayupLogger() as layup_logger:
        logger = layup_logger.get_logger(__name__)
        logger.info("About to call `log_cli`.")
        log_cli(layup_logger)
        logger.info("All done.")


if __name__ == "__main__":
    main()
