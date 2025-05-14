# logging_config.py
import os
import logging


def setup_logging():
    log_level_name = os.getenv(
        "TCFG_LOG_LEVEL", "WARNING"
    ).upper()  # Default to WARNING, set 'export TCFG_LOG_LEVEL=DEBUG' on the terminal to change
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = log_levels.get(log_level_name, logging.WARNING)
    # Create a logger for the library
    logger = logging.getLogger("transformers_cfg")
    # the level will propagate to loggers with submodule scope
    logger.setLevel(log_level)
