# logging_config.py
import os
import logging


def setup_logging():
    log_level_name = os.getenv(
        "TCFG_LOG_LEVEL", "WARNING"
    ).upper()  # Default to WARNING if not set
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = log_levels.get(log_level_name, logging.WARNING)
    logging.basicConfig(level=log_level)
