"""
logger.py

Centralized logging architecture for ICARUS.
Directs standard execution telemetry to a runfile while printing
critical alerts and warnings to standard output.
"""

__project__ = "ICARUS"
__author__ = "Joseph Havens"
__version__ = "3.4.0"

import logging
import sys


def get_icarus_logger(
    name: str = "ICARUS", log_file: str = "icarus_run.log"
) -> logging.Logger:
    """
    Initializes and returns the standard pipeline logger.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Format for the permanent log file
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Format for the terminal console
        console_formatter = logging.Formatter(fmt="[%(levelname)s] %(message)s")
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger
