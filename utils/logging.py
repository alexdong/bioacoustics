# -*- coding: utf-8 -*-
"""Basic logging setup."""

import sys
from typing import Dict

# Logging Levels
LOG_LEVELS: Dict[str, int] = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
# Set default log level here or configure via env variable/args later
CURRENT_LOG_LEVEL: int = LOG_LEVELS["INFO"]


def log(level: str, message: str) -> None:
    """Prints a message if its level is >= CURRENT_LOG_LEVEL."""
    if LOG_LEVELS.get(level.upper(), 99) >= CURRENT_LOG_LEVEL:
        # Print errors to stderr, others to stdout
        stream = sys.stderr if level.upper() == "ERROR" else sys.stdout
        print(f"[{level.upper():<5}] {message}", file=stream, flush=True)


def set_log_level(level: str) -> None:
    """Sets the global log level."""
    global CURRENT_LOG_LEVEL
    new_level_int = LOG_LEVELS.get(level.upper())
    if new_level_int is not None:
        CURRENT_LOG_LEVEL = new_level_int
        log("INFO", f"Log level set to {level.upper()}")
    else:
        log("WARN", f"Invalid log level '{level}'. Keeping current level.")
