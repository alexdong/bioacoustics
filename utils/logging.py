# -*- coding: utf-8 -*-
"""
DEPRECATED: This module conflicts with Python's standard library logging module.
Please use utils.log_utils instead.
"""

import sys
import warnings
from typing import Dict

# Show deprecation warning
warnings.warn(
    "The utils.logging module conflicts with Python's standard library. "
    "Please use utils.log_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from log_utils for backward compatibility
from utils.log_utils import log, set_log_level, LOG_LEVELS, CURRENT_LOG_LEVEL
