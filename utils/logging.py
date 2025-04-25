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

# Use relative import for when the module is imported as part of the utils package
try:
    from .log_utils import log, set_log_level, LOG_LEVELS, CURRENT_LOG_LEVEL
except ImportError:
    # Fallback for when the script is run directly
    import os
    import sys
    
    # Add the parent directory to sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.log_utils import log, set_log_level, LOG_LEVELS, CURRENT_LOG_LEVEL
