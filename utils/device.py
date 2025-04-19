# -*- coding: utf-8 -*-
"""Device selection utility."""

import torch

from .logging import log  # Use relative import within the package/utils


def select_device() -> torch.device:
    """Selects the best available device (MPS, CUDA, CPU)."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Check if MPS is available and built
        # Note: is_available() can be True even if not usable. is_built() is a better check.
        log("INFO", "🚀 Using MPS device (Mac Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        log("INFO", "🚀 Using CUDA device (NVIDIA GPU)")
        # Optionally add logic here to select a specific CUDA device
        # e.g., cuda_device = 0; torch.cuda.set_device(cuda_device)
        return torch.device("cuda")
    else:
        log("WARN", "🐢 No MPS or CUDA detected. Falling back to CPU.")
        return torch.device("cpu")
