# -*- coding: utf-8 -*-
"""Audio processing functions, including Spectrogram transforms."""

from torch import nn
from torchaudio import transforms as T

from .config import FMAX, FMIN, HOP_LENGTH, N_FFT, N_MELS, POWER, TARGET_SAMPLE_RATE
from .logging import log


def create_spectrogram_transforms() -> nn.Sequential:
    """Creates the Mel Spectrogram and dB conversion transforms."""
    log("INFO", "🔧 Creating Mel Spectrogram transforms...")
    log(
        "INFO",
        f" Params: SR={TARGET_SAMPLE_RATE}, N_FFT={N_FFT}, HOP={HOP_LENGTH}, N_MELS={N_MELS}, F_MAX={FMAX}",
    )

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=FMIN,
        f_max=FMAX,
        power=POWER,
        # Add other parameters like win_length, window_fn if needed
    )

    amplitude_to_db = T.AmplitudeToDB(
        stype="power", top_db=80,
    )  # Convert power spec to dB

    # Normalize the spectrogram
    # InstanceNorm2d normalizes each spectrogram (over H, W dimensions) independently within a batch.
    # It learns affine parameters (gamma, beta) by default.
    # Input needs to be (N, C, H, W), hence the Lambda layer.
    normalize = nn.InstanceNorm2d(1, affine=True)  # Learnable affine params often help

    # Using Sequential for clarity
    return nn.Sequential(
        mel_spectrogram,
        amplitude_to_db,
        # Add channel dimension C for Conv2D layers and InstanceNorm
        # Input shape (..., n_mels, time) -> (..., 1, n_mels, time)
        nn.Lambda(
            lambda x: x.unsqueeze(-3),
        ),  # Use -3 for robustness to batch dim presence
        normalize,
    )
