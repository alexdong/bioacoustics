# -*- coding: utf-8 -*-
"""Shared configuration constants."""

from pathlib import Path

# Project Wide
PROJECT_NAME: str = "Bioacoustic_Species_Classification_Arch_Comparison"

# Data Locations & Params (Modify if locations change)
DATASET_BASE_DIR: Path = Path("./datasets/xeno-canto-brazil-small")
AUDIO_BASE_DIR: Path = DATASET_BASE_DIR  # Updated requirement
CLASS_MAPPING_FILE: Path = DATASET_BASE_DIR / "class_map.json"
NUM_CLASSES: int = 100  # From project description

# Shared Output / Temp Dirs
OUTPUT_BASE_DIR: Path = Path("./output/arch")
TEMP_BASE_DIR: Path = Path("/tmp") / PROJECT_NAME

# Audio & Spectrogram Parameters (Ensure consistency across experiments)
TARGET_SAMPLE_RATE: int = 32000
SEGMENT_DURATION_S: float = 5.0
SEGMENT_SAMPLES: int = int(TARGET_SAMPLE_RATE * SEGMENT_DURATION_S)
# Mel Spectrogram Params (Set B - High Freq Detail, Max Mels)
N_FFT: int = 2048
HOP_LENGTH: int = 256
N_MELS: int = 256
FMIN: int = 0
FMAX: int = TARGET_SAMPLE_RATE // 2
POWER: float = 2.0
# Normalization stats (Consider calculating these properly from dataset)
# NORMALIZATION_MEAN = [-40.0] # Using InstanceNorm instead now
# NORMALIZATION_STD = [20.0]

# Common Training Defaults (Can be overridden in specific train scripts)
DEFAULT_LEARNING_RATE: float = 3e-4
DEFAULT_WEIGHT_DECAY: float = 1e-2
DEFAULT_NUM_EPOCHS: int = 100
DEFAULT_EARLY_STOPPING_PATIENCE: int = 15
DEFAULT_LR_SCHEDULER_T_MAX_FACTOR: int = 1  # T_max = num_epochs * factor
DEFAULT_LR_WARMUP_EPOCHS: int = 5
DEFAULT_OPTIMIZER_CLS: type = torch.optim.AdamW  # type: ignore # AdamW is a class

# Augmentation Params (Can be overridden)
DEFAULT_AUG_GAUSSIAN_NOISE_P: float = 0.3
DEFAULT_AUG_TIME_STRETCH_P: float = 0.3
DEFAULT_AUG_PITCH_SHIFT_P: float = 0.3

# Make sure base output/temp dirs exist
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
TEMP_BASE_DIR.mkdir(parents=True, exist_ok=True)
