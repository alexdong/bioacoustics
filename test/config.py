# test/config.py

import sys
from pathlib import Path

import torch

# --- Paths ---
# NOTE: Path to the test audio directory provided by the competition
TEST_AUDIO_DIR: Path = Path("./input/birdclef-2025/test_soundscapes")  # TODO: Update!

# NOTE: Path to the test metadata or file listing test audio (if provided)
# If not provided, the script might need to glob the TEST_AUDIO_DIR directly.
# Often competitions provide a 'test.csv' or 'sample_submission.csv' to define rows.
TEST_METADATA_PATH: Path | None = Path(
    "./input/birdclef-2025/sample_submission.csv",
)  # TODO: Update or set to None!

# NOTE: Path to the TRAINED model checkpoint (from fine-tuning)
MODEL_CHECKPOINT_PATH: Path = Path(
    "./output/checkpoints/best_model_step_XYZ.pt",
)  # TODO: Update with actual best checkpoint!

# NOTE: Path for the final submission file
SUBMISSION_OUTPUT_PATH: Path = Path("./submission.csv")

# NOTE: Directory containing fine-tuning config/code (needed for NUM_CLASSES etc.)
# Assumes fine_tune is in the parent directory relative to test/
FINE_TUNE_DIR: Path = Path(__file__).parent.parent / "fine_tune"

# --- Audio Processing (Must match fine-tuning settings) ---
# Import relevant settings from fine-tune config
# This is a bit fragile, assumes relative path works. Consider duplicating essential values.
sys.path.insert(0, str(FINE_TUNE_DIR.resolve()))
try:
    import config as fine_tune_config

    TARGET_SAMPLE_RATE: int = fine_tune_config.TARGET_SAMPLE_RATE
    CHUNK_DURATION_SEC: float = 5.0  # Competition standard: 5 seconds
    N_MELS: int = fine_tune_config.N_MELS
    HOP_LENGTH: int = fine_tune_config.HOP_LENGTH
    # Get NUM_CLASSES indirectly if needed, or define explicitly
    # Need species list for submission header
except ImportError:
    print(
        "[ERROR] Could not import fine_tune/config.py. Define test config values manually.",
    )
    # Define fallback values manually if import fails - Adjust these!
    TARGET_SAMPLE_RATE = 32000
    CHUNK_DURATION_SEC = 5.0
    N_MELS = 128
    HOP_LENGTH = 160
sys.path.pop(0)  # Clean up path modification

# --- Inference ---
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_BATCH_SIZE: int = 64  # Adjust based on GPU/CPU memory for inference
NUM_WORKERS: int = 2  # For test DataLoader

# --- Model Loading Options (controlled via command line args in inference.py) ---
# These are just defaults or placeholders; actual loading depends on args.
DEFAULT_MODEL_FORMAT: str = "pytorch"  # Options: pytorch, int8, torchscript

# Paths for quantized/scripted models (optional, can be generated)
QUANTIZED_MODEL_PATH: Path | None = None  # Path("./output/quantized_model.pt")
TORCHSCRIPT_MODEL_PATH: Path | None = None  # Path("./output/scripted_model.pt")
