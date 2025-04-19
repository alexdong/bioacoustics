# config.py

from pathlib import Path

import torch

# --- Paths ---
# Define paths with consistent structure
PROJECT_ROOT: Path = Path(
    __file__,
).parent.parent.resolve()  # Get project root directory
DATA_DIR: Path = PROJECT_ROOT / "data"  # Data directory in project root

# Audio and metadata paths
BASE_DATA_DIR: Path = DATA_DIR / "birdclef-2025"  # Customizable dataset folder name
AUDIO_DIR: Path = BASE_DATA_DIR / "train_audio"
TRAIN_METADATA_PATH: Path = BASE_DATA_DIR / "train_metadata.csv"
TAXONOMY_PATH: Path = BASE_DATA_DIR / "taxonomy.csv"

# SSL encoder weights from ssl directory output
SSL_DIR: Path = PROJECT_ROOT / "ssl"
ENCODER_CHECKPOINT_PATH: Path | None = SSL_DIR / "output_ssl" / "ssl_encoder_final.pt"
# Set to None to use base Whisper instead of SSL pre-trained encoder

# Output directories
OUTPUT_DIR: Path = PROJECT_ROOT / "output" / "fine-tune"
CHECKPOINT_DIR: Path = OUTPUT_DIR / "checkpoints"
LOG_DIR: Path = OUTPUT_DIR / "logs"
TEMP_DIR: Path = Path(
    "/tmp/birdclef_finetune",
)  # For intermediate files (not cleaned up)

# --- Audio Processing ---
TARGET_SAMPLE_RATE: int = (
    32000  # Whisper models often use 16kHz or 32kHz, check your base
)
CHUNK_DURATION_SEC: float = 5.0  # As per competition requirement
N_MELS: int = 128  # Typical for Whisper-like models, adjust if needed
HOP_LENGTH: int = 160  # Corresponds roughly to 10ms frame shift at 16kHz, adjust for SR

# --- Model ---
# NOTE: Specify the base Whisper model size used for your encoder
# Options: "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"
WHISPER_MODEL_SIZE: str = "large-v3"  # TODO: Update if different!
NUM_CLASSES: int = 182  # TODO: Update with actual number of species in Xeno-Canto data
POOLING_TYPE: str = "mean"  # Options: "mean", "max"

# --- Training ---
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE: int = 32  # Adjust based on GPU memory
NUM_WORKERS: int = 4  # For DataLoader
LEARNING_RATE_ENCODER: float = 1e-5  # Lower LR for pre-trained part
LEARNING_RATE_HEAD: float = 1e-4  # Higher LR for new classifier head
WEIGHT_DECAY: float = 0.01
NUM_EPOCHS: int = 10  # Adjust as needed
ACCUMULATE_GRAD_BATCHES: int = 1  # Increase if BATCH_SIZE is too small for memory
VALIDATION_INTERVAL_STEPS: int = 500  # How often to run validation within an epoch

# --- Loss Function ---
# Weighting factor for the hierarchical distance penalty
HIERARCHICAL_LOSS_ALPHA: float = 0.25  # Tune this based on validation

# --- Miscellaneous ---
RANDOM_SEED: int = 42

# --- Create output directories ---
# Fail early if paths can't be created (though mkdir parents=True is forgiving)
print("[CONFIG] Ensuring output directories exist...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
print(f"[CONFIG] Using Temporary Directory: {TEMP_DIR.resolve()}")
