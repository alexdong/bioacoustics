# ssl/config.py

from pathlib import Path

import torch

# --- Paths ---
# Define paths with consistent structure
PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()  # Get project root directory
DATA_DIR: Path = PROJECT_ROOT / "data"  # Data directory in project root

# Path to your large unlabeled dataset (e.g., iNaturalist audio)
UNLABELED_AUDIO_DIR: Path = DATA_DIR / "unlabeled_audio"

# Path to the base Whisper model for starting SSL
# If None, loads directly from Hugging Face. If a path, loads custom weights.
BASE_ENCODER_CHECKPOINT_PATH: Path | None = None

# Output directories
SSL_OUTPUT_DIR: Path = PROJECT_ROOT / "output" / "ssl"
SSL_ENCODER_CHECKPOINT_PATH: Path = SSL_OUTPUT_DIR / "ssl_encoder_final.pt"
SSL_LOG_DIR: Path = SSL_OUTPUT_DIR / "logs"
SSL_TEMP_DIR: Path = Path("/tmp/birdclef_ssl")  # For intermediate files

# --- Audio Processing (Should match fine-tuning config where applicable) ---
TARGET_SAMPLE_RATE: int = 32000 # Whisper models often use 16kHz or 32kHz
CHUNK_DURATION_SEC: float = 5.0 # Use same chunk size as fine-tuning? Or maybe longer? Let's stick to 5s.
N_MELS: int = 128 # Must match the fine-tuning input if using the same base
HOP_LENGTH: int = 160 # Adjust for SR if needed

# --- SSL Model (MAE) ---
WHISPER_MODEL_SIZE: str = "large-v3" # Base model size
MASKING_RATIO: float = 0.75 # Proportion of time frames to mask (high ratio is common for MAE)
# MAE Decoder configuration
DECODER_EMBED_DIM: int = 512 # Dimension for the decoder layers (can be smaller than encoder)
DECODER_DEPTH: int = 4 # Number of decoder layers (much shallower than encoder)
DECODER_NUM_HEADS: int = 8 # Number of attention heads in decoder layers

# --- SSL Training ---
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE: int = 16 # Adjust based on GPU memory (MAE can be memory intensive)
NUM_WORKERS: int = 4
LEARNING_RATE: float = 1e-4 # Single LR for encoder+decoder during SSL
WEIGHT_DECAY: float = 0.05 # Often higher in MAE pre-training
NUM_EPOCHS: int = 50 # SSL often requires many epochs
ACCUMULATE_GRAD_BATCHES: int = 2 # Adjust based on BATCH_SIZE and memory
CHECKPOINT_INTERVAL_STEPS: int = 1000 # How often to save full MAE model checkpoint (optional)

# --- Miscellaneous ---
RANDOM_SEED: int = 42

# --- Create output directories ---
print("[SSL CONFIG] Ensuring output directories exist...")
SSL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SSL_LOG_DIR.mkdir(parents=True, exist_ok=True)
SSL_TEMP_DIR.mkdir(parents=True, exist_ok=True)
print(f"[SSL CONFIG] Using Temporary Directory: {SSL_TEMP_DIR.resolve()}")
