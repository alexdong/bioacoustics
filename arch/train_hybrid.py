# -*- coding: utf-8 -*-
"""
Trains a Hybrid CNN-Transformer model for bioacoustic species classification.
Reuses utility functions.
"""

import math
import sys
import time
from typing import Any, Dict, Optional, Tuple, TypeAlias

import torch
import wandb
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAUROC

from utils.audio_processing import create_spectrogram_transforms
from utils.augmentation import create_augmentation_pipeline
from utils.config import (
    CLASS_MAPPING_FILE,
    DATASET_BASE_DIR,
    DEFAULT_AUG_GAUSSIAN_NOISE_P,
    DEFAULT_AUG_PITCH_SHIFT_P,
    DEFAULT_AUG_TIME_STRETCH_P,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_LR_WARMUP_EPOCHS,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_OPTIMIZER_CLS,
    DEFAULT_WEIGHT_DECAY,
    HOP_LENGTH,
    N_MELS,
    NUM_CLASSES,
    OUTPUT_BASE_DIR,
    PROJECT_NAME,
    SEGMENT_DURATION_S,
    SEGMENT_SAMPLES,
    TARGET_SAMPLE_RATE,
    TEMP_BASE_DIR,
)
from utils.data_utils import get_dataloaders, load_class_mapping
from utils.device import select_device

# --- Project Utilities ---
from utils.logging import log, set_log_level
from utils.model_utils import count_parameters, load_checkpoint, save_checkpoint
from utils.training_utils import (
    log_wandb_metrics,
    setup_wandb,
    train_one_epoch,
    validate,
)

# Define type alias
Metric: TypeAlias = float  # Or MulticlassAUROC

# --- Experiment Specific Configuration ---
EXPERIMENT_NAME = (
    "Hybrid_CNN_Transformer_Tiny"  # Example: Adjust size (Tiny, Small, Base)
)
MODEL_ARCH = "Hybrid-CNN-Transformer"

# Training Hyperparameters (Can override defaults from config)
BATCH_SIZE = 32  # Adjust based on memory usage
LEARNING_RATE = 1e-4  # Transformers often benefit from smaller LR
WEIGHT_DECAY = DEFAULT_WEIGHT_DECAY
NUM_EPOCHS = DEFAULT_NUM_EPOCHS
EARLY_STOPPING_PATIENCE = DEFAULT_EARLY_STOPPING_PATIENCE
LR_WARMUP_EPOCHS = DEFAULT_LR_WARMUP_EPOCHS
OPTIMIZER_CLS = DEFAULT_OPTIMIZER_CLS
GRAD_CLIP_VALUE: Optional[float] = 1.0  # Max grad norm, often useful for transformers

# Augmentation Parameters
AUG_GAUSSIAN_NOISE_P = DEFAULT_AUG_GAUSSIAN_NOISE_P
AUG_TIME_STRETCH_P = DEFAULT_AUG_TIME_STRETCH_P
AUG_PITCH_SHIFT_P = DEFAULT_AUG_PITCH_SHIFT_P

# --- Hybrid Model Specific Hyperparameters ---
# Target embedding size matching EfficientNet-B1 classifier input
TARGET_EMBED_DIM = 1280  # As per spec for fair comparison head input

# CNN Stem Config (Example: Light stem)
# Output channels of last CNN layer before flatten/project
CNN_STEM_OUT_CHANNELS = 256  # Intermediate size
# Output feature map height/width reduction factor (depends on pooling layers)
# Example: 2 MaxPool layers with kernel 2 -> 4x reduction each dim
# Calculate expected H, W after stem to determine input sequence length for Transformer
# Input: (B, 1, N_MELS, Time) = (B, 1, 256, ~250) # Time = ceil(SEGMENT_SAMPLES / HOP_LENGTH)
# After Stem (example): (B, CNN_STEM_OUT_CHANNELS, N_MELS/4, Time/4) = (B, 256, 64, ~62)
# Flattened: (B, CNN_STEM_OUT_CHANNELS * N_MELS/4 * Time/4) OR (B, SeqLen, Features)
# We want (B, SeqLen, D_Model) -> Flatten H, W into SeqLen? Or just Time?
# Let's flatten time dimension to be sequence length for Transformer: (B, Time/4, CNN_STEM_OUT_CHANNELS * N_MELS/4)
# Or flatten spatial dims (H*W) and project? (B, Features, Time) -> (B, D_Model, Time) -> (B, Time, D_Model)
# Common approach: Flatten spatial dims (H,W) -> (B, C, H*W) -> permute -> (B, H*W, C) -> Linear Project -> (B, H*W, D_Model)
# Let's try flattening H*W after the CNN stem.
# Expected output dims after CNN stem need calculation based on actual stem layers

# Transformer Config (Example: Tiny/Small size)
# D_MODEL must match the output projection from CNN stem
D_MODEL = 512  # Transformer internal dimension (can differ from final TARGET_EMBED_DIM)
NHEAD = 8  # Number of attention heads (must divide D_MODEL)
NUM_ENCODER_LAYERS = 4  # Number of transformer blocks
DIM_FEEDFORWARD = 1024  # Hidden dim in FFN (often 2*D_MODEL or 4*D_MODEL)
TRANSFORMER_DROPOUT = 0.1  # Dropout within transformer layers

# --- Derived Paths ---
OUTPUT_DIR = OUTPUT_BASE_DIR / EXPERIMENT_NAME
TEMP_DIR = TEMP_BASE_DIR / EXPERIMENT_NAME
BEST_MODEL_PATH = OUTPUT_DIR / f"{EXPERIMENT_NAME}_best_model.pth"

# Make sure experiment-specific dirs exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# --- Model Definition ---


class HybridCNNTransformer(nn.Module):
    """Hybrid CNN stem + Transformer Encoder model."""

    def __init__(
        self,
        num_classes: int,
        n_mels: int = N_MELS,
        time_steps: int = math.ceil(
            SEGMENT_SAMPLES / HOP_LENGTH,
        ),  # Approx input time dim
        cnn_out_channels: int = CNN_STEM_OUT_CHANNELS,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_encoder_layers: int = NUM_ENCODER_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        transformer_dropout: float = TRANSFORMER_DROPOUT,
        target_embed_dim: int = TARGET_EMBED_DIM,
    ) -> None:
        super().__init__()
        log("INFO", "🏗️ Creating Hybrid CNN-Transformer model...")
        log(
            "INFO",
            f" Params: CNN_out={cnn_out_channels}, D_Model={d_model}, Heads={nhead}, Layers={num_encoder_layers}",
        )
        self.num_classes = num_classes
        self.d_model = d_model

        # --- Simple CNN Stem Example ---
        # Input: (B, 1, n_mels, time_steps) = (B, 1, 256, ~250)
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2, W/2 -> 128, ~125
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4, W/4 -> 64, ~62
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Optional: Add another MaxPool if needed, depends on desired seq length / features
            # nn.MaxPool2d(kernel_size=2, stride=2), # H/8, W/8 -> 32, ~31
            nn.Conv2d(128, cnn_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(inplace=True),
            # Output: (B, cnn_out_channels, n_mels/4, time_steps/4) = (B, 256, 64, ~62)
        )
        self.cnn_output_shape: Optional[Tuple[int, int, int]] = (
            None  # To store C, H, W after stem
        )

        # Calculate CNN output spatial dimensions dynamically (run dummy input once)
        self._determine_cnn_output_shape(n_mels, time_steps)
        assert self.cnn_output_shape is not None
        cnn_feat_h, cnn_feat_w = self.cnn_output_shape[1], self.cnn_output_shape[2]
        cnn_flattened_features = cnn_out_channels * cnn_feat_h  # Features per time step

        # --- Flattening & Projection ---
        # Flatten H dimension, keep W (time) as sequence length
        # Input: (B, C, H, W) -> Reshape -> (B, W, C*H) = (B, SeqLen, Features)
        self.flatten = nn.Flatten(
            start_dim=1, end_dim=2,
        )  # Flatten C and H together -> (B, C*H, W)
        self.permute = lambda x: x.permute(
            0, 2, 1,
        )  # (B, C*H, W) -> (B, W, C*H) = (B, SeqLen, Features)

        self.projection = nn.Linear(cnn_flattened_features, d_model)
        # Output: (B, SeqLen, d_model) = (B, time_steps/4, d_model)

        # --- Positional Encoding ---
        # Simple learned positional encoding
        self.seq_len = cnn_feat_w
        self.pos_encoder = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.1)
        log("INFO", f" Learned Positional Encoding for SeqLen={self.seq_len}")

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation=F.relu,
            batch_first=True,  # IMPORTANT: Input/Output as (Batch, Seq, Feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),  # Optional final norm
        )
        # Output: (B, SeqLen, d_model)

        # --- Classification Head (Matching EfficientNet B1 target) ---
        # Global Average Pooling over the sequence dimension
        self.gap = nn.AdaptiveAvgPool1d(
            1,
        )  # Input (B, SeqLen, d_model) -> (B, d_model, SeqLen) needed for pool
        # Output: (B, d_model, 1) -> Squeeze -> (B, d_model)

        # Final linear layer to project to target embedding size (if different from d_model)
        # And then to number of classes
        self.fc1 = nn.Linear(d_model, target_embed_dim)
        self.relu = nn.ReLU(inplace=True)  # Optional activation
        self.fc2 = nn.Linear(target_embed_dim, num_classes)
        log(
            "INFO",
            f"✅ Classification Head: GAP -> Linear({d_model}->{target_embed_dim}) -> ReLU -> Linear({target_embed_dim}->{num_classes})",
        )

    def _determine_cnn_output_shape(self, n_mels: int, time_steps: int) -> None:
        """Runs a dummy input through CNN stem to find output shape."""
        log("DEBUG", "Determining CNN stem output shape...")
        dummy_input = torch.randn(1, 1, n_mels, time_steps)
        with torch.no_grad():
            dummy_output = self.cnn_stem(dummy_input)
        # Shape: (1, C, H, W)
        self.cnn_output_shape = dummy_output.shape[1:]  # Store (C, H, W)
        log(
            "INFO",
            f" CNN Stem output feature map shape (C, H, W): {self.cnn_output_shape}",
        )
        assert (
            len(self.cnn_output_shape) == 3
        ), "💥 Unexpected CNN output shape dimension"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (B, 1, n_mels, time_steps) from dataloader/transforms
        assert x.dim() == 4 and x.size(1) == 1, f"💥 Unexpected input shape: {x.shape}"

        # 1. CNN Stem
        x = self.cnn_stem(x)  # (B, C, H, W)
        # Shape check after stem
        assert (
            x.shape[1:] == self.cnn_output_shape
        ), f"💥 Unexpected shape after CNN stem: {x.shape}"

        # 2. Flatten & Permute for Transformer
        # Flatten C and H dims, keep W (time) as sequence dim
        # (B, C, H, W) -> (B, C*H, W) -> (B, W, C*H)
        b, c, h, w = x.shape
        x = x.view(b, c * h, w)  # -> (B, C*H, W)
        x = x.permute(0, 2, 1)  # -> (B, W, C*H) = (B, SeqLen, Features)
        assert x.shape == (b, w, c * h), "💥 Shape mismatch after flatten/permute"

        # 3. Project to d_model
        x = self.projection(x)  # (B, SeqLen, d_model)
        assert x.shape == (b, w, self.d_model), "💥 Shape mismatch after projection"

        # 4. Add Positional Encoding
        # Ensure pos_encoder matches batch size (broadcasting) and seq len
        assert self.pos_encoder.shape == (
            1,
            self.seq_len,
            self.d_model,
        ), "💥 Positional encoder shape mismatch"
        x = x + self.pos_encoder  # Broadcasting adds pos encoding to each item in batch

        # 5. Transformer Encoder
        x = self.transformer_encoder(x)  # (B, SeqLen, d_model)
        assert x.shape == (b, w, self.d_model), "💥 Shape mismatch after transformer"

        # 6. Classification Head
        # Global Average Pooling requires (B, C, L) = (B, d_model, SeqLen)
        x = x.permute(0, 2, 1)  # (B, d_model, SeqLen)
        x = self.gap(x)  # (B, d_model, 1)
        x = x.squeeze(-1)  # (B, d_model)

        # Final Linear layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # (B, num_classes)
        assert x.shape == (
            b,
            self.num_classes,
        ), "💥 Shape mismatch after final classifier"

        return x


def get_model(num_classes: int) -> nn.Module:
    """Creates the specific model for this experiment."""
    model = HybridCNNTransformer(
        num_classes=num_classes,
        # Pass other specific hyperparameters if needed
        cnn_out_channels=CNN_STEM_OUT_CHANNELS,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        transformer_dropout=TRANSFORMER_DROPOUT,
        target_embed_dim=TARGET_EMBED_DIM,
    )
    return model


# --- Helper Functions for Orchestration (Reused Structure) ---
# These functions (setup_environment, prepare_data, setup_training_components)
# are largely identical to the EfficientNet script, just parameter changes.


def setup_environment(
    wandb_config: Dict[str, Any],
) -> Tuple[torch.device, Dict[str, int]]:
    """Sets up logging, device, wandb, and loads class map."""
    set_log_level("INFO")
    log("INFO", f"🐍 Starting Training Script - {EXPERIMENT_NAME}")
    log("INFO", f"PyTorch Version: {torch.__version__}")

    device = select_device()
    class_map = load_class_mapping(CLASS_MAPPING_FILE)
    num_classes_actual = len(class_map)
    assert (
        num_classes_actual == NUM_CLASSES
    ), f"💥 Mismatch: NUM_CLASSES constant is {NUM_CLASSES}, loaded map has {num_classes_actual}"

    # Initialize WandB
    setup_wandb(wandb_config, PROJECT_NAME, EXPERIMENT_NAME)

    return device, class_map


def prepare_data(
    class_map: Dict[str, int], spectrogram_transform: nn.Module, device: torch.device,
) -> Dict[str, torch.utils.data.DataLoader[Any]]:
    """Prepares datasets and dataloaders."""
    log("INFO", "Preparing datasets and dataloaders...")
    augment_pipeline = create_augmentation_pipeline(
        p_gaussian_noise=AUG_GAUSSIAN_NOISE_P,
        p_time_stretch=AUG_TIME_STRETCH_P,
        p_pitch_shift=AUG_PITCH_SHIFT_P,
        sample_rate=TARGET_SAMPLE_RATE,
    )

    dataloaders = get_dataloaders(
        class_map=class_map,
        spectrogram_transform=spectrogram_transform,
        augment_pipeline=augment_pipeline,
        batch_size=BATCH_SIZE,
        pin_memory=device.type == "cuda",
    )
    return dataloaders


def setup_training_components(
    model: nn.Module, num_epochs: int, steps_per_epoch: int,
) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, Metric, str]:  # type: ignore
    """Sets up optimizer, loss, scheduler, and metric."""
    log("INFO", "Setting up optimizer, loss, scheduler, metrics...")
    optimizer = OPTIMIZER_CLS(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    num_training_steps = steps_per_epoch * (num_epochs - LR_WARMUP_EPOCHS)
    log(
        "INFO",
        f"Scheduler: CosineAnnealingLR with T_max = {num_training_steps} steps after warmup",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_training_steps,
    )

    metric = MulticlassAUROC(num_classes=NUM_CLASSES, average="macro", thresholds=None)
    metric_name = "Macro_ROC_AUC"
    log("INFO", f"Primary validation metric: {metric_name}")

    return criterion, optimizer, scheduler, metric, metric_name


# --- Main Training Orchestration Function (Top Level - Reused Structure) ---


def train_model() -> None:
    """Main function to set up and run the training process."""
    script_start_time = time.monotonic()

    # --- Basic Setup ---
    wandb_config = {
        "architecture": MODEL_ARCH,
        "dataset": DATASET_BASE_DIR.name,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": OPTIMIZER_CLS.__name__,
        "lr_scheduler": "CosineAnnealingLR",
        "warmup_epochs": LR_WARMUP_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "grad_clip_value": GRAD_CLIP_VALUE,
        "segment_duration_s": SEGMENT_DURATION_S,
        "sample_rate": TARGET_SAMPLE_RATE,
        # Hybrid Specific Params
        "cnn_stem_out_channels": CNN_STEM_OUT_CHANNELS,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_encoder_layers": NUM_ENCODER_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "transformer_dropout": TRANSFORMER_DROPOUT,
        "target_embed_dim": TARGET_EMBED_DIM,
        # Augmentation Params
        "augmentation_gaussian_p": AUG_GAUSSIAN_NOISE_P,
        "augmentation_stretch_p": AUG_TIME_STRETCH_P,
        "augmentation_pitch_p": AUG_PITCH_SHIFT_P,
    }
    device, class_map = setup_environment(wandb_config)

    # --- Data Preparation ---
    spectrogram_transforms = create_spectrogram_transforms()
    dataloaders = prepare_data(class_map, spectrogram_transforms, device)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # --- Model & Training Setup ---
    model = get_model(num_classes=NUM_CLASSES).to(device)
    param_count = count_parameters(model)
    log("INFO", f" Model parameters: {param_count:,}")
    wandb.config.update({"parameters": param_count})
    wandb.watch(model, log_freq=100)

    criterion, optimizer, scheduler, val_metric, metric_name = (
        setup_training_components(model, NUM_EPOCHS, len(train_loader))
    )
    val_metric = val_metric.to(device)

    # --- Training Loop ---
    log("INFO", f"🏁 Starting training for max {NUM_EPOCHS} epochs...")
    best_val_metric = 0.0
    epochs_without_improvement = 0
    start_epoch = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start_time = time.monotonic()

        # Manual LR Warmup
        current_lr = LEARNING_RATE  # Default LR
        if epoch < LR_WARMUP_EPOCHS:
            warmup_factor = (epoch + 1) / LR_WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group["lr"] = LEARNING_RATE * warmup_factor
            current_lr = LEARNING_RATE * warmup_factor
            log(
                "DEBUG",
                f"Warmup Epoch {epoch+1}/{LR_WARMUP_EPOCHS}, LR set to {current_lr:.6f}",
            )
        elif epoch == LR_WARMUP_EPOCHS:
            log(
                "INFO",
                f"Warmup complete after epoch {epoch}. Starting Cosine Annealing schedule.",
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = LEARNING_RATE
            current_lr = LEARNING_RATE
        else:
            current_lr = scheduler.get_last_lr()[0]

        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            grad_clip_value=GRAD_CLIP_VALUE,
        )

        # Validate
        val_loss, val_metric_value = validate(
            model, val_loader, criterion, val_metric, device, epoch,
        )

        # Step scheduler after validation (if applicable)
        if epoch >= LR_WARMUP_EPOCHS:
            scheduler.step()

        epoch_duration = time.monotonic() - epoch_start_time
        log(
            "INFO",
            f"--- Epoch {epoch+1}/{NUM_EPOCHS} Summary --- Duration: {epoch_duration:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val {metric_name}: {val_metric_value:.4f} | LR: {current_lr:.6f}",
        )

        log_wandb_metrics(
            epoch,
            train_loss,
            val_loss,
            val_metric_value,
            metric_name,
            current_lr,
            epoch_duration,
        )

        # --- Early Stopping & Checkpointing ---
        if val_metric_value > best_val_metric:
            log(
                "INFO",
                f"🎉 Hooray! Validation {metric_name} improved from {best_val_metric:.4f} to {val_metric_value:.4f}.",
            )
            best_val_metric = val_metric_value
            epochs_without_improvement = 0
            save_checkpoint(
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_metric,
                class_map,
                BEST_MODEL_PATH,
            )
            wandb.summary[f"best_val_{metric_name.lower()}"] = best_val_metric
            wandb.summary["best_epoch"] = epoch + 1
        else:
            epochs_without_improvement += 1
            log(
                "INFO",
                f"😕 No improvement in validation {metric_name} for {epochs_without_improvement} epoch(s). Best: {best_val_metric:.4f}",
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            log(
                "WARN",
                f"🛑 Patience [{EARLY_STOPPING_PATIENCE}] exhausted. Stopping early at epoch {epoch+1}.",
            )
            break

    # --- Final Evaluation on Test Set ---
    log(
        "INFO",
        "⏳ Training finished or stopped early. Loading best model and evaluating on test set...",
    )
    if BEST_MODEL_PATH.exists():
        _, best_metric_loaded = load_checkpoint(BEST_MODEL_PATH, model, device=device)
        log(
            "INFO",
            f"✅ Loaded best model from checkpoint (Val {metric_name}: {best_metric_loaded:.4f}) for final test evaluation.",
        )

        test_loss, test_metric = validate(
            model, test_loader, criterion, val_metric, device, epoch=-1,
        )

        log("INFO", f"--- Final Test Results ({EXPERIMENT_NAME}) ---")
        log("INFO", f"📊 Test Loss: {test_loss:.4f}")
        log("INFO", f"📊 Test {metric_name}: {test_metric:.4f}")

        wandb.summary[f"test_{metric_name.lower()}"] = test_metric
        wandb.summary["test_loss"] = test_loss
    else:
        log("ERROR", "💥 No best model checkpoint found. Cannot evaluate on test set.")

    script_duration = time.monotonic() - script_start_time
    log(
        "INFO",
        f"🏁 Training script finished in {script_duration:.2f} seconds ({script_duration/60:.1f} minutes).",
    )
    wandb.finish()
    log("INFO", "✅ wandb run finished.")


# --- Main Execution Block ---

if __name__ == "__main__":
    log("INFO", f"🚀 Script invoked directly. Running {EXPERIMENT_NAME} training.")
    try:
        train_model()
        log(
            "INFO",
            f"✅ {EXPERIMENT_NAME} training demonstration completed successfully.",
        )
        assert (
            BEST_MODEL_PATH.exists()
        ), f"💥 Verification failed: Best model file '{BEST_MODEL_PATH}' was not created."
        log(
            "INFO",
            f"✅ Verification passed: Best model file found at {BEST_MODEL_PATH}",
        )

    except Exception as e:
        log("ERROR", f"💥 Training run failed: {e}")
        import traceback

        log("ERROR", traceback.format_exc())
        if wandb.run is not None:
            wandb.finish(exit_code=1)
        sys.exit(1)
