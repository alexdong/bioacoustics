# -*- coding: utf-8 -*-
"""
Trains an EfficientNet-B1 model for bioacoustic species classification.
Refactored to use utility functions.
"""

import time
from typing import Any, Dict, Optional, Tuple

import torch
import wandb
from torch import nn
from torchmetrics.classification import MulticlassAUROC
from torchvision.models import efficientnet_b1

from utils.audio_processing import create_spectrogram_transforms

# Assumes utils/augmentation.py exists and has create_augmentation_pipeline
from utils.augmentation import create_augmentation_pipeline
from utils.config import (
    CLASS_MAPPING_FILE,
    DATASET_BASE_DIR,
    DEFAULT_AUG_GAUSSIAN_NOISE_P,
    DEFAULT_AUG_PITCH_SHIFT_P,
    DEFAULT_AUG_TIME_STRETCH_P,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LR_WARMUP_EPOCHS,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_OPTIMIZER_CLS,
    DEFAULT_WEIGHT_DECAY,
    NUM_CLASSES,
    OUTPUT_BASE_DIR,
    PROJECT_NAME,
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

# --- Experiment Specific Configuration ---
EXPERIMENT_NAME = "EfficientNet-B1_Baseline_Refactored"
MODEL_ARCH = "EfficientNet-B1"

# Training Hyperparameters (Can override defaults from config)
BATCH_SIZE = 32
LEARNING_RATE = DEFAULT_LEARNING_RATE
WEIGHT_DECAY = DEFAULT_WEIGHT_DECAY
NUM_EPOCHS = DEFAULT_NUM_EPOCHS
EARLY_STOPPING_PATIENCE = DEFAULT_EARLY_STOPPING_PATIENCE
LR_WARMUP_EPOCHS = DEFAULT_LR_WARMUP_EPOCHS
OPTIMIZER_CLS = DEFAULT_OPTIMIZER_CLS
GRAD_CLIP_VALUE: Optional[float] = 1.0  # Max grad norm

# Augmentation Parameters
AUG_GAUSSIAN_NOISE_P = DEFAULT_AUG_GAUSSIAN_NOISE_P
AUG_TIME_STRETCH_P = DEFAULT_AUG_TIME_STRETCH_P
AUG_PITCH_SHIFT_P = DEFAULT_AUG_PITCH_SHIFT_P

# --- Derived Paths ---
OUTPUT_DIR = OUTPUT_BASE_DIR / EXPERIMENT_NAME
TEMP_DIR = TEMP_BASE_DIR / EXPERIMENT_NAME  # For intermediate files if needed
BEST_MODEL_PATH = OUTPUT_DIR / f"{EXPERIMENT_NAME}_best_model.pth"

# Make sure experiment-specific dirs exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# --- Model Definition ---


def get_model(num_classes: int) -> nn.Module:
    """Creates the specific model for this experiment."""
    log("INFO", f"🏗️ Creating {MODEL_ARCH} model...")
    # Load EfficientNet-B1 without pre-trained weights (as per project plan)
    model = efficientnet_b1(weights=None)  # Train from scratch

    in_features = model.classifier[1].in_features
    log("INFO", f" Original classifier input features: {in_features}")
    assert (
        in_features == 1280
    ), f"💥 Expected EfficientNet-B1 feature size 1280, got {in_features}"

    # Replace the classifier head (matches Perch: GAP -> Linear)
    # EfficientNet already applies GAP before the classifier layer
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),  # Standard dropout for EffNet head
        nn.Linear(in_features, num_classes),
    )
    log("INFO", f"✅ Replaced classifier head. Output features: {num_classes}")
    return model


# --- Helper Functions for Orchestration ---


def setup_environment(
    wandb_config: Dict[str, Any],
) -> Tuple[torch.device, Dict[str, int]]:
    """Sets up logging, device, wandb, and loads class map."""
    set_log_level("INFO")  # Or load from env/args
    log("INFO", f"🐍 Starting Training Script - {EXPERIMENT_NAME}")
    log("INFO", f"PyTorch Version: {torch.__version__}")
    # Add torchaudio, torchvision versions if desired

    device = select_device()
    class_map = load_class_mapping(CLASS_MAPPING_FILE)
    num_classes_actual = len(class_map)
    assert (
        num_classes_actual == NUM_CLASSES
    ), f"💥 Mismatch: NUM_CLASSES constant is {NUM_CLASSES}, loaded map has {num_classes_actual}"

    # Initialize WandB (Mandatory)
    setup_wandb(wandb_config, PROJECT_NAME, EXPERIMENT_NAME)

    return device, class_map


def prepare_data(
    class_map: Dict[str, int],
    spectrogram_transform: nn.Module,
    device: torch.device,  # Needed for pin_memory decision
) -> Dict[str, torch.utils.data.DataLoader[Any]]:
    """Prepares datasets and dataloaders."""
    log("INFO", "Preparing datasets and dataloaders...")
    augment_pipeline = create_augmentation_pipeline(
        # Pass specific params if needed, assuming defaults from config/augmentation.py
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
        pin_memory=device.type == "cuda",  # Enable pin_memory only for CUDA
    )
    return dataloaders


def setup_training_components(
    model: nn.Module,
    num_epochs: int,  # Needed for scheduler T_max
    steps_per_epoch: int,  # Needed for OneCycleLR if used
) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, Metric]:  # type: ignore
    """Sets up optimizer, loss, scheduler, and metric."""
    log("INFO", "Setting up optimizer, loss, scheduler, metrics...")
    optimizer = OPTIMIZER_CLS(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()  # Standard classification loss

    # Scheduler with warmup + cosine decay
    # T_max is the number of steps *after* warmup
    num_training_steps = steps_per_epoch * (num_epochs - LR_WARMUP_EPOCHS)
    log(
        "INFO",
        f"Scheduler: CosineAnnealingLR with T_max = {num_training_steps} steps after warmup",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_training_steps,
    )
    # We will handle warmup manually in the training loop for simplicity here

    # Primary metric: Macro-Averaged ROC-AUC
    metric = MulticlassAUROC(num_classes=NUM_CLASSES, average="macro", thresholds=None)
    metric_name = "Macro_ROC_AUC"  # Used for logging/tracking
    log("INFO", f"Primary validation metric: {metric_name}")

    return criterion, optimizer, scheduler, metric, metric_name


# --- Main Training Orchestration Function (Top Level) ---


def train_model() -> None:
    """
    Main function to set up and run the training process.
    """
    script_start_time = time.monotonic()

    # --- Basic Setup ---
    wandb_config = {  # Collect hyperparams for logging
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
        # Add mel spec params if desired
        "augmentation_gaussian_p": AUG_GAUSSIAN_NOISE_P,
        "augmentation_stretch_p": AUG_TIME_STRETCH_P,
        "augmentation_pitch_p": AUG_PITCH_SHIFT_P,
    }
    device, class_map = setup_environment(wandb_config)

    # --- Data Preparation ---
    # Create transforms once, potentially move to device if they have state
    spectrogram_transforms = (
        create_spectrogram_transforms()
    )  # .to(device) # Keep on CPU for dataloader usually
    dataloaders = prepare_data(class_map, spectrogram_transforms, device)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # --- Model & Training Setup ---
    model = get_model(num_classes=NUM_CLASSES).to(device)
    param_count = count_parameters(model)
    log("INFO", f" Model parameters: {param_count:,}")
    wandb.config.update({"parameters": param_count})  # Add param count to wandb config
    wandb.watch(model, log_freq=100)  # Log gradients and parameters

    criterion, optimizer, scheduler, val_metric, metric_name = (
        setup_training_components(model, NUM_EPOCHS, len(train_loader))
    )
    val_metric = val_metric.to(device)  # Move metric object state to device

    # --- Training Loop ---
    log("INFO", f"🏁 Starting training for max {NUM_EPOCHS} epochs...")
    best_val_metric = 0.0
    epochs_without_improvement = 0
    start_epoch = 0  # Default start

    # Optional: Resume from checkpoint
    # if BEST_MODEL_PATH.exists():
    #     log("INFO", "Attempting to resume from checkpoint...")
    #     start_epoch, best_val_metric = load_checkpoint(
    #         BEST_MODEL_PATH, model, optimizer, scheduler, device
    #     )
    #     log("INFO", f"Resuming from epoch {start_epoch}. Best metric was {best_val_metric:.4f}")
    #     epochs_without_improvement = 0 # Reset patience counter

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start_time = time.monotonic()

        # Manual LR Warmup
        current_lr = LEARNING_RATE  # Default LR for logging if no warmup/schedule step
        if epoch < LR_WARMUP_EPOCHS:
            # Simple linear warmup
            warmup_factor = (epoch + 1) / LR_WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                # Assuming only one param group or all use the same base LR
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
            # Set LR to initial LR for scheduler (just after warmup)
            for param_group in optimizer.param_groups:
                param_group["lr"] = LEARNING_RATE
            current_lr = LEARNING_RATE
            # Scheduler step will happen *after* this epoch's train/val
        else:
            # Scheduler step happens after optimizer.step() in train loop implicitly via validate
            # We step it *before* the next training epoch starts (after validation)
            current_lr = scheduler.get_last_lr()[0]  # Get LR for logging

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

        # Step the scheduler *after* validation and *before* the next training epoch, but only after warmup
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
        # Load best model state
        _, best_metric_loaded = load_checkpoint(BEST_MODEL_PATH, model, device=device)
        log(
            "INFO",
            f"✅ Loaded best model from checkpoint (Val {metric_name}: {best_metric_loaded:.4f}) for final test evaluation.",
        )

        # Create a separate metric instance for test evaluation if needed (or reset existing)
        # Reusing val_metric here, but ensure it's reset if state matters across splits
        test_loss, test_metric = validate(
            model, test_loader, criterion, val_metric, device, epoch=-1,
        )  # Use epoch=-1 for clarity

        log("INFO", f"--- Final Test Results ({EXPERIMENT_NAME}) ---")
        log("INFO", f"📊 Test Loss: {test_loss:.4f}")
        log("INFO", f"📊 Test {metric_name}: {test_metric:.4f}")

        # Log final test results to wandb summary
        wandb.summary[f"test_{metric_name.lower()}"] = test_metric
        wandb.summary["test_loss"] = test_loss
        # Optionally save the best model file artifact to wandb
        # wandb.save(str(BEST_MODEL_PATH))
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
        # Final check
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
        # Ensure wandb is finished even on error if it was initialized
        if wandb.run is not None:
            wandb.finish(exit_code=1)  # Mark run as failed
        sys.exit(1)  # Exit with error code
import sys
from typing import TypeAlias

# Define Metric type alias
Metric: TypeAlias = float  # Or whatever the appropriate type is

# Define SEGMENT_DURATION_S constant
SEGMENT_DURATION_S: float = 5.0  # Or whatever the appropriate value is
