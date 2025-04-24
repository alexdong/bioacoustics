# -*- coding: utf-8 -*-
"""Utilities related to model definition and handling."""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

from .logging import log


def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],  # type: ignore # Private but common type hint
    best_metric_value: float,
    class_map: dict,
    filepath: Path,
) -> None:
    """Saves model checkpoint."""
    log("INFO", f"💾 Saving checkpoint to {filepath}...")
    try:
        state = {
            "epoch": epoch + 1,  # Save 1-based epoch
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_metric_value": best_metric_value,
            "class_map": class_map,
        }
        torch.save(state, filepath)
        log("INFO", "✅ Checkpoint saved successfully.")
    except Exception as e:
        log("ERROR", f"💥 Failed to save checkpoint to {filepath}: {e}")


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,  # type: ignore
    device: torch.device = torch.device("cpu"),  # Load to specified device
) -> Tuple[int, float]:
    """Loads model checkpoint. Returns starting epoch and best metric value."""
    assert filepath.exists(), f"💥 Checkpoint file not found: {filepath}"
    log("INFO", f"⏳ Loading checkpoint from {filepath}...")
    try:
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        start_epoch = checkpoint.get("epoch", 0)  # Default to 0 if not found
        best_metric_value = checkpoint.get(
            "best_metric_value",
            0.0,
        )  # Default if not found

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            log("INFO", "✅ Optimizer state loaded.")
        elif optimizer:
            log(
                "WARN",
                "⚠️ Checkpoint loaded, but optimizer state not found or not loaded.",
            )

        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            log("INFO", "✅ Scheduler state loaded.")
        elif scheduler:
            log(
                "WARN",
                "⚠️ Checkpoint loaded, but scheduler state not found or not loaded.",
            )

        log(
            "INFO",
            f"✅ Checkpoint loaded. Resuming from epoch {start_epoch}, Best metric so far: {best_metric_value:.4f}",
        )
        return start_epoch, best_metric_value

    except Exception as e:
        log("ERROR", f"💥 Failed to load checkpoint from {filepath}: {e}")
        raise  # Re-raise error if loading fails critically
