# -*- coding: utf-8 -*-
"""Utility functions for the training loop."""

import time
from typing import Any, Dict, Optional, Tuple

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric  # Use Metric base class for type hint

from .logging import log


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int = 50,  # Log progress every N batches
    grad_clip_value: Optional[float] = None,  # Optional gradient clipping
) -> float:
    """Runs one epoch of training."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    start_time = time.monotonic()
    last_log_time = start_time

    log("INFO", f"--- Epoch {epoch+1} Training ---")
    for batch_idx, batch in enumerate(dataloader):
        # Assuming batch is a tuple (data, target)
        # More robust: check type or expect a dict? Let's assume tuple for now.
        assert (
            isinstance(batch, (tuple, list)) and len(batch) == 2
        ), f"💥 Dataloader expected to yield (data, target) tuples, got {type(batch)}"
        data, target = batch
        data, target = data.to(device, non_blocking=True), target.to(
            device, non_blocking=True,
        )

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Basic checks
        assert not torch.isnan(
            loss,
        ), f"💥 NaN loss encountered at epoch {epoch+1}, batch {batch_idx}! Check inputs/gradients."
        assert not torch.isinf(
            loss,
        ), f"💥 Inf loss encountered at epoch {epoch+1}, batch {batch_idx}! Check inputs/gradients."

        loss.backward()

        # Gradient Clipping
        if grad_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)

        optimizer.step()

        total_loss += loss.item()

        # Logging
        current_time = time.monotonic()
        if batch_idx % log_interval == 0 or batch_idx == num_batches - 1:
            elapsed_since_last_log = current_time - last_log_time
            batches_since_last_log = (
                batch_idx - (batch_idx // log_interval * log_interval) + 1
                if batch_idx % log_interval == 0
                else log_interval
            )
            samples_processed = (batch_idx + 1) * data.size(0)  # Use actual batch size
            total_samples = len(dataloader.dataset)  # type: ignore # Assume dataset has __len__
            percent_complete = samples_processed / total_samples * 100

            throughput = (
                (batches_since_last_log * data.size(0)) / elapsed_since_last_log
                if elapsed_since_last_log > 0
                else 0
            )

            log(
                "DEBUG",
                f" Epoch {epoch+1} | Batch {batch_idx+1:>4}/{num_batches} ({percent_complete:>3.0f}%) | Loss: {loss.item():.4f} | Throughput: {throughput:>5.0f} samples/sec",
            )
            last_log_time = current_time

    avg_loss = total_loss / num_batches
    epoch_duration = time.monotonic() - start_time
    log(
        "INFO",
        f"--- Epoch {epoch+1} Training Complete --- Avg Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f}s",
    )
    return avg_loss


@torch.no_grad()  # Disable gradient calculations for validation
def validate(
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: nn.Module,
    metric: Metric,  # Expecting a torchmetrics object (e.g., AUROC)
    device: torch.device,
    epoch: int,  # For logging purposes
) -> Tuple[float, float]:
    """Runs validation for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    metric.reset()  # Reset metric state for the epoch
    start_time = time.monotonic()

    log("INFO", f"--- Epoch {epoch+1} Validation ---")
    for batch_idx, batch in enumerate(dataloader):
        assert (
            isinstance(batch, (tuple, list)) and len(batch) == 2
        ), f"💥 Dataloader expected to yield (data, target) tuples, got {type(batch)}"
        data, target = batch
        data, target = data.to(device, non_blocking=True), target.to(
            device, non_blocking=True,
        )

        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()

        # Update metric state
        # Assuming metric expects raw logits or probabilities based on its internal state
        # E.g., AUROC often needs probabilities (post-softmax)
        # Let's assume probabilities are needed here. If logits, remove softmax.
        probabilities = torch.softmax(output, dim=1)
        try:
            metric.update(probabilities, target)
            log(
                "DEBUG",
                f" Validation Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f}",
            )
        except Exception as e:
            log(
                "ERROR",
                f"💥 Metric update failed at epoch {epoch+1}, batch {batch_idx}: {e}",
            )
            # Decide: continue without this batch's metric or raise? Let's continue but log error.

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Compute final metric for the epoch
    epoch_metric_value = 0.0
    try:
        epoch_metric_value = metric.compute().item()  # .item() gets the scalar value
        log("DEBUG", f"Metric state computed successfully for epoch {epoch+1}")
    except ValueError as e:
        log(
            "WARN",
            f"⚠️ Could not compute metric for epoch {epoch+1}. Maybe missing classes or other issue? Error: {e}",
        )
        # epoch_metric_value remains 0.0
    except Exception as e:
        log("ERROR", f"💥 Unexpected error computing metric for epoch {epoch+1}: {e}")
        # epoch_metric_value remains 0.0

    epoch_duration = time.monotonic() - start_time
    metric_name = metric.__class__.__name__  # Get metric name dynamically
    log(
        "INFO",
        f"--- Epoch {epoch+1} Validation Complete --- Avg Loss: {avg_loss:.4f} | {metric_name}: {epoch_metric_value:.4f}, Duration: {epoch_duration:.2f}s",
    )

    return avg_loss, epoch_metric_value


def setup_wandb(
    config: Dict[str, Any], project_name: str, experiment_name: str,
) -> None:
    """Initializes Weights & Biases."""
    log("INFO", "Initializing Weights & Biases... ✨")
    try:
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
        )
        log("INFO", "✅ wandb initialized.")
    except ImportError:
        log("ERROR", "💥 wandb import failed, but was required. Please install wandb.")
        raise  # Re-raise because it's mandatory now
    except Exception as e:
        log("ERROR", f"💥 Failed to initialize wandb: {e}")
        raise  # Re-raise other init errors


def log_wandb_metrics(
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_metric: float,
    metric_name: str,
    learning_rate: float,
    epoch_duration: float,
) -> None:
    """Logs metrics to Weights & Biases."""
    try:
        wandb.log(
            {
                "epoch": epoch + 1,  # Log 1-based epoch
                "train_loss": train_loss,
                "val_loss": val_loss,
                f"val_{metric_name.lower()}": val_metric,  # Use metric name in key
                "learning_rate": learning_rate,
                "epoch_duration_sec": epoch_duration,
            },
        )
    except Exception as e:
        log("ERROR", f"💥 Failed to log metrics to wandb: {e}")
