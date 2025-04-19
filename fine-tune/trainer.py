# trainer.py
"""Functions for training loop and validation."""

import time
from pathlib import Path
from typing import Optional, Union

import config  # Import config variables
import model  # Import the model module
import numpy as np  # For distance matrix type hint fix
import torch
import torch.nn as nn
import torch.optim as optim
from loss import hierarchical_distance_loss

# Assuming DistanceMatrix is numpy initially, convert to tensor before loss
from taxonomy import DistanceMatrix as NumpyDistanceMatrix
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Example scheduler
from torch.utils.data import DataLoader, Dataset

# Type Alias
Tensor = torch.Tensor
Device = torch.device


# --- Top Level Function ---
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    distance_matrix_np: NumpyDistanceMatrix,
    optimizer: optim.Optimizer,
    scheduler: Optional[
        Union[
            torch.optim.lr_scheduler._LRScheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ]
    ],  # Scheduler type can vary
    num_epochs: int,
    device: Device,
    checkpoint_dir: Path,
    validation_interval: int,
    accumulate_grad_batches: int,
) -> None:
    """Main training loop."""
    print(f"[TRAIN] Starting training for {num_epochs} epochs on device: {device}")
    model.to(device)
    # Convert numpy distance matrix to tensor and move to device once
    distance_matrix = torch.from_numpy(distance_matrix_np).float().to(device)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        model.train()  # Set model to training mode
        train_loss_accum = 0.0
        optimizer.zero_grad()  # Zero gradients at the start of epoch/accumulation cycle

        for step, batch in enumerate(train_loader):
            step_start_time = time.time()
            # Move batch to device
            # Assuming batch is (mel_specs, labels)
            mel_specs, labels = batch
            mel_specs = mel_specs.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(mel_specs)

            # Calculate loss
            loss = hierarchical_distance_loss(logits, labels, distance_matrix)
            loss = (
                loss / accumulate_grad_batches
            )  # Scale loss for gradient accumulation

            # Backward pass
            loss.backward()

            train_loss_accum += (
                loss.item() * accumulate_grad_batches
            )  # Unscale for logging

            # Optimizer step (perform after accumulating gradients)
            if (step + 1) % accumulate_grad_batches == 0:
                optimizer.step()
                optimizer.zero_grad()  # Zero gradients after step
                global_step += 1
                step_end_time = time.time()

                # Logging (print statement as per convention)
                lr_encoder = optimizer.param_groups[0][
                    "lr"
                ]  # Assuming first group is encoder
                lr_head = optimizer.param_groups[1][
                    "lr"
                ]  # Assuming second group is head
                print(
                    f"[TRAIN] Epoch {epoch+1}, Step {global_step}, Loss: {loss.item() * accumulate_grad_batches:.4f}, LR Enc: {lr_encoder:.1e}, LR Head: {lr_head:.1e}, Time/Step: {step_end_time - step_start_time:.2f}s",
                )

                # --- Validation Step ---
                if val_loader is not None and global_step % validation_interval == 0:
                    val_start_time = time.time()
                    val_loss = _validate_epoch(
                        model, val_loader, distance_matrix, device,
                    )
                    val_end_time = time.time()
                    print(
                        f"[VAL] Step {global_step}, Validation Loss: {val_loss:.4f}, Time: {val_end_time - val_start_time:.2f}s",
                    )

                    # Scheduler step (example: based on validation loss)
                    if scheduler is not None and isinstance(
                        scheduler, ReduceLROnPlateau,
                    ):
                        scheduler.step(val_loss)

                    # Save checkpoint if validation loss improved
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_path = checkpoint_dir / f"best_model_step_{global_step}.pt"
                        torch.save(model.state_dict(), save_path)
                        print(
                            f"[CHECKPOINT] ✨ Best validation loss improved! Saved model to {save_path} ✨",
                        )

        # --- End of Epoch ---
        avg_train_loss = train_loss_accum / len(train_loader)
        epoch_end_time = time.time()
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"  Average Training Loss: {avg_train_loss:.4f}")
        print(f"  Epoch Time: {epoch_end_time - epoch_start_time:.2f}s")

        # Optional: Save checkpoint at the end of each epoch
        # epoch_save_path = checkpoint_dir / f"epoch_{epoch+1}_model.pt"
        # torch.save(model.state_dict(), epoch_save_path)
        # print(f"[CHECKPOINT] Saved epoch model to {epoch_save_path}")

    print("\n[TRAIN] 🎉 Training finished! Jeeves, fetch the celebratory crumpets! 🎉")


# --- Helper Function for Validation ---
def _validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    distance_matrix: Tensor,  # Expecting tensor here
    device: Device,
) -> float:
    """Runs a validation epoch and returns the average loss."""
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0.0
    with torch.no_grad():  # Disable gradient calculations
        for batch in val_loader:
            mel_specs, labels = batch
            mel_specs = mel_specs.to(device)
            labels = labels.to(device)

            logits = model(mel_specs)
            loss = hierarchical_distance_loss(logits, labels, distance_matrix)
            total_val_loss += loss.item()

    return total_val_loss / len(val_loader)


# --- Main Block for Testing/Demonstration ---
if __name__ == "__main__":
    print("--- Running trainer.py demonstration ---")
    print(
        "[DEMO] NOTE: This demo uses dummy data and won't produce meaningful results.",
    )

    # Create dummy components
    class DummyEncoder(nn.Module):
        def __init__(self, feature_size: int = 128) -> None:
            super().__init__()
            self.feature_size = feature_size
            self.config = type(
                "obj", (object,), {"d_model": feature_size},
            )()  # Mock config
            self.dummy_layer = nn.Linear(
                config.N_MELS * 10, self.feature_size,
            )  # Rough size match

        def forward(self, input_features: Tensor) -> object:
            # Flatten N_MELS and some time dim for Linear input
            bs, n_mels, n_frames = input_features.shape
            # Need a fixed size input for linear, this is tricky without real encoder
            # Let's just return random tensor of expected shape
            # Assume sequence length becomes 50 after encoder processing
            seq_len_out = 50
            return type(
                "obj",
                (object,),
                {"last_hidden_state": torch.randn(bs, seq_len_out, self.feature_size)},
            )()

    class DummyDataset(Dataset):
        def __init__(self, length: int = 20) -> None:
            self.len = length

        def __len__(self) -> int:
            return self.len

        def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
            # Rough estimate of frame count for 5s at 32kHz, hop 160, n_fft 1024 -> ~1000 frames
            frame_count = 1000  # Matches competition 5s window inference more closely
            mel = torch.randn(config.N_MELS, frame_count)
            # Target needs NUM_CLASSES = 4 for dummy distance matrix
            target = torch.randint(0, 2, (4,)).float()
            return mel, target

    dummy_encoder = DummyEncoder()
    dummy_model = model.BirdClefClassifier(
        dummy_encoder, 128, 4,
    )  # 4 classes for dummy dist matrix
    dummy_train_dataset = DummyDataset(
        length=40 * config.ACCUMULATE_GRAD_BATCHES,
    )  # Ensure enough steps
    dummy_val_dataset = DummyDataset(length=10)
    dummy_train_loader = DataLoader(dummy_train_dataset, batch_size=config.BATCH_SIZE)
    dummy_val_loader = DataLoader(dummy_val_dataset, batch_size=config.BATCH_SIZE)
    dummy_dist_matrix_np = np.array(
        [  # 4x4
            [0, 2, 6, 8],
            [2, 0, 2, 6],
            [6, 2, 0, 2],
            [8, 6, 2, 0],
        ],
        dtype=np.float32,
    )

    # Separate params for differential learning rate
    encoder_params = dummy_model.encoder.parameters()
    head_params = dummy_model.classifier_head.parameters()
    dummy_optimizer = optim.AdamW(
        [
            {"params": encoder_params, "lr": config.LEARNING_RATE_ENCODER},
            {"params": head_params, "lr": config.LEARNING_RATE_HEAD},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )

    dummy_scheduler = ReduceLROnPlateau(dummy_optimizer, "min", patience=2, factor=0.5)
    dummy_device = torch.device("cpu")  # Force CPU for demo

    try:
        print("[DEMO] Starting dummy training loop...")
        train_model(
            model=dummy_model,
            train_loader=dummy_train_loader,
            val_loader=dummy_val_loader,
            distance_matrix_np=dummy_dist_matrix_np,
            optimizer=dummy_optimizer,
            scheduler=dummy_scheduler,
            num_epochs=1,  # Just one epoch for demo
            device=dummy_device,
            checkpoint_dir=config.CHECKPOINT_DIR,
            validation_interval=1,  # Validate every step
            accumulate_grad_batches=config.ACCUMULATE_GRAD_BATCHES,
        )
        print("\n[DEMO] Dummy training loop completed.")

    except Exception as e:
        import traceback

        print(f"[ERROR] Demonstration failed: {e}")
        traceback.print_exc()

    print("--- End trainer.py demonstration ---")
