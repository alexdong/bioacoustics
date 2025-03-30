# ssl/trainer.py
"""Functions for the MAE SSL training loop."""

import time
from pathlib import Path
from typing import Any

# Use config settings from the ssl directory
import config as ssl_config
import data_loader  # Import data_loader module
import model  # Import the model module
import torch
import torch.nn as nn
import torch.optim as optim
from loss import mae_reconstruction_loss

# Example scheduler: Cosine Annealing
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# Type Alias
Tensor = torch.Tensor
Device = torch.device

# --- Top Level Function ---
def train_ssl_model(
    model: nn.Module, # The MAE model (encoder + decoder)
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Any, # Scheduler type can vary
    num_epochs: int,
    device: Device,
    output_dir: Path, # Dir to save final encoder and optional MAE checkpoints
    checkpoint_interval: int, # Steps between saving full MAE model (0 to disable)
    accumulate_grad_batches: int,
) -> None:
    """Main SSL training loop for MAE."""
    print(f"[SSL TRAIN] Starting SSL training for {num_epochs} epochs on device: {device}")
    model.to(device)

    global_step = 0
    final_encoder_save_path = output_dir / "ssl_encoder_final.pt" # Predefined name

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n--- SSL Epoch {epoch+1}/{num_epochs} ---")

        model.train() # Set model to training mode
        train_loss_accum = 0.0
        optimizer.zero_grad() # Zero gradients at the start of epoch/accumulation cycle

        for step, batch in enumerate(train_loader):
            step_start_time = time.time()
            # Move batch to device
            # Assuming batch is (masked_spec, original_spec, mask)
            masked_spec, original_spec, mask = batch
            masked_spec = masked_spec.to(device)
            original_spec = original_spec.to(device)
            mask = mask.to(device)

            # Forward pass through MAE model
            predictions = model(masked_spec, mask)

            # Calculate reconstruction loss
            loss = mae_reconstruction_loss(predictions, original_spec, mask)
            loss = loss / accumulate_grad_batches # Scale loss

            # Backward pass
            loss.backward()

            train_loss_accum += loss.item() * accumulate_grad_batches # Unscale for logging

            # Optimizer step
            if (step + 1) % accumulate_grad_batches == 0:
                optimizer.step()
                optimizer.zero_grad() # Zero gradients after step
                if scheduler is not None: # Step scheduler after optimizer step
                     scheduler.step()
                global_step += 1
                step_end_time = time.time()

                # Logging
                current_lr = optimizer.param_groups[0]['lr'] # Get LR
                print(f"[SSL TRAIN] Epoch {epoch+1}, Step {global_step}, Loss: {loss.item() * accumulate_grad_batches:.4f}, LR: {current_lr:.1e}, Time/Step: {step_end_time - step_start_time:.2f}s")

                # Optional: Save full MAE model checkpoint periodically
                if checkpoint_interval > 0 and global_step % checkpoint_interval == 0:
                    mae_save_path = output_dir / f"mae_model_step_{global_step}.pt"
                    torch.save(model.state_dict(), mae_save_path)
                    print(f"[CHECKPOINT] Saved full MAE model to {mae_save_path}")


        # --- End of Epoch ---
        avg_train_loss = train_loss_accum / len(train_loader)
        epoch_end_time = time.time()
        print(f"--- SSL Epoch {epoch+1} Summary ---")
        print(f"  Average Training Loss: {avg_train_loss:.4f}")
        print(f"  Epoch Time: {epoch_end_time - epoch_start_time:.2f}s")

    # --- End of Training ---
    print("\n[SSL TRAIN] 🎉 SSL Training finished! Saving final encoder weights... 🎉")
    # Extract and save only the encoder's state dictionary
    encoder_state_dict = model.encoder.state_dict()
    torch.save(encoder_state_dict, final_encoder_save_path)
    print(f"[SSL TRAIN] Final encoder model saved to: {final_encoder_save_path}")


# --- Main Block for Testing/Demonstration ---
if __name__ == "__main__":
    print("--- Running ssl/trainer.py demonstration ---")
    print("[DEMO] NOTE: This demo uses dummy data and won't produce meaningful results.")

    # Create dummy components
    class DummySSLDataset(Dataset):
        def __init__(self, length: int = 20) -> None:
            self.len = length

        def __len__(self) -> int:
            return self.len

        def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
            frames = 100 # Shorter sequence for faster demo
            masked = torch.randn(ssl_config.N_MELS, frames)
            orig = torch.randn(ssl_config.N_MELS, frames)
            mask = torch.rand(frames) > 0.3 # Boolean mask
            return masked, orig, mask

    # Build dummy MAE model
    dummy_mae_model = model.build_ssl_model() # Reuse build function
    dummy_train_dataset = DummySSLDataset(length=40 * ssl_config.ACCUMULATE_GRAD_BATCHES)
    dummy_train_loader = DataLoader(
        dummy_train_dataset,
        batch_size=ssl_config.BATCH_SIZE,
        collate_fn=data_loader._ssl_collate_fn # Use the collate fn
    )

    dummy_optimizer = optim.AdamW(dummy_mae_model.parameters(), lr=ssl_config.LEARNING_RATE)
    # Dummy scheduler (e.g., step once per 'batch' effectively)
    dummy_scheduler = CosineAnnealingLR(dummy_optimizer, T_max=len(dummy_train_loader) // ssl_config.ACCUMULATE_GRAD_BATCHES)
    dummy_device = torch.device("cpu") # Force CPU for demo

    try:
        print("[DEMO] Starting dummy SSL training loop...")
        train_ssl_model(
            model=dummy_mae_model,
            train_loader=dummy_train_loader,
            optimizer=dummy_optimizer,
            scheduler=dummy_scheduler,
            num_epochs=1, # Just one epoch for demo
            device=dummy_device,
            output_dir=ssl_config.SSL_OUTPUT_DIR, # Use configured output dir
            checkpoint_interval=1, # Save checkpoint every step
            accumulate_grad_batches=ssl_config.ACCUMULATE_GRAD_BATCHES,
        )
        print("\n[DEMO] Dummy SSL training loop completed.")
        # Check if encoder file was created
        assert ssl_config.SSL_ENCODER_CHECKPOINT_PATH.exists(), "Final encoder file not saved"
        print(f"[DEMO] Final encoder file found: {ssl_config.SSL_ENCODER_CHECKPOINT_PATH}")


    except Exception as e:
        import traceback
        print(f"[ERROR] Demonstration failed: {e}")
        traceback.print_exc()

    print("--- End ssl/trainer.py demonstration ---")
