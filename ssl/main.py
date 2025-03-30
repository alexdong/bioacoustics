# ssl/main.py
"""Main script to run the SSL MAE pre-training process."""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR # Example scheduler
import numpy as np
import random
from pathlib import Path

# Import local SSL modules
import config as ssl_config
import data_loader as ssl_data_loader
import model as ssl_model
import trainer as ssl_trainer

def set_seed(seed: int) -> None:
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic algorithms are used if available
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

# --- Main Execution ---
print("--- Starting SSL Pre-training Script ---")
set_seed(ssl_config.RANDOM_SEED)
device = torch.device(ssl_config.DEVICE)
print(f"[SSL MAIN] Using device: {device}")

# 1. Create SSL DataLoader
print("[SSL MAIN] Creating SSL dataloader...")
ssl_loader = ssl_data_loader.create_ssl_dataloader(
    audio_dir=ssl_config.UNLABELED_AUDIO_DIR,
    batch_size=ssl_config.BATCH_SIZE,
    num_workers=ssl_config.NUM_WORKERS,
)
assert ssl_loader is not None, "SSL DataLoader creation failed"

# 2. Build MAE Model
print("[SSL MAIN] Building MAE model...")
mae_model = ssl_model.build_ssl_model()

# 3. Setup Optimizer and Scheduler
print("[SSL MAIN] Setting up optimizer and scheduler...")
# MAE typically trains encoder and decoder together with same LR initially
optimizer = optim.AdamW(
    mae_model.parameters(),
    lr=ssl_config.LEARNING_RATE,
    weight_decay=ssl_config.WEIGHT_DECAY
)

# Example scheduler: Cosine annealing over total training steps
num_training_steps = (len(ssl_loader) // ssl_config.ACCUMULATE_GRAD_BATCHES) * ssl_config.NUM_EPOCHS
scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-6) # Decay to near zero
print(f"[SSL MAIN] Using CosineAnnealingLR scheduler over {num_training_steps} steps.")

# 4. Run SSL Training
print("[SSL MAIN] Starting MAE model training...")
try:
    ssl_trainer.train_ssl_model(
        model=mae_model,
        train_loader=ssl_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=ssl_config.NUM_EPOCHS,
        device=device,
        output_dir=ssl_config.SSL_OUTPUT_DIR,
        checkpoint_interval=ssl_config.CHECKPOINT_INTERVAL_STEPS,
        accumulate_grad_batches=ssl_config.ACCUMULATE_GRAD_BATCHES,
    )
except Exception as e:
    print(f"\n[ERROR] SSL Training loop failed: {e}")
    import traceback
    traceback.print_exc()
    print("[SSL MAIN] Training aborted due to error.")
else:
    print("\n[SSL MAIN] SSL Training completed successfully.")
    assert ssl_config.SSL_ENCODER_CHECKPOINT_PATH.exists(), "Final SSL encoder checkpoint not found!"
    print(f"[SSL MAIN] Final encoder checkpoint saved to: {ssl_config.SSL_ENCODER_CHECKPOINT_PATH}")

print("\n--- SSL Pre-training Script Finished ---")
print("Jeeves seems pleased with the self-supervised endeavour. Onwards!")