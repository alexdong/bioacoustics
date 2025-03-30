# main.py

import random

import config
import data_loader
import model
import numpy as np
import pandas as pd
import taxonomy
import torch
import torch.optim as optim
import trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau

# import predictor # Optional: if running prediction after training

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
        # torch.backends.cudnn.benchmark = False # Can slow down, but more deterministic

# --- Main Execution ---
print("--- Starting Fine-Tuning Script ---")
set_seed(config.RANDOM_SEED)
device = torch.device(config.DEVICE)
print(f"[MAIN] Using device: {device}")

# 1. Load Metadata and Define Species List
print("[MAIN] Loading training metadata...")
assert config.TRAIN_METADATA_PATH.is_file(), f"Metadata not found: {config.TRAIN_METADATA_PATH}"
train_meta_df = pd.read_csv(config.TRAIN_METADATA_PATH)
# Verify that primary_label column exists in metadata
assert 'primary_label' in train_meta_df.columns, "Missing 'primary_label' column in metadata"
species_list = sorted(list(train_meta_df['primary_label'].unique()))
num_classes = len(species_list)
print(f"[MAIN] Found {num_classes} unique species.")
if num_classes != config.NUM_CLASSES:
    print(f"[WARNING] Mismatch between detected classes ({num_classes}) and config.NUM_CLASSES ({config.NUM_CLASSES})")
    print("[WARNING] Updating config.NUM_CLASSES to match detected classes")
    config.NUM_CLASSES = num_classes

# 2. Load Taxonomy and Compute Distance Matrix
print("[MAIN] Loading taxonomy and computing distance matrix...")
try:
    _, distance_matrix_np = taxonomy.load_and_compute_distance_matrix(
        config.TAXONOMY_PATH, species_list,
    )
    # Save distance matrix for potential reuse/analysis
    dist_matrix_path = config.TEMP_DIR / "distance_matrix.npy"
    np.save(dist_matrix_path, distance_matrix_np)
    print(f"[MAIN] Saved distance matrix to: {dist_matrix_path}")
except Exception as e:
    print(f"[WARNING] Error computing distance matrix: {e}")
    print("[WARNING] Using identity matrix as fallback")
    # Create identity matrix as fallback (no hierarchical penalty)
    distance_matrix_np = np.eye(num_classes)


# 3. Create DataLoaders with validation split
print("[MAIN] Creating dataloaders with validation split...")
train_loader, val_loader = data_loader.create_dataloaders(
    metadata_path=config.TRAIN_METADATA_PATH,
    audio_dir=config.AUDIO_DIR,
    species_list=species_list,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    val_split=0.2,  # Use 20% of data for validation
    random_state=config.RANDOM_SEED,
)
# Basic check
assert train_loader is not None, "Train loader creation failed"
assert val_loader is not None, "Validation loader creation failed - required for proper evaluation"


# 4. Build Model
print("[MAIN] Building model...")
fine_tune_model = model.build_model(
    num_classes=num_classes,
    encoder_checkpoint_path=config.ENCODER_CHECKPOINT_PATH,
)


# 5. Setup Optimizer and Scheduler
print("[MAIN] Setting up optimizer and scheduler...")
# Apply differential learning rates
encoder_params = fine_tune_model.encoder.parameters()
head_params = fine_tune_model.classifier_head.parameters()
optimizer = optim.AdamW([
    {'params': encoder_params, 'lr': config.LEARNING_RATE_ENCODER, 'weight_decay': config.WEIGHT_DECAY},
    {'params': head_params, 'lr': config.LEARNING_RATE_HEAD, 'weight_decay': config.WEIGHT_DECAY}, # Can have different WD if needed
])

# Example scheduler: Reduce learning rate when validation loss plateaus
# Use validation loader existence to decide if scheduler is active
scheduler = None
if val_loader is not None:
     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)
     print("[MAIN] Using ReduceLROnPlateau scheduler based on validation loss.")
else:
     print("[MAIN] No validation loader, scheduler not used.")


# 6. Run Training
print("[MAIN] Starting model training...")
try:
    trainer.train_model(
        model=fine_tune_model,
        train_loader=train_loader,
        val_loader=val_loader, # Pass None if no validation set
        distance_matrix_np=distance_matrix_np,
        optimizer=optimizer,
        scheduler=scheduler, # Pass None if not used
        num_epochs=config.NUM_EPOCHS,
        device=device,
        checkpoint_dir=config.CHECKPOINT_DIR,
        validation_interval=config.VALIDATION_INTERVAL_STEPS,
        accumulate_grad_batches=config.ACCUMULATE_GRAD_BATCHES,
    )
except Exception as e:
    print(f"\n[ERROR] Training loop failed: {e}")
    import traceback
    traceback.print_exc()
    print("[MAIN] Training aborted due to error.")
else:
    print("\n[MAIN] Training completed successfully.")

# 7. Optional: Run Prediction/Evaluation on a test set after training
# print("\n[MAIN] Running prediction on test set (example)...")
# try:
#     # Assume a test dataloader exists: test_loader
#     # Load the best checkpoint
#     best_checkpoint_path = config.CHECKPOINT_DIR / "best_model_step_XYZ.pt" # Find the actual best
#     if best_checkpoint_path.exists():
#         inference_model = predictor.load_model_for_inference(
#             best_checkpoint_path, num_classes, device
#         )
#         # predictions = predictor.predict_on_dataset(inference_model, test_loader, device)
#         # TODO: Save predictions or calculate final metrics
#         print("[MAIN] Prediction example finished.")
#     else:
#         print("[MAIN] Best checkpoint not found, skipping prediction.")
# except Exception as e:
#     print(f"[ERROR] Prediction failed: {e}")


print("\n--- Fine-Tuning Script Finished ---")
print("Jeeves approves. Remember, temporary files in /tmp were not tidied up, as requested. Pip pip!")
