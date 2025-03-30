# predictor.py
"""Functions for running inference with a trained model."""

from pathlib import Path

import config
import model  # To build model structure
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Type Alias
Tensor = torch.Tensor
Device = torch.device

# --- Top Level Function ---
def predict_on_dataset(
    model_instance: nn.Module,
    dataloader: DataLoader,
    device: Device,
) -> np.ndarray:
    """Runs inference on a dataloader and returns probabilities."""
    print(f"[PREDICT] Starting prediction on device: {device}")
    model_instance.to(device)
    model_instance.eval() # Set model to evaluation mode

    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            # Assuming dataloader yields (mel_specs, labels) or just (mel_specs)
            # Handle both cases if needed. Assuming (mel_specs, _) for now.
            mel_specs, _ = batch # Ignore labels if present
            mel_specs = mel_specs.to(device)

            logits = model_instance(mel_specs)
            probabilities = torch.sigmoid(logits) # Convert logits to probabilities

            all_probabilities.append(probabilities.cpu().numpy())

    print("[PREDICT] Prediction finished.")
    # Concatenate results from all batches
    if not all_probabilities:
        return np.array([]) # Return empty if no data
    predictions_np = np.concatenate(all_probabilities, axis=0)
    return predictions_np


def load_model_for_inference(
    checkpoint_path: Path,
    num_classes: int,
    device: Device,
) -> nn.Module:
    """Loads model structure and weights from a checkpoint."""
    print(f"[PREDICT] Loading model from checkpoint: {checkpoint_path}")
    assert checkpoint_path.is_file(), f"Checkpoint not found: {checkpoint_path}"

    # Build the model structure first (without loading SSL weights here)
    model_instance = model.build_model(num_classes, encoder_checkpoint_path=None)

    # Load the fine-tuned state dict
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if isinstance(state_dict, dict) and "model" in state_dict:
        # Handle case where state_dict is in the "model" key (e.g., PyTorch Lightning)
        state_dict = state_dict["model"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        # Handle case where state_dict is in the "state_dict" key (many frameworks)
        state_dict = state_dict["state_dict"]

    # Adapt keys if needed (e.g., if saved with 'module.' prefix)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Use strict loading to ensure all keys match
    try:
        model_instance.load_state_dict(state_dict, strict=True)
        print("[PREDICT] Model weights loaded successfully with strict=True")
    except Exception as e:
        print(f"[WARNING] Strict loading failed: {e}")
        print("[WARNING] Attempting non-strict loading as fallback")
        missing_keys, unexpected_keys = model_instance.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"[WARNING] Missing keys when loading fine-tuned state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys when loading fine-tuned state_dict: {unexpected_keys}")
        print("[WARNING] Model loaded with non-strict matching - results may be unreliable")
    print("[PREDICT] Model weights loaded successfully.")

    model_instance.to(device)
    model_instance.eval()
    return model_instance


# --- Main Block for Testing/Demonstration ---
if __name__ == "__main__":
    print("--- Running predictor.py demonstration ---")

    # Use dummy data/model from trainer demo
    n_classes_demo = 4
    dummy_device = torch.device("cpu")

    # 1. Create and save a dummy checkpoint
    dummy_encoder_inf = model.DummyEncoder() # Use the same dummy encoder
    dummy_model_inf = model.BirdClefClassifier(dummy_encoder_inf, 128, n_classes_demo)
    dummy_checkpoint_path = config.CHECKPOINT_DIR / "dummy_predictor_model.pt"
    torch.save(dummy_model_inf.state_dict(), dummy_checkpoint_path)
    print(f"[DEMO] Saved dummy model checkpoint: {dummy_checkpoint_path}")

    # 2. Load the model
    try:
        loaded_model = load_model_for_inference(dummy_checkpoint_path, n_classes_demo, dummy_device)
        print("[DEMO] Model loaded successfully for inference.")

        # 3. Create dummy inference data
        dummy_inf_dataset = model.DummyDataset(length=5) # Reuse dummy dataset
        dummy_inf_loader = DataLoader(dummy_inf_dataset, batch_size=2)
        print(f"[DEMO] Created dummy inference dataloader with {len(dummy_inf_dataset)} samples.")

        # 4. Run prediction
        predictions = predict_on_dataset(loaded_model, dummy_inf_loader, dummy_device)
        print(f"[DEMO] Predictions array shape: {predictions.shape}") # Should be (5, 4)
        print(f"[DEMO] Sample predictions:\n{predictions[:2]}")
        assert predictions.shape == (len(dummy_inf_dataset), n_classes_demo), "Prediction shape mismatch"
        assert np.all((predictions >= 0) & (predictions <= 1)), "Probabilities out of range"

    except Exception as e:
        import traceback
        print(f"[ERROR] Demonstration failed: {e}")
        traceback.print_exc()


    print("--- End predictor.py demonstration ---")
