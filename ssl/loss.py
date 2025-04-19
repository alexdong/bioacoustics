# ssl/loss.py
"""Reconstruction loss for MAE SSL."""

import torch

# Use config settings from the ssl directory
# import config as ssl_config # Not strictly needed here

# Type Alias
Tensor = torch.Tensor


# --- Top Level Function ---
def mae_reconstruction_loss(
    predictions: Tensor,  # Shape: (B, N_FRAMES, N_MELS) - Output from Decoder
    targets: Tensor,  # Shape: (B, N_MELS, N_FRAMES) - Original Mel Spec
    mask: Tensor,  # Shape: (B, N_FRAMES) - Boolean mask (True=masked)
) -> Tensor:
    """
    Computes the Mean Squared Error (MSE) loss only on the masked patches.

    Args:
        predictions: Predicted spectrogram patches from the decoder.
        targets: Original spectrogram patches.
        mask: Boolean mask indicating which time frames were masked.

    Returns:
        Scalar loss value normalized by the number of masked patches.
    """
    assert predictions.ndim == 3, "Predictions should be 3D (B, T_pred, C)"
    assert targets.ndim == 3, "Targets should be 3D (B, C, T_target)"
    assert mask.ndim == 2, "Mask should be 2D (B, T_mask)"
    # Allow for potential frame mismatch between prediction and target/mask due to encoder
    # assert predictions.shape[0] == targets.shape[0] == mask.shape[0], "Batch size mismatch"
    # assert predictions.shape[1] == mask.shape[1], "Prediction frame count mismatch with mask"
    # assert targets.shape[2] == mask.shape[1], "Target frame count mismatch with mask"
    # assert predictions.shape[2] == targets.shape[1], "Channel (N_MELS) mismatch"

    # Permute predictions to match target shape (B, N_MELS, N_FRAMES)
    predictions = predictions.permute(0, 2, 1)

    # Handle potential frame count mismatch between predictions and targets/mask
    # This can happen if Whisper encoder conv layers change sequence length.
    # Simplest: truncate target/mask to match prediction length if needed.
    batch_size, n_mels, n_frames_pred = predictions.shape
    n_frames_mask = mask.shape[1]

    if n_frames_pred != n_frames_mask:
        print(
            f"[WARNING][LOSS] Frame mismatch! Pred: {n_frames_pred}, Mask: {n_frames_mask}. Truncating mask/target.",
        )
        min_frames = min(n_frames_pred, n_frames_mask)
        predictions = predictions[:, :, :min_frames]
        targets = targets[:, :, :min_frames]
        mask = mask[:, :min_frames]

    # Expand mask to match the shape of targets/predictions (B, N_MELS, N_FRAMES)
    mask_expanded = mask.unsqueeze(1).expand_as(targets)  # (B, 1, T) -> (B, C, T)

    # Calculate squared error only on masked patches
    squared_error = (predictions - targets) ** 2
    # Zero out error on unmasked patches
    masked_squared_error = squared_error * mask_expanded

    # Sum the squared error over the masked patches and normalize
    # Sum over channel and time dimensions, then mean over batch
    loss_per_sample = torch.sum(
        masked_squared_error, dim=(1, 2),
    )  # Sum over C, T -> Shape (B,)
    # Normalize by the number of masked elements *in each sample* (mask sum * N_MELS)
    # Add epsilon to avoid division by zero if a sample somehow has no masked frames
    num_masked_per_sample = torch.sum(mask, dim=1) * n_mels + 1e-7  # Shape (B,)
    normalized_loss_per_sample = loss_per_sample / num_masked_per_sample

    # Mean loss across the batch
    final_loss = torch.mean(normalized_loss_per_sample)

    return final_loss


# --- Main Block for Testing/Demonstration ---
if __name__ == "__main__":
    print("--- Running ssl/loss.py demonstration ---")

    # Dummy data
    batch_demo = 2
    classes_demo = 8  # N_MELS
    frames_demo = 20
    # Decoder output shape (B, T, C)
    dummy_preds = torch.randn(batch_demo, frames_demo, classes_demo)
    # Target shape (B, C, T)
    dummy_targets = torch.randn(batch_demo, classes_demo, frames_demo)
    # Mask shape (B, T), boolean, True=masked
    dummy_mask = torch.rand(batch_demo, frames_demo) > 0.3  # ~70% masked

    try:
        print("[DEMO] Inputs:")
        print(f"  Predictions shape: {dummy_preds.shape}")
        print(f"  Targets shape: {dummy_targets.shape}")
        print(f"  Mask shape: {dummy_mask.shape}")
        print(f"  Masked elements per sample: {dummy_mask.sum(dim=1).tolist()}")

        loss_val = mae_reconstruction_loss(dummy_preds, dummy_targets, dummy_mask)
        print(f"\n[DEMO] Calculated MAE Loss: {loss_val.item()}")
        assert loss_val >= 0, "Loss should be non-negative"

        # Test with frame mismatch
        print("\n[DEMO] Testing frame mismatch handling...")
        dummy_preds_short = dummy_preds[:, :-1, :]  # Shorter prediction
        loss_mismatch = mae_reconstruction_loss(
            dummy_preds_short, dummy_targets, dummy_mask,
        )
        print(f"[DEMO] Calculated Loss (mismatch): {loss_mismatch.item()}")

    except Exception as e:
        import traceback

        print(f"[ERROR] Demonstration failed: {e}")
        traceback.print_exc()

    print("--- End ssl/loss.py demonstration ---")
