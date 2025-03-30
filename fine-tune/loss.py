# loss.py
"""Hierarchical distance-weighted loss function."""


import config  # Import config variables
import torch
import torch.nn.functional as functional  # Change from 'as F' to 'as functional'

# Type Alias
Tensor = torch.Tensor
DistanceMatrix = torch.Tensor # Expecting tensor form in loss function

# --- Top Level Function ---
def hierarchical_distance_loss(
    logits: Tensor,
    targets: Tensor,
    distance_matrix: DistanceMatrix,
    alpha: float = config.HIERARCHICAL_LOSS_ALPHA,
) -> Tensor:
    """
    Computes Binary Cross Entropy loss weighted by taxonomic distance.

    Args:
        logits: Raw output from the model (batch_size, num_classes).
        targets: Ground truth labels (batch_size, num_classes), 0 or 1.
        distance_matrix: Precomputed (num_classes, num_classes) matrix of
                         taxonomic distances. Should be on the same device as logits.
        alpha: Scaling factor for the distance penalty.

    Returns:
        Scalar loss value.
    """
    assert logits.ndim == 2, "Logits should be 2D (batch, classes)"
    assert targets.ndim == 2, "Targets should be 2D (batch, classes)"
    assert logits.shape == targets.shape, "Logits and targets shape mismatch"
    assert distance_matrix.ndim == 2, "Distance matrix should be 2D"
    assert distance_matrix.shape[0] == distance_matrix.shape[1] == logits.shape[1], \
        "Distance matrix dimension mismatch"
    assert alpha >= 0, "Alpha must be non-negative"

    # Use BCEWithLogitsLoss for numerical stability
    # Calculate element-wise loss *without* reduction
    bce_loss_elements = functional.binary_cross_entropy_with_logits(
        logits, targets, reduction='none',
    )

    # --- Calculate weights based on distance ---
    batch_size, num_classes = targets.shape
    # Ensure distance matrix is on the correct device
    dist_matrix_dev = distance_matrix.to(logits.device)

    # Initialize weights to 1
    weights = torch.ones_like(bce_loss_elements)

    # Iterate over samples (can this be vectorized?) - Vectorization is tricky here
    # due to the dependency on true positives within each sample.
    # Looping might be clearer and acceptable if batch size isn't enormous.
    for i in range(batch_size):
        true_pos_mask = targets[i] > 0.5 # Mask for true positives in this sample
        false_pos_mask = targets[i] < 0.5 # Mask for true negatives (potential FPs)

        # Check if there are any true positives in this sample
        if torch.any(true_pos_mask):
            # Get indices of true positive classes
            true_pos_indices = torch.where(true_pos_mask)[0]

            # Get indices of potential false positive classes (where target is 0)
            false_pos_indices = torch.where(false_pos_mask)[0]

            if len(false_pos_indices) > 0:
                # Calculate min distances from each potential FP to any TP
                # D[false_pos_indices, :]: Slice rows for FPs
                # [:, true_pos_indices]: Slice columns for TPs
                # Result: (num_fp, num_tp) matrix of distances
                fp_to_tp_distances = dist_matrix_dev[false_pos_indices][:, true_pos_indices]

                # Find the minimum distance for each FP row
                min_dists, _ = torch.min(fp_to_tp_distances, dim=1) # Shape: (num_fp)

                # Calculate weights: 1.0 + alpha * distance
                fp_weights = 1.0 + alpha * min_dists

                # Update weights for the false positive positions in this sample
                weights[i, false_pos_indices] = fp_weights

    # --- Apply weights and calculate final loss ---
    weighted_bce_loss = weights * bce_loss_elements
    final_loss = torch.mean(weighted_bce_loss) # Mean over batch and classes

    return final_loss

# --- Main Block for Testing/Demonstration ---
if __name__ == "__main__":
    print("--- Running loss.py demonstration ---")

    # Dummy data
    batch_demo = 2
    classes_demo = 4
    dummy_logits = torch.randn(batch_demo, classes_demo)
    # Sample 0: TP at index 0, FP at index 2 (far)
    # Sample 1: TP at index 1, FP at index 2 (close)
    dummy_targets = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    # Dummy distance matrix: D(0,1)=2, D(0,2)=6, D(0,3)=8
    #                       D(1,2)=2, D(1,3)=6
    #                       D(2,3)=2
    dummy_dist_matrix = torch.tensor([
        [0, 2, 6, 8],
        [2, 0, 2, 6],
        [6, 2, 0, 2],
        [8, 6, 2, 0],
    ], dtype=torch.float32)
    alpha_demo = 0.5

    try:
        print("[DEMO] Inputs:")
        print(f"  Logits:\n{dummy_logits}")
        print(f"  Targets:\n{dummy_targets}")
        print(f"  Distance Matrix:\n{dummy_dist_matrix}")
        print(f"  Alpha: {alpha_demo}")

        loss_val = hierarchical_distance_loss(
            dummy_logits, dummy_targets, dummy_dist_matrix, alpha_demo,
        )
        print(f"\n[DEMO] Calculated Loss: {loss_val.item()}")

        # Manual check for sample 0, false positive at index 2:
        # True positive is index 0. Distance D(2, 0) = 6.
        # Weight = 1.0 + 0.5 * 6 = 4.0
        # Manual check for sample 1, false positive at index 2:
        # True positive is index 1. Distance D(2, 1) = 2.
        # Weight = 1.0 + 0.5 * 2 = 2.0
        # The loss contribution for the FP at index 2 should be higher in sample 0.

        # Test case with no true positives
        dummy_targets_no_tp = torch.zeros_like(dummy_targets)
        loss_no_tp = hierarchical_distance_loss(
             dummy_logits, dummy_targets_no_tp, dummy_dist_matrix, alpha_demo,
        )
        # Should be equivalent to standard BCEWithLogitsLoss in this case
        standard_bce = functional.binary_cross_entropy_with_logits(dummy_logits, dummy_targets_no_tp)
        print(f"\n[DEMO] Loss with no TPs: {loss_no_tp.item()}")
        print(f"[DEMO] Standard BCE (no TPs): {standard_bce.item()}")
        assert torch.isclose(loss_no_tp, standard_bce), "Loss mismatch when no TPs"

    except Exception as e:
        import traceback
        print(f"[ERROR] Demonstration failed: {e}")
        traceback.print_exc()

    print("--- End loss.py demonstration ---")
