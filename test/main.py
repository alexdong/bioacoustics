# test/main.py

import argparse
from pathlib import Path
from typing import List

import config as test_config
import data_loader as test_data_loader
import model_loader as test_model_loader
import numpy as np
import pandas as pd
import torch

# Import fine_tune config to get species list (if needed and successful)
try:
    import config as fine_tune_config
except ImportError:
    fine_tune_config = None

# Type Alias
Device = torch.device


# --- Top Level Function ---
def run_inference(
    model_format: str,
    checkpoint_path: Path,
    test_audio_dir: Path,
    test_metadata_path: Path | None,
    submission_path: Path,
    batch_size: int,
    num_workers: int,
    device: Device,
    quantized_model_path: Path | None = None,
    torchscript_model_path: Path | None = None,
) -> None:
    """Runs the full inference pipeline."""
    print("[INFERENCE] Starting inference process...")
    print(f"[INFERENCE]  - Model Format: {model_format}")
    print(f"[INFERENCE]  - Checkpoint: {checkpoint_path}")
    print(f"[INFERENCE]  - Test Audio Dir: {test_audio_dir}")
    print(f"[INFERENCE]  - Test Metadata: {test_metadata_path}")
    print(f"[INFERENCE]  - Batch Size: {batch_size}")
    print(f"[INFERENCE]  - Device: {device}")

    # 1. Load Model
    model = test_model_loader.load_inference_model(
        model_format=model_format,
        checkpoint_path=checkpoint_path,
        device=device,
        quantized_path=quantized_model_path,
        torchscript_path=torchscript_model_path,
    )

    # 2. Create Test DataLoader
    test_loader, row_ids = test_data_loader.create_test_dataloader(
        test_audio_dir=test_audio_dir,
        metadata_path=test_metadata_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # 3. Run Prediction in Batches
    all_probabilities_list: List[np.ndarray] = []
    processed_row_ids: List[str] = []  # Store row_ids in order of processing

    model.eval()  # Ensure model is in eval mode
    print(f"[INFERENCE] Processing {len(row_ids)} test chunks in batches...")
    with torch.no_grad():
        for batch_data, batch_row_ids in test_loader:
            # batch_data is the Mel spectrogram tensor
            batch_data = batch_data.to(device)

            # Handle different model types (PyTorch vs TorchScript)
            if isinstance(model, torch.jit.ScriptModule):
                logits = model(batch_data)
            elif isinstance(model, torch.nn.Module):
                logits = model(batch_data)
            else:
                raise TypeError(f"Loaded model has unexpected type: {type(model)}")

            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(logits)
            all_probabilities_list.append(probabilities.cpu().numpy())
            processed_row_ids.extend(batch_row_ids)  # Store row_ids as they come

    print("[INFERENCE] Prediction loop finished.")

    # Assert that the number of processed row_ids matches the dataloader size *if* metadata was used
    if test_metadata_path:
        assert len(processed_row_ids) == len(
            row_ids,
        ), "Mismatch in processed row IDs count"
        # Reorder results based on original row_ids from metadata if needed
        # This might be necessary if DataLoader order isn't guaranteed (though shuffle=False helps)
        # For simplicity, assuming DataLoader preserves order when shuffle=False

    # Concatenate all predictions
    assert len(all_probabilities_list) > 0, "No predictions were generated."
    all_probabilities = np.concatenate(all_probabilities_list, axis=0)
    print(
        f"[INFERENCE] Final predictions array shape: {all_probabilities.shape}",
    )  # (num_samples, num_classes)

    # 4. Format Submission File
    print("[INFERENCE] Formatting submission file...")
    # Get the list of species names (column headers)
    # Need to load this reliably, e.g., from saved fine-tuning data or metadata
    try:
        # Attempt to get from fine_tune config metadata read
        # This relies on the import structure and might fail.
        ft_meta = pd.read_csv(
            test_config.FINE_TUNE_DIR / "config.py::TRAIN_METADATA_PATH",
        )  # Hacky way
        species_list = sorted(list(ft_meta["primary_label"].unique()))
    except Exception:
        # Fallback: Try loading from sample submission columns if metadata was used
        print(
            "[WARNING] Could not reliably load species list from fine_tune config. Trying sample submission...",
        )
        if test_metadata_path and test_config.TEST_METADATA_PATH:
            sample_df = pd.read_csv(test_config.TEST_METADATA_PATH)
            # Assume species columns are all columns except 'row_id'
            species_list = [col for col in sample_df.columns if col != "row_id"]
            print(
                f"[INFERENCE] Using {len(species_list)} species from sample submission columns.",
            )
        else:
            raise ValueError("Cannot determine species list for submission header!")

    assert all_probabilities.shape[1] == len(
        species_list,
    ), f"Prediction dimension ({all_probabilities.shape[1]}) doesn't match species list ({len(species_list)})"

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "row_id": processed_row_ids,  # Use the row_ids processed by the loader
        },
    )
    prob_df = pd.DataFrame(all_probabilities, columns=species_list)
    submission_df = pd.concat([submission_df, prob_df], axis=1)

    # Reorder according to original row_ids from metadata if necessary and available
    if test_metadata_path and test_config.TEST_METADATA_PATH:
        sample_df = pd.read_csv(test_config.TEST_METADATA_PATH)
        original_order_df = pd.DataFrame({"row_id": sample_df["row_id"]})
        # Ensure all processed IDs are in the original list (and vice versa if strict)
        assert set(processed_row_ids) == set(
            original_order_df["row_id"],
        ), "Mismatch between processed and expected row IDs"
        submission_df = pd.merge(
            original_order_df, submission_df, on="row_id", how="left",
        )
        print("[INFERENCE] Reordered submission based on sample_submission.csv")

    # Save submission file
    submission_df.to_csv(submission_path, index=False)
    print(f"[INFERENCE] ✅ Submission file saved successfully to: {submission_path} ✅")


# --- Argument Parsing and Main Execution ---
if __name__ == "__main__":
    print("--- Running Inference Script ---")
    parser = argparse.ArgumentParser(
        description="Run inference for BirdCLEF competition.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default=str(test_config.MODEL_CHECKPOINT_PATH),
        help="Path to the fine-tuned model checkpoint (.pt).",
    )
    parser.add_argument(
        "--test_audio_dir",
        type=str,
        required=False,
        default=str(test_config.TEST_AUDIO_DIR),
        help="Path to the directory containing test audio files.",
    )
    parser.add_argument(
        "--test_metadata",
        type=str,
        required=False,
        default=(
            str(test_config.TEST_METADATA_PATH)
            if test_config.TEST_METADATA_PATH
            else None
        ),
        help="Path to the test metadata file (e.g., sample_submission.csv). If None, glob test_audio_dir.",
    )
    parser.add_argument(
        "--submission_file",
        type=str,
        required=False,
        default=str(test_config.SUBMISSION_OUTPUT_PATH),
        help="Path to save the output submission CSV file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=test_config.INFERENCE_BATCH_SIZE,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=test_config.NUM_WORKERS,
        help="Number of workers for DataLoader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default=test_config.DEVICE,
        choices=["cuda", "cpu"],
        help="Device to run inference on ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--model_format",
        type=str,
        required=False,
        default=test_config.DEFAULT_MODEL_FORMAT,
        choices=["pytorch", "int8", "torchscript"],
        help="Format of the model to load.",
    )
    parser.add_argument(
        "--quantized_path",
        type=str,
        required=False,
        default=(
            str(test_config.QUANTIZED_MODEL_PATH)
            if test_config.QUANTIZED_MODEL_PATH
            else None
        ),
        help="Path to the quantized model (needed for format 'int8', specific usage depends on quantization type).",
    )
    parser.add_argument(
        "--torchscript_path",
        type=str,
        required=False,
        default=(
            str(test_config.TORCHSCRIPT_MODEL_PATH)
            if test_config.TORCHSCRIPT_MODEL_PATH
            else None
        ),
        help="Path to the TorchScript model (needed for format 'torchscript').",
    )

    args = parser.parse_args()

    # Convert paths from string arguments to Path objects
    checkpoint_p = Path(args.checkpoint)
    test_audio_dir_p = Path(args.test_audio_dir)
    test_metadata_p = Path(args.test_metadata) if args.test_metadata else None
    submission_path_p = Path(args.submission_file)
    quantized_model_p = Path(args.quantized_path) if args.quantized_path else None
    torchscript_model_p = Path(args.torchscript_path) if args.torchscript_path else None
    selected_device = torch.device(args.device)

    # Basic validation of paths
    assert checkpoint_p.is_file(), f"Checkpoint file not found: {checkpoint_p}"
    assert (
        test_audio_dir_p.is_dir()
    ), f"Test audio directory not found: {test_audio_dir_p}"
    if test_metadata_p:
        assert (
            test_metadata_p.is_file()
        ), f"Test metadata file not found: {test_metadata_p}"
    if args.model_format == "int8" and quantized_model_p is None:
        # Note: Dynamic quantization uses the FP32 checkpoint, so this check might
        # be removed if only dynamic quantization is supported via model_loader.
        # But keeping it for clarity if static quantization is ever used.
        print(
            "[WARNING] --model_format 'int8' selected but --quantized_path not provided. Dynamic quantization will use the main --checkpoint path.",
        )
        # assert quantized_model_p is not None, "--quantized_path is required for --model_format 'int8'"
        # assert quantized_model_p.is_file(), f"Quantized model file not found: {quantized_model_p}"
    if args.model_format == "torchscript":
        assert (
            torchscript_model_p is not None
        ), "--torchscript_path is required for --model_format 'torchscript'"
        assert (
            torchscript_model_p.is_file()
        ), f"TorchScript model file not found: {torchscript_model_p}"

    # Run the main inference function
    try:
        run_inference(
            model_format=args.model_format,
            checkpoint_path=checkpoint_p,
            test_audio_dir=test_audio_dir_p,
            test_metadata_path=test_metadata_p,
            submission_path=submission_path_p,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=selected_device,
            quantized_model_path=quantized_model_p,
            torchscript_model_path=torchscript_model_p,
        )
    except Exception as e:
        print(f"\n[ERROR] Inference script failed: {e}")
        import traceback

        traceback.print_exc()
        print("[INFERENCE] Script aborted.")

    print("\n--- Inference Script Finished ---")
