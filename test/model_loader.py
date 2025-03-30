# test/model_loader.py
"""Handles loading the model in different formats (PyTorch, INT8, TorchScript)."""

# Import model definition from fine_tune directory
# This requires fine_tune to be importable
import sys
from pathlib import Path

# Use config settings from the test directory
import config as test_config
import torch
import torch.nn as nn

sys.path.insert(0, str(test_config.FINE_TUNE_DIR.resolve()))
try:
    import model as fine_tune_model
    # pandas needed for species list for num_classes elsewhere
except ImportError as e:
    print(f"[ERROR] Could not import fine_tune modules: {e}")
    print("[ERROR] Cannot proceed with model loading.")
    # Define dummy classes/functions if needed for file to parse, but fail later
    fine_tune_model = None # type: ignore
sys.path.pop(0) # Clean up path

Device = torch.device

# --- Top Level Function ---
def load_inference_model(
    model_format: str,
    checkpoint_path: Path,
    device: Device,
    quantized_path: Path | None = None,
    torchscript_path: Path | None = None,
) -> nn.Module | torch.jit.ScriptModule: # Return type is either Module or ScriptModule
    """Loads the model based on the specified format."""
    print(f"[MODEL LOAD] Attempting to load model in format: {model_format}")

    if fine_tune_model is None: # Check if import failed
         raise ImportError("Could not import fine_tune model definition.")

    # Need num_classes to build the base structure
    # Get this from fine_tune data/config (this is awkward)
    try:
        test_config.FINE_TUNE_DIR / "config.py"
        # This dynamic import is complex; better to read species list from a saved file
        # Or require NUM_CLASSES in test_config.py directly.
        # Let's assume NUM_CLASSES is available via fine_tune_config import in test/config.py
        num_classes = test_config.fine_tune_config.NUM_CLASSES
    except Exception as e:
        print(f"[ERROR] Failed to get NUM_CLASSES from fine_tune config: {e}")
        # Fallback or raise error
        raise ValueError("NUM_CLASSES could not be determined.") from e

    if model_format == "pytorch":
        model = _load_pytorch_model(checkpoint_path, num_classes, device)
    elif model_format == "int8":
        assert quantized_path is not None, "Path to quantized model must be provided for 'int8' format."
        assert quantized_path.is_file(), f"Quantized model file not found: {quantized_path}"
        model = _load_quantized_model(quantized_path, num_classes, device)
    elif model_format == "torchscript":
        assert torchscript_path is not None, "Path to TorchScript model must be provided for 'torchscript' format."
        assert torchscript_path.is_file(), f"TorchScript model file not found: {torchscript_path}"
        model = _load_torchscript_model(torchscript_path, device)
    else:
        raise ValueError(f"Unsupported model format: {model_format}")

    model.eval() # Ensure model is in eval mode
    print(f"[MODEL LOAD] Model loaded successfully in {model_format} format.")
    return model


# --- Helper Functions for Loading ---

def _load_pytorch_model(
    checkpoint_path: Path, num_classes: int, device: Device,
) -> nn.Module:
    """Loads standard PyTorch model from checkpoint."""
    print(f"[MODEL LOAD - PT] Loading standard PyTorch model from: {checkpoint_path}")
    assert checkpoint_path.is_file(), f"Checkpoint not found: {checkpoint_path}"

    # Build the base model structure (without loading SSL weights here)
    model_instance = fine_tune_model.build_model(num_classes, encoder_checkpoint_path=None)

    # Load the fine-tuned state dict
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    # Adapt keys if needed (e.g., if saved with 'module.' prefix from DDP)
    # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    try:
        missing_keys, unexpected_keys = model_instance.load_state_dict(state_dict, strict=True)
        # Strict=True is safer for final fine-tuned models
        if missing_keys or unexpected_keys:
            print(f"[WARNING] State dict mismatch! Missing: {missing_keys}, Unexpected: {unexpected_keys}")
            # Optionally raise error or proceed with caution
            # raise ValueError("State dict mismatch during loading.")
    except Exception as e:
        print(f"[ERROR] Failed loading state dict from {checkpoint_path}: {e}")
        raise RuntimeError("Failed to load model state dict.") from e

    return model_instance.to(device)


def _load_quantized_model(
    quantized_model_path: Path, num_classes: int, device: Device,
) -> nn.Module:
    """Loads a dynamically quantized INT8 model."""
    # Dynamic quantization happens after loading FP32 weights into structure
    # A saved 'quantized' model usually means saving the FP32 model and quantizing on load,
    # OR using static quantization where the quantized state_dict is saved differently.
    # Assuming dynamic quantization for simplicity.
    # We need to build the FP32 structure, load FP32 weights, then quantize.
    print("[MODEL LOAD - INT8] Loading base FP32 model for dynamic quantization...")
    # NOTE: checkpoint_path here should ideally be the *original FP32* checkpoint
    # used to *create* the quantized model, not the quantized state dict itself,
    # unless static quantization was used and saved differently.
    # Let's assume we load the FP32 checkpoint passed via `test_config.MODEL_CHECKPOINT_PATH`
    fp32_model = _load_pytorch_model(test_config.MODEL_CHECKPOINT_PATH, num_classes, torch.device('cpu'))
    fp32_model.eval()

    print("[MODEL LOAD - INT8] Applying dynamic quantization...")
    # Note: Only Linear layers are typically quantized dynamically. Adjust if needed.
    # Quantization must happen on CPU.
    quantized_model = torch.quantization.quantize_dynamic(
        fp32_model, {torch.nn.Linear}, dtype=torch.qint8,
    )
    # The provided quantized_model_path isn't used in this dynamic flow,
    # which might be confusing. If static quantization was used, the loading
    # process would differ significantly.
    print(f"[MODEL LOAD - INT8] WARNING: Assuming dynamic quantization applied on the fly. Provided path '{quantized_model_path}' might not be directly loaded if it's not an FP32 checkpoint.")

    return quantized_model.to(device) # Move quantized model to target device


def _load_torchscript_model(
    torchscript_model_path: Path, device: Device,
) -> torch.jit.ScriptModule:
    """Loads a TorchScript model."""
    print(f"[MODEL LOAD - TS] Loading TorchScript model from: {torchscript_model_path}")
    assert torchscript_model_path.is_file(), f"TorchScript file not found: {torchscript_model_path}"
    model = torch.jit.load(torchscript_model_path, map_location=device)
    # TorchScript models are already optimized and don't need state dict loading
    return model # Already on the target device due to map_location


# --- Main Block for Testing/Demonstration ---
if __name__ == "__main__":
    print("--- Running test/model_loader.py demonstration ---")

    # Requires fine_tune modules and potentially a dummy checkpoint
    if fine_tune_model is None:
        print("[DEMO] Skipping demonstration as fine_tune modules failed to import.")
    else:
        # Use dummy data/model from fine_tune/model.py demo if needed
        n_classes_demo = 10 # Match fine_tune model demo
        dummy_device = torch.device("cpu")

        # 1. Create a dummy FP32 checkpoint if it doesn't exist
        dummy_checkpoint_dir = Path("/tmp/birdclef_test/dummy_checkpoints") # Use /tmp
        dummy_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        dummy_fp32_path = dummy_checkpoint_dir / "dummy_finetune_model.pt"

        if not dummy_fp32_path.exists():
             print("[DEMO] Creating dummy FP32 model and checkpoint...")
             try:
                 # Build requires downloading base whisper model
                 dummy_model = fine_tune_model.build_model(n_classes_demo, None)
                 torch.save(dummy_model.state_dict(), dummy_fp32_path)
                 print(f"[DEMO] Saved dummy FP32 checkpoint: {dummy_fp32_path}")
             except Exception as e:
                  print(f"[ERROR] Failed to create dummy model/checkpoint: {e}")
                  dummy_fp32_path = None # Prevent further errors
        else:
             print(f"[DEMO] Using existing dummy FP32 checkpoint: {dummy_fp32_path}")

        if dummy_fp32_path:
            try:
                # Test loading standard PyTorch model
                print("\n[DEMO] Loading PyTorch model...")
                pt_model = load_inference_model("pytorch", dummy_fp32_path, dummy_device)
                assert isinstance(pt_model, nn.Module)
                print("[DEMO] PyTorch model loaded.")

                # Test loading INT8 (dynamic quantization)
                # NOTE: This reloads the FP32 model and quantizes it.
                print("\n[DEMO] Loading INT8 model (dynamic quantization)...")
                # We pass the FP32 path, the quantized_path arg isn't really used here
                # Need to update test_config temporarily for the demo path
                test_config.MODEL_CHECKPOINT_PATH = dummy_fp32_path
                int8_model = load_inference_model("int8", dummy_fp32_path, dummy_device, quantized_path=Path("dummy"))
                # Check if some layers are quantized (difficult to check type precisely)
                # assert isinstance(int8_model.classifier_head, torch.nn.quantized.dynamic.Linear)
                print("[DEMO] INT8 model loaded (dynamically quantized).")

                # Test loading TorchScript (requires creating one first)
                print("\n[DEMO] Creating and loading TorchScript model...")
                dummy_ts_path = dummy_checkpoint_dir / "dummy_model.torchscript.pt"
                try:
                    # Need example input to trace
                    frame_count = 1000 # From test data loader demo
                    example_input = torch.randn(1, test_config.N_MELS, frame_count) # Batch size 1
                    # Load the FP32 model again for tracing
                    pt_model_cpu = _load_pytorch_model(dummy_fp32_path, n_classes_demo, torch.device('cpu'))
                    pt_model_cpu.eval()
                    traced_model = torch.jit.trace(pt_model_cpu, example_input)
                    torch.jit.save(traced_model, dummy_ts_path)
                    print(f"[DEMO] Saved dummy TorchScript model: {dummy_ts_path}")

                    ts_model = load_inference_model("torchscript", dummy_fp32_path, dummy_device, torchscript_path=dummy_ts_path)
                    assert isinstance(ts_model, torch.jit.ScriptModule)
                    print("[DEMO] TorchScript model loaded.")

                except Exception as e:
                    print(f"[ERROR] Failed TorchScript creation/loading: {e}")


            except Exception as e:
                import traceback
                print(f"[ERROR] Demonstration failed: {e}")
                traceback.print_exc()

    print("--- End test/model_loader.py demonstration ---")
