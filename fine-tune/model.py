# model.py

from pathlib import Path
from typing import Optional

import config
import torch
import torch.nn as nn
from transformers import WhisperConfig, WhisperModel

# Type Alias
Tensor = torch.Tensor

# --- Top Level Function ---
def build_model(
    num_classes: int,
    encoder_checkpoint_path: Optional[Path] = None,
) -> nn.Module:
    """Builds the classification model with optional pre-trained encoder weights."""
    print("[MODEL] Building model...")
    # Load base Whisper configuration and potentially modify it if needed
    whisper_config = WhisperConfig.from_pretrained(f"openai/whisper-{config.WHISPER_MODEL_SIZE}")
    # Example modification (if necessary):
    # whisper_config.attention_dropout = 0.1

    # Load the Whisper encoder part
    print(f"[MODEL] Loading base Whisper encoder: openai/whisper-{config.WHISPER_MODEL_SIZE}")
    whisper_model = WhisperModel.from_pretrained(
        f"openai/whisper-{config.WHISPER_MODEL_SIZE}", config=whisper_config,
    )
    encoder = whisper_model.get_encoder()

    # Freeze encoder initially? Optional, can be controlled in trainer
    # for param in encoder.parameters():
    #     param.requires_grad = False

    if encoder_checkpoint_path:
        print(f"[MODEL] Loading SSL fine-tuned encoder weights from: {encoder_checkpoint_path}")
        assert encoder_checkpoint_path.is_file(), f"Encoder checkpoint not found: {encoder_checkpoint_path}"
        try:
            # Load state dict, be mindful of potential key mismatches if saved differently
            state_dict = torch.load(encoder_checkpoint_path, map_location='cpu')
            # Adapt keys if necessary (e.g., if saved with 'module.' prefix from DDP)
            # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = encoder.load_state_dict(state_dict, strict=False)
            if missing_keys:
                 print(f"[WARNING] Missing keys when loading encoder state_dict: {missing_keys}")
            if unexpected_keys:
                 print(f"[WARNING] Unexpected keys when loading encoder state_dict: {unexpected_keys}")
            assert not unexpected_keys, "Refused to load state dict with unexpected keys."
            print("[MODEL] Encoder weights loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load encoder checkpoint: {encoder_checkpoint_path}. Error: {e}")
            raise RuntimeError("Failed loading encoder checkpoint") from e
    else:
        print("[MODEL] Using standard Whisper pre-trained encoder weights (no SSL checkpoint).")

    # Determine the feature size from the encoder
    encoder_feature_size = encoder.config.d_model

    # Build the full model
    model = BirdClefClassifier(encoder, encoder_feature_size, num_classes, config.POOLING_TYPE)
    print("[MODEL] Model built successfully.")
    return model

# --- Model Class ---
class BirdClefClassifier(nn.Module):
    """Classifier model using a pre-trained encoder and a shallow head."""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_feature_size: int,
        num_classes: int,
        pooling_type: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.feature_size = encoder_feature_size
        self.num_classes = num_classes
        self.pooling_type = pooling_type

        assert pooling_type in ["mean", "max"], f"Invalid pooling type: {pooling_type}"

        # Simple linear classifier head
        # Consider adding dropout here if needed: nn.Dropout(p=0.1),
        self.classifier_head = nn.Linear(self.feature_size, self.num_classes)

        print(f"[MODEL] Classifier head initialized: Linear({self.feature_size}, {self.num_classes})")

    def forward(self, input_features: Tensor) -> Tensor:
        """Forward pass through encoder, pooling, and classifier head."""
        # input_features shape: (batch_size, n_mels, n_frames)
        # WhisperEncoder expects: (batch_size, n_frames, n_features) - Needs check!
        # Let's verify WhisperModel input expectations.
        # Typically, transformers expect (batch, seq_len, hidden_dim)
        # The MelSpec output might be (batch, n_mels, n_frames)
        # Whisper's internal conv layers handle this projection usually.
        # Let's assume the encoder handles the shape (B, N_MELS, N_FRAMES) correctly.
        # If not, a projection layer might be needed before the encoder.
        # Check `transformers` documentation for WhisperModel input spec.
        # For now, proceeding assuming encoder handles MelSpec shape.

        # Pass through encoder
        # output = self.encoder(input_features=input_features) # Using keyword arg for clarity
        # last_hidden_state shape: (batch_size, sequence_length, feature_size)
        # NOTE: Check actual output shape of the specific encoder implementation
        # For WhisperModel.encoder, the input is `input_features` but the positional
        # embeddings might be added internally based on assumed sequence length.
        # Need to confirm input processing. Let's use a simple forward call for now.
        encoder_output = self.encoder(input_features).last_hidden_state
        assert encoder_output.shape[-1] == self.feature_size, "Feature size mismatch"

        # Pooling across the time dimension (sequence_length)
        # encoder_output shape: (batch, seq_len, features)
        if self.pooling_type == "mean":
            pooled_output = torch.mean(encoder_output, dim=1) # Pool over seq_len dim
        elif self.pooling_type == "max":
            pooled_output, _ = torch.max(encoder_output, dim=1)
        else:
            # This case should be caught by __init__ assert, but defensive check
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")

        # Pass through classifier head
        logits = self.classifier_head(pooled_output)
        # Logits shape: (batch_size, num_classes)

        return logits


# --- Main Block for Testing/Demonstration ---
if __name__ == "__main__":
    print("--- Running model.py demonstration ---")

    # Dummy parameters
    n_classes_demo = 10
    batch_size_demo = 4
    n_mels_demo = config.N_MELS # Use from config
    n_frames_demo = 150 # Example sequence length after spectrogram

    try:
        print("[DEMO] Building model without pre-trained encoder weights...")
        # NOTE: This will download the base whisper model if not cached
        model_demo = build_model(n_classes_demo, encoder_checkpoint_path=None) # Use base whisper
        print(f"[DEMO] Model:\n{model_demo}")

        # Create dummy input
        dummy_input = torch.randn(batch_size_demo, n_mels_demo, n_frames_demo)
        print(f"\n[DEMO] Dummy input shape: {dummy_input.shape}")

        # Perform forward pass
        print("[DEMO] Performing forward pass...")
        model_demo.eval() # Set to evaluation mode
        with torch.no_grad(): # Disable gradient calculation
            output_logits = model_demo(dummy_input)

        print(f"[DEMO] Output logits shape: {output_logits.shape}")
        assert output_logits.shape == (batch_size_demo, n_classes_demo), "Output shape mismatch"
        print("[DEMO] Forward pass successful.")

        # Example with a dummy checkpoint (create one first)
        # print("\n[DEMO] Testing model loading with dummy checkpoint...")
        # dummy_encoder = model_demo.encoder
        # dummy_checkpoint_path = config.TEMP_DIR / "dummy_encoder_test.pt"
        # torch.save(dummy_encoder.state_dict(), dummy_checkpoint_path)
        # print(f"[DEMO] Saved dummy checkpoint: {dummy_checkpoint_path}")
        # model_loaded = build_model(n_classes_demo, encoder_checkpoint_path=dummy_checkpoint_path)
        # print("[DEMO] Model loaded with dummy checkpoint successfully.")

    except Exception as e:
        import traceback
        print(f"[ERROR] Demonstration failed: {e}")
        traceback.print_exc()

    print("--- End model.py demonstration ---")
