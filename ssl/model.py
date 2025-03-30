# ssl/model.py
"""Defines the Masked Autoencoder (MAE) model for SSL."""


# Use config settings from the ssl directory
import config as ssl_config
import torch
import torch.nn as nn
from transformers import WhisperConfig, WhisperModel
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoderLayer,  # For decoder
)

# Type Alias
Tensor = torch.Tensor

# --- Top Level Function ---
def build_ssl_model() -> nn.Module:
    """Builds the MAE model including encoder and decoder."""
    print("[SSL MODEL] Building MAE model...")

    # 1. Load Base Whisper Encoder
    print(f"[SSL MODEL] Loading base Whisper encoder: openai/whisper-{ssl_config.WHISPER_MODEL_SIZE}")
    whisper_config = WhisperConfig.from_pretrained(f"openai/whisper-{ssl_config.WHISPER_MODEL_SIZE}")
    whisper_model = WhisperModel.from_pretrained(
        f"openai/whisper-{ssl_config.WHISPER_MODEL_SIZE}", config=whisper_config
    )
    encoder = whisper_model.get_encoder()
    encoder_embed_dim = encoder.config.d_model

    # TODO: Load custom base weights if specified in config
    if ssl_config.BASE_ENCODER_CHECKPOINT_PATH:
        print(f"[SSL MODEL] WARNING: Loading BASE encoder weights from {ssl_config.BASE_ENCODER_CHECKPOINT_PATH} - implement loading logic if needed.")
        # Implement loading logic similar to fine_tune/model.py if required

    # 2. Define Decoder
    print(f"[SSL MODEL] Defining Simple MAE Decoder (Depth: {ssl_config.DECODER_DEPTH}, Dim: {ssl_config.DECODER_EMBED_DIM})")
    decoder = SimpleMAEDecoder(
        encoder_embed_dim=encoder_embed_dim,
        decoder_embed_dim=ssl_config.DECODER_EMBED_DIM,
        decoder_depth=ssl_config.DECODER_DEPTH,
        decoder_num_heads=ssl_config.DECODER_NUM_HEADS,
        output_patch_dim=ssl_config.N_MELS # Predict N_MELS for each time frame
    )

    # 3. Combine into MAE Model
    mae_model = MaskedAutoencoderModel(encoder, decoder)

    print("[SSL MODEL] MAE model built successfully.")
    return mae_model

# --- Decoder Class ---
class SimpleMAEDecoder(nn.Module):
    """Simple Decoder for MAE based on Transformer blocks."""
    def __init__(
        self,
        encoder_embed_dim: int,
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        output_patch_dim: int, # Dimension of the output patch (e.g., N_MELS)
    ) -> None:
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)) # Learnable mask token

        # Simple positional embedding for the decoder sequence length
        # NOTE: Max length assumption - adjust if needed, or use sinusoidal
        # This assumes a fixed number of frames after encoder/masking? Risky.
        # A better approach might reuse/adapt encoder pos embedding or use sinusoidal.
        # For now, placeholder - Whisper's encoder handles pos embedding internally.
        # Let's assume we add pos embedding *after* combining encoder output & mask tokens.
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, MAX_FRAMES, decoder_embed_dim))

        # Stack of Transformer blocks (using Whisper's layer for consistency)
        # Need to create a dummy config for the decoder layers
        decoder_config = WhisperConfig( # Use WhisperConfig structure but adjust params
             d_model=decoder_embed_dim,
             encoder_attention_heads=decoder_num_heads,
             encoder_ffn_dim=decoder_embed_dim * 4, # Standard ratio
             # Other params like dropout can be added if needed
        )
        self.decoder_layers = nn.ModuleList([
            WhisperEncoderLayer(decoder_config) for _ in range(decoder_depth)
        ])

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, output_patch_dim, bias=True) # Predict N_MELS

        # Initialize mask token and potentially pos embeds
        torch.nn.init.normal_(self.mask_token, std=.02)
        # self.apply(self._init_weights) # Optional: Custom weight init

    # def _init_weights(self, m): ... # Weight init function if needed

    def forward(self, x: Tensor, masked_indices: Tensor) -> Tensor:
        """
        Forward pass of the decoder.

        Args:
            x: Tensor of shape (batch, total_frames, encoder_embed_dim) - Output from encoder.
            masked_indices: Boolean tensor (batch, total_frames) where True means masked.
                            Used to know where to insert mask tokens.
        """
        batch_size, total_frames, embed_dim = x.shape

        # Project encoder output features to decoder dimension
        x = self.decoder_embed(x)  # (batch, total_frames, decoder_dim)

        # Create mask token expanded to match batch size and embedding dimension
        mask_token_expanded = self.mask_token.expand(batch_size, 1, -1)  # (batch, 1, decoder_dim)

        # Create a properly expanded mask for broadcasting
        mask_expanded = masked_indices.unsqueeze(-1).to(x.dtype)  # (batch, total_frames, 1)

        # Replace masked positions with mask tokens
        # For positions where mask_expanded is 1, use mask_token
        # For positions where mask_expanded is 0, use encoder features
        # We need to expand mask_token to all masked positions
        mask_tokens = mask_token_expanded.repeat(1, total_frames, 1)  # (batch, total_frames, decoder_dim)
        x = x * (1 - mask_expanded) + mask_tokens * mask_expanded

        # Apply Transformer blocks
        for layer in self.decoder_layers:
            layer_outputs = layer(x)  # WhisperEncoderLayer returns tuple
            x = layer_outputs[0]  # Get hidden state

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)  # (batch, total_frames, output_patch_dim)

        return x

# --- MAE Model Class ---
class MaskedAutoencoderModel(nn.Module):
    """Main MAE model combining encoder and decoder."""
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # NOTE: Positional embeddings are handled within WhisperEncoder

    def forward(self, masked_features: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for MAE.

        Args:
            masked_features: Input Mel spectrogram with masked frames (B, N_MELS, N_FRAMES).
                             Zero-masking assumed from data loader.
            mask: Boolean tensor (B, N_FRAMES) indicating masked positions (True=masked).

        Returns:
            predictions: Reconstructed spectrogram frames (B, N_FRAMES, N_MELS).
        """
        # 1. Encode the masked input
        # Whisper encoder expects input shape (B, N_MELS, N_FRAMES)
        encoder_output = self.encoder(input_features=masked_features).last_hidden_state
        # encoder_output shape: (B, N_FRAMES_ENC, encoder_embed_dim)

        # Handle potential frame dimension mismatch between input and encoder output
        encoder_frame_count = encoder_output.shape[1]
        input_frame_count = mask.shape[1]

        if encoder_frame_count != input_frame_count:
            print(f"[WARNING] Frame count mismatch: encoder={encoder_frame_count}, input={input_frame_count}")
            # Two options:
            # 1. Resize the mask to match encoder output
            # 2. Resize encoder output to match mask
            # We'll choose option 1 as it's simpler

            # Interpolate mask to match encoder frame count
            # Convert boolean mask to float for interpolation
            mask_float = mask.float().unsqueeze(1)  # Add channel dim: (B, 1, N_FRAMES)

            # Use interpolate to resize mask
            resized_mask = torch.nn.functional.interpolate(
                mask_float,
                size=encoder_frame_count,
                mode='nearest'
            ).squeeze(1)  # Remove channel dim: (B, N_FRAMES_ENC)

            # Convert back to boolean
            mask = resized_mask > 0.5

        # 2. Decode - Pass encoder output and mask to decoder
        predictions = self.decoder(encoder_output, mask)

        # Validate output shape
        assert predictions.shape[0] == masked_features.shape[0], "Batch size mismatch"
        assert predictions.shape[1] == encoder_frame_count, "Frame count mismatch in output"
        assert predictions.shape[2] == ssl_config.N_MELS, "Mel dimension mismatch"

        return predictions

# --- Main Block for Testing/Demonstration ---
if __name__ == "__main__":
    print("--- Running ssl/model.py demonstration ---")

    # Dummy parameters
    batch_size_demo = 4
    n_mels_demo = ssl_config.N_MELS
    # Estimate frame count based on config (e.g., 5s @ 32kHz, hop 160 -> ~1000 frames)
    n_frames_demo = int(ssl_config.CHUNK_DURATION_SEC * ssl_config.TARGET_SAMPLE_RATE / ssl_config.HOP_LENGTH) + 1
    print(f"[DEMO] Estimated frame count: {n_frames_demo}")

    try:
        print("[DEMO] Building SSL MAE model...")
        # NOTE: This will download the base whisper model if not cached
        mae_model_demo = build_ssl_model()
        print("[DEMO] MAE Model built.") # Printing full model can be very long

        # Create dummy input
        dummy_masked_input = torch.randn(batch_size_demo, n_mels_demo, n_frames_demo)
        dummy_mask = torch.rand(batch_size_demo, n_frames_demo) > (1.0 - ssl_config.MASKING_RATIO) # Boolean mask
        print(f"\n[DEMO] Dummy masked input shape: {dummy_masked_input.shape}")
        print(f"[DEMO] Dummy mask shape: {dummy_mask.shape}")

        # Perform forward pass
        print("[DEMO] Performing forward pass...")
        mae_model_demo.eval() # Set to evaluation mode
        with torch.no_grad(): # Disable gradient calculation
            output_predictions = mae_model_demo(dummy_masked_input, dummy_mask)

        print(f"[DEMO] Output predictions shape: {output_predictions.shape}")
        # Expected shape: (batch, n_frames, n_mels) - Decoder output shape
        # assert output_predictions.shape == (batch_size_demo, n_frames_demo, n_mels_demo), "Output shape mismatch"
        print("[DEMO] NOTE: Output shape assertion commented out due to potential frame mismatch from Whisper encoder conv layers. Verify manually.")
        print("[DEMO] Forward pass completed.")

    except Exception as e:
        import traceback
        print(f"[ERROR] Demonstration failed: {e}")
        traceback.print_exc()

    print("--- End ssl/model.py demonstration ---")
