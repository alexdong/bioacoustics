# ssl/data_loader.py
"""Dataset class and functions for loading unlabeled audio for SSL."""

import random  # For selecting files if needed
from pathlib import Path

# Use config settings from the ssl directory
import config as ssl_config
import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Suppress torchaudio warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# Type Aliases
AudioTensor = torch.Tensor

# --- Top Level Function ---
def create_ssl_dataloader(
    audio_dir: Path,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Creates training dataloader for SSL."""
    print("[SSL DATA] Creating SSL DataLoader...")
    assert audio_dir.is_dir(), f"Unlabeled audio directory not found: {audio_dir}"

    # List all audio files (consider glob for specific extensions e.g., '*.ogg', '*.wav')
    # NOTE: This could be memory intensive if the directory is huge.
    # Consider using `Path.rglob` for recursive search.
    # Limit number of files for testing if needed.
    all_audio_files = list(audio_dir.glob("*.ogg")) # TODO: Adapt extension if needed
    print(f"[SSL DATA] Found {len(all_audio_files)} audio files for SSL.")
    assert len(all_audio_files) > 0, f"No audio files found in {audio_dir}"

    ssl_dataset = SSLDataset(all_audio_files)

    ssl_loader = DataLoader(
        ssl_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True, # Important for consistent batch sizes
        collate_fn=_ssl_collate_fn, # Custom collate to handle masking if needed here
    )

    print("[SSL DATA] SSL DataLoader created successfully.")
    return ssl_loader

# --- Masking Function ---
def _random_masking(
    features: Tensor, mask_ratio: float = 0.75,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Performs random masking on the time dimension of a spectrogram.

    Args:
        features: Input Mel spectrogram tensor (N_MELS, N_FRAMES).
        mask_ratio: Proportion of frames to mask.

    Returns:
        Tuple of:
            - masked_features: Spectrogram with masked frames zeroed out.
            - original_features: The input features tensor (for target).
            - mask: Boolean tensor (N_FRAMES), True where frames are masked.
    """
    n_mels, n_frames = features.shape
    num_masked = int(mask_ratio * n_frames)
    assert num_masked < n_frames, "Mask ratio too high, cannot mask all frames"

    # Generate random indices to mask
    indices = np.random.permutation(n_frames)[:num_masked]

    # Create mask: True indicates masked
    mask = torch.zeros(n_frames, dtype=torch.bool)
    mask[indices] = True

    # Create masked features (simple zero-masking)
    masked_features = features.clone()
    masked_features[:, mask] = 0.0 # Zero out the masked time frames across all mel bins

    return masked_features, features, mask # Return masked, original, and the mask itself


# --- Dataset Class ---
class SSLDataset(Dataset[tuple[AudioTensor, AudioTensor, Tensor]]):
    """PyTorch Dataset for SSL pre-training on unlabeled audio."""

    def __init__(
        self,
        audio_files: list[Path],
    ) -> None:
        super().__init__()
        self.audio_files = audio_files
        self.target_sr = ssl_config.TARGET_SAMPLE_RATE
        self.chunk_samples = int(ssl_config.CHUNK_DURATION_SEC * self.target_sr)
        self.masking_ratio = ssl_config.MASKING_RATIO

        # Initialize Mel Spectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024, # Example, match fine-tuning?
            win_length=None,
            hop_length=ssl_config.HOP_LENGTH,
            n_mels=ssl_config.N_MELS,
            power=2.0,
        )
        print(f"[SSL DATASET] Initialized SSL Dataset with {len(self.audio_files)} files.")

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, index: int) -> tuple[AudioTensor, AudioTensor, Tensor]:
        """Loads audio, processes, applies masking."""
        audio_path = self.audio_files[index]
        # Minimal check, assuming path validity from initial scan
        # assert audio_path.is_file(), f"Audio file disappeared?: {audio_path}"

        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"[ERROR] Failed to load audio: {audio_path}. Error: {e}")
            # Return dummy data of correct shape might be problematic downstream. Fail fast.
            raise RuntimeError(f"Failed to load audio: {audio_path}") from e

        # Handle channels and resampling (same as fine-tuning loader)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        waveform = waveform.squeeze(0)

        # Pad or truncate (same as fine-tuning loader)
        current_len = waveform.shape[0]
        if current_len < self.chunk_samples:
            padding = self.chunk_samples - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_len > self.chunk_samples:
            # Random crop might be better for SSL than fixed truncation
            start = random.randint(0, current_len - self.chunk_samples)
            waveform = waveform[start : start + self.chunk_samples]
            # waveform = waveform[:self.chunk_samples] # Fixed truncation

        assert waveform.shape[0] == self.chunk_samples, "Padding/crop failed"

        # Compute Mel Spectrogram & Log scale (same as fine-tuning loader)
        mel_spec = self.mel_spectrogram(waveform.unsqueeze(0)).squeeze(0)
        mel_spec = torch.log(mel_spec + 1e-7)

        # --- Apply Masking ---
        masked_mel_spec, original_mel_spec, mask_indices = _random_masking(
            mel_spec, self.masking_ratio,
        )

        # Type checks
        assert isinstance(masked_mel_spec, torch.Tensor)
        assert isinstance(original_mel_spec, torch.Tensor)
        assert isinstance(mask_indices, torch.Tensor)
        assert masked_mel_spec.shape == original_mel_spec.shape
        assert mask_indices.ndim == 1

        return masked_mel_spec, original_mel_spec, mask_indices

# --- Custom Collate Function (Optional but Recommended) ---
def _ssl_collate_fn(
    batch: list[tuple[AudioTensor, AudioTensor, Tensor]],
) -> tuple[AudioTensor, AudioTensor, Tensor]:
    """Pads sequences within a batch and stacks tensors."""
    # Unzip the batch
    masked_specs, original_specs, masks = zip(*batch)

    # Pad Mel spectrograms (time dimension might vary slightly if transforms change it)
    # Assuming fixed length due to padding/cropping in __getitem__ for now.
    # If variable length: use torch.nn.utils.rnn.pad_sequence
    masked_specs_padded = torch.stack(masked_specs, dim=0)
    original_specs_padded = torch.stack(original_specs, dim=0)

    # Pad boolean masks
    masks_padded = torch.stack(masks, dim=0) # Assuming masks have same length

    return masked_specs_padded, original_specs_padded, masks_padded


# --- Main Block for Testing/Demonstration ---
if __name__ == "__main__":
    print("--- Running ssl/data_loader.py demonstration ---")

    # Create dummy audio files in SSL temp dir
    dummy_ssl_audio_dir = ssl_config.SSL_TEMP_DIR / "dummy_ssl_audio"
    dummy_ssl_audio_dir.mkdir(parents=True, exist_ok=True)
    num_dummy_files = 10
    dummy_fpaths = []
    for i in range(num_dummy_files):
        fname = f"ssl_dummy_{i}.ogg"
        fpath = dummy_ssl_audio_dir / fname
        dummy_fpaths.append(fpath)
        if not fpath.exists():
            dummy_wav = torch.randn((1, ssl_config.TARGET_SAMPLE_RATE * 4)) # 4 sec random
            torchaudio.save(fpath, dummy_wav, ssl_config.TARGET_SAMPLE_RATE)
            # print(f"[DEMO] Created dummy SSL audio file: {fpath}")
    print(f"[DEMO] Ensured {num_dummy_files} dummy SSL audio files exist in {dummy_ssl_audio_dir}")

    try:
        print("[DEMO] Creating dummy SSL Dataset...")
        ssl_dataset_demo = SSLDataset(dummy_fpaths)
        print(f"[DEMO] SSL Dataset length: {len(ssl_dataset_demo)}")

        print("[DEMO] Getting first item...")
        masked_demo, orig_demo, mask_demo = ssl_dataset_demo[0]
        print(f"[DEMO]   Masked Spec shape: {masked_demo.shape}")
        print(f"[DEMO]   Original Spec shape: {orig_demo.shape}")
        print(f"[DEMO]   Mask shape: {mask_demo.shape}") # Should be [N_FRAMES]
        print(f"[DEMO]   Number of masked frames: {mask_demo.sum().item()}")
        print(f"[DEMO]   Mask ratio check: {mask_demo.sum().item() / mask_demo.shape[0]:.2f} (Target: {ssl_config.MASKING_RATIO})")

        print("[DEMO] Creating dummy SSL DataLoader...")
        ssl_loader_demo = create_ssl_dataloader(
            dummy_ssl_audio_dir, batch_size=4, num_workers=0,
        )

        print("[DEMO] Getting first batch...")
        batch_masked, batch_orig, batch_mask = next(iter(ssl_loader_demo))
        print(f"[DEMO]   Batch Masked shape: {batch_masked.shape}") # Should be [BATCH, N_MELS, N_FRAMES]
        print(f"[DEMO]   Batch Original shape: {batch_orig.shape}") # Should be [BATCH, N_MELS, N_FRAMES]
        print(f"[DEMO]   Batch Mask shape: {batch_mask.shape}")     # Should be [BATCH, N_FRAMES]

    except Exception as e:
        import traceback
        print(f"[ERROR] Demonstration failed: {e}")
        traceback.print_exc() # Print full traceback for debugging

    print("--- End ssl/data_loader.py demonstration ---")
