# data_loader.py
"""Dataset class and functions for loading audio data."""

from pathlib import Path

import config  # Import config variables
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

# Suppress torchaudio warnings about sox/ffmpeg backends if desired
# warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# Type Aliases
AudioTensor = torch.Tensor
LabelTensor = torch.Tensor


# --- Top Level Function ---
def create_dataloaders(
    metadata_path: Path,
    audio_dir: Path,
    species_list: list[str],
    batch_size: int,
    num_workers: int,
    val_split: float = 0.2,
    random_state: int = 42,
) -> tuple[DataLoader, DataLoader | None]:
    """
    Creates training and validation dataloaders.

    Args:
        metadata_path: Path to CSV file containing audio metadata
        audio_dir: Directory containing audio files
        species_list: List of species names for classification
        batch_size: Batch size for dataloader
        num_workers: Number of worker processes for dataloader
        val_split: Fraction of data to use for validation (0-1)
        random_state: Random seed for reproducible train/val splits
    """
    print("[DATA] Creating DataLoaders...")
    assert metadata_path.is_file(), f"Metadata file not found: {metadata_path}"
    assert audio_dir.is_dir(), f"Audio directory not found: {audio_dir}"

    full_df = pd.read_csv(metadata_path)

    # Implement train/validation split
    if val_split > 0:
        # Stratified split to maintain class distribution
        from sklearn.model_selection import train_test_split

        # Get stratification labels (primary_label)
        strat_labels = full_df["primary_label"]

        # Perform split
        train_df, val_df = train_test_split(
            full_df,
            test_size=val_split,
            random_state=random_state,
            stratify=strat_labels,
        )

        # Reset indices
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
    else:
        # Use all data for training
        train_df = full_df
        val_df = None

    print(f"[DATA] Training samples: {len(train_df)}")
    if val_df is not None:
        print(f"[DATA] Validation samples: {len(val_df)}")

    train_dataset = BirdClefDataset(train_df, audio_dir, species_list)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Usually good for GPU training
        drop_last=True,  # Drop last incomplete batch
    )

    val_loader = None
    if val_df is not None:
        val_dataset = BirdClefDataset(val_df, audio_dir, species_list)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffle for validation
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    print("[DATA] DataLoaders created successfully.")
    return train_loader, val_loader


# --- Dataset Class ---
class BirdClefDataset(Dataset[tuple[AudioTensor, LabelTensor]]):
    """PyTorch Dataset for BirdCLEF audio classification."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        audio_dir: Path,
        species_list: list[str],
    ) -> None:
        super().__init__()
        self.metadata = metadata_df
        self.audio_dir = audio_dir
        self.species_list = species_list
        self.species_to_idx = {name: i for i, name in enumerate(species_list)}
        self.num_classes = len(species_list)
        self.target_sr = config.TARGET_SAMPLE_RATE
        self.chunk_samples = int(config.CHUNK_DURATION_SEC * self.target_sr)

        # Initialize Mel Spectrogram transform (can be done here or in __getitem__)
        # Doing it here avoids re-creation on every call
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024,  # Example, adjust as needed
            win_length=None,  # Defaults to n_fft
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            power=2.0,  # Power spectrogram
        )
        print(f"[DATASET] Initialized Dataset with {len(self.metadata)} samples.")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[AudioTensor, LabelTensor]:
        """Loads audio, processes, and returns spectrogram and labels."""
        row = self.metadata.iloc[index]
        # TODO: Adapt filename generation based on metadata.csv columns
        filename = row["filename"]
        primary_label = row["primary_label"]
        # secondary_labels = eval(row['secondary_labels']) # Assuming list string format
        # TODO: Handle secondary labels if needed for multi-label targets

        audio_path = self.audio_dir / filename
        assert audio_path.is_file(), f"Audio file not found: {audio_path}"

        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"[ERROR] Failed to load audio: {audio_path}. Error: {e}")
            # Return dummy data or raise error? Let's fail explicitly.
            raise RuntimeError(f"Failed to load audio: {audio_path}") from e

        # Handle channels (convert to mono) and resampling
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        # Ensure waveform is 1D
        waveform = waveform.squeeze(0)

        # Pad or truncate to target chunk length
        current_len = waveform.shape[0]
        if current_len < self.chunk_samples:
            padding = self.chunk_samples - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_len > self.chunk_samples:
            # TODO: Choose strategy: truncate start/end, random crop?
            # Simple truncation from start for now
            waveform = waveform[: self.chunk_samples]

        assert waveform.shape[0] == self.chunk_samples, "Padding/truncation failed"

        # Compute Mel Spectrogram
        # Add channel dim temporarily: (1, T) -> (1, N_MELS, N_FRAMES)
        mel_spec = self.mel_spectrogram(waveform.unsqueeze(0))
        # Remove channel dim: (N_MELS, N_FRAMES)
        mel_spec = mel_spec.squeeze(0)

        # Log-scale Mel Spectrogram (common practice)
        # Add epsilon for numerical stability
        mel_spec = torch.log(mel_spec + 1e-7)

        # Create label tensor (multi-hot encoding if multiple labels allowed)
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        if primary_label in self.species_to_idx:
            label_idx = self.species_to_idx[primary_label]
            label[label_idx] = 1.0
        # TODO: Add secondary labels if needed

        # Type check (optional but good practice)
        assert isinstance(mel_spec, torch.Tensor), "Mel spec is not a Tensor"
        assert isinstance(label, torch.Tensor), "Label is not a Tensor"
        assert mel_spec.ndim == 2, f"Expected 2D Mel spec, got {mel_spec.ndim}"
        assert label.shape == (self.num_classes,), "Label shape mismatch"

        return mel_spec, label


# --- Main Block for Testing/Demonstration ---
if __name__ == "__main__":
    print("--- Running data_loader.py demonstration ---")

    # Use config for paths, create dummy files if they don't exist
    dummy_audio_dir = config.TEMP_DIR / "dummy_audio"
    dummy_audio_dir.mkdir(parents=True, exist_ok=True)

    dummy_metadata_data = {
        "filename": ["bird1.ogg", "bird2.ogg", "mixed.ogg"],
        "primary_label": ["sp_a1", "sp_b1", "sp_a2"],
        # 'secondary_labels': [[], ['sp_a1'], []] # Example
    }
    dummy_meta_df = pd.DataFrame(dummy_metadata_data)
    dummy_meta_path = config.TEMP_DIR / "dummy_metadata.csv"
    dummy_meta_df.to_csv(dummy_meta_path, index=False)
    print(f"[DEMO] Created dummy metadata file: {dummy_meta_path}")

    # Create dummy audio files (silent)
    sample_rate_orig = 16000
    duration_orig = 6  # seconds
    num_frames_orig = sample_rate_orig * duration_orig
    for fname in dummy_metadata_data["filename"]:
        fpath = dummy_audio_dir / fname
        if not fpath.exists():  # Avoid rewriting if run multiple times
            dummy_wav = torch.zeros((1, num_frames_orig))
            torchaudio.save(fpath, dummy_wav, sample_rate_orig)
            print(f"[DEMO] Created dummy audio file: {fpath}")

    dummy_species = ["sp_a1", "sp_a2", "sp_b1", "sp_c1"]  # Needs to match taxonomy

    try:
        print("[DEMO] Creating dummy Dataset...")
        dataset = BirdClefDataset(dummy_meta_df, dummy_audio_dir, dummy_species)
        print(f"[DEMO] Dataset length: {len(dataset)}")

        print("[DEMO] Getting first item...")
        mel_spec_demo, label_demo = dataset[0]
        print(
            f"[DEMO]   Mel Spec shape: {mel_spec_demo.shape}",
        )  # Should be [N_MELS, N_FRAMES]
        print(f"[DEMO]   Label shape: {label_demo.shape}")  # Should be [num_species]
        print(f"[DEMO]   Label: {label_demo}")

        print("[DEMO] Creating dummy DataLoader...")
        dataloader, _ = create_dataloaders(
            dummy_meta_path,
            dummy_audio_dir,
            dummy_species,
            batch_size=2,
            num_workers=0,
        )
        print("[DEMO] Getting first batch...")
        batch_mels, batch_labels = next(iter(dataloader))
        print(
            f"[DEMO]   Batch Mels shape: {batch_mels.shape}",
        )  # Should be [BATCH, N_MELS, N_FRAMES]
        print(
            f"[DEMO]   Batch Labels shape: {batch_labels.shape}",
        )  # Should be [BATCH, num_species]

    except Exception as e:
        import traceback

        print(f"[ERROR] Demonstration failed: {e}")
        traceback.print_exc()  # Print full traceback for debugging

    print("--- End data_loader.py demonstration ---")
