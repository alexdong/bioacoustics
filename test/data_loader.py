# test/data_loader.py
"""Dataset class and functions for loading test audio data."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

# Use config settings from the test directory
import config as test_config
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

# Type Aliases
AudioTensor = torch.Tensor
RowID = str

# --- Top Level Function ---
def create_test_dataloader(
    test_audio_dir: Path,
    metadata_path: Path | None, # Can be None if we just glob audio files
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, List[RowID]]:
    """Creates DataLoader for test audio and extracts row IDs."""
    print("[TEST DATA] Creating Test DataLoader...")
    assert test_audio_dir.is_dir(), f"Test audio directory not found: {test_audio_dir}"

    test_metadata: List[Dict[str, Any]] = []
    row_ids: List[RowID] = []

    if metadata_path:
        # Assume metadata defines the chunks (e.g., sample_submission.csv)
        assert metadata_path.is_file(), f"Test metadata file not found: {metadata_path}"
        test_df = pd.read_csv(metadata_path)
        # Expect columns like 'row_id', potentially 'filename', 'seconds' or similar
        # Example format: row_id = filename_seconds (e.g., 'soundscape_12345_5')
        assert 'row_id' in test_df.columns, "Test metadata must contain 'row_id' column"

        # Extract info needed for TestDataset
        for _, row in test_df.iterrows():
            row_id = row['row_id']
            try:
                # Infer filename and end_time from row_id (adjust parsing logic!)
                parts = row_id.split('_')
                filename_stem = "_".join(parts[:-1])
                end_time_sec = int(parts[-1])
                # Assume files are .ogg (TODO: make configurable or detect)
                audio_filename = f"{filename_stem}.ogg"
                start_time_sec = end_time_sec - test_config.CHUNK_DURATION_SEC
                test_metadata.append({
                    "row_id": row_id,
                    "filename": audio_filename,
                    "start_sec": start_time_sec,
                    "end_sec": end_time_sec,
                })
                row_ids.append(row_id)
            except Exception as e:
                 print(f"[ERROR] Could not parse row_id: {row_id}. Error: {e}. Skipping.")
                 continue # Skip rows that can't be parsed

    else:
        # Glob directory and create 5-second chunks manually
        print(f"[WARNING] No test metadata provided. Globbing {test_audio_dir} and creating 5s chunks.")
        # This part needs careful implementation based on competition rules
        # Example: Iterate through files, get duration, create row_ids and metadata dicts
        all_test_files = list(test_audio_dir.glob("*.ogg")) # Adjust extension
        for audio_path in all_test_files:
            try:
                 info = torchaudio.info(str(audio_path))
                 duration_sec = info.num_frames / info.sample_rate
                 num_chunks = int(np.ceil(duration_sec / test_config.CHUNK_DURATION_SEC))
                 for i in range(num_chunks):
                     end_time_sec = (i + 1) * test_config.CHUNK_DURATION_SEC
                     # Ensure end_time doesn't exceed actual duration if needed precisely
                     # end_time_sec = min(end_time_sec, duration_sec) # This makes last chunk shorter
                     start_time_sec = i * test_config.CHUNK_DURATION_SEC
                     row_id = f"{audio_path.stem}_{int(end_time_sec)}" # Construct row_id
                     test_metadata.append({
                        "row_id": row_id,
                        "filename": audio_path.name,
                        "start_sec": start_time_sec,
                        "end_sec": end_time_sec, # End of the 5s window
                     })
                     row_ids.append(row_id)
            except Exception as e:
                 print(f"[ERROR] Failed processing test file {audio_path}: {e}")
                 continue

    assert len(test_metadata) > 0, "No test samples found or processed."
    print(f"[TEST DATA] Total test chunks to process: {len(test_metadata)}")

    test_dataset = TestDataset(test_metadata, test_audio_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # DO NOT shuffle test data
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print("[TEST DATA] Test DataLoader created successfully.")
    return test_loader, row_ids


# --- Dataset Class ---
class TestDataset(Dataset[Tuple[AudioTensor, RowID]]):
    """PyTorch Dataset for loading test audio chunks."""

    def __init__(
        self,
        metadata: List[Dict[str, Any]], # List of {'row_id', 'filename', 'start_sec', 'end_sec'}
        audio_dir: Path,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.audio_dir = audio_dir
        self.target_sr = test_config.TARGET_SAMPLE_RATE
        self.chunk_duration_sec = test_config.CHUNK_DURATION_SEC
        self.chunk_samples = int(self.chunk_duration_sec * self.target_sr)

        # Initialize Mel Spectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024, # Example, match fine-tuning
            win_length=None,
            hop_length=test_config.HOP_LENGTH,
            n_mels=test_config.N_MELS,
            power=2.0,
        )
        print(f"[TEST DATASET] Initialized Test Dataset with {len(self.metadata)} chunks.")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple[AudioTensor, RowID]:
        """Loads a specific 5-second audio chunk and returns its spectrogram and row_id."""
        row = self.metadata[index]
        row_id = row['row_id']
        filename = row['filename']
        start_sec = row['start_sec']
        row['end_sec'] # This might slightly differ from start_sec + 5 if duration wasn't multiple of 5

        audio_path = self.audio_dir / filename
        assert audio_path.is_file(), f"Test audio file not found: {audio_path}"

        # Calculate start and end frames
        start_frame = int(start_sec * self.target_sr)
        num_frames_to_load = self.chunk_samples # Load exactly 5 seconds worth

        try:
            # Load only the required segment
            waveform, sr = torchaudio.load(
                audio_path, frame_offset=start_frame, num_frames=num_frames_to_load,
            )
        except Exception as e:
            # Handle cases where the file is shorter than expected or other load errors
            print(f"[WARNING] Failed loading segment for {row_id} from {audio_path}. Error: {e}. Returning silence.")
            # Return silent audio of the correct length
            waveform = torch.zeros((1, self.chunk_samples))
            sr = self.target_sr # Assume target sr for silence

        # Handle channels and resampling (should match training)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.target_sr:
            # This shouldn't happen often if source SR is consistent, but good to check
            print(f"[WARNING] Resampling needed for {row_id} (SR: {sr}).")
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        waveform = waveform.squeeze(0)

        # Pad if loaded segment was shorter than requested (e.g., end of file)
        current_len = waveform.shape[0]
        if current_len < self.chunk_samples:
            padding = self.chunk_samples - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Ensure exact length
        assert waveform.shape[0] == self.chunk_samples, f"Waveform length incorrect for {row_id}"

        # Compute Mel Spectrogram & Log scale (should match training)
        mel_spec = self.mel_spectrogram(waveform.unsqueeze(0)).squeeze(0)
        mel_spec = torch.log(mel_spec + 1e-7)

        # Type check
        assert isinstance(mel_spec, torch.Tensor), "Mel spec is not a Tensor"
        assert mel_spec.ndim == 2, f"Expected 2D Mel spec, got {mel_spec.ndim}"

        return mel_spec, row_id


# --- Main Block for Testing/Demonstration ---
if __name__ == "__main__":
    print("--- Running test/data_loader.py demonstration ---")

    # Create dummy test audio dir and file
    dummy_test_audio_dir = Path("/tmp/birdclef_test/dummy_test_audio") # Use /tmp
    dummy_test_audio_dir.mkdir(parents=True, exist_ok=True)
    dummy_audio_fname = "soundscape_999.ogg"
    dummy_audio_path = dummy_test_audio_dir / dummy_audio_fname
    if not dummy_audio_path.exists():
        sr_orig = test_config.TARGET_SAMPLE_RATE
        duration_orig = 12 # seconds -> should yield 3 chunks (0-5, 5-10, 10-15)
        num_frames_orig = sr_orig * duration_orig
        dummy_wav = torch.randn((1, num_frames_orig))
        torchaudio.save(dummy_audio_path, dummy_wav, sr_orig)
        print(f"[DEMO] Created dummy test audio file: {dummy_audio_path}")

    # Create dummy metadata (sample submission style)
    dummy_metadata_data = {
        'row_id': [f"{dummy_audio_path.stem}_5", f"{dummy_audio_path.stem}_10", f"{dummy_audio_path.stem}_15"],
        # Add dummy columns for other species if mimicking sample submission
        'species1': [0.5] * 3, 'species2': [0.5] * 3,
    }
    dummy_meta_df = pd.DataFrame(dummy_metadata_data)
    dummy_meta_path = Path("/tmp/birdclef_test/dummy_sample_submission.csv")
    dummy_meta_df.to_csv(dummy_meta_path, index=False)
    print(f"[DEMO] Created dummy test metadata file: {dummy_meta_path}")

    try:
        print("[DEMO] Creating Test DataLoader using metadata...")
        dataloader_meta, row_ids_meta = create_test_dataloader(
            dummy_test_audio_dir, dummy_meta_path, batch_size=2, num_workers=0,
        )
        print(f"[DEMO]   Row IDs from metadata: {row_ids_meta}")
        assert len(row_ids_meta) == 3

        print("[DEMO] Getting first batch from metadata loader...")
        batch_mels_meta, batch_ids_meta = next(iter(dataloader_meta))
        print(f"[DEMO]   Batch Mels shape: {batch_mels_meta.shape}") # Should be [BATCH, N_MELS, N_FRAMES]
        print(f"[DEMO]   Batch Row IDs: {batch_ids_meta}")

        print("\n[DEMO] Creating Test DataLoader by globbing directory...")
        # Temporarily remove dummy metadata path to trigger globbing
        dummy_meta_path.unlink()
        dataloader_glob, row_ids_glob = create_test_dataloader(
            dummy_test_audio_dir, None, batch_size=2, num_workers=0,
        )
        print(f"[DEMO]   Row IDs from globbing: {row_ids_glob}")
        # Should still be 3 if duration=12, CHUNK_DURATION=5
        # assert len(row_ids_glob) == 3

        print("[DEMO] Getting first batch from glob loader...")
        batch_mels_glob, batch_ids_glob = next(iter(dataloader_glob))
        print(f"[DEMO]   Batch Mels shape: {batch_mels_glob.shape}")
        print(f"[DEMO]   Batch Row IDs: {batch_ids_glob}")


    except Exception as e:
        import traceback
        print(f"[ERROR] Demonstration failed: {e}")
        traceback.print_exc()

    print("--- End test/data_loader.py demonstration ---")
