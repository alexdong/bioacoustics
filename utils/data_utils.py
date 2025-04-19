# -*- coding: utf-8 -*-
"""Data loading utilities."""

import json
import os
import random
import typing
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from .config import (
    AUDIO_BASE_DIR,
    HOP_LENGTH,
    N_MELS,
    SEGMENT_SAMPLES,
    TARGET_SAMPLE_RATE,
)
from .logging import log


def load_class_mapping(filepath: Path) -> Dict[str, int]:
    """Loads the class name to integer index mapping from a JSON file."""
    assert filepath.exists(), f"💥 Class mapping file not found: {filepath}"
    try:
        with open(filepath, "r") as f:
            class_map = json.load(f)
        log("INFO", f"📚 Loaded {len(class_map)} classes from {filepath}")
        return class_map
    except json.JSONDecodeError as e:
        log("ERROR", f"💥 Failed to decode JSON from {filepath}: {e}")
        raise
    except Exception as e:
        log("ERROR", f"💥 Unexpected error loading class map {filepath}: {e}")
        raise


class BrazillianRandom100BirdSongDataset(Dataset[Tuple[torch.Tensor, int]]):
    """
    PyTorch Dataset for loading Brazilian bird song segments from the specific
    100-species split, applying augmentations, and converting to Mel spectrograms.
    """

    def __init__(
        self,
        split_file: Path,
        class_map: Dict[str, int],
        spectrogram_transform: torch.nn.Module,
        audio_dir: Path = AUDIO_BASE_DIR,  # Default from config
        sample_rate: int = TARGET_SAMPLE_RATE,
        segment_samples: int = SEGMENT_SAMPLES,
        augmentations: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    ) -> None:
        super().__init__()
        assert split_file.exists(), f"💥 Split file not found: {split_file}"
        self.split_file = split_file
        self.audio_dir = audio_dir
        self.class_map = class_map
        self.spectrogram_transform = spectrogram_transform
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.augmentations = augmentations
        self._expected_time_steps = int(np.ceil(self.segment_samples / HOP_LENGTH))

        log("INFO", f"🔍 Reading dataset split file: {self.split_file}")
        try:
            # Assuming tab-separated: recording_id.ogg\tspecies_name
            self.file_list = pd.read_csv(
                self.split_file, sep="\t", header=None, names=["filename", "species"],
            )
        except Exception as e:
            log("ERROR", f"💥 Failed to read split file {self.split_file}: {e}")
            raise
        log(
            "INFO",
            f"📊 Found {len(self.file_list)} potential samples in {self.split_file.name}",
        )

        # --- File Existence Check and Filtering ---
        initial_count = len(self.file_list)
        log("INFO", f"Checking existence of audio files in {self.audio_dir}...")
        # This can be slow for large datasets, consider optimizing if needed
        # (e.g., listdir once and check against the set)
        self.file_list["full_path"] = self.file_list["filename"].apply(
            lambda f: self.audio_dir / f,
        )
        exists_mask = self.file_list["full_path"].apply(lambda p: p.exists())
        self.file_list = self.file_list[exists_mask]
        filtered_count = len(self.file_list)

        if initial_count != filtered_count:
            num_missing = initial_count - filtered_count
            log(
                "WARN",
                f"⚠️ Dropped {num_missing} samples ({num_missing/initial_count:.1%}) because audio files were missing.",
            )
            # Optional: Log missing files if debugging is needed
            # missing_files = df[~exists_mask]['filename'].tolist()
            # log("DEBUG", f"Missing files example: {missing_files[:5]}")
        assert (
            filtered_count > 0
        ), f"💥 No valid audio files found for split {self.split_file.name} in {self.audio_dir}"

        # --- Pre-calculate labels ---
        log("INFO", "Mapping species to labels...")
        self.file_list["label"] = self.file_list["species"].map(self.class_map)
        if self.file_list["label"].isnull().any():
            missing_species = self.file_list[self.file_list["label"].isnull()][
                "species"
            ].unique()
            log(
                "ERROR",
                f"💥 Found samples with species not in class_map! Missing: {missing_species}",
            )
            # Decide whether to drop these or raise error
            # For now, let's drop them and issue a strong warning
            log("WARN", "Dropping samples with species not found in class map.")
            self.file_list = self.file_list.dropna(subset=["label"])
            assert (
                not self.file_list.empty
            ), "💥 All samples dropped due to missing species in class map!"

        # Convert label column to integer type after ensuring no NaNs
        self.file_list["label"] = self.file_list["label"].astype(int)

        log("INFO", f"✅ Dataset initialized with {len(self.file_list)} valid samples.")
        log("DEBUG", f"First 5 samples:\n{self.file_list.head()}")

    def __len__(self) -> int:
        return len(self.file_list)

    def _load_and_segment_audio(self, audio_path: Path) -> Optional[torch.Tensor]:
        """Loads, validates, and segments/pads a single audio file."""
        try:
            waveform, sr = torchaudio.load(audio_path)  # type: ignore
            waveform = typing.cast(torch.Tensor, waveform)
            sr = typing.cast(int, sr)

            assert (
                sr == self.sample_rate
            ), f"Sample rate mismatch! Expected {self.sample_rate}, got {sr} for {audio_path.name}"
            assert (
                waveform.ndim == 2
            ), f"Expected 2D waveform (channels, time), got {waveform.ndim}D for {audio_path.name}"
            assert (
                waveform.shape[0] == 1
            ), f"Expected mono audio (1 channel), got {waveform.shape[0]} channels for {audio_path.name}"

            waveform = waveform.squeeze(0)  # -> (time,)

            current_samples = waveform.shape[0]
            if current_samples == self.segment_samples:
                return waveform  # Perfect match

            elif current_samples < self.segment_samples:
                padding = self.segment_samples - current_samples
                pad_left = padding // 2
                pad_right = padding - pad_left
                waveform = torch.nn.functional.pad(
                    waveform, (pad_left, pad_right), mode="constant", value=0,
                )
                log(
                    "DEBUG",
                    f" Padded {audio_path.name} from {current_samples} to {self.segment_samples} samples.",
                )
                return waveform

            else:  # current_samples > self.segment_samples
                max_start = current_samples - self.segment_samples
                start_idx = random.randint(0, max_start)
                waveform = waveform[start_idx : start_idx + self.segment_samples]
                # log("DEBUG", f" Segmented {audio_path.name} from {current_samples} to {self.segment_samples} samples.") # Too verbose
                return waveform

        except FileNotFoundError:
            log("ERROR", f"💥 File not found during loading: {audio_path}")
            return None
        except AssertionError as e:
            log("ERROR", f"💥 Assertion failed loading {audio_path.name}: {e}")
            return None
        except Exception as e:
            log(
                "ERROR",
                f"💥 Unexpected error loading/segmenting {audio_path.name}: {e}",
            )
            return None  # Return None to signal failure

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Loads audio, processes, and returns (spectrogram, label)."""
        item = self.file_list.iloc[index]
        audio_path = item["full_path"]
        label_int = item["label"]  # Already validated and converted to int in __init__

        waveform = self._load_and_segment_audio(audio_path)

        # Handle loading failure gracefully
        if waveform is None:
            log(
                "WARN",
                f"Returning dummy data for index {index} ({audio_path.name}) due to loading error.",
            )
            # Return a dummy tensor and label 0 to avoid crashing the batch loading
            dummy_spec = torch.zeros(
                (1, N_MELS, self._expected_time_steps), dtype=torch.float32,
            )
            # Perhaps return a special label like -1 if loss function handles ignores? For now, 0.
            return dummy_spec, 0

        assert (
            waveform.shape[0] == self.segment_samples
        ), f"💥 Waveform shape incorrect after load/segment: {waveform.shape}, expected ({self.segment_samples},) for {audio_path.name}"

        # Apply augmentations (if training)
        waveform_np = waveform.numpy().astype(np.float32)
        if self.augmentations:
            try:
                waveform_aug_np = self.augmentations(
                    samples=waveform_np, sample_rate=self.sample_rate,
                )
                waveform = torch.from_numpy(waveform_aug_np.astype(np.float32))
            except Exception as e:
                log(
                    "ERROR",
                    f"💥 Augmentation failed for {audio_path.name}: {e}. Using original waveform.",
                )
                # Fallback to unaugmented waveform
                waveform = torch.from_numpy(waveform_np)
        else:
            waveform = torch.from_numpy(waveform_np)  # Ensure it's a tensor

        # Generate spectrogram
        try:
            # Move transforms to appropriate device *before* applying them if they have state
            # Assuming transforms are generally stateless or handled internally by PyTorch
            # self.spectrogram_transform.to(waveform.device) # Not strictly needed if transforms are on CPU
            spectrogram = self.spectrogram_transform(waveform)

            # Shape check after transforms: (batch=1, channel=1, n_mels, time_steps)
            assert (
                spectrogram.ndim == 4
            ), f"💥 Expected 4D spectrogram, got {spectrogram.ndim}D for {audio_path.name}"
            assert (
                spectrogram.shape[1] == 1
            ), f"💥 Expected 1 channel in spectrogram, got {spectrogram.shape[1]} for {audio_path.name}"
            assert (
                spectrogram.shape[2] == N_MELS
            ), f"💥 Expected {N_MELS} mel bins, got {spectrogram.shape[2]} for {audio_path.name}"
            # Allow slight variation in time steps due to potential edge effects in transforms
            assert (
                abs(spectrogram.shape[3] - self._expected_time_steps) <= 1
            ), f"💥 Unexpected time steps in spectrogram: {spectrogram.shape[3]}, expected ~{self._expected_time_steps} for {audio_path.name}"

            log(
                "DEBUG",
                f" Processed {audio_path.name}: Spec shape={spectrogram.shape}, Label={label_int}",
            )
            # Squeeze the batch dimension added by the Lambda layer in transform
            # We want (C, H, W) = (1, n_mels, time_steps) for Conv2D input
            return spectrogram.squeeze(0), label_int

        except Exception as e:
            log("ERROR", f"💥 Spectrogram generation failed for {audio_path.name}: {e}")
            # Return dummy data again
            dummy_spec = torch.zeros(
                (1, N_MELS, self._expected_time_steps), dtype=torch.float32,
            )
            return dummy_spec, 0


def get_dataloaders(
    class_map: Dict[str, int],
    spectrogram_transform: torch.nn.Module,
    augment_pipeline: Optional[Callable[[np.ndarray, int], np.ndarray]],
    batch_size: int,
    num_workers: Optional[int] = None,
    pin_memory: bool = False,
    split_dir: Path = DATASET_BASE_DIR,  # Default from config
) -> Dict[str, DataLoader[Any]]:
    """Creates train, validation, and test DataLoaders."""

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 4)  # Sensible default
        log(
            "INFO",
            f"Auto-detected {os.cpu_count()} CPUs. Using {num_workers} workers for DataLoaders.",
        )

    split_files = {
        "train": split_dir / "train_split.txt",
        "val": split_dir / "val_split.txt",
        "test": split_dir / "test_split.txt",
    }

    datasets = {
        "train": BrazillianRandom100BirdSongDataset(
            split_file=split_files["train"],
            class_map=class_map,
            spectrogram_transform=spectrogram_transform,
            augmentations=augment_pipeline,  # Only train uses augmentation
        ),
        "val": BrazillianRandom100BirdSongDataset(
            split_file=split_files["val"],
            class_map=class_map,
            spectrogram_transform=spectrogram_transform,
            augmentations=None,
        ),
        "test": BrazillianRandom100BirdSongDataset(
            split_file=split_files["test"],
            class_map=class_map,
            spectrogram_transform=spectrogram_transform,
            augmentations=None,
        ),
    }

    dataloaders = {}
    for split, ds in datasets.items():
        is_train = split == "train"
        # Use larger batch size for eval if desired, simple approach: same batch size
        current_batch_size = batch_size  # if is_train else batch_size * 2
        dataloaders[split] = DataLoader(
            ds,
            batch_size=current_batch_size,
            shuffle=is_train,  # Shuffle only training data
            num_workers=num_workers,
            pin_memory=pin_memory,
            # persistent_workers=True if num_workers > 0 else False, # Can speed up if workers are costly to init
            drop_last=is_train,  # Drop last incomplete batch only for training
        )
        log(
            "INFO",
            f"✅ {split.capitalize()} DataLoader created. Samples: {len(ds)}, Batches: {len(dataloaders[split])}",
        )

    return dataloaders
