import argparse
import json
import logging
import pathlib
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_filename(filepath):
    """
    Parses the filename to extract species, recording ID, and segment number.
    Assumes filename format: {specie_en}-{recording_id}-{segment}.png
    Handles potential underscores or hyphens within the species name.
    """
    try:
        # Use pathlib for robust path handling
        p = pathlib.Path(filepath)
        filename = p.name
        base_name = filename.rsplit(".", 1)[0]  # Remove .png extension

        # Split from the right to isolate segment and recording ID
        parts = base_name.rsplit("-", 2)
        if len(parts) == 3:
            species_name = parts[0]
            recording_id = parts[1]
            segment_num = parts[2]
            # Basic validation: recording_id and segment_num should be numeric
            if recording_id.isdigit() and segment_num.isdigit():
                return {
                    "filepath": str(filepath),  # Store full path as string
                    "species": species_name.replace("_", " "),  # Normalize species name
                    "recording_id": int(recording_id),
                    "segment": int(segment_num),
                }
            else:
                logging.warning(f"Could not parse numeric IDs in filename: {filename}")
                return None
        else:
            logging.warning(f"Filename does not match expected format: {filename}")
            return None
    except Exception as e:
        logging.error(f"Error parsing filename {filepath}: {e}")
        return None


def create_splits(input_dir, output_dir, test_size=0.1, val_size=0.1, random_state=42) -> None:
    """
    Scans the input directory, splits data by recording ID with stratification by species,
    and saves train, validation, and test file lists (with labels) to CSV files.
    Also saves a species-to-integer label mapping.

    Args:
        input_dir (str): Path to the directory containing PNG spectrogram files.
        output_dir (str): Path to the directory where CSV files and label map will be saved.
        test_size (float): Proportion of unique recordings for the test set.
        val_size (float): Proportion of unique recordings for the validation set (from the remainder after test split).
        random_state (int): Random seed for reproducible splits.
    """
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)

    if not input_path.is_dir():
        logging.error(f"Input directory not found: {input_dir}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Scanning directory: {input_dir}")
    all_files_data = []
    recordings_data = defaultdict(lambda: {"species": None, "files": []})
    all_species = set()

    # Use rglob to find all PNG files recursively
    for filepath in input_path.rglob("*.png"):
        parsed = parse_filename(filepath)
        if parsed:
            all_files_data.append(parsed)
            rec_id = parsed["recording_id"]
            species = parsed["species"]
            if recordings_data[rec_id]["species"] is None:
                recordings_data[rec_id]["species"] = species
            elif recordings_data[rec_id]["species"] != species:
                logging.warning(
                    f"Recording ID {rec_id} found with multiple species: "
                    f"'{recordings_data[rec_id]['species']}' and '{species}'. "
                    f"Using the first one found ('{recordings_data[rec_id]['species']}').",
                )
                # Keep the first species associated with the recording ID
                parsed["species"] = recordings_data[rec_id]["species"]

            recordings_data[rec_id]["files"].append(parsed["filepath"])
            all_species.add(species)

    if not recordings_data:
        logging.error(f"No valid PNG files found or parsed in {input_dir}")
        return

    logging.info(
        f"Found {len(all_files_data)} total segments from {len(recordings_data)} unique recordings.",
    )
    logging.info(f"Found {len(all_species)} unique species.")

    # Prepare for splitting: lists of unique recording IDs and their corresponding species
    unique_recording_ids = list(recordings_data.keys())
    recording_species = [
        recordings_data[rec_id]["species"] for rec_id in unique_recording_ids
    ]

    # --- Create Label Mapping ---
    sorted_species = sorted(list(all_species))
    label_map = {species: i for i, species in enumerate(sorted_species)}
    label_map_file = output_path / "label_map.json"
    with open(label_map_file, "w") as f:
        json.dump(label_map, f, indent=4)
    logging.info(f"Saved species-to-label mapping to {label_map_file}")
    num_classes = len(label_map)
    logging.info(f"Number of classes: {num_classes}")

    # --- Split 1: Separate Test Set ---
    # Adjust val_size relative to the remaining data after test split
    remaining_size = 1.0 - test_size
    relative_val_size = val_size / remaining_size if remaining_size > 0 else 0

    try:
        train_val_ids, test_ids, train_val_species, _ = train_test_split(
            unique_recording_ids,
            recording_species,
            test_size=test_size,
            random_state=random_state,
            stratify=recording_species,  # Stratify by species
        )
        logging.info(f"Split {len(test_ids)} recordings for test set.")
    except ValueError as e:
        logging.error(
            f"Could not perform initial train/test split, possibly due to small classes: {e}",
        )
        # Handle case where stratification might fail with very small classes
        if "n_splits=2" in str(e):
            logging.warning(
                "Attempting split without stratification due to small classes...",
            )
            train_val_ids, test_ids = train_test_split(
                unique_recording_ids, test_size=test_size, random_state=random_state,
            )
            # Need to re-derive species lists for the next step
            train_val_species = [
                recordings_data[rec_id]["species"] for rec_id in train_val_ids
            ]
            logging.info(
                f"Split {len(test_ids)} recordings for test set (no stratification).",
            )
        else:
            raise e  # Re-raise other errors

    # --- Split 2: Separate Train and Validation Sets ---
    if (
        len(train_val_ids) > 1 and relative_val_size > 0
    ):  # Need at least 2 samples to split
        try:
            train_ids, val_ids, _, _ = train_test_split(
                train_val_ids,
                train_val_species,  # Use the species subset corresponding to train_val_ids
                test_size=relative_val_size,  # Use relative val size
                random_state=random_state,
                stratify=train_val_species,  # Stratify by species within this subset
            )
            logging.info(
                f"Split {len(train_ids)} recordings for train, {len(val_ids)} for validation.",
            )
        except ValueError as e:
            logging.error(
                f"Could not perform train/validation split, possibly due to small classes: {e}",
            )
            if "n_splits=2" in str(e):
                logging.warning(
                    "Attempting split without stratification due to small classes...",
                )
                train_ids, val_ids = train_test_split(
                    train_val_ids,
                    test_size=relative_val_size,
                    random_state=random_state,
                )
                logging.info(
                    f"Split {len(train_ids)} recordings for train, {len(val_ids)} for validation (no stratification).",
                )
            else:
                # If split fails even without strat, assign all remaining to train
                logging.warning(
                    "Assigning all remaining train_val samples to train set.",
                )
                train_ids = train_val_ids
                val_ids = []
                logging.info(
                    f"Split {len(train_ids)} recordings for train, {len(val_ids)} for validation.",
                )

    elif len(train_val_ids) > 0:
        # If not enough data to split further or val_size is 0, assign all to train
        logging.warning(
            "Not enough data or val_size=0, assigning remaining to train set.",
        )
        train_ids = train_val_ids
        val_ids = []
        logging.info(
            f"Split {len(train_ids)} recordings for train, {len(val_ids)} for validation.",
        )
    else:
        # Should not happen if test_size < 1, but handle defensively
        train_ids = []
        val_ids = []
        logging.info("No recordings left for train/validation after test split.")

    # --- Collect Filepaths for Each Split ---
    def get_split_data(recording_ids):
        split_files = []
        for rec_id in recording_ids:
            species = recordings_data[rec_id]["species"]
            label = label_map[species]  # Get integer label
            for fpath in recordings_data[rec_id]["files"]:
                split_files.append(
                    {
                        "filepath": fpath,
                        "species": species,
                        "label": label,
                        "recording_id": rec_id,
                    },
                )
        return split_files

    train_data = get_split_data(train_ids)
    val_data = get_split_data(val_ids)
    test_data = get_split_data(test_ids)

    # --- Create DataFrames and Save to CSV ---
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    train_csv_path = output_path / "train_split.csv"
    val_csv_path = output_path / "val_split.csv"
    test_csv_path = output_path / "test_split.csv"

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    logging.info(f"Saved train split ({len(train_df)} segments) to {train_csv_path}")
    logging.info(f"Saved validation split ({len(val_df)} segments) to {val_csv_path}")
    logging.info(f"Saved test split ({len(test_df)} segments) to {test_csv_path}")
    logging.info("--- Split Summary ---")
    logging.info(f"Total Recordings: {len(recordings_data)}")
    logging.info(f"Train Recordings: {len(train_ids)}")
    logging.info(f"Validation Recordings: {len(val_ids)}")
    logging.info(f"Test Recordings: {len(test_ids)}")
    logging.info(f"Total Segments: {len(all_files_data)}")
    logging.info(f"Train Segments: {len(train_df)}")
    logging.info(f"Validation Segments: {len(val_df)}")
    logging.info(f"Test Segments: {len(test_df)}")
    logging.info("--------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split bird song spectrogram dataset by recording ID.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/xeno-canto-brazil-small",  # Default input dir
        help="Path to the directory containing PNG spectrogram files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/data_splits",  # Default output dir
        help="Path to the directory where split CSV files and label map will be saved.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Proportion of unique recordings for the test set (e.g., 0.1 for 10%).",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Proportion of unique recordings for the validation set (relative to total, e.g., 0.1 for 10%).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )

    args = parser.parse_args()

    # Validate sizes
    if not (0.0 <= args.test_size < 1.0):
        parser.error("test_size must be between 0.0 and 1.0 (exclusive of 1.0)")
    if not (0.0 <= args.val_size < 1.0):
        parser.error("val_size must be between 0.0 and 1.0 (exclusive of 1.0)")
    if args.test_size + args.val_size >= 1.0:
        parser.error("The sum of test_size and val_size must be less than 1.0")

    create_splits(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )
