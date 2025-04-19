import collections
import csv
import os
from typing import Dict, Set, Tuple, Optional, Union, List


def parse_duration(duration_str: str) -> int:
    """Converts M:SS or H:MM:SS string to total seconds."""
    parts = duration_str.split(":")
    seconds = 0
    try:
        if len(parts) == 2:  # M:SS format
            minutes = int(parts[0])
            secs = int(parts[1])
            seconds = minutes * 60 + secs
        elif len(parts) == 3:  # H:MM:SS format
            hours = int(parts[0])
            minutes = int(parts[1])
            secs = int(parts[2])
            seconds = hours * 3600 + minutes * 60 + secs
        # Silently ignore invalid formats, returning 0 seconds
    except (ValueError, IndexError):
        # Silently ignore invalid numbers or structure, returning 0 seconds
        pass
    return seconds


def aggregate_species_data(file_path: str) -> Tuple[Optional[Dict], Optional[Set]]:
    """
    Aggregates recording data by species from the input CSV.

    Args:
        file_path (str): The path to the input CSV file.

    Returns:
        tuple: Contains:
            - species_agg_data (dict): {species_name: {agg_details}}
            - all_qualities (set): A set of all unique quality codes found.
            Returns None, None on critical errors.
    """
    if not os.path.exists(file_path):
        print(f"Error: Input file not found at {file_path}")
        return None, None

    # Stores aggregated data. Key: species_name ('en')
    # Value: dict with 'id', 'gen', 'sp', 'group', 'total_duration_seconds', 'quality_seconds' (nested dict)
    species_agg_data = {}
    all_qualities = set()  # Keep track of all quality codes encountered

    try:
        with open(file_path, mode="r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            # Check for essential columns
            required_cols = {"id", "en", "gen", "sp", "group", "q", "length"}
            if not required_cols.issubset(reader.fieldnames):
                missing = required_cols - set(reader.fieldnames)
                print(
                    f"Error: Missing required columns in input CSV: {', '.join(missing)}",
                )
                return None, None

            line_num = 1
            for row in reader:
                line_num += 1
                try:
                    species_name = row.get("en", "").strip()
                    # Skip rows entirely if species name is missing (adjust if "" is valid)
                    # if not species_name:
                    #     print(f"Warning: Skipping row {line_num} due to missing species name ('en').")
                    #     continue

                    quality = row.get("q", "").strip()
                    duration_str = row.get("length", "").strip()
                    duration_seconds = parse_duration(duration_str)

                    # Track unique quality codes encountered
                    if quality:
                        all_qualities.add(quality)

                    # If this species is seen for the first time, initialize its entry
                    if species_name not in species_agg_data:
                        species_agg_data[species_name] = {
                            "id": row.get("id", ""),  # Store the first encountered ID
                            "en": species_name,
                            "gen": row.get("gen", ""),
                            "sp": row.get("sp", ""),
                            "group": row.get("group", ""),
                            "total_duration_seconds": 0,
                            "quality_seconds": collections.defaultdict(
                                int,
                            ),  # Store seconds per quality
                        }

                    # Update aggregates for this species
                    species_agg_data[species_name][
                        "total_duration_seconds"
                    ] += duration_seconds
                    if quality:  # Only add to quality breakdown if quality exists
                        species_agg_data[species_name]["quality_seconds"][
                            quality
                        ] += duration_seconds

                except Exception as e:
                    print(f"Warning: Error processing row {line_num}: {row} - {e}")
                    continue  # Skip to next row on error

    except Exception as e:
        print(f"An unexpected error occurred while reading the input file: {e}")
        return None, None

    return species_agg_data, all_qualities


def write_aggregated_csv(output_path: str, species_data: Dict, quality_codes: Set) -> Union[bool, None]:
    """
    Writes the aggregated species data to a single CSV file.

    Args:
        output_path (str): Path for the output CSV file.
        species_data (dict): The aggregated data dictionary from aggregate_species_data.
        quality_codes (set): Set of unique quality codes found.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not species_data:
        print("No data to write.")
        return False

    # Define base fieldnames and add dynamic quality columns
    base_fieldnames = ["id", "en", "gen", "sp", "group", "total_duration_seconds"]
    # Sort quality codes for consistent column order (e.g., A, B, C, D...)
    sorted_qualities = sorted(list(quality_codes))
    quality_fieldnames = [f"q_{q}_seconds" for q in sorted_qualities]
    output_fieldnames = base_fieldnames + quality_fieldnames

    try:
        print(f"Writing aggregated data to {output_path}...")
        with open(output_path, mode="w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
            writer.writeheader()

            # Sort species by name for predictable output order
            sorted_species_names = sorted(species_data.keys())

            for species_name in sorted_species_names:
                data = species_data[species_name]
                output_row = {
                    "id": data.get("id", ""),
                    "en": data.get("en", ""),
                    "gen": data.get("gen", ""),
                    "sp": data.get("sp", ""),
                    "group": data.get("group", ""),
                    "total_duration_seconds": data.get("total_duration_seconds", 0),
                }
                # Add the seconds for each quality grade
                for q in sorted_qualities:
                    q_key = f"q_{q}_seconds"
                    # Use .get on the defaultdict to handle cases where a species
                    # might not have recordings of a specific quality.
                    output_row[q_key] = data["quality_seconds"].get(q, 0)

                writer.writerow(output_row)

        print("...Write complete.")
        return True

    except IOError as e:
        print(f"Error writing file {output_path}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred writing the output CSV: {e}")
        return False


# --- Main execution ---
if __name__ == "__main__":
    # --- Configuration ---
    input_csv_file = "datasets/xeno-canto-brazil.csv"
    output_csv_file = "species_summary_output.csv"
    # -------------------

    print(f"Starting aggregation for {input_csv_file}...")

    # Aggregate data
    aggregated_data, unique_qualities = aggregate_species_data(input_csv_file)

    if aggregated_data is not None and unique_qualities is not None:
        print(
            f"Aggregation complete. Found {len(aggregated_data)} unique species entries.",
        )
        print(f"Unique quality codes found: {sorted(list(unique_qualities))}")

        # Write the aggregated data to the output CSV
        success = write_aggregated_csv(
            output_csv_file, aggregated_data, unique_qualities,
        )

        if success:
            print("\nProcessing finished successfully.")
        else:
            print("\nProcessing finished with errors during CSV writing.")
    else:
        print("\nProcessing aborted due to errors during data aggregation.")
