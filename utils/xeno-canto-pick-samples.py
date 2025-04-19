import csv
import json
import os
import random
import sys
import time  # To estimate time

# --- Configuration ---
SUMMARY_CSV_PATH = "datasets/xeno-canto-summary.csv"
JSON_DIR_PATH = "./datasets/xeno-canto/"
OUTPUT_FILE_PATH = "datasets/xeno-canto-dataset.txt"
TARGET_SPECIES_COUNT = 100
MIN_USABLE_SECONDS = 300  # 5 minutes threshold

# Tier definitions (based on A+B quality seconds)
TIER_LOW_UPPER_BOUND = 720  # Up to 12 mins
TIER_MED_UPPER_BOUND = 1800  # Up to 30 mins
# Tier High is > TIER_MED_UPPER_BOUND

# Sampling targets per tier (adjust if needed to sum ~ TARGET_SPECIES_COUNT)
SAMPLE_COUNT_TIER_HIGH = 30
SAMPLE_COUNT_TIER_MED = 40
SAMPLE_COUNT_TIER_LOW = 30

# For reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# --- Helper Function ---
def parse_seconds(value: str) -> float:
    """Safely convert string to float, return 0.0 on error."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


# --- Phase 1: Species Selection from Summary CSV ---
print("--- Phase 1: Selecting Species from Summary CSV ---")
eligible_species = []  # List of tuples: (species_name, usable_seconds)

try:
    with open(SUMMARY_CSV_PATH, "r", encoding="utf-8") as f_csv:
        reader = csv.reader(f_csv)
        header = next(reader)  # Skip header
        # Find column indices dynamically to be more robust
        try:
            en_idx = header.index("en")
            qa_idx = header.index("q_A_seconds")
            qb_idx = header.index("q_B_seconds")
        except ValueError as e:
            print(f"ERROR: Missing required column in {SUMMARY_CSV_PATH}: {e}")
            sys.exit(1)

        row_count = 0
        for row in reader:
            row_count += 1
            # Basic check for row length consistency
            if len(row) <= max(en_idx, qa_idx, qb_idx):
                # print(f"Warning: Skipping malformed row {row_count+1} in summary CSV.")
                continue  # skip rows that dont have enough columns

            species_name = row[en_idx].strip()
            q_a = parse_seconds(row[qa_idx])
            q_b = parse_seconds(row[qb_idx])
            usable_seconds = q_a + q_b

            if (
                usable_seconds >= MIN_USABLE_SECONDS
                and species_name.lower() != "identity unknown"
                and species_name
            ):  # Ensure name is not empty
                eligible_species.append((species_name, usable_seconds))

except FileNotFoundError:
    print(f"ERROR: Summary file not found at {SUMMARY_CSV_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR reading summary file {SUMMARY_CSV_PATH}: {e}")
    sys.exit(1)

print(
    f"Found {len(eligible_species)} eligible species meeting the >= {MIN_USABLE_SECONDS}s threshold.",
)

if not eligible_species:
    print("ERROR: No species meet the minimum usable seconds requirement.")
    sys.exit(1)
if len(eligible_species) < TARGET_SPECIES_COUNT:
    print(
        f"WARNING: Fewer than {TARGET_SPECIES_COUNT} eligible species available ({len(eligible_species)}). Will select all available.",
    )
    TARGET_SPECIES_COUNT = len(eligible_species)  # Adjust target if needed

# Assign to Tiers
tier_high_spp = []
tier_med_spp = []
tier_low_spp = []

for name, seconds in eligible_species:
    if seconds > TIER_MED_UPPER_BOUND:
        tier_high_spp.append(name)
    elif seconds > TIER_LOW_UPPER_BOUND:
        tier_med_spp.append(name)
    else:  # Handles >= MIN_USABLE_SECONDS up to TIER_LOW_UPPER_BOUND
        tier_low_spp.append(name)

print(f"  Tier High (> {TIER_MED_UPPER_BOUND}s): {len(tier_high_spp)} species")
print(
    f"  Tier Medium ({TIER_LOW_UPPER_BOUND}s - {TIER_MED_UPPER_BOUND}s): {len(tier_med_spp)} species",
)
print(
    f"  Tier Low ({MIN_USABLE_SECONDS}s - {TIER_LOW_UPPER_BOUND}s): {len(tier_low_spp)} species",
)

# Tiered Sampling
selected_species_list = []

actual_sample_high = min(SAMPLE_COUNT_TIER_HIGH, len(tier_high_spp))
actual_sample_med = min(SAMPLE_COUNT_TIER_MED, len(tier_med_spp))
actual_sample_low = min(SAMPLE_COUNT_TIER_LOW, len(tier_low_spp))

print(f"Sampling {actual_sample_high} from Tier High...")
selected_species_list.extend(random.sample(tier_high_spp, k=actual_sample_high))

print(f"Sampling {actual_sample_med} from Tier Medium...")
selected_species_list.extend(random.sample(tier_med_spp, k=actual_sample_med))

print(f"Sampling {actual_sample_low} from Tier Low...")
selected_species_list.extend(random.sample(tier_low_spp, k=actual_sample_low))

# Ensure uniqueness and handle potential shortfall
selected_species_set = set(selected_species_list)
final_selected_count = len(selected_species_set)
print(f"Sampled {final_selected_count} unique species initially.")

if final_selected_count < TARGET_SPECIES_COUNT:
    print(f"Attempting to select additional species to reach {TARGET_SPECIES_COUNT}...")
    all_eligible_names = {name for name, _ in eligible_species}
    remaining_species = list(all_eligible_names - selected_species_set)
    needed = TARGET_SPECIES_COUNT - final_selected_count

    if len(remaining_species) >= needed:
        additional_sample = random.sample(remaining_species, k=needed)
        selected_species_set.update(additional_sample)
        print(f"Added {len(additional_sample)} more species.")
    else:
        print(f"Could only add {len(remaining_species)} more species.")
        selected_species_set.update(remaining_species)

# Final sorted list of selected species names
selected_species_en = sorted(list(selected_species_set))
print(f"Final selected species count: {len(selected_species_en)}")
# Use a set for fast lookups in the next phase
selected_species_lookup = set(selected_species_en)

# --- Phase 2: Collect Recording IDs from Individual JSONs ---
print("\n--- Phase 2: Collecting Recording IDs from JSON Files ---")
print(f"Scanning directory: {JSON_DIR_PATH}")
print("!! This phase can be VERY SLOW depending on the number of JSON files !!")

species_to_ids = {
    species: [] for species in selected_species_en
}  # Pre-populate dictionary
start_time = time.time()
files_scanned = 0
errors_encountered = 0
relevant_found = 0

if not os.path.isdir(JSON_DIR_PATH):
    print(f"ERROR: JSON directory not found: {JSON_DIR_PATH}")
    sys.exit(1)

try:
    # Using os.scandir for potentially better performance than listdir
    for entry in os.scandir(JSON_DIR_PATH):
        files_scanned += 1
        if files_scanned % 10000 == 0:  # Print progress update
            elapsed = time.time() - start_time
            rate = files_scanned / elapsed if elapsed > 0 else 0
            print(
                f"  Scanned {files_scanned} files... ({rate:.0f} files/sec). Found {relevant_found} relevant. Errors: {errors_encountered}",
                end="\r",
            )

        if entry.is_file() and entry.name.endswith(".json"):
            recording_id = entry.name[:-5]  # Remove '.json' extension
            json_path = entry.path

            try:
                with open(json_path, "r", encoding="utf-8") as f_json:
                    data = json.load(f_json)

                # Check if the recording belongs to a selected species and is from Brazil
                # Use .get() for safer dictionary access
                species = data.get("en", "").strip()
                country = data.get("cnt", "").strip()

                if species in selected_species_lookup and country.lower() == "brazil":
                    species_to_ids[species].append(recording_id)
                    relevant_found += 1

            except json.JSONDecodeError:
                # print(f"\nWarning: Could not decode JSON: {json_path}")
                errors_encountered += 1
            except (
                Exception
            ):  # Catch other potential errors during file processing
                # print(f"\nError processing file {json_path}: {e}")
                errors_encountered += 1
except Exception as e:
    print(f"\nERROR during directory scan: {e}")
    sys.exit(1)

end_time = time.time()
print(
    f"\nFinished scanning {files_scanned} files in {end_time - start_time:.2f} seconds.",
)
print(f"Found {relevant_found} recordings for the selected species in Brazil.")
if errors_encountered > 0:
    print(f"Encountered {errors_encountered} errors reading/parsing JSON files.")


# --- Phase 3: Write Output File ---
print("\n--- Phase 3: Writing Output File ---")
print(f"Writing output to {OUTPUT_FILE_PATH}...")
lines_written = 0
with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f_out:
    for species_name in selected_species_en:  # Iterate using the sorted list
        ids_list = species_to_ids.get(
            species_name, [],
        )  # Get list, default to empty if somehow missing
        ids_string = ",".join(
            sorted(ids_list),
        )  # Join IDs into comma-separated string, sort for consistency
        f_out.write(f"{species_name}|{ids_string}\n")
        lines_written += 1
        if not ids_list:
            print(
                f"  WARNING: No Brazil recordings found in metadata for selected species: {species_name}",
            )


print(
    f"\nProcess complete. Wrote {lines_written} species mappings to {OUTPUT_FILE_PATH}",
)
if lines_written != len(selected_species_en):
    print(
        f"WARNING: Mismatch in lines written ({lines_written}) vs species selected ({len(selected_species_en)}). Check warnings.",
    )
