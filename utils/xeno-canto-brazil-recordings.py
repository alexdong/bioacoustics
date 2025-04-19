"""
Purpose:
Processes a directory of JSON files containing bird sound recording metadata.
Specifically, it:
1. Scans a specified input folder for files ending with '.json'.
2. Reads each JSON file.
3. Filters records, keeping only those where the 'cnt' (country) field is 'Brazil'.
4. For the filtered records, extracts specific fields:
   'id', 'en' (English name), 'type', 'q' (quality), 'length',
   'gen' (genus), 'sp' (species), 'group'.
5. Writes the extracted data into a CSV file named 'stats.csv'.
6. Displays a progress bar using tqdm during file processing.
7. Reports summary statistics upon completion.
"""

import json
import csv
from pathlib import Path
import sys # To print errors to stderr
from tqdm import tqdm # For progress bar

# --- Configuration ---
# Set the path to your folder containing the JSON files
INPUT_FOLDER = Path('datasets/xeno-canto')

# Set the desired name for the output CSV file
OUTPUT_CSV_FILE = Path('stats.csv')

# Fields to extract from the JSON if the condition is met
# Added 'gen', 'sp', 'group' as requested
FIELDS_TO_EXTRACT = ['id', 'en', 'type', 'q', 'length', 'gen', 'sp', 'group']

# The field and value to filter by
FILTER_KEY = 'cnt'
FILTER_VALUE = 'Brazil'
# --- End Configuration ---

# Check if input folder exists
if not INPUT_FOLDER.is_dir():
    print(f"Error: Input folder '{INPUT_FOLDER}' not found or is not a directory.", file=sys.stderr)
    sys.exit(1) # Exit the script with an error code

# Counter variables for summary
processed_files_count = 0
extracted_records_count = 0
error_files_count = 0

print(f"Starting processing of JSON files in: {INPUT_FOLDER}")
print(f"Filtering records where '{FILTER_KEY}' is '{FILTER_VALUE}'...")
print(f"Extracting fields: {', '.join(FIELDS_TO_EXTRACT)}")
print(f"Output will be written to: {OUTPUT_CSV_FILE}")

try:
    # --- Get total file count for tqdm ---
    # Convert glob generator to a list to count files for tqdm progress bar
    # This loads all filenames into memory, which is usually fine,
    # but consider iterating directly if memory is extremely constrained
    # and an approximate progress bar is acceptable.
    print("Counting files...")
    json_files = list(INPUT_FOLDER.glob('*.json'))
    total_files = len(json_files)
    print(f"Found {total_files} JSON files.")
    # --- End file count ---


    # Open the output CSV file for writing
    # newline='' prevents extra blank rows in the CSV on Windows
    # encoding='utf-8' is generally a good default for text data
    with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write the header row to the CSV
        csv_writer.writerow(FIELDS_TO_EXTRACT)

        # Iterate through all files using tqdm for progress bar
        for json_file_path in tqdm(json_files, desc="Processing JSON files", unit="file"):
            processed_files_count += 1

            try:
                # Open and load the JSON data from the current file
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 1. Check the filter condition: 'cnt' must be 'Brazil'
                # Use .get() to avoid KeyError if 'cnt' is missing
                if data.get(FILTER_KEY) == FILTER_VALUE:

                    # 2. Extract the required fields
                    # Create a list containing the values for the specified fields
                    # Use data.get(field, '') to handle cases where a field might be missing,
                    # defaulting to an empty string in that case.
                    row_data = [data.get(field, '') for field in FIELDS_TO_EXTRACT]

                    # 3. Write the extracted data as a row in the CSV
                    csv_writer.writerow(row_data)
                    extracted_records_count += 1

            except json.JSONDecodeError:
                # Handle files that are not valid JSON
                # Suppress warning spamming if using tqdm, maybe log elsewhere if needed
                # print(f"Warning: Skipping invalid JSON file: {json_file_path.name}", file=sys.stderr)
                error_files_count += 1
            except IOError as e:
                # Handle file reading errors
                print(f"\nWarning: Could not read file {json_file_path.name}: {e}", file=sys.stderr)
                error_files_count += 1
            except Exception as e:
                # Catch any other unexpected errors during processing a single file
                print(f"\nWarning: Unexpected error processing file {json_file_path.name}: {e}", file=sys.stderr)
                error_files_count += 1

except IOError as e:
    print(f"\nError: Could not open or write to output file {OUTPUT_CSV_FILE}: {e}", file=sys.stderr)
    sys.exit(1) # Exit if the output file can't be handled
except Exception as e:
    # Catch potential errors during file counting or CSV setup
    print(f"\nAn unexpected error occurred during setup: {e}", file=sys.stderr)
    sys.exit(1)


# --- Summary ---
# Ensure summary prints on a new line after tqdm potentially finishing mid-line
print("\n--- Processing Complete ---")
print(f"Total JSON files scanned: {processed_files_count} (out of {total_files} found)")
print(f"Records matching filter ('{FILTER_KEY}' = '{FILTER_VALUE}') and extracted: {extracted_records_count}")
print(f"Files skipped due to errors (invalid JSON, read errors, etc.): {error_files_count}")
print(f"Output saved to: {OUTPUT_CSV_FILE}")
