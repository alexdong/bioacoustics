# -*- coding: utf-8 -*-
"""
Script to scan a directory for JSON files and validate the 'date' field.

Purpose:
- Iterates through all files ending with '.json' in the specified directory.
- For each JSON file, it checks if a 'date' key exists.
- If the 'date' key exists, it attempts to parse its value as a date
  using the 'YYYY-MM-DD' format.
- If the parsing fails (indicating an invalid or non-standard date format),
  the script replaces the invalid date value with today's date (YYYY-MM-DD).
- Files with valid dates, missing 'date' keys, or non-JSON files are skipped.
- Reports a summary of processed, modified, and skipped files, along with any errors.

WARNING: This script modifies files in place. It is STRONGLY recommended
         to back up your data before running it.
"""

import os
import json
from datetime import date, datetime
import sys

# --- Configuration ---
# Set the path to the folder containing your JSON files
JSON_FOLDER_PATH = './datasets/xeno-canto/'  # Use '.' for the current directory, or specify a full path like '/path/to/your/json/files'

# Define the expected valid date format
EXPECTED_DATE_FORMAT = "%Y-%m-%d"

# --- Additions for Time Cleaning ---
# Define the expected valid time format (use HH:MM:SS for safety)
EXPECTED_TIME_FORMAT = "%H:%M:%S"
DEFAULT_TIME_STR = "09:00:00" # Default value for invalid/missing times
# --- End Configuration ---

def fix_invalid_json_dates(folder_path, date_format):
    """
    Scans a folder for JSON files, checks if the 'date' field can be parsed,
    and replaces unparseable dates with today's date.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1) # Exit if the folder doesn't exist

    today_date_str = date.today().strftime(date_format) # Format: YYYY-MM-DD
    files_processed = 0
    files_modified = 0
    files_skipped_no_date = 0
    files_skipped_valid_date = 0
    files_skipped_no_time = 0
    files_skipped_valid_time = 0
    errors = []

    print(f"Scanning folder: {os.path.abspath(folder_path)}")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            files_processed += 1
            needs_update = False
            invalid_date_value = None # Store the original invalid value for logging

            try:
                # Read the JSON file
                # Use 'utf-8-sig' to handle potential BOM (Byte Order Mark)
                with open(filepath, 'r', encoding='utf-8-sig') as f_read:
                    data = json.load(f_read)

                # Check if 'date' key exists
                if 'date' in data:
                    current_date_value = data.get('date') # Use .get for safety

                    # Attempt to parse the date value
                    try:
                        # Check if it's actually a string before trying to parse
                        if isinstance(current_date_value, str):
                             datetime.strptime(current_date_value, date_format)
                             # If parsing succeeds, the date is considered valid
                             files_skipped_valid_date += 1
                        else:
                             # Handle non-string types (like null, numbers) as invalid
                             invalid_date_value = current_date_value
                             print(f"Found non-string date value in {filename}: '{invalid_date_value}'. Updating...")
                             needs_update = True

                    except ValueError:
                        # Parsing failed - date string is invalid or wrong format
                        invalid_date_value = current_date_value
                        print(f"Found invalid/unparseable date format in {filename}: '{invalid_date_value}'. Updating...")
                        needs_update = True

                    # If flagged for update, set the date to today
                    if needs_update:
                        data['date'] = today_date_str

                else:
                    # 'date' key doesn't exist
                    files_skipped_no_date += 1
                    pass # print(f"Skipping {filename}: No 'date' key found.")

                # --- Add Time Field Check ---
                if 'time' in data:
                    current_time_value = data.get('time')
                    invalid_time_value = None

                    try:
                        # Check if it's a string before parsing
                        if isinstance(current_time_value, str):
                            # Attempt to parse using HH:MM:SS format
                            # We need to parse it against *some* date, today is fine
                            datetime.strptime(current_time_value, EXPECTED_TIME_FORMAT)
                            # If parsing succeeds, time is valid
                            files_skipped_valid_time += 1
                        else:
                            # Handle non-string types as invalid
                            invalid_time_value = current_time_value
                            print(f"Found non-string time value in {filename}: '{invalid_time_value}'. Updating...")
                            needs_update = True
                            data['time'] = DEFAULT_TIME_STR # Set default time

                    except ValueError:
                         # Parsing failed - time string is invalid or wrong format (like "?")
                         invalid_time_value = current_time_value
                         print(f"Found invalid/unparseable time format in {filename}: '{invalid_time_value}'. Updating...")
                         needs_update = True
                         data['time'] = DEFAULT_TIME_STR # Set default time
                else:
                    files_skipped_no_time += 1


                # Write the modified data back to the file ONLY if it was updated
                if needs_update:
                    with open(filepath, 'w', encoding='utf-8') as f_write:
                        # Use indent=4 for pretty printing, adjust if needed
                        json.dump(data, f_write, indent=4, ensure_ascii=False) # ensure_ascii=False if you have non-ascii chars
                    files_modified += 1
                    print(f"Successfully updated: {filename}")


            except json.JSONDecodeError:
                error_msg = f"Error decoding JSON in file: {filename}. Skipping this file."
                print(error_msg)
                errors.append(error_msg)
            except IOError as e:
                error_msg = f"Error reading/writing file {filename}: {e}. Skipping this file."
                print(error_msg)
                errors.append(error_msg)
            except Exception as e:
                error_msg = f"An unexpected error occurred with {filename}: {e}. Skipping this file."
                print(error_msg)
                errors.append(error_msg)

    print("\n" + "=" * 30)
    print("Scan Complete.")
    print(f"Total JSON files processed: {files_processed}")
    if errors:
        print("\nErrors encountered during processing:")
        for err in errors:
            print(f"- {err}")
    print("=" * 30)


# --- Run the script ---
if __name__ == "__main__":
    fix_invalid_json_dates(JSON_FOLDER_PATH, EXPECTED_DATE_FORMAT)
