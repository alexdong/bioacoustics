import duckdb
import glob
import os
import math

# --- Configuration ---
db_file = '/Users/alexdong/Projects/bird-songs/datasets/xeno-canto.duckdb' # Use the persistent DB file
# Use the absolute path for globbing in Python too, for consistency
json_pattern = '/Users/alexdong/Projects/bird-songs/datasets/xeno-canto/*.json'
table_name = 'recordings'
# Adjust batch size based on memory and testing. Start smaller if needed.
# 182k files / 1000 files/batch = ~183 batches. Seems reasonable.
batch_size = 1000
# --- Configuration ---

# Get the full list of files using Python's glob
print(f"Finding files matching: {json_pattern}")
json_files = sorted(glob.glob(json_pattern)) # Use Python's glob

if not json_files:
    print(f"Error: Python glob could not find any files matching pattern '{json_pattern}'")
    exit()

total_files = len(json_files)
print(f"Found {total_files} JSON files.")

# Connect to DuckDB
# Consider increasing memory limit here too, as loading data still uses RAM
con = duckdb.connect(database=db_file, read_only=False)
print(f"Connected to DuckDB database: {db_file}")
# con.execute("PRAGMA memory_limit='10GB';") # Optional: Uncomment and adjust if needed

# Step 1: Create the table schema using ONLY THE FIRST file
# This avoids asking read_json_auto to process the huge list for schema inference
try:
    print(f"Creating table '{table_name}' schema based on first file: {os.path.basename(json_files[0])}")
    # Use read_json_auto on a single file list parameter
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_json_auto(?) LIMIT 0", [[json_files[0]]])
    print(f"Table '{table_name}' created successfully.")
except Exception as e:
    print(f"Error creating table schema from {json_files[0]}: {e}")
    con.close()
    exit()

# Step 2: Process files in batches, inserting data
num_batches = math.ceil(total_files / batch_size)
print(f"Starting data insertion in {num_batches} batches of up to {batch_size} files each...")

for i in range(0, total_files, batch_size):
    batch_files = json_files[i:min(i + batch_size, total_files)]
    current_batch_num = (i // batch_size) + 1
    print(f"Processing Batch {current_batch_num}/{num_batches} ({len(batch_files)} files starting with {os.path.basename(batch_files[0])}...)", end='', flush=True)

    try:
        # Pass the explicit LIST of filenames for this batch as a parameter
        # read_json_auto([file1, file2, ...]) is the syntax needed
        con.execute(f"INSERT INTO {table_name} SELECT * FROM read_json_auto(?);", [batch_files])
        print(" Done.")
    except Exception as e:
        print(f"\nError processing batch {current_batch_num} starting with {os.path.basename(batch_files[0])}: {e}")
        # Decide how to handle errors: stop, skip batch, log, etc.
        # For now, we'll stop the script on error.
        con.close()
        exit()

print(f"\nFinished processing all {total_files} files.")

# Optional: Verify count
try:
    count_result = con.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()
    if count_result:
        print(f"Final count in table '{table_name}': {count_result[0]}")
    else:
        print(f"Could not get count for table '{table_name}'.")
except Exception as e:
    print(f"Error getting final count: {e}")

con.close()
print("Database connection closed.")

