import concurrent.futures
import json
import multiprocessing
import os
import time
from typing import Any, Dict, List, Tuple

import requests

from utils.download_utils import (
    create_session,
    download_file,
    ensure_directory,
    get_file_extension,
    save_json,
)

# --- Configuration ---
API_URL = "https://api.coreo.io/graphql"
# IMPORTANT: This JWT token might expire. Get a fresh one from browser dev tools if needed.
AUTH_TOKEN = "JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJrZXkiOiJiMzA0MjUxNjljODkyMWZkYjA2ZTc3NjY1YjJiZDg5NSIsImlhdCI6MTYxNzY5Nzc2NSwiaXNzIjoiY29yZW8ifQ.v-S8zGmYdOcJGaw5XBQ3VGOu-pVdydOHiohELd9-8CU"
DOWNLOAD_DIR = "./datasets/dawn-chorus"
RECORDS_PER_PAGE = 12
MAX_PAGES = None
START_OFFSET = 0
SURVEY_ID = 734
# Set MAX_WORKERS to CPU count * 2 to saturate CPUs (mostly I/O bound task)
MAX_WORKERS = multiprocessing.cpu_count() * 2

# --- GraphQL Query (Exactly as observed in the request) ---
GRAPHQL_QUERY = """
query DCGetRecordsWeb($surveyId: Int!, $offset: Int!){
    records(limit: 12, offset: $offset, where: { surveyId: $surveyId, state: { not: 1 }, data: {location_private: { not: true }} }, order: "reverse:data.date_time"){
      id
      data  # Requesting the full data object is fine
      state
      likes { userId } # Included to match browser request
      associates {     # Included to match browser request
        record {
          id
          data
          userId
        }
      }
    }
  }
"""


def download_audio(session: requests.Session, record: Dict[str, Any]) -> bool:
    """Downloads an audio file and saves record data as JSON."""
    record_id = record.get("id")
    record_data = record.get("data", {})

    if not isinstance(record_data, dict):
        print(
            f"  Skipping record ID {record_id}: 'data' field is missing or not a dictionary.",
        )
        return False

    audio_url = record_data.get("audio")
    if (
        not audio_url
        or not isinstance(audio_url, str)
        or not audio_url.startswith("http")
    ):
        print(f"  Skipping record ID {record_id}: No valid 'audio' URL found.")
        return False

    # Get file extension from URL
    file_extension = get_file_extension(audio_url)

    # Create filenames
    audio_filename = f"{record_id}{file_extension}"
    json_filename = f"{record_id}.json"

    download_dir_path = ensure_directory(DOWNLOAD_DIR)
    audio_filepath = download_dir_path / audio_filename
    json_filepath = download_dir_path / json_filename

    # Check if files already exist
    if os.path.exists(audio_filepath) and os.path.exists(json_filepath):
        print(f"  Skipping record ID {record_id}: Files already exist")
        return False

    # Download audio file with retry logic
    print(f"  Downloading: {audio_url} -> {audio_filename}")
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            if download_file(audio_url, audio_filepath):
                # Save record data as JSON
                if save_json(record, json_filepath):
                    print(f"  Successfully saved: {audio_filename} and {json_filename}")
                    return True
                break
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries - 1:
                print(
                    f"  Connection dropped while downloading {audio_filename}. Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})",
                )
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(
                    f"  Failed to download {audio_filename} after {max_retries} attempts: {e}",
                )

    return False


def fetch_records_page(
    session: requests.Session,
    offset: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Fetch a page of records from the API.

    Returns:
        Tuple containing (list of records, should_continue flag)
    """
    payload = {
        "query": GRAPHQL_QUERY,
        "variables": {
            "surveyId": SURVEY_ID,
            "offset": offset,
        },
    }

    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            response = session.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            break  # Success, exit the retry loop

        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries - 1:
                print(
                    f"Connection dropped. Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})",
                )
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            print(f"Failed to connect after {max_retries} attempts: {e}")
            return [], False  # Stop processing

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(
                    f"Request timed out. Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})",
                )
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            print(f"Request timed out after {max_retries} attempts.")
            return [], False  # Stop processing

        except requests.exceptions.RequestException as e:
            print(f"HTTP Request failed: {e}")
            if "response" in locals():
                print(f"Status Code: {response.status_code}")
                print(f"Response Text: {response.text[:500]}...")
            return [], False  # Stop processing

        except json.JSONDecodeError:
            print("Failed to decode JSON response. Response text:")
            print(response.text[:1000])
            return [], False  # Stop processing

    # If we got here without a successful response, return empty
    if "data" not in locals():
        return [], False

    # Check for GraphQL errors within the response
    if "errors" in data:
        print("GraphQL API returned errors:")
        print(json.dumps(data["errors"], indent=2))
        # Check for auth errors
        if any(
            "Unauthorized" in err.get("message", "") for err in data["errors"]
        ) or any(
            "invalid" in err.get("message", "").lower()
            and "token" in err.get("message", "").lower()
            for err in data["errors"]
        ):
            print("\nAuthorization Error: Your JWT token might be invalid or expired.")
            print(
                "Please obtain a new token from your browser's developer tools network tab",
            )
            print(
                "while browsing https://dawn-chorus.org/the-chorus/ and update the AUTH_TOKEN variable.",
            )
        return [], False  # Stop processing

    records = data.get("data", {}).get("records", [])
    return records, True  # Continue processing


def process_records_page(
    session: requests.Session,
    executor: concurrent.futures.ThreadPoolExecutor,
    records: List[Dict[str, Any]],
) -> int:
    """Process a page of records by downloading audio files in parallel.

    Returns:
        Number of successfully downloaded files
    """
    if not records:
        return 0

    # Submit download tasks to the thread pool
    futures = []
    for record in records:
        futures.append(executor.submit(download_audio, session, record))

    # Wait for all downloads to complete
    page_download_count = 0
    for future in concurrent.futures.as_completed(futures):
        if future.result():
            page_download_count += 1

    return page_download_count


def download_all_records() -> None:
    """Main function to download all records."""
    print(f"Starting download from Dawn Chorus (Survey ID: {SURVEY_ID})")
    print(f"Saving files to: {DOWNLOAD_DIR}")
    print(f"Using {MAX_WORKERS} parallel workers for downloads")

    # Create download directory
    ensure_directory(DOWNLOAD_DIR)

    current_offset = START_OFFSET
    pages_fetched = 0
    total_downloaded = 0

    # Set up session with appropriate headers
    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Authorization": AUTH_TOKEN,
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "Origin": "https://explore.dawn-chorus.org",
        "Referer": "https://explore.dawn-chorus.org/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }
    session = create_session(headers)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

    try:
        while True:
            if MAX_PAGES is not None and pages_fetched >= MAX_PAGES:
                print(f"\nReached maximum page limit ({MAX_PAGES}). Stopping.")
                break

            print(
                f"\nFetching page {pages_fetched + 1} (offset: {current_offset}, limit: {RECORDS_PER_PAGE})...",
            )

            # Fetch records
            records, should_continue = fetch_records_page(session, current_offset)
            if not should_continue:
                break

            if not records:
                print(
                    "No more records found in the response for this offset. Download likely complete.",
                )
                break

            print(f"Found {len(records)} records on this page.")
            pages_fetched += 1

            # Process records
            page_download_count = process_records_page(session, executor, records)
            total_downloaded += page_download_count

            print(f"Downloaded {page_download_count} files from this page.")

            # Prepare for next page
            if len(records) == 0 and page_download_count == 0:
                print("Received an empty list of records, assuming end of data.")
                break

            current_offset += len(records)
            print(f"Total files downloaded so far: {total_downloaded}")

            # Be polite to the server - wait before fetching the next page
            # Only sleep between GraphQL queries, not between downloads
            time.sleep(2)

    finally:
        # Ensure executor is properly shut down
        executor.shutdown()
        print(f"\nFinished. Total files downloaded in this run: {total_downloaded}")


if __name__ == "__main__":
    download_all_records()
