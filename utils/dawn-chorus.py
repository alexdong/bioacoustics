import requests
import json
import os
import time
import re
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, Optional, List

# --- Configuration ---
API_URL = "https://api.coreo.io/graphql"
# IMPORTANT: This JWT token might expire. Get a fresh one from browser dev tools if needed.
# Ensure this token is CURRENT.
AUTH_TOKEN = "JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJrZXkiOiJiMzA0MjUxNjljODkyMWZkYjA2ZTc3NjY1YjJiZDg5NSIsImlhdCI6MTYxNzY5Nzc2NSwiaXNzIjoiY29yZW8ifQ.v-S8zGmYdOcJGaw5XBQ3VGOu-pVdydOHiohELd9-8CU" # <-- REPLACE IF NEEDED
DOWNLOAD_DIR = "./datasets/dawn-chorus"
RECORDS_PER_PAGE = 12  # Confirmed from the query (limit: 12)
MAX_PAGES = None       # Set to a number (e.g., 10) to limit pages, or None for all
START_OFFSET = 0       # Set to a higher value to resume download if needed
SURVEY_ID = 734        # Confirmed from the variables (surveyId: 734)
MAX_WORKERS = 8        # Number of parallel downloads

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

# --- Helper Functions ---
def get_file_extension(audio_url: str) -> str:
    """Extract file extension from URL, defaulting to .wav if not found."""
    try:
        file_extension = os.path.splitext(audio_url)[1]
        if not file_extension or len(file_extension) > 5:  # Basic sanity check
            print(f"    Warning: Unusual file extension '{file_extension}' for URL {audio_url}. Defaulting to .wav")
            file_extension = ".wav"
        return file_extension
    except Exception:
        print(f"    Warning: Could not parse extension from URL {audio_url}. Defaulting to .wav")
        return ".wav"  # Default if extraction fails

def download_audio(session: requests.Session, record: Dict[str, Any]) -> bool:
    """Downloads an audio file and saves record data as JSON."""
    record_id = record.get("id")
    record_data = record.get("data", {})
    
    if not isinstance(record_data, dict):
        print(f"  Skipping record ID {record_id}: 'data' field is missing or not a dictionary.")
        return False
    
    audio_url = record_data.get("audio")
    if not audio_url or not isinstance(audio_url, str) or not audio_url.startswith('http'):
        print(f"  Skipping record ID {record_id}: No valid 'audio' URL found.")
        return False
    
    # Get file extension from URL
    file_extension = get_file_extension(audio_url)
    
    # Create filenames
    audio_filename = f"{record_id}{file_extension}"
    json_filename = f"{record_id}.json"
    
    audio_filepath = os.path.join(DOWNLOAD_DIR, audio_filename)
    json_filepath = os.path.join(DOWNLOAD_DIR, json_filename)
    
    # Check if files already exist
    if os.path.exists(audio_filepath) and os.path.exists(json_filepath):
        print(f"  Skipping record ID {record_id}: Files already exist")
        return False
    
    # Download audio file
    try:
        print(f"  Downloading: {audio_url} -> {audio_filename}")
        with session.get(audio_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(audio_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Save record data as JSON
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2)
        
        print(f"  Successfully saved: {audio_filename} and {json_filename}")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"  Error downloading {audio_url}: {e}")
        # Clean up potentially incomplete files
        for filepath in [audio_filepath, json_filepath]:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
        return False
    
    except Exception as e:
        print(f"  Unexpected error during download of {audio_url}: {e}")
        for filepath in [audio_filepath, json_filepath]:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
        return False

# --- Main Script ---
if __name__ == "__main__":
    print(f"Starting download from Dawn Chorus (Survey ID: {SURVEY_ID})")
    print(f"Saving files to: {DOWNLOAD_DIR}")
    print(f"Using {MAX_WORKERS} parallel workers for downloads")

    # Create download directory if it doesn't exist
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    current_offset = START_OFFSET
    pages_fetched = 0
    total_downloaded = 0
    
    # Set up session with headers
    session = requests.Session()
    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Authorization": AUTH_TOKEN,
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "Origin": "https://explore.dawn-chorus.org",
        "Referer": "https://explore.dawn-chorus.org/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    session.headers.update(headers)

    # Create ThreadPoolExecutor for parallel downloads
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)


    while True:
        if MAX_PAGES is not None and pages_fetched >= MAX_PAGES:
            print(f"\nReached maximum page limit ({MAX_PAGES}). Stopping.")
            break

        print(f"\nFetching page {pages_fetched + 1} (offset: {current_offset}, limit: {RECORDS_PER_PAGE})...")

        # Construct the payload with the correct query and variables
        payload = {
            "query": GRAPHQL_QUERY,
            "variables": {
                "surveyId": SURVEY_ID,
                "offset": current_offset
                # 'limit' is hardcoded in the query string now
            }
        }

        try:
            response = session.post(API_URL, json=payload, timeout=30) # Use session.post
            response.raise_for_status()  # Check for HTTP errors (4xx or 5xx)
            data = response.json()

        except requests.exceptions.Timeout:
            print("Request timed out. Retrying after 10 seconds...")
            time.sleep(10)
            continue
        except requests.exceptions.RequestException as e:
            print(f"HTTP Request failed: {e}")
            if response is not None:
                print(f"Status Code: {response.status_code}")
                print(f"Response Text: {response.text[:500]}...") # Print start of response
            print("Stopping script.")
            break
        except json.JSONDecodeError:
            print("Failed to decode JSON response. Response text:")
            print(response.text[:1000]) # Print more if it's not JSON
            print("Stopping script.")
            break

        # Check for GraphQL errors within the response
        if "errors" in data:
            print("GraphQL API returned errors:")
            print(json.dumps(data["errors"], indent=2))
            # Specific check for auth errors (might need adjustment based on actual error message)
            if any("Unauthorized" in err.get("message", "") for err in data["errors"]) or \
               any("invalid" in err.get("message", "").lower() and "token" in err.get("message", "").lower() for err in data["errors"]):
                 print("\nAuthorization Error: Your JWT token might be invalid or expired.")
                 print("Please obtain a new token from your browser's developer tools network tab")
                 print("while browsing https://dawn-chorus.org/the-chorus/ and update the AUTH_TOKEN variable.")
            print("Stopping script.")
            break

        records = data.get("data", {}).get("records", [])

        if not records:
            print("No more records found in the response for this offset. Download likely complete.")
            break

        print(f"Found {len(records)} records on this page.")
        pages_fetched += 1
        
        # Submit download tasks to the thread pool
        futures = []
        for record in records:
            futures.append(executor.submit(download_audio, session, record))
        
        # Wait for all downloads to complete
        page_download_count = 0
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                page_download_count += 1
                total_downloaded += 1
        
        print(f"Downloaded {page_download_count} files from this page.")

        # Prepare for next page
        if len(records) == 0 and page_download_count == 0:
            print("Received an empty list of records, assuming end of data.")
            break

        current_offset += len(records)
        print(f"Total files downloaded so far: {total_downloaded}")
        
        # Be polite to the server - wait before fetching the next page
        time.sleep(2)

    # Shutdown the executor
    executor.shutdown()
    print(f"\nFinished. Total files downloaded in this run: {total_downloaded}")
