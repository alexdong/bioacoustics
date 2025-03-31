import requests
import json
import os
import time
import re

# --- Configuration ---
API_URL = "https://api.coreo.io/graphql"
# IMPORTANT: This JWT token might expire. Get a fresh one from browser dev tools if needed.
# Ensure this token is CURRENT.
AUTH_TOKEN = "JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJrZXkiOiJiMzA0MjUxNjljODkyMWZkYjA2ZTc3NjY1YjJiZDg5NSIsImlhdCI6MTYxNzY5Nzc2NSwiaXNzIjoiY29yZW8ifQ.v-S8zGmYdOcJGaw5XBQ3VGOu-pVdydOHiohELd9-8CU" # <-- REPLACE IF NEEDED
DOWNLOAD_DIR = "dawn_chorus_audio"
RECORDS_PER_PAGE = 12  # Confirmed from the query (limit: 12)
MAX_PAGES = None       # Set to a number (e.g., 10) to limit pages, or None for all
START_OFFSET = 0       # Set to a higher value to resume download if needed
SURVEY_ID = 734        # Confirmed from the variables (surveyId: 734)

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
def sanitize_filename(filename):
    """Removes or replaces characters invalid in filenames."""
    if filename is None:
        return "None"
    # Remove invalid characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "", str(filename))
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # Reduce consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores/periods
    sanitized = sanitized.strip('_.')
    # Handle potential empty strings after sanitization
    if not sanitized:
        return "InvalidChars"
    return sanitized

def download_audio(audio_url, filepath):
    """Downloads an audio file from a URL to a local path."""
    if not audio_url or not isinstance(audio_url, str) or not audio_url.startswith('http'):
        print(f"    Invalid audio URL skipped: {audio_url}")
        return False
    try:
        print(f"    Downloading: {audio_url}")
        # Use the session object for the download request as well
        with session.get(audio_url, stream=True, timeout=60) as r:
            r.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            # Check content type if possible (optional but good practice)
            # content_type = r.headers.get('content-type', '').lower()
            # if 'audio' not in content_type:
            #     print(f"    Warning: URL {audio_url} did not return an audio content type ({content_type}). Skipping download.")
            #     return False

            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"    Saved to: {filepath}")
        return True
    except requests.exceptions.MissingSchema:
        print(f"    Invalid URL format (Missing http/https): {audio_url}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"    Error downloading {audio_url}: {e}")
        # Clean up potentially incomplete file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError: pass # Ignore error if file can't be removed immediately
        return False
    except Exception as e:
        print(f"    Unexpected error during download of {audio_url}: {e}")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError: pass
        return False

# --- Main Script ---
if __name__ == "__main__":
    print(f"Starting download from Dawn Chorus (Survey ID: {SURVEY_ID})")
    print(f"Saving audio files to: {DOWNLOAD_DIR}")

    # Create download directory if it doesn't exist
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    current_offset = START_OFFSET
    pages_fetched = 0
    total_downloaded = 0
    session = requests.Session() # Use a session for potential connection reuse & headers

    # Set headers for the session (applies to all requests made with this session)
    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9", # Keep it general
        "Authorization": AUTH_TOKEN,
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "Origin": "https://explore.dawn-chorus.org", # Crucial header
        "Referer": "https://explore.dawn-chorus.org/", # Crucial header
        # Mimic browser User-Agent
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" # Or use the one from your request
    }
    session.headers.update(headers)


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
            # It's possible to get an empty list before the *very* end if a page has only private records filtered out
            # A more robust check might be needed if the API guarantees *something* until the absolute end
            print("No more records found in the response for this offset. Download likely complete.")
            break

        print(f"Found {len(records)} records on this page.")
        pages_fetched += 1
        page_download_count = 0

        for record in records:
            record_id = record.get("id")
            record_data = record.get("data", {}) # Get the 'data' sub-dictionary

            if not isinstance(record_data, dict): # Basic check if data is missing or wrong type
                 print(f"  Skipping record ID {record_id}: 'data' field is missing or not a dictionary.")
                 continue

            audio_url = record_data.get("audio")

            if not audio_url:
                print(f"  Skipping record ID {record_id}: No 'audio' URL found in record data.")
                continue

            # --- Construct Filename ---
            species_list = record_data.get("species", [])
            # Ensure species names are sanitized
            species_str = sanitize_filename("_".join(species_list)) if species_list else "UnknownSpecies"
            # Sanitize date/time - handle potential None
            date_time_raw = record_data.get("date_time", "UnknownDate")
            date_str = "UnknownDate"
            if date_time_raw:
                 date_str = date_time_raw.split("T")[0] # Get YYYY-MM-DD part

            city_str = sanitize_filename(record_data.get("city", "UnknownCity"))
            country_str = sanitize_filename(record_data.get("country", "UnknownCountry"))

            filename_base = f"{country_str}_{city_str}_{species_str}_{record_id}_{date_str}"

            # Extract file extension safely
            try:
                file_extension = os.path.splitext(audio_url)[1]
                if not file_extension or len(file_extension) > 5: # Basic sanity check
                    print(f"    Warning: Unusual file extension '{file_extension}' for URL {audio_url}. Defaulting to .wav")
                    file_extension = ".wav"
            except Exception:
                 print(f"    Warning: Could not parse extension from URL {audio_url}. Defaulting to .wav")
                 file_extension = ".wav" # Default if extraction fails


            filename = f"{filename_base}{file_extension}"
            filepath = os.path.join(DOWNLOAD_DIR, filename)

            # --- Download ---
            if os.path.exists(filepath):
                print(f"  Skipping record ID {record_id}: File already exists ({filepath})")
            else:
                if download_audio(audio_url, filepath):
                    total_downloaded += 1
                    page_download_count += 1
                    time.sleep(0.5) # Small delay between downloads
                else:
                    # Log failure and continue with the next record
                    print(f"  Failed to download record ID {record_id}. Continuing...")
                    # Optional: Add failed URLs to a list for later retry
                    time.sleep(2) # Longer pause after a failure

        # --- Prepare for next page ---
        # IMPORTANT: Increment offset by the number of records *received* on the page
        # This handles cases where the last page might have fewer than RECORDS_PER_PAGE items.
        if len(records) == 0 and page_download_count == 0:
             print("Received an empty list of records, assuming end of data.")
             break # Exit loop if we received an empty list

        current_offset += len(records) # Increment offset correctly
        print(f"Total audio files downloaded so far: {total_downloaded}")

        # Be polite to the server - wait before fetching the next page
        time.sleep(2) # Wait 2 seconds between page requests

    print(f"\nFinished. Total audio files downloaded in this run: {total_downloaded}")