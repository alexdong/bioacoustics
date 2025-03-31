import argparse
import os
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def download_file(url: str, folder_path: str) -> bool:
    """Downloads a single file from a URL into the specified folder."""
    try:
        # Ensure the folder exists
        os.makedirs(folder_path, exist_ok=True)

        # Get the filename from the URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            print(f"-> Could not determine filename for URL: {url}. Skipping.")
            return False

        local_filepath = os.path.join(folder_path, filename)

        print(f"--> Downloading {filename} from {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            with open(local_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"--> Successfully saved to {local_filepath}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"--> ERROR downloading {url}: {e}")
    except OSError as e:
        print(f"--> ERROR saving file {local_filepath}: {e}")
    except Exception as e:
         print(f"--> An unexpected error occurred for {url}: {e}")
    return False

def main() -> None:
    parser = argparse.ArgumentParser(description="Download all files with a specific extension from a given URL.")
    parser.add_argument("url", help="The URL of the page containing links to files.")
    parser.add_argument("extension", help="The file extension to download (e.g., .mp3, .pdf, *.jpg).")
    parser.add_argument("-o", "--output-dir", default="downloads",
                        help="Directory to save downloaded files (default: ./downloads).")

    args = parser.parse_args()

    base_url = args.url
    output_dir = args.output_dir
    file_extension = args.extension

    # Standardize extension (remove *, ensure leading .)
    if file_extension.startswith("*"):
        file_extension = file_extension[1:]
    if not file_extension.startswith("."):
        file_extension = "." + file_extension
    file_extension = file_extension.lower() # Make comparison case-insensitive

    print(f"[*] Fetching page: {base_url}")
    try:
        response = requests.get(base_url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[!] Error fetching base URL {base_url}: {e}")
        return

    print("[*] Parsing HTML content...")
    soup = BeautifulSoup(response.text, 'html.parser')

    links_found = 0
    links_downloaded = 0

    print(f"[*] Searching for links ending with '{file_extension}'...")
    # Find all 'a' tags with an 'href' attribute
    for link in soup.find_all('a', href=True):
        href = link['href']

        # Construct the absolute URL
        # urljoin handles both absolute (http://...) and relative (/path/to/file, file.ext) links
        absolute_url = urljoin(base_url, href)

        # Check if the absolute URL ends with the desired extension (case-insensitive)
        parsed_link_url = urlparse(absolute_url)
        if parsed_link_url.path.lower().endswith(file_extension):
            links_found += 1
            print(f"\n[*] Found potential file: {absolute_url}")
            if download_file(absolute_url, output_dir):
                links_downloaded += 1
        # Optional: Add checks for other tags like <img> src, <audio> src etc. if needed
        # Example for images:
        # elif soup.find_all('img', src=True) and file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
        #     src = link['src']
        #     absolute_url = urljoin(base_url, src)
        #     ... check and download ...


    print("\n[*] --- Summary ---")
    if links_found == 0:
        print(f"[*] No links ending with '{file_extension}' found on {base_url}.")
    else:
        print(f"[*] Found {links_found} potential link(s).")
        print(f"[*] Successfully downloaded {links_downloaded} file(s) to '{output_dir}'.")
    print("[*] Done.")


if __name__ == "__main__":
    main()
