import argparse
import os
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from utils.download_utils import download_file, ensure_directory


def find_and_download_files(base_url: str, output_dir: str) -> tuple[int, int]:
    """Find and download all files with the specified extension from a URL.

    Returns:
        Tuple of (links_found, links_downloaded)
    """
    print(f"[*] Fetching page: {base_url}")
    try:
        response = requests.get(base_url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[!] Error fetching base URL {base_url}: {e}")
        return (0, 0)

    print("[*] Parsing HTML content...")
    soup = BeautifulSoup(response.text, "html.parser")

    links_found = 0
    links_downloaded = 0

    # Ensure output directory exists
    output_path = ensure_directory(output_dir)

    # Find all 'a' tags with an 'href' attribute
    for link in soup.find_all("a", href=True):
        href = link["href"]

        # Construct the absolute URL
        # urljoin handles both absolute (http://...) and relative (/path/to/file, file.ext) links
        absolute_url = urljoin(base_url, href)
        print(f"-> Found link: {absolute_url}")

        # Check if the absolute URL contains the desired extension (case-insensitive)
        link_url = urlparse(absolute_url)
        filename = os.path.basename(link_url.path)
        if not filename or not filename.endswith((".ogg", ".wav", ".mp3", ".flac")):
            continue

        links_found += 1
        local_filepath = output_path / filename

        print(f"\n[*] Found potential file: {absolute_url}")
        if download_file(absolute_url, local_filepath):
            print(f"--> Successfully saved to {local_filepath}")
            links_downloaded += 1

    return links_found, links_downloaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all files with a specific extension from a given URL.",
    )
    parser.add_argument("url", help="The URL of the page containing links to files.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="datasets",
        help="Directory to save downloaded files (default: ./datasets).",
    )

    args = parser.parse_args()
    print(args.output_dir)
    links_found, links_downloaded = find_and_download_files(args.url, args.output_dir)

    print("\n[*] --- Summary ---")
    if links_found == 0:
        print(f"[*] No links ending with '{args.extension}' found on {args.url}.")
    else:
        print(f"[*] Found {links_found} potential link(s).")
        print(
            f"[*] Successfully downloaded {links_downloaded} file(s) to '{args.output_dir}'.",
        )
    print("[*] Done.")


if __name__ == "__main__":
    main()
