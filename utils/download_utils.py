import json
import os
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Optional, Union

import requests
from mypy_boto3_s3.client import S3Client
from tqdm import tqdm

# --- Common Constants ---
DEFAULT_TIMEOUT = 60
DEFAULT_CHUNK_SIZE = 8192
DEFAULT_REQUEST_DELAY = 1.1  # Slightly more than 1 second to be safe

# --- File Handling Functions ---

def ensure_directory(directory_path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist and return Path object."""
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_file_extension(url: str, default_ext: str = ".wav") -> str:
    """Extract file extension from URL, defaulting if not found."""
    try:
        file_extension = os.path.splitext(url)[1]
        if not file_extension or len(file_extension) > 5:
            print(f"    Warning: Unusual file extension '{file_extension}' for URL {url}. Defaulting to {default_ext}")
            file_extension = default_ext
        return file_extension
    except Exception:
        print(f"    Warning: Could not parse extension from URL {url}. Defaulting to {default_ext}")
        return default_ext

# --- Request Handling ---

def create_session(headers: Optional[Dict[str, str]] = None) -> requests.Session:
    """Create and configure a requests session with appropriate headers."""
    session = requests.Session()
    if headers:
        session.headers.update(headers)
    return session

def download_file(
    url: str,
    output_path: Union[str, Path],
    session: Optional[requests.Session] = None,
    timeout: int = DEFAULT_TIMEOUT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    log_func: Callable[[str], None] = print,
) -> bool:
    """Downloads a file from URL to the specified path.
    
    Args:
        url: The URL to download from
        output_path: Path where the file should be saved
        session: Optional requests session to use
        timeout: Request timeout in seconds
        chunk_size: Size of chunks to download
        log_func: Function to use for logging (e.g., print or tqdm.write)
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Use provided session or create a new one
        req = session.get if session else requests.get
        
        with req(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
        
        return True
    
    except requests.exceptions.RequestException as e:
        log_func(f"Error downloading {url}: {e}")
        # Clean up potentially incomplete file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return False
    
    except Exception as e:
        log_func(f"Unexpected error during download of {url}: {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return False

def save_json(data: Dict[str, Any], output_path: Union[str, Path]) -> bool:
    """Save data as JSON to the specified path."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON to {output_path}: {e}")
        return False

# --- Progress Tracking ---

def create_progress_bar(
    total: int,
    desc: str = "Downloading",
    unit: str = "file",
    ncols: int = 100,
) -> tqdm:
    """Create a tqdm progress bar with standard formatting."""
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        ncols=ncols,
    )

# --- S3 Utilities ---

def check_s3_file_exists(
    s3_client: S3Client,
    bucket: str,
    s3_key: str,
) -> bool:
    """Checks if a file exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=s3_key)
        return True
    except Exception:
        return False

def upload_to_s3(
    s3_client: S3Client,
    bucket: str,
    s3_key: str,
    data: Union[bytes, BinaryIO],  # Can be bytes or file-like object
    content_type: Optional[str] = None,
    quiet: bool = False,
    log_func: Callable[[str], None] = print,
) -> bool:
    """Uploads data to a specific S3 key."""
    if not quiet:
        log_func(f"Uploading to s3://{bucket}/{s3_key}...")
    try:
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type

        # Handle both bytes and file-like objects
        if isinstance(data, bytes):
            s3_client.put_object(Bucket=bucket, Key=s3_key, Body=data, **extra_args)
        else:
            s3_client.upload_fileobj(data, bucket, s3_key, ExtraArgs=extra_args)
            
        if not quiet:
            log_func(f"Successfully uploaded to s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        log_func(f"Failed to upload {s3_key} to S3: {e}")
        return False
