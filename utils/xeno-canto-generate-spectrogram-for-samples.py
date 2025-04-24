import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

# --- Configuration ---
TARGET_SAMPLE_RATE = 32000
SEGMENT_DURATION = 5  # seconds

# Mel Spectrogram Parameters (Set B - High Freq Detail, Max Mels)
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256  # Maximized Mels
FMIN = 0  # Minimum frequency
FMAX = TARGET_SAMPLE_RATE / 2  # Maximum frequency (Nyquist)
POWER = 2.0  # Exponent for the magnitude spectrogram

# S3 Configuration
S3_BUCKET_PREFIX = "s3://alexdong-bioacoustics/xeno-canto/"


# --- Helper Functions ---


def run_cli_command(
    command_list: list[str],
    command_desc: str = "Command",
    timeout: int = 300,
) -> tuple[bool, str]:
    """Runs a generic command line command using subprocess."""
    try:
        subprocess.run(
            command_list,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return True, ""
    except subprocess.TimeoutExpired:
        # print(f"ERROR: {command_desc} timed out.")
        # print(f"  Command: {' '.join(command_list)}")
        return False, "Timeout"
    except subprocess.CalledProcessError as e:
        (e.stderr or "No stderr captured")[:1000] + (
            "..." if e.stderr and len(e.stderr) > 1000 else ""
        )
        # print(error_msg)
        # print(f"  Command: {' '.join(e.cmd)}")
        # print(f"  Stderr: {stderr_preview}")
        return False, f"{command_desc} Error (Code {e.returncode})"
    except FileNotFoundError:
        # Specifically handle aws cli not found
        if command_list[0] == "aws":
            print(
                "\nERROR: 'aws' command not found. Is AWS CLI installed and in your PATH?",
            )
            # This should likely stop the execution, but for parallel processing, return False
            return False, "AWS CLI not found"
        else:
            # print(f"ERROR: Command not found: {command_list[0]}")
            return False, f"Command not found: {command_list[0]}"
    except Exception as e:
        # print(error_msg)
        # print(f"  Command: {' '.join(command_list)}")
        return False, f"Unexpected Error: {e}"


def generate_spectrogram(
    audio_path: str,
    output_png_path: str,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: int | float,
    fmax: int | float,
    power: float,
) -> tuple[bool, str]:
    """Loads an audio segment, generates a Mel spectrogram, and saves as PNG."""
    try:
        waveform, sr = torchaudio.load(audio_path)

        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:  # Ensure mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if waveform.numel() == 0:
            return False, "Empty waveform"

        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=power,
            normalized=False,
        )
        amplitude_to_db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=80,
        )

        mel_spec = mel_spectrogram_transform(waveform)
        mel_spec_db = amplitude_to_db_transform(mel_spec)

        plot_spec = mel_spec_db.squeeze(0).cpu().numpy()

        if not np.isfinite(plot_spec).any():
            return False, "Spectrogram contains only non-finite values (NaN/Inf)"

        if np.isinf(plot_spec).any():
            finite_min = np.nanmin(plot_spec[np.isfinite(plot_spec)])
            if np.isfinite(finite_min):
                plot_spec[np.isinf(plot_spec)] = finite_min
            else:
                plot_spec[np.isinf(plot_spec)] = -80.0

        fig, ax = plt.subplots(1, 1)
        ax.imshow(plot_spec, aspect="auto", origin="lower", cmap="magma")
        plt.axis("off")
        fig.patch.set_alpha(0)
        plt.savefig(
            output_png_path,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
        plt.close(fig)

        return True, ""

    except FileNotFoundError:
        return False, "Audio file not found"
    except RuntimeError as e:
        err_str = str(e).lower()
        if (
            "requires output shape of" in err_str
            or "invalid audio data" in err_str
            or "input tensor must have" in err_str
            or "frame_length must be" in err_str
        ):
            return False, "Torchaudio Runtime Error (Likely too short/corrupt)"
        else:
            return False, f"Torchaudio Runtime Error: {e}"
    except Exception as e:
        return False, f"Spectrogram Error: {e}"


# --- Download Function ---


def download_s3_file(
    recording_id: str,
    json_dir: str,
    s3_bucket_prefix: str,
    mp3_dir: str,
    skip_existing: bool = True,
) -> dict:
    """Downloads a single MP3 file from S3 based on its JSON metadata."""
    status = {"id": recording_id, "success": False, "reason": "", "skipped": False}
    json_path = os.path.join(json_dir, f"{recording_id}.json")
    local_mp3_path = os.path.join(mp3_dir, f"{recording_id}.mp3")

    if (
        skip_existing
        and os.path.exists(local_mp3_path)
        and os.path.getsize(local_mp3_path) > 0
    ):
        status["success"] = True
        status["skipped"] = True
        status["reason"] = "File exists"
        return status

    if not os.path.exists(json_path):
        status["reason"] = "JSON not found"
        return status

    try:
        with open(json_path, "r", encoding="utf-8") as f_json:
            metadata = json.load(f_json)

        s3_filename = metadata.get("file-name")
        if not s3_filename or s3_filename == "null":
            status["reason"] = "Invalid/missing 'file-name' in JSON"
            return status

        s3_path = f"{s3_bucket_prefix.rstrip('/')}/{s3_filename}"
        # Use --only-show-errors to mimic shell script behavior
        download_command = [
            "aws",
            "s3",
            "cp",
            "--only-show-errors",
            s3_path,
            local_mp3_path,
        ]

        success, reason = run_cli_command(download_command, "S3 Download")
        if success:
            # Verify file was created and isn't empty
            if os.path.exists(local_mp3_path) and os.path.getsize(local_mp3_path) > 0:
                status["success"] = True
            else:
                status["success"] = False
                status["reason"] = (
                    "Download command succeeded but file is missing/empty"
                )
                if os.path.exists(local_mp3_path):  # Clean up zero-byte file
                    try:
                        os.remove(local_mp3_path)
                    except OSError:
                        pass
        else:
            status["reason"] = f"Download failed: {reason}"
            # Attempt to clean up potentially partial file
            if os.path.exists(local_mp3_path):
                try:
                    os.remove(local_mp3_path)
                except OSError:
                    pass

    except json.JSONDecodeError:
        status["reason"] = "JSON decode error"
    except Exception as e:
        status["reason"] = f"Error processing JSON/Download: {e}"

    return status


# --- Audio Processing Function ---


def process_audio_files(
    rec_id: str,
    species_name: str,
    mp3_dir: str,
    output_dir: str,
    temp_dir: str,
) -> dict:
    """Processes a single recording: convert, segment, generate spectrograms."""
    species_name = species_name.replace("-", "_")

    mp3_path = os.path.join(mp3_dir, f"{rec_id}.mp3")
    # Use a unique temp OGG name within the shared temp dir
    temp_ogg_path = os.path.join(temp_dir, f"{species_name}-{rec_id}_temp.ogg")
    status = {
        "id": rec_id,
        "species": species_name,
        "success": False,
        "reason": "",
        "segments_created": 0,
        "specs_created": 0,
        "spec_errors": 0,
    }

    # Double-check MP3 existence, though it should exist after download phase
    if not os.path.exists(mp3_path):
        status["reason"] = "MP3 not found for processing"
        return status

    # Step A: Convert MP3 to Temporary OGG (Mono, 32kHz)
    convert_command = [
        "ffmpeg",
        "-i",
        mp3_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "libvorbis",
        "-q:a",
        "4",
        temp_ogg_path,
    ]
    success, reason = run_cli_command(convert_command, "MP3->OGG conversion")
    if not success:
        status["reason"] = f"Conversion failed: {reason}"
        if os.path.exists(temp_ogg_path):  # Clean up
            try:
                os.remove(temp_ogg_path)
            except OSError:
                pass
        return status

    # Step B: Segment the OGG file directly into the final output directory
    segment_pattern = os.path.join(output_dir, f"{species_name}-{rec_id}-%04d.ogg")
    segment_command = [
        "ffmpeg",
        "-i",
        temp_ogg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "segment",
        "-segment_time",
        str(SEGMENT_DURATION),
        "-c",
        "copy",
        "-reset_timestamps",
        "1",
        segment_pattern,
    ]
    success, reason = run_cli_command(segment_command, "OGG segmentation")

    # Clean up temporary full OGG file *regardless* of segmentation success/failure here
    if os.path.exists(temp_ogg_path):
        try:
            os.remove(temp_ogg_path)
        except OSError as e:
            print(f"\nWarning: Could not remove temp file {temp_ogg_path}: {e}")

    if not success:
        status["reason"] = f"Segmentation failed: {reason}"
        return status

    # Step C: Generate Spectrograms for each segment
    segments_found = []
    try:
        prefix = f"{species_name}-{rec_id}-"
        suffix = ".ogg"
        for filename in os.listdir(output_dir):
            if filename.startswith(prefix) and filename.endswith(suffix):
                try:
                    int(filename[len(prefix) : -len(suffix)])
                    segments_found.append(filename)
                except ValueError:
                    continue
    except Exception as e:
        status["reason"] = f"Error listing segments: {e}"
        return status

    status["segments_created"] = len(segments_found)
    specs_created_count = 0
    spec_errors_count = 0

    for segment_file in segments_found:
        segment_path = os.path.join(output_dir, segment_file)
        segment_basename = os.path.splitext(segment_file)[0]
        spec_path = os.path.join(output_dir, f"{segment_basename}.png")

        spec_success, spec_reason = generate_spectrogram(
            segment_path,
            spec_path,
            TARGET_SAMPLE_RATE,
            N_FFT,
            HOP_LENGTH,
            N_MELS,
            FMIN,
            FMAX,
            POWER,
        )
        if spec_success:
            specs_created_count += 1
        else:
            spec_errors_count += 1

    status["specs_created"] = specs_created_count
    status["spec_errors"] = spec_errors_count
    status["success"] = True  # Mark as success if conversion and segmentation worked
    status["reason"] = ""

    if (
        status["segments_created"] == 0 and status["success"]
    ):  # Check again after finding segments
        status["success"] = False
        status["reason"] = "Segmentation command succeeded but 0 files found"

    return status


# --- Main Execution ---
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download S3 files, convert, segment, generate spectrograms in parallel.",
    )
    parser.add_argument(
        "--input-list-file",
        default="datasets/xeno-canto-brazil-small.txt",
        help="Path to the text file containing 'species|recording_id1,id2,...'",
    )
    parser.add_argument(
        "--json-dir",
        default="./datasets/xeno-canto/",
        help="Directory containing the individual recording JSON metadata files.",
    )
    parser.add_argument(
        "--mp3-dir",
        default="./datasets/xeno-canto-brazil-small",
        help="Directory to save downloaded MP3 files.",
    )
    parser.add_argument(
        "--output-dir",
        default="./datasets/xeno-canto-brazil-small",
        help="Directory to save the final segmented OGG and spectrogram PNG files.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max number of parallel processes (default: number of CPU cores).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to the first N recordings (for testing).",
    )
    parser.add_argument(
        "--skip-existing-mp3",
        action="store_true",
        help="Skip downloading MP3s if they already exist.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the S3 download phase entirely (assumes MP3s exist).",
    )

    args = parser.parse_args()

    start_total_time = time.time()

    # --- 1. Read Input File and Prepare Data ---
    print(f"Reading recording IDs and species map from: {args.input_list_file}")
    id_to_species_map = {}
    all_recording_ids = set()
    try:
        with open(args.input_list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                species_part, ids_part = line.split("|", 1)
                # Replace problematic characters for filenames
                species_name = (
                    species_part.strip()
                    .replace(" ", "_")
                    .replace("/", "_")
                    .replace("\\", "_")
                )
                if ids_part:
                    ids_list = [
                        rec_id.strip()
                        for rec_id in ids_part.split(",")
                        if rec_id.strip()
                    ]
                    all_recording_ids.update(ids_list)
                    for rec_id in ids_list:
                        id_to_species_map[rec_id] = species_name
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {args.input_list_file}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read input file {args.input_list_file}: {e}")
        sys.exit(1)

    if not all_recording_ids:
        print("ERROR: No recording IDs found in the input file.")
        sys.exit(1)

    sorted_ids = sorted(list(all_recording_ids))

    if args.limit:
        print(f"Limiting processing to the first {args.limit} recordings.")
        sorted_ids = sorted_ids[: args.limit]

    print(f"Found {len(sorted_ids)} unique recording IDs to process.")
    recordings_to_process = set(sorted_ids)  # Start with all selected IDs

    # --- 2. Ensure Output Directories Exist ---
    os.makedirs(args.mp3_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Ensured MP3 download directory exists: {args.mp3_dir}")
    print(f"Ensured final output directory exists: {args.output_dir}")

    # --- 3. Download Phase ---
    if not args.skip_download:
        print("\n--- Starting S3 Download Phase (Parallel) ---")
        download_tasks = []
        for rec_id in sorted_ids:
            # Check if ID has species mapping, required for JSON path logic potentially
            if rec_id in id_to_species_map:
                download_tasks.append(
                    (
                        rec_id,
                        args.json_dir,
                        S3_BUCKET_PREFIX,
                        args.mp3_dir,
                        args.skip_existing_mp3,
                    ),
                )
            else:
                print(
                    f"Warning: No species mapping for {rec_id}, cannot determine JSON path if needed. Skipping download.",
                )

        download_results = []
        download_errors = 0
        download_skips = 0
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(download_s3_file, *task): task
                for task in download_tasks
            }
            for future in tqdm(
                as_completed(futures),
                total=len(download_tasks),
                desc="Downloading MP3s",
            ):
                try:
                    result = future.result()
                    download_results.append(result)
                    if not result["success"]:
                        download_errors += 1
                        if result["skipped"]:
                            download_skips += (
                                1  # Count skips separately from actual errors
                            )
                            download_errors -= 1  # Don't count skip as error
                        # else:
                        #     print(f"\nDownload Error (ID: {result['id']}): {result['reason']}")
                except Exception as e:
                    rec_id = futures[future][0]
                    download_results.append(
                        {
                            "id": rec_id,
                            "success": False,
                            "reason": f"Task level error: {e}",
                            "skipped": False,
                        },
                    )
                    download_errors += 1
                    # print(f"\nDownload Task Error (ID: {rec_id}): {e}")

        print("\n--- Download Summary ---")
        print(f"Tasks attempted: {len(download_tasks)}")
        successful_downloads = sum(
            1 for r in download_results if r["success"] and not r["skipped"]
        )
        print(f"Successful downloads: {successful_downloads}")
        print(f"Skipped (already existed): {download_skips}")
        print(f"Failed downloads: {download_errors}")

        # Update the set of recordings to process based on download success/existence
        successfully_downloaded_or_skipped_ids = {
            r["id"] for r in download_results if r["success"]
        }
        recordings_to_process = recordings_to_process.intersection(
            successfully_downloaded_or_skipped_ids,
        )
        print(
            f"Proceeding to process {len(recordings_to_process)} recordings with available MP3s.",
        )
        if not recordings_to_process:
            print(
                "No recordings available for processing after download phase. Exiting.",
            )
            sys.exit(0)
    else:
        print("\n--- Skipping S3 Download Phase ---")
        # Assume all requested IDs have MP3s available
        print(
            f"Proceeding to process {len(recordings_to_process)} recordings (assuming MP3s exist).",
        )

    # --- 4. Audio Processing Phase ---
    # Create a shared temporary directory for intermediate OGGs for this phase
    audio_temp_dir = tempfile.mkdtemp(prefix="audio_proc_temp_")
    print("\n--- Starting Audio Processing Phase (Parallel) ---")
    print(f"Created temporary directory for OGG conversion: {audio_temp_dir}")

    audio_tasks = []
    for rec_id in sorted(
        list(recordings_to_process),
    ):  # Process only those confirmed available
        species_name = id_to_species_map.get(rec_id)
        if not species_name:
            print(
                f"Warning: No species name found for processing recording ID {rec_id}. Skipping.",
            )
            continue
        audio_tasks.append(
            (rec_id, species_name, args.mp3_dir, args.output_dir, audio_temp_dir),
        )

    audio_results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_audio_files, *task): task for task in audio_tasks
        }
        for future in tqdm(
            as_completed(futures),
            total=len(audio_tasks),
            desc="Processing Audio",
        ):
            try:
                result = future.result()
                audio_results.append(result)
                # Optionally print errors immediately
                # if not result["success"]:
                #      print(f"\nAudio Processing Error (ID: {result['id']}): {result['reason']}")
            except Exception as e:
                rec_id = futures[future][0]  # Get rec_id from original task tuple
                audio_results.append(
                    {
                        "id": rec_id,
                        "species": "unknown",
                        "success": False,
                        "reason": f"Task level error: {e}",
                        "segments_created": 0,
                        "specs_created": 0,
                        "spec_errors": 0,
                    },
                )
                # print(f"\nAudio Processing Task Error (ID: {rec_id}): {e}")

    # --- 5. Summarize Audio Processing Results ---
    print("\n--- Audio Processing Summary ---")
    successful_audio_proc = 0
    total_segments = 0
    total_specs = 0
    total_spec_errors = 0
    audio_failed_reasons = {}

    for res in audio_results:
        if res["success"]:
            successful_audio_proc += 1
            total_segments += res["segments_created"]
            total_specs += res["specs_created"]
            total_spec_errors += res["spec_errors"]
        else:
            reason = res.get("reason", "Unknown error")
            audio_failed_reasons[reason] = audio_failed_reasons.get(reason, 0) + 1

    print(f"Total recordings attempted: {len(audio_tasks)}")
    print(f"Successfully processed (Conversion+Segmentation): {successful_audio_proc}")
    print(f"Total 5s segments created: {total_segments}")
    print(f"Total spectrograms generated: {total_specs}")
    print(f"Spectrogram generation errors/warnings: {total_spec_errors}")
    print(f"Failed audio processing: {len(audio_tasks) - successful_audio_proc}")
    if audio_failed_reasons:
        print("Audio Processing Failure Reasons:")
        # Sort reasons for consistent output
        for reason, count in sorted(audio_failed_reasons.items()):
            print(f"  - {reason}: {count}")

    # --- 6. Clean up Temporary Directory ---
    try:
        print(f"Cleaning up temporary directory: {audio_temp_dir}")
        shutil.rmtree(audio_temp_dir)
    except Exception as e:
        print(f"Warning: Could not remove temporary directory {audio_temp_dir}: {e}")

    end_total_time = time.time()
    print(f"\nTotal execution time: {end_total_time - start_total_time:.2f} seconds")


if __name__ == "__main__":
    # Add check for soundfile if needed by torchaudio backend
    try:
        import importlib.util

        has_soundfile = importlib.util.find_spec("soundfile") is not None
    except ImportError:
        has_soundfile = False
    main()
