"""
Machine Learning Dataset Directory Analyzer

Purpose:
Analyzes directory structures of ML training datasets to provide insights about:
- Class distribution (subdirectory file counts)
- File type distribution
- Dataset structure visualization
- Basic statistical summary
- Audio file properties (sample rate, duration, etc.)

Usage:
python ml_dataset_analyzer.py <dataset_root> [--graph] [--output <filename>] [--audio] [--max-files <num>]

Example:
python ml_dataset_analyzer.py ./train_dataset --graph --audio --output dataset_report.png
"""

import argparse
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

# Import librosa for audio analysis
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def analyze_audio_file(file_path: str) -> Dict[str, Any]:
    """
    Analyzes a single audio file and returns its properties

    Args:
        file_path (str): Path to the audio file

    Returns:
        dict: Audio properties including sample rate, duration, etc.
    """
    try:
        y, sr = librosa.load(
            file_path, sr=None, duration=10,
        )  # Load first 10 seconds for efficiency
        duration = librosa.get_duration(y=y, sr=sr)

        # Calculate basic audio properties
        properties = {
            "sample_rate": sr,
            "duration": duration,
            "channels": 1 if y.ndim == 1 else y.shape[0],
            "samples": len(y),
        }

        return properties
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")
        return {}


def analyze_ml_dataset(root_dir: str, analyze_audio: bool = False, max_files_per_class: int = 100) -> tuple[dict, dict, dict, dict]:
    """
    Analyzes an ML dataset directory structure and file distribution

    Args:
        root_dir (str): Path to root directory of dataset (should contain class directories)
        analyze_audio (bool): Whether to analyze audio properties
        max_files_per_class (int): Maximum number of files to analyze per class

    Returns:
        tuple: (class_counts, file_types, stats, audio_stats)
        - class_counts: Dict of {class_name: file_count}
        - file_types: Dict of {file_extension: count}
        - stats: Dict of summary statistics
        - audio_stats: Dict of audio statistics (if analyze_audio=True)
    """
    class_counts = {}
    file_types = defaultdict(int)
    total_files = 0
    file_counts = []

    # Audio statistics containers
    audio_files_analyzed = 0
    sample_rates = []
    durations = []
    audio_stats = {}

    # Analyze directory structure
    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        if os.path.isdir(class_path):
            file_list = [
                f
                for f in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, f))
            ]
            count = len(file_list)
            class_counts[class_dir] = count
            file_counts.append(count)
            total_files += count

            # Analyze file types
            for f in file_list:
                _, ext = os.path.splitext(f)
                file_types[ext.lower()] += 1

            # Analyze audio properties for a subset of files
            if analyze_audio and LIBROSA_AVAILABLE:
                audio_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
                audio_files = [
                    f
                    for f in file_list
                    if os.path.splitext(f)[1].lower() in audio_extensions
                ]

                # Limit the number of files to analyze per class
                sample_files = audio_files[:max_files_per_class]

                with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                    file_paths = [os.path.join(class_path, f) for f in sample_files]
                    results = list(executor.map(analyze_audio_file, file_paths))

                    for result in results:
                        if result:  # Skip empty results (errors)
                            audio_files_analyzed += 1
                            sample_rates.append(result["sample_rate"])
                            durations.append(result["duration"])

    # Calculate statistics
    stats = {
        "total_classes": len(class_counts),
        "total_files": total_files,
        "min_files": min(file_counts) if file_counts else 0,
        "max_files": max(file_counts) if file_counts else 0,
        "median_files": (
            sorted(file_counts)[len(file_counts) // 2] if file_counts else 0
        ),
        "avg_files": total_files / len(class_counts) if class_counts else 0,
    }

    # Calculate audio statistics
    if analyze_audio and LIBROSA_AVAILABLE and audio_files_analyzed > 0:
        audio_stats = {
            "files_analyzed": audio_files_analyzed,
            "sample_rates": {
                "min": min(sample_rates) if sample_rates else 0,
                "max": max(sample_rates) if sample_rates else 0,
                "most_common": (
                    max(set(sample_rates), key=sample_rates.count)
                    if sample_rates
                    else 0
                ),
            },
            "durations": {
                "min": min(durations) if durations else 0,
                "max": max(durations) if durations else 0,
                "mean": np.mean(durations) if durations else 0,
                "median": np.median(durations) if durations else 0,
                "total": sum(durations) if durations else 0,
            },
        }

    return class_counts, dict(file_types), stats, audio_stats


def generate_ml_report(
    class_counts: dict, file_types: dict, stats: dict, audio_stats: dict | None = None, output_file: str = "dataset_report.png",
) -> None:
    """
    Generates visualization report for ML dataset analysis

    Args:
        class_counts (dict): Class distribution dictionary
        file_types (dict): File type distribution dictionary
        stats (dict): Dataset statistics dictionary
        audio_stats (dict): Audio statistics dictionary
        output_file (str): Output filename for the report
    """
    # Determine number of subplots based on whether we have audio stats
    num_plots = 3 if audio_stats and audio_stats.get("files_analyzed", 0) > 0 else 2

    plt.figure(figsize=(15, 10))

    # Class distribution plot
    plt.subplot(1, num_plots, 1)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_classes) if sorted_classes else ([], [])
    plt.barh(classes[-20:], counts[-20:])  # Show top 20 classes
    plt.title("Class Distribution (Top 20)")
    plt.xlabel("Number of Samples")
    plt.gca().invert_yaxis()

    # File type distribution plot
    plt.subplot(1, num_plots, 2)
    file_labels = [ext if ext else "no_extension" for ext in file_types.keys()]
    plt.pie(file_types.values(), labels=file_labels, autopct="%1.1f%%")
    plt.title("File Type Distribution")

    # Audio statistics plot (if available)
    if audio_stats and audio_stats.get("files_analyzed", 0) > 0:
        plt.subplot(1, num_plots, 3)

        # Create a duration histogram
        durations = [
            d for d in audio_stats["durations"].values() if isinstance(d, (int, float))
        ]
        if isinstance(audio_stats["durations"], dict):
            # If durations is a dict with statistics
            durations = []
        else:
            # If durations is a list of actual durations
            durations = audio_stats["durations"]

        if durations:
            plt.hist(durations, bins=20)
            plt.title("Audio Duration Distribution")
            plt.xlabel("Duration (seconds)")
            plt.ylabel("Number of Files")

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Dataset report saved as {output_file}")


def main() -> None:
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description="ML Dataset Analysis Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_root", help="Root directory of the ML training dataset",
    )
    parser.add_argument(
        "--graph", action="store_true", help="Generate visual analysis report",
    )
    parser.add_argument(
        "--output",
        default="dataset_report.png",
        help="Output filename for the generated report",
    )
    parser.add_argument(
        "--audio", action="store_true", help="Analyze audio file properties",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=100,
        help="Maximum number of audio files to analyze per class",
    )
    args = parser.parse_args()

    # Validate dataset directory
    if not os.path.isdir(args.dataset_root):
        print(f"Error: Invalid dataset directory '{args.dataset_root}'")
        return

    # Check for librosa if audio analysis is requested
    if args.audio and not LIBROSA_AVAILABLE:
        print("Warning: librosa is required for audio analysis but not installed.")
        print("Install with: pip install librosa")
        print("Continuing without audio analysis...")
        args.audio = False

    # Perform analysis
    print(f"Analyzing dataset at {args.dataset_root}...")
    if args.audio:
        print("Audio analysis enabled (this may take some time)...")

    class_counts, file_types, stats, audio_stats = analyze_ml_dataset(
        args.dataset_root, analyze_audio=args.audio, max_files_per_class=args.max_files,
    )

    # Print text report
    print("\n=== ML Dataset Analysis Report ===")
    print(f"Dataset location: {os.path.abspath(args.dataset_root)}")
    print(f"\nClass Distribution ({stats['total_classes']} classes):")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[
        :20
    ]:  # Show top 20
        print(f"- {cls}: {count} samples")

    if stats["total_classes"] > 20:
        print(f"  ... and {stats['total_classes'] - 20} more classes")

    print(f"\nFile Type Distribution ({stats['total_files']} total files):")
    for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
        print(f"- {ext if ext else 'no_ext'}: {count} files")

    print("\nStatistics:")
    print(f"- Minimum samples per class: {stats['min_files']}")
    print(f"- Maximum samples per class: {stats['max_files']}")
    print(f"- Average samples per class: {stats['avg_files']:.1f}")
    print(f"- Median samples per class: {stats['median_files']}")

    # Print audio statistics if available
    if args.audio and audio_stats:
        print("\nAudio Statistics:")
        print(f"- Files analyzed: {audio_stats.get('files_analyzed', 0)}")

        if audio_stats.get("files_analyzed", 0) > 0:
            sr = audio_stats["sample_rates"]
            dur = audio_stats["durations"]

            print("\nSample Rates:")
            print(f"- Min: {sr['min']} Hz")
            print(f"- Max: {sr['max']} Hz")
            print(f"- Most common: {sr['most_common']} Hz")

            print("\nDurations:")
            print(f"- Min: {dur['min']:.2f} seconds")
            print(f"- Max: {dur['max']:.2f} seconds")
            print(f"- Mean: {dur['mean']:.2f} seconds")
            print(f"- Median: {dur['median']:.2f} seconds")
            print(f"- Total duration: {dur['total']/3600:.2f} hours")

    # Generate visual report
    if args.graph:
        try:
            generate_ml_report(
                class_counts, file_types, stats, audio_stats, args.output,
            )
        except ImportError:
            print("Error: matplotlib is required for graph generation")
            print("Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
