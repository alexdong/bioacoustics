"""
Machine Learning Dataset Directory Analyzer

Purpose:
Analyzes directory structures of ML training datasets to provide insights about:
- Class distribution (subdirectory file counts)
- File type distribution
- Dataset structure visualization
- Basic statistical summary

Usage:
python ml_dataset_analyzer.py <dataset_root> [--graph] [--output <filename>]

Example:
python ml_dataset_analyzer.py ./train_dataset --graph --output dataset_report.png
"""

import os
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_ml_dataset(root_dir):
    """
    Analyzes an ML dataset directory structure and file distribution
    
    Args:
        root_dir (str): Path to root directory of dataset (should contain class directories)
    
    Returns:
        tuple: (class_counts, file_types, stats)
        - class_counts: Dict of {class_name: file_count}
        - file_types: Dict of {file_extension: count}
        - stats: Dict of summary statistics
    """
    class_counts = {}
    file_types = defaultdict(int)
    total_files = 0
    file_counts = []

    # Analyze directory structure
    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        if os.path.isdir(class_path):
            file_list = [f for f in os.listdir(class_path) 
                       if os.path.isfile(os.path.join(class_path, f))]
            count = len(file_list)
            class_counts[class_dir] = count
            file_counts.append(count)
            total_files += count

            # Analyze file types
            for f in file_list:
                _, ext = os.path.splitext(f)
                file_types[ext.lower()] += 1

    # Calculate statistics
    stats = {
        'total_classes': len(class_counts),
        'total_files': total_files,
        'min_files': min(file_counts) if file_counts else 0,
        'max_files': max(file_counts) if file_counts else 0,
        'median_files': sorted(file_counts)[len(file_counts)//2] if file_counts else 0,
        'avg_files': total_files/len(class_counts) if class_counts else 0
    }

    return class_counts, dict(file_types), stats

def generate_ml_report(class_counts, file_types, stats, output_file='dataset_report.png'):
    """
    Generates visualization report for ML dataset analysis
    
    Args:
        class_counts (dict): Class distribution dictionary
        file_types (dict): File type distribution dictionary
        stats (dict): Dataset statistics dictionary
        output_file (str): Output filename for the report
    """
    plt.figure(figsize=(15, 8))

    # Class distribution plot
    plt.subplot(1, 2, 1)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_classes) if sorted_classes else ([], [])
    plt.barh(classes[-20:], counts[-20:])  # Show top 20 classes
    plt.title('Class Distribution (Top 20)')
    plt.xlabel('Number of Samples')
    plt.gca().invert_yaxis()

    # File type distribution plot
    plt.subplot(1, 2, 2)
    file_labels = [ext if ext else 'no_extension' for ext in file_types.keys()]
    plt.pie(file_types.values(), labels=file_labels, autopct='%1.1f%%')
    plt.title('File Type Distribution')

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Dataset report saved as {output_file}")

def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description="ML Dataset Analysis Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('dataset_root', 
                      help='Root directory of the ML training dataset')
    parser.add_argument('--graph', action='store_true',
                      help='Generate visual analysis report')
    parser.add_argument('--output', default='dataset_report.png',
                      help='Output filename for the generated report')
    args = parser.parse_args()

    # Validate dataset directory
    if not os.path.isdir(args.dataset_root):
        print(f"Error: Invalid dataset directory '{args.dataset_root}'")
        return

    # Perform analysis
    class_counts, file_types, stats = analyze_ml_dataset(args.dataset_root)

    # Print text report
    print("\n=== ML Dataset Analysis Report ===")
    print(f"Dataset location: {os.path.abspath(args.dataset_root)}")
    print(f"\nClass Distribution ({stats['total_classes']} classes):")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"- {cls}: {count} samples")

    print(f"\nFile Type Distribution ({stats['total_files']} total files):")
    for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
        print(f"- {ext if ext else 'no_ext'}: {count} files")

    print("\nStatistics:")
    print(f"- Minimum samples per class: {stats['min_files']}")
    print(f"- Maximum samples per class: {stats['max_files']}")
    print(f"- Average samples per class: {stats['avg_files']:.1f}")
    print(f"- Median samples per class: {stats['median_files']}")

    # Generate visual report
    if args.graph:
        try:
            generate_ml_report(class_counts, file_types, stats, args.output)
        except ImportError:
            print("Error: matplotlib is required for graph generation")
            print("Install with: pip install matplotlib")

if __name__ == "__main__":
    main()
