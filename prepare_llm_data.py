"""
Helper script to prepare data files for LLM training
Creates train_files.txt and train_labels.txt from a dataset directory
"""

import os
import argparse
from pathlib import Path


def prepare_data_lists(
    data_dir: str,
    output_dir: str,
    file_extension: str = ".csv",
    train_ratio: float = 0.8,
):
    """
    Prepare train/eval file lists and labels
    
    Args:
        data_dir: Directory containing signal files and labels
        output_dir: Output directory for generated lists
        file_extension: File extension to look for (default: .csv)
        train_ratio: Ratio of training data (default: 0.8)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning directory: {data_dir}")
    
    # Find all signal files
    signal_files = sorted(data_dir.glob(f"*{file_extension}"))
    print(f"Found {len(signal_files)} signal files")
    
    if len(signal_files) == 0:
        print(f"❌ No files with extension '{file_extension}' found!")
        return
    
    # Prepare file-label pairs
    # Assumes labels are in a separate file or encoded in filename
    # Example 1: label_0.csv, label_1.csv, ...
    # Example 2: labels.txt with one label per line
    
    # Try to find labels file
    labels_file = data_dir / "labels.txt"
    if labels_file.exists():
        print(f"Loading labels from: {labels_file}")
        with open(labels_file, 'r') as f:
            labels = [int(line.strip()) for line in f if line.strip()]
        
        if len(labels) != len(signal_files):
            print(f"⚠️  Warning: Number of labels ({len(labels)}) != number of files ({len(signal_files)})")
            print(f"   Using minimum length")
            min_len = min(len(labels), len(signal_files))
            signal_files = signal_files[:min_len]
            labels = labels[:min_len]
    else:
        # Try to extract label from filename
        print("No labels.txt found, trying to extract labels from filenames...")
        labels = []
        for f in signal_files:
            # Example: "label_0_sample_1.csv" -> extract 0
            # Adjust this logic based on your filename convention
            try:
                # Look for pattern: label_X or class_X
                stem = f.stem.lower()
                if 'label_' in stem:
                    label = int(stem.split('label_')[1].split('_')[0])
                elif 'class_' in stem:
                    label = int(stem.split('class_')[1].split('_')[0])
                else:
                    # Default to 0 if no pattern found
                    label = 0
                labels.append(label)
            except (ValueError, IndexError):
                print(f"⚠️  Could not extract label from: {f.name}, using 0")
                labels.append(0)
    
    # Split into train and eval
    num_train = int(len(signal_files) * train_ratio)
    
    train_files = signal_files[:num_train]
    train_labels = labels[:num_train]
    
    eval_files = signal_files[num_train:]
    eval_labels = labels[num_train:]
    
    print(f"\nData split:")
    print(f"  Training: {len(train_files)} samples")
    print(f"  Evaluation: {len(eval_files)} samples")
    
    # Write training data
    train_files_path = output_dir / "train_files.txt"
    train_labels_path = output_dir / "train_labels.txt"
    
    with open(train_files_path, 'w') as f:
        for file in train_files:
            f.write(f"{file.absolute()}\n")
    
    with open(train_labels_path, 'w') as f:
        for label in train_labels:
            f.write(f"{label}\n")
    
    print(f"\n✅ Saved training data:")
    print(f"  Files: {train_files_path}")
    print(f"  Labels: {train_labels_path}")
    
    # Write evaluation data if available
    if len(eval_files) > 0:
        eval_files_path = output_dir / "eval_files.txt"
        eval_labels_path = output_dir / "eval_labels.txt"
        
        with open(eval_files_path, 'w') as f:
            for file in eval_files:
                f.write(f"{file.absolute()}\n")
        
        with open(eval_labels_path, 'w') as f:
            for label in eval_labels:
                f.write(f"{label}\n")
        
        print(f"\n✅ Saved evaluation data:")
        print(f"  Files: {eval_files_path}")
        print(f"  Labels: {eval_labels_path}")
    
    # Print label distribution
    from collections import Counter
    train_dist = Counter(train_labels)
    print(f"\nTraining label distribution:")
    for label, count in sorted(train_dist.items()):
        print(f"  Label {label}: {count} samples ({count/len(train_labels)*100:.1f}%)")
    
    if len(eval_labels) > 0:
        eval_dist = Counter(eval_labels)
        print(f"\nEvaluation label distribution:")
        for label, count in sorted(eval_dist.items()):
            print(f"  Label {label}: {count} samples ({count/len(eval_labels)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser('Prepare LLM training data lists')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing signal files')
    parser.add_argument('--output_dir', type=str, default='./data_lists',
                        help='Output directory for generated lists')
    parser.add_argument('--file_extension', type=str, default='.csv',
                        help='File extension to look for (default: .csv)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data (default: 0.8)')
    
    args = parser.parse_args()
    
    prepare_data_lists(
        args.data_dir,
        args.output_dir,
        args.file_extension,
        args.train_ratio,
    )


if __name__ == "__main__":
    main()
