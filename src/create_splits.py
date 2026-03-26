"""
Split Generator for PolyU-IITD Palmprint Dataset

Generates subject-disjoint train/val/test splits (70/10/20) ensuring
no identity leakage across splits.
"""

import os
import random
import json
from pathlib import Path
from typing import List, Tuple, Dict
import argparse


def get_valid_subjects(data_root: str) -> List[str]:
    """
    Scan dataset directory and return list of valid subject IDs.
    A subject is valid if their folder exists and contains at least one image.
    
    Args:
        data_root: Path to dataset root (e.g., datasets/Grayscale_128_128)
    
    Returns:
        Sorted list of valid subject IDs (e.g., ['H_ID001', 'H_ID002', ...])
    """
    data_path = Path(data_root)
    valid_subjects = []
    
    for subject_dir in sorted(data_path.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith('H_ID'):
            continue
        
        # Check if subject has any images (in L or R subfolders)
        has_images = False
        for hand in ['L', 'R']:
            hand_dir = subject_dir / hand
            if hand_dir.exists():
                # Case-insensitive check for .jpg files
                images = list(hand_dir.glob('*.JPG')) + list(hand_dir.glob('*.jpg'))
                if images:
                    has_images = True
                    break
        
        if has_images:
            valid_subjects.append(subject_dir.name)
    
    return valid_subjects


def create_splits(
    subjects: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Create subject-disjoint train/val/test splits.
    
    Args:
        subjects: List of subject IDs
        train_ratio: Fraction for training (default 0.7)
        val_ratio: Fraction for validation (default 0.1)
        test_ratio: Fraction for testing (default 0.2)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing subject lists
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Shuffle subjects deterministically
    subjects = subjects.copy()
    random.seed(seed)
    random.shuffle(subjects)
    
    n = len(subjects)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        'train': sorted(subjects[:n_train]),
        'val': sorted(subjects[n_train:n_train + n_val]),
        'test': sorted(subjects[n_train + n_val:])
    }
    
    return splits


def get_subject_stats(data_root: str, subjects: List[str]) -> Dict[str, int]:
    """
    Count total images for a list of subjects.
    
    Args:
        data_root: Path to dataset root
        subjects: List of subject IDs
    
    Returns:
        Dictionary with image counts per hand and total
    """
    data_path = Path(data_root)
    stats = {'left': 0, 'right': 0, 'total': 0}
    
    for subject in subjects:
        subject_dir = data_path / subject
        for hand, key in [('L', 'left'), ('R', 'right')]:
            hand_dir = subject_dir / hand
            if hand_dir.exists():
                images = list(hand_dir.glob('*.JPG')) + list(hand_dir.glob('*.jpg'))
                # Remove duplicates (in case both patterns match same file)
                images = list(set(images))
                stats[key] += len(images)
    
    stats['total'] = stats['left'] + stats['right']
    return stats


def save_splits(
    splits: Dict[str, List[str]],
    output_dir: str,
    data_root: str
) -> None:
    """
    Save splits to files and print statistics.
    
    Args:
        splits: Dictionary with train/val/test subject lists
        output_dir: Directory to save split files
        data_root: Path to dataset root (for stats)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save individual text files (one subject per line)
    for split_name, subjects in splits.items():
        txt_file = output_path / f'{split_name}_subjects.txt'
        with open(txt_file, 'w') as f:
            f.write('\n'.join(subjects))
        print(f"Saved {txt_file}")
    
    # Save combined JSON with metadata
    metadata = {
        'splits': splits,
        'stats': {},
        'config': {
            'train_ratio': 0.7,
            'val_ratio': 0.1,
            'test_ratio': 0.2,
            'seed': 42
        }
    }
    
    # Calculate stats for each split
    print("\n" + "=" * 60)
    print("Split Statistics")
    print("=" * 60)
    
    total_subjects = 0
    total_images = 0
    
    for split_name, subjects in splits.items():
        stats = get_subject_stats(data_root, subjects)
        metadata['stats'][split_name] = {
            'num_subjects': len(subjects),
            'num_images': stats['total'],
            'left_images': stats['left'],
            'right_images': stats['right']
        }
        
        print(f"\n{split_name.upper():}")
        print(f"  Subjects: {len(subjects)}")
        print(f"  Images:   {stats['total']} (L: {stats['left']}, R: {stats['right']})")
        
        total_subjects += len(subjects)
        total_images += stats['total']
    
    print("\n" + "-" * 60)
    print(f"TOTAL: {total_subjects} subjects, {total_images} images")
    print("=" * 60)
    
    # Save JSON
    json_file = output_path / 'splits.json'
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate subject-disjoint splits for palmprint dataset'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='datasets/Grayscale_128_128',
        help='Path to dataset root directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='splits',
        help='Directory to save split files'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Fraction of subjects for training'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Fraction of subjects for validation'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.2,
        help='Fraction of subjects for testing'
    )
    
    args = parser.parse_args()
    
    print(f"Scanning dataset at: {args.data_root}")
    subjects = get_valid_subjects(args.data_root)
    print(f"Found {len(subjects)} valid subjects")
    
    if len(subjects) == 0:
        print("ERROR: No valid subjects found!")
        return
    
    splits = create_splits(
        subjects,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    save_splits(splits, args.output_dir, args.data_root)


if __name__ == '__main__':
    main()
