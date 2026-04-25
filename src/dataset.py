"""
PyTorch Dataset for PolyU-IITD Palmprint Dataset

Supports both classification-based training (for embedding learning)
and verification pair generation.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Callable
import json

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms


class PalmprintDataset(Dataset):
    """
    PyTorch Dataset for palmprint images.
    
    Treats each (subject, hand) combination as a unique identity for classification.
    This doubles the number of classes but preserves hand distinction.
    
    Args:
        data_root: Path to dataset root (e.g., 'datasets/Grayscale_128_128')
        subjects: List of subject IDs to include (e.g., ['H_ID001', 'H_ID002'])
        transform: Optional torchvision transforms to apply
        hands: Which hands to include ('both', 'left', 'right')
        return_metadata: If True, also return (subject_id, hand, filename)
    """
    
    def __init__(
        self,
        data_root: str,
        subjects: List[str],
        transform: Optional[Callable] = None,
        hands: str = 'both',
        return_metadata: bool = False
    ):
        self.data_root = Path(data_root)
        self.subjects = subjects
        self.transform = transform
        self.hands = hands
        self.return_metadata = return_metadata
        
        # Build sample list and class mapping
        self.samples: List[Tuple[Path, int, str, str]] = []  # (path, class_id, subject, hand)
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        
        self._build_dataset()
    
    def _build_dataset(self) -> None:
        """Scan directories and build sample list."""
        hand_dirs = []
        if self.hands in ('both', 'left'):
            hand_dirs.append('L')
        if self.hands in ('both', 'right'):
            hand_dirs.append('R')
        
        class_idx = 0
        
        for subject in sorted(self.subjects):
            subject_dir = self.data_root / subject
            
            for hand in hand_dirs:
                hand_dir = subject_dir / hand
                if not hand_dir.exists():
                    continue
                
                # Get all images (case-insensitive)
                images = list(hand_dir.glob('*.JPG')) + list(hand_dir.glob('*.jpg'))
                images = sorted(set(images))  # Remove duplicates, sort for determinism
                
                if not images:
                    continue
                
                # Create class label for this (subject, hand) combination
                class_name = f"{subject}_{hand}"
                self.class_to_idx[class_name] = class_idx
                self.idx_to_class[class_idx] = class_name
                
                for img_path in images:
                    self.samples.append((img_path, class_idx, subject, hand))
                
                class_idx += 1
        
        self.num_classes = class_idx
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        img_path, class_id, subject, hand = self.samples[idx]
        
        # Load image
        image = Image.open(img_path)
        
        # Convert grayscale to RGB for pretrained models (if needed)
        if image.mode == 'L':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.return_metadata:
            return image, class_id, {
                'subject': subject,
                'hand': hand,
                'filename': img_path.name
            }
        
        return image, class_id
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name (subject_hand) from class ID."""
        return self.idx_to_class.get(class_id, f"unknown_{class_id}")
    
    def get_samples_by_class(self, class_id: int) -> List[Path]:
        """Get all image paths for a given class."""
        return [s[0] for s in self.samples if s[1] == class_id]


def load_subjects_from_file(filepath: str) -> List[str]:
    """Load subject list from a text file (one subject per line)."""
    with open(filepath, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]
    return subjects


def load_subjects_from_json(filepath: str, split: str) -> List[str]:
    """Load subject list from splits.json file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['splits'][split]


def get_transforms(
    split: str = 'train',
    image_size: int = 128,
    normalize: bool = True,
    strong_augment: bool = True
) -> transforms.Compose:
    """
    Get appropriate transforms for train/val/test splits.
    
    Args:
        split: One of 'train', 'val', 'test'
        image_size: Target image size (assumes square)
        normalize: Whether to apply ImageNet normalization
        strong_augment: Use stronger augmentations to prevent overfitting
    
    Returns:
        Composed transforms
    """
    # ImageNet normalization (works well for pretrained models)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if split == 'train':
        if strong_augment:
            # Stronger augmentations to prevent overfitting
            transform_list = [
                transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=5
                ),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomGrayscale(p=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # Cutout-like
            ]
        else:
            transform_list = [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ]
    else:
        # Val/Test: no augmentation
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    
    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def create_dataloaders(
    data_root: str,
    splits_dir: str,
    batch_size: int = 32,
    image_size: int = 128,
    num_workers: int = 4,
    hands: str = 'both'
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create train/val/test dataloaders from split files.
    
    Args:
        data_root: Path to dataset root
        splits_dir: Directory containing split files
        batch_size: Batch size for dataloaders
        image_size: Target image size
        num_workers: Number of dataloader workers
        hands: Which hands to include ('both', 'left', 'right')
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes)
    """
    splits_path = Path(splits_dir)
    
    # Load subject lists
    train_subjects = load_subjects_from_file(splits_path / 'train_subjects.txt')
    val_subjects = load_subjects_from_file(splits_path / 'val_subjects.txt')
    test_subjects = load_subjects_from_file(splits_path / 'test_subjects.txt')
    
    # Create datasets
    train_dataset = PalmprintDataset(
        data_root=data_root,
        subjects=train_subjects,
        transform=get_transforms('train', image_size),
        hands=hands
    )
    
    val_dataset = PalmprintDataset(
        data_root=data_root,
        subjects=val_subjects,
        transform=get_transforms('val', image_size),
        hands=hands
    )
    
    test_dataset = PalmprintDataset(
        data_root=data_root,
        subjects=test_subjects,
        transform=get_transforms('test', image_size),
        hands=hands
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes


if __name__ == '__main__':
    # Quick test
    import argparse
    
    parser = argparse.ArgumentParser(description='Test dataset loading')
    parser.add_argument('--data-root', type=str, default='datasets/Grayscale_128_128')
    parser.add_argument('--splits-dir', type=str, default='splits')
    args = parser.parse_args()
    
    print("Testing dataset loading...")
    
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        data_root=args.data_root,
        splits_dir=args.splits_dir,
        batch_size=32,
        num_workers=0  # Use 0 for debugging
    )
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label range: [{labels.min()}, {labels.max()}]")
