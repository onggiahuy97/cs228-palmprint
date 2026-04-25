"""
Robustness Evaluation Suite for Palmprint Verification

Systematically evaluates how verification performance degrades under
various real-world capture variations (corruptions).

Outputs:
- EER vs severity curves for each corruption
- Summary table of clean vs corrupted performance
- Identification of most damaging corruptions
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset import PalmprintDataset, load_subjects_from_file
from model import PalmprintVerifier
from evaluate import extract_embeddings, compute_verification_pairs, compute_eer, compute_far_at_frr
from corruptions import (
    get_corruption, get_all_corruptions, CorruptedTransform,
    CORRUPTION_DESCRIPTIONS, SEVERITY_DESCRIPTIONS
)


@dataclass
class RobustnessResult:
    """Results for a single corruption/severity combination."""
    corruption: str
    severity: int
    eer: float
    eer_threshold: float
    far_at_1pct_frr: float
    genuine_mean: float
    genuine_std: float
    impostor_mean: float
    impostor_std: float
    num_pairs: int


class CorruptedPalmprintDataset(Dataset):
    """
    Dataset wrapper that applies corruptions to images.
    """
    
    def __init__(
        self,
        base_dataset: PalmprintDataset,
        corruption_name: str,
        severity: int,
        image_size: int = 128
    ):
        self.base_dataset = base_dataset
        self.corruption_name = corruption_name
        self.severity = severity
        self.corruption_fn = get_corruption(corruption_name, severity)
        
        # Standard normalization (must match training)
        self.post_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int):
        # Get original sample (without transforms)
        img_path, class_id, subject, hand = self.base_dataset.samples[idx]
        
        # Load image
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply corruption
        corrupted = self.corruption_fn(image)
        
        # Apply standard transforms
        tensor = self.post_transform(corrupted)
        
        return tensor, class_id


def evaluate_corruption(
    model: torch.nn.Module,
    base_dataset: PalmprintDataset,
    corruption_name: str,
    severity: int,
    device: str,
    batch_size: int = 32,
    num_pairs: int = 3000,
    num_workers: int = 0  # Must be 0 - corruption functions aren't picklable
) -> RobustnessResult:
    """
    Evaluate model on a specific corruption at a specific severity.
    """
    # Create corrupted dataset
    corrupted_dataset = CorruptedPalmprintDataset(
        base_dataset, corruption_name, severity
    )
    
    dataloader = DataLoader(
        corrupted_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Must be 0 - corruption lambdas aren't picklable
    )
    
    # Extract embeddings
    embeddings, labels, _ = extract_embeddings(model, dataloader, device)
    
    # Compute verification pairs
    scores, is_genuine = compute_verification_pairs(
        embeddings, labels,
        num_genuine=num_pairs,
        num_impostor=num_pairs
    )
    
    # Compute metrics
    eer, threshold = compute_eer(scores, is_genuine)
    far_at_1 = compute_far_at_frr(scores, is_genuine, target_frr=0.01)
    
    genuine_scores = scores[is_genuine == 1]
    impostor_scores = scores[is_genuine == 0]
    
    return RobustnessResult(
        corruption=corruption_name,
        severity=severity,
        eer=eer,
        eer_threshold=threshold,
        far_at_1pct_frr=far_at_1,
        genuine_mean=genuine_scores.mean(),
        genuine_std=genuine_scores.std(),
        impostor_mean=impostor_scores.mean(),
        impostor_std=impostor_scores.std(),
        num_pairs=len(genuine_scores) + len(impostor_scores)
    )


def run_robustness_benchmark(
    model: torch.nn.Module,
    base_dataset: PalmprintDataset,
    device: str,
    corruptions: Optional[List[str]] = None,
    severities: Optional[List[int]] = None,
    batch_size: int = 32,
    num_pairs: int = 3000,
    num_workers: int = 4
) -> List[RobustnessResult]:
    """
    Run full robustness benchmark across all corruptions and severities.
    """
    if corruptions is None:
        corruptions = get_all_corruptions()
    if severities is None:
        severities = [1, 2, 3, 4, 5]
    
    results = []
    
    # First evaluate clean performance
    print("\nEvaluating clean (no corruption)...")
    clean_loader = DataLoader(
        base_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    embeddings, labels, _ = extract_embeddings(model, clean_loader, device)
    scores, is_genuine = compute_verification_pairs(embeddings, labels, num_pairs, num_pairs)
    eer, threshold = compute_eer(scores, is_genuine)
    far_at_1 = compute_far_at_frr(scores, is_genuine, target_frr=0.01)
    genuine_scores = scores[is_genuine == 1]
    impostor_scores = scores[is_genuine == 0]
    
    clean_result = RobustnessResult(
        corruption='clean',
        severity=0,
        eer=eer,
        eer_threshold=threshold,
        far_at_1pct_frr=far_at_1,
        genuine_mean=genuine_scores.mean(),
        genuine_std=genuine_scores.std(),
        impostor_mean=impostor_scores.mean(),
        impostor_std=impostor_scores.std(),
        num_pairs=len(genuine_scores) + len(impostor_scores)
    )
    results.append(clean_result)
    print(f"  Clean EER: {eer*100:.2f}%")
    
    # Evaluate each corruption
    total_evals = len(corruptions) * len(severities)
    pbar = tqdm(total=total_evals, desc="Robustness evaluation")
    
    for corruption_name in corruptions:
        for severity in severities:
            result = evaluate_corruption(
                model, base_dataset, corruption_name, severity,
                device, batch_size, num_pairs, num_workers
            )
            results.append(result)
            
            pbar.set_postfix({
                'corruption': corruption_name,
                'severity': severity,
                'EER': f'{result.eer*100:.2f}%'
            })
            pbar.update(1)
    
    pbar.close()
    return results


def results_to_dataframe(results: List[RobustnessResult]) -> pd.DataFrame:
    """Convert results list to pandas DataFrame."""
    return pd.DataFrame([asdict(r) for r in results])


def create_eer_heatmap(df: pd.DataFrame, save_path: str) -> None:
    """Create heatmap of EER values across corruptions and severities."""
    # Filter out clean
    df_corrupt = df[df['corruption'] != 'clean'].copy()
    
    # Pivot for heatmap
    pivot = df_corrupt.pivot(index='corruption', columns='severity', values='eer')
    pivot = pivot * 100  # Convert to percentage
    
    # Get clean EER for reference
    clean_eer = df[df['corruption'] == 'clean']['eer'].values[0] * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',  # Red = bad (high EER), Green = good (low EER)
        cbar_kws={'label': 'EER (%)'},
        vmin=0,
        vmax=max(50, pivot.max().max())
    )
    plt.title(f'Robustness Benchmark: EER (%) by Corruption & Severity\n(Clean baseline: {clean_eer:.2f}%)')
    plt.xlabel('Severity Level')
    plt.ylabel('Corruption Type')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {save_path}")


def create_eer_curves(df: pd.DataFrame, save_path: str) -> None:
    """Create line plots of EER vs severity for each corruption."""
    df_corrupt = df[df['corruption'] != 'clean'].copy()
    clean_eer = df[df['corruption'] == 'clean']['eer'].values[0] * 100
    
    plt.figure(figsize=(12, 6))
    
    corruptions = df_corrupt['corruption'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(corruptions)))
    
    for corruption, color in zip(corruptions, colors):
        subset = df_corrupt[df_corrupt['corruption'] == corruption]
        plt.plot(
            subset['severity'],
            subset['eer'] * 100,
            marker='o',
            label=corruption,
            color=color,
            linewidth=2,
            markersize=6
        )
    
    # Add clean baseline
    plt.axhline(y=clean_eer, color='black', linestyle='--', linewidth=2, label='Clean baseline')
    
    plt.xlabel('Severity Level')
    plt.ylabel('EER (%)')
    plt.title('Verification Performance Degradation Under Corruptions')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved EER curves to {save_path}")


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary table showing impact of each corruption."""
    clean_eer = df[df['corruption'] == 'clean']['eer'].values[0]
    
    summary_rows = []
    for corruption in df['corruption'].unique():
        if corruption == 'clean':
            continue
        
        subset = df[df['corruption'] == corruption]
        max_eer = subset['eer'].max()
        mean_eer = subset['eer'].mean()
        worst_severity = subset.loc[subset['eer'].idxmax(), 'severity']
        
        # Relative degradation from clean
        relative_deg = (max_eer - clean_eer) / clean_eer * 100
        
        summary_rows.append({
            'Corruption': corruption,
            'Description': CORRUPTION_DESCRIPTIONS.get(corruption, ''),
            'Clean EER (%)': clean_eer * 100,
            'Mean EER (%)': mean_eer * 100,
            'Worst EER (%)': max_eer * 100,
            'Worst Severity': int(worst_severity),
            'Rel. Degradation (%)': relative_deg
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('Worst EER (%)', ascending=False)
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Run robustness evaluation benchmark')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default='datasets/Grayscale_128_128')
    parser.add_argument('--splits-dir', type=str, default='splits')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: checkpoint dir)')
    parser.add_argument('--corruptions', type=str, nargs='+', default=None,
                        help='Specific corruptions to evaluate (default: all)')
    parser.add_argument('--severities', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help='Severity levels to evaluate')
    parser.add_argument('--num-pairs', type=int, default=3000,
                        help='Number of genuine/impostor pairs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Output directory
    if args.output_dir is None:
        args.output_dir = Path(args.checkpoint).parent / 'robustness'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    state_dict = checkpoint['model_state_dict']
    num_classes = state_dict['arcface.weight'].shape[0]
    embedding_dim = state_dict['arcface.weight'].shape[1]
    
    model = PalmprintVerifier(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        backbone='resnet18',
        pretrained=False
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Load test data (without transforms - corruptions applied separately)
    print(f"\nLoading {args.split} data...")
    subjects = load_subjects_from_file(
        Path(args.splits_dir) / f'{args.split}_subjects.txt'
    )
    
    # Create base dataset without transforms
    base_dataset = PalmprintDataset(
        data_root=args.data_root,
        subjects=subjects,
        transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        hands='both',
        return_metadata=False
    )
    
    print(f"Dataset: {len(base_dataset)} images, {base_dataset.num_classes} classes")
    
    # List available corruptions
    available = get_all_corruptions()
    if args.corruptions:
        corruptions = [c for c in args.corruptions if c in available]
        if len(corruptions) != len(args.corruptions):
            print(f"Warning: Some corruptions not found. Using: {corruptions}")
    else:
        corruptions = available
    
    print(f"\nEvaluating corruptions: {corruptions}")
    print(f"Severities: {args.severities}")
    
    # Run benchmark
    print("\n" + "=" * 60)
    print("ROBUSTNESS BENCHMARK")
    print("=" * 60)
    
    results = run_robustness_benchmark(
        model=model,
        base_dataset=base_dataset,
        device=device,
        corruptions=corruptions,
        severities=args.severities,
        batch_size=args.batch_size,
        num_pairs=args.num_pairs,
        num_workers=args.num_workers
    )
    
    # Convert to DataFrame
    df = results_to_dataframe(results)
    
    # Save raw results
    csv_path = output_dir / 'robustness_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw results to {csv_path}")
    
    json_path = output_dir / 'robustness_results.json'
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Saved JSON results to {json_path}")
    
    # Create visualizations
    create_eer_heatmap(df, output_dir / 'eer_heatmap.png')
    create_eer_curves(df, output_dir / 'eer_curves.png')
    
    # Create and display summary
    summary_df = create_summary_table(df)
    summary_path = output_dir / 'robustness_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY (sorted by worst degradation)")
    print("=" * 60)
    
    clean_eer = df[df['corruption'] == 'clean']['eer'].values[0]
    print(f"\nClean Baseline EER: {clean_eer*100:.2f}%")
    print("\nCorruption Rankings:")
    print("-" * 60)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Corruption']:15s} | "
              f"Worst: {row['Worst EER (%)']:6.2f}% (sev {row['Worst Severity']}) | "
              f"Deg: +{row['Rel. Degradation (%)']:.1f}%")
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
