"""
Verification Evaluation for Palmprint Model

Evaluates model using proper verification metrics (EER, FAR@FRR)
on genuine/impostor pairs, which is the correct evaluation for
subject-disjoint splits.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import PalmprintDataset, get_transforms, load_subjects_from_file
from model import PalmprintVerifier, cosine_similarity


def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    """
    Extract embeddings for all samples in dataloader.
    
    Returns:
        embeddings: [N, D] tensor of embeddings
        labels: [N] tensor of class labels
        metadata: List of dicts with subject/hand info
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    all_metadata = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting embeddings'):
            if len(batch) == 3:
                images, labels, metadata = batch
            else:
                images, labels = batch
                metadata = None
            
            images = images.to(device)
            embeddings = model.get_embedding(images)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
            if metadata:
                all_metadata.extend([{k: v[i] for k, v in metadata.items()} 
                                    for i in range(len(labels))])
    
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return embeddings, labels, all_metadata


def compute_verification_pairs(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_genuine: int = 5000,
    num_impostor: int = 5000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate genuine and impostor pairs and compute similarity scores.
    
    Genuine pairs: same class (same subject+hand)
    Impostor pairs: different class
    
    Returns:
        scores: Array of similarity scores
        is_genuine: Array of binary labels (1=genuine, 0=impostor)
    """
    np.random.seed(seed)
    
    n = len(embeddings)
    unique_labels = torch.unique(labels).numpy()
    
    # Group indices by label
    label_to_indices = {}
    for idx, label in enumerate(labels.numpy()):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    # Generate genuine pairs (same class)
    genuine_scores = []
    for label in unique_labels:
        indices = label_to_indices[label]
        if len(indices) < 2:
            continue
        # Generate pairs within this class
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if len(genuine_scores) >= num_genuine:
                    break
                emb1 = embeddings[indices[i]]
                emb2 = embeddings[indices[j]]
                score = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                genuine_scores.append(score)
            if len(genuine_scores) >= num_genuine:
                break
        if len(genuine_scores) >= num_genuine:
            break
    
    # Generate impostor pairs (different class)
    impostor_scores = []
    labels_np = labels.numpy()
    attempts = 0
    max_attempts = num_impostor * 10
    
    while len(impostor_scores) < num_impostor and attempts < max_attempts:
        i, j = np.random.randint(0, n, size=2)
        if labels_np[i] != labels_np[j]:
            score = F.cosine_similarity(
                embeddings[i].unsqueeze(0), 
                embeddings[j].unsqueeze(0)
            ).item()
            impostor_scores.append(score)
        attempts += 1
    
    # Combine
    scores = np.array(genuine_scores + impostor_scores)
    is_genuine = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
    
    return scores, is_genuine


def compute_eer(scores: np.ndarray, is_genuine: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and threshold.
    
    Returns:
        eer: Equal Error Rate (0-1)
        threshold: Threshold at EER
    """
    fpr, tpr, thresholds = roc_curve(is_genuine, scores)
    fnr = 1 - tpr
    
    # Find where FPR ≈ FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    threshold = thresholds[eer_idx]
    
    return eer, threshold


def compute_far_at_frr(
    scores: np.ndarray, 
    is_genuine: np.ndarray,
    target_frr: float = 0.01
) -> float:
    """
    Compute FAR at a fixed FRR (e.g., FAR when FRR=1%).
    
    Returns:
        far: False Accept Rate at target FRR
    """
    fpr, tpr, thresholds = roc_curve(is_genuine, scores)
    fnr = 1 - tpr
    
    # Find threshold where FNR is closest to target
    idx = np.nanargmin(np.abs(fnr - target_frr))
    far = fpr[idx]
    
    return far


def evaluate_verification(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    num_pairs: int = 5000
) -> Dict[str, float]:
    """
    Full verification evaluation.
    
    Returns:
        Dictionary with EER, threshold, FAR@FRR=1%, etc.
    """
    # Extract embeddings
    embeddings, labels, _ = extract_embeddings(model, dataloader, device)
    
    # Compute pairs
    scores, is_genuine = compute_verification_pairs(
        embeddings, labels,
        num_genuine=num_pairs,
        num_impostor=num_pairs
    )
    
    # Compute metrics
    eer, threshold = compute_eer(scores, is_genuine)
    far_at_1_frr = compute_far_at_frr(scores, is_genuine, target_frr=0.01)
    far_at_01_frr = compute_far_at_frr(scores, is_genuine, target_frr=0.001)
    
    # Score statistics
    genuine_scores = scores[is_genuine == 1]
    impostor_scores = scores[is_genuine == 0]
    
    return {
        'eer': eer,
        'eer_threshold': threshold,
        'far_at_1pct_frr': far_at_1_frr,
        'far_at_01pct_frr': far_at_01_frr,
        'genuine_mean': genuine_scores.mean(),
        'genuine_std': genuine_scores.std(),
        'impostor_mean': impostor_scores.mean(),
        'impostor_std': impostor_scores.std(),
        'num_genuine_pairs': len(genuine_scores),
        'num_impostor_pairs': len(impostor_scores)
    }


def plot_score_distribution(
    scores: np.ndarray,
    is_genuine: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """Plot genuine vs impostor score distributions."""
    genuine_scores = scores[is_genuine == 1]
    impostor_scores = scores[is_genuine == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(impostor_scores, bins=50, alpha=0.7, label='Impostor', color='red', density=True)
    plt.hist(genuine_scores, bins=50, alpha=0.7, label='Genuine', color='green', density=True)
    plt.xlabel('Cosine Similarity Score')
    plt.ylabel('Density')
    plt.title('Score Distribution: Genuine vs Impostor Pairs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate palmprint verification')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default='datasets/Grayscale_128_128')
    parser.add_argument('--splits-dir', type=str, default='splits')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--num-pairs', type=int, default=5000)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--plot', action='store_true', help='Save score distribution plot')
    
    args = parser.parse_args()
    
    # Device
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
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # We need to know num_classes to create model - get from checkpoint
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
    
    # Load data
    subjects = load_subjects_from_file(
        Path(args.splits_dir) / f'{args.split}_subjects.txt'
    )
    
    dataset = PalmprintDataset(
        data_root=args.data_root,
        subjects=subjects,
        transform=get_transforms('test', image_size=128),
        hands='both',
        return_metadata=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    print(f"\nEvaluating on {args.split} split ({len(dataset)} images, {dataset.num_classes} classes)")
    
    # Evaluate
    metrics = evaluate_verification(model, dataloader, device, args.num_pairs)
    
    # Print results
    print("\n" + "=" * 50)
    print("VERIFICATION RESULTS")
    print("=" * 50)
    print(f"EER:              {metrics['eer']*100:.2f}%")
    print(f"EER Threshold:    {metrics['eer_threshold']:.4f}")
    print(f"FAR @ 1% FRR:     {metrics['far_at_1pct_frr']*100:.2f}%")
    print(f"FAR @ 0.1% FRR:   {metrics['far_at_01pct_frr']*100:.2f}%")
    print("-" * 50)
    print(f"Genuine pairs:    {metrics['num_genuine_pairs']}")
    print(f"Impostor pairs:   {metrics['num_impostor_pairs']}")
    print(f"Genuine mean:     {metrics['genuine_mean']:.4f} ± {metrics['genuine_std']:.4f}")
    print(f"Impostor mean:    {metrics['impostor_mean']:.4f} ± {metrics['impostor_std']:.4f}")
    print("=" * 50)
    
    # Plot if requested
    if args.plot:
        embeddings, labels, _ = extract_embeddings(model, dataloader, device)
        scores, is_genuine = compute_verification_pairs(
            embeddings, labels, args.num_pairs, args.num_pairs
        )
        
        checkpoint_dir = Path(args.checkpoint).parent
        plot_path = checkpoint_dir / f'score_dist_{args.split}.png'
        plot_score_distribution(scores, is_genuine, str(plot_path))


if __name__ == '__main__':
    main()
