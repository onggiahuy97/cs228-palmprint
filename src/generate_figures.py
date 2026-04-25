"""
Generate all figures for the final report.

Usage:
    python src/generate_figures.py --figures all
    python src/generate_figures.py --figures training_curves roc_curve
"""

import sys
import os
import gc
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from PIL import Image

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from model import PalmprintVerifier
from dataset import PalmprintDataset, get_transforms, load_subjects_from_file
from evaluate import extract_embeddings, compute_verification_pairs, compute_eer
from corruptions import get_corruption, get_all_corruptions


PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints' / 'baseline_v2'
FIGURES_DIR = PROJECT_ROOT / 'figures'
DATA_ROOT = PROJECT_ROOT / 'datasets' / 'Grayscale_128_128'
SPLITS_DIR = PROJECT_ROOT / 'splits'


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_training_history():
    """Extract training metrics from checkpoint files."""
    history = []

    checkpoint_files = sorted(CHECKPOINT_DIR.glob('checkpoint_epoch_*.pt'),
                              key=lambda p: int(p.stem.split('_')[-1]))

    for ckpt_path in ['best_model.pt', 'final_model.pt']:
        full = CHECKPOINT_DIR / ckpt_path
        if full.exists() and full not in checkpoint_files:
            checkpoint_files.append(full)

    seen_epochs = set()
    for ckpt_path in checkpoint_files:
        print(f'  Loading {ckpt_path.name}...')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        epoch = ckpt['epoch']
        if epoch in seen_epochs:
            del ckpt
            gc.collect()
            continue
        seen_epochs.add(epoch)
        m = ckpt['metrics']
        history.append({
            'epoch': epoch,
            'train_loss': m['train']['loss'],
            'train_acc': m['train']['accuracy'],
            'val_eer': m['val']['eer'],
        })
        del ckpt
        gc.collect()

    history.sort(key=lambda x: x['epoch'])
    return history


def load_model_and_test_data(device):
    """Load best model and test dataloader."""
    ckpt = torch.load(CHECKPOINT_DIR / 'best_model.pt', map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']
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
    del ckpt
    gc.collect()

    subjects = load_subjects_from_file(SPLITS_DIR / 'test_subjects.txt')
    dataset = PalmprintDataset(
        data_root=str(DATA_ROOT),
        subjects=subjects,
        transform=get_transforms('test', image_size=128),
        hands='both',
        return_metadata=False
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    return model, dataloader


def compute_scores(model, dataloader, device):
    """Extract embeddings and compute verification scores."""
    embeddings, labels, _ = extract_embeddings(model, dataloader, device)
    scores, is_genuine = compute_verification_pairs(
        embeddings, labels, num_genuine=5000, num_impostor=5000
    )
    return embeddings, labels, scores, is_genuine


# =====================================================================
# Figure generators
# =====================================================================

def generate_training_curves():
    """Figure 1: Training convergence (3-panel)."""
    print('Generating training_curves.png...')
    history = load_training_history()

    epochs = [h['epoch'] for h in history]
    losses = [h['train_loss'] for h in history]
    accs = [h['train_acc'] * 100 for h in history]
    eers = [h['val_eer'] * 100 for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Loss
    axes[0].plot(epochs, losses, 'o-', color='#2196F3', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('(a) Training Loss', fontsize=13)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Accuracy
    axes[1].plot(epochs, accs, 's-', color='#4CAF50', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Training Accuracy (%)', fontsize=12)
    axes[1].set_title('(b) Training Accuracy', fontsize=13)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Val EER
    axes[2].plot(epochs, eers, 'D-', color='#F44336', linewidth=2, markersize=6)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Validation EER (%)', fontsize=12)
    axes[2].set_title('(c) Validation EER', fontsize=13)
    axes[2].grid(True, alpha=0.3)

    # Annotate best EER
    best_idx = np.argmin(eers)
    axes[2].annotate(
        f'Best: {eers[best_idx]:.2f}%\n(epoch {epochs[best_idx]})',
        xy=(epochs[best_idx], eers[best_idx]),
        xytext=(epochs[best_idx] + 3, eers[best_idx] + 0.5),
        arrowprops=dict(arrowstyle='->', color='black'),
        fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved figures/training_curves.png')


def generate_roc_curve(scores=None, is_genuine=None, model=None, dataloader=None, device=None):
    """Figure 2: ROC curve."""
    print('Generating roc_curve.png...')

    if scores is None:
        _, _, scores, is_genuine = compute_scores(model, dataloader, device)

    fpr, tpr, _ = roc_curve(is_genuine, scores)
    roc_auc = auc(fpr, tpr)

    # Compute EER point
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer_fpr, eer_tpr = fpr[eer_idx], tpr[eer_idx]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color='#2196F3', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.plot(eer_fpr, eer_tpr, 'ro', markersize=10, label=f'EER = {(1-eer_tpr)*100:.2f}%')

    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curve — Palmprint Verification', fontsize=14)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved figures/roc_curve.png')


def generate_det_curve(scores=None, is_genuine=None, model=None, dataloader=None, device=None):
    """Figure 3: DET curve (log-log)."""
    print('Generating det_curve.png...')

    if scores is None:
        _, _, scores, is_genuine = compute_scores(model, dataloader, device)

    fpr, tpr, _ = roc_curve(is_genuine, scores)
    fnr = 1 - tpr

    # Remove zeros for log scale
    mask = (fpr > 0) & (fnr > 0)
    fpr_nz, fnr_nz = fpr[mask], fnr[mask]

    # EER point
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer_val = (fpr[eer_idx] + fnr[eer_idx]) / 2

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.loglog(fpr_nz * 100, fnr_nz * 100, color='#F44336', linewidth=2, label='DET Curve')
    ax.loglog([0.01, 100], [0.01, 100], 'k--', linewidth=1, alpha=0.5, label='EER line (FPR=FNR)')
    ax.plot(eer_val * 100, eer_val * 100, 'bo', markersize=10,
            label=f'EER = {eer_val*100:.2f}%')

    ax.set_xlabel('False Positive Rate (%)', fontsize=13)
    ax.set_ylabel('False Negative Rate (%)', fontsize=13)
    ax.set_title('DET Curve — Palmprint Verification', fontsize=14)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([0.1, 100])
    ax.set_ylim([0.1, 100])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'det_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved figures/det_curve.png')


def generate_tsne(embeddings=None, labels=None, model=None, dataloader=None, device=None):
    """Figure 4: t-SNE embedding visualization."""
    print('Generating tsne_embeddings.png...')

    if embeddings is None:
        embeddings, labels, _, _ = compute_scores(model, dataloader, device)

    # Subsample 20 random classes for clarity
    np.random.seed(42)
    unique_labels = torch.unique(labels).numpy()
    selected = np.random.choice(unique_labels, size=min(20, len(unique_labels)), replace=False)
    mask = np.isin(labels.numpy(), selected)

    sub_emb = embeddings[mask].numpy()
    sub_lab = labels[mask].numpy()

    print(f'  Running t-SNE on {len(sub_emb)} samples from {len(selected)} classes...')
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(sub_emb)

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(len(selected))

    label_to_idx = {l: i for i, l in enumerate(selected)}
    colors = [label_to_idx[l] for l in sub_lab]

    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, cmap=cmap,
                         s=25, alpha=0.7, edgecolors='none')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=13)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=13)
    ax.set_title('t-SNE Embedding Visualization (20 Identities)', fontsize=14)
    ax.grid(True, alpha=0.2)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label('Identity Class', fontsize=11)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'tsne_embeddings.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved figures/tsne_embeddings.png')


def generate_corruption_samples():
    """Figure 5: Corruption sample grid."""
    print('Generating corruption_samples.png...')

    # Find a sample image
    sample_path = list(DATA_ROOT.glob('H_ID001/L/*.JPG'))
    if not sample_path:
        sample_path = list(DATA_ROOT.glob('*/L/*.JPG'))
    img = Image.open(sample_path[0]).convert('RGB')

    corruptions = get_all_corruptions()
    np.random.seed(42)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    # Original
    axes[0].imshow(img)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 9 corruptions at severity 3
    for i, name in enumerate(corruptions):
        corruption_fn = get_corruption(name, severity=3)
        corrupted = corruption_fn(img.copy())
        axes[i + 1].imshow(corrupted)
        axes[i + 1].set_title(name.replace('_', ' ').title(), fontsize=11)
        axes[i + 1].axis('off')

    plt.suptitle('Corruption Types at Severity 3', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'corruption_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved figures/corruption_samples.png')


# =====================================================================
# Main
# =====================================================================

FIGURE_GENERATORS = {
    'training_curves': generate_training_curves,
    'corruption_samples': generate_corruption_samples,
    # These three share embeddings/scores, handled specially
    'roc_curve': generate_roc_curve,
    'det_curve': generate_det_curve,
    'tsne_embeddings': generate_tsne,
}

EMBEDDING_FIGURES = {'roc_curve', 'det_curve', 'tsne_embeddings'}


def main():
    parser = argparse.ArgumentParser(description='Generate figures for final report')
    parser.add_argument('--figures', nargs='+', default=['all'],
                        choices=['all'] + list(FIGURE_GENERATORS.keys()),
                        help='Which figures to generate')
    args = parser.parse_args()

    requested = set(args.figures)
    if 'all' in requested:
        requested = set(FIGURE_GENERATORS.keys())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Standalone figures (no model needed)
    standalone = requested - EMBEDDING_FIGURES
    for name in standalone:
        FIGURE_GENERATORS[name]()

    # Embedding-based figures (share computation)
    embedding_figs = requested & EMBEDDING_FIGURES
    if embedding_figs:
        device = get_device()
        print(f'\nUsing device: {device}')
        print('Loading model and computing embeddings (shared across ROC/DET/t-SNE)...')
        model, dataloader = load_model_and_test_data(device)
        embeddings, labels, scores, is_genuine = compute_scores(model, dataloader, device)

        # Free model from memory
        del model
        gc.collect()

        if 'roc_curve' in embedding_figs:
            generate_roc_curve(scores=scores, is_genuine=is_genuine)
        if 'det_curve' in embedding_figs:
            generate_det_curve(scores=scores, is_genuine=is_genuine)
        if 'tsne_embeddings' in embedding_figs:
            generate_tsne(embeddings=embeddings, labels=labels)

    print(f'\nDone! Generated {len(requested)} figures in {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
