"""
Baseline Floor: Pretrained-only ResNet18 (no ArcFace fine-tuning).

Produces the quantitative floor that Checkpoint 2 results compare against.
Uses ImageNet-pretrained ResNet18 features, L2-normalized, scored by cosine
similarity on the identical test pair set as evaluate.py.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from dataset import PalmprintDataset, get_transforms, load_subjects_from_file
from evaluate import evaluate_verification


class PretrainedResNet18Embedder(nn.Module):
    """ImageNet ResNet18 with FC stripped; emits L2-normalized 512-D features."""

    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return F.normalize(feats, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_embedding(x)


def main():
    device = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f"Using device: {device}")

    model = PretrainedResNet18Embedder().to(device).eval()

    subjects = load_subjects_from_file('splits/test_subjects.txt')
    dataset = PalmprintDataset(
        data_root='datasets/Grayscale_128_128',
        subjects=subjects,
        transform=get_transforms('test', image_size=128),
        hands='both',
        return_metadata=False,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"Evaluating {len(dataset)} images / {dataset.num_classes} classes")
    metrics = evaluate_verification(model, loader, device, num_pairs=5000)

    out_dir = Path('checkpoints/baseline_floor')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'metrics.json').write_text(json.dumps(
        {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()},
        indent=2,
    ))

    print("\n" + "=" * 50)
    print("BASELINE FLOOR (Pretrained-only ResNet18)")
    print("=" * 50)
    print(f"EER:              {metrics['eer']*100:.2f}%")
    print(f"FAR @ 1% FRR:     {metrics['far_at_1pct_frr']*100:.2f}%")
    print(f"FAR @ 0.1% FRR:   {metrics['far_at_01pct_frr']*100:.2f}%")
    print(f"Genuine mean:     {metrics['genuine_mean']:.4f} ± {metrics['genuine_std']:.4f}")
    print(f"Impostor mean:    {metrics['impostor_mean']:.4f} ± {metrics['impostor_std']:.4f}")
    print(f"\nSaved: {out_dir / 'metrics.json'}")


if __name__ == '__main__':
    main()
