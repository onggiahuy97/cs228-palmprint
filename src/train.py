"""
Training Script for Palmprint Verification

Trains a palmprint embedding model using ArcFace loss.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_device
from dataset import create_dataloaders, PalmprintDataset, get_transforms, load_subjects_from_file
from model import PalmprintVerifier
from evaluate import evaluate_verification


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_scheduler(optimizer, config: Config, steps_per_epoch: int):
    """Create learning rate scheduler."""
    tc = config.training
    
    if tc.scheduler == 'none':
        return None
    
    # Warmup scheduler
    def warmup_lambda(epoch):
        if epoch < tc.warmup_epochs:
            return (epoch + 1) / tc.warmup_epochs
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, warmup_lambda)
    
    if tc.scheduler == 'cosine':
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=tc.epochs - tc.warmup_epochs,
            eta_min=tc.min_lr
        )
    elif tc.scheduler == 'step':
        main_scheduler = StepLR(
            optimizer,
            step_size=tc.step_size,
            gamma=tc.gamma
        )
    else:
        return warmup_scheduler
    
    return warmup_scheduler, main_scheduler


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    device: str,
    epoch: int,
    label_smoothing: float = 0.1
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(images, labels)
        
        # Loss with label smoothing for regularization
        loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return {
        'loss': total_loss / total,
        'accuracy': correct / total
    }


@torch.no_grad()
def validate_verification(
    model: nn.Module,
    val_loader,
    device: str,
    num_pairs: int = 2000
) -> dict:
    """
    Validate using proper verification metrics (EER).
    
    For subject-disjoint splits, classification accuracy is meaningless
    because val classes don't exist in training. Instead, we evaluate
    the quality of embeddings using genuine/impostor pair similarity.
    """
    metrics = evaluate_verification(model, val_loader, device, num_pairs)
    return {
        'eer': metrics['eer'],
        'eer_threshold': metrics['eer_threshold'],
        'genuine_mean': metrics['genuine_mean'],
        'impostor_mean': metrics['impostor_mean']
    }


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: dict,
    path: str
) -> None:
    """Save training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, path)


def train(config: Config) -> None:
    """Main training function."""
    tc = config.training
    dc = config.data
    mc = config.model
    
    # Setup
    set_seed(tc.seed)
    device = get_device(tc.device)
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        data_root=dc.data_root,
        splits_dir=dc.splits_dir,
        batch_size=tc.batch_size,
        image_size=dc.image_size,
        num_workers=dc.num_workers,
        hands=dc.hands
    )
    
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = PalmprintVerifier(
        num_classes=num_classes,
        embedding_dim=mc.embedding_dim,
        backbone=mc.backbone,
        pretrained=mc.pretrained,
        scale=mc.arcface_scale,
        margin=mc.arcface_margin
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=tc.learning_rate,
        weight_decay=tc.weight_decay
    )
    
    # Scheduler
    schedulers = get_scheduler(optimizer, config, len(train_loader))
    
    # Checkpoint directory
    checkpoint_dir = Path(tc.checkpoint_dir) / config.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save(checkpoint_dir / 'config.json')
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    print("Note: Val EER (Equal Error Rate) is the key metric - lower is better!")
    print("=" * 60)
    
    best_val_eer = 1.0  # Lower is better for EER
    patience_counter = 0
    
    for epoch in range(1, tc.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate using verification metrics (EER)
        val_metrics = validate_verification(model, val_loader, device, num_pairs=2000)
        
        # Update scheduler
        if schedulers is not None:
            if isinstance(schedulers, tuple):
                warmup_sched, main_sched = schedulers
                if epoch <= tc.warmup_epochs:
                    warmup_sched.step()
                else:
                    main_sched.step()
            else:
                schedulers.step()
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Print metrics
        print(f"Epoch {epoch:3d}/{tc.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']*100:.2f}% | "
              f"Val EER: {val_metrics['eer']*100:.2f}% | "
              f"Gen/Imp: {val_metrics['genuine_mean']:.3f}/{val_metrics['impostor_mean']:.3f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        # Check for improvement (lower EER is better)
        if val_metrics['eer'] < best_val_eer - tc.min_delta:
            best_val_eer = val_metrics['eer']
            patience_counter = 0
            
            # Save best model
            save_checkpoint(
                model, optimizer, epoch,
                {'train': train_metrics, 'val': val_metrics},
                checkpoint_dir / 'best_model.pt'
            )
            print(f"  → New best model saved! (Val EER: {best_val_eer*100:.2f}%)")
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if epoch % tc.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch,
                {'train': train_metrics, 'val': val_metrics},
                checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            )
        
        # Early stopping
        if patience_counter >= tc.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Save final model
    save_checkpoint(
        model, optimizer, epoch,
        {'train': train_metrics, 'val': val_metrics},
        checkpoint_dir / 'final_model.pt'
    )
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best validation EER: {best_val_eer*100:.2f}%")
    print(f"Checkpoints saved to: {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train palmprint verification model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON file')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Override data root path')
    parser.add_argument('--splits-dir', type=str, default=None,
                        help='Override splits directory')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Experiment name for checkpoints')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/mps/cpu)')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    
    # Apply overrides
    if args.data_root:
        config.data.data_root = args.data_root
    if args.splits_dir:
        config.data.splits_dir = args.splits_dir
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.experiment:
        config.experiment_name = args.experiment
    if args.device:
        config.training.device = args.device
    
    # Add timestamp to experiment name if default
    if config.experiment_name == 'baseline':
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config.experiment_name = f'baseline_{timestamp}'
    
    train(config)


if __name__ == '__main__':
    main()
