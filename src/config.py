"""
Configuration for Palmprint Verification Training
"""

from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path


@dataclass
class DataConfig:
    """Data-related configuration."""
    data_root: str = "datasets/Grayscale_128_128"
    splits_dir: str = "splits"
    image_size: int = 128
    hands: str = "both"  # 'both', 'left', 'right'
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    backbone: str = "resnet18"  # 'resnet18', 'resnet34', 'mobilenetv3'
    embedding_dim: int = 256
    pretrained: bool = True
    dropout: float = 0.5  # Increased dropout for regularization
    
    # ArcFace parameters - reduced margin to prevent overfitting
    arcface_scale: float = 30.0
    arcface_margin: float = 0.3  # Reduced from 0.5 for better generalization


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 3e-4  # Reduced from 1e-3
    weight_decay: float = 1e-3  # Increased from 1e-4 for more regularization
    
    # Label smoothing
    label_smoothing: float = 0.1
    
    # Learning rate scheduler
    scheduler: str = "cosine"  # 'cosine', 'step', 'none'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Step scheduler params (if scheduler='step')
    step_size: int = 15
    gamma: float = 0.1
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5  # Save checkpoint every N epochs
    
    # Device
    device: str = "auto"  # 'auto', 'cuda', 'mps', 'cpu'
    
    # Reproducibility
    seed: int = 42


@dataclass
class Config:
    """Complete configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment tracking
    experiment_name: str = "baseline"
    
    def save(self, path: str) -> None:
        """Save config to JSON file."""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'experiment_name': self.experiment_name
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        config.data = DataConfig(**config_dict.get('data', {}))
        config.model = ModelConfig(**config_dict.get('model', {}))
        config.training = TrainingConfig(**config_dict.get('training', {}))
        config.experiment_name = config_dict.get('experiment_name', 'baseline')
        return config


def get_device(preference: str = 'auto') -> str:
    """
    Get the best available device.
    
    Args:
        preference: 'auto', 'cuda', 'mps', or 'cpu'
    
    Returns:
        Device string for PyTorch
    """
    import torch
    
    if preference == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    return preference


# Default configuration for quick experiments
DEFAULT_CONFIG = Config()


if __name__ == '__main__':
    # Print default config
    config = Config()
    print("Default Configuration")
    print("=" * 50)
    print(f"\nData Config:")
    for k, v in config.data.__dict__.items():
        print(f"  {k}: {v}")
    
    print(f"\nModel Config:")
    for k, v in config.model.__dict__.items():
        print(f"  {k}: {v}")
    
    print(f"\nTraining Config:")
    for k, v in config.training.__dict__.items():
        print(f"  {k}: {v}")
    
    # Test save/load
    config.save('configs/test_config.json')
    loaded = Config.load('configs/test_config.json')
    print("\n\nConfig save/load test passed!")
