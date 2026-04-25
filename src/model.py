"""
Model Architecture for Palmprint Verification

ResNet18-based embedding network with ArcFace margin loss for
classification-based embedding learning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PalmprintEmbedder(nn.Module):
    """
    Embedding network for palmprint verification.
    
    Uses a pretrained backbone (ResNet18 by default) followed by
    a projection head to produce fixed-size embeddings.
    
    Args:
        embedding_dim: Dimension of output embeddings (default: 256)
        backbone: Backbone architecture ('resnet18', 'resnet34', 'mobilenetv3')
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate before final projection
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Load backbone
        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            backbone_dim = 512
        elif backbone == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            backbone_dim = 512
        elif backbone == 'mobilenetv3':
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            backbone_dim = 576
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove classification head
        if backbone.startswith('resnet'):
            self.backbone.fc = nn.Identity()
        elif backbone == 'mobilenetv3':
            self.backbone.classifier = nn.Identity()
        
        # Projection head: backbone_dim -> embedding_dim
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.BatchNorm1d(backbone_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get embeddings.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            L2-normalized embeddings [B, embedding_dim]
        """
        features = self.backbone(x)
        embeddings = self.projector(features)
        # L2 normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class ArcFaceHead(nn.Module):
    """
    ArcFace (Additive Angular Margin) classification head.
    
    Implements the margin-based softmax loss from:
    "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    
    Args:
        embedding_dim: Dimension of input embeddings
        num_classes: Number of identity classes
        scale: Scaling factor (default: 30.0)
        margin: Angular margin in radians (default: 0.5)
        easy_margin: Whether to use easy margin (default: False)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.5,
        easy_margin: bool = False
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Learnable class weight matrix (normalized during forward)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute margin terms
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with ArcFace margin.
        
        Args:
            embeddings: L2-normalized embeddings [B, embedding_dim]
            labels: Ground truth class labels [B]
        
        Returns:
            Scaled logits with margin applied [B, num_classes]
        """
        # Normalize weights
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity (embeddings are already normalized)
        cosine = F.linear(embeddings, normalized_weight)
        
        # Apply angular margin
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encoding for labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Apply margin only to ground truth class
        output = torch.where(one_hot == 1, phi, cosine)
        output = output * self.scale
        
        return output


class PalmprintVerifier(nn.Module):
    """
    Complete palmprint verification model.
    
    Combines the embedding network with ArcFace head for training.
    During inference, only the embedder is used.
    
    Args:
        num_classes: Number of identity classes for training
        embedding_dim: Dimension of embeddings
        backbone: Backbone architecture
        pretrained: Whether to use pretrained weights
        scale: ArcFace scale parameter
        margin: ArcFace margin parameter
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 256,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        scale: float = 30.0,
        margin: float = 0.5
    ):
        super().__init__()
        
        self.embedder = PalmprintEmbedder(
            embedding_dim=embedding_dim,
            backbone=backbone,
            pretrained=pretrained
        )
        
        self.arcface = ArcFaceHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=scale,
            margin=margin
        )
    
    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            labels: Ground truth labels [B] (required for training)
        
        Returns:
            If labels provided: scaled logits [B, num_classes]
            If no labels: embeddings [B, embedding_dim]
        """
        embeddings = self.embedder(x)
        
        if labels is not None:
            return self.arcface(embeddings, labels)
        
        return embeddings
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings for verification (inference mode)."""
        return self.embedder(x)


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two sets of embeddings.
    
    Args:
        emb1: First embeddings [B, D] or [D]
        emb2: Second embeddings [B, D] or [D]
    
    Returns:
        Cosine similarity scores [B] or scalar
    """
    if emb1.dim() == 1:
        emb1 = emb1.unsqueeze(0)
    if emb2.dim() == 1:
        emb2 = emb2.unsqueeze(0)
    
    return F.cosine_similarity(emb1, emb2, dim=1)


if __name__ == '__main__':
    # Quick test
    print("Testing model architecture...")
    
    # Create model
    model = PalmprintVerifier(
        num_classes=1000,
        embedding_dim=256,
        backbone='resnet18',
        pretrained=True
    )
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 128, 128)
    labels = torch.randint(0, 1000, (batch_size,))
    
    # Training mode
    logits = model(x, labels)
    print(f"Training output shape: {logits.shape}")
    
    # Inference mode
    embeddings = model.get_embedding(x)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding norm: {torch.norm(embeddings, dim=1)}")  # Should be ~1.0
    
    # Test similarity
    sim = cosine_similarity(embeddings[0], embeddings[1])
    print(f"Cosine similarity (sample 0 vs 1): {sim.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
