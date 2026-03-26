#!/usr/bin/env python3
"""
Palmprint Verification Demo

Verifies whether two palmprint images belong to the same person.

Usage:
    python src/demo.py --image1 path/to/palm1.jpg --image2 path/to/palm2.jpg
    
    # Or use sample images from dataset:
    python src/demo.py --sample-match      # Two images from same person
    python src/demo.py --sample-nonmatch   # Two images from different people
"""

import sys
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model import PalmprintVerifier


def load_model(checkpoint_path: str, device: str) -> PalmprintVerifier:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
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
    
    return model


def load_and_preprocess(image_path: str, image_size: int = 128) -> torch.Tensor:
    """Load and preprocess a palmprint image."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path)
    
    # Convert to RGB if needed
    if image.mode == 'L':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension


def compute_similarity(
    model: PalmprintVerifier,
    image1_path: str,
    image2_path: str,
    device: str
) -> float:
    """Compute cosine similarity between two palmprint images."""
    # Load images
    img1 = load_and_preprocess(image1_path).to(device)
    img2 = load_and_preprocess(image2_path).to(device)
    
    # Get embeddings
    with torch.no_grad():
        emb1 = model.get_embedding(img1)
        emb2 = model.get_embedding(img2)
    
    # Cosine similarity
    similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
    
    return similarity


def verify(similarity: float, threshold: float = 0.30) -> tuple:
    """Make verification decision based on similarity and threshold."""
    is_match = similarity >= threshold
    confidence = abs(similarity - threshold) / (1.0 - threshold) if is_match else abs(similarity - threshold) / threshold
    confidence = min(confidence, 1.0)
    return is_match, confidence


def get_sample_images(match: bool, data_root: str = 'datasets/Grayscale_128_128') -> tuple:
    """Get sample image pairs for demo."""
    data_path = Path(data_root)
    
    if match:
        # Same person: get two images from same subject/hand
        subject_dir = list(data_path.glob('H_ID001/L'))[0]
        images = sorted(subject_dir.glob('*.JPG'))[:2]
        if len(images) < 2:
            images = sorted(subject_dir.glob('*.jpg'))[:2]
        return str(images[0]), str(images[1])
    else:
        # Different people: get images from different subjects
        subject1 = list(data_path.glob('H_ID001/L/*.JPG')) or list(data_path.glob('H_ID001/L/*.jpg'))
        subject2 = list(data_path.glob('H_ID050/L/*.JPG')) or list(data_path.glob('H_ID050/L/*.jpg'))
        return str(subject1[0]), str(subject2[0])


def print_result(
    image1: str,
    image2: str,
    similarity: float,
    is_match: bool,
    threshold: float
):
    """Print formatted verification result."""
    print("\n" + "=" * 60)
    print("🖐️  PALMPRINT VERIFICATION RESULT")
    print("=" * 60)
    print(f"\n📷 Image 1: {Path(image1).name}")
    print(f"📷 Image 2: {Path(image2).name}")
    print(f"\n📊 Similarity Score: {similarity:.4f}")
    print(f"📏 Threshold: {threshold:.4f}")
    
    if is_match:
        print(f"\n✅ Decision: MATCH - Same person")
    else:
        print(f"\n❌ Decision: NO MATCH - Different people")
    
    # Visual similarity bar
    bar_len = 40
    filled = int(similarity * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    thresh_pos = int(threshold * bar_len)
    print(f"\n[{bar}]")
    print(f" {'':>{thresh_pos}}↑ threshold")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Palmprint Verification Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/demo.py --image1 palm1.jpg --image2 palm2.jpg
  python src/demo.py --sample-match
  python src/demo.py --sample-nonmatch
        """
    )
    
    parser.add_argument('--image1', type=str, help='Path to first palmprint image')
    parser.add_argument('--image2', type=str, help='Path to second palmprint image')
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/baseline_v2/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.30,
                        help='Verification threshold (default: 0.30)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, mps, cpu')
    parser.add_argument('--sample-match', action='store_true',
                        help='Run demo with matching sample pair')
    parser.add_argument('--sample-nonmatch', action='store_true',
                        help='Run demo with non-matching sample pair')
    
    args = parser.parse_args()
    
    # Handle sample mode
    if args.sample_match:
        args.image1, args.image2 = get_sample_images(match=True)
        print(f"Using sample MATCHING pair")
    elif args.sample_nonmatch:
        args.image1, args.image2 = get_sample_images(match=False)
        print(f"Using sample NON-MATCHING pair")
    
    # Validate inputs
    if not args.image1 or not args.image2:
        parser.error("Please provide --image1 and --image2, or use --sample-match/--sample-nonmatch")
    
    if not Path(args.image1).exists():
        print(f"Error: Image not found: {args.image1}")
        sys.exit(1)
    if not Path(args.image2).exists():
        print(f"Error: Image not found: {args.image2}")
        sys.exit(1)
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
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
    
    # Load model
    print(f"Loading model... (device: {device})")
    model = load_model(args.checkpoint, device)
    
    # Compute similarity
    similarity = compute_similarity(model, args.image1, args.image2, device)
    
    # Make decision
    is_match, confidence = verify(similarity, args.threshold)
    
    # Print result
    print_result(args.image1, args.image2, similarity, is_match, args.threshold)


if __name__ == '__main__':
    main()
