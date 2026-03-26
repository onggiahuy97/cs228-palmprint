"""
Image Corruptions for Robustness Evaluation

Implements common real-world capture variations for palmprint verification:
- Rotation (hand pose changes)
- Scale (distance variations)
- Brightness/Contrast (lighting conditions)
- Blur (motion/focus issues)
- JPEG compression (transmission/storage)
- Occlusion (partial hand coverage)
"""

import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torchvision import transforms
from typing import Callable, Dict, List, Tuple, Optional
import cv2


# =============================================================================
# ROTATION CORRUPTIONS
# =============================================================================

def rotate(image: Image.Image, angle: float) -> Image.Image:
    """Rotate image by given angle (degrees)."""
    return image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=0)


def get_rotation_corruption(severity: int) -> Callable:
    """
    Get rotation corruption function.
    Severity 1-5 maps to ±10°, ±15°, ±20°, ±25°, ±30°
    """
    angles = [10, 15, 20, 25, 30]
    angle = angles[severity - 1]
    # Randomly choose direction
    def corruption(img):
        direction = np.random.choice([-1, 1])
        return rotate(img, direction * angle)
    return corruption


# =============================================================================
# SCALE CORRUPTIONS
# =============================================================================

def scale(image: Image.Image, factor: float) -> Image.Image:
    """Scale image by given factor, maintaining original size with padding/cropping."""
    w, h = image.size
    new_w, new_h = int(w * factor), int(h * factor)
    
    # Resize
    scaled = image.resize((new_w, new_h), Image.BILINEAR)
    
    # Create output image (same size as original)
    result = Image.new(image.mode, (w, h), 0)
    
    if factor < 1.0:
        # Smaller: center the scaled image
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2
        result.paste(scaled, (offset_x, offset_y))
    else:
        # Larger: crop the center
        offset_x = (new_w - w) // 2
        offset_y = (new_h - h) // 2
        result = scaled.crop((offset_x, offset_y, offset_x + w, offset_y + h))
    
    return result


def get_scale_corruption(severity: int) -> Callable:
    """
    Get scale corruption function.
    Severity 1-5 maps to scale factors further from 1.0
    """
    # Alternate between zoom in and zoom out
    factors = [0.9, 1.1, 0.85, 1.15, 0.8, 1.2, 0.75, 1.25, 0.7, 1.3]
    factor = factors[(severity - 1) * 2 + np.random.randint(0, 2)]
    return lambda img: scale(img, factor)


# =============================================================================
# BRIGHTNESS/CONTRAST CORRUPTIONS
# =============================================================================

def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Adjust brightness. factor > 1 = brighter, < 1 = darker."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Adjust contrast. factor > 1 = more contrast, < 1 = less."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def get_brightness_corruption(severity: int) -> Callable:
    """
    Get brightness corruption function.
    Severity 1-5 maps to increasingly extreme brightness changes.
    """
    # Factors: 1.0 = no change, <1 = darker, >1 = brighter
    factors = [
        (0.8, 1.2),   # Mild
        (0.7, 1.3),   # Light
        (0.6, 1.5),   # Moderate
        (0.5, 1.7),   # Strong
        (0.4, 2.0),   # Severe
    ]
    low, high = factors[severity - 1]
    def corruption(img):
        factor = np.random.uniform(low, high)
        return adjust_brightness(img, factor)
    return corruption


def get_contrast_corruption(severity: int) -> Callable:
    """
    Get contrast corruption function.
    Severity 1-5 maps to increasingly extreme contrast changes.
    """
    factors = [
        (0.8, 1.2),
        (0.7, 1.4),
        (0.6, 1.6),
        (0.5, 1.8),
        (0.4, 2.0),
    ]
    low, high = factors[severity - 1]
    def corruption(img):
        factor = np.random.uniform(low, high)
        return adjust_contrast(img, factor)
    return corruption


# =============================================================================
# BLUR CORRUPTIONS
# =============================================================================

def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Apply Gaussian blur with given sigma."""
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def motion_blur(image: Image.Image, kernel_size: int, angle: float = 0) -> Image.Image:
    """Apply motion blur with given kernel size and angle."""
    # Convert to numpy
    img_array = np.array(image)
    
    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    
    # Rotate kernel for angle
    if angle != 0:
        center = (kernel_size // 2, kernel_size // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
        kernel = kernel / kernel.sum()
    
    # Apply filter
    if len(img_array.shape) == 3:
        result = cv2.filter2D(img_array, -1, kernel)
    else:
        result = cv2.filter2D(img_array, -1, kernel)
    
    return Image.fromarray(result.astype(np.uint8))


def get_gaussian_blur_corruption(severity: int) -> Callable:
    """
    Get Gaussian blur corruption function.
    Severity 1-5 maps to sigma values.
    """
    sigmas = [0.5, 1.0, 1.5, 2.0, 3.0]
    sigma = sigmas[severity - 1]
    return lambda img: gaussian_blur(img, sigma)


def get_motion_blur_corruption(severity: int) -> Callable:
    """
    Get motion blur corruption function.
    Severity 1-5 maps to kernel sizes.
    """
    kernel_sizes = [3, 5, 7, 9, 11]
    kernel_size = kernel_sizes[severity - 1]
    def corruption(img):
        angle = np.random.uniform(0, 360)
        return motion_blur(img, kernel_size, angle)
    return corruption


# =============================================================================
# JPEG COMPRESSION CORRUPTION
# =============================================================================

def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG compression with given quality (1-100)."""
    # Save to buffer with JPEG compression
    buffer = io.BytesIO()
    
    # Convert to RGB if needed (JPEG doesn't support all modes)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    
    # Load back
    return Image.open(buffer).copy()


def get_jpeg_corruption(severity: int) -> Callable:
    """
    Get JPEG compression corruption function.
    Severity 1-5 maps to quality levels (lower = more compression).
    """
    qualities = [80, 60, 40, 25, 10]
    quality = qualities[severity - 1]
    return lambda img: jpeg_compress(img, quality)


# =============================================================================
# OCCLUSION CORRUPTION
# =============================================================================

def random_occlusion(image: Image.Image, area_fraction: float) -> Image.Image:
    """
    Apply random rectangular occlusion covering given fraction of image area.
    """
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Calculate occlusion rectangle size
    total_area = h * w
    occlude_area = total_area * area_fraction
    
    # Random aspect ratio between 0.5 and 2.0
    aspect = np.random.uniform(0.5, 2.0)
    occlude_h = int(np.sqrt(occlude_area / aspect))
    occlude_w = int(occlude_area / occlude_h)
    
    # Clamp to image size
    occlude_h = min(occlude_h, h)
    occlude_w = min(occlude_w, w)
    
    # Random position
    top = np.random.randint(0, h - occlude_h + 1)
    left = np.random.randint(0, w - occlude_w + 1)
    
    # Apply occlusion (black rectangle)
    result = img_array.copy()
    if len(result.shape) == 3:
        result[top:top+occlude_h, left:left+occlude_w, :] = 0
    else:
        result[top:top+occlude_h, left:left+occlude_w] = 0
    
    return Image.fromarray(result)


def get_occlusion_corruption(severity: int) -> Callable:
    """
    Get random occlusion corruption function.
    Severity 1-5 maps to area fractions: 10%, 15%, 20%, 25%, 30%
    """
    fractions = [0.10, 0.15, 0.20, 0.25, 0.30]
    fraction = fractions[severity - 1]
    return lambda img: random_occlusion(img, fraction)


# =============================================================================
# NOISE CORRUPTION
# =============================================================================

def gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    """Add Gaussian noise with given standard deviation."""
    img_array = np.array(image).astype(np.float32)
    
    noise = np.random.normal(0, sigma, img_array.shape)
    noisy = img_array + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy)


def get_noise_corruption(severity: int) -> Callable:
    """
    Get Gaussian noise corruption function.
    Severity 1-5 maps to noise sigma values.
    """
    sigmas = [5, 10, 20, 35, 50]
    sigma = sigmas[severity - 1]
    return lambda img: gaussian_noise(img, sigma)


# =============================================================================
# CORRUPTION REGISTRY
# =============================================================================

CORRUPTION_FUNCTIONS = {
    'rotation': get_rotation_corruption,
    'scale': get_scale_corruption,
    'brightness': get_brightness_corruption,
    'contrast': get_contrast_corruption,
    'gaussian_blur': get_gaussian_blur_corruption,
    'motion_blur': get_motion_blur_corruption,
    'jpeg': get_jpeg_corruption,
    'occlusion': get_occlusion_corruption,
    'noise': get_noise_corruption,
}

CORRUPTION_DESCRIPTIONS = {
    'rotation': 'Hand rotation/pose changes',
    'scale': 'Distance/scale variations',
    'brightness': 'Lighting brightness changes',
    'contrast': 'Lighting contrast changes',
    'gaussian_blur': 'Focus/defocus blur',
    'motion_blur': 'Hand motion blur',
    'jpeg': 'JPEG compression artifacts',
    'occlusion': 'Partial hand occlusion',
    'noise': 'Sensor/environmental noise',
}

SEVERITY_DESCRIPTIONS = {
    1: 'Mild',
    2: 'Light',
    3: 'Moderate',
    4: 'Strong',
    5: 'Severe',
}


def get_corruption(name: str, severity: int) -> Callable:
    """
    Get a corruption function by name and severity.
    
    Args:
        name: Corruption type name
        severity: Severity level 1-5
    
    Returns:
        Callable that takes PIL Image and returns corrupted PIL Image
    """
    if name not in CORRUPTION_FUNCTIONS:
        raise ValueError(f"Unknown corruption: {name}. Available: {list(CORRUPTION_FUNCTIONS.keys())}")
    if severity < 1 or severity > 5:
        raise ValueError(f"Severity must be 1-5, got {severity}")
    
    return CORRUPTION_FUNCTIONS[name](severity)


def get_all_corruptions() -> List[str]:
    """Get list of all available corruption names."""
    return list(CORRUPTION_FUNCTIONS.keys())


class CorruptedTransform:
    """
    Transform wrapper that applies a corruption to images.
    Can be composed with other torchvision transforms.
    """
    
    def __init__(self, corruption_name: str, severity: int):
        self.corruption_name = corruption_name
        self.severity = severity
        self.corruption_fn = get_corruption(corruption_name, severity)
    
    def __call__(self, image: Image.Image) -> Image.Image:
        return self.corruption_fn(image)
    
    def __repr__(self) -> str:
        return f"CorruptedTransform({self.corruption_name}, severity={self.severity})"


if __name__ == '__main__':
    # Test all corruptions
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Load a sample image
    sample_path = Path('datasets/Grayscale_128_128/H_ID001/L/roi_H_ID001_L_01_.JPG')
    if not sample_path.exists():
        # Try to find any image
        sample_path = list(Path('datasets/Grayscale_128_128').glob('*/L/*.JPG'))[0]
    
    img = Image.open(sample_path).convert('RGB')
    
    print(f"Testing corruptions on: {sample_path}")
    print(f"Original size: {img.size}")
    
    # Create visualization
    corruptions = get_all_corruptions()
    n_corruptions = len(corruptions)
    n_severities = 5
    
    fig, axes = plt.subplots(n_corruptions + 1, n_severities + 1, 
                              figsize=(3 * (n_severities + 1), 3 * (n_corruptions + 1)))
    
    # Original image in top-left
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Labels for severity
    for s in range(1, n_severities + 1):
        axes[0, s].text(0.5, 0.5, f'Severity {s}\n{SEVERITY_DESCRIPTIONS[s]}',
                        ha='center', va='center', fontsize=12)
        axes[0, s].axis('off')
    
    # Apply each corruption
    for i, corruption_name in enumerate(corruptions):
        # Label row
        axes[i + 1, 0].text(0.5, 0.5, corruption_name.replace('_', '\n'),
                           ha='center', va='center', fontsize=10)
        axes[i + 1, 0].axis('off')
        
        for s in range(1, n_severities + 1):
            corruption_fn = get_corruption(corruption_name, s)
            corrupted = corruption_fn(img.copy())
            axes[i + 1, s].imshow(corrupted)
            axes[i + 1, s].axis('off')
    
    plt.tight_layout()
    plt.savefig('corruption_samples.png', dpi=150, bbox_inches='tight')
    print("Saved corruption_samples.png")
    plt.close()
