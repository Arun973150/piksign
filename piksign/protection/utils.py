# -*- coding: utf-8 -*-
"""
PikSign Utility Functions
Common utilities for image processing and metrics.
"""

import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as calc_ssim
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
import torch
import warnings

warnings.filterwarnings('ignore')


def load_image(image_path: str) -> tuple:
    """
    Load an image and return as float32 array and torch tensor.
    
    Returns:
        tuple: (img_array, img_tensor, original_pil)
    """
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_array, img_tensor, img


def save_image(img_array: np.ndarray, output_path: str, quality: int = 100) -> str:
    """
    Save a float32 image array to file.
    
    Args:
        img_array: Image as float32 array (0-1 range)
        output_path: Output file path
        quality: JPEG quality (default 100)
    
    Returns:
        str: Saved file path
    """
    img_uint8 = (img_array * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    
    if output_path.lower().endswith('.png'):
        img_pil.save(output_path, compress_level=1)
    else:
        img_pil.save(output_path, quality=quality)
    
    return output_path


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute PSNR between two images."""
    orig_uint8 = (original * 255).astype(np.uint8) if original.max() <= 1 else original.astype(np.uint8)
    proc_uint8 = (processed * 255).astype(np.uint8) if processed.max() <= 1 else processed.astype(np.uint8)
    return calc_psnr(orig_uint8, proc_uint8)


def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute SSIM between two images."""
    orig_uint8 = (original * 255).astype(np.uint8) if original.max() <= 1 else original.astype(np.uint8)
    proc_uint8 = (processed * 255).astype(np.uint8) if processed.max() <= 1 else processed.astype(np.uint8)
    return calc_ssim(orig_uint8, proc_uint8, channel_axis=2, data_range=255)


def array_to_tensor(img_array: np.ndarray, device: torch.device = None) -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    if device:
        tensor = tensor.to(device)
    return tensor


def tensor_to_array(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return img_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


def print_banner(title: str, char: str = "=", width: int = 80):
    """Print a formatted banner."""
    print(char * width)
    try:
        print(f"\U0001f6e1\ufe0f  {title}")
    except (UnicodeEncodeError, UnicodeDecodeError):
        print(f"[*] {title}")
    print(char * width)
