# -*- coding: utf-8 -*-
"""
PikSign Configuration Module
Quality-optimized settings for image protection.
"""

import torch


class Config:
    """Configuration optimized for high quality output (PSNR ~40dB, SSIM ~0.93+)"""
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Image size limits
    MIN_SIZE = 256
    MAX_SIZE = 2048
    
    # Epsilon values for perturbation (reduced for quality)
    EPSILON_BASE = 1.5 / 255
    EPSILON_MAX = 3.0 / 255
    
    # Quality targets
    TARGET_PSNR = 40.0
    TARGET_SSIM = 0.93
    
    # CLIP vulnerability threshold
    CLIP_VULNERABILITY_THRESHOLD = 0.7
    SEMANTIC_DRIFT_TARGET = 0.08
    
    # Watermark settings (reduced for quality)
    WATERMARK_STRENGTH = 0.03
    WATERMARK_MESSAGE = b'PikSign2026'
    WATERMARK_METHOD = 'dwtDct'
    
    # Perceptual Hash settings
    PHASH_SIZE = 16
    PHASH_THRESHOLD = 8
    
    # Multi-band Frequency settings (reduced for quality)
    MULTIBAND_STRENGTH = 0.02
    FREQUENCY_BANDS = 4
    
    # Blending ratios (favor original for quality)
    TRANSFORM_BLEND_RATIO = 0.15
    FREQ_DISTORT_BLEND = 0.1

    # LEAT (Latent Ensemble Attack) settings
    LEAT_ENABLED = True
    LEAT_ITERATIONS = 50       # PGD iterations (increased for stronger attack)
    LEAT_STEP_SIZE = 0.01      # Per-iteration step size (a in Algorithm 1)
    LEAT_EPSILON = 0.08        # Max L-inf perturbation (~20/255, stronger bound)
    LEAT_ENCODERS = ['arcface', 'stylegan', 'diffae', 'icface', 'vgg', 'sdvae', 'sdxl_vae']
    LEAT_BLEND_RATIO = 0.75    # Higher blend ratio to let LEAT dominate
    LEAT_FREQ_ATTACK = True    # Enable frequency-domain attack
    LEAT_FREQ_WEIGHT = 0.5     # Frequency attack weight in the ensemble
    
    @classmethod
    def get_info(cls) -> dict:
        """Return configuration summary."""
        return {
            'device': str(cls.DEVICE),
            'epsilon_range': f"{cls.EPSILON_BASE*255:.2f}-{cls.EPSILON_MAX*255:.2f}/255",
            'watermark_strength': cls.WATERMARK_STRENGTH,
            'multiband_strength': cls.MULTIBAND_STRENGTH,
            'target_psnr': cls.TARGET_PSNR,
            'target_ssim': cls.TARGET_SSIM,
            'leat_enabled': cls.LEAT_ENABLED,
            'leat_iterations': cls.LEAT_ITERATIONS,
            'leat_epsilon': cls.LEAT_EPSILON,
            'leat_encoders': cls.LEAT_ENCODERS,
        }
