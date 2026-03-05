# -*- coding: utf-8 -*-
"""
PikSign Confusion Transform
Applies perceptual-safe noise and frequency distortion.
"""

import numpy as np
import torch
from scipy.fftpack import dct, idct


class ConfusionTransform:
    """Applies confusion transforms to disrupt AI interpretation."""
    
    def __init__(self, epsilon: float, freq_weights: tuple):
        self.epsilon = epsilon
        self.freq_weights = freq_weights
    
    def apply(self, img_tensor: torch.Tensor, 
              img_array: np.ndarray) -> tuple:
        """
        Apply confusion transforms.
        
        Args:
            img_tensor: Image as torch tensor
            img_array: Image as numpy array
            
        Returns:
            tuple: (patch_noise, freq_distorted_array)
        """
        patch_noise = self._generate_patch_noise(img_tensor)
        freq_distorted = self._apply_frequency_distortion(img_array)
        return patch_noise, freq_distorted
    
    def _generate_patch_noise(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Generate patch-level perceptual noise."""
        _, _, h, w = img_tensor.shape
        
        # Reduced noise strength for quality
        noise = torch.randn_like(img_tensor) * self.epsilon * 0.2
        
        patch_size = 16  # Larger patches = smoother
        mask = torch.ones_like(img_tensor)
        
        for i in range(0, h, patch_size * 2):
            for j in range(0, w, patch_size * 2):
                if np.random.rand() > 0.8:  # Less frequent patches
                    mask[:, :, i:min(i+patch_size, h), j:min(j+patch_size, w)] = 0.3
        
        return noise * mask
    
    def _apply_frequency_distortion(self, img_array: np.ndarray) -> np.ndarray:
        """Apply frequency domain distortion."""
        distorted = img_array.copy()
        
        for c in range(3):
            channel = img_array[:, :, c]
            h, w = channel.shape
            
            try:
                dct_coef = dct(dct(channel.T, norm='ortho').T, norm='ortho')
                
                h_quarter = h // 4
                w_quarter = w // 4
                h_three_quarter = 3 * h // 4
                w_three_quarter = 3 * w // 4
                
                # Reduced noise in frequency domain
                if h_quarter > 0 and w_quarter > 0:
                    noise_low = (
                        np.random.randn(h_quarter, w_quarter) 
                        * self.epsilon * self.freq_weights[0] * 0.15
                    )
                    dct_coef[:h_quarter, :w_quarter] += noise_low
                
                mid_h_size = h_three_quarter - h_quarter
                mid_w_size = w_three_quarter - w_quarter
                if mid_h_size > 0 and mid_w_size > 0:
                    noise_mid = (
                        np.random.randn(mid_h_size, mid_w_size) 
                        * self.epsilon * self.freq_weights[1] * 0.15
                    )
                    dct_coef[h_quarter:h_three_quarter, w_quarter:w_three_quarter] += noise_mid
                
                high_h_size = h - h_three_quarter
                high_w_size = w - w_three_quarter
                if high_h_size > 0 and high_w_size > 0:
                    noise_high = (
                        np.random.randn(high_h_size, high_w_size) 
                        * self.epsilon * self.freq_weights[2] * 0.15
                    )
                    dct_coef[h_three_quarter:, w_three_quarter:] += noise_high
                
                distorted[:, :, c] = idct(idct(dct_coef.T, norm='ortho').T, norm='ortho')
            except:
                pass
        
        return np.clip(distorted, 0, 1)
