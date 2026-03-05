# -*- coding: utf-8 -*-
"""
PikSign Spectral Watermark
FFT-based watermark embedding in frequency domain.
"""

import numpy as np


class SpectralWatermark:
    """Spectral (FFT-based) watermark embedding and detection."""
    
    def __init__(self, strength: float = 0.03):
        self.strength = strength
        self.watermark_pattern = self._generate_pattern()
    
    def _generate_pattern(self) -> np.ndarray:
        """Generate watermark pattern."""
        np.random.seed(42)
        return np.random.randn(32, 32)
    
    def embed(self, img_array: np.ndarray) -> np.ndarray:
        """
        Embed spectral watermark in image.
        
        Args:
            img_array: Image as float32 array (0-1 range)
            
        Returns:
            np.ndarray: Watermarked image
        """
        watermarked = img_array.copy()
        h, w = img_array.shape[:2]
        
        for c in range(3):
            try:
                f_transform = np.fft.fft2(img_array[:, :, c])
                f_shift = np.fft.fftshift(f_transform)
                
                center_h, center_w = h // 2, w // 2
                pattern_h, pattern_w = self.watermark_pattern.shape
                
                start_h = max(0, center_h - pattern_h // 2)
                start_w = max(0, center_w - pattern_w // 2)
                end_h = min(h, start_h + pattern_h)
                end_w = min(w, start_w + pattern_w)
                
                actual_h = end_h - start_h
                actual_w = end_w - start_w
                
                if actual_h > 0 and actual_w > 0:
                    pattern_crop = self.watermark_pattern[:actual_h, :actual_w]
                    magnitude = np.abs(f_shift[start_h:end_h, start_w:end_w])
                    # Reduced embedding strength for quality
                    f_shift[start_h:end_h, start_w:end_w] += (
                        pattern_crop * magnitude * self.strength * 0.5
                    )
                
                f_ishift = np.fft.ifftshift(f_shift)
                img_back = np.fft.ifft2(f_ishift)
                watermarked[:, :, c] = np.real(img_back)
            except:
                pass
        
        return np.clip(watermarked, 0, 1)
    
    def detect(self, img_array: np.ndarray) -> float:
        """
        Detect spectral watermark in image.
        
        Args:
            img_array: Image to check
            
        Returns:
            float: Correlation score (higher = stronger detection)
        """
        correlations = []
        h, w = img_array.shape[:2]
        
        for c in range(3):
            try:
                f_transform = np.fft.fft2(img_array[:, :, c])
                f_shift = np.fft.fftshift(f_transform)
                
                center_h, center_w = h // 2, w // 2
                pattern_h, pattern_w = self.watermark_pattern.shape
                
                start_h = max(0, center_h - pattern_h // 2)
                start_w = max(0, center_w - pattern_w // 2)
                end_h = min(h, start_h + pattern_h)
                end_w = min(w, start_w + pattern_w)
                
                actual_h = end_h - start_h
                actual_w = end_w - start_w
                
                if actual_h > 0 and actual_w > 0:
                    extracted = np.abs(f_shift[start_h:end_h, start_w:end_w])
                    pattern_crop = self.watermark_pattern[:actual_h, :actual_w]
                    
                    corr = np.corrcoef(
                        extracted.flatten(), 
                        pattern_crop.flatten()
                    )[0, 1]
                    
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            except:
                pass
        
        return np.mean(correlations) if correlations else 0.0
