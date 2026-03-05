# -*- coding: utf-8 -*-
"""
PikSign Multi-Band Frequency Watermark
Advanced DCT-based multi-band frequency domain watermarking.
"""

import numpy as np
import cv2
from scipy.fftpack import dct, idct


class MultiBandFrequencyWatermark:
    """Multi-band frequency domain watermarking (DCT-based)."""
    
    def __init__(self, strength: float = 0.02, num_bands: int = 4):
        self.strength = strength
        self.num_bands = num_bands
        self.watermark_seeds = self._generate_band_seeds()
    
    def _generate_band_seeds(self) -> dict:
        """Generate unique seeds for each frequency band."""
        seeds = {}
        for i in range(self.num_bands):
            np.random.seed(42 + i * 10)
            seeds[f'band_{i}'] = np.random.randn(16, 16)
        return seeds
    
    def _create_frequency_bands(self, coefficients: np.ndarray) -> list:
        """Divide frequency domain into bands."""
        h, w = coefficients.shape
        bands = []
        band_size = max(h // (2 ** self.num_bands), 4)
        
        for i in range(self.num_bands):
            start = i * band_size
            end = min((i + 1) * band_size, min(h, w) // 2)
            if start < end:
                bands.append((start, end))
        
        return bands
    
    def embed(self, img_array: np.ndarray) -> np.ndarray:
        """
        Embed multi-band frequency watermark.
        
        Args:
            img_array: Image as float32 array (0-1 range)
            
        Returns:
            np.ndarray: Watermarked image
        """
        watermarked = img_array.copy()
        h, w = img_array.shape[:2]
        
        for c in range(3):
            channel = img_array[:, :, c]
            
            try:
                dct_coef = dct(dct(channel.T, norm='ortho').T, norm='ortho')
                bands = self._create_frequency_bands(dct_coef)
                
                for band_idx, (start, end) in enumerate(bands):
                    if band_idx >= self.num_bands:
                        break
                    
                    band_key = f'band_{band_idx}'
                    pattern = self.watermark_seeds[band_key]
                    
                    band_height = end - start
                    band_width = end - start
                    
                    if band_height > 0 and band_width > 0:
                        pattern_resized = cv2.resize(
                            pattern,
                            (band_width, band_height),
                            interpolation=cv2.INTER_LINEAR
                        )
                        
                        # Adaptive strength - higher bands get slightly more
                        band_strength = self.strength * (
                            0.3 + 0.2 * (band_idx / self.num_bands)
                        )
                        
                        dct_coef[start:end, start:end] += (
                            pattern_resized * band_strength
                        )
                
                watermarked[:, :, c] = idct(
                    idct(dct_coef.T, norm='ortho').T, norm='ortho'
                )
            except Exception:
                pass
        
        return np.clip(watermarked, 0, 1)
    
    def detect(self, img_array: np.ndarray) -> float:
        """
        Detect multi-band frequency watermark.
        
        Returns:
            float: Average correlation across bands
        """
        correlations = []
        
        for c in range(3):
            channel = img_array[:, :, c]
            
            try:
                dct_coef = dct(dct(channel.T, norm='ortho').T, norm='ortho')
                bands = self._create_frequency_bands(dct_coef)
                band_correlations = []
                
                for band_idx, (start, end) in enumerate(bands):
                    if band_idx >= self.num_bands:
                        break
                    
                    band_key = f'band_{band_idx}'
                    pattern = self.watermark_seeds[band_key]
                    
                    band_height = end - start
                    band_width = end - start
                    
                    if band_height > 0 and band_width > 0:
                        pattern_resized = cv2.resize(
                            pattern,
                            (band_width, band_height),
                            interpolation=cv2.INTER_LINEAR
                        )
                        
                        band_coef = dct_coef[start:end, start:end]
                        
                        correlation = np.corrcoef(
                            band_coef.flatten(),
                            pattern_resized.flatten()
                        )[0, 1]
                        
                        if not np.isnan(correlation):
                            band_correlations.append(abs(correlation))
                
                if band_correlations:
                    correlations.append(np.mean(band_correlations))
            except Exception:
                pass
        
        return np.mean(correlations) if correlations else 0.0
    
    def get_band_strengths(self, img_array: np.ndarray) -> dict:
        """Get watermark strength in each frequency band."""
        band_strengths = {f'band_{i}': [] for i in range(self.num_bands)}
        
        for c in range(3):
            channel = img_array[:, :, c]
            
            try:
                dct_coef = dct(dct(channel.T, norm='ortho').T, norm='ortho')
                bands = self._create_frequency_bands(dct_coef)
                
                for band_idx, (start, end) in enumerate(bands):
                    if band_idx >= self.num_bands:
                        break
                    
                    band_key = f'band_{band_idx}'
                    pattern = self.watermark_seeds[band_key]
                    
                    band_height = end - start
                    band_width = end - start
                    
                    if band_height > 0 and band_width > 0:
                        pattern_resized = cv2.resize(
                            pattern,
                            (band_width, band_height),
                            interpolation=cv2.INTER_LINEAR
                        )
                        
                        band_coef = dct_coef[start:end, start:end]
                        
                        correlation = np.corrcoef(
                            band_coef.flatten(),
                            pattern_resized.flatten()
                        )[0, 1]
                        
                        if not np.isnan(correlation):
                            band_strengths[band_key].append(abs(correlation))
            except:
                pass
        
        result = {}
        for band_key in band_strengths:
            if band_strengths[band_key]:
                result[band_key] = np.mean(band_strengths[band_key])
            else:
                result[band_key] = 0.0
        
        return result
