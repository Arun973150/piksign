# -*- coding: utf-8 -*-
"""
PikSign Real Image Validator

Pre-classification calibration that runs BEFORE AI detectors.
Checks for real camera/photo indicators to reduce false positives.

PURPOSE:
- Detect signs of genuine camera-captured images
- If 3+ real indicators present -> penalize AI detection scores
- Prevents false positives on real photos

INDICATORS CHECKED:
1. JPEG 8x8 DCT grid alignment (cameras produce aligned grids)
2. Sensor noise patterns (Gaussian noise distribution)
3. Chromatic aberration (lens imperfections)
4. Natural color histogram distribution
5. Image statistics (real photos have specific characteristics)
"""

import numpy as np
import cv2
from PIL import Image
from scipy.fftpack import dct
from typing import Dict, Any, Tuple
import warnings

warnings.filterwarnings('ignore')


def _safe_prob(x: float) -> float:
    """Clamp value to valid probability range [0, 1]."""
    return float(min(max(x, 0.0), 1.0))


class RealImageValidator:
    """
    Real Image Likelihood Estimator.
    
    Checks for indicators that suggest an image is a real camera photo:
    1. JPEG compression artifacts (8x8 DCT blocking pattern)
    2. Sensor noise patterns (read noise, shot noise characteristics)
    3. Chromatic aberration (lens imperfections)
    4. Natural color distribution (histogram patterns)
    5. Image statistics consistency
    
    If 3+ indicators are present, the image is likely real and AI scores
    should be penalized.
    """
    
    def __init__(self):
        # Thresholds for each indicator
        self.jpeg_grid_threshold = 0.05  # Grid strength > this = real JPEG
        self.noise_gaussian_threshold = 0.85  # Gaussian fit > this = real noise
        self.chromatic_threshold = 0.3  # Aberration strength > this = real lens
        self.histogram_natural_threshold = 0.6  # Natural histogram score
        
    def validate(self, image_path: str) -> Dict[str, Any]:
        """
        Validate if image shows signs of being a real camera photo.
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: Validation results with real_indicators count and scores
        """
        try:
            # Load image
            pil_image = Image.open(image_path)
            image = np.array(pil_image.convert('RGB'))
            
            results = {
                'status': 'success',
                'real_indicators': 0,
                'indicators': {},
                'recommendation': 'proceed_with_ai_detection'
            }
            
            # 1. JPEG Grid Detection
            jpeg_score, has_jpeg_grid = self._check_jpeg_grid(image)
            results['indicators']['jpeg_grid'] = {
                'score': jpeg_score,
                'is_real_indicator': has_jpeg_grid
            }
            if has_jpeg_grid:
                results['real_indicators'] += 1
                
            # 2. Sensor Noise Pattern
            noise_score, has_real_noise = self._check_sensor_noise(image)
            results['indicators']['sensor_noise'] = {
                'score': noise_score,
                'is_real_indicator': has_real_noise
            }
            if has_real_noise:
                results['real_indicators'] += 1
                
            # 3. Chromatic Aberration
            ca_score, has_ca = self._check_chromatic_aberration(image)
            results['indicators']['chromatic_aberration'] = {
                'score': ca_score,
                'is_real_indicator': has_ca
            }
            if has_ca:
                results['real_indicators'] += 1
                
            # 4. Natural Color Distribution
            color_score, has_natural_color = self._check_color_distribution(image)
            results['indicators']['color_distribution'] = {
                'score': color_score,
                'is_real_indicator': has_natural_color
            }
            if has_natural_color:
                results['real_indicators'] += 1
                
            # 5. Image Statistics
            stats_score, has_real_stats = self._check_image_statistics(image)
            results['indicators']['image_statistics'] = {
                'score': stats_score,
                'is_real_indicator': has_real_stats
            }
            if has_real_stats:
                results['real_indicators'] += 1
            
            # Recommendation based on indicator count
            if results['real_indicators'] >= 3:
                results['recommendation'] = 'penalize_ai_scores'
                results['penalty_factor'] = 0.7  # Reduce AI scores by 30%
            elif results['real_indicators'] >= 2:
                results['recommendation'] = 'slight_penalty'
                results['penalty_factor'] = 0.85
            else:
                results['recommendation'] = 'proceed_normally'
                results['penalty_factor'] = 1.0
                
            # Overall real image likelihood
            total_score = (
                jpeg_score * 0.25 +
                noise_score * 0.20 +
                ca_score * 0.15 +
                color_score * 0.20 +
                stats_score * 0.20
            )
            results['real_image_likelihood'] = _safe_prob(total_score)
            
            return results
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'real_indicators': 0,
                'penalty_factor': 1.0,
                'recommendation': 'proceed_normally'
            }
    
    def _check_jpeg_grid(self, image: np.ndarray) -> Tuple[float, bool]:
        """
        Check for 8x8 JPEG DCT block artifacts.
        
        Real JPEG photos from cameras have aligned 8x8 grid patterns.
        AI-generated images (often PNG) lack this pattern.
        
        Returns:
            (score, is_real_indicator): Score 0-1, and whether it indicates real
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
            
            # Compute horizontal and vertical differences
            h_diff = np.abs(gray[:-1, :] - gray[1:, :])
            v_diff = np.abs(gray[:, :-1] - gray[:, 1:])
            
            # Sum along rows/cols to find periodicity
            h_sum = np.sum(h_diff, axis=1)
            v_sum = np.sum(v_diff, axis=0)
            
            def check_8x8_periodicity(signal):
                """Check for peaks at multiples of 8."""
                if len(signal) < 24:
                    return 0.0
                    
                # Get values at block boundaries (every 8th pixel)
                boundaries = signal[7::8]  # indices 7, 15, 23, ...
                # Get values at mid-block positions
                midpoints = signal[3::8]   # indices 3, 11, 19, ...
                
                if len(boundaries) < 2 or len(midpoints) < 2:
                    return 0.0
                    
                mean_boundary = np.mean(boundaries)
                mean_midpoint = np.mean(midpoints)
                
                if mean_midpoint < 1e-6:
                    return 0.0
                    
                # Ratio of boundary to midpoint energy
                periodicity = (mean_boundary - mean_midpoint) / (mean_midpoint + 1e-6)
                return max(0.0, periodicity)
            
            h_period = check_8x8_periodicity(h_sum)
            v_period = check_8x8_periodicity(v_sum)
            
            grid_strength = (h_period + v_period) / 2.0
            
            # Strong grid = likely real JPEG
            is_real = grid_strength > self.jpeg_grid_threshold
            
            # Convert to 0-1 score (higher = more likely real)
            score = min(grid_strength / 0.1, 1.0)
            
            return (score, is_real)
            
        except Exception:
            return (0.0, False)
    
    def _check_sensor_noise(self, image: np.ndarray) -> Tuple[float, bool]:
        """
        Check for camera sensor noise characteristics.
        
        Real camera photos have specific noise patterns:
        - Gaussian-distributed noise
        - Consistent noise level in smooth areas
        - Presence of shot noise (intensity-dependent)
        
        AI images tend to have too-smooth areas or artificial noise.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
            
            # Extract noise using high-pass filter
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = gray - blurred
            
            # Analyze noise distribution
            noise_mean = np.mean(noise)
            noise_std = np.std(noise)
            
            # Check if noise follows Gaussian distribution
            # Real sensor noise is approximately Gaussian
            if noise_std < 0.5:
                # Too smooth - likely AI or heavily denoised
                return (0.2, False)
            
            # Compute histogram of noise
            hist, bins = np.histogram(noise, bins=50, range=(-50, 50))
            hist = hist.astype(np.float64) / (np.sum(hist) + 1e-10)
            
            # Generate ideal Gaussian distribution
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ideal_gaussian = np.exp(-0.5 * ((bin_centers - noise_mean) / (noise_std + 1e-6)) ** 2)
            ideal_gaussian /= (np.sum(ideal_gaussian) + 1e-10)
            
            # Compute correlation with ideal Gaussian
            correlation = np.corrcoef(hist, ideal_gaussian)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
                
            # Score based on Gaussian fit
            score = _safe_prob((correlation + 1) / 2)  # Map -1,1 to 0,1
            
            # Also check noise level - real photos have moderate noise (2-15 std)
            noise_level_ok = 2.0 < noise_std < 15.0
            
            is_real = score > self.noise_gaussian_threshold and noise_level_ok
            
            return (score, is_real)
            
        except Exception:
            return (0.0, False)
    
    def _check_chromatic_aberration(self, image: np.ndarray) -> Tuple[float, bool]:
        """
        Check for chromatic aberration (lens imperfection).
        
        Real camera lenses cause color fringing at high-contrast edges,
        especially near image corners. AI generators rarely simulate this.
        """
        try:
            # Split channels
            b, g, r = cv2.split(image)
            
            # Detect edges in each channel
            edges_r = cv2.Canny(r, 50, 150)
            edges_g = cv2.Canny(g, 50, 150)
            edges_b = cv2.Canny(b, 50, 150)
            
            # ChromaticAberration causes R/B edges to be slightly shifted from G
            # Check edge alignment between channels
            
            # Dilate green edges slightly
            kernel = np.ones((3, 3), np.uint8)
            edges_g_dilated = cv2.dilate(edges_g, kernel, iterations=1)
            
            # Check how much R/B edges fall outside dilated G
            r_outside = np.sum(edges_r & ~edges_g_dilated)
            b_outside = np.sum(edges_b & ~edges_g_dilated)
            total_edges = np.sum(edges_g) + 1
            
            # Chromatic aberration ratio
            ca_ratio = (r_outside + b_outside) / total_edges
            
            # Focus on corners where CA is stronger
            h, w = image.shape[:2]
            corner_size = min(h, w) // 4
            
            corners = [
                (0, corner_size, 0, corner_size),
                (0, corner_size, w-corner_size, w),
                (h-corner_size, h, 0, corner_size),
                (h-corner_size, h, w-corner_size, w)
            ]
            
            corner_ca = 0
            for y1, y2, x1, x2 in corners:
                r_corner = np.sum(edges_r[y1:y2, x1:x2] & ~edges_g_dilated[y1:y2, x1:x2])
                b_corner = np.sum(edges_b[y1:y2, x1:x2] & ~edges_g_dilated[y1:y2, x1:x2])
                g_corner = np.sum(edges_g[y1:y2, x1:x2]) + 1
                corner_ca += (r_corner + b_corner) / g_corner
            
            corner_ca /= 4
            
            # Score: higher CA = more likely real
            score = _safe_prob(min(ca_ratio * 5, 1.0))
            corner_score = _safe_prob(min(corner_ca * 3, 1.0))
            
            combined_score = 0.4 * score + 0.6 * corner_score
            is_real = combined_score > self.chromatic_threshold
            
            return (combined_score, is_real)
            
        except Exception:
            return (0.0, False)
    
    def _check_color_distribution(self, image: np.ndarray) -> Tuple[float, bool]:
        """
        Check for natural color histogram distribution.
        
        Real photos follow specific color distributions:
        - Smooth, continuous histograms
        - No artificial peaks or gaps
        - Natural saturation distribution
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            
            # Analyze saturation histogram
            sat_hist, _ = np.histogram(s, bins=32, range=(0, 256))
            sat_hist = sat_hist.astype(np.float64) / (np.sum(sat_hist) + 1e-10)
            
            # Natural photos have smooth saturation distribution
            # Compute smoothness (low variation between adjacent bins)
            sat_diff = np.abs(np.diff(sat_hist))
            sat_smoothness = 1.0 - np.mean(sat_diff) * 10
            
            # Analyze value histogram
            val_hist, _ = np.histogram(v, bins=32, range=(0, 256))
            val_hist = val_hist.astype(np.float64) / (np.sum(val_hist) + 1e-10)
            
            # Natural photos have broader value distribution (not clipped)
            val_range = np.sum(val_hist > 0.01)  # Bins with significant content
            val_spread = val_range / 32.0
            
            # Check for unnatural gaps in histogram
            gaps = np.sum(val_hist < 0.001) / 32.0
            no_gaps_score = 1.0 - gaps
            
            # Combined score
            score = _safe_prob(
                0.4 * max(sat_smoothness, 0) +
                0.3 * val_spread +
                0.3 * no_gaps_score
            )
            
            is_real = score > self.histogram_natural_threshold
            
            return (score, is_real)
            
        except Exception:
            return (0.0, False)
    
    def _check_image_statistics(self, image: np.ndarray) -> Tuple[float, bool]:
        """
        Check general image statistics that differ between real and AI.
        
        Real photos characteristics:
        - Moderate entropy (not too uniform or too random)
        - Natural contrast range
        - Realistic luminance distribution
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Entropy check
            hist, _ = np.histogram(gray, bins=256, range=(0, 256))
            hist = hist.astype(np.float64) / (np.sum(hist) + 1e-10)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # Natural images have entropy between 5 and 7.5
            if 5.0 < entropy < 7.5:
                entropy_score = 1.0
            elif 4.0 < entropy < 8.0:
                entropy_score = 0.6
            else:
                entropy_score = 0.2
            
            # Local contrast check (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            lap_var = np.var(laplacian)
            
            # Real photos have moderate Laplacian variance (100-2000)
            if 100 < lap_var < 2000:
                contrast_score = 1.0
            elif 50 < lap_var < 3000:
                contrast_score = 0.6
            else:
                contrast_score = 0.3
            
            # Dynamic range check
            dynamic_range = float(np.max(gray)) - float(np.min(gray))
            dr_score = min(dynamic_range / 200.0, 1.0)
            
            # Combined score
            score = _safe_prob(
                0.35 * entropy_score +
                0.35 * contrast_score +
                0.30 * dr_score
            )
            
            is_real = score > 0.6
            
            return (score, is_real)
            
        except Exception:
            return (0.0, False)


# Convenience function
def validate_real_image(image_path: str) -> Dict[str, Any]:
    """
    Validate if an image shows signs of being a real camera photo.
    
    Args:
        image_path: Path to image file
        
    Returns:
        dict: Validation results with real_indicators count
    """
    validator = RealImageValidator()
    return validator.validate(image_path)
