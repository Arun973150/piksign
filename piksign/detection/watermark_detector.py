# -*- coding: utf-8 -*-
"""
PikSign Watermark Detector

Detects AI-embedded spectral watermarks in images.

PURPOSE:
- Detect Stable Signature (Meta/Stability AI) watermarks
- Detect generic DCT/DFT spread-spectrum watermarks
- SynthID heuristic detection via spectral entropy

WATERMARK TYPES:
1. Stable Signature: DWT-DCT embedding in mid-frequency bands
2. Generic DCT: Spread-spectrum in frequency domain
3. SynthID: Token selection patterns (heuristic detection)
"""

import numpy as np
import cv2
from PIL import Image
from scipy.fftpack import dct, idct, fft2, fftshift
from typing import Dict, Any, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Try to import imwatermark for Stable Signature
try:
    from imwatermark import WatermarkDecoder
    HAS_IMWATERMARK = True
except ImportError:
    HAS_IMWATERMARK = False

# Try to import pywt for wavelet analysis
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


def _safe_prob(x: float) -> float:
    """Clamp value to valid probability range [0, 1]."""
    return float(min(max(x, 0.0), 1.0))


class WatermarkDetector:
    """
    Spectral Watermark Detector.
    
    Detects various AI-embedded watermarks:
    
    1. Stable Signature (Meta/Stability AI):
       - Uses DWT-DCT embedding in mid-frequency bands
       - 48-100 bits embedded redundantly
       - Detection via imwatermark library
    
    2. Generic DCT/DFT Watermarks:
       - Spread-spectrum in frequency domain
       - 2D autocorrelation for periodic peaks
       - Cross-correlation with known patterns
    
    3. SynthID Heuristic (Google):
       - Modifies token selection during generation
       - Detectable via spectral entropy analysis
       - Lower entropy in specific frequency bands
    """
    
    def __init__(self):
        self.has_imwatermark = HAS_IMWATERMARK
        self.has_pywt = HAS_PYWT
        
        # Stable Signature decoder
        self.stable_decoder = None
        if HAS_IMWATERMARK:
            try:
                # Try different watermark methods
                self.stable_decoder = WatermarkDecoder('dwtDct', 64)
            except:
                try:
                    self.stable_decoder = WatermarkDecoder('rivaGan', 32)
                except:
                    self.stable_decoder = None
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Detect watermarks in an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: Watermark detection results
        """
        try:
            # Load image
            image = np.array(Image.open(image_path).convert('RGB'))
            
            results = {
                'status': 'success',
                'watermark_detected': False,
                'watermark_confidence': 0.0,
                'watermark_type': None,
                'details': {}
            }
            
            # 1. Try Stable Signature detection
            stable_result = self._detect_stable_signature(image)
            results['details']['stable_signature'] = stable_result
            if stable_result.get('detected', False):
                results['watermark_detected'] = True
                results['watermark_type'] = 'stable_signature'
                results['watermark_confidence'] = stable_result.get('confidence', 0.8)
            
            # 2. Generic DCT watermark detection
            dct_result = self._detect_dct_watermark(image)
            results['details']['dct_watermark'] = dct_result
            if dct_result.get('detected', False) and not results['watermark_detected']:
                results['watermark_detected'] = True
                results['watermark_type'] = 'dct_spread_spectrum'
                results['watermark_confidence'] = dct_result.get('confidence', 0.6)
            
            # 3. SynthID heuristic detection
            synthid_result = self._detect_synthid_heuristic(image)
            results['details']['synthid_heuristic'] = synthid_result
            if synthid_result.get('likely_synthid', False) and not results['watermark_detected']:
                results['watermark_detected'] = True
                results['watermark_type'] = 'synthid_heuristic'
                results['watermark_confidence'] = synthid_result.get('confidence', 0.5)
            
            # Overall AI indicator from watermarks
            if results['watermark_detected']:
                results['recommendation'] = 'likely_ai_watermarked'
            else:
                # Check for signs that watermark was stripped
                strip_result = self._check_watermark_stripping(image)
                results['details']['stripping_indicators'] = strip_result
                if strip_result.get('likely_stripped', False):
                    results['recommendation'] = 'possibly_watermark_stripped'
                else:
                    results['recommendation'] = 'no_watermark_detected'
            
            return results
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'watermark_detected': False,
                'watermark_confidence': 0.0
            }
    
    def _detect_stable_signature(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect Stable Signature watermark using imwatermark.
        
        Stable Signature:
        - DWT decomposition (3 levels)
        - DCT coefficients in LH, HL, HH subbands
        - 48-100 bits embedded redundantly
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'payload': None
        }
        
        if not self.has_imwatermark or self.stable_decoder is None:
            result['note'] = 'imwatermark not available'
            return result
        
        try:
            # Convert to BGR for OpenCV compatibility
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Try to decode watermark
            payload = self.stable_decoder.decode(bgr_image, 'dwtDct')
            
            if payload is not None:
                # Check if payload looks valid (not all zeros/noise)
                payload_bytes = bytes(payload)
                non_zero = sum(1 for b in payload_bytes if b != 0)
                
                if non_zero > len(payload_bytes) * 0.1:  # At least 10% non-zero
                    result['detected'] = True
                    result['confidence'] = 0.85
                    result['payload'] = payload_bytes.hex()[:32]  # Truncate for display
                    
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _detect_dct_watermark(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect generic DCT spread-spectrum watermarks.
        
        Approach:
        1. Compute 2D DCT of image
        2. Analyze mid-frequency band for periodic patterns
        3. Check autocorrelation for watermark periodicity
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'periodicity_score': 0.0
        }
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
            
            # Compute 2D DCT
            dct_coef = dct(dct(gray.T, norm='ortho').T, norm='ortho')
            
            # Extract mid-frequency band (where watermarks typically hide)
            h, w = dct_coef.shape
            mid_band = dct_coef[h//8:h//2, w//8:w//2]
            
            # Compute autocorrelation via FFT (O(n log n) vs O(n²) for np.correlate)
            flat = mid_band.flatten()
            n = len(flat)
            fft_flat = np.fft.rfft(flat, n=2 * n)
            autocorr = np.fft.irfft(fft_flat * np.conj(fft_flat))[:n].real
            autocorr = autocorr / (autocorr[0] + 1e-10)  # Normalize
            
            # Look for periodic peaks (watermark signature)
            # Watermarks often have periodicity at 8, 16, or 32 pixels
            peaks = []
            for lag in [8, 16, 32, 64]:
                if lag < len(autocorr):
                    peak_val = autocorr[lag]
                    if peak_val > 0.1:  # Significant correlation
                        peaks.append((lag, peak_val))
            
            if len(peaks) >= 2:
                avg_peak = np.mean([p[1] for p in peaks])
                result['periodicity_score'] = float(avg_peak)
                
                if avg_peak > 0.3:
                    result['detected'] = True
                    result['confidence'] = min(avg_peak, 0.8)
                    result['detected_periods'] = [p[0] for p in peaks]
            
            # Also check for unnatural symmetry in mid-frequencies
            # (watermarked images often have more symmetric DCT)
            left_half = mid_band[:, :mid_band.shape[1]//2]
            right_half = np.fliplr(mid_band[:, mid_band.shape[1]//2:])
            
            min_w = min(left_half.shape[1], right_half.shape[1])
            if min_w > 0:
                symmetry = np.corrcoef(
                    left_half[:, :min_w].flatten(),
                    right_half[:, :min_w].flatten()
                )[0, 1]
                
                if not np.isnan(symmetry) and symmetry > 0.7:
                    result['dct_symmetry'] = float(symmetry)
                    if symmetry > 0.85:
                        result['detected'] = True
                        result['confidence'] = max(result['confidence'], symmetry * 0.8)
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _detect_synthid_heuristic(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Heuristic detection for Google SynthID.
        
        SynthID modifies token selection probabilities during generation,
        which can cause:
        - Lower spectral entropy in specific frequency bands
        - Non-random clustering in DCT mid-frequencies
        """
        result = {
            'likely_synthid': False,
            'confidence': 0.0,
            'spectral_entropy': 0.0,
            'entropy_anomaly': False
        }
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
            
            # Compute FFT
            f_transform = fft2(gray)
            f_shift = fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Analyze spectral entropy in different frequency bands
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2
            
            # Define frequency bands
            low_freq = magnitude[center_h-30:center_h+30, center_w-30:center_w+30]
            mid_freq = magnitude[center_h-100:center_h+100, center_w-100:center_w+100]
            
            # Compute entropy for each band
            def compute_entropy(arr):
                arr = arr.flatten()
                arr = arr / (np.sum(arr) + 1e-10)
                arr = arr[arr > 1e-10]
                return -np.sum(arr * np.log2(arr))
            
            low_entropy = compute_entropy(low_freq)
            mid_entropy = compute_entropy(mid_freq)
            
            result['low_freq_entropy'] = float(low_entropy)
            result['mid_freq_entropy'] = float(mid_entropy)
            result['spectral_entropy'] = float((low_entropy + mid_entropy) / 2)
            
            # SynthID images tend to have lower mid-frequency entropy
            # compared to natural distribution
            if mid_entropy < 10.0:  # Threshold based on natural image statistics
                result['entropy_anomaly'] = True
                result['likely_synthid'] = True
                result['confidence'] = _safe_prob((12.0 - mid_entropy) / 4.0)
            
            # Also check for non-random clustering in DCT
            dct_coef = dct(dct(gray.T, norm='ortho').T, norm='ortho')
            mid_dct = dct_coef[h//8:h//4, w//8:w//4]
            
            # Compute coefficient distribution uniformity
            hist, _ = np.histogram(mid_dct.flatten(), bins=50)
            hist = hist / (np.sum(hist) + 1e-10)
            dct_entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            result['dct_entropy'] = float(dct_entropy)
            
            # Very low DCT entropy suggests structured embedding
            if dct_entropy < 3.5:
                result['likely_synthid'] = True
                result['confidence'] = max(result['confidence'], 0.5)
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _check_watermark_stripping(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Check for signs that a watermark may have been stripped.
        
        Common stripping techniques leave artifacts:
        - JPEG recompression artifacts
        - Slight blurring in mid-frequencies
        - Unnatural noise patterns
        """
        result = {
            'likely_stripped': False,
            'confidence': 0.0,
            'indicators': []
        }
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
            
            # Check for excessive mid-frequency smoothing
            # (common when trying to remove DCT watermarks)
            dct_coef = dct(dct(gray.T, norm='ortho').T, norm='ortho')
            h, w = dct_coef.shape
            
            # Energy distribution across frequency bands
            low_energy = np.sum(np.abs(dct_coef[:h//8, :w//8]))
            mid_energy = np.sum(np.abs(dct_coef[h//8:h//2, w//8:w//2]))
            high_energy = np.sum(np.abs(dct_coef[h//2:, w//2:]))
            total = low_energy + mid_energy + high_energy + 1e-10
            
            mid_ratio = mid_energy / total
            
            # Naturally, mid-frequency should have ~30-50% of energy
            # If too low, might be stripped
            if mid_ratio < 0.15:
                result['indicators'].append('low_mid_frequency')
                result['confidence'] += 0.3
            
            # Check for unnatural smoothness gradients
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            lap_std = np.std(laplacian)
            
            if lap_std < 10:  # Very smooth
                result['indicators'].append('excessive_smoothness')
                result['confidence'] += 0.2
            
            if result['confidence'] > 0.4:
                result['likely_stripped'] = True
                
        except:
            pass
        
        return result


# Convenience function
def detect_watermark(image_path: str) -> Dict[str, Any]:
    """
    Detect watermarks in an image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        dict: Watermark detection results
    """
    detector = WatermarkDetector()
    return detector.detect(image_path)
