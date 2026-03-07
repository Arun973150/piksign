# -*- coding: utf-8 -*-
"""
PikSign Forensics Module
Embedding and frequency domain forensic analysis for manipulation detection.

PURPOSE:
This is a PURE AUXILIARY module that provides forensic analysis scores.
It should NEVER be used to override classifier decisions.

The module outputs individual component scores without any decision logic.
Thresholds and decisions should be applied only in the CLI or reporting layer.
"""

import numpy as np
import cv2
from PIL import Image
from scipy.fftpack import dct, fft2, fftshift
from typing import Dict, Any, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


def _safe_prob(x: float) -> float:
    """Clamp value to valid probability range [0, 1]."""
    return float(min(max(x, 0.0), 1.0))


class FrequencyForensics:
    """
    Frequency domain forensic analysis.
    
    Detects manipulation artifacts in:
    - FFT spectrum
    - DCT coefficients
    - Wavelet domain
    """
    
    def __init__(self):
        pass
    
    def compute_fft_spectrum(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Compute FFT spectrum analysis.
        
        Args:
            image: RGB image array
            
        Returns:
            dict: FFT analysis results
        """
        gray = cv2.cvtColor((image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8), 
                           cv2.COLOR_RGB2GRAY)
        
        # Compute 2D FFT
        f_transform = fft2(gray.astype(np.float64))
        f_shift = fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Analyze spectrum for anomalies
        center_h, center_w = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        
        # Energy in different frequency bands
        low_freq = magnitude_spectrum[center_h-20:center_h+20, center_w-20:center_w+20]
        mid_freq_h = magnitude_spectrum[center_h-60:center_h+60, center_w-60:center_w+60]
        
        low_energy = np.mean(low_freq)
        mid_energy = np.mean(mid_freq_h) - low_energy
        high_energy = np.mean(magnitude_spectrum) - np.mean(mid_freq_h)
        
        # Symmetry analysis (manipulated images often have asymmetric spectra)
        upper_half = magnitude_spectrum[:center_h, :]
        lower_half = np.flipud(magnitude_spectrum[center_h:, :])
        
        min_h = min(upper_half.shape[0], lower_half.shape[0])
        symmetry_score = np.corrcoef(
            upper_half[:min_h, :].flatten(),
            lower_half[:min_h, :].flatten()
        )[0, 1]
        
        return {
            'low_freq_energy': float(low_energy),
            'mid_freq_energy': float(mid_energy),
            'high_freq_energy': float(high_energy),
            'symmetry_score': float(symmetry_score) if not np.isnan(symmetry_score) else 0.0,
            'spectral_anomaly': _safe_prob(1.0 - symmetry_score) if not np.isnan(symmetry_score) else 0.5
        }

    def detect_jpeg_grid(self, image: np.ndarray) -> float:
        """
        Detect 8x8 JPEG compression grid.
        AI images often LACK this grid (if PNG/generated) or have MISALIGNED grid.
        Range: 0.0 (Strong Grid - Real) to 1.0 (No Grid - AI/PNG)
        """
        try:
            # High-pass filter to extract block artifacts
            gray = cv2.cvtColor((image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8), 
                               cv2.COLOR_RGB2GRAY)
            
            # Simple grid detection by checking periodicity in difference
            # Sum rows/cols to find peaks at multiples of 8
            
            # Differencing
            h_diff = np.abs(gray[:-1, :] - gray[1:, :])
            v_diff = np.abs(gray[:, :-1] - gray[:, 1:])
            
            h_sum = np.sum(h_diff, axis=1)
            v_sum = np.sum(v_diff, axis=0)
            
            # Check periodicity at 8
            def check_periodicity(signal):
                if len(signal) < 16: return 0.0
                peaks = signal[7::8] # 8, 16, 24... (index 7, 15, 23...)
                valleys = signal[3::8] # 4, 12, 20...
                
                mean_peak = np.mean(peaks) if len(peaks) > 0 else 0
                mean_valley = np.mean(valleys) if len(valleys) > 0 else 0
                
                if mean_valley == 0: return 0.0
                return max(0, (mean_peak - mean_valley) / mean_valley)
            
            h_period = check_periodicity(h_sum)
            v_period = check_periodicity(v_sum)
            
            grid_strength = (h_period + v_period) / 2.0
            
            # If grid is strong, it's likely a real JPEG (or correctly saved AI)
            # If grid is weak/missing, it matches AI generation (often PNG source)
            
            # Grid strength > 0.05 is usually real JPEG
            if grid_strength > 0.05:
                return 0.1 # Strong grid -> Real
            elif grid_strength > 0.02:
                return 0.4 # Weak grid -> Uncertain
            else:
                return 0.8 # No grid -> AI (or uncompressed)
                
        except:
            return 0.5
    
    def compute_dct_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Compute DCT-based forensic analysis.
        JPEG compression artifacts and manipulation detection.
        """
        gray = cv2.cvtColor((image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8), 
                           cv2.COLOR_RGB2GRAY)
        
        # Compute DCT
        dct_coef = dct(dct(gray.T.astype(np.float64), norm='ortho').T, norm='ortho')
        
        # Analyze DCT coefficient distribution
        abs_coef = np.abs(dct_coef)
        
        # Check for double compression artifacts
        ac_coefficients = abs_coef[1:50, 1:50].flatten()
        
        # Histogram of AC coefficients
        hist, bins = np.histogram(ac_coefficients, bins=50)
        hist_normalized = hist / (np.sum(hist) + 1e-10)
        
        # Entropy of DCT coefficient distribution
        entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10))
        
        # Double JPEG detection
        block_artifacts = self._detect_block_artifacts(dct_coef)
        
        return {
            'dct_entropy': float(entropy),
            'block_artifact_score': float(block_artifacts),
            'coefficient_sparsity': float(np.sum(abs_coef < 1) / abs_coef.size),
            'manipulation_score': _safe_prob(block_artifacts * 0.5 + (1 - entropy/10) * 0.5)
        }
    
    def _detect_block_artifacts(self, dct_coef: np.ndarray) -> float:
        """Detect 8x8 block artifacts from JPEG compression."""
        h, w = dct_coef.shape
        
        # Check periodicity at block boundaries
        block_size = 8
        boundary_energy = 0
        total_energy = 0
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = dct_coef[i:i+block_size, j:j+block_size]
                boundary_energy += np.sum(np.abs(block[0, :])) + np.sum(np.abs(block[:, 0]))
                total_energy += np.sum(np.abs(block))
        
        if total_energy > 0:
            return _safe_prob(float(boundary_energy / total_energy))
        return 0.0
    
    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform complete frequency forensic analysis.
        """
        fft_results = self.compute_fft_spectrum(image)
        dct_results = self.compute_dct_analysis(image)
        
        # Combined manipulation score - Tuned for higher sensitivity
        # Spectral anomalies are very strong indicators, so we boost them
        combined_score = _safe_prob(
            fft_results['spectral_anomaly'] * 0.6 +
            dct_results['manipulation_score'] * 0.4
        )
        
        # Boost if spectral anomaly is high
        if fft_results['spectral_anomaly'] > 0.6:
            combined_score = max(combined_score, fft_results['spectral_anomaly'])

        
        return {
            'fft_analysis': fft_results,
            'dct_analysis': dct_results,
            'combined_manipulation_score': combined_score
        }


class EmbeddingForensics:
    """
    Embedding-based forensic analysis.
    
    V2: Now includes Gram Matrix Analysis (VGG19) and Noise Residual Analysis
    in addition to ResNet50 patch embeddings.
    
    Gram Matrix: Captures texture/style statistics - AI images have unnatural
    texture correlations compared to real photos.
    
    Noise Residual: Analyzes high-frequency noise patterns - AI images have
    different noise characteristics than camera sensors.
    """
    
    def __init__(self, device=None):
        import torch
        from torchvision import models
        import torchvision.transforms as transforms
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load ResNet for patch analysis (backward compatibility)
        try:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except:
            self.model = models.resnet50(pretrained=True)
        
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # V2: Load VGG19 for Gram matrix analysis
        self.vgg_model = None
        try:
            try:
                vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            except:
                vgg = models.vgg19(pretrained=True)
            
            # Extract features from multiple layers
            self.vgg_layers = ['0', '5', '10', '19', '28']  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
            self.vgg_model = vgg.features.to(self.device)
            self.vgg_model.eval()
        except Exception as e:
            print(f"   [\!] VGG19 for Gram Matrix: Not available ({e})")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_patch_embeddings(self, image: np.ndarray,
                                  patch_size: int = 64,
                                  stride: int = 64) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Extract embeddings from image patches."""
        import torch
        
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        h, w = image.shape[:2]
        embeddings = []
        positions = []
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[y:y+patch_size, x:x+patch_size]
                
                try:
                    patch_tensor = self.transform(patch).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        embedding = self.model(patch_tensor)
                        embedding = embedding.flatten().cpu().numpy()
                    
                    embeddings.append(embedding)
                    positions.append((y, x))
                except:
                    pass
        
        return embeddings, positions
    
    def compute_patch_inconsistency(self, embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """
        Compute patch-level inconsistency score.
        
        Manipulated regions often have different embedding characteristics.
        """
        if len(embeddings) < 4:
            return {'inconsistency_score': 0.0, 'outlier_patches': []}
        
        embeddings_array = np.array(embeddings)
        
        # Compute mean embedding
        mean_embedding = np.mean(embeddings_array, axis=0)
        
        # Compute distances from mean
        distances = np.array([
            np.linalg.norm(emb - mean_embedding) 
            for emb in embeddings_array
        ])
        
        # Identify outliers (potential manipulation regions)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        outlier_indices = np.where(distances > mean_dist + 2 * std_dist)[0]
        
        # Inconsistency score based on variance
        inconsistency = np.std(distances) / (mean_dist + 1e-10)
        
        return {
            'inconsistency_score': _safe_prob(inconsistency),
            'mean_distance': float(mean_dist),
            'outlier_count': len(outlier_indices),
            'outlier_indices': outlier_indices.tolist()
        }
    
    def compute_gram_matrix(self, image: np.ndarray) -> Dict[str, Any]:
        """
        V2: Compute Gram matrix analysis using VGG19 features.
        
        Gram matrices capture texture/style statistics. AI-generated images
        have unnatural texture correlations between feature channels.
        
        Returns:
            dict: Gram matrix anomaly scores
        """
        import torch
        
        if self.vgg_model is None:
            return {'status': 'unavailable', 'gram_anomaly': 0.0}
        
        try:
            # Prepare image
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
            
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features from VGG layers
            features = {}
            x = img_tensor
            
            with torch.no_grad():
                for name, layer in self.vgg_model._modules.items():
                    x = layer(x)
                    if name in self.vgg_layers:
                        features[name] = x
            
            # Compute Gram matrices
            gram_matrices = {}
            for name, feat in features.items():
                b, c, h, w = feat.size()
                feat_flat = feat.view(b, c, h * w)
                gram = torch.bmm(feat_flat, feat_flat.transpose(1, 2))
                gram = gram / (c * h * w)
                gram_matrices[name] = gram.cpu().numpy()
            
            # Analyze Gram matrix statistics
            # AI images tend to have:
            # 1. Lower variance in Gram matrix entries (too uniform textures)
            # 2. Lower trace-to-frobenius ratio (less texture diversity)
            
            anomaly_scores = []
            for name, gram in gram_matrices.items():
                gram_flat = gram.flatten()
                
                # Variance analysis
                variance = np.var(gram_flat)
                # Real photos: high variance (0.001-0.1)
                # AI images: lower variance (0.0001-0.01)
                if variance < 0.0001:
                    var_score = 0.9  # Suspiciously low - likely AI
                elif variance < 0.001:
                    var_score = 0.6
                else:
                    var_score = 0.2
                
                # Sparsity analysis
                sparsity = np.sum(np.abs(gram_flat) < 0.0001) / len(gram_flat)
                # AI images often have more sparse Gram matrices
                sparse_score = min(sparsity * 2, 1.0)
                
                layer_score = 0.6 * var_score + 0.4 * sparse_score
                anomaly_scores.append(layer_score)
            
            # Average across layers
            gram_anomaly = np.mean(anomaly_scores) if anomaly_scores else 0.0
            
            return {
                'status': 'success',
                'gram_anomaly': _safe_prob(gram_anomaly),
                'layer_scores': {f'layer_{i}': float(s) for i, s in enumerate(anomaly_scores)}
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'gram_anomaly': 0.0}
    
    def compute_noise_residual(self, image: np.ndarray) -> Dict[str, Any]:
        """
        V2: Analyze noise residual patterns.
        
        Real camera photos have consistent Gaussian sensor noise.
        AI images have different noise characteristics:
        - Too smooth (denoised)
        - Non-Gaussian patterns
        - Inconsistent noise across regions
        
        Returns:
            dict: Noise residual anomaly scores
        """
        try:
            if image.max() <= 1:
                image = (image * 255).astype(np.float64)
            else:
                image = image.astype(np.float64)
            
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
            
            # Extract noise using high-pass filter
            denoised = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = gray - denoised
            
            # Analyze noise statistics
            noise_std = np.std(noise)
            noise_mean = np.mean(noise)
            
            # 1. Noise level check
            # Real photos: std 2-15 (depending on ISO)
            # AI images: often < 2 or unnaturally high
            if noise_std < 1.0:
                level_score = 0.8  # Too smooth - likely AI
            elif noise_std < 2.0:
                level_score = 0.5
            elif noise_std > 20.0:
                level_score = 0.6  # Too noisy - possibly enhanced AI
            else:
                level_score = 0.2  # Natural noise level
            
            # 2. Gaussianity check (compare with ideal Gaussian)
            hist, bins = np.histogram(noise, bins=50, range=(-30, 30))
            hist = hist.astype(np.float64) / (np.sum(hist) + 1e-10)
            
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ideal_gaussian = np.exp(-0.5 * ((bin_centers - noise_mean) / (noise_std + 1e-6)) ** 2)
            ideal_gaussian /= (np.sum(ideal_gaussian) + 1e-10)
            
            # Correlation with Gaussian
            corr = np.corrcoef(hist, ideal_gaussian)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            
            # Real photos: high correlation (>0.9)
            # AI images: lower correlation (<0.8)
            gaussian_score = 1.0 - max(corr, 0.0)  # Invert so higher = more suspicious
            
            # 3. Spatial consistency check
            # Split image into quadrants and check noise consistency
            h, w = gray.shape
            quadrants = [
                noise[:h//2, :w//2],
                noise[:h//2, w//2:],
                noise[h//2:, :w//2],
                noise[h//2:, w//2:]
            ]
            
            quad_stds = [np.std(q) for q in quadrants]
            quad_variance = np.std(quad_stds) / (np.mean(quad_stds) + 1e-10)
            
            # Real photos: consistent noise (low variance)
            # AI images: inconsistent (high variance)
            consistency_score = min(quad_variance * 2, 1.0)
            
            # Combined noise anomaly
            noise_anomaly = (
                0.35 * level_score +
                0.35 * gaussian_score +
                0.30 * consistency_score
            )
            
            return {
                'status': 'success',
                'noise_anomaly': _safe_prob(noise_anomaly),
                'noise_std': float(noise_std),
                'gaussianity': float(corr),
                'spatial_consistency': float(1.0 - consistency_score)
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'noise_anomaly': 0.0}
    
    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform embedding-based forensic analysis.
        
        V2: Now includes Gram matrix and noise residual analysis
        for improved AI detection accuracy.
        """
        # Original patch embedding analysis
        embeddings, positions = self.extract_patch_embeddings(image)
        
        if not embeddings:
            patch_analysis = {
                'status': 'error',
                'error': 'Could not extract embeddings',
                'inconsistency_score': 0.0
            }
        else:
            inconsistency = self.compute_patch_inconsistency(embeddings)
            patch_analysis = {
                'status': 'success',
                'patch_count': len(embeddings),
                'inconsistency_analysis': inconsistency
            }
        
        # V2: Gram matrix analysis
        gram_analysis = self.compute_gram_matrix(image)
        
        # V2: Noise residual analysis
        noise_analysis = self.compute_noise_residual(image)
        
        # Combine all signals
        patch_score = patch_analysis.get('inconsistency_analysis', {}).get('inconsistency_score', 0.0)
        gram_score = gram_analysis.get('gram_anomaly', 0.0)
        noise_score = noise_analysis.get('noise_anomaly', 0.0)
        
        # Weighted combination
        # Gram matrix and noise are stronger indicators for AI detection
        manipulation_probability = _safe_prob(
            patch_score * 0.25 +   # Patch inconsistency
            gram_score * 0.40 +    # Gram matrix (texture)
            noise_score * 0.35     # Noise residual
        )
        
        return {
            'status': 'success',
            'patch_analysis': patch_analysis,
            'gram_analysis': gram_analysis,
            'noise_analysis': noise_analysis,
            'manipulation_probability': manipulation_probability
        }


class ColorChannelForensics:
    """
    Color channel forensic analysis.
    
    Detects manipulation through color space inconsistencies.
    """
    
    def analyze_channel_correlation(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze correlation between color channels."""
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        r_channel = image[:, :, 0].flatten().astype(np.float64)
        g_channel = image[:, :, 1].flatten().astype(np.float64)
        b_channel = image[:, :, 2].flatten().astype(np.float64)
        
        # Compute correlations
        rg_corr = np.corrcoef(r_channel, g_channel)[0, 1]
        rb_corr = np.corrcoef(r_channel, b_channel)[0, 1]
        gb_corr = np.corrcoef(g_channel, b_channel)[0, 1]
        
        # Natural images typically have high channel correlation
        avg_corr = (rg_corr + rb_corr + gb_corr) / 3
        
        return {
            'rg_correlation': float(rg_corr) if not np.isnan(rg_corr) else 0.0,
            'rb_correlation': float(rb_corr) if not np.isnan(rb_corr) else 0.0,
            'gb_correlation': float(gb_corr) if not np.isnan(gb_corr) else 0.0,
            'average_correlation': float(avg_corr) if not np.isnan(avg_corr) else 0.0,
            'anomaly_score': _safe_prob(1 - avg_corr) if not np.isnan(avg_corr) else 0.5
        }
    
    def analyze_noise_patterns(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze noise patterns across color channels."""
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        noise_levels = []
        
        for c in range(3):
            channel = image[:, :, c].astype(np.float64)
            
            # High-pass filter to extract noise
            blurred = cv2.GaussianBlur(channel, (5, 5), 0)
            noise = channel - blurred
            noise_level = np.std(noise)
            noise_levels.append(noise_level)
        
        # Check for noise inconsistency
        noise_variance = np.var(noise_levels)
        
        # Inconsistency score (higher variance = higher likelihood of manipulation)
        # Normalize: typical variance for spliced images is > 5.0
        inconsistency = float(min(noise_variance / 10.0, 1.0))
        
        return {
            'noise_levels': [float(n) for n in noise_levels],
            'noise_variance': float(noise_variance),
            'noise_inconsistency': _safe_prob(inconsistency)
        }



class ProtectionForensics:
    """
    Detects artifacts specifically from PikSign's protection module.
    
    Checks for:
    - Spectral Watermarks (FFT peaks)
    - Multi-band Frequency Watermarks (DCT energy)
    - Stable Signatures (DWT-DCT)
    """
    
    def __init__(self):
        self.spectral = None
        self.multiband = None
        self.stable = None
        self.enabled = False
        
        try:
            # Lazy import to avoid circular dependencies
            from piksign.protection.spectral_watermark import SpectralWatermark
            from piksign.protection.multiband_watermark import MultiBandFrequencyWatermark
            from piksign.protection.stable_signature import StableSignatureWatermark
            
            self.spectral = SpectralWatermark()
            self.multiband = MultiBandFrequencyWatermark()
            self.stable = StableSignatureWatermark()
            self.enabled = True
        except ImportError:
            # If protection module is missing, just disable this check
            print("   [\!] Protection modules not found - skipping protection artifact checks")
            self.enabled = False
            
    def analyze(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image for PikSign protection artifacts.
        
        Args:
            image_array: Float32 image (0-1) [H, W, 3]
            
        Returns:
            Dict with detection scores
        """
        results = {
            'spectral_score': 0.0,
            'multiband_score': 0.0,
            'stable_signature_score': 0.0,
            'protection_detected': False
        }
        
        if not self.enabled:
            return results
            
        try:
            # 1. Spectral Watermark
            # detect() returns accumulation score
            results['spectral_score'] = float(self.spectral.detect(image_array))
            
            # 2. Multi-band Watermark
            # detect() returns average correlation
            results['multiband_score'] = float(self.multiband.detect(image_array))
            
            # 3. Stable Signature
            # extract() returns (message, confidence)
            _, conf = self.stable.extract(image_array)
            results['stable_signature_score'] = float(conf)
            
            # Global protection decision
            # If any watermark is strongly detected, we flag it
            if (results['spectral_score'] > 0.4 or 
                results['multiband_score'] > 0.4 or 
                results['stable_signature_score'] > 0.8):
                results['protection_detected'] = True
                
            results['max_protection_score'] = max(
                results['spectral_score'],
                results['multiband_score'],
                results['stable_signature_score']
            )
            
        except Exception as e:
            print(f"Error in protection forensics: {e}")
            
        return results


class ForensicsAnalyzer:
    """
    Main Forensics Analyzer combining all forensic techniques.

    This is a PURE AUXILIARY module. It provides component scores only.
    No decision logic or thresholds are applied here.

    Uses fixed fusion weights:
    - Frequency analysis: 35%
    - Color analysis: 20%
    - Embedding analysis: 20%
    - Manipulation forensics: 25%
    """

    def __init__(self, device=None):
        print("Initializing Forensics Analyzer...")

        self.frequency_forensics = FrequencyForensics()
        print("   Frequency analysis: Ready")

        self.embedding_forensics = EmbeddingForensics(device)
        print("   Embedding analysis: Ready")

        self.color_forensics = ColorChannelForensics()
        print("   Color analysis: Ready")

        self.protection_forensics = ProtectionForensics()
        print("   Protection artifact analysis: Ready")

        # Patch-level manipulation forensics (GLCM, LBP, Wavelet, Edge, Benford)
        from piksign.detection.manipulation_forensics import ManipulationForensics
        self.manipulation_forensics = ManipulationForensics()
        print("   Manipulation forensics: Ready (GLCM/LBP/Wavelet/Edge/Benford)")
    
    
    def analyze(self, image_path: str, ai_confidence: float = 0.0) -> Dict[str, Any]:
        """
        Perform complete forensic analysis.
        
        Args:
            image_path: Path to image file
            ai_confidence: Probability from AI detection (0.0 - 1.0).
                          Used to switch between Manipulation Mode and AI Generation Mode.
            
        Returns:
            dict: Forensic analysis results.
        """
        try:
            image = np.array(Image.open(image_path).convert('RGB'))
            image_float = image.astype(np.float32) / 255.0
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
        
        results = {
            'status': 'success',
            'frequency_analysis': {},
            'embedding_analysis': {},
            'color_analysis': {},
            'manipulation_analysis': {},
            'protection_analysis': {}
        }

        # Frequency domain analysis
        results['frequency_analysis'] = self.frequency_forensics.analyze(image_float)

        # AI-Specific: Check for JPEG Grid
        grid_anomaly = self.frequency_forensics.detect_jpeg_grid(image_float)
        results['frequency_analysis']['grid_anomaly'] = grid_anomaly

        # Embedding analysis
        results['embedding_analysis'] = self.embedding_forensics.analyze(image)

        # Color channel analysis
        results['color_analysis'] = {
            'correlation': self.color_forensics.analyze_channel_correlation(image),
            'noise': self.color_forensics.analyze_noise_patterns(image)
        }

        # Patch-level manipulation forensics (GLCM, LBP, Wavelet, Edge, Benford)
        results['manipulation_analysis'] = self.manipulation_forensics.analyze(image_float)

        # Protection artifact analysis
        results['protection_analysis'] = self.protection_forensics.analyze(image_float)

        # Extract individual component scores
        freq_score = _safe_prob(results['frequency_analysis'].get('combined_manipulation_score', 0))
        embed_score = _safe_prob(results['embedding_analysis'].get('manipulation_probability', 0))
        manip_score = _safe_prob(results['manipulation_analysis'].get('manipulation_score', 0))

        # Enhanced Color Score
        corr_anomaly = results['color_analysis']['correlation'].get('anomaly_score', 0)
        noise_inconsistency = results['color_analysis']['noise'].get('noise_inconsistency', 0)
        color_score = _safe_prob(max(corr_anomaly, noise_inconsistency))

        # MODE SELECTION LOGIC
        # ---------------------------------------------------------

        if ai_confidence > 0.60:
            # MODE 2: AI GENERATION INDICATORS
            # Prioritize: Grid Anomaly, Spectral Anomalies, Benford, Color Correlation
            w_freq = 0.35
            w_color = 0.20
            w_embed = 0.15
            w_manip = 0.30   # Benford + LBP strong for AI

            # Boost frequency score with grid anomaly
            final_freq_score = (freq_score * 0.4 + grid_anomaly * 0.6)
            final_color_score = corr_anomaly

            combined = (
                final_freq_score * w_freq +
                final_color_score * w_color +
                embed_score * w_embed +
                manip_score * w_manip
            )

        else:
            # MODE 1: TRADITIONAL MANIPULATION
            # Prioritize: Patch-level forensics (GLCM, LBP, Wavelet, Edge)
            w_freq = 0.25
            w_color = 0.15
            w_embed = 0.20
            w_manip = 0.40   # Patch-level methods are king for subtle edits

            final_freq_score = freq_score
            final_color_score = color_score

            combined = (
                final_freq_score * w_freq +
                final_color_score * w_color +
                embed_score * w_embed +
                manip_score * w_manip
            )

            # Boost if any single indicator is very strong (max pooling)
            max_indicator = max(final_freq_score, final_color_score, embed_score, manip_score)
            if max_indicator > 0.7:
                combined = max(combined, max_indicator * 0.9)
        
        # Special Case: If protection is detected, we know it's "manipulated" by protection
        # But this is a "good" manipulation. 
        # The user asked to "correctly identify" artifacts.
        # We don't necessarily want to increase the "fake" score if it's just protected.
        # However, technically it IS manipulated. 
        # We will report the protection score separately. 
        # For the pure manipulation score, we leave it as is, but maybe we can suppress it 
        # if we know it's just protection? 
        # For now, let's just report it.
        
        # Start of component scores
        results['frequency_score'] = final_freq_score
        results['color_score'] = final_color_score
        results['embedding_score'] = embed_score
        results['manipulation_score'] = manip_score
        results['combined_forensic_score'] = _safe_prob(combined)
        
        # Keep legacy field for backward compatibility
        results['overall_manipulation_score'] = _safe_prob(combined)
        
        return results

