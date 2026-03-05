# -*- coding: utf-8 -*-
"""
PikSign Manipulation Forensics Module
Patch-level numerical analysis for detecting subtle image manipulations.

Methods:
- GLCM (Gray-Level Co-occurrence Matrix): texture inconsistency
- LBP (Local Binary Pattern): micro-texture mismatch at edit boundaries
- Wavelet high-frequency kurtosis: detects inpainting / content-aware fill
- Edge density variance: detects object removal / background replacement
- Benford's Law on DCT coefficients: detects any re-saving after editing
"""

import numpy as np
import cv2
from scipy.fftpack import dct
from scipy.stats import kurtosis as scipy_kurtosis
from typing import Dict, Any, List, Tuple


def _safe_prob(x: float) -> float:
    """Clamp value to valid probability range [0, 1]."""
    return float(min(max(x, 0.0), 1.0))


class ManipulationForensics:
    """
    Patch-level forensic analysis for detecting subtle image manipulations.

    All methods are pure numerical (numpy/scipy/OpenCV) -- no neural networks.
    Analysis is done per-patch (default 64x64) to catch localized edits
    that whole-image averages would miss.
    """

    def __init__(self, patch_size: int = 64, stride: int = 32):
        self.patch_size = patch_size
        self.stride = stride

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------

    def _to_uint8(self, image: np.ndarray) -> np.ndarray:
        if image.max() <= 1.0:
            return (image * 255).astype(np.uint8)
        return image.astype(np.uint8)

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        img = self._to_uint8(image)
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def _extract_patches(self, gray: np.ndarray) -> List[Tuple[np.ndarray, int, int]]:
        """Extract overlapping patches from a grayscale image."""
        h, w = gray.shape
        ps = self.patch_size
        patches = []
        for y in range(0, h - ps + 1, self.stride):
            for x in range(0, w - ps + 1, self.stride):
                patches.append((gray[y:y + ps, x:x + ps], y, x))
        return patches

    # ------------------------------------------------------------------
    # 1. GLCM (Gray-Level Co-occurrence Matrix)
    # ------------------------------------------------------------------

    def _compute_glcm_features(self, patch: np.ndarray, levels: int = 32) -> Dict[str, float]:
        """
        Compute GLCM texture features for a single patch.
        Quantize to `levels` gray levels for speed.
        """
        # Quantize
        q = (patch / 256.0 * levels).astype(np.int32)
        q = np.clip(q, 0, levels - 1)

        # Build co-occurrence matrix (horizontal, distance=1)
        glcm = np.zeros((levels, levels), dtype=np.float64)
        rows, cols = q.shape
        for dy, dx in [(0, 1), (1, 0)]:  # horizontal + vertical
            for r in range(rows - dy):
                for c in range(cols - dx):
                    i, j = q[r, c], q[r + dy, c + dx]
                    glcm[i, j] += 1
                    glcm[j, i] += 1

        # Normalize
        total = glcm.sum()
        if total == 0:
            return {'contrast': 0.0, 'energy': 0.0, 'homogeneity': 0.0, 'correlation': 0.0}
        glcm /= total

        # Features
        ii, jj = np.meshgrid(np.arange(levels), np.arange(levels), indexing='ij')
        diff = (ii - jj).astype(np.float64)

        contrast = float(np.sum(glcm * diff ** 2))
        energy = float(np.sum(glcm ** 2))
        homogeneity = float(np.sum(glcm / (1.0 + np.abs(diff))))

        # Correlation
        mu_i = np.sum(ii * glcm)
        mu_j = np.sum(jj * glcm)
        sigma_i = np.sqrt(np.sum(glcm * (ii - mu_i) ** 2))
        sigma_j = np.sqrt(np.sum(glcm * (jj - mu_j) ** 2))
        if sigma_i < 1e-10 or sigma_j < 1e-10:
            correlation = 0.0
        else:
            correlation = float(np.sum(glcm * (ii - mu_i) * (jj - mu_j)) / (sigma_i * sigma_j))

        return {
            'contrast': contrast,
            'energy': energy,
            'homogeneity': homogeneity,
            'correlation': correlation,
        }

    def analyze_glcm(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Patch-level GLCM analysis.
        Edited patches have different contrast/homogeneity than surrounding areas.
        """
        gray = self._to_gray(image)
        patches = self._extract_patches(gray)
        if len(patches) < 4:
            return {'status': 'too_small', 'glcm_anomaly': 0.0}

        features_list = []
        for patch, y, x in patches:
            feats = self._compute_glcm_features(patch)
            features_list.append(feats)

        # Collect per-feature arrays
        contrasts = np.array([f['contrast'] for f in features_list])
        energies = np.array([f['energy'] for f in features_list])
        homogeneities = np.array([f['homogeneity'] for f in features_list])

        # Outlier detection: patches whose features deviate > 2 std from mean
        def outlier_ratio(arr):
            if len(arr) < 4:
                return 0.0
            mu, sigma = np.mean(arr), np.std(arr)
            if sigma < 1e-10:
                return 0.0
            outliers = np.sum(np.abs(arr - mu) > 2.0 * sigma)
            return float(outliers / len(arr))

        contrast_outliers = outlier_ratio(contrasts)
        energy_outliers = outlier_ratio(energies)
        homogeneity_outliers = outlier_ratio(homogeneities)

        # Coefficient of variation (CV) -- high CV = inconsistent textures
        def cv(arr):
            mu = np.mean(arr)
            if abs(mu) < 1e-10:
                return 0.0
            return float(np.std(arr) / abs(mu))

        contrast_cv = cv(contrasts)
        energy_cv = cv(energies)

        # Combined anomaly score
        outlier_score = (contrast_outliers + energy_outliers + homogeneity_outliers) / 3.0
        cv_score = min((contrast_cv + energy_cv) / 2.0, 1.0)

        glcm_anomaly = _safe_prob(0.5 * outlier_score + 0.5 * cv_score)

        return {
            'status': 'success',
            'glcm_anomaly': glcm_anomaly,
            'contrast_cv': contrast_cv,
            'energy_cv': energy_cv,
            'outlier_ratio': outlier_score,
            'num_patches': len(patches),
        }

    # ------------------------------------------------------------------
    # 2. LBP (Local Binary Pattern)
    # ------------------------------------------------------------------

    def _compute_lbp(self, patch: np.ndarray) -> np.ndarray:
        """Compute basic LBP (8 neighbors, radius 1) for a patch."""
        h, w = patch.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        center = patch[1:-1, 1:-1].astype(np.int16)

        # 8 neighbors clockwise from top-left
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                   (1, 1), (1, 0), (1, -1), (0, -1)]
        for bit, (dy, dx) in enumerate(offsets):
            neighbor = patch[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx].astype(np.int16)
            lbp |= ((neighbor >= center).astype(np.uint8) << bit)

        return lbp

    def _lbp_histogram(self, lbp: np.ndarray) -> np.ndarray:
        """Compute normalized 256-bin LBP histogram."""
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        total = hist.sum()
        if total > 0:
            hist = hist.astype(np.float64) / total
        return hist

    def analyze_lbp(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Patch-level LBP analysis.
        Edited regions break local texture patterns -- their LBP histograms
        differ from the majority of patches.
        """
        gray = self._to_gray(image)
        patches = self._extract_patches(gray)
        if len(patches) < 4:
            return {'status': 'too_small', 'lbp_anomaly': 0.0}

        histograms = []
        for patch, y, x in patches:
            lbp = self._compute_lbp(patch)
            hist = self._lbp_histogram(lbp)
            histograms.append(hist)

        histograms = np.array(histograms)

        # Mean histogram
        mean_hist = np.mean(histograms, axis=0)

        # Chi-squared distance of each patch from the mean
        distances = []
        for hist in histograms:
            chi2 = np.sum((hist - mean_hist) ** 2 / (mean_hist + 1e-10))
            distances.append(chi2)
        distances = np.array(distances)

        # Outlier detection
        mu, sigma = np.mean(distances), np.std(distances)
        if sigma < 1e-10:
            outlier_ratio = 0.0
        else:
            outlier_ratio = float(np.sum(distances > mu + 2.0 * sigma) / len(distances))

        # Variance of distances -- high variance = inconsistent textures
        dist_cv = float(np.std(distances) / (mu + 1e-10))

        lbp_anomaly = _safe_prob(0.5 * outlier_ratio + 0.5 * min(dist_cv, 1.0))

        return {
            'status': 'success',
            'lbp_anomaly': lbp_anomaly,
            'outlier_ratio': outlier_ratio,
            'distance_cv': dist_cv,
            'num_patches': len(patches),
        }

    # ------------------------------------------------------------------
    # 3. Wavelet high-frequency kurtosis
    # ------------------------------------------------------------------

    def _haar_decompose(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Single-level Haar wavelet decomposition (no pywt dependency)."""
        h, w = image.shape
        # Make even
        h2 = h - h % 2
        w2 = w - w % 2
        img = image[:h2, :w2].astype(np.float64)

        # Row-wise
        lo = (img[:, 0::2] + img[:, 1::2]) / 2.0
        hi = (img[:, 0::2] - img[:, 1::2]) / 2.0

        # Col-wise on lo
        LL = (lo[0::2, :] + lo[1::2, :]) / 2.0
        LH = (lo[0::2, :] - lo[1::2, :]) / 2.0

        # Col-wise on hi
        HL = (hi[0::2, :] + hi[1::2, :]) / 2.0
        HH = (hi[0::2, :] - hi[1::2, :]) / 2.0

        return LL, LH, HL, HH

    def analyze_wavelet(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Patch-level wavelet kurtosis analysis.
        Inpainting/clone-stamp/content-aware fill smooth out high-freq details,
        causing the HH band kurtosis to drop in edited regions.
        """
        gray = self._to_gray(image)
        patches = self._extract_patches(gray)
        if len(patches) < 4:
            return {'status': 'too_small', 'wavelet_anomaly': 0.0}

        kurtosis_values = []
        for patch, y, x in patches:
            _, LH, HL, HH = self._haar_decompose(patch)
            # Combine all high-frequency sub-bands
            hf = np.concatenate([LH.flatten(), HL.flatten(), HH.flatten()])
            if len(hf) > 10:
                k = scipy_kurtosis(hf, fisher=True)
                if np.isnan(k) or np.isinf(k):
                    k = 0.0
                kurtosis_values.append(k)

        if len(kurtosis_values) < 4:
            return {'status': 'insufficient_patches', 'wavelet_anomaly': 0.0}

        kurtosis_values = np.array(kurtosis_values)

        # Real unedited images: heavy-tailed (high kurtosis) in HF bands
        # Edited patches: lower kurtosis (smoother HF)
        mu, sigma = np.mean(kurtosis_values), np.std(kurtosis_values)

        # Outlier ratio: patches with unusually LOW kurtosis
        if sigma < 1e-10:
            outlier_ratio = 0.0
        else:
            outlier_ratio = float(np.sum(kurtosis_values < mu - 2.0 * sigma) / len(kurtosis_values))

        # Coefficient of variation
        cv = float(sigma / (abs(mu) + 1e-10))

        wavelet_anomaly = _safe_prob(0.5 * outlier_ratio + 0.5 * min(cv / 2.0, 1.0))

        return {
            'status': 'success',
            'wavelet_anomaly': wavelet_anomaly,
            'mean_kurtosis': float(mu),
            'kurtosis_cv': cv,
            'low_kurtosis_outliers': outlier_ratio,
            'num_patches': len(kurtosis_values),
        }

    # ------------------------------------------------------------------
    # 4. Edge density variance
    # ------------------------------------------------------------------

    def analyze_edge_density(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Patch-level edge density analysis.
        Edited regions (object removal, background replacement) have
        unnaturally smooth or sharp edges compared to the rest.
        """
        gray = self._to_gray(image)
        patches = self._extract_patches(gray)
        if len(patches) < 4:
            return {'status': 'too_small', 'edge_anomaly': 0.0}

        densities = []
        for patch, y, x in patches:
            edges = cv2.Canny(patch, 50, 150)
            density = float(np.sum(edges > 0)) / edges.size
            densities.append(density)

        densities = np.array(densities)

        mu, sigma = np.mean(densities), np.std(densities)

        # Outlier detection: patches with very different edge density
        if sigma < 1e-10:
            outlier_ratio = 0.0
        else:
            outlier_ratio = float(np.sum(np.abs(densities - mu) > 2.0 * sigma) / len(densities))

        # CV of edge densities
        cv = float(sigma / (mu + 1e-10))

        edge_anomaly = _safe_prob(0.5 * outlier_ratio + 0.5 * min(cv, 1.0))

        return {
            'status': 'success',
            'edge_anomaly': edge_anomaly,
            'mean_density': float(mu),
            'density_cv': cv,
            'outlier_ratio': outlier_ratio,
            'num_patches': len(patches),
        }

    # ------------------------------------------------------------------
    # 5. Benford's Law on DCT coefficients
    # ------------------------------------------------------------------

    def analyze_benford(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Benford's Law analysis on DCT coefficients.
        Natural images follow Benford's distribution in their DCT first digits.
        Any re-saving or editing shifts this distribution away from the ideal.
        """
        gray = self._to_gray(image).astype(np.float64)

        # Compute block-wise DCT (8x8 blocks like JPEG)
        h, w = gray.shape
        h8 = h - h % 8
        w8 = w - w % 8
        gray = gray[:h8, :w8]

        all_coeffs = []
        for r in range(0, h8, 8):
            for c in range(0, w8, 8):
                block = gray[r:r + 8, c:c + 8]
                d = dct(dct(block.T, norm='ortho').T, norm='ortho')
                # Skip DC coefficient (d[0,0])
                ac = d.flatten()[1:]
                all_coeffs.extend(ac)

        all_coeffs = np.abs(np.array(all_coeffs))
        # Keep only non-zero
        all_coeffs = all_coeffs[all_coeffs > 0.5]

        if len(all_coeffs) < 100:
            return {'status': 'insufficient_data', 'benford_anomaly': 0.0}

        # Extract first significant digit
        first_digits = np.floor(all_coeffs / (10 ** np.floor(np.log10(all_coeffs + 1e-30)))).astype(int)
        first_digits = first_digits[(first_digits >= 1) & (first_digits <= 9)]

        if len(first_digits) < 100:
            return {'status': 'insufficient_digits', 'benford_anomaly': 0.0}

        # Observed distribution
        observed = np.zeros(9)
        for d in range(1, 10):
            observed[d - 1] = np.sum(first_digits == d)
        observed /= (observed.sum() + 1e-10)

        # Benford's expected distribution
        expected = np.array([np.log10(1 + 1.0 / d) for d in range(1, 10)])

        # Chi-squared divergence
        chi2 = np.sum((observed - expected) ** 2 / (expected + 1e-10))

        # Correlation with Benford
        corr = np.corrcoef(observed, expected)[0, 1]
        if np.isnan(corr):
            corr = 0.0

        # Higher chi2 = more deviation from Benford = more likely edited
        # Typical values: unedited < 0.01, edited > 0.05, heavy edit > 0.2
        if chi2 < 0.01:
            benford_score = 0.1   # Very close to Benford -- likely original
        elif chi2 < 0.05:
            benford_score = 0.3
        elif chi2 < 0.15:
            benford_score = 0.6
        else:
            benford_score = 0.85  # Strong deviation -- likely edited/AI

        return {
            'status': 'success',
            'benford_anomaly': _safe_prob(benford_score),
            'chi_squared': float(chi2),
            'benford_correlation': float(corr),
            'observed_distribution': observed.tolist(),
            'expected_distribution': expected.tolist(),
        }

    # ------------------------------------------------------------------
    # Combined analysis
    # ------------------------------------------------------------------

    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run all 5 manipulation forensic methods and combine results.

        Returns a dict with individual scores and a combined manipulation score.
        """
        glcm = self.analyze_glcm(image)
        lbp = self.analyze_lbp(image)
        wavelet = self.analyze_wavelet(image)
        edge = self.analyze_edge_density(image)
        benford = self.analyze_benford(image)

        glcm_score = glcm.get('glcm_anomaly', 0.0)
        lbp_score = lbp.get('lbp_anomaly', 0.0)
        wavelet_score = wavelet.get('wavelet_anomaly', 0.0)
        edge_score = edge.get('edge_anomaly', 0.0)
        benford_score = benford.get('benford_anomaly', 0.0)

        # Weighted combination
        # Benford + LBP are strongest for subtle edits
        combined = _safe_prob(
            benford_score * 0.25 +
            lbp_score * 0.25 +
            glcm_score * 0.20 +
            wavelet_score * 0.15 +
            edge_score * 0.15
        )

        # Max-pooling boost: if any single method is very confident, trust it
        max_score = max(glcm_score, lbp_score, wavelet_score, edge_score, benford_score)
        if max_score > 0.7:
            combined = max(combined, max_score * 0.85)

        return {
            'status': 'success',
            'glcm': glcm,
            'lbp': lbp,
            'wavelet': wavelet,
            'edge_density': edge,
            'benford': benford,
            'manipulation_score': combined,
            'individual_scores': {
                'glcm': glcm_score,
                'lbp': lbp_score,
                'wavelet': wavelet_score,
                'edge_density': edge_score,
                'benford': benford_score,
            }
        }
