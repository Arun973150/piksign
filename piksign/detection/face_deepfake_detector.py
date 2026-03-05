# -*- coding: utf-8 -*-
"""
PikSign Face Deepfake Detector
Uses MediaPipe for face detection and heuristic analysis for manipulation artifacts.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any, List, Tuple
import logging

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

def _safe_prob(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))

class FaceDeepfakeDetector:
    """
    Face-specific Deepfake Detector.

    Uses OpenCV Haar cascade for face detection (reliable, no external model downloads)
    and analyzes face regions for manipulation artifacts:
       - Boundary inconsistencies (blending artifacts)
       - Noise mismatch (face vs background)
       - Texture smoothness (AI faces often overly smooth)
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enabled = False
        self.face_cascade = None

        try:
            # Use OpenCV Haar cascade - always available
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            if self.face_cascade.empty():
                # Try alternative paths
                alt_paths = [
                    '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                    '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
                ]
                for path in alt_paths:
                    self.face_cascade = cv2.CascadeClassifier(path)
                    if not self.face_cascade.empty():
                        break

            if not self.face_cascade.empty():
                self.enabled = True
                print("   [OK] FaceDeepfakeDetector: Ready (OpenCV Haar)")
            else:
                print("   [--] FaceDeepfakeDetector: Haar cascade not found")
        except Exception as e:
            print(f"   [--] FaceDeepfakeDetector: Init failed ({e})")
            self.enabled = False
            
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Detect faces and analyze them for manipulation.
        """
        if not self.enabled:
            return {'status': 'error', 'error': 'Face detector not available', 'faces_detected': 0}

        try:
            # Load image using OpenCV
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    return {'status': 'error', 'error': 'Could not read image', 'faces_detected': 0}
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_path, np.ndarray):
                image_rgb = image_path if image_path.shape[2] == 3 else cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
            else:
                 return {'status': 'error', 'error': 'Invalid image format', 'faces_detected': 0}

            h, w, _ = image_rgb.shape

            face_data = []
            max_face_score = 0.0

            # Use Haar cascade for face detection
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))

            for (x, y, bw, bh) in faces:
                # Clamp bounding box
                x = max(0, x)
                y = max(0, y)
                bw = min(bw, w - x)
                bh = min(bh, h - y)

                if bw < 20 or bh < 20: continue  # Skip tiny faces

                face_crop = image_rgb[y:y+bh, x:x+bw]

                # Analyze face artifacts
                analysis = self._analyze_face_artifacts(face_crop, image_rgb, (x, y, bw, bh))
                score = analysis['manipulation_score']

                face_data.append({
                    'bbox': [x, y, bw, bh],
                    'score': score,
                    'analysis': analysis
                })

                max_face_score = max(max_face_score, score)

            return {
                'status': 'success',
                'faces_detected': len(face_data),
                'faces': face_data,
                'final_face_score': float(max_face_score)
            }

        except Exception as e:
            return {'status': 'error', 'error': str(e), 'faces_detected': 0}

    def _analyze_face_artifacts(self, face_crop: np.ndarray, full_image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Heuristic analysis of face crop for manipulation.
        """
        # 1. Texture Smoothness Analysis
        # AI faces (especially GANs/filters) are often smoother than real skin
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        
        # Laplacian variance (sharpness/texture measure)
        sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        # Real faces usually usually > 100 (depending on resolution), AI often < 50 (smooth)
        # But blurry photos also low. So this is weak signal alone.
        smoothness_score = 1.0 if sharpness < 50 else 0.2
        
        # 2. Boundary Analysis (Edge artifacts)
        # Check gradient magnitude at the bounding box edges vs inside
        x, y, w, h = bbox
        
        # Extract boundary region (5px margin)
        margin = 5
        x1, y1 = max(0, x-margin), max(0, y-margin)
        x2, y2 = min(full_image.shape[1], x+w+margin), min(full_image.shape[0], y+h+margin)
        
        boundary_region = full_image[y1:y2, x1:x2]
        
        # If we can't extract boundary, skip
        if boundary_region.size == 0:
            boundary_score = 0.0
        else:
            # Calculate gradient magnitude at boundaries
            # Deepfake face boundaries often show discontinuities in gradients
            gray_boundary = cv2.cvtColor(boundary_region, cv2.COLOR_RGB2GRAY)
            
            # Sobel edge detection
            sobelx = cv2.Sobel(gray_boundary, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_boundary, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Calculate local variance of gradients (high variance at spliced edges)
            grad_variance = np.var(gradient_magnitude)
            
            # Compare boundary gradient variance to internal face texture
            face_gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            face_sobelx = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
            face_sobely = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
            face_grad_mag = np.sqrt(face_sobelx**2 + face_sobely**2)
            face_grad_var = np.var(face_grad_mag)
            
            # If boundary has much higher gradient variance, possible splicing
            if face_grad_var > 0:
                var_ratio = grad_variance / face_grad_var
                # Normal faces have edge-to-interior variance ratio around 1-2
                # Spliced faces often show ratios > 3
                boundary_score = _safe_prob(min((var_ratio - 1.5) / 2.0, 1.0)) if var_ratio > 1.5 else 0.0
            else:
                boundary_score = 0.0
            
        # 3. Noise Analysis (Face vs Background)
        # Estimate noise in face
        face_noise_std = self._estimate_noise(face_crop)
        
        # Estimate noise in a background patch (random crop outside face)
        # Simplified: just use whole image noise estimate for now
        bg_noise_std = self._estimate_noise(full_image) # Rough approx
        
        # Discrepancy
        noise_diff = abs(face_noise_std - bg_noise_std)
        # If face noise is significantly different from global noise -> potential splicing
        noise_score = _safe_prob(min(noise_diff / 5.0, 1.0))
        
        # Combined Score
        # Heuristic combination
        combined_score = (smoothness_score * 0.3 + noise_score * 0.7)
        
        return {
            'manipulation_score': combined_score,
            'sharpness': sharpness,
            'noise_diff': noise_diff
        }
        
    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise standard deviation."""
        # Handle grayscale (2D) or color (3D) images
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        # Standard noise estimation using high-pass filter
        h, w = gray.shape
        if h < 5 or w < 5: return 0.0
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(np.float32) - blurred.astype(np.float32)
        return float(np.std(noise))

