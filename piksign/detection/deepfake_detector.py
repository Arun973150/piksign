# -*- coding: utf-8 -*-
"""
PikSign Deepfake Detector
AI-generated image detection using Reality Defender cloud API.

PURPOSE:
- Detects AI-generated and deepfake imagery using Reality Defender API
- Primary decision-maker for AI-generated content detection

Uses:
- Reality Defender API (cloud) - Enterprise-grade detection
- LocalFaceAnalyzer (OpenCV) - Face region detection
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, Optional, List
import warnings
import logging

warnings.filterwarnings('ignore')

import traceback

# Reality Defender wrapper
from .reality_defender import RealityDefenderDetector


def _safe_prob(x: float) -> float:
    """Clamp value to valid probability range [0, 1]."""
    return float(min(max(x, 0.0), 1.0))


class LocalFaceAnalyzer:
    """Local face analysis using OpenCV."""
    def __init__(self):
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except:
            self.face_cascade = None
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        if self.face_cascade is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return [{'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)} for (x, y, w, h) in faces]


class DeepfakeDetector:
    """
    Main Deepfake Detector using Reality Defender cloud API.
    
    Replaces local models (SigLIP, XceptionNet, DIRE, ResNet50)
    with a single cloud API call for enterprise-grade detection.
    
    Reality Defender returns:
    - Overall status: AUTHENTIC / FAKE / SUSPICIOUS
    - Overall score: 0-1
    - Per-model breakdown with individual scores
    """
    
    def __init__(self, device=None, api_key: Optional[str] = None):
        """
        Initialize the detector with Reality Defender API.
        
        Args:
            device: Unused (kept for API compatibility). Cloud API handles compute.
            api_key: Reality Defender API key. If None, reads from
                     REALITY_DEFENDER_API_KEY environment variable.
        """
        print("Initializing Deepfake Detector...")
        
        # Reality Defender cloud detector
        self.rd_detector = RealityDefenderDetector(api_key=api_key)
        rd_status = 'Ready (cloud API)' if self.rd_detector.enabled else 'Not available'
        print(f"   Reality Defender: {rd_status}")
        
        # Face analyzer (local, always works)
        self.face_analyzer = LocalFaceAnalyzer()
        print("   Local Face Analyzer: Ready")
        
        if not self.rd_detector.enabled:
            print("\n   WARNING: Reality Defender not available!")
            print("      Install SDK: pip install realitydefender")
            print("      Set API key: set REALITY_DEFENDER_API_KEY=your-key")
        
        # Store last computed sub-scores for compatibility
        self._last_sub_scores = {
            'reality_defender': 0.0,
            'rd_models': []
        }
    
    def detect(self, image_path: str, threshold: float = 0.5, use_api: bool = False) -> Dict[str, Any]:
        """
        Detect if an image is AI-generated using Reality Defender API.
        
        Args:
            image_path: Path to image file
            threshold: Detection threshold (0.5 default)
            use_api: Unused, kept for API compatibility
            
        Returns:
            Dict with deepfake_probability, sub_scores, and metadata
        """
        # Validate image file exists
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'deepfake_probability': 0.0}
        
        results = {
            'status': 'success',
            'detectors_used': [],
            'individual_results': {}
        }
        
        # Call Reality Defender API
        if self.rd_detector.enabled:
            rd_result = self.rd_detector.detect(image_path)
            results['individual_results']['reality_defender'] = rd_result
            
            if rd_result.get('status') == 'success':
                final = _safe_prob(rd_result['probability'])
                results['detectors_used'].append('RealityDefender')
                
                # Store per-model breakdown
                rd_models = rd_result.get('models', [])
                self._last_sub_scores = {
                    'reality_defender': final,
                    'rd_models': rd_models
                }
                
                # Also store in legacy sub_scores format for backward compatibility
                results['sub_scores'] = {
                    'reality_defender': final,
                    'ai_source': final,      # Map RD score to legacy keys
                    'universal': 0.0,
                    'dire': 0.0,
                    'statistical': 0.0
                }
                
                # Print detailed breakdown
                print(f"\n   [#] Reality Defender Analysis:")
                print(f"      * Status:  {rd_result.get('rd_status', 'UNKNOWN')}")
                print(f"      * Score:   {final*100:.1f}%")
                
                if rd_models:
                    print(f"      * Models ({len(rd_models)}):")
                    for model in rd_models:
                        score_pct = model.get('score', 0.0) * 100
                        bar = '#' * int(score_pct / 10) + '.' * (10 - int(score_pct / 10))
                        print(f"        - {model['name']}: {score_pct:.1f}% {bar}")
                
            else:
                # API call failed
                final = 0.0
                results['sub_scores'] = {
                    'reality_defender': 0.0,
                    'ai_source': 0.0,
                    'universal': 0.0,
                    'dire': 0.0,
                    'statistical': 0.0
                }
                error_msg = rd_result.get('error', 'Unknown error')
                print(f"\n   [\!] Reality Defender error: {error_msg}")
                results['status'] = 'partial_failure'
        else:
            # Detector not available
            final = 0.0
            results['status'] = 'no_detectors_available'
            results['sub_scores'] = {
                'reality_defender': 0.0,
                'ai_source': 0.0,
                'universal': 0.0,
                'dire': 0.0,
                'statistical': 0.0
            }
            print("\n   [\!] No detectors available")
        
        # Determine manipulation type
        if final > 0.7:
            manipulation_type = 'likely_deepfake'
        elif final > 0.4:
            manipulation_type = 'possible_manipulation'
        else:
            manipulation_type = 'likely_authentic'
        
        results['deepfake_probability'] = final
        results['manipulation_type'] = manipulation_type
        results['confidence'] = abs(final - 0.5) * 2
        
        return results
    
    def detect_multi_scale(self, image_path: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        V2: Multi-Scale Detection.
        
        Runs detection at 3 different resolutions:
        1. Original resolution
        2. Downscaled to 512px (max dimension)
        3. Upscaled to 2048px (max dimension)
        """
        import tempfile
        import os
        
        try:
            original_image = Image.open(image_path).convert('RGB')
            w, h = original_image.size
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'deepfake_probability': 0.0}
        
        results = {
            'status': 'success',
            'multi_scale': True,
            'scales': {},
            'scale_agreement': 0.0,
            'inconclusive': False
        }
        
        # Scale 1: Original
        print("   [#] Multi-Scale Testing:")
        print(f"      * Original ({w}x{h})...", end=" ")
        original_result = self.detect(image_path, threshold)
        results['scales']['original'] = {
            'resolution': f'{w}x{h}',
            'probability': original_result.get('deepfake_probability', 0.0)
        }
        print(f"{results['scales']['original']['probability']*100:.1f}%")
        
        probabilities = [original_result.get('deepfake_probability', 0.0)]
        
        # Scale 2: Downscaled to 512px
        max_dim = 512
        if max(w, h) > max_dim:
            scale_factor = max_dim / max(w, h)
            new_size = (int(w * scale_factor), int(h * scale_factor))
            
            downscaled = original_image.resize(new_size, Image.Resampling.LANCZOS)
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                downscaled.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                print(f"      * Downscaled ({new_size[0]}x{new_size[1]})...", end=" ")
                down_result = self.detect(tmp_path, threshold)
                results['scales']['downscaled'] = {
                    'resolution': f'{new_size[0]}x{new_size[1]}',
                    'probability': down_result.get('deepfake_probability', 0.0)
                }
                print(f"{results['scales']['downscaled']['probability']*100:.1f}%")
                probabilities.append(down_result.get('deepfake_probability', 0.0))
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        # Scale 3: Upscaled to 2048px
        max_dim = 2048
        if max(w, h) < max_dim:
            scale_factor = max_dim / max(w, h)
            new_size = (int(w * scale_factor), int(h * scale_factor))
            
            upscaled = original_image.resize(new_size, Image.Resampling.LANCZOS)
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                upscaled.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                print(f"      * Upscaled ({new_size[0]}x{new_size[1]})...", end=" ")
                up_result = self.detect(tmp_path, threshold)
                results['scales']['upscaled'] = {
                    'resolution': f'{new_size[0]}x{new_size[1]}',
                    'probability': up_result.get('deepfake_probability', 0.0)
                }
                print(f"{results['scales']['upscaled']['probability']*100:.1f}%")
                probabilities.append(up_result.get('deepfake_probability', 0.0))
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        # Compute agreement between scales
        if len(probabilities) >= 2:
            prob_std = np.std(probabilities)
            prob_mean = np.mean(probabilities)
            
            results['scale_agreement'] = max(0.0, 1.0 - prob_std * 3)
            
            max_prob = max(probabilities)
            min_prob = min(probabilities)
            
            if max_prob > 0.6 and min_prob < 0.4:
                results['inconclusive'] = True
                results['note'] = 'Significant divergence between scales'
                final_prob = prob_mean * 0.9
            else:
                final_prob = (
                    probabilities[0] * 0.5 +
                    np.mean(probabilities[1:]) * 0.5
                ) if len(probabilities) > 1 else probabilities[0]
        else:
            results['scale_agreement'] = 1.0
            final_prob = probabilities[0]
        
        results['deepfake_probability'] = _safe_prob(final_prob)
        results['original_result'] = original_result
        
        print(f"      -> Final (multi-scale): {results['deepfake_probability']*100:.1f}%")
        if results['inconclusive']:
            print("      [\!] INCONCLUSIVE: Scales disagree significantly")
        
        return results
    
    def export_feature_vector(self) -> List[float]:
        """
        Export feature vector for future learned fusion (MLP).
        
        Returns:
            List with [reality_defender_score]
        """
        return [self._last_sub_scores.get('reality_defender', 0.0)]

    def detect_video(self, video_path):
        """Detect deepfakes in video (not implemented)."""
        return {'status': 'not_implemented'}