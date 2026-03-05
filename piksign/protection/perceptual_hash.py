# -*- coding: utf-8 -*-
"""
PikSign Perceptual Hash System
Robust perceptual hashing for image verification and tampering detection.
"""

import numpy as np
import cv2
import hashlib
from PIL import Image

# Try to import imagehash
try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False


class PerceptualHashSystem:
    """Robust perceptual hashing for image verification."""
    
    def __init__(self, hash_size: int = 16):
        self.hash_size = hash_size
    
    def compute_phash(self, img_array: np.ndarray) -> tuple:
        """Compute standard perceptual hash (DCT-based)."""
        if not HAS_IMAGEHASH:
            return self._fallback_hash(img_array), None
        
        img_uint8 = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        phash = imagehash.phash(pil_img, hash_size=self.hash_size)
        return str(phash), phash
    
    def compute_average_hash(self, img_array: np.ndarray) -> tuple:
        """Compute average hash."""
        if not HAS_IMAGEHASH:
            return self._fallback_hash(img_array), None
        
        img_uint8 = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        ahash = imagehash.average_hash(pil_img, hash_size=self.hash_size)
        return str(ahash), ahash
    
    def compute_dhash(self, img_array: np.ndarray) -> tuple:
        """Compute difference hash."""
        if not HAS_IMAGEHASH:
            return self._fallback_hash(img_array), None
        
        img_uint8 = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        dhash = imagehash.dhash(pil_img, hash_size=self.hash_size)
        return str(dhash), dhash
    
    def compute_wavelet_hash(self, img_array: np.ndarray) -> tuple:
        """Compute wavelet-based hash."""
        if not HAS_IMAGEHASH:
            return self._fallback_hash(img_array), None
        
        img_uint8 = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        whash = imagehash.whash(pil_img, hash_size=self.hash_size)
        return str(whash), whash
    
    def compute_color_hash(self, img_array: np.ndarray) -> tuple:
        """Compute color-sensitive hash."""
        if not HAS_IMAGEHASH:
            return self._fallback_hash(img_array), None
        
        img_uint8 = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        chash = imagehash.colorhash(pil_img, binbits=3)
        return str(chash), chash
    
    def _fallback_hash(self, img_array: np.ndarray) -> str:
        """Simple fallback hash when imagehash is not available."""
        img_small = cv2.resize((img_array * 255).astype(np.uint8), (8, 8))
        return hashlib.md5(img_small.tobytes()).hexdigest()[:16]
    
    def generate_all_hashes(self, img_array: np.ndarray) -> dict:
        """Generate all perceptual hashes."""
        phash_str, phash_obj = self.compute_phash(img_array)
        ahash_str, ahash_obj = self.compute_average_hash(img_array)
        dhash_str, dhash_obj = self.compute_dhash(img_array)
        whash_str, whash_obj = self.compute_wavelet_hash(img_array)
        chash_str, chash_obj = self.compute_color_hash(img_array)
        
        return {
            'phash': {'string': phash_str, 'object': phash_obj},
            'ahash': {'string': ahash_str, 'object': ahash_obj},
            'dhash': {'string': dhash_str, 'object': dhash_obj},
            'whash': {'string': whash_str, 'object': whash_obj},
            'chash': {'string': chash_str, 'object': chash_obj}
        }
    
    def compute_hash_distance(self, hash1, hash2) -> int:
        """Compute Hamming distance between two hashes."""
        if hash1 is None or hash2 is None:
            return 0
        return hash1 - hash2
    
    def verify_hashes(self, original_hashes: dict, test_hashes: dict, 
                      threshold: int = 8) -> dict:
        """Verify if test image matches original based on hash distances."""
        results = {}
        
        for hash_type in ['phash', 'ahash', 'dhash', 'whash']:
            if original_hashes[hash_type]['object'] is not None:
                distance = self.compute_hash_distance(
                    original_hashes[hash_type]['object'],
                    test_hashes[hash_type]['object']
                )
                results[hash_type] = {
                    'distance': distance,
                    'match': distance <= threshold
                }
            else:
                results[hash_type] = {'distance': 0, 'match': True}
        
        # Color hash uses different comparison
        if original_hashes['chash']['object'] is not None:
            results['chash'] = {
                'distance': self.compute_hash_distance(
                    original_hashes['chash']['object'],
                    test_hashes['chash']['object']
                ),
                'match': True
            }
        else:
            results['chash'] = {'distance': 0, 'match': True}
        
        return results
    
    def generate_composite_fingerprint(self, hashes: dict) -> dict:
        """Create a composite fingerprint from all hashes."""
        fingerprint_str = (
            f"{hashes['phash']['string']}|"
            f"{hashes['ahash']['string']}|"
            f"{hashes['dhash']['string']}|"
            f"{hashes['whash']['string']}|"
            f"{hashes['chash']['string']}"
        )
        composite_hash = hashlib.sha256(fingerprint_str.encode()).hexdigest()
        
        return {
            'fingerprint': fingerprint_str,
            'composite_hash': composite_hash
        }
