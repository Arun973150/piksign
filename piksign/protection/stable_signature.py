# -*- coding: utf-8 -*-
"""
PikSign Stable Signature Watermark
Integration with imwatermark library for robust watermarking.
"""

import numpy as np
import cv2

# Try to import imwatermark
try:
    from imwatermark import WatermarkEncoder, WatermarkDecoder
    HAS_IMWATERMARK = True
except ImportError:
    HAS_IMWATERMARK = False


class StableSignatureWatermark:
    """Stable Signature watermark using imwatermark library."""
    
    def __init__(self, message: bytes = b'PikSign2026', method: str = 'dwtDct'):
        self.message = message
        self.method = method
        self.enabled = HAS_IMWATERMARK
        
        if self.enabled:
            self.encoder = WatermarkEncoder()
            self.decoder = WatermarkDecoder('bytes', 88)
            self.encoder.set_watermark('bytes', self.message)
    
    def embed(self, img_array: np.ndarray) -> np.ndarray:
        """
        Embed stable signature watermark.
        
        Args:
            img_array: Image as float32 array (0-1 range)
            
        Returns:
            np.ndarray: Watermarked image
        """
        if not self.enabled:
            return img_array
        
        try:
            img_uint8 = (img_array * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            
            watermarked_bgr = self.encoder.encode(img_bgr, self.method)
            watermarked_rgb = cv2.cvtColor(watermarked_bgr, cv2.COLOR_BGR2RGB)
            
            return watermarked_rgb.astype(np.float32) / 255.0
        except:
            return img_array
    
    def extract(self, img_array: np.ndarray) -> tuple:
        """
        Extract watermark from image.
        
        Returns:
            tuple: (extracted_message, confidence)
        """
        if not self.enabled:
            return b'', 0.0
        
        try:
            img_uint8 = (img_array * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            
            extracted = self.decoder.decode(img_bgr, self.method)
            
            if extracted == self.message:
                return extracted, 1.0
            
            matches = sum(a == b for a, b in zip(extracted, self.message))
            confidence = matches / len(self.message) if len(self.message) > 0 else 0
            
            return extracted, confidence
        except:
            return b'', 0.0
    
    @property
    def is_available(self) -> bool:
        """Check if stable signature is available."""
        return self.enabled
