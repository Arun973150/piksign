# -*- coding: utf-8 -*-
"""
PikSign GPU Client
HTTP client for calling the Colab GPU server via ngrok.
"""

import time
import base64
import io
import numpy as np
import requests
from PIL import Image
from typing import Optional, Dict, Any


class ColabGPUClient:
    """HTTP client for PikSign Colab GPU server."""

    def __init__(self, base_url: str, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
        }
        self._healthy = None
        self._last_check = 0.0

    def _encode_image(self, img) -> str:
        """Encode PIL Image or numpy array to base64 PNG string."""
        if isinstance(img, np.ndarray):
            # Convert float [0,1] to uint8 if needed
            if img.dtype in (np.float32, np.float64):
                img = (img * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _decode_perturbation(self, b64_str: str, shape: list) -> np.ndarray:
        """Decode base64-encoded float32 numpy array."""
        raw = base64.b64decode(b64_str)
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)

    def is_available(self) -> bool:
        """Check if Colab GPU server is reachable. Caches for 30s."""
        now = time.time()
        if self._healthy is not None and (now - self._last_check) < 30:
            return self._healthy
        try:
            r = requests.get(
                f"{self.base_url}/api/v1/health",
                headers=self.headers,
                timeout=5,
            )
            self._healthy = r.status_code == 200
        except Exception:
            self._healthy = False
        self._last_check = now
        return self._healthy

    def get_health(self) -> Optional[Dict]:
        """Get server health info."""
        try:
            r = requests.get(
                f"{self.base_url}/api/v1/health",
                headers=self.headers,
                timeout=5,
            )
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def generate_leat(
        self,
        image,
        iterations: int = 50,
        epsilon: float = 0.08,
        step_size: float = 0.01,
    ) -> Optional[Dict]:
        """
        Call LEAT generation endpoint.

        Args:
            image: PIL Image or numpy array
            iterations: PGD iterations
            epsilon: L-inf perturbation bound
            step_size: Step size per iteration

        Returns:
            Dict with 'perturbation' (numpy array) and 'leat_metrics' (dict),
            or None on failure.
        """
        try:
            payload = {
                "image_b64": self._encode_image(image),
                "iterations": iterations,
                "epsilon": epsilon,
                "step_size": step_size,
            }
            r = requests.post(
                f"{self.base_url}/api/v1/leat/generate",
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            if r.status_code == 200:
                data = r.json()
                perturbation = self._decode_perturbation(
                    data["perturbation_b64"], data["perturbation_shape"]
                )
                return {
                    "perturbation": perturbation,
                    "leat_metrics": data.get("leat_metrics", {}),
                }
        except Exception as e:
            print(f"      [!] GPU LEAT call failed: {e}")
        return None

    def analyze_content(self, image) -> Optional[Dict]:
        """
        Call content analysis endpoint.

        Args:
            image: PIL Image or numpy array

        Returns:
            Dict with has_faces, face_count, clip_vulnerability,
            texture_complexity, risk_level. Or None on failure.
        """
        try:
            payload = {"image_b64": self._encode_image(image)}
            r = requests.post(
                f"{self.base_url}/api/v1/content/analyze",
                json=payload,
                headers=self.headers,
                timeout=30,
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            print(f"      [!] GPU content analysis failed: {e}")
        return None

    def compute_drift(self, original, protected) -> Optional[float]:
        """
        Call embedding drift computation endpoint.

        Args:
            original: PIL Image or numpy array (original image)
            protected: PIL Image or numpy array (protected image)

        Returns:
            Float drift value, or None on failure.
        """
        try:
            payload = {
                "original_b64": self._encode_image(original),
                "protected_b64": self._encode_image(protected),
            }
            r = requests.post(
                f"{self.base_url}/api/v1/drift/compute",
                json=payload,
                headers=self.headers,
                timeout=30,
            )
            if r.status_code == 200:
                data = r.json()
                return float(data.get("embedding_drift", 0.0))
        except Exception as e:
            print(f"      [!] GPU drift call failed: {e}")
        return None
