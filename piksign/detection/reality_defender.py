# -*- coding: utf-8 -*-
"""
PikSign Reality Defender Integration

Wraps the Reality Defender cloud API for AI-generated image detection.
Replaces local AI classifiers (SigLIP, XceptionNet, DIRE, ResNet50)
with a single cloud API call.

Install: pip install realitydefender
API Key: Obtain from https://app.realitydefender.ai
"""

import asyncio
import os
import traceback
from typing import Dict, Any, Optional

# Load .env file automatically
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

# Try to import Reality Defender SDK
try:
    from realitydefender import RealityDefender
    HAS_REALITY_DEFENDER = True
except ImportError:
    HAS_REALITY_DEFENDER = False


def _safe_prob(x: float) -> float:
    """Clamp value to valid probability range [0, 1]."""
    return float(min(max(x, 0.0), 1.0))


class RealityDefenderDetector:
    """
    Reality Defender API-based AI detection.

    Uses the Reality Defender cloud service to detect AI-generated,
    deepfake, and manipulated media with enterprise-grade accuracy.

    Provides a synchronous interface wrapping the async SDK.

    Response format from API:
    {
        "status": "AUTHENTIC" | "FAKE" | "SUSPICIOUS" | "NOT_APPLICABLE",
        "score": float (0-1),
        "models": [
            {"name": str, "status": str, "score": float},
            ...
        ]
    }
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Reality Defender detector.

        Args:
            api_key: Reality Defender API key. If None, reads from
                     REALITY_DEFENDER_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("REALITY_DEFENDER_API_KEY", "")
        self.enabled = False
        self.error_message = None
        self.rd = None

        if not HAS_REALITY_DEFENDER:
            self.error_message = "realitydefender SDK not installed. Install with: pip install realitydefender"
            print("   [--] Reality Defender: SDK not installed")
            print("      Install with: pip install realitydefender")
            return

        if not self.api_key:
            self.error_message = "No API key provided. Set REALITY_DEFENDER_API_KEY or pass api_key parameter."
            print("   [--] Reality Defender: No API key configured")
            print("      Set REALITY_DEFENDER_API_KEY environment variable or pass api_key to constructor")
            return

        try:
            self.rd = RealityDefender(api_key=self.api_key)
            self.enabled = True
            print("   [OK] Reality Defender: Ready (cloud API)")
        except Exception as e:
            self.error_message = f"Initialization failed: {str(e)}"
            print(f"   [--] Reality Defender: Init failed: {e}")
            traceback.print_exc()

    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Detect if image is AI-generated using Reality Defender API.

        Uploads the image to Reality Defender's cloud service,
        polls for results, and returns a structured response.

        Args:
            image_path: Path to image file

        Returns:
            Dict with:
                - status: 'success' | 'error' | 'model_unavailable'
                - probability: float (0-1) AI generation probability
                - rd_status: str - Reality Defender's verdict
                - models: list of per-model results
                - raw_response: full API response
        """
        if not self.enabled:
            return {
                'status': 'model_unavailable',
                'probability': 0.0,
                'rd_status': 'unavailable',
                'models': [],
                'error': self.error_message
            }

        try:
            # Run the async upload + polling in a sync context
            raw = self._run_async_detection(image_path)
            return self._parse_result(raw)

        except Exception as e:
            tb = traceback.format_exc()
            print(f"   [!] Reality Defender exception: {type(e).__name__}: {e}")
            print(tb)
            return {
                'status': 'error',
                'probability': 0.0,
                'rd_status': 'error',
                'models': [],
                'error': str(e)
            }

    def _run_async_detection(self, image_path: str) -> Dict[str, Any]:
        """
        Run async Reality Defender detection synchronously.

        Creates a fresh RealityDefender instance inside a dedicated thread so
        the SDK's internal aiohttp session is always bound to a live event loop.
        """
        import concurrent.futures
        api_key = self.api_key

        def _in_thread():
            try:
                return asyncio.run(self._async_detect_fresh(api_key, image_path))
            except Exception as _te:
                tb = traceback.format_exc()
                print(f"\n   [DBG] Thread exception: {type(_te).__name__}: {_te}")
                print(tb)
                raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_in_thread).result(timeout=120)

    @staticmethod
    async def _async_detect_fresh(api_key: str, image_path: str) -> Dict[str, Any]:
        """
        Async detection with a freshly created RealityDefender instance.
        Instantiating inside the coroutine ensures the SDK binds to the
        current (live) event loop, not a stale one from a previous run.
        """
        rd = RealityDefender(api_key=api_key)

        # Step 1: Upload
        print("   Uploading to Reality Defender...", end="\r")
        response = await rd.upload(file_path=image_path)
        request_id = response["request_id"]
        print(f"   Uploaded (ID: {request_id[:8]}...)           ")

        # Step 2: Poll for results
        print("   Waiting for analysis...", end="\r")
        result = await rd.get_result(request_id)
        print("   Analysis complete                          ")

        return result

    def _parse_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Reality Defender API response into PikSign format.

        Maps:
        - "FAKE" -> high probability
        - "SUSPICIOUS" -> moderate probability
        - "AUTHENTIC" -> low probability
        - "NOT_APPLICABLE" -> 0.0

        Args:
            result: Raw API response dict

        Returns:
            Structured detection result
        """
        rd_status = result.get("status", "UNKNOWN")
        rd_score = result.get("score", 0.0)

        # Parse per-model results
        models = []
        for model in result.get("models", []):
            models.append({
                'name': model.get('name', 'unknown'),
                'status': model.get('status', 'unknown'),
                'score': _safe_prob(model.get('score', 0.0))
            })

        # Convert RD score to AI probability
        # RD score is 0-1 where higher = more likely fake
        probability = _safe_prob(rd_score)

        # Map status to ensure probability aligns with verdict
        if rd_status == "FAKE":
            probability = max(probability, 0.6)  # Floor at 60%
        elif rd_status == "SUSPICIOUS":
            probability = max(probability, 0.4)  # Floor at 40%
        elif rd_status == "AUTHENTIC":
            probability = min(probability, 0.4)  # Cap at 40%
        elif rd_status == "NOT_APPLICABLE":
            probability = 0.0

        return {
            'status': 'success',
            'probability': probability,
            'rd_status': rd_status,
            'models': models,
            'model_count': len(models),
            'raw_response': result
        }
