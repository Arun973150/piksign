# -*- coding: utf-8 -*-
"""
PikSign GPU Server for Google Colab
FastAPI server exposing GPU-heavy operations via HTTP.

Run with:
    uvicorn colab_server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import base64
import io
import time
import traceback
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

app = FastAPI(title="PikSign GPU Server", version="1.0.0")

# ---------------------------------------------------------------------------
# Global singletons (initialized at startup)
# ---------------------------------------------------------------------------
DEVICE = None
leat_attack = None
content_analyzer = None
drift_controller = None
startup_info = {}


def _decode_image(b64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _image_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """Convert PIL Image to (1, 3, H, W) float tensor in [0, 1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def _image_to_array(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to float32 numpy array in [0, 1]."""
    return np.array(img).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Startup: load all models into GPU
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    global DEVICE, leat_attack, content_analyzer, drift_controller, startup_info

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0

    print(f"[*] Device: {DEVICE} ({gpu_name})")
    print(f"[*] VRAM: {vram:.1f} GB")

    startup_info = {
        "device": str(DEVICE),
        "gpu": gpu_name,
        "vram_gb": round(vram, 1),
        "encoders_loaded": [],
    }

    # Load ContentAnalyzer
    try:
        from piksign.protection.content_analyzer import ContentAnalyzer
        content_analyzer = ContentAnalyzer(DEVICE)
        print("[OK] ContentAnalyzer loaded")
    except Exception as e:
        print(f"[!] ContentAnalyzer failed: {e}")

    # Load SemanticDriftController
    try:
        from piksign.protection.semantic_drift import SemanticDriftController
        drift_controller = SemanticDriftController(DEVICE)
        print("[OK] SemanticDriftController loaded")
    except Exception as e:
        print(f"[!] SemanticDriftController failed: {e}")

    # Load LEAT Attack (heaviest - 7 encoders)
    try:
        from piksign.protection.leat_attack import LEATAttack
        leat_attack = LEATAttack(
            device=DEVICE,
            iterations=50,
            step_size=0.01,
            epsilon=0.08,
        )
        startup_info["encoders_loaded"] = list(leat_attack.encoders.keys())
        print(f"[OK] LEAT loaded with encoders: {startup_info['encoders_loaded']}")
    except Exception as e:
        print(f"[!] LEAT failed: {e}")
        traceback.print_exc()

    print(f"\n[*] PikSign GPU Server ready on {DEVICE}")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class LEATRequest(BaseModel):
    image_b64: str
    iterations: int = 50
    epsilon: float = 0.08
    step_size: float = 0.01

class ContentRequest(BaseModel):
    image_b64: str

class DriftRequest(BaseModel):
    original_b64: str
    protected_b64: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/v1/health")
async def health():
    """Health check and capability report."""
    return {
        "status": "ok",
        "gpu": startup_info.get("gpu", "unknown"),
        "vram_gb": startup_info.get("vram_gb", 0),
        "device": startup_info.get("device", "cpu"),
        "encoders_loaded": startup_info.get("encoders_loaded", []),
        "leat_available": leat_attack is not None,
        "content_analyzer_available": content_analyzer is not None,
        "drift_controller_available": drift_controller is not None,
    }


@app.post("/api/v1/leat/generate")
async def leat_generate(req: LEATRequest):
    """Generate LEAT adversarial perturbation."""
    if leat_attack is None:
        raise HTTPException(status_code=503, detail="LEAT not loaded")

    try:
        img = _decode_image(req.image_b64)
        img_tensor = _image_to_tensor(img, DEVICE)

        # Override iteration params if provided
        leat_attack.iterations = req.iterations
        leat_attack.step_size = req.step_size
        leat_attack.epsilon = req.epsilon

        t0 = time.time()
        perturbation = leat_attack.generate_perturbation(img_tensor)
        leat_metrics = leat_attack.compute_latent_disruption(img_tensor, perturbation)
        elapsed = time.time() - t0

        # Convert perturbation to base64 (keep float32 precision)
        pert_np = perturbation.cpu().detach().numpy()
        pert_b64 = base64.b64encode(pert_np.astype(np.float32).tobytes()).decode("utf-8")

        # Serialize metrics (convert tensors/numpy to floats)
        clean_metrics = {}
        for enc_name, enc_data in leat_metrics.items():
            if isinstance(enc_data, dict):
                clean_metrics[enc_name] = {
                    k: float(v) if hasattr(v, '__float__') else v
                    for k, v in enc_data.items()
                }
            else:
                clean_metrics[enc_name] = float(enc_data) if hasattr(enc_data, '__float__') else enc_data

        return {
            "perturbation_b64": pert_b64,
            "perturbation_shape": list(pert_np.shape),
            "leat_metrics": clean_metrics,
            "elapsed_seconds": round(elapsed, 2),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/content/analyze")
async def content_analyze(req: ContentRequest):
    """Analyze image content (faces, vulnerability, texture, risk)."""
    if content_analyzer is None:
        raise HTTPException(status_code=503, detail="ContentAnalyzer not loaded")

    try:
        img = _decode_image(req.image_b64)
        img_tensor = _image_to_tensor(img, DEVICE)
        img_array = _image_to_array(img)

        result = content_analyzer.analyze(img_tensor, img_array)

        # Ensure all values are JSON serializable
        return {
            "has_faces": bool(result.get("has_faces", False)),
            "face_count": int(result.get("face_count", 0)),
            "clip_vulnerability": float(result.get("clip_vulnerability", 0.0)),
            "texture_complexity": float(result.get("texture_complexity", 0.5)),
            "risk_level": str(result.get("risk_level", "LOW")),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/drift/compute")
async def drift_compute(req: DriftRequest):
    """Compute embedding drift between original and protected images."""
    if drift_controller is None:
        raise HTTPException(status_code=503, detail="DriftController not loaded")

    try:
        orig_img = _decode_image(req.original_b64)
        prot_img = _decode_image(req.protected_b64)

        orig_tensor = _image_to_tensor(orig_img, DEVICE)
        prot_tensor = _image_to_tensor(prot_img, DEVICE)

        drift = drift_controller.compute_embedding_drift(orig_tensor, prot_tensor)

        return {"embedding_drift": float(drift)}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
