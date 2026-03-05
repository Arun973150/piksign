"""
Error Level Analysis (ELA) for image forensics.
From ai-image pipeline.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


def compute_ela(
    image: Union[str, Path, np.ndarray],
    quality: int = 90,
    scale: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    else:
        img = np.asarray(image)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    tmp_path = Path(".ela_tmp.jpg")
    cv2.imwrite(str(tmp_path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    compressed = cv2.imread(str(tmp_path))
    try:
        tmp_path.unlink(missing_ok=True)
    except OSError:
        pass

    if compressed is None:
        raise RuntimeError("Failed to write/read temporary JPEG for ELA")

    diff = cv2.absdiff(img, compressed)
    diff_float = np.mean(diff, axis=2).astype(np.float32)
    ela_map = np.clip(diff_float * scale, 0, 255).astype(np.uint8)
    return ela_map, diff_float


def ela_anomaly_score(ela_map: np.ndarray) -> dict:
    mean_val = float(np.mean(ela_map))
    std_val = float(np.std(ela_map))
    max_val = int(np.max(ela_map))
    kernel_size = 32
    if min(ela_map.shape) >= kernel_size:
        local_means = cv2.blur(ela_map.astype(np.float32), (kernel_size, kernel_size))
        spatial_variance = float(np.var(local_means))
    else:
        spatial_variance = 0.0
    anomaly_score = min(1.0, (std_val / 30.0) * 0.5 + (min(spatial_variance / 100.0, 1.0)) * 0.5)
    return {
        "mean": mean_val,
        "std": std_val,
        "max": max_val,
        "spatial_variance": spatial_variance,
        "anomaly_score": anomaly_score,
    }


def double_compression_metric(image: Union[str, Path, np.ndarray]) -> dict:
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
    else:
        img = np.asarray(image)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image}")
    ela_90, _ = compute_ela(img, quality=90, scale=15)
    ela_75, _ = compute_ela(img, quality=75, scale=15)
    s90 = ela_anomaly_score(ela_90)
    s75 = ela_anomaly_score(ela_75)
    return {
        "quality_90": s90,
        "quality_75": s75,
        "double_compression_hint": s90["anomaly_score"] > 0.5 or s75["anomaly_score"] > 0.5,
    }
