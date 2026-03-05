"""
PRNU / Noiseprint-style analysis for image forensics.
From ai-image pipeline.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


def extract_noise_residual(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    else:
        img = np.asarray(image)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_float = img.astype(np.float64) / 255.0
    blur = cv2.blur(img_float, (5, 5))
    blur_strong = cv2.blur(img_float, (15, 15))
    residual = img_float - 0.5 * blur - 0.5 * blur_strong
    return residual.astype(np.float32)


def residual_anomaly_map(
    residual: np.ndarray,
    block_size: int = 64,
) -> Tuple[np.ndarray, dict]:
    h, w = residual.shape[:2]
    bh, bw = block_size, block_size
    ny = (h + bh - 1) // bh
    nx = (w + bw - 1) // bw
    block_energies = np.zeros((ny, nx), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            y0, y1 = i * bh, min((i + 1) * bh, h)
            x0, x1 = j * bw, min((j + 1) * bw, w)
            block = residual[y0:y1, x0:x1]
            block_energies[i, j] = np.var(block)
    mean_energy = float(np.mean(block_energies))
    std_energy = float(np.std(block_energies))
    if std_energy > 1e-10:
        z = np.abs(block_energies - mean_energy) / std_energy
        anomaly_block = np.clip(z, 0, 3) / 3.0
    else:
        anomaly_block = np.zeros_like(block_energies)
    anomaly_map = cv2.resize(anomaly_block, (w, h), interpolation=cv2.INTER_LINEAR)
    anomaly_score = min(1.0, std_energy / (mean_energy + 1e-6) * 0.3)
    stats = {"mean_energy": mean_energy, "std_energy": std_energy, "anomaly_score": anomaly_score}
    return anomaly_map, stats


def prnu_analysis(
    image: Union[str, Path, np.ndarray],
    block_size: int = 64,
) -> dict:
    residual = extract_noise_residual(image)
    anomaly_map, stats = residual_anomaly_map(residual, block_size=block_size)
    return {"residual": residual, "anomaly_map": anomaly_map, "stats": stats}
