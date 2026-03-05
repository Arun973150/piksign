"""
Geometric and shadow consistency for image forensics.
From ai-image pipeline.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


def gradient_direction_histogram(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180
    if mask is not None:
        mag = mag * np.maximum(mask, 0.01)
    strong = mag > np.percentile(mag, 70)
    angles = ang[strong].ravel()
    hist, _ = np.histogram(angles, bins=36, range=(0, 180))
    return hist.astype(np.float32) / (np.sum(hist) + 1e-8)


def shadow_consistency_score(
    image: Union[str, Path, np.ndarray],
    n_regions: int = 4,
) -> dict:
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    else:
        img = np.asarray(image)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    h, w = gray.shape
    histograms: List[np.ndarray] = []
    step_h, step_w = h // n_regions, w // n_regions
    for i in range(n_regions):
        for j in range(n_regions):
            y0, y1 = i * step_h, (i + 1) * step_h if i < n_regions - 1 else h
            x0, x1 = j * step_w, (j + 1) * step_w if j < n_regions - 1 else w
            region = gray[y0:y1, x0:x1]
            hist = gradient_direction_histogram(region)
            histograms.append(hist)
    n_hist = len(histograms)
    correlations: List[float] = []
    for a in range(n_hist):
        for b in range(a + 1, n_hist):
            c = np.corrcoef(histograms[a], histograms[b])[0, 1]
            correlations.append(float(c) if not np.isnan(c) else 0.0)
    mean_corr = float(np.mean(correlations)) if correlations else 0.0
    consistency = float(np.clip((mean_corr + 1) / 2, 0, 1))
    return {"consistency_score": consistency, "mean_histogram_correlation": mean_corr, "n_regions": n_hist}


def edge_density_by_region(image: np.ndarray, n_regions: int = 4) -> dict:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape
    step_h, step_w = h // n_regions, w // n_regions
    densities: List[float] = []
    for i in range(n_regions):
        for j in range(n_regions):
            y0, y1 = i * step_h, (i + 1) * step_h if i < n_regions - 1 else h
            x0, x1 = j * step_w, (j + 1) * step_w if j < n_regions - 1 else w
            region = edges[y0:y1, x0:x1]
            densities.append(float(np.mean(region) / 255.0))
    std_density = float(np.std(densities))
    return {"edge_densities": densities, "std_edge_density": std_density, "inconsistency_hint": std_density > 0.1}


def geometric_analysis(image: Union[str, Path, np.ndarray]) -> dict:
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    else:
        img = np.asarray(image)
    shadow = shadow_consistency_score(img, n_regions=4)
    edge = edge_density_by_region(img, n_regions=4)
    geom_anomaly = (1.0 - shadow["consistency_score"]) * 0.6 + min(1.0, edge["std_edge_density"] * 5) * 0.4
    return {
        "shadow_consistency": shadow,
        "edge_density": edge,
        "geometric_anomaly_score": float(np.clip(geom_anomaly, 0, 1)),
    }
