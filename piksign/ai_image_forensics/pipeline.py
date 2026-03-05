"""
Unified image forensics pipeline: ELA, PRNU, geometric consistency, and DIRE.
Exact logic from ai-image (FORENSICS_LOGIC.md).
"""

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from .ela import compute_ela, ela_anomaly_score, double_compression_metric
from .prnu import prnu_analysis
from .geometric import geometric_analysis
from .dire_detector import DIREDetector

# Model search: this package dir, DIRE-repo
_PKG_DIR = Path(__file__).resolve().parent
_PKG_PARENT = _PKG_DIR.parent  # piksign
# Workspace root is now just the parent of the package
_WORKSPACE_ROOT = _PKG_PARENT.parent 
_DIRE_REPO = _WORKSPACE_ROOT / "DIRE-repo"

if not _DIRE_REPO.is_dir():
    # Fallback if running from within piksign/
    if (_PKG_PARENT / "DIRE-repo").is_dir():
        _DIRE_REPO = _PKG_PARENT / "DIRE-repo"
    else:
        _DIRE_REPO = Path.cwd() / "DIRE-repo"

_DIRE_REPO_CKPT = _DIRE_REPO / "data" / "exp" / "ckpt"

DIRE_MODEL_PREFERENCE = (
    "model_epoch_latest.pth",
    "imagenet_adm.pth",
    "lsun_adm.pth",
    "lsun_iddpm.pth",
    "lsun_pndm.pth",
    "celebahq_sdv2.pth",
    "lsun_stylegan.pth",
)


def get_default_dire_model_path(choice: Optional[str] = None) -> Optional[Path]:
    """Return path to DIRE checkpoint. Checks: ai_image_forensics/model/ and DIRE-repo/data/exp/ckpt/*/."""
    # 1) Explicit choice in known dirs
    model_dir = _PKG_DIR / "model"
    if model_dir.is_dir():
        if choice:
            base = choice if choice.endswith(".pth") else f"{choice}.pth"
            candidate = model_dir / base
            if candidate.exists():
                return candidate
        else:
            for name in DIRE_MODEL_PREFERENCE:
                candidate = model_dir / name
                if candidate.exists():
                    return candidate
            pth_files = sorted(model_dir.glob("*.pth"))
            if pth_files:
                return pth_files[0]

    # 2) DIRE-repo: data/exp/ckpt/<exp>/model_epoch_latest.pth (official default)
    if _DIRE_REPO_CKPT.is_dir():
        for exp_dir in _DIRE_REPO_CKPT.iterdir():
            if exp_dir.is_dir():
                latest = exp_dir / "model_epoch_latest.pth"
                if latest.exists():
                    return latest
        for pth in _DIRE_REPO_CKPT.rglob("*.pth"):
            return pth
    if _DIRE_REPO.is_dir():
        for pth in _DIRE_REPO.rglob("*.pth"):
            return pth
    return None


def get_default_diffusion_model_path() -> Optional[Path]:
    """Return path to diffusion model for DIRE map computation (e.g. 256x256_diffusion_uncond.pt)."""
    for base in (_DIRE_REPO, _DIRE_REPO / "guided-diffusion"):
        if not base.is_dir():
            continue
        for name in ("256x256_diffusion_uncond.pt", "lsun_bedroom.pt"):
            candidate = base / "models" / name if (base / "models").is_dir() else base / name
            if candidate.exists():
                return candidate
    return None


def compute_dire_map_image(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    diffusion_model_path: Union[str, Path],
    dire_repo_path: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """
    Compute DIRE map for one image using official DIRE repo's guided-diffusion (no MPI).
    Returns path to the saved DIRE map image, or None on failure.
    """
    import subprocess
    import tempfile
    dire_repo = Path(dire_repo_path or _DIRE_REPO)
    script = dire_repo / "guided-diffusion" / "compute_dire_single.py"
    if not script.exists():
        return None
    diffusion_path = Path(diffusion_model_path)
    if not diffusion_path.exists():
        return None
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "-i", str(Path(image_path).resolve()),
        "-o", str(out_path.resolve()),
        "-m", str(diffusion_path.resolve()),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(script.parent),
            capture_output=True,
            text=True,
            timeout=1200,  # 20 min per image (CPU can be slow for diffusion)
        )
        if proc.returncode == 0 and out_path.exists():
            return out_path
        import logging
        logging.getLogger(__name__).warning(
            "DIRE map subprocess failed (rc=%s): %s",
            proc.returncode,
            (proc.stderr or proc.stdout or "")[:500],
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("DIRE map subprocess error: %s", exc)
    return None


def resolve_dire_model_path(
    path: Optional[Union[str, Path]],
    dire_model_name: Optional[str] = None,
) -> Optional[Path]:
    if dire_model_name:
        resolved = get_default_dire_model_path(dire_model_name)
        if resolved is not None:
            return resolved
    if path is None:
        return get_default_dire_model_path()
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file():
        return p if p.suffix.lower() == ".pth" else None
    latest = p / "model_epoch_latest.pth"
    if latest.exists():
        return latest
    for name in DIRE_MODEL_PREFERENCE:
        candidate = p / name
        if candidate.exists():
            return candidate
    pth_files = list(p.glob("*.pth"))
    return pth_files[0] if pth_files else None


VERDICT_REAL = "real"
VERDICT_AI_GENERATED = "ai_generated"
VERDICT_AI_MANIPULATED = "ai_manipulated"


@dataclass
class ForensicsResult:
    path: str
    ela: Dict[str, Any] = field(default_factory=dict)
    prnu: Dict[str, Any] = field(default_factory=dict)
    geometric: Dict[str, Any] = field(default_factory=dict)
    dire: Optional[float] = None
    artifacts: Dict[str, np.ndarray] = field(default_factory=dict)
    combined_anomaly_score: float = 0.0
    errors: List[str] = field(default_factory=list)
    verdict: str = VERDICT_REAL
    verdict_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "verdict": self.verdict,
            "verdict_confidence": self.verdict_confidence,
            "ela": self.ela,
            "prnu": self.prnu,
            "geometric": self.geometric,
            "dire_synthetic_prob": self.dire,
            "combined_anomaly_score": self.combined_anomaly_score,
            "errors": self.errors,
        }


def classify_verdict(
    result: "ForensicsResult",
    dire_threshold: float = 0.55,
    manipulation_threshold: float = 0.35,
) -> tuple:
    """
    Verdict logic:
    1. DIRE >= threshold (only when DIRE maps were used) -> ai_generated
    2. ELA/PRNU/geometric combined >= manipulation_threshold -> ai_manipulated
    3. Otherwise -> real
    """
    dire_prob = result.dire

    ela_anomaly = result.ela.get("anomaly_score", 0.0) if result.ela else 0.0
    prnu_anomaly = result.prnu.get("anomaly_score", 0.0) if result.prnu else 0.0
    geom_anomaly = result.geometric.get("geometric_anomaly_score", 0.0) if result.geometric else 0.0

    # DIRE verdict (only meaningful when computed on DIRE maps, not raw images)
    if dire_prob is not None and 0.01 < dire_prob < 0.98 and dire_prob >= dire_threshold:
        confidence = min(1.0, (dire_prob - dire_threshold) / (1.0 - dire_threshold) + 0.5)
        return (VERDICT_AI_GENERATED, float(np.clip(confidence, 0, 1)))

    # Weighted combination of ELA, PRNU, geometric (ELA is strongest signal)
    manipulation_score = ela_anomaly * 0.45 + prnu_anomaly * 0.25 + geom_anomaly * 0.30

    if manipulation_score >= manipulation_threshold:
        confidence = min(1.0, (manipulation_score - manipulation_threshold) / 0.5)
        return (VERDICT_AI_MANIPULATED, float(np.clip(confidence, 0, 1)))

    confidence = 1.0 - manipulation_score
    return (VERDICT_REAL, float(np.clip(confidence, 0, 1)))


def run_forensics_pipeline(
    image: Union[str, Path],
    *,
    run_ela: bool = True,
    run_prnu: bool = True,
    run_geometric: bool = True,
    run_dire: bool = True,
    dire_model_path: Optional[Union[str, Path]] = None,
    dire_model_name: Optional[str] = None,
    use_dire_maps: bool = False,
    diffusion_model_path: Optional[Union[str, Path]] = None,
    dire_repo_path: Optional[Union[str, Path]] = None,
    save_artifacts: bool = False,
) -> ForensicsResult:
    """
    Run full forensics on one image.
    If use_dire_maps=True (and diffusion model path is set), computes DIRE map first
    then runs the classifier on the map for full DIRE accuracy.
    """
    path = str(Path(image).resolve())
    result = ForensicsResult(path=path)
    artifacts: Dict[str, np.ndarray] = {}

    img_cv = cv2.imread(path)
    if img_cv is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if run_ela:
        try:
            ela_map, _ = compute_ela(path, quality=90, scale=15)
            ela_stats = ela_anomaly_score(ela_map)
            result.ela = ela_stats
            if save_artifacts:
                artifacts["ela_map"] = ela_map
            try:
                dc = double_compression_metric(path)
                result.ela["double_compression"] = dc
            except Exception:
                pass
        except Exception as e:
            result.ela = {"error": str(e)}

    if run_prnu:
        try:
            prnu_out = prnu_analysis(path, block_size=64)
            result.prnu = prnu_out["stats"]
            if save_artifacts:
                artifacts["prnu_anomaly_map"] = prnu_out["anomaly_map"]
        except Exception as e:
            result.prnu = {"error": str(e)}

    if run_geometric:
        try:
            result.geometric = geometric_analysis(path)
        except Exception as e:
            result.geometric = {"error": str(e)}

    # DIRE: only meaningful when run on DIRE maps (reconstruction error images).
    # The official classifier was trained on DIRE maps, NOT raw images.
    # On raw images it always outputs ~0, which is useless and drags down combined score.
    # So: only run DIRE when use_dire_maps=True AND we can actually compute the map.
    if run_dire and use_dire_maps:
        dire_checkpoint = resolve_dire_model_path(dire_model_path, dire_model_name=dire_model_name)
        if dire_checkpoint is not None:
            dire_map_tmp = None
            try:
                diff_path = Path(diffusion_model_path) if diffusion_model_path else get_default_diffusion_model_path()
                if diff_path and diff_path.exists():
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        dire_map_tmp = f.name
                    computed = compute_dire_map_image(
                        path, dire_map_tmp, diff_path, dire_repo_path=dire_repo_path
                    )
                    if computed is not None:
                        detector = DIREDetector(model_path=str(dire_checkpoint), arch="resnet50")
                        raw_dire = detector.predict(str(computed))
                        # If classifier output is saturated (all maps score 0 or 1), ignore DIRE
                        if 0.02 <= raw_dire <= 0.98:
                            result.dire = raw_dire
                        else:
                            result.dire = None
                    else:
                        result.errors.append("DIRE map computation failed (subprocess error or timeout)")
                else:
                    result.errors.append(
                        f"DIRE maps requested but no diffusion model found "
                        f"(searched DIRE-repo={_DIRE_REPO})"
                    )
            except Exception as e:
                result.dire = None
                result.errors.append(f"DIRE: {e}")
            finally:
                if dire_map_tmp and Path(dire_map_tmp).exists():
                    try:
                        Path(dire_map_tmp).unlink()
                    except OSError:
                        pass

    result.artifacts = artifacts

    # Combined score from ELA/PRNU/geometric (weighted, same as verdict)
    ela_s = result.ela.get("anomaly_score", 0.0) if result.ela else 0.0
    prnu_s = result.prnu.get("anomaly_score", 0.0) if result.prnu else 0.0
    geom_s = result.geometric.get("geometric_anomaly_score", 0.0) if result.geometric else 0.0
    combined = ela_s * 0.45 + prnu_s * 0.25 + geom_s * 0.30
    if result.dire is not None and 0.01 < result.dire < 0.98:
        combined = combined * 0.5 + result.dire * 0.5
    result.combined_anomaly_score = float(np.clip(combined, 0, 1))

    result.verdict, result.verdict_confidence = classify_verdict(result)
    return result
