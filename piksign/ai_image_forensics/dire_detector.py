"""
DIRE (DIffusion REstoration Error) detector for diffusion-generated images.
Aligned with official implementation: https://github.com/ZhendongWang6/DIRE
Pre-trained models: BaiduDrive/RecDrive (password: dire).
"""

from pathlib import Path
from typing import Optional, Union

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from .networks.resnet import resnet50


def get_dire_network(arch: str = "resnet50"):
    """Build ResNet with single-class output (same as official DIRE get_network)."""
    if "resnet50" in arch:
        return resnet50(num_classes=1)
    elif "resnet18" in arch:
        from .networks.resnet import resnet18
        return resnet18(num_classes=1)
    raise ValueError(f"Unsupported arch: {arch}")


def _load_checkpoint(model: torch.nn.Module, path: Union[str, Path]) -> None:
    """Load DIRE checkpoint; supports 'model' and 'state_dict' keys (official format)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DIRE model not found: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if state is None:
        raise ValueError(f"Empty checkpoint: {path}")
    # Official demo: state_dict["model"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Strip 'module.' prefix if present (DDP / DataParallel)
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        result = model.load_state_dict(state, strict=False)
        missing = getattr(result, "missing_keys", [])
        unexpected = getattr(result, "unexpected_keys", [])
        if missing or unexpected:
            import warnings
            warnings.warn(
                f"DIRE checkpoint {path}: strict load failed. Missing: {len(missing)}, Unexpected: {len(unexpected)}. "
                "If DIRE was trained on DIRE maps (not raw images), scores on raw images can be near zero."
            )
        else:
            raise RuntimeError(f"Failed to load DIRE checkpoint {path}: {e}") from e


class DIREDetector:
    """
    DIRE classifier wrapper. Outputs P(synthetic) in [0,1].
    Same preprocessing as official demo: Resize(256), CenterCrop(224), ImageNet normalize.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        arch: str = "resnet50",
        device: Optional[str] = None,
        aug_norm: bool = True,
    ):
        self.arch = arch
        self.aug_norm = aug_norm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_dire_network(arch)
        if model_path:
            _load_checkpoint(self.model, model_path)
        self.model.eval()
        self.model.to(self.device)
        self.transforms = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ])

    def preprocess(self, img: Union[Image.Image, "np.ndarray"]) -> torch.Tensor:
        import numpy as np
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensor = self.transforms(img)
        if self.aug_norm:
            tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image, "np.ndarray"]) -> float:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        x = self.preprocess(image)
        logit = self.model(x)
        return torch.sigmoid(logit).item()

    def __call__(self, image: Union[str, Path, Image.Image, "np.ndarray"]) -> float:
        return self.predict(image)
