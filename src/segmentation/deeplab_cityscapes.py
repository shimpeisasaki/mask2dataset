from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np

import torch
from torchvision import transforms


class DeepLabCityscapesEngine:
    """DeepLabV3 (MobileNetV2) engine intended for Cityscapes.

    This engine tries to load pretrained Cityscapes weights from
    `models/deeplab_cityscapes_mobilenetv2.pth` in the repo root. If not
    found, it will construct a torchvision DeepLab model (backbone may
    differ) and proceed but accuracy will depend on available weights.
    """

    def __init__(self) -> None:
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Minimal id->label mapping including common Cityscapes names used by templates.
        self.id2label: Dict[int, str] = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "traffic light",
            7: "traffic sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motorcycle",
            18: "bicycle",
            19: "parking",
        }

        # preprocessing
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            # Lazy import to avoid requiring heavy libs unless used
            import torchvision.models.segmentation as seg

            # Try to load a user-provided weights file first
            repo_root = Path(__file__).resolve().parents[2]
            models_dir = repo_root / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            weights_path = models_dir / "deeplab_cityscapes_mobilenetv2.pth"

            # If weights file missing, attempt automatic download from config or env
            if not weights_path.exists():
                self._maybe_download_weights(weights_path)
            # The model architecture: use torchvision's DeepLabV3 with mobilenet_v3 backbone if available
            try:
                model = seg.deeplabv3_mobilenet_v3_large(pretrained=False, progress=True)
            except Exception:
                # Fall back to deeplabv3_resnet50 if mobilenet is not available
                model = seg.deeplabv3_resnet50(pretrained=False, progress=True)

            if weights_path.exists():
                state = torch.load(str(weights_path), map_location="cpu")
                model.load_state_dict(state)
            else:
                # No Cityscapes weights found; use the model as-is.
                pass

            model.to(self._device)
            model.eval()
            self._model = model
        except Exception as e:
            raise RuntimeError(f"failed to initialize DeepLab engine: {e}")

    def predict_city_ids(self, rgb: np.ndarray) -> np.ndarray:
        """Predict Cityscapes label ids for a single RGB tile (H,W,3 uint8).

        Returns an integer numpy array of shape (H,W) with label ids matching
        `self.id2label` where possible. If the underlying model has a different
        label set, integer ids correspond to model output channels.
        """
        if self._model is None:
            self.ensure_loaded()

        # Preserve shape
        h, w = rgb.shape[:2]

        img = rgb.astype(np.uint8)
        # transform expects HWC uint8 -> tensor
        inp = self._transform(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            out = self._model(inp)['out']  # type: ignore[index]
            # out: (N, C, H', W')
            probs = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

        # Resize to original tile size if needed
        if probs.shape[0] != h or probs.shape[1] != w:
            import cv2

            probs = cv2.resize(probs.astype('int32'), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

        return probs

    def _maybe_download_weights(self, target_path: Path) -> None:
        """Attempt to download weights to `target_path`.

        Sources checked (in order):
        - environment variable `DEEPLAB_WEIGHTS_URL`
        - config/deeplab_weights.yaml -> key `deeplab_cityscapes_mobilenetv2_url`

        If no URL found, do nothing.
        """
        import urllib.request
        import yaml

        # 1) env var
        url = os.environ.get("DEEPLAB_WEIGHTS_URL")

        # 2) config file
        if not url:
            repo_root = Path(__file__).resolve().parents[2]
            cfg = repo_root / "config" / "deeplab_weights.yaml"
            if cfg.exists():
                try:
                    raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
                    url = raw.get("deeplab_cityscapes_mobilenetv2_url")
                except Exception:
                    url = None

        if not url:
            return

        try:
            print(f"Downloading DeepLab weights from {url} -> {target_path}")

            def _report(block_num, block_size, total_size):
                if total_size <= 0:
                    return
                downloaded = block_num * block_size
                pct = min(100, downloaded * 100 / total_size)
                print(f"downloaded {pct:.1f}%\r", end="")

            urllib.request.urlretrieve(url, filename=str(target_path), reporthook=_report)
            print("\nDownload complete")
        except Exception as e:
            print(f"failed to download weights: {e}")
