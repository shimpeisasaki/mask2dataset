from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision.models._utils import IntermediateLayerGetter
from torchvision import transforms

logger = logging.getLogger(__name__)


class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        return F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels: int, low_level_channels: int, num_classes: int, aspp_dilate: Iterable[int] = (6, 12, 18)) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.aspp = ASPP(in_channels, tuple(aspp_dilate))
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )
        self._init_weight()

    def forward(self, feature: Dict[str, torch.Tensor]) -> torch.Tensor:
        low_level_feature = self.project(feature["low_level"])
        output_feature = self.aspp(feature["out"])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode="bilinear", align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class DeepLabHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, aspp_dilate: Iterable[int] = (6, 12, 18)) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            ASPP(in_channels, tuple(aspp_dilate)),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )
        self._init_weight()

    def forward(self, feature: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.classifier(feature["out"])

    def _init_weight(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def _init_weight(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = super().forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: Tuple[int, int, int]) -> None:
        super().__init__()
        out_channels = 256
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]

        rate1, rate2, rate3 = atrous_rates
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabCityscapesEngine:
    """DeepLabV3+ MobileNetV2 engine for Cityscapes-trained weights."""

    def __init__(self, output_stride: int = 16) -> None:
        self._model: Optional[nn.Module] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_stride = output_stride
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
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _build_model(self) -> nn.Module:
        backbone = mobilenet_v2(weights=None)
        backbone.low_level_features = backbone.features[0:4]
        backbone.high_level_features = backbone.features[4:-1]
        backbone.features = None
        backbone.classifier = None

        aspp_dilate = (12, 24, 36) if self.output_stride == 8 else (6, 12, 18)
        inplanes = 320
        low_level_planes = 24
        return_layers = {"high_level_features": "out", "low_level_features": "low_level"}
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, 19, aspp_dilate)
        return _SimpleSegmentationModel(backbone, classifier)

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return

        repo_root = Path(__file__).resolve().parents[2]
        models_dir = repo_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        candidates = [
            models_dir / "deeplab_cityscapes_mobilenetv2.pth",
            models_dir / "cc_ai_best_deeplabv3plus_mobilenet_cityscapes_os16.pth",
        ]
        weights_path = next((path for path in candidates if path.exists()), candidates[0])
        if not weights_path.exists():
            self._maybe_download_weights(weights_path)

        model = self._build_model()
        if weights_path.exists():
            checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
            state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            try:
                result = model.load_state_dict(state, strict=True)
                if result.missing_keys or result.unexpected_keys:
                    logger.warning(
                        "Loaded Cityscapes checkpoint with key mismatch: missing %d keys, unexpected %d keys.",
                        len(result.missing_keys),
                        len(result.unexpected_keys),
                    )
            except Exception:
                result = model.load_state_dict(state, strict=False)
                logger.warning(
                    "Loaded Cityscapes checkpoint with fallback strict=False: missing %d keys, unexpected %d keys.",
                    len(result.missing_keys),
                    len(result.unexpected_keys),
                )
                if result.missing_keys:
                    logger.debug("Missing keys (first 10): %s", result.missing_keys[:10])
                if result.unexpected_keys:
                    logger.debug("Unexpected keys (first 10): %s", result.unexpected_keys[:10])
        else:
            logger.info("No Cityscapes weights file found; using randomly initialized model.")

        model.to(self._device)
        model.eval()
        self._model = model

    def predict_city_ids(self, rgb: np.ndarray) -> np.ndarray:
        if self._model is None:
            self.ensure_loaded()

        h, w = rgb.shape[:2]
        inp = self._transform(rgb.astype(np.uint8)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            out = self._model(inp)
            pred = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

        if pred.shape[:2] != (h, w):
            import cv2

            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        return pred

    def _maybe_download_weights(self, target_path: Path) -> None:
        import urllib.request
        import yaml

        url = os.environ.get("DEEPLAB_WEIGHTS_URL")
        if not url:
            repo_root = Path(__file__).resolve().parents[2]
            cfg = repo_root / "config" / "deeplab_weights.yaml"
            if cfg.exists():
                try:
                    raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        url = raw.get("deeplab_cityscapes_mobilenetv2_url") or None
                except Exception:
                    url = None

        if not url:
            return

        try:
            print(f"Downloading DeepLab Cityscapes weights from {url} -> {target_path}")

            def _report(block_num: int, block_size: int, total_size: int) -> None:
                if total_size <= 0:
                    return
                downloaded = block_num * block_size
                pct = min(100.0, downloaded * 100.0 / total_size)
                print(f"downloaded {pct:.1f}%\r", end="")

            urllib.request.urlretrieve(url, filename=str(target_path), reporthook=_report)
            print("\nDownload complete")
        except Exception as exc:
            print(f"failed to download weights: {exc}")
