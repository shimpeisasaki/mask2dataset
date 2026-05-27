from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class Mask2FormerADEEngine:
    """Semantic segmentation engine using Mask2Former pretrained on ADE20K.

    - Loads `facebook/mask2former-swin-large-ade-semantic` (or local folder if provided).
    - Returns per-pixel ADE20K label ids.
    """

    model_name_or_path: str = "facebook/mask2former-swin-large-ade-semantic"
    device_preference: str = "cuda"  # 'cuda' or 'cpu'

    _torch: Optional[object] = None
    _processor: Optional[object] = None
    _model: Optional[object] = None
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None and self._processor is not None and self._torch is not None:
                return

            try:
                import torch
                from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
            except Exception as e:
                raise RuntimeError(
                    "Mask2Former requires 'torch' and 'transformers'. Install requirements.txt first."
                ) from e

            self._torch = torch
            self._processor = AutoImageProcessor.from_pretrained(self.model_name_or_path)
            self._model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_name_or_path)
            self._model.eval()

            if self.device_preference == "cuda" and torch.cuda.is_available():
                try:
                    self._model.to("cuda", dtype=torch.float16)
                except Exception:
                    self._model.to("cuda")
            else:
                self._model.to("cpu")

    @property
    def id2label(self) -> Dict[int, str]:
        self.ensure_loaded()
        cfg = getattr(self._model, "config", None)
        id2label = getattr(cfg, "id2label", None)
        if not isinstance(id2label, dict) or not id2label:
            raise RuntimeError("Mask2Former config.id2label missing")
        return {int(k): str(v) for k, v in id2label.items()}

    def predict_ade_ids(self, rgb_u8: np.ndarray) -> np.ndarray:
        """Returns ADE label id map with shape (H, W), dtype int32."""
        with self._lock:
            self.ensure_loaded()
            if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
                raise ValueError("rgb_u8 must be HxWx3")

            torch = self._torch
            processor = self._processor
            model = self._model
            assert torch is not None and processor is not None and model is not None

            h, w = rgb_u8.shape[:2]
            inputs = processor(images=[rgb_u8], return_tensors="pt")
            device = model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)

            pred = processor.post_process_semantic_segmentation(outputs, target_sizes=[(h, w)])[0]
            return pred.detach().to("cpu").numpy().astype(np.int32)
