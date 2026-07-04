from __future__ import annotations

import threading
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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
    _ade_name_to_id_cache: Optional[Dict[str, int]] = None
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    @staticmethod
    def _normalize_label_name(name: str) -> str:
        return str(name).strip().lower()

    @staticmethod
    def _compact_label_name(name: str) -> str:
        # Compact form improves matching robustness for names like "street light" vs "streetlight".
        return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())

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
                try:
                    self._model.to(memory_format=torch.channels_last)
                except Exception:
                    pass
                try:
                    torch.backends.cudnn.benchmark = True
                except Exception:
                    pass
            else:
                self._model.to("cpu")

            # Invalidate cache after model (re)load.
            self._ade_name_to_id_cache = None

    @property
    def id2label(self) -> Dict[int, str]:
        self.ensure_loaded()
        cfg = getattr(self._model, "config", None)
        id2label = getattr(cfg, "id2label", None)
        if not isinstance(id2label, dict) or not id2label:
            raise RuntimeError("Mask2Former config.id2label missing")
        return {int(k): str(v) for k, v in id2label.items()}

    @property
    def ade_name_to_id(self) -> Dict[str, int]:
        """Returns a robust ADE name index including normalized/compact aliases."""
        with self._lock:
            if self._ade_name_to_id_cache is not None:
                return dict(self._ade_name_to_id_cache)

            id2label = self.id2label
            out: Dict[str, int] = {}
            for ade_id, raw_name in id2label.items():
                norm = self._normalize_label_name(raw_name)
                compact = self._compact_label_name(raw_name)
                out.setdefault(norm, int(ade_id))
                out.setdefault(compact, int(ade_id))

                # Additional alias from comma-separated ADE labels, if present.
                for token in str(raw_name).split(","):
                    tok_norm = self._normalize_label_name(token)
                    tok_compact = self._compact_label_name(token)
                    if tok_norm:
                        out.setdefault(tok_norm, int(ade_id))
                    if tok_compact:
                        out.setdefault(tok_compact, int(ade_id))

            self._ade_name_to_id_cache = out
            return dict(out)

    def predict_ade_ids(self, rgb_u8: np.ndarray) -> np.ndarray:
        """Returns ADE label id map with shape (H, W), dtype int32."""
        out = self.predict_ade_ids_batch([rgb_u8], batch_size=1)
        if not out:
            raise RuntimeError("mask2former returned no prediction")
        return out[0]

    def predict_ade_ids_batch(self, rgb_u8_list: Sequence[np.ndarray], batch_size: int = 4) -> List[np.ndarray]:
        """Returns ADE label maps for input images, preserving input order."""
        with self._lock:
            self.ensure_loaded()
            if not rgb_u8_list:
                return []

            torch = self._torch
            processor = self._processor
            model = self._model
            assert torch is not None and processor is not None and model is not None

            bs = max(1, int(batch_size))
            device = model.device
            results: List[np.ndarray] = []

            i = 0
            current_bs = bs
            while i < len(rgb_u8_list):
                chunk = list(rgb_u8_list[i : i + current_bs])
                sanitized_chunk: List[np.ndarray] = []
                target_sizes: List[Tuple[int, int]] = []
                for rgb_u8 in chunk:
                    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
                        raise ValueError("rgb_u8 must be HxWx3")
                    if rgb_u8.dtype != np.uint8:
                        rgb_u8 = rgb_u8.clip(0, 255).astype(np.uint8)
                    sanitized_chunk.append(np.ascontiguousarray(rgb_u8))
                    target_sizes.append((int(rgb_u8.shape[0]), int(rgb_u8.shape[1])))

                try:
                    inputs = processor(images=sanitized_chunk, return_tensors="pt")
                    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

                    with torch.inference_mode():
                        if device.type == "cuda":
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                outputs = model(**inputs)
                        else:
                            outputs = model(**inputs)

                    preds = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
                    for pred in preds:
                        results.append(pred.detach().to("cpu").numpy().astype(np.int32))
                    i += len(chunk)
                except RuntimeError as e:
                    msg = str(e).lower()
                    is_oom = ("out of memory" in msg) or ("cuda error" in msg and "memory" in msg)
                    if device.type == "cuda" and is_oom and current_bs > 1:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        current_bs = max(1, current_bs // 2)
                        continue
                    raise

            return results
