from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from src.segmentation.class_map import ClassMap


def _normalize_prompt(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    return x0, y0, x1, y1


def _extract_embedding_tensor(torch: object, obj: object) -> "object":
    """Return a torch.Tensor-like embedding from various transformers outputs."""
    Tensor = getattr(torch, "Tensor", None)
    if Tensor is not None and isinstance(obj, Tensor):
        return obj

    # Common outputs
    for attr in ("text_embeds", "image_embeds", "pooler_output"):
        v = getattr(obj, attr, None)
        if Tensor is not None and isinstance(v, Tensor):
            return v

    # Fallback: use CLS token of last_hidden_state
    v = getattr(obj, "last_hidden_state", None)
    if Tensor is not None and isinstance(v, Tensor) and v.ndim >= 2:
        return v[:, 0, :]

    raise TypeError(f"Unexpected CLIP features type: {type(obj)}")


@dataclass
class FastSAMPromptEngine:
    """FastSAM instance masks + CLIP prompt classification -> semantic label map.

    This keeps the rest of the codebase compatible with mmseg-style labels:
    returns a (H, W) uint8 map of dataset ids (0..7) or 255(ignore).
    """

    fastsam_weights: str = "FastSAM-s.pt"
    clip_model_name_or_path: str = "openai/clip-vit-base-patch32"
    device_preference: str = "cuda"  # 'cuda' or 'cpu'

    fastsam_imgsz: int = 1024
    fastsam_conf: float = 0.25
    fastsam_iou: float = 0.9

    max_masks: int = 64
    min_area_ratio: float = 0.001  # relative to image area

    _torch: Optional[object] = None
    _sam: Optional[object] = None
    _clip_model: Optional[object] = None
    _clip_processor: Optional[object] = None

    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._sam is not None and self._clip_model is not None and self._clip_processor is not None:
                return

            try:
                import torch
            except Exception as e:
                raise RuntimeError("FastSAM prompt mode requires 'torch'.") from e

            try:
                # Ultralytics FastSAM implementation.
                try:
                    from ultralytics import FastSAM  # type: ignore
                except Exception:
                    from ultralytics.models.fastsam import FastSAM  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "FastSAM prompt mode requires 'ultralytics'. Install requirements.txt first."
                ) from e

            try:
                from transformers import CLIPModel, CLIPProcessor
            except Exception as e:
                raise RuntimeError(
                    "FastSAM prompt mode requires 'transformers'. Install requirements.txt first."
                ) from e

            self._torch = torch

            # FastSAM
            try:
                self._sam = FastSAM(self.fastsam_weights)
            except Exception as e:
                raise RuntimeError(
                    "Failed to load FastSAM weights. "
                    f"weights='{self.fastsam_weights}'. "
                    "If you don't have the weights file yet, download it (e.g. FastSAM-s.pt) "
                    "and place it where this path points, or change FastSAMPromptEngine.fastsam_weights."
                ) from e

            # CLIP
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name_or_path)
            self._clip_model = CLIPModel.from_pretrained(self.clip_model_name_or_path)
            self._clip_model.eval()

            if self.device_preference == "cuda" and torch.cuda.is_available():
                try:
                    self._clip_model.to("cuda", dtype=torch.float16)
                except Exception:
                    self._clip_model.to("cuda")
            else:
                self._clip_model.to("cpu")

    def _device(self) -> str:
        self.ensure_loaded()
        torch = self._torch
        assert torch is not None
        if self.device_preference == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _predict_instance_masks(self, rgb_u8: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """Return (masks, scores). masks are boolean (H,W)."""
        self.ensure_loaded()
        torch = self._torch
        sam = self._sam
        assert torch is not None and sam is not None

        device = self._device()
        h, w = rgb_u8.shape[:2]

        # Ultralytics API varies slightly across versions; support both call styles.
        try:
            results = sam.predict(
                source=rgb_u8,
                imgsz=int(self.fastsam_imgsz),
                conf=float(self.fastsam_conf),
                iou=float(self.fastsam_iou),
                device=device,
                verbose=False,
            )
        except Exception:
            results = sam(
                rgb_u8,
                imgsz=int(self.fastsam_imgsz),
                conf=float(self.fastsam_conf),
                iou=float(self.fastsam_iou),
                device=device,
                verbose=False,
            )

        if not results:
            return [], []

        r0 = results[0]
        masks_obj = getattr(r0, "masks", None)
        if masks_obj is None:
            return [], []

        data = getattr(masks_obj, "data", None)
        if data is None:
            return [], []

        if isinstance(data, torch.Tensor):
            data = data.detach().to("cpu")
            masks_np = data.numpy()
        else:
            masks_np = np.asarray(data)

        if masks_np.ndim != 3:
            return [], []

        # Some Ultralytics/FastSAM variants return masks in the model canvas size
        # instead of the original image size. Force them back to the caller image size.
        if masks_np.shape[1] != h or masks_np.shape[2] != w:
            resized = np.empty((masks_np.shape[0], h, w), dtype=np.float32)
            for i in range(masks_np.shape[0]):
                resized[i] = cv2.resize(
                    masks_np[i].astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            masks_np = resized

        # Confidence, if available
        confs: List[float] = []
        boxes = getattr(r0, "boxes", None)
        conf_t = getattr(boxes, "conf", None) if boxes is not None else None
        if isinstance(conf_t, torch.Tensor):
            confs = conf_t.detach().to("cpu").numpy().astype(float).tolist()
        elif conf_t is not None:
            try:
                confs = np.asarray(conf_t, dtype=float).tolist()
            except Exception:
                confs = []

        masks: List[np.ndarray] = []
        scores: List[float] = []
        for i in range(masks_np.shape[0]):
            m = masks_np[i]
            if m.dtype != np.bool_:
                m = m > 0.5
            masks.append(m)
            scores.append(float(confs[i]) if i < len(confs) else 1.0)

        return masks, scores

    def _build_text_index(self, cm: ClassMap) -> Tuple[List[str], Dict[int, List[int]]]:
        """Return (unique_prompts, class_id -> prompt_indices)."""
        unique: List[str] = []
        index: Dict[str, int] = {}
        cls_to_idx: Dict[int, List[int]] = {}

        for cls_id, prompts in cm.id_to_prompts.items():
            idxs: List[int] = []
            for p in prompts:
                key = _normalize_prompt(p)
                if not key:
                    continue
                if key not in index:
                    index[key] = len(unique)
                    unique.append(p)
                idxs.append(index[key])
            if idxs:
                cls_to_idx[int(cls_id)] = idxs

        if not unique:
            raise ValueError("class_map has no prompts")

        return unique, cls_to_idx

    def _clip_text_features(self, prompts: List[str]) -> np.ndarray:
        self.ensure_loaded()
        torch = self._torch
        model = self._clip_model
        processor = self._clip_processor
        assert torch is not None and model is not None and processor is not None

        device = model.device
        inputs = processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            feats = model.get_text_features(**inputs)
        feats = _extract_embedding_tensor(torch, feats)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feats.detach().to("cpu").numpy().astype(np.float32)

    def _clip_image_features(self, images_rgb_u8: List[np.ndarray]) -> np.ndarray:
        self.ensure_loaded()
        torch = self._torch
        model = self._clip_model
        processor = self._clip_processor
        assert torch is not None and model is not None and processor is not None

        from PIL import Image

        device = model.device
        pil_imgs = [Image.fromarray(im, mode="RGB") for im in images_rgb_u8]
        inputs = processor(images=pil_imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            feats = model.get_image_features(**inputs)
        feats = _extract_embedding_tensor(torch, feats)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feats.detach().to("cpu").numpy().astype(np.float32)

    def predict_dataset_ids(self, rgb_u8: np.ndarray, cm: ClassMap) -> np.ndarray:
        """Return label map (H,W) uint8 of dataset ids or ignore_id."""
        if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
            raise ValueError("rgb_u8 must be HxWx3")

        h, w = rgb_u8.shape[:2]
        img_area = float(h * w)
        min_area = max(1.0, img_area * float(self.min_area_ratio))

        masks, scores = self._predict_instance_masks(rgb_u8)
        if not masks:
            out = np.full((h, w), int(cm.unmapped), dtype=np.uint8)
            if cm.unmapped == cm.ignore_id:
                out[:] = np.uint8(cm.ignore_id)
            return out

        # Filter + sort masks
        kept: List[Tuple[np.ndarray, float, float]] = []  # (mask, score, area)
        for m, s in zip(masks, scores):
            area = float(np.count_nonzero(m))
            if area < min_area:
                continue
            kept.append((m, float(s), area))

        if not kept:
            out = np.full((h, w), int(cm.unmapped), dtype=np.uint8)
            if cm.unmapped == cm.ignore_id:
                out[:] = np.uint8(cm.ignore_id)
            return out

        kept.sort(key=lambda t: (t[1] * t[2]), reverse=True)
        kept = kept[: int(self.max_masks)]

        prompts, cls_to_idxs = self._build_text_index(cm)
        text_feats = self._clip_text_features(prompts)  # (P, D)

        crops: List[np.ndarray] = []
        crop_masks: List[np.ndarray] = []
        crop_bboxes: List[Tuple[int, int, int, int]] = []

        for m, _s, _a in kept:
            bbox = _mask_bbox(m)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            crop = rgb_u8[y0:y1, x0:x1].copy()
            cmask = m[y0:y1, x0:x1]
            crop[~cmask] = 0
            crops.append(crop)
            crop_masks.append(m)
            crop_bboxes.append(bbox)

        if not crops:
            out = np.full((h, w), int(cm.unmapped), dtype=np.uint8)
            if cm.unmapped == cm.ignore_id:
                out[:] = np.uint8(cm.ignore_id)
            return out

        # Batch CLIP image features
        img_feats = self._clip_image_features(crops)  # (N, D)

        sims = img_feats @ text_feats.T  # (N, P)

        # Build semantic label map
        label = np.full((h, w), int(cm.ignore_id), dtype=np.uint8)

        for i in range(sims.shape[0]):
            best_cls: Optional[int] = None
            best_sim = -1e9

            for cls_id, idxs in cls_to_idxs.items():
                v = float(np.max(sims[i, idxs]))
                if v > best_sim:
                    best_sim = v
                    best_cls = int(cls_id)

            if best_cls is None:
                continue

            m = crop_masks[i]
            label[(label == int(cm.ignore_id)) & m] = np.uint8(best_cls)

        if cm.unmapped != cm.ignore_id:
            label[label == int(cm.ignore_id)] = np.uint8(int(cm.unmapped))

        return label
