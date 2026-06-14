from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MMSegDatasetWriter:
    root: Path
    val_ratio: float = 0.2
    seed: int = 42

    def ensure_dirs(self) -> None:
        for split in ("train", "val"):
            (self.root / "img_dir" / split).mkdir(parents=True, exist_ok=True)
            (self.root / "ann_dir" / split).mkdir(parents=True, exist_ok=True)

    def choose_split(self, u: float) -> str:
        return "val" if u < float(self.val_ratio) else "train"

    def save_image(self, split: str, filename: str, rgb_u8: np.ndarray) -> Path:
        out = self.root / "img_dir" / split / filename
        Image.fromarray(rgb_u8, mode="RGB").save(out)
        return out

    def save_label(self, split: str, filename: str, label_u8: np.ndarray) -> Path:
        out = self.root / "ann_dir" / split / filename
        if label_u8.dtype != np.uint8:
            raise ValueError("label must be uint8")
        # IMPORTANT: Save as 8-bit grayscale where pixel values are class ids (0..7) or 255.
        Image.fromarray(label_u8, mode="L").save(out)
        return out


@dataclass
class CocoDatasetWriter:
    root: Path
    category_names_by_dataset_id: Dict[int, str]
    ignore_id: int = 255
    val_ratio: float = 0.2
    seed: int = 42

    _cat_id_by_dataset_id: Dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        dataset_ids = sorted(
            int(x)
            for x in self.category_names_by_dataset_id.keys()
            if 0 <= int(x) <= 254 and int(x) != int(self.ignore_id)
        )
        self._cat_id_by_dataset_id = {did: idx + 1 for idx, did in enumerate(dataset_ids)}

    def ensure_dirs(self) -> None:
        (self.root / "images").mkdir(parents=True, exist_ok=True)

    def choose_split(self, u: float) -> str:
        return "val" if u < float(self.val_ratio) else "train"

    def save_image(self, split: str, filename: str, rgb_u8: np.ndarray) -> Path:
        out = self.root / "images" / filename
        Image.fromarray(rgb_u8, mode="RGB").save(out)
        return out

    def _build_shapes_for_label(self, label_u8: np.ndarray) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for dataset_id in self._cat_id_by_dataset_id.keys():
            cls_mask = (label_u8 == np.uint8(dataset_id)).astype(np.uint8)
            if cls_mask.max() == 0:
                continue

            contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cnt is None or len(cnt) < 3:
                    continue

                area = float(cv2.contourArea(cnt))
                if area <= 0.0:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                pts = cnt.reshape(-1, 2)
                if pts.shape[0] < 3:
                    continue

                points = [[float(p[0]), float(p[1])] for p in pts.tolist()]
                out.append(
                    {
                        "kie_linking": [],
                        "label": str(self.category_names_by_dataset_id.get(dataset_id, f"class_{dataset_id}")),
                        "score": None,
                        "points": points,
                        "group_id": None,
                        "description": "",
                        "difficult": False,
                        "shape_type": "polygon",
                        "flags": {},
                        "attributes": {},
                    }
                )
        return out

    def add_sample(self, split: str, filename: str, rgb_u8: np.ndarray, label_u8: np.ndarray) -> None:
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        if label_u8.dtype != np.uint8:
            raise ValueError("label must be uint8")
        if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
            raise ValueError("rgb image must be HxWx3")
        if rgb_u8.shape[:2] != label_u8.shape:
            raise ValueError("rgb and label shape mismatch")

        out_path = self.save_image(split, filename, rgb_u8)
        h, w = int(rgb_u8.shape[0]), int(rgb_u8.shape[1])
        shapes = self._build_shapes_for_label(label_u8=label_u8)
        payload = {
            "version": "3.3.10",
            "flags": {},
            "shapes": shapes,
            "imagePath": out_path.name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
            "description": "",
        }

        json_path = out_path.with_suffix(".json")
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _categories(self) -> List[Dict[str, object]]:
        cats: List[Dict[str, object]] = []
        for dataset_id in sorted(self._cat_id_by_dataset_id.keys()):
            cats.append(
                {
                    "id": int(self._cat_id_by_dataset_id[dataset_id]),
                    "name": str(self.category_names_by_dataset_id.get(dataset_id, f"class_{dataset_id}")),
                    "supercategory": "object",
                    "dataset_id": int(dataset_id),
                }
            )
        return cats

    def save_annotations(self) -> None:
        # Kept for backward compatibility with existing pipeline calls.
        return
