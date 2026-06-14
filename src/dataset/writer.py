from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image

try:
    from shapely import coverage_simplify
    from shapely.ops import unary_union
    from shapely.geometry import MultiPolygon, Polygon

    _HAS_SHAPELY = True
except Exception:
    coverage_simplify = None
    unary_union = None
    MultiPolygon = None
    Polygon = None
    _HAS_SHAPELY = False


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
    simplify_epsilon_px: float = 0.0
    polygon_backend: str = "fast"  # fast | topology
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
        backend = str(self.polygon_backend).strip().lower()
        if backend not in ("fast", "topology"):
            self.polygon_backend = "fast"
        else:
            self.polygon_backend = backend

    def ensure_dirs(self) -> None:
        (self.root / "images").mkdir(parents=True, exist_ok=True)

    def choose_split(self, u: float) -> str:
        return "val" if u < float(self.val_ratio) else "train"

    def save_image(self, split: str, filename: str, rgb_u8: np.ndarray) -> Path:
        out = self.root / "images" / filename
        Image.fromarray(rgb_u8, mode="RGB").save(out)
        return out

    def _shape_obj(self, dataset_id: int, points: List[List[float]]) -> Dict[str, object]:
        return {
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

    def _build_shapes_with_opencv(self, label_u8: np.ndarray) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        eps = max(0.0, float(self.simplify_epsilon_px))
        present_ids = set(int(x) for x in np.unique(label_u8).tolist())

        for dataset_id in sorted(self._cat_id_by_dataset_id.keys()):
            if int(dataset_id) not in present_ids:
                continue
            cls_mask = (label_u8 == np.uint8(dataset_id)).astype(np.uint8)
            if int(np.count_nonzero(cls_mask)) == 0:
                continue

            contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cnt is None or len(cnt) < 3:
                    continue
                raw_cnt = cnt
                if eps > 0.0:
                    cnt = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
                if cnt is None or len(cnt) < 3:
                    cnt = raw_cnt
                if cnt is None or len(cnt) < 3:
                    continue

                points: List[List[float]] = []
                for p in cnt[:, 0, :].tolist():
                    x, y = int(p[0]), int(p[1])
                    points.append([float(x), float(y)])
                if len(points) < 3:
                    continue
                out.append(self._shape_obj(dataset_id, points))

        return out

    @staticmethod
    def _row_runs(mask_row: np.ndarray) -> List[tuple[int, int]]:
        runs: List[tuple[int, int]] = []
        start = -1
        for x, v in enumerate(mask_row.tolist()):
            on = bool(v)
            if on and start < 0:
                start = x
            elif not on and start >= 0:
                runs.append((start, x - 1))
                start = -1
        if start >= 0:
            runs.append((start, int(mask_row.shape[0]) - 1))
        return runs

    def _rect_decompose_mask(self, mask_u8: np.ndarray) -> List[tuple[int, int, int, int]]:
        h, _w = mask_u8.shape[:2]
        active: Dict[tuple[int, int], List[int]] = {}
        out: List[tuple[int, int, int, int]] = []

        for y in range(h):
            runs = self._row_runs(mask_u8[y])
            run_set = set(runs)

            for key in list(active.keys()):
                if key not in run_set:
                    x0, x1, y0, y1 = active.pop(key)
                    out.append((x0, y0, x1 + 1, y1 + 1))

            for run in runs:
                if run in active:
                    active[run][3] = y
                else:
                    x0, x1 = run
                    active[run] = [x0, x1, y, y]

        for x0, x1, y0, y1 in active.values():
            out.append((x0, y0, x1 + 1, y1 + 1))

        return out

    def _build_shapes_with_shared_topology(self, label_u8: np.ndarray) -> List[Dict[str, object]]:
        if not _HAS_SHAPELY:
            return self._build_shapes_with_opencv(label_u8)

        labeled_geoms: List[tuple[int, object]] = []
        for dataset_id in sorted(self._cat_id_by_dataset_id.keys()):
            cls_mask = (label_u8 == np.uint8(dataset_id)).astype(np.uint8)
            if int(np.count_nonzero(cls_mask)) == 0:
                continue

            rects = self._rect_decompose_mask(cls_mask)
            if not rects:
                continue
            boxes = [Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]) for x0, y0, x1, y1 in rects]
            geom = unary_union(boxes)
            if geom is None or geom.is_empty:
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
                if geom.is_empty:
                    continue

            if isinstance(geom, MultiPolygon):
                for g in geom.geoms:
                    if not g.is_empty and g.area > 0:
                        labeled_geoms.append((dataset_id, g))
            else:
                if geom.area > 0:
                    labeled_geoms.append((dataset_id, geom))

        if not labeled_geoms:
            return []

        ids = [x[0] for x in labeled_geoms]
        geoms = [x[1] for x in labeled_geoms]
        eps = max(0.0, float(self.simplify_epsilon_px))
        if eps > 0.0:
            geoms = list(coverage_simplify(geoms, eps))

        out: List[Dict[str, object]] = []
        for dataset_id, geom in zip(ids, geoms):
            if geom is None or geom.is_empty:
                continue
            if isinstance(geom, MultiPolygon):
                poly_iter = list(geom.geoms)
            else:
                poly_iter = [geom]

            for poly in poly_iter:
                if poly.is_empty or poly.area <= 0:
                    continue
                coords = list(poly.exterior.coords)
                if len(coords) < 4:
                    continue
                coords = coords[:-1]
                if len(coords) < 3:
                    continue
                points = [[float(x), float(y)] for x, y in coords]
                out.append(self._shape_obj(dataset_id, points))

        return out

    def _build_shapes_for_label(self, label_u8: np.ndarray) -> List[Dict[str, object]]:
        if self.polygon_backend == "topology":
            return self._build_shapes_with_shared_topology(label_u8)
        return self._build_shapes_with_opencv(label_u8)

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
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

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
