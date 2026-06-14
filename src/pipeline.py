from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.dataset.writer import CocoDatasetWriter
from src.segmentation.class_map import ClassMap
from src.segmentation.mask2former import Mask2FormerADEEngine
from src.segmentation.palette import default_palette
from src.utils.logging import Logger
from src.v360 import V360Projector, ViewSpec


@dataclass(frozen=True)
class ExtractConfig:
    fov: float  # 90, 120, or 150
    out_size: int
    yaw_offset: float  # slider value in [-180, 180]
    up_pitch_deg: float
    down_pitch_deg: float
    seg_stride_px: int

    use_up_4: bool
    use_up_6: bool
    use_top: bool
    use_h_4: bool
    use_h_6: bool
    use_down_4: bool
    use_down_6: bool


def build_view_specs(cfg: ExtractConfig) -> Tuple[List[ViewSpec], List[ViewSpec], List[ViewSpec]]:
    """Returns (up_row, mid_row, down_row) specs."""
    yaw_center = (float(cfg.yaw_offset) + 180.0) % 360.0
    up_pitch = float(cfg.up_pitch_deg)
    down_pitch = float(cfg.down_pitch_deg)

    def angle_token(v: float) -> str:
        x = abs(float(v))
        if abs(x - round(x)) < 1e-6:
            return str(int(round(x)))
        return f"{x:.1f}".replace(".", "p")

    up_tag = f"up{angle_token(up_pitch)}"
    down_tag = f"down{angle_token(down_pitch)}"

    def yaws(n: int) -> List[float]:
        step = 360.0 / float(n)
        return [(yaw_center + i * step) % 360.0 for i in range(n)]

    up: List[ViewSpec] = []
    mid: List[ViewSpec] = []
    down: List[ViewSpec] = []

    if cfg.use_up_4:
        for i, yaw in enumerate(yaws(4)):
            up.append(ViewSpec(name=f"{up_tag}_4_{i}", yaw=yaw, pitch=up_pitch))
    if cfg.use_up_6:
        for i, yaw in enumerate(yaws(6)):
            up.append(ViewSpec(name=f"{up_tag}_6_{i}", yaw=yaw, pitch=up_pitch))
    if cfg.use_top:
        up.append(ViewSpec(name="top", yaw=yaw_center, pitch=+90.0))

    if cfg.use_h_4:
        for i, yaw in enumerate(yaws(4)):
            mid.append(ViewSpec(name=f"h_4_{i}", yaw=yaw, pitch=0.0))
    if cfg.use_h_6:
        for i, yaw in enumerate(yaws(6)):
            mid.append(ViewSpec(name=f"h_6_{i}", yaw=yaw, pitch=0.0))

    if cfg.use_down_4:
        for i, yaw in enumerate(yaws(4)):
            down.append(ViewSpec(name=f"{down_tag}_4_{i}", yaw=yaw, pitch=down_pitch))
    if cfg.use_down_6:
        for i, yaw in enumerate(yaws(6)):
            down.append(ViewSpec(name=f"{down_tag}_6_{i}", yaw=yaw, pitch=down_pitch))

    return up, mid, down


def resize_equirect_for_speed(bgr: np.ndarray, out_size: int) -> np.ndarray:
    """Resize equirect image to (2*out_size, out_size) for fast processing."""
    target_h = int(out_size)
    target_w = int(out_size) * 2
    if bgr.shape[0] == target_h and bgr.shape[1] == target_w:
        return bgr
    return cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)


def overlay_segmentation(
    rgb: np.ndarray,
    label: np.ndarray,
    alpha: float = 0.45,
    palette_by_class: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    max_id = int(label[label != 255].max()) if np.any(label != 255) else 0
    palette = default_palette(max_id + 1)
    out = rgb.copy()
    color = np.zeros_like(out)

    class_ids = np.unique(label)
    for cls_id_raw in class_ids.tolist():
        cls_id = int(cls_id_raw)
        if cls_id == 255 or cls_id < 0:
            continue
        mask = label == np.uint8(cls_id)
        if not np.any(mask):
            continue
        if palette_by_class is not None and cls_id in palette_by_class:
            r, g, b = palette_by_class[cls_id]
        elif cls_id < len(palette):
            r, g, b = palette[cls_id]
        else:
            ext = default_palette(cls_id + 1)
            r, g, b = ext[cls_id]
        color[mask] = (r, g, b)

    ignore_mask = label == 255
    color[ignore_mask] = (0, 0, 0)

    out = (out.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return out


@dataclass
class PreviewResult:
    preview1_rgb: np.ndarray
    preview1_time_s: Optional[float]
    up_tiles_rgb: List[np.ndarray]
    mid_tiles_rgb: List[np.ndarray]
    down_tiles_rgb: List[np.ndarray]
    up_tiles_seg: List[np.ndarray]
    mid_tiles_seg: List[np.ndarray]
    down_tiles_seg: List[np.ndarray]
    up_tiles_reports: List[Dict[str, str]]
    mid_tiles_reports: List[Dict[str, str]]
    down_tiles_reports: List[Dict[str, str]]
    up_tile_names: List[str]
    mid_tile_names: List[str]
    down_tile_names: List[str]


class GeneratorPipeline:
    def __init__(self, *, ffmpeg: str = "ffmpeg", logger: Optional[Logger] = None) -> None:
        self.logger = logger or Logger()
        self.projector = V360Projector(ffmpeg=ffmpeg)
        self.engine = Mask2FormerADEEngine()
        try:
            infer_batch = int(os.environ.get("MASK2DATASET_INFER_BATCH", "4"))
        except Exception:
            infer_batch = 4
        self.infer_batch_size = max(1, infer_batch)

        polygon_backend = str(os.environ.get("MASK2DATASET_POLYGON_BACKEND", "fast")).strip().lower()
        self.polygon_backend = polygon_backend if polygon_backend in ("fast", "topology") else "fast"
        self.class_map_path = Path(__file__).resolve().parent.parent / "config" / "new_class_map.yaml"
        self.palette_by_class: Dict[int, Tuple[int, int, int]] = {}
        self._class_map: Optional[ClassMap] = None
        self._ade_to_dataset_lut: Optional[np.ndarray] = None

    def set_palette_overrides(self, palette_by_class: Dict[int, Tuple[int, int, int]]) -> None:
        self.palette_by_class = {
            int(k): (int(v[0]), int(v[1]), int(v[2]))
            for k, v in palette_by_class.items()
            if 0 <= int(k) <= 254 and isinstance(v, (tuple, list)) and len(v) == 3
        }

    @staticmethod
    def _normalize_label_name(name: str) -> str:
        return str(name).strip().lower()

    @staticmethod
    def _dataset_id_by_name(cm: ClassMap, name: str, default: int) -> int:
        want = GeneratorPipeline._normalize_label_name(name)
        for dataset_id, dataset_name in cm.id_to_name.items():
            if GeneratorPipeline._normalize_label_name(dataset_name) == want:
                return int(dataset_id)
        return int(default)

    @staticmethod
    def _format_target_report(
        *,
        ade: np.ndarray,
        lbl: np.ndarray,
        id2label: Dict[int, str],
        target_id: int,
        target_name: str,
        topk: int = 12,
    ) -> str:
        if ade.shape != lbl.shape:
            return f"[warn] {target_name} report: shape mismatch"

        target_id_u8 = np.uint8(int(target_id))
        mask = lbl == target_id_u8
        total = int(lbl.size)
        cnt = int(mask.sum())
        if total <= 0:
            return f"[info] {target_name}: 0/0"
        if cnt <= 0:
            return f"[info] {target_name}: 0 (0.00%)"

        ade_unl = ade[mask]
        counts: Dict[int, int] = {}
        for x in ade_unl.reshape(-1):
            xi = int(x)
            counts[xi] = counts.get(xi, 0) + 1

        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(topk))]
        ratio = cnt / float(total)

        lines: List[str] = []
        lines.append(f"{target_name}: {cnt}/{total} ({ratio:.2%})")
        for ade_id, c in items:
            name = id2label.get(int(ade_id), "<unknown>")
            pct = c / float(cnt)
            lines.append(f"- {name} (ade_id={ade_id}): {c} ({pct:.2%})")
        return "\n".join(lines)

    def _load_class_map(self) -> ClassMap:
        self.engine.ensure_loaded()
        id2label = self.engine.id2label
        path = self.class_map_path
        if not path.exists():
            raise FileNotFoundError(f"class map yaml not found: {path}")
        cm = ClassMap.from_yaml(path, id2label)
        self._class_map = cm
        self._ade_to_dataset_lut = self._build_ade_to_dataset_lut(cm=cm, id2label=id2label)
        self.logger.log(f"Loaded class_map: {cm.summarize()}")
        return cm

    def reload_class_map(self) -> ClassMap:
        self._class_map = None
        return self._load_class_map()

    def _ensure_class_map(self, *, force_reload: bool = False) -> ClassMap:
        if force_reload or self._class_map is None:
            return self._load_class_map()
        return self._class_map

    @staticmethod
    def _build_ade_to_dataset_lut(cm: ClassMap, id2label: Dict[int, str]) -> np.ndarray:
        max_ade = 0
        if id2label:
            max_ade = max(max_ade, max(int(k) for k in id2label.keys()))
        for ade_id in cm.ade_id_to_dataset_id.keys():
            if int(ade_id) >= 0:
                max_ade = max(max_ade, int(ade_id))

        unmapped = int(cm.ade_id_to_dataset_id.get(-1, 255))
        lut = np.full((max_ade + 1,), unmapped, dtype=np.uint8)
        for ade_id, dataset_id in cm.ade_id_to_dataset_id.items():
            if int(ade_id) < 0:
                continue
            if int(ade_id) < lut.shape[0]:
                lut[int(ade_id)] = np.uint8(int(dataset_id))
        return lut

    @staticmethod
    def _remap_ade_to_dataset_ids(ade: np.ndarray, cm: ClassMap, lut: Optional[np.ndarray] = None) -> np.ndarray:
        unmapped = int(cm.ade_id_to_dataset_id.get(-1, 255))
        if lut is None:
            out = np.full(ade.shape, unmapped, dtype=np.uint8)
            for ade_id, dataset_id in cm.ade_id_to_dataset_id.items():
                if ade_id < 0:
                    continue
                out[ade == int(ade_id)] = np.uint8(int(dataset_id))
            return out

        out = np.full(ade.shape, unmapped, dtype=np.uint8)
        valid = (ade >= 0) & (ade < int(lut.shape[0]))
        if np.any(valid):
            out[valid] = lut[ade[valid]]
        return out

    def _predict_ade_ids_scaled(self, rgb_u8: np.ndarray, seg_stride_px: int) -> np.ndarray:
        """Predict ADE ids at reduced resolution and restore to original size with nearest upsampling."""
        h, w = rgb_u8.shape[:2]
        stride = max(1, int(seg_stride_px))
        if stride <= 1:
            return self.engine.predict_ade_ids(rgb_u8)

        h_s = max(1, int(round(h / float(stride))))
        w_s = max(1, int(round(w / float(stride))))
        small = cv2.resize(rgb_u8, (w_s, h_s), interpolation=cv2.INTER_AREA)
        ade_small = self.engine.predict_ade_ids(small)
        ade_full = cv2.resize(ade_small.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
        return ade_full.astype(np.int32)

    def _predict_ade_ids_scaled_batch(self, rgb_u8_list: Sequence[np.ndarray], seg_stride_px: int) -> List[np.ndarray]:
        if not rgb_u8_list:
            return []

        stride = max(1, int(seg_stride_px))
        if stride <= 1:
            return self.engine.predict_ade_ids_batch(rgb_u8_list, batch_size=self.infer_batch_size)

        resized: List[np.ndarray] = []
        original_hw: List[Tuple[int, int]] = []
        for rgb_u8 in rgb_u8_list:
            h, w = rgb_u8.shape[:2]
            original_hw.append((h, w))
            h_s = max(1, int(round(h / float(stride))))
            w_s = max(1, int(round(w / float(stride))))
            small = cv2.resize(rgb_u8, (w_s, h_s), interpolation=cv2.INTER_AREA)
            resized.append(small)

        ade_small_list = self.engine.predict_ade_ids_batch(resized, batch_size=self.infer_batch_size)
        out: List[np.ndarray] = []
        for ade_small, (h, w) in zip(ade_small_list, original_hw):
            ade_full = cv2.resize(ade_small.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
            out.append(ade_full.astype(np.int32))
        return out

    def _project_and_segment_group(
        self,
        *,
        pano_rgb: np.ndarray,
        specs: Sequence[ViewSpec],
        cfg: ExtractConfig,
        cm: ClassMap,
        build_seg: bool,
        build_reports: bool,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[Dict[str, str]], List[str]]:
        if not specs:
            return [], [], [], [], []

        if should_stop is not None and should_stop():
            return [], [], [], [], []

        rgbs = self.projector.project_many_rgb_from_array(
            pano_rgb,
            specs,
            out_size=cfg.out_size,
            fov=cfg.fov,
        )
        ade_list = self._predict_ade_ids_scaled_batch(rgbs, cfg.seg_stride_px)
        lut = self._ade_to_dataset_lut

        labels: List[np.ndarray] = []
        segs: List[np.ndarray] = []
        reports: List[Dict[str, str]] = []
        names: List[str] = []

        id2label: Dict[int, str] = self.engine.id2label if build_reports else {}
        for spec, rgb, ade in zip(specs, rgbs, ade_list):
            if should_stop is not None and should_stop():
                break
            lbl = self._remap_ade_to_dataset_ids(ade, cm, lut=lut)
            labels.append(lbl)
            if build_seg:
                segs.append(overlay_segmentation(rgb, lbl, palette_by_class=self.palette_by_class))
            if build_reports:
                report_by_class: Dict[str, str] = {}
                for class_id, class_name in sorted(cm.id_to_name.items()):
                    report_by_class[str(class_id)] = self._format_target_report(
                        ade=ade,
                        lbl=lbl,
                        id2label=id2label,
                        target_id=class_id,
                        target_name=f"{class_id}: {class_name}",
                    )
                reports.append(report_by_class)
            names.append(spec.name)
        return rgbs, labels, segs, reports, names

    @staticmethod
    def _filter_specs_by_names(specs: Sequence[ViewSpec], include_names: Optional[Sequence[str]]) -> List[ViewSpec]:
        if include_names is None:
            return list(specs)
        want = set(str(x) for x in include_names)
        return [s for s in specs if s.name in want]

    def build_preview(
        self,
        *,
        input_bgr: np.ndarray,
        preview_time_s: Optional[float],
        cfg: ExtractConfig,
        reload_class_map: bool = False,
    ) -> PreviewResult:
        cm = self._ensure_class_map(force_reload=reload_class_map)

        # Downscale equirect early for speed (also becomes the actual dataset basis).
        pano_bgr = resize_equirect_for_speed(input_bgr, cfg.out_size)
        pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)

        up_specs, mid_specs, down_specs = build_view_specs(cfg)

        up_rgb, _, up_seg, up_rep, up_names = self._project_and_segment_group(
            pano_rgb=pano_rgb,
            specs=up_specs,
            cfg=cfg,
            cm=cm,
            build_seg=True,
            build_reports=True,
        )
        mid_rgb, _, mid_seg, mid_rep, mid_names = self._project_and_segment_group(
            pano_rgb=pano_rgb,
            specs=mid_specs,
            cfg=cfg,
            cm=cm,
            build_seg=True,
            build_reports=True,
        )
        down_rgb, _, down_seg, down_rep, down_names = self._project_and_segment_group(
            pano_rgb=pano_rgb,
            specs=down_specs,
            cfg=cfg,
            cm=cm,
            build_seg=True,
            build_reports=True,
        )

        return PreviewResult(
            preview1_rgb=pano_rgb,
            preview1_time_s=preview_time_s,
            up_tiles_rgb=up_rgb,
            mid_tiles_rgb=mid_rgb,
            down_tiles_rgb=down_rgb,
            up_tiles_seg=up_seg,
            mid_tiles_seg=mid_seg,
            down_tiles_seg=down_seg,
            up_tiles_reports=up_rep,
            mid_tiles_reports=mid_rep,
            down_tiles_reports=down_rep,
            up_tile_names=up_names,
            mid_tile_names=mid_names,
            down_tile_names=down_names,
        )

    def generate_dataset_from_images(
        self,
        *,
        image_paths: Sequence[Path],
        output_root: Path,
        cfg: ExtractConfig,
        include_spec_names: Optional[Sequence[str]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        cm = self._ensure_class_map()
        writer = CocoDatasetWriter(
            root=output_root,
            category_names_by_dataset_id=cm.id_to_name,
            ignore_id=cm.ignore_id,
            simplify_epsilon_px=max(0.0, 0.75 * float(max(1, cfg.seg_stride_px) - 1)),
            polygon_backend=self.polygon_backend,
        )
        writer.ensure_dirs()

        rng = random.Random(writer.seed)

        up_specs, mid_specs, down_specs = build_view_specs(cfg)
        all_specs = list(up_specs) + list(mid_specs) + list(down_specs)
        all_specs = self._filter_specs_by_names(all_specs, include_spec_names)
        if not all_specs:
            raise ValueError("no enabled view directions to generate")

        self.logger.log(f"Generate from images: count={len(image_paths)} views={len(all_specs)}")

        cancelled = False
        for src_idx, path in enumerate(image_paths):
            if should_stop is not None and should_stop():
                cancelled = True
                break

            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                self.logger.log(f"skip unreadable image: {path}")
                continue

            pano_bgr = resize_equirect_for_speed(bgr, cfg.out_size)
            pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)

            base = path.stem

            # Choose split per source image to avoid leakage across train/val.
            split = writer.choose_split(rng.random())

            rgb_tiles, lbl_tiles, _, _, _ = self._project_and_segment_group(
                pano_rgb=pano_rgb,
                specs=all_specs,
                cfg=cfg,
                cm=cm,
                build_seg=False,
                build_reports=False,
                should_stop=should_stop,
            )

            for spec, rgb, lbl in zip(all_specs, rgb_tiles, lbl_tiles):
                if should_stop is not None and should_stop():
                    cancelled = True
                    break
                filename = f"{base}_{src_idx:06d}_{spec.name}.png"
                writer.add_sample(split, filename, rgb, lbl)

            if cancelled:
                break

            if (src_idx + 1) % 5 == 0:
                self.logger.log(f"processed {src_idx+1}/{len(image_paths)}")

        writer.save_annotations()
        if cancelled:
            self.logger.log("cancelled")
        else:
            self.logger.log("done")

    def generate_dataset_from_video(
        self,
        *,
        video_path: Path,
        output_root: Path,
        fps: float,
        cfg: ExtractConfig,
        include_spec_names: Optional[Sequence[str]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        cm = self._ensure_class_map()
        writer = CocoDatasetWriter(
            root=output_root,
            category_names_by_dataset_id=cm.id_to_name,
            ignore_id=cm.ignore_id,
            simplify_epsilon_px=max(0.0, 0.75 * float(max(1, cfg.seg_stride_px) - 1)),
            polygon_backend=self.polygon_backend,
        )
        writer.ensure_dirs()

        rng = random.Random(writer.seed)

        up_specs, mid_specs, down_specs = build_view_specs(cfg)
        all_specs = list(up_specs) + list(mid_specs) + list(down_specs)
        all_specs = self._filter_specs_by_names(all_specs, include_spec_names)
        if not all_specs:
            raise ValueError("no enabled view directions to generate")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 0:
            src_fps = 30.0

        step_s = 1.0 / max(1e-6, float(fps))
        next_t = 0.0
        frame_idx = 0
        saved_idx = 0

        self.logger.log(f"Generate from video: fps={fps} step={step_s:.3f}s views={len(all_specs)}")

        cancelled = False
        try:
            while True:
                if should_stop is not None and should_stop():
                    cancelled = True
                    break

                ok, frame = cap.read()
                if not ok:
                    break
                t = frame_idx / src_fps
                frame_idx += 1
                if t + 1e-6 < next_t:
                    continue
                next_t += step_s

                # Choose split per source frame to avoid leakage across train/val.
                split = writer.choose_split(rng.random())

                pano_bgr = resize_equirect_for_speed(frame, cfg.out_size)
                pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)

                rgb_tiles, lbl_tiles, _, _, _ = self._project_and_segment_group(
                    pano_rgb=pano_rgb,
                    specs=all_specs,
                    cfg=cfg,
                    cm=cm,
                    build_seg=False,
                    build_reports=False,
                    should_stop=should_stop,
                )

                for spec, rgb, lbl in zip(all_specs, rgb_tiles, lbl_tiles):
                    if should_stop is not None and should_stop():
                        cancelled = True
                        break
                    filename = f"frame_{saved_idx:06d}_{spec.name}.png"
                    writer.add_sample(split, filename, rgb, lbl)

                if cancelled:
                    break

                saved_idx += 1
                if saved_idx % 5 == 0:
                    self.logger.log(f"saved frames: {saved_idx}")
        finally:
            cap.release()

        writer.save_annotations()
        if cancelled:
            self.logger.log("cancelled")
        else:
            self.logger.log("done")
