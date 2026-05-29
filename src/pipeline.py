from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from src.dataset.writer import MMSegDatasetWriter
from src.segmentation.class_map import ClassMap
from src.segmentation.mask2former import Mask2FormerADEEngine
from src.segmentation.deeplab_cityscapes import DeepLabCityscapesEngine
from src.segmentation.palette import default_palette_8
from src.utils.logging import Logger
from src.v360 import V360Projector, ViewSpec


@dataclass(frozen=True)
class ExtractConfig:
    fov: float  # 90 or 120
    out_size: int
    yaw_offset: float  # slider value in [-180, 180]

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

    def yaws(n: int) -> List[float]:
        step = 360.0 / float(n)
        return [(yaw_center + i * step) % 360.0 for i in range(n)]

    up: List[ViewSpec] = []
    mid: List[ViewSpec] = []
    down: List[ViewSpec] = []

    if cfg.use_up_4:
        for i, yaw in enumerate(yaws(4)):
            up.append(ViewSpec(name=f"up45_4_{i}", yaw=yaw, pitch=+45.0))
    if cfg.use_up_6:
        for i, yaw in enumerate(yaws(6)):
            up.append(ViewSpec(name=f"up45_6_{i}", yaw=yaw, pitch=+45.0))
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
            down.append(ViewSpec(name=f"down45_4_{i}", yaw=yaw, pitch=-45.0))
    if cfg.use_down_6:
        for i, yaw in enumerate(yaws(6)):
            down.append(ViewSpec(name=f"down45_6_{i}", yaw=yaw, pitch=-45.0))

    return up, mid, down


def resize_equirect_for_speed(bgr: np.ndarray, out_size: int) -> np.ndarray:
    """Resize equirect image to (2*out_size, out_size) for fast processing."""
    target_h = int(out_size)
    target_w = int(out_size) * 2
    if bgr.shape[0] == target_h and bgr.shape[1] == target_w:
        return bgr
    return cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)


def overlay_segmentation(rgb: np.ndarray, label: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    palette = default_palette_8()
    out = rgb.copy()
    color = np.zeros_like(out)

    for cls_id in range(8):
        mask = label == cls_id
        if not np.any(mask):
            continue
        r, g, b = palette[cls_id]
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


class GeneratorPipeline:
    def __init__(self, *, ffmpeg: str = "ffmpeg", logger: Optional[Logger] = None) -> None:
        self.logger = logger or Logger()
        self.projector = V360Projector(ffmpeg=ffmpeg)
        # Choose engine: if a Cityscapes class_map exists, use DeepLab Cityscapes engine and mapping.
        repo_root = Path(__file__).resolve().parent.parent
        cs_map = repo_root / "config" / "class_map_cityscapes.yaml"
        if cs_map.exists():
            self.engine = DeepLabCityscapesEngine()
            self.class_map_path = cs_map
        else:
            self.engine = Mask2FormerADEEngine()
            self.class_map_path = repo_root / "config" / "class_map.yaml"
        self._class_map: Optional[ClassMap] = None

    def _load_class_map(self) -> ClassMap:
        self.engine.ensure_loaded()
        id2label = self.engine.id2label
        path = self.class_map_path
        if not path.exists():
            raise FileNotFoundError(f"class map yaml not found: {path}")
        cm = ClassMap.from_yaml(path, id2label)
        self._class_map = cm
        self.logger.log(f"Loaded class_map: {cm.summarize()}")
        return cm

    def _ensure_class_map(self) -> ClassMap:
        return self._class_map or self._load_class_map()

    def build_preview(
        self,
        *,
        input_bgr: np.ndarray,
        preview_time_s: Optional[float],
        cfg: ExtractConfig,
    ) -> PreviewResult:
        cm = self._ensure_class_map()

        # Downscale equirect early for speed (also becomes the actual dataset basis).
        pano_bgr = resize_equirect_for_speed(input_bgr, cfg.out_size)
        pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)

        # If the engine supports per-tile Cityscapes inference, use tile-first flow.
        tile_first = hasattr(self.engine, "predict_city_ids")

        up_specs, mid_specs, down_specs = build_view_specs(cfg)

        with tempfile.TemporaryDirectory(prefix="v360_preview_") as td:
            td_path = Path(td)
            pano_path = td_path / "pano.png"
            label_path = td_path / "pano_label.png"
            Image.fromarray(pano_rgb, mode="RGB").save(pano_path)
            Image.fromarray(out, mode="L").save(label_path)

            def project_group(specs: Sequence[ViewSpec]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
                if not specs:
                    return [], []

                rgb_outs = [td_path / f"{spec.name}.png" for spec in specs]
                self.projector.project_many_rgb(pano_path, specs, rgb_outs, out_size=cfg.out_size, fov=cfg.fov)

                rgbs: List[np.ndarray] = []
                segs: List[np.ndarray] = []

                if tile_first:
                    # Run per-tile inference and map to dataset ids
                    for rgb_p in rgb_outs:
                        rgb = np.array(Image.open(rgb_p).convert("RGB"), dtype=np.uint8)
                        # engine returns model-specific ids
                        model_ids = self.engine.predict_city_ids(rgb)
                        # Map model ids to dataset ids via ClassMap
                        unmapped = cm.ade_id_to_dataset_id.get(-1, 255)
                        out_lbl = np.full(model_ids.shape, int(unmapped), dtype=np.uint8)
                        for model_id, dataset_id in cm.ade_id_to_dataset_id.items():
                            if model_id < 0:
                                continue
                            out_lbl[model_ids == int(model_id)] = np.uint8(int(dataset_id))
                        rgbs.append(rgb)
                        segs.append(overlay_segmentation(rgb, out_lbl))
                    return rgbs, segs
                else:
                    lbl_outs = [td_path / f"{spec.name}_lbl.png" for spec in specs]
                    self.projector.project_many_label_nearest(label_path, specs, lbl_outs, out_size=cfg.out_size, fov=cfg.fov)

                    for rgb_p, lbl_p in zip(rgb_outs, lbl_outs):
                        rgb = np.array(Image.open(rgb_p).convert("RGB"), dtype=np.uint8)
                        lbl = np.array(Image.open(lbl_p).convert("L"), dtype=np.uint8)
                        rgbs.append(rgb)
                        segs.append(overlay_segmentation(rgb, lbl))
                    return rgbs, segs

            up_rgb, up_seg = project_group(up_specs)
            mid_rgb, mid_seg = project_group(mid_specs)
            down_rgb, down_seg = project_group(down_specs)

        return PreviewResult(
            preview1_rgb=pano_rgb,
            preview1_time_s=preview_time_s,
            up_tiles_rgb=up_rgb,
            mid_tiles_rgb=mid_rgb,
            down_tiles_rgb=down_rgb,
            up_tiles_seg=up_seg,
            mid_tiles_seg=mid_seg,
            down_tiles_seg=down_seg,
        )

    def generate_dataset_from_images(
        self,
        *,
        image_paths: Sequence[Path],
        output_root: Path,
        cfg: ExtractConfig,
    ) -> None:
        cm = self._ensure_class_map()
        writer = MMSegDatasetWriter(root=output_root)
        writer.ensure_dirs()

        rng = random.Random(writer.seed)

        up_specs, mid_specs, down_specs = build_view_specs(cfg)
        all_specs = list(up_specs) + list(mid_specs) + list(down_specs)

        self.logger.log(f"Generate from images: count={len(image_paths)} views={len(all_specs)}")

        for src_idx, path in enumerate(image_paths):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                self.logger.log(f"skip unreadable image: {path}")
                continue

            pano_bgr = resize_equirect_for_speed(bgr, cfg.out_size)
            pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)
            tile_first = hasattr(self.engine, "predict_city_ids")

            if not tile_first:
                ade = self.engine.predict_ade_ids(pano_rgb)

                unmapped = cm.ade_id_to_dataset_id.get(-1, 255)
                pano_label = np.full(ade.shape, int(unmapped), dtype=np.uint8)
                for ade_id, dataset_id in cm.ade_id_to_dataset_id.items():
                    if ade_id < 0:
                        continue
                    pano_label[ade == int(ade_id)] = np.uint8(int(dataset_id))

            base = path.stem

            # Choose split per source image to avoid leakage across train/val.
            split = writer.choose_split(rng.random())

            with tempfile.TemporaryDirectory(prefix="v360_gen_") as td:
                td_path = Path(td)
                pano_path = td_path / "pano.png"
                label_path = td_path / "pano_label.png"
                Image.fromarray(pano_rgb, mode="RGB").save(pano_path)
                Image.fromarray(pano_label, mode="L").save(label_path)

                rgb_outs = [td_path / f"{spec.name}.png" for spec in all_specs]
                self.projector.project_many_rgb(pano_path, all_specs, rgb_outs, out_size=cfg.out_size, fov=cfg.fov)

                for spec, rgb_out in zip(all_specs, rgb_outs):
                    rgb = np.array(Image.open(rgb_out).convert("RGB"), dtype=np.uint8)

                    if tile_first:
                        model_ids = self.engine.predict_city_ids(rgb)
                        unmapped = cm.ade_id_to_dataset_id.get(-1, 255)
                        lbl = np.full(model_ids.shape, int(unmapped), dtype=np.uint8)
                        for model_id, dataset_id in cm.ade_id_to_dataset_id.items():
                            if model_id < 0:
                                continue
                            lbl[model_ids == int(model_id)] = np.uint8(int(dataset_id))
                    else:
                        # fallback: load projected pano labels
                        lbl_out = td_path / f"{spec.name}_lbl.png"
                        lbl = np.array(Image.open(lbl_out).convert("L"), dtype=np.uint8)

                    # Enforce allowed ids
                    bad = (lbl != 255) & (lbl > 7)
                    if np.any(bad):
                        lbl[bad] = 255

                    filename = f"{base}_{src_idx:06d}_{spec.name}.png"
                    writer.save_image(split, filename, rgb)
                    writer.save_label(split, filename, lbl)

            if (src_idx + 1) % 5 == 0:
                self.logger.log(f"processed {src_idx+1}/{len(image_paths)}")

        self.logger.log("done")

    def generate_dataset_from_video(
        self,
        *,
        video_path: Path,
        output_root: Path,
        fps: float,
        cfg: ExtractConfig,
    ) -> None:
        cm = self._ensure_class_map()
        writer = MMSegDatasetWriter(root=output_root)
        writer.ensure_dirs()

        rng = random.Random(writer.seed)

        up_specs, mid_specs, down_specs = build_view_specs(cfg)
        all_specs = list(up_specs) + list(mid_specs) + list(down_specs)

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

        while True:
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
            tile_first = hasattr(self.engine, "predict_city_ids")

            if not tile_first:
                ade = self.engine.predict_ade_ids(pano_rgb)

                unmapped = cm.ade_id_to_dataset_id.get(-1, 255)
                pano_label = np.full(ade.shape, int(unmapped), dtype=np.uint8)
                for ade_id, dataset_id in cm.ade_id_to_dataset_id.items():
                    if ade_id < 0:
                        continue
                    pano_label[ade == int(ade_id)] = np.uint8(int(dataset_id))

            with tempfile.TemporaryDirectory(prefix="v360_vid_") as td:
                td_path = Path(td)
                pano_path = td_path / "pano.png"
                label_path = td_path / "pano_label.png"
                Image.fromarray(pano_rgb, mode="RGB").save(pano_path)
                Image.fromarray(pano_label, mode="L").save(label_path)

                rgb_outs = [td_path / f"{spec.name}.png" for spec in all_specs]
                self.projector.project_many_rgb(pano_path, all_specs, rgb_outs, out_size=cfg.out_size, fov=cfg.fov)

                for spec, rgb_out in zip(all_specs, rgb_outs):
                    rgb = np.array(Image.open(rgb_out).convert("RGB"), dtype=np.uint8)

                    if tile_first:
                        model_ids = self.engine.predict_city_ids(rgb)
                        unmapped = cm.ade_id_to_dataset_id.get(-1, 255)
                        lbl = np.full(model_ids.shape, int(unmapped), dtype=np.uint8)
                        for model_id, dataset_id in cm.ade_id_to_dataset_id.items():
                            if model_id < 0:
                                continue
                            lbl[model_ids == int(model_id)] = np.uint8(int(dataset_id))
                    else:
                        lbl_out = td_path / f"{spec.name}_lbl.png"
                        lbl = np.array(Image.open(lbl_out).convert("L"), dtype=np.uint8)

                    bad = (lbl != 255) & (lbl > 7)
                    if np.any(bad):
                        lbl[bad] = 255

                    filename = f"frame_{saved_idx:06d}_{spec.name}.png"
                    writer.save_image(split, filename, rgb)
                    writer.save_label(split, filename, lbl)

            saved_idx += 1
            if saved_idx % 5 == 0:
                self.logger.log(f"saved frames: {saved_idx}")

        cap.release()
        self.logger.log("done")
