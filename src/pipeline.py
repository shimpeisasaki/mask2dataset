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
from src.segmentation.palette import default_palette_8
from src.utils.logging import Logger
from src.v360 import V360Projector, ViewSpec


@dataclass(frozen=True)
class ExtractConfig:
    fov: float  # 90, 120, or 150
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
        self.class_map_path = Path(__file__).resolve().parent.parent / "config" / "class_map.yaml"
        self._class_map: Optional[ClassMap] = None

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
    def _postprocess_road_sidewalk_boundary(
        *,
        ade: np.ndarray,
        lbl: np.ndarray,
        cm: ClassMap,
        kernel_size: int = 10,
        road_name: str = "road",
        sidewalk_name: str = "sidewalk",
        avoid_name: str = "avoid",
        path_name: str = "path",
    ) -> np.ndarray:
        """Mark path-like/road-like boundary as avoid, then merge road-like into path.

        Output label map `lbl` is modified in-place and returned.
        """

        if ade.shape != lbl.shape:
            raise ValueError("ade and lbl must have the same shape")

        # Support treating multiple ADE labels as road-like.
        road_id = cm.ade_name_to_id.get(GeneratorPipeline._normalize_label_name(road_name))
        pool_id = cm.ade_name_to_id.get(GeneratorPipeline._normalize_label_name("pool"))
        if pool_id is None:
            pool_id = cm.ade_name_to_id.get(GeneratorPipeline._normalize_label_name("swimming pool"))
        earth_id = cm.ade_name_to_id.get(GeneratorPipeline._normalize_label_name("earth"))
        fountain_id = cm.ade_name_to_id.get(GeneratorPipeline._normalize_label_name("fountain"))
        water_id = cm.ade_name_to_id.get(GeneratorPipeline._normalize_label_name("water"))

        # Build list of road-like ADE ids.
        road_ids = [int(x) for x in (road_id, pool_id, earth_id, fountain_id, water_id) if x is not None]
        if not road_ids:
            return lbl

        avoid_id = GeneratorPipeline._dataset_id_by_name(cm, avoid_name, default=1)
        path_id = GeneratorPipeline._dataset_id_by_name(cm, path_name, default=2)

        # Non-road path-like ADE labels for step (1) boundary extraction.
        path_like_ids: List[int] = []
        for nm in (sidewalk_name, "floor", "rug", path_name):
            x = cm.ade_name_to_id.get(GeneratorPipeline._normalize_label_name(nm))
            if x is not None:
                path_like_ids.append(int(x))

        k = max(1, int(kernel_size))
        kernel = np.ones((k, k), np.uint8)

        # 1) Boundary between path-like and road-like -> avoid.
        if path_like_ids:
            path_like_mask = np.isin(ade, path_like_ids).astype(np.uint8)
            road_like_mask = np.isin(ade, road_ids).astype(np.uint8)

            path_like_d = cv2.dilate(path_like_mask, kernel, iterations=1)
            road_like_d = cv2.dilate(road_like_mask, kernel, iterations=1)
            boundary_1 = cv2.bitwise_and(path_like_d, road_like_d)
            lbl[boundary_1 > 0] = np.uint8(int(avoid_id))

        # 2) Merge road-like ADE into path.
        lbl[np.isin(ade, road_ids)] = np.uint8(int(path_id))

        return lbl

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
    def _remap_ade_to_dataset_ids(ade: np.ndarray, cm: ClassMap) -> np.ndarray:
        unmapped = cm.ade_id_to_dataset_id.get(-1, 255)
        out = np.full(ade.shape, int(unmapped), dtype=np.uint8)
        for ade_id, dataset_id in cm.ade_id_to_dataset_id.items():
            if ade_id < 0:
                continue
            out[ade == int(ade_id)] = np.uint8(int(dataset_id))
        return out

    def _project_and_segment_group(
        self,
        *,
        td_path: Path,
        pano_path: Path,
        specs: Sequence[ViewSpec],
        cfg: ExtractConfig,
        cm: ClassMap,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[Dict[str, str]], List[str]]:
        if not specs:
            return [], [], [], [], []

        rgb_outs = [td_path / f"{spec.name}.png" for spec in specs]
        self.projector.project_many_rgb(pano_path, specs, rgb_outs, out_size=cfg.out_size, fov=cfg.fov)

        rgbs: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        segs: List[np.ndarray] = []
        reports: List[Dict[str, str]] = []
        names: List[str] = []

        unlabeled_id = self._dataset_id_by_name(cm, "unlabeled", default=5)
        avoid_id = self._dataset_id_by_name(cm, "avoid", default=1)
        path_id = self._dataset_id_by_name(cm, "path", default=2)
        id2label = self.engine.id2label
        for spec, rgb_p in zip(specs, rgb_outs):
            rgb = np.array(Image.open(rgb_p).convert("RGB"), dtype=np.uint8)
            ade = self.engine.predict_ade_ids(rgb)
            lbl = self._remap_ade_to_dataset_ids(ade, cm)
            lbl = self._postprocess_road_sidewalk_boundary(ade=ade, lbl=lbl, cm=cm)
            reports.append({
                "unlabeled": self._format_target_report(
                    ade=ade,
                    lbl=lbl,
                    id2label=id2label,
                    target_id=unlabeled_id,
                    target_name="unlabeled",
                ),
                "avoid": self._format_target_report(
                    ade=ade,
                    lbl=lbl,
                    id2label=id2label,
                    target_id=avoid_id,
                    target_name="avoid",
                ),
                "path": self._format_target_report(
                    ade=ade,
                    lbl=lbl,
                    id2label=id2label,
                    target_id=path_id,
                    target_name="path",
                ),
            })
            rgbs.append(rgb)
            labels.append(lbl)
            segs.append(overlay_segmentation(rgb, lbl))
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

        with tempfile.TemporaryDirectory(prefix="v360_preview_") as td:
            td_path = Path(td)
            pano_path = td_path / "pano.png"
            Image.fromarray(pano_rgb, mode="RGB").save(pano_path)

            up_rgb, _, up_seg, up_rep, up_names = self._project_and_segment_group(td_path=td_path, pano_path=pano_path, specs=up_specs, cfg=cfg, cm=cm)
            mid_rgb, _, mid_seg, mid_rep, mid_names = self._project_and_segment_group(td_path=td_path, pano_path=pano_path, specs=mid_specs, cfg=cfg, cm=cm)
            down_rgb, _, down_seg, down_rep, down_names = self._project_and_segment_group(td_path=td_path, pano_path=pano_path, specs=down_specs, cfg=cfg, cm=cm)

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
    ) -> None:
        cm = self._ensure_class_map()
        writer = MMSegDatasetWriter(root=output_root)
        writer.ensure_dirs()

        rng = random.Random(writer.seed)

        up_specs, mid_specs, down_specs = build_view_specs(cfg)
        all_specs = list(up_specs) + list(mid_specs) + list(down_specs)
        all_specs = self._filter_specs_by_names(all_specs, include_spec_names)
        if not all_specs:
            raise ValueError("no enabled view directions to generate")

        self.logger.log(f"Generate from images: count={len(image_paths)} views={len(all_specs)}")

        for src_idx, path in enumerate(image_paths):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                self.logger.log(f"skip unreadable image: {path}")
                continue

            pano_bgr = resize_equirect_for_speed(bgr, cfg.out_size)
            pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)

            base = path.stem

            # Choose split per source image to avoid leakage across train/val.
            split = writer.choose_split(rng.random())

            with tempfile.TemporaryDirectory(prefix="v360_gen_") as td:
                td_path = Path(td)
                pano_path = td_path / "pano.png"
                Image.fromarray(pano_rgb, mode="RGB").save(pano_path)

                rgb_tiles, lbl_tiles, _, _, _ = self._project_and_segment_group(
                    td_path=td_path,
                    pano_path=pano_path,
                    specs=all_specs,
                    cfg=cfg,
                    cm=cm,
                )

                for spec, rgb, lbl in zip(all_specs, rgb_tiles, lbl_tiles):
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
        include_spec_names: Optional[Sequence[str]] = None,
    ) -> None:
        cm = self._ensure_class_map()
        writer = MMSegDatasetWriter(root=output_root)
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

            with tempfile.TemporaryDirectory(prefix="v360_vid_") as td:
                td_path = Path(td)
                pano_path = td_path / "pano.png"
                Image.fromarray(pano_rgb, mode="RGB").save(pano_path)

                rgb_tiles, lbl_tiles, _, _, _ = self._project_and_segment_group(
                    td_path=td_path,
                    pano_path=pano_path,
                    specs=all_specs,
                    cfg=cfg,
                    cm=cm,
                )

                for spec, rgb, lbl in zip(all_specs, rgb_tiles, lbl_tiles):
                    filename = f"frame_{saved_idx:06d}_{spec.name}.png"
                    writer.save_image(split, filename, rgb)
                    writer.save_label(split, filename, lbl)

            saved_idx += 1
            if saved_idx % 5 == 0:
                self.logger.log(f"saved frames: {saved_idx}")

        cap.release()
        self.logger.log("done")
