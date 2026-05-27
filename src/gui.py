from __future__ import annotations

import glob
import os
import threading
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.pipeline import ExtractConfig, GeneratorPipeline, PreviewResult
from src.utils.logging import Logger


def _first_image_in_folder(folder: Path) -> Optional[Path]:
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(str(folder / p)))
    files = sorted(set(files))
    return Path(files[0]) if files else None


def _load_video_frame_with_ffmpeg(ffmpeg: str, video_path: Path, time_s: float) -> np.ndarray:
    # Extract a single frame to a temp file (fast + robust).
    with tempfile.TemporaryDirectory(prefix="preview1_") as td:
        out_path = Path(td) / "frame.jpg"
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(float(time_s)),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out_path),
        ]
        import subprocess

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out_path.exists():
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"ffmpeg preview failed: {err}")

        bgr = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("failed to read extracted preview frame")
        return bgr


def _pil_from_rgb(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


@dataclass
class Tile:
    label: tk.Label
    photo: Optional[ImageTk.PhotoImage] = None
    rgb: Optional[np.ndarray] = None
    seg: Optional[np.ndarray] = None


class AppGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("360 Dataset Generator (Mask2Former ADE20K)")

        self.var_input_type = tk.StringVar(value="video")  # video | images

        self.var_video_path = tk.StringVar(value="")
        self.var_images_dir = tk.StringVar(value="")
        self.var_output_dir = tk.StringVar(value="")

        self.var_fps = tk.StringVar(value="1.0")
        self.var_preview_time = tk.StringVar(value="0.0")
        self.var_yaw_offset = tk.DoubleVar(value=0.0)

        self.var_fov = tk.StringVar(value="90")
        self.var_out_size = tk.StringVar(value="512")

        self.var_up_4 = tk.BooleanVar(value=False)
        self.var_up_6 = tk.BooleanVar(value=True)
        self.var_top = tk.BooleanVar(value=True)
        self.var_h_4 = tk.BooleanVar(value=False)
        self.var_h_6 = tk.BooleanVar(value=True)
        self.var_down_4 = tk.BooleanVar(value=False)
        self.var_down_6 = tk.BooleanVar(value=True)

        self.var_show_seg = tk.BooleanVar(value=False)

        self.preview1_photo: Optional[ImageTk.PhotoImage] = None
        self.preview1_rgb: Optional[np.ndarray] = None
        self.preview_loaded_time: Optional[float] = None

        self.tiles_up: List[Tile] = []
        self.tiles_mid: List[Tile] = []
        self.tiles_down: List[Tile] = []

        self._build_layout()

        self.logger = Logger(sink=self._append_log)
        self.pipeline = GeneratorPipeline(logger=self.logger)

        self._refresh_input_state()

    def _build_layout(self) -> None:
        self.root.geometry("1300x900")

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        # Top controls
        frm_top = ttk.Frame(main)
        frm_top.pack(fill="x", padx=10, pady=8)

        # Input type
        frm_in = ttk.LabelFrame(frm_top, text="入力")
        frm_in.pack(fill="x", side="top")

        row0 = ttk.Frame(frm_in)
        row0.pack(fill="x", padx=8, pady=4)
        ttk.Radiobutton(row0, text="動画", variable=self.var_input_type, value="video", command=self._refresh_input_state).pack(
            side="left"
        )
        ttk.Radiobutton(row0, text="画像フォルダ", variable=self.var_input_type, value="images", command=self._refresh_input_state).pack(
            side="left", padx=(12, 0)
        )

        # Video path
        row1 = ttk.Frame(frm_in)
        row1.pack(fill="x", padx=8, pady=2)
        ttk.Label(row1, text="動画パス").pack(side="left")
        self.ent_video = ttk.Entry(row1, textvariable=self.var_video_path)
        self.ent_video.pack(side="left", fill="x", expand=True, padx=6)
        self.btn_video = ttk.Button(row1, text="参照", command=self._browse_video)
        self.btn_video.pack(side="left")

        # Images dir
        row2 = ttk.Frame(frm_in)
        row2.pack(fill="x", padx=8, pady=2)
        ttk.Label(row2, text="画像フォルダ").pack(side="left")
        self.ent_images = ttk.Entry(row2, textvariable=self.var_images_dir)
        self.ent_images.pack(side="left", fill="x", expand=True, padx=6)
        self.btn_images = ttk.Button(row2, text="参照", command=self._browse_images_dir)
        self.btn_images.pack(side="left")

        # Output dir
        row3 = ttk.Frame(frm_in)
        row3.pack(fill="x", padx=8, pady=2)
        ttk.Label(row3, text="出力フォルダ").pack(side="left")
        self.ent_output = ttk.Entry(row3, textvariable=self.var_output_dir)
        self.ent_output.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row3, text="参照", command=self._browse_output_dir).pack(side="left")

        # Video FPS
        row4 = ttk.Frame(frm_in)
        row4.pack(fill="x", padx=8, pady=(2, 6))
        ttk.Label(row4, text="FPS(動画) ").pack(side="left")
        self.ent_fps = ttk.Entry(row4, textvariable=self.var_fps, width=10)
        self.ent_fps.pack(side="left", padx=(0, 10))
        ttk.Label(row4, text="プレビュー時刻[秒] ").pack(side="left")
        self.ent_preview_time = ttk.Entry(row4, textvariable=self.var_preview_time, width=10)
        self.ent_preview_time.pack(side="left")

        # Preview1 + settings
        body = ttk.Frame(main)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(body)
        right.pack(side="right", fill="y")

        frm_p1 = ttk.LabelFrame(left, text="プレビュー1 (正面指定: yawスライダー)")
        frm_p1.pack(fill="x", padx=0, pady=6)

        rowp = ttk.Frame(frm_p1)
        rowp.pack(fill="x", padx=8, pady=4)
        ttk.Button(rowp, text="プレビュー1更新", command=self._on_preview1).pack(side="left")
        self.lbl_time = ttk.Label(rowp, text="")
        self.lbl_time.pack(side="left", padx=(12, 0))

        self.yaw_slider = tk.Scale(
            frm_p1,
            from_=-180,
            to=180,
            orient="horizontal",
            resolution=1,
            variable=self.var_yaw_offset,
            length=900,
            command=lambda _v: self._refresh_preview1_overlay(),
        )
        self.yaw_slider.pack(fill="x", padx=8, pady=(0, 4))

        self.lbl_preview1 = tk.Label(frm_p1, borderwidth=1, relief="solid")
        self.lbl_preview1.pack(fill="x", padx=8, pady=(0, 8))

        frm_set = ttk.LabelFrame(right, text="切り出し設定")
        frm_set.pack(fill="x", padx=6, pady=6)

        r0 = ttk.Frame(frm_set)
        r0.pack(fill="x", padx=8, pady=4)
        ttk.Label(r0, text="FOV(正方形)").pack(side="left")
        ttk.Radiobutton(r0, text="90", variable=self.var_fov, value="90").pack(side="left", padx=6)
        ttk.Radiobutton(r0, text="120", variable=self.var_fov, value="120").pack(side="left")

        r1 = ttk.Frame(frm_set)
        r1.pack(fill="x", padx=8, pady=4)
        ttk.Label(r1, text="1辺 解像度(px)").pack(side="left")
        ttk.Entry(r1, textvariable=self.var_out_size, width=10).pack(side="left", padx=6)

        grid = ttk.Frame(frm_set)
        grid.pack(fill="x", padx=8, pady=6)

        ttk.Label(grid, text="上方向(上45°)").grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(grid, text="4分割", variable=self.var_up_4).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(grid, text="6分割", variable=self.var_up_6).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(grid, text="真上", variable=self.var_top).grid(row=0, column=3, sticky="w")

        ttk.Label(grid, text="水平方向").grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(grid, text="4分割", variable=self.var_h_4).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(grid, text="6分割", variable=self.var_h_6).grid(row=1, column=2, sticky="w")

        ttk.Label(grid, text="下方向(下45°)").grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(grid, text="4分割", variable=self.var_down_4).grid(row=2, column=1, sticky="w")
        ttk.Checkbutton(grid, text="6分割", variable=self.var_down_6).grid(row=2, column=2, sticky="w")

        frm_p2 = ttk.LabelFrame(left, text="プレビュー2")
        frm_p2.pack(fill="both", expand=True, padx=0, pady=6)

        rowp2 = ttk.Frame(frm_p2)
        rowp2.pack(fill="x", padx=8, pady=4)
        ttk.Button(rowp2, text="プレビュー2生成", command=self._on_preview2).pack(side="left")
        ttk.Checkbutton(
            rowp2, text="セグメンテーション表示", variable=self.var_show_seg, command=self._refresh_preview2_images
        ).pack(side="left", padx=(14, 0))

        # Rows: up (max 7), mid (max 6), down (max 6)
        self.frm_row_up = ttk.Frame(frm_p2)
        self.frm_row_up.pack(fill="x", padx=8, pady=4)
        self.frm_row_mid = ttk.Frame(frm_p2)
        self.frm_row_mid.pack(fill="x", padx=8, pady=4)
        self.frm_row_down = ttk.Frame(frm_p2)
        self.frm_row_down.pack(fill="x", padx=8, pady=4)

        frm_run = ttk.Frame(main)
        frm_run.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Button(frm_run, text="実行", command=self._on_run).pack(side="left")

        frm_log = ttk.LabelFrame(main, text="ログ")
        frm_log.pack(fill="both", expand=False, padx=10, pady=(0, 10))
        self.txt_log = tk.Text(frm_log, height=10, state="disabled")
        self.txt_log.pack(fill="both", expand=True, padx=8, pady=6)

    def _append_log(self, line: str) -> None:
        def _do() -> None:
            self.txt_log.configure(state="normal")
            self.txt_log.insert(tk.END, line + "\n")
            self.txt_log.see(tk.END)
            self.txt_log.configure(state="disabled")

        self.root.after(0, _do)

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.mov *.mkv *.avi"), ("All", "*.*")]
        )
        if path:
            self.var_video_path.set(path)

    def _browse_images_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.var_images_dir.set(path)

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.var_output_dir.set(path)

    def _refresh_input_state(self) -> None:
        is_video = self.var_input_type.get() == "video"
        state_video = "normal" if is_video else "disabled"
        state_images = "normal" if not is_video else "disabled"

        for w in (self.ent_video, self.btn_video, self.ent_fps, self.ent_preview_time):
            w.configure(state=state_video)
        for w in (self.ent_images, self.btn_images):
            w.configure(state=state_images)

    def _parse_cfg(self) -> ExtractConfig:
        try:
            fov = float(self.var_fov.get())
        except Exception:
            raise ValueError("FOV must be 90 or 120")
        if fov not in (90.0, 120.0):
            raise ValueError("FOV must be 90 or 120")

        try:
            out_size = int(self.var_out_size.get())
        except Exception:
            raise ValueError("out size must be integer")
        if out_size < 128:
            raise ValueError("out size must be >= 128")

        cfg = ExtractConfig(
            fov=fov,
            out_size=out_size,
            yaw_offset=float(self.var_yaw_offset.get()),
            use_up_4=bool(self.var_up_4.get()),
            use_up_6=bool(self.var_up_6.get()),
            use_top=bool(self.var_top.get()),
            use_h_4=bool(self.var_h_4.get()),
            use_h_6=bool(self.var_h_6.get()),
            use_down_4=bool(self.var_down_4.get()),
            use_down_6=bool(self.var_down_6.get()),
        )

        # Validate view count 1..19
        from src.pipeline import build_view_specs

        up, mid, down = build_view_specs(cfg)
        count = len(up) + len(mid) + len(down)
        if count < 1 or count > 19:
            raise ValueError("view count must be 1..19")
        return cfg

    def _load_preview1_bgr(self) -> Tuple[np.ndarray, Optional[float]]:
        if self.var_input_type.get() == "video":
            video = Path(self.var_video_path.get().strip())
            if not video.exists():
                raise ValueError("動画パスが正しくありません")
            try:
                t = float(self.var_preview_time.get())
            except Exception:
                raise ValueError("プレビュー時刻[秒]は数値で入力してください")
            if t < 0:
                raise ValueError("プレビュー時刻[秒]は0以上")
            bgr = _load_video_frame_with_ffmpeg(self.pipeline.projector.ffmpeg, video, t)
            return bgr, t

        folder = Path(self.var_images_dir.get().strip())
        if not folder.exists():
            raise ValueError("画像フォルダが正しくありません")
        first = _first_image_in_folder(folder)
        if first is None:
            raise ValueError("画像フォルダに画像が見つかりません")
        bgr = cv2.imread(str(first), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("画像の読み込みに失敗しました")
        return bgr, None

    def _on_preview1(self) -> None:
        try:
            bgr, t = self._load_preview1_bgr()
            cfg = self._parse_cfg()
        except Exception as e:
            messagebox.showerror("エラー", str(e))
            return

        # Keep the base RGB for overlay refresh
        pano_bgr = cv2.resize(bgr, (cfg.out_size * 2, cfg.out_size), interpolation=cv2.INTER_AREA)
        pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)
        self.preview1_rgb = pano_rgb
        self.preview_loaded_time = t
        self._refresh_preview1_overlay()

        if t is None:
            self.lbl_time.config(text="")
        else:
            self.lbl_time.config(text=f"t = {t:.2f} s")

    def _refresh_preview1_overlay(self) -> None:
        if self.preview1_rgb is None:
            return

        rgb = self.preview1_rgb
        h, w = rgb.shape[:2]
        yaw0 = float(self.var_yaw_offset.get())
        yaw_disp = (yaw0 + 180.0) % 360.0
        x = int((yaw_disp / 360.0) * w)

        img = rgb.copy()
        x = max(0, min(w - 1, x))
        img[:, max(0, x - 1) : min(w, x + 2), :] = (0, 255, 255)

        pil = _pil_from_rgb(img)
        # Scale to fit
        max_w = 980
        scale = min(1.0, max_w / max(1, pil.size[0]))
        pil = pil.resize((int(pil.size[0] * scale), int(pil.size[1] * scale)), Image.Resampling.BILINEAR)

        photo = ImageTk.PhotoImage(pil)
        self.preview1_photo = photo
        self.lbl_preview1.configure(image=photo)

    def _clear_preview2(self) -> None:
        for frm in (self.frm_row_up, self.frm_row_mid, self.frm_row_down):
            for child in frm.winfo_children():
                child.destroy()

        self.tiles_up = []
        self.tiles_mid = []
        self.tiles_down = []

    def _set_tiles(self, container: ttk.Frame, images: List[np.ndarray], segs: List[np.ndarray], max_cols: int) -> List[Tile]:
        tiles: List[Tile] = []
        thumb = 160
        for i, (rgb, seg) in enumerate(zip(images, segs)):
            if i >= max_cols:
                break
            show = seg if self.var_show_seg.get() else rgb
            pil = _pil_from_rgb(show)
            pil.thumbnail((thumb, thumb), Image.Resampling.BILINEAR)
            photo = ImageTk.PhotoImage(pil)
            lbl = tk.Label(container, image=photo, borderwidth=1, relief="solid")
            lbl.grid(row=0, column=i, padx=4, pady=2)
            tile = Tile(label=lbl, photo=photo, rgb=rgb, seg=seg)
            tiles.append(tile)
        return tiles

    def _refresh_preview2_images(self) -> None:
        def refresh_group(tiles: List[Tile]) -> None:
            for tile in tiles:
                if tile.rgb is None or tile.seg is None:
                    continue
                show = tile.seg if self.var_show_seg.get() else tile.rgb
                pil = _pil_from_rgb(show)
                pil.thumbnail((160, 160), Image.Resampling.BILINEAR)
                tile.photo = ImageTk.PhotoImage(pil)
                tile.label.configure(image=tile.photo)

        refresh_group(self.tiles_up)
        refresh_group(self.tiles_mid)
        refresh_group(self.tiles_down)

    def _on_preview2(self) -> None:
        try:
            cfg = self._parse_cfg()
            bgr, t = self._load_preview1_bgr()
        except Exception as e:
            messagebox.showerror("エラー", str(e))
            return

        self._append_log("[info] preview2: generating...")

        def worker() -> None:
            try:
                result = self.pipeline.build_preview(input_bgr=bgr, preview_time_s=t, cfg=cfg)
            except Exception as e:
                self.logger.log(f"preview2 failed: {e}")
                return

            def apply() -> None:
                self._clear_preview2()
                self.tiles_up = self._set_tiles(self.frm_row_up, result.up_tiles_rgb, result.up_tiles_seg, max_cols=7)
                self.tiles_mid = self._set_tiles(self.frm_row_mid, result.mid_tiles_rgb, result.mid_tiles_seg, max_cols=6)
                self.tiles_down = self._set_tiles(self.frm_row_down, result.down_tiles_rgb, result.down_tiles_seg, max_cols=6)

            self.root.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    def _on_run(self) -> None:
        try:
            cfg = self._parse_cfg()
        except Exception as e:
            messagebox.showerror("エラー", str(e))
            return

        out_dir = Path(self.var_output_dir.get().strip())
        if not out_dir:
            messagebox.showerror("エラー", "出力フォルダを指定してください")
            return
        out_dir.mkdir(parents=True, exist_ok=True)

        def worker() -> None:
            try:
                if self.var_input_type.get() == "video":
                    video = Path(self.var_video_path.get().strip())
                    if not video.exists():
                        raise ValueError("動画パスが正しくありません")
                    try:
                        fps = float(self.var_fps.get())
                    except Exception:
                        raise ValueError("FPSは数値で入力してください")
                    if fps <= 0:
                        raise ValueError("FPSは0より大きくしてください")
                    self.pipeline.generate_dataset_from_video(video_path=video, output_root=out_dir, fps=fps, cfg=cfg)
                else:
                    folder = Path(self.var_images_dir.get().strip())
                    if not folder.exists():
                        raise ValueError("画像フォルダが正しくありません")
                    patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
                    files: List[Path] = []
                    for pat in patterns:
                        files.extend([Path(p) for p in glob.glob(str(folder / pat))])
                    files = sorted(set(files))
                    if not files:
                        raise ValueError("画像フォルダに画像がありません")
                    self.pipeline.generate_dataset_from_images(image_paths=files, output_root=out_dir, cfg=cfg)

            except Exception as e:
                self.logger.log(f"run failed: {e}")
                return

        threading.Thread(target=worker, daemon=True).start()
