from __future__ import annotations

import glob
import json
import os
import threading
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk
import yaml

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.pipeline import ExtractConfig, GeneratorPipeline, PreviewResult, overlay_segmentation, resize_equirect_for_speed
from src.segmentation.palette import default_palette_8
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

        # Image-folder preview navigation
        self._image_files: List[Path] = []
        self._image_index: int = 0

        self.preview1_photo: Optional[ImageTk.PhotoImage] = None
        self.preview1_rgb: Optional[np.ndarray] = None
        self.preview1_seg_rgb: Optional[np.ndarray] = None
        self.preview_loaded_time: Optional[float] = None

        self._preview1_req_id: int = 0

        self.tiles_up: List[Tile] = []
        self.tiles_mid: List[Tile] = []
        self.tiles_down: List[Tile] = []

        self._load_persisted_paths()

        self._build_layout()

        self.logger = Logger(sink=self._append_log)
        self.pipeline = GeneratorPipeline(logger=self.logger)

        self._build_legend()

        self._refresh_input_state()

    def _build_layout(self) -> None:
        self.root.geometry("1300x900")

        # Scrollable root container
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)

        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._scroll_canvas = canvas
        self._scroll_outer = outer

        main = ttk.Frame(canvas)
        self._scroll_main = main
        window_id = canvas.create_window((0, 0), window=main, anchor="nw")
        self._scroll_window_id = window_id

        def _on_configure(_e: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(_e: tk.Event) -> None:
            # Keep content width in sync with canvas width.
            canvas.itemconfigure(window_id, width=canvas.winfo_width())

        main.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event: tk.Event) -> None:
            # Linux: Button-4/5. Windows/macOS: MouseWheel.
            if getattr(event, "num", None) == 4:
                canvas.yview_scroll(-2, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(2, "units")
            else:
                delta = int(-1 * (getattr(event, "delta", 0) / 120))
                if delta != 0:
                    canvas.yview_scroll(delta, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)

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
        self.ent_video.bind("<FocusOut>", lambda _e: self._on_video_path_edited())
        self.btn_video = ttk.Button(row1, text="参照", command=self._browse_video)
        self.btn_video.pack(side="left")

        # Images dir
        row2 = ttk.Frame(frm_in)
        row2.pack(fill="x", padx=8, pady=2)
        ttk.Label(row2, text="画像フォルダ").pack(side="left")
        self.ent_images = ttk.Entry(row2, textvariable=self.var_images_dir)
        self.ent_images.pack(side="left", fill="x", expand=True, padx=6)
        self.ent_images.bind("<FocusOut>", lambda _e: self._on_images_dir_edited())
        self.btn_images = ttk.Button(row2, text="参照", command=self._browse_images_dir)
        self.btn_images.pack(side="left")

        # Output dir
        row3 = ttk.Frame(frm_in)
        row3.pack(fill="x", padx=8, pady=2)
        ttk.Label(row3, text="出力フォルダ").pack(side="left")
        self.ent_output = ttk.Entry(row3, textvariable=self.var_output_dir)
        self.ent_output.pack(side="left", fill="x", expand=True, padx=6)
        self.ent_output.bind("<FocusOut>", lambda _e: self._persist_paths())
        ttk.Button(row3, text="参照", command=self._browse_output_dir).pack(side="left")

        # Preview1 + (settings + legend) + preview2
        body = ttk.Frame(main)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)

        frm_p1 = ttk.LabelFrame(left, text="プレビュー1 (正面指定: yawスライダー)")
        frm_p1.pack(fill="x", padx=0, pady=6)
        self.frm_p1 = frm_p1

        rowp = ttk.Frame(frm_p1)
        rowp.pack(fill="x", padx=8, pady=4)
        self.btn_prev_image = ttk.Button(rowp, text="←", command=self._on_prev_image, width=3)
        self.btn_prev_image.pack(side="left")
        self.btn_next_image = ttk.Button(rowp, text="→", command=self._on_next_image, width=3)
        self.btn_next_image.pack(side="left", padx=(4, 12))

        ttk.Label(rowp, text="プレビュー時刻[秒]").pack(side="left")
        self.ent_preview_time = ttk.Entry(rowp, textvariable=self.var_preview_time, width=10)
        self.ent_preview_time.pack(side="left", padx=(6, 0))
        self.ent_preview_time.bind("<Return>", lambda _e: self._on_preview1())
        self.ent_preview_time.bind("<FocusOut>", lambda _e: self._on_preview1())

        self.lbl_time = ttk.Label(rowp, text="")
        self.lbl_time.pack(side="left", padx=(10, 0))

        self.lbl_image_idx = ttk.Label(rowp, text="")
        self.lbl_image_idx.pack(side="right")

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

        frm_mid = ttk.Frame(left)
        frm_mid.pack(fill="x", padx=0, pady=(0, 6))
        frm_mid.grid_columnconfigure(0, weight=1)
        frm_mid.grid_columnconfigure(1, weight=1)

        frm_set = ttk.LabelFrame(frm_mid, text="切り出し設定")
        frm_set.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)

        frm_legend = ttk.LabelFrame(frm_mid, text="クラス凡例")
        frm_legend.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.frm_legend = frm_legend

        r0 = ttk.Frame(frm_set)
        r0.pack(fill="x", padx=8, pady=4)
        ttk.Label(r0, text="FOV(正方形)").pack(side="left")
        ttk.Radiobutton(r0, text="90", variable=self.var_fov, value="90").pack(side="left", padx=6)
        ttk.Radiobutton(r0, text="120", variable=self.var_fov, value="120").pack(side="left")

        r1 = ttk.Frame(frm_set)
        r1.pack(fill="x", padx=8, pady=4)
        ttk.Label(r1, text="1辺 解像度(px)").pack(side="left")
        ttk.Entry(r1, textvariable=self.var_out_size, width=10).pack(side="left", padx=6)

        r_fps = ttk.Frame(frm_set)
        r_fps.pack(fill="x", padx=8, pady=4)
        ttk.Label(r_fps, text="FPS(動画)").pack(side="left")
        self.ent_fps = ttk.Entry(r_fps, textvariable=self.var_fps, width=10)
        self.ent_fps.pack(side="left", padx=(6, 0))

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
        self.frm_p2 = frm_p2

        rowp2 = ttk.Frame(frm_p2)
        rowp2.pack(fill="x", padx=8, pady=4)
        ttk.Button(rowp2, text="プレビュー2生成", command=self._on_preview2).pack(side="left")
        ttk.Checkbutton(
            rowp2, text="セグメンテーション表示", variable=self.var_show_seg, command=self._on_toggle_show_seg
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

    def _read_class_names_from_yaml(self) -> Dict[int, str]:
        """Return {dataset_id: name} from config/class_map.yaml (no model load)."""
        path = Path(__file__).resolve().parent.parent / "config" / "class_map.yaml"
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            raw = None

        out: Dict[int, str] = {}
        if isinstance(raw, dict):
            classes = raw.get("classes")
            if isinstance(classes, dict):
                for k, spec in classes.items():
                    try:
                        did = int(k)
                    except Exception:
                        continue
                    if not (0 <= did <= 7):
                        continue
                    if not isinstance(spec, dict):
                        continue
                    out[did] = str(spec.get("name", f"class{did}"))
        return dict(sorted(out.items(), key=lambda kv: kv[0]))

    def _build_legend(self) -> None:
        """Populate legend frame with class name + color swatch."""
        frm = getattr(self, "frm_legend", None)
        if frm is None:
            return

        for child in frm.winfo_children():
            child.destroy()

        palette = default_palette_8()
        names = self._read_class_names_from_yaml()

        grid = ttk.Frame(frm)
        grid.pack(fill="x", padx=8, pady=6)

        # Keep this compact for side-by-side layout.
        cols = 2
        for i, (cls_id, cls_name) in enumerate(names.items()):
            if not (0 <= cls_id < len(palette)):
                continue

            r = i // cols
            c = i % cols

            item = ttk.Frame(grid)
            item.grid(row=r, column=c, sticky="w", padx=(0, 16), pady=2)

            rr, gg, bb = palette[cls_id]
            color_hex = f"#{rr:02x}{gg:02x}{bb:02x}"
            sw = tk.Canvas(item, width=16, height=16, highlightthickness=0)
            # Outline helps when fill is black.
            sw.create_rectangle(1, 1, 15, 15, fill=color_hex, outline="#ffffff")
            sw.pack(side="left")

            ttk.Label(item, text=f"{cls_id}: {cls_name}").pack(side="left", padx=(6, 0))

    def _on_toggle_show_seg(self) -> None:
        if self.var_show_seg.get() and self.preview1_rgb is not None and self.preview1_seg_rgb is None:
            self._start_preview1_segmentation(self.preview1_rgb)
        self._refresh_preview1_overlay()
        self._refresh_preview2_images()

    def _start_preview1_segmentation(self, pano_rgb: np.ndarray, *, req_id: Optional[int] = None) -> None:
        if req_id is None:
            self._preview1_req_id += 1
            req_id = self._preview1_req_id

        self._append_log("[info] preview1: segmenting...")

        def worker() -> None:
            try:
                self.pipeline.engine.ensure_loaded()
                cm = self.pipeline._ensure_class_map()  # cache if already loaded

                ade = self.pipeline.engine.predict_ade_ids(pano_rgb)
                unmapped = cm.ade_id_to_dataset_id.get(-1, 255)
                lbl = np.full(ade.shape, int(unmapped), dtype=np.uint8)
                for ade_id, dataset_id in cm.ade_id_to_dataset_id.items():
                    if ade_id < 0:
                        continue
                    lbl[ade == int(ade_id)] = np.uint8(int(dataset_id))

                seg_rgb = overlay_segmentation(pano_rgb, lbl)
            except Exception as e:
                self.logger.log(f"preview1 segmentation failed: {e}")
                return

            def apply() -> None:
                if req_id != self._preview1_req_id:
                    return
                self.preview1_seg_rgb = seg_rgb
                self._refresh_preview1_overlay()

            self.root.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

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
            self.var_input_type.set("video")
            self._refresh_input_state()
            self.var_video_path.set(path)
            self.var_preview_time.set("0.0")
            self._persist_paths()
            self._on_preview1()

    def _browse_images_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.var_input_type.set("images")
            self._refresh_input_state()
            self.var_images_dir.set(path)
            self._set_image_folder(Path(path))
            self._persist_paths()
            self._on_preview1()

    def _on_video_path_edited(self) -> None:
        self._persist_paths()
        if self.var_input_type.get() != "video":
            return
        video_str = self.var_video_path.get().strip()
        if not video_str:
            return
        if not self.var_preview_time.get().strip():
            self.var_preview_time.set("0.0")
        self._on_preview1()

    def _on_images_dir_edited(self) -> None:
        self._persist_paths()
        folder_str = self.var_images_dir.get().strip()
        if not folder_str:
            return
        self._set_image_folder(Path(folder_str), keep_index=False)
        if self.var_input_type.get() != "images":
            return
        self._on_preview1()

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.var_output_dir.set(path)
            self._persist_paths()

    def _refresh_input_state(self) -> None:
        is_video = self.var_input_type.get() == "video"
        state_video = "normal" if is_video else "disabled"
        state_images = "normal" if not is_video else "disabled"

        for w in (self.ent_video, self.btn_video, self.ent_fps, self.ent_preview_time):
            w.configure(state=state_video)
        for w in (self.ent_images, self.btn_images):
            w.configure(state=state_images)

        for w in (self.btn_prev_image, self.btn_next_image):
            w.configure(state=state_images)

        if not is_video:
            # Refresh file list if path is already set.
            folder_str = self.var_images_dir.get().strip()
            if folder_str:
                self._set_image_folder(Path(folder_str), keep_index=True)
        self._update_image_index_label()

    def _set_image_folder(self, folder: Path, *, keep_index: bool = False) -> None:
        if not folder.is_dir():
            self._image_files = []
            self._image_index = 0
            self._update_image_index_label()
            return

        patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
        files: List[Path] = []
        for pat in patterns:
            files.extend([Path(p) for p in glob.glob(str(folder / pat))])
        files = sorted(set(files))
        self._image_files = files
        if not keep_index:
            self._image_index = 0
        else:
            self._image_index = max(0, min(self._image_index, max(0, len(files) - 1)))
        self._update_image_index_label()

    def _update_image_index_label(self) -> None:
        if not hasattr(self, "lbl_image_idx"):
            return
        if self.var_input_type.get() != "images":
            self.lbl_image_idx.config(text="")
            return
        n = len(self._image_files)
        if n <= 0:
            self.lbl_image_idx.config(text="(0/0)")
            return
        cur = self._image_files[self._image_index]
        self.lbl_image_idx.config(text=f"({self._image_index+1}/{n}) {cur.name}")

    def _on_prev_image(self) -> None:
        if self.var_input_type.get() != "images":
            return
        if not self._image_files:
            self._set_image_folder(Path(self.var_images_dir.get().strip()))
        if not self._image_files:
            return
        self._image_index = (self._image_index - 1) % len(self._image_files)
        self._update_image_index_label()
        self._on_preview1()

    def _on_next_image(self) -> None:
        if self.var_input_type.get() != "images":
            return
        if not self._image_files:
            self._set_image_folder(Path(self.var_images_dir.get().strip()))
        if not self._image_files:
            return
        self._image_index = (self._image_index + 1) % len(self._image_files)
        self._update_image_index_label()
        self._on_preview1()

    def _state_file_path(self) -> Path:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg) if xdg else (Path.home() / ".config")
        return base / "mask2dataset" / "state.json"

    def _load_persisted_paths(self) -> None:
        path = self._state_file_path()
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        if isinstance(raw.get("video_path"), str):
            self.var_video_path.set(raw["video_path"])
        if isinstance(raw.get("images_dir"), str):
            self.var_images_dir.set(raw["images_dir"])
            self._set_image_folder(Path(raw["images_dir"]), keep_index=False)
        if isinstance(raw.get("output_dir"), str):
            self.var_output_dir.set(raw["output_dir"])

    def _persist_paths(self) -> None:
        path = self._state_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "video_path": self.var_video_path.get().strip(),
            "images_dir": self.var_images_dir.get().strip(),
            "output_dir": self.var_output_dir.get().strip(),
        }
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            # Best-effort; ignore persistence failures.
            return

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
            video_str = self.var_video_path.get().strip()
            if not video_str:
                raise ValueError("動画パスが未指定です")
            video = Path(video_str)
            if not video.is_file():
                raise ValueError("動画パスが正しくありません")
            try:
                t = float(self.var_preview_time.get())
            except Exception:
                raise ValueError("プレビュー時刻[秒]は数値で入力してください")
            if t < 0:
                raise ValueError("プレビュー時刻[秒]は0以上")
            bgr = _load_video_frame_with_ffmpeg(self.pipeline.projector.ffmpeg, video, t)
            return bgr, t

        folder_str = self.var_images_dir.get().strip()
        if not folder_str:
            raise ValueError("画像フォルダが未指定です")
        folder = Path(folder_str)
        if not folder.is_dir():
            raise ValueError("画像フォルダが正しくありません")

        # Use current selection if available; otherwise fallback to first file.
        if not self._image_files:
            self._set_image_folder(folder, keep_index=True)
        if self._image_files:
            path = self._image_files[self._image_index]
        else:
            path = _first_image_in_folder(folder)
        if path is None:
            raise ValueError("画像フォルダに画像が見つかりません")

        self._update_image_index_label()

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
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

        self._preview1_req_id += 1
        req_id = self._preview1_req_id

        # Keep the base RGB for overlay refresh
        pano_bgr = resize_equirect_for_speed(bgr, cfg.out_size)
        pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)
        self.preview1_rgb = pano_rgb
        self.preview1_seg_rgb = None
        self.preview_loaded_time = t
        self._refresh_preview1_overlay()

        if self.var_show_seg.get():
            self._start_preview1_segmentation(pano_rgb, req_id=req_id)

        if t is None:
            self.lbl_time.config(text="")
        else:
            self.lbl_time.config(text=f"t = {t:.2f} s")

    def _refresh_preview1_overlay(self) -> None:
        if self.preview1_rgb is None:
            return

        rgb = self.preview1_seg_rgb if (self.var_show_seg.get() and self.preview1_seg_rgb is not None) else self.preview1_rgb
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

        out_dir_str = self.var_output_dir.get().strip()
        if not out_dir_str:
            messagebox.showerror("エラー", "出力フォルダを指定してください")
            return
        out_dir = Path(out_dir_str)
        out_dir.mkdir(parents=True, exist_ok=True)

        self._persist_paths()

        def worker() -> None:
            try:
                if self.var_input_type.get() == "video":
                    video_str = self.var_video_path.get().strip()
                    if not video_str:
                        raise ValueError("動画パスが未指定です")
                    video = Path(video_str)
                    if not video.is_file():
                        raise ValueError("動画パスが正しくありません")
                    try:
                        fps = float(self.var_fps.get())
                    except Exception:
                        raise ValueError("FPSは数値で入力してください")
                    if fps <= 0:
                        raise ValueError("FPSは0より大きくしてください")
                    self.pipeline.generate_dataset_from_video(video_path=video, output_root=out_dir, fps=fps, cfg=cfg)
                else:
                    folder_str = self.var_images_dir.get().strip()
                    if not folder_str:
                        raise ValueError("画像フォルダが未指定です")
                    folder = Path(folder_str)
                    if not folder.is_dir():
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
