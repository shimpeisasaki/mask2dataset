from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
