from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from src.pipeline import GeneratorPipeline
from src.segmentation.class_map import ClassMap
from src.segmentation.mask2former import Mask2FormerADEEngine


def _topk(items: Dict[int, int], k: int) -> List[Tuple[int, int]]:
    return sorted(items.items(), key=lambda kv: kv[1], reverse=True)[:k]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Report which ADE20K labels end up as dataset unlabeled(=5 by default).\n"
            "Useful to decide what to add to config/class_map.yaml."
        )
    )
    ap.add_argument("image", type=Path, help="Path to an RGB image (e.g. a generated tile in img_dir/*/*.png)")
    ap.add_argument(
        "--class-map",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config" / "class_map.yaml",
        help="Path to class_map.yaml",
    )
    ap.add_argument("--kernel", type=int, default=10, help="Boundary kernel size (same meaning as pipeline)")
    ap.add_argument(
        "--unlabeled-id",
        type=int,
        default=5,
        help="Dataset id considered 'unlabeled' (default: 5, matches current config)",
    )
    ap.add_argument("--topk", type=int, default=20, help="Show top-K ADE labels within unlabeled pixels")

    args = ap.parse_args()

    img_path: Path = args.image
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    engine = Mask2FormerADEEngine()
    engine.ensure_loaded()

    id2label = engine.id2label
    cm = ClassMap.from_yaml(args.class_map, id2label)

    rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    ade = engine.predict_ade_ids(rgb)

    lbl = GeneratorPipeline._remap_ade_to_dataset_ids(ade, cm)
    lbl = GeneratorPipeline._postprocess_road_sidewalk_boundary(ade=ade, lbl=lbl, cm=cm, kernel_size=int(args.kernel))

    unlabeled_id = int(args.unlabeled_id)
    unlabeled_mask = lbl == np.uint8(unlabeled_id)

    total = int(lbl.size)
    unlabeled = int(unlabeled_mask.sum())
    ratio = 0.0 if total == 0 else (unlabeled / total)

    print(f"image: {img_path}")
    print(f"class_map: {args.class_map}")
    print(f"total_pixels: {total}")
    print(f"unlabeled_pixels: {unlabeled} ({ratio:.2%})")

    if unlabeled == 0:
        print("No unlabeled pixels.")
        return 0

    ade_unlabeled = ade[unlabeled_mask]

    counts: Dict[int, int] = {}
    for x in ade_unlabeled.reshape(-1):
        xi = int(x)
        counts[xi] = counts.get(xi, 0) + 1

    print("\nTop ADE labels within unlabeled pixels:")
    for ade_id, c in _topk(counts, int(args.topk)):
        name = id2label.get(int(ade_id), "<unknown>")
        pct = c / max(1, unlabeled)
        print(f"  ade_id={ade_id:3d}  count={c:8d}  pct={pct:6.2%}  name={name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
