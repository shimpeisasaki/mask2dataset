from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Pair:
    relpath: Path
    img_path: Path
    ann_path: Path


def _iter_image_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _collect_pairs(img_train: Path, ann_train: Path) -> List[Pair]:
    pairs: List[Pair] = []
    for img_path in _iter_image_files(img_train):
        rel = img_path.relative_to(img_train)
        ann_path = ann_train / rel
        if ann_path.exists() and ann_path.is_file():
            pairs.append(Pair(relpath=rel, img_path=img_path, ann_path=ann_path))
    return pairs


def _copy_pair(pair: Pair, out_root: Path, split: str) -> None:
    out_img = out_root / "img_dir" / split / pair.relpath
    out_ann = out_root / "ann_dir" / split / pair.relpath
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_ann.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pair.img_path, out_img)
    shutil.copy2(pair.ann_path, out_ann)


def _ensure_dataset_dirs(out_root: Path) -> None:
    for split in ("train", "val"):
        (out_root / "img_dir" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "ann_dir" / split).mkdir(parents=True, exist_ok=True)


def _sample_one_split(
    img_split_dir: Path,
    ann_split_dir: Path,
    out_root: Path,
    split: str,
    n: int,
    seed: int,
    dry_run: bool,
    logger: Callable[[str], None],
) -> List[Pair]:
    if not img_split_dir.exists() or not img_split_dir.is_dir():
        raise FileNotFoundError(f"img_dir/{split} directory not found: {img_split_dir}")
    if not ann_split_dir.exists() or not ann_split_dir.is_dir():
        raise FileNotFoundError(f"ann_dir/{split} directory not found: {ann_split_dir}")
    if n < 0:
        raise ValueError(f"num_{split} must be 0 or a positive integer")

    pairs = _collect_pairs(img_train=img_split_dir, ann_train=ann_split_dir)
    if n > len(pairs):
        raise ValueError(f"Requested num_{split}={n}, but only {len(pairs)} matched pairs exist")

    rnd = random.Random(int(seed))
    selected = rnd.sample(pairs, k=n) if n > 0 else []

    logger(f"[{split}] matched pairs: {len(pairs)}")
    logger(f"[{split}] sampling: {n} (seed={seed})")

    if dry_run:
        for p in selected:
            logger(f"[DRY-RUN][{split}] {p.relpath}")
        return selected

    for p in selected:
        _copy_pair(p, out_root, split=split)

    logger(f"[{split}] copied: {len(selected)}")
    return selected


def sample_dataset_splits(
    dataset_root: Path,
    out_root: Path,
    num_train: int,
    num_val: int,
    seed: int = 42,
    dry_run: bool = False,
    logger: Callable[[str], None] = print,
) -> Dict[str, List[Pair]]:
    dataset_root = dataset_root.resolve()
    out_root = out_root.resolve()

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    _ensure_dataset_dirs(out_root)
    logger(f"Dataset root: {dataset_root}")
    logger(f"Output root: {out_root}")

    sampled_train = _sample_one_split(
        img_split_dir=dataset_root / "img_dir" / "train",
        ann_split_dir=dataset_root / "ann_dir" / "train",
        out_root=out_root,
        split="train",
        n=int(num_train),
        seed=int(seed),
        dry_run=dry_run,
        logger=logger,
    )
    sampled_val = _sample_one_split(
        img_split_dir=dataset_root / "img_dir" / "val",
        ann_split_dir=dataset_root / "ann_dir" / "val",
        out_root=out_root,
        split="val",
        n=int(num_val),
        seed=int(seed) + 1,
        dry_run=dry_run,
        logger=logger,
    )

    logger("Done.")
    logger(f"Copied train={len(sampled_train)}, val={len(sampled_val)}")
    logger(f"  {out_root / 'img_dir' / 'train'}")
    logger(f"  {out_root / 'img_dir' / 'val'}")
    logger(f"  {out_root / 'ann_dir' / 'train'}")
    logger(f"  {out_root / 'ann_dir' / 'val'}")
    return {"train": sampled_train, "val": sampled_val}


def sample_and_copy_pairs(
    img_train: Path,
    ann_train: Path,
    out_root: Path,
    n: int,
    seed: int = 42,
    dry_run: bool = False,
    logger: Callable[[str], None] = print,
) -> List[Pair]:
    img_train = img_train.resolve()
    ann_train = ann_train.resolve()
    out_root = out_root.resolve()

    if not img_train.exists() or not img_train.is_dir():
        raise FileNotFoundError(f"img-train directory not found: {img_train}")
    if not ann_train.exists() or not ann_train.is_dir():
        raise FileNotFoundError(f"ann-train directory not found: {ann_train}")
    if n <= 0:
        raise ValueError("num must be a positive integer")

    _ensure_dataset_dirs(out_root)

    pairs = _collect_pairs(img_train=img_train, ann_train=ann_train)
    if not pairs:
        raise RuntimeError("No matched pairs found between img-train and ann-train")
    if n > len(pairs):
        raise ValueError(f"Requested num={n}, but only {len(pairs)} matched pairs exist")

    rnd = random.Random(int(seed))
    selected = rnd.sample(pairs, k=n)

    logger(f"Found matched pairs: {len(pairs)}")
    logger(f"Sampling: {n} (seed={seed})")
    logger(f"Output root: {out_root}")

    if dry_run:
        for p in selected:
            logger(f"[DRY-RUN] {p.relpath}")
        return selected

    for p in selected:
        _copy_pair(p, out_root, split="train")

    logger("Done.")
    logger(f"Copied {n} pairs into:")
    logger(f"  {out_root / 'img_dir' / 'train'}")
    logger(f"  {out_root / 'ann_dir' / 'train'}")
    return selected


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Randomly sample paired files from img_dir/train and ann_dir/train, "
            "then copy them into another dataset-style output folder."
        )
    )
    ap.add_argument("--img-train", type=Path, help="Path to source img_dir/train")
    ap.add_argument("--ann-train", type=Path, help="Path to source ann_dir/train")
    ap.add_argument("--dataset-root", type=Path, help="Path to source dataset root containing img_dir and ann_dir")
    ap.add_argument("--out-root", type=Path, required=True, help="Destination root directory")
    ap.add_argument("-n", "--num", type=int, help="Number of train pairs to sample (legacy mode)")
    ap.add_argument("--num-train", type=int, default=0, help="Number of train pairs to sample")
    ap.add_argument("--num-val", type=int, default=0, help="Number of val pairs to sample")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--dry-run", action="store_true", help="Print planned copies without writing files")

    args = ap.parse_args()

    if args.dataset_root is not None:
        sample_dataset_splits(
            dataset_root=args.dataset_root,
            out_root=args.out_root,
            num_train=int(args.num_train),
            num_val=int(args.num_val),
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
            logger=print,
        )
        return 0

    if args.img_train is None or args.ann_train is None or args.num is None:
        raise ValueError("Either --dataset-root mode, or (--img-train --ann-train -n) is required")

    sample_and_copy_pairs(
        img_train=args.img_train,
        ann_train=args.ann_train,
        out_root=args.out_root,
        n=int(args.num),
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        logger=print,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
