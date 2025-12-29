#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class PreparedPaths:
    dataset_root: Path
    train_images: Path
    val_images: Path
    train_labels: Path
    val_labels: Path
    data_yaml: Path


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists():
            dst.unlink()
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _extract_if_zip(path: Path) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    if path.is_file() and path.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="cvat_yolo_")
        out = Path(tmp.name)
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(out)
        return out, tmp
    return path, None


def _find_export_assets(export_root: Path) -> Tuple[List[Path], Dict[str, Path]]:
    """
    Returns:
      images: list of image paths found inside export_root (may be empty)
      labels_by_stem: mapping stem -> label .txt path
    """
    all_files = list(export_root.rglob("*"))
    images = [p for p in all_files if p.is_file() and _is_image(p)]

    # Candidate label txts: exclude obj.names/obj.data/train.txt etc by checking content pattern
    labels_by_stem: Dict[str, Path] = {}
    for p in all_files:
        if not (p.is_file() and p.suffix.lower() == ".txt"):
            continue
        name = p.name.lower()
        if name in {"train.txt", "valid.txt", "test.txt", "obj.names", "obj.data", "classes.txt"}:
            continue

        # Accept empty label files OR files where first non-empty line starts with an int class id
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue

        if txt == "":
            labels_by_stem[p.stem] = p
            continue

        first = txt.splitlines()[0].strip()
        if len(first) > 0 and first[0].isdigit():
            labels_by_stem[p.stem] = p

    return images, labels_by_stem


def _load_images_from_dir(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"crops-dir does not exist: {images_dir}")
    imgs = [p for p in images_dir.iterdir() if p.is_file() and _is_image(p)]
    imgs.sort()
    if not imgs:
        raise RuntimeError(f"No images found in crops-dir: {images_dir}")
    return imgs


def _write_data_yaml(out_root: Path, class_name: str) -> Path:
    data_yaml = out_root / "data.yaml"
    # Ultralytics format
    content = (
        f"path: {out_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: {class_name}\n"
    )
    data_yaml.write_text(content, encoding="utf-8")
    return data_yaml


def _split(items: List[Path], val_split: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_val = int(round(n * float(val_split)))
    n_val = max(1, min(n - 1, n_val)) if n >= 2 else 0
    val = items[:n_val]
    train = items[n_val:]
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare an Ultralytics YOLO dataset from CVAT YOLO 1.1 export + crops.")
    ap.add_argument("--cvat-export", required=True, help="Path to CVAT YOLO 1.1 export .zip or extracted folder")
    ap.add_argument("--crops-dir", default=None, help="Directory containing crop images (if export zip does not include images)")
    ap.add_argument("--out", required=True, help="Output dataset root, e.g. datasets/ball")
    ap.add_argument("--class-name", default="ball", help="Class name for data.yaml (default: ball)")
    ap.add_argument("--val-split", type=float, default=0.15, help="Validation split fraction (default: 0.15)")
    ap.add_argument("--seed", type=int, default=13, help="Random seed for train/val split")
    ap.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy", help="How to place images/labels into dataset")
    args = ap.parse_args()

    export_path = Path(args.cvat_export).expanduser().resolve()
    export_root, tmp = _extract_if_zip(export_path)

    out_root = Path(args.out).expanduser().resolve()
    train_images = out_root / "images" / "train"
    val_images = out_root / "images" / "val"
    train_labels = out_root / "labels" / "train"
    val_labels = out_root / "labels" / "val"
    for p in (train_images, val_images, train_labels, val_labels):
        _safe_mkdir(p)

    export_images, labels_by_stem = _find_export_assets(export_root)

    if export_images:
        images = sorted(export_images)
        source_images_desc = f"CVAT export images ({len(images)})"
    else:
        if args.crops_dir is None:
            raise RuntimeError(
                "CVAT export did not include images. Provide --crops-dir pointing to your crop images folder."
            )
        images = _load_images_from_dir(Path(args.crops_dir).expanduser().resolve())
        source_images_desc = f"crops-dir images ({len(images)})"

    train_imgs, val_imgs = _split(images, args.val_split, args.seed)

    # Copy images + labels; create empty label file if none exists.
    def place_one(img_path: Path, split: str) -> Tuple[bool, bool]:
        # returns (has_label_file, is_positive)
        if split == "train":
            img_dst = train_images / img_path.name
            lbl_dst = train_labels / f"{img_path.stem}.txt"
        else:
            img_dst = val_images / img_path.name
            lbl_dst = val_labels / f"{img_path.stem}.txt"

        _copy_or_link(img_path, img_dst, args.mode)

        src_lbl = labels_by_stem.get(img_path.stem)
        if src_lbl is None:
            lbl_dst.write_text("", encoding="utf-8")
            return False, False

        _copy_or_link(src_lbl, lbl_dst, args.mode if args.mode != "hardlink" else "copy")

        # Determine if positive
        try:
            t = lbl_dst.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            t = ""
        return True, (t != "")

    stats = {"train": {"n": 0, "labeled": 0, "pos": 0}, "val": {"n": 0, "labeled": 0, "pos": 0}}

    for img in train_imgs:
        has_lbl, is_pos = place_one(img, "train")
        stats["train"]["n"] += 1
        stats["train"]["labeled"] += int(has_lbl)
        stats["train"]["pos"] += int(is_pos)

    for img in val_imgs:
        has_lbl, is_pos = place_one(img, "val")
        stats["val"]["n"] += 1
        stats["val"]["labeled"] += int(has_lbl)
        stats["val"]["pos"] += int(is_pos)

    data_yaml = _write_data_yaml(out_root, args.class_name)

    print("\n=== Prepared dataset ===")
    print(f"Source images: {source_images_desc}")
    print(f"CVAT labels discovered: {len(labels_by_stem)}")
    print(f"Output dataset: {out_root}")
    print(f"data.yaml: {data_yaml}")
    print(f"Train: {stats['train']['n']} images | positives: {stats['train']['pos']} | labeled files found: {stats['train']['labeled']}")
    print(f"Val:   {stats['val']['n']} images | positives: {stats['val']['pos']} | labeled files found: {stats['val']['labeled']}")
    print("\nNotes:")
    print("- Images with no ball are kept and get an EMPTY .txt label file (this is correct).")
    print("- If CVAT didn’t export empty txts for unlabeled images, this script creates them.")

    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
