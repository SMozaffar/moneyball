#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2


def _read_labels(lbl_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not lbl_path.exists():
        return []
    txt = lbl_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return []
    out = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
        out.append((cls, xc, yc, w, h))
    return out


def _draw_yolo_box(img, xc, yc, w, h):
    H, W = img.shape[:2]
    x1 = int((xc - w / 2.0) * W)
    y1 = int((yc - h / 2.0) * H)
    x2 = int((xc + w / 2.0) * W)
    y2 = int((yc + h / 2.0) * H)
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset root (contains images/ and labels/)")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", required=True, help="Output folder for previews")
    args = ap.parse_args()

    ds = Path(args.dataset).expanduser().resolve()
    img_dir = ds / "images" / args.split
    lbl_dir = ds / "labels" / args.split
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in img_dir.iterdir() if p.is_file()])
    if not imgs:
        raise RuntimeError(f"No images found in {img_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(imgs)
    imgs = imgs[: min(len(imgs), args.n)]

    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        lbl = lbl_dir / f"{p.stem}.txt"
        boxes = _read_labels(lbl)
        for _, xc, yc, w, h in boxes:
            _draw_yolo_box(img, xc, yc, w, h)
        cv2.imwrite(str(out_dir / p.name), img)

    print(f"Wrote {len(imgs)} previews to {out_dir}")


if __name__ == "__main__":
    main()