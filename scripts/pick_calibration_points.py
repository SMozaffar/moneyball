# scripts/pick_calibration_points.py

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


Point2 = Tuple[float, float]


@dataclass
class Picked:
    image_points: List[Point2]
    field_points: List[Point2]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True, help="Path to video")
    ap.add_argument("--frame-index", type=int, default=0, help="Frame index to use for picking points")
    ap.add_argument("--out-yaml", type=str, default="", help="Optional path to write YAML snippet")
    ap.add_argument("--pitch-length", type=float, default=105.0)
    ap.add_argument("--pitch-width", type=float, default=68.0)
    return ap.parse_args()


def read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame index {frame_idx} from {video_path}")
    return frame


def main() -> None:
    args = parse_args()
    frame = read_frame(args.video, args.frame_index)
    disp = frame.copy()

    picked: Picked = Picked(image_points=[], field_points=[])

    win = "Pick calibration points (click) - press 'q' when done"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ix, iy = float(x), float(y)
        print(f"\nClicked image point: ({ix:.1f}, {iy:.1f})")
        print("Now enter FIELD point in meters for this same location.")
        print("Convention: x in [0..pitch_width], y in [0..pitch_length]")
        while True:
            raw = input("Enter field point as 'x y' (e.g. 13.84 16.5): ").strip()
            parts = raw.split()
            if len(parts) != 2:
                print("Please enter exactly two numbers.")
                continue
            try:
                fx, fy = float(parts[0]), float(parts[1])
            except ValueError:
                print("Invalid float(s). Try again.")
                continue
            break

        picked.image_points.append((ix, iy))
        picked.field_points.append((fx, fy))

        cv2.circle(disp, (int(ix), int(iy)), 6, (0, 0, 255), -1)
        cv2.putText(
            disp,
            f"{len(picked.image_points)}",
            (int(ix) + 8, int(iy) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(win, disp)

    cv2.setMouseCallback(win, on_mouse)

    print("\nInstructions:")
    print("- Click a point on the image.")
    print("- In the terminal, enter its matching pitch coord (meters).")
    print("- Repeat for >= 4 points (6-10 is better).")
    print("- Press 'q' when done.\n")

    cv2.imshow(win, disp)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    if len(picked.image_points) < 4:
        raise RuntimeError("Need at least 4 points. Re-run and pick more.")

    yaml = []
    yaml.append("field:")
    yaml.append("  enabled: true")
    yaml.append(f"  pitch_length_m: {args.pitch_length}")
    yaml.append(f"  pitch_width_m: {args.pitch_width}")
    yaml.append("  ransac_reproj_threshold: 4.0")
    yaml.append("  image_points:")
    for (x, y) in picked.image_points:
        yaml.append(f"    - [{x:.1f}, {y:.1f}]")
    yaml.append("  field_points:")
    for (x, y) in picked.field_points:
        yaml.append(f"    - [{x:.3f}, {y:.3f}]")

    yaml_text = "\n".join(yaml) + "\n"
    print("\n--- Paste into configs/default.yaml ---\n")
    print(yaml_text)

    if args.out_yaml:
        outp = Path(args.out_yaml)
        outp.write_text(yaml_text, encoding="utf-8")
        print(f"Wrote YAML snippet to: {outp}")


if __name__ == "__main__":
    main()
