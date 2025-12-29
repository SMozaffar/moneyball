from __future__ import annotations

import cv2
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple
import numpy as np

@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int

def get_video_info(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return VideoInfo(path=path, fps=fps, width=width, height=height, frame_count=frame_count)

def iter_frames(path: str, step: int = 1, start: int = 0, end: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if end is None:
        end = total - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    while idx <= end:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        # skip
        for _ in range(step - 1):
            ok = cap.grab()
            if not ok:
                break
            idx += 1
        idx += 1

    cap.release()

def make_writer(path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, size)

def resize_keep_aspect(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
