from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    conf: float
    cls: str

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

@dataclass
class Track:
    track_id: int
    xyxy: Tuple[float, float, float, float]
    conf: float
    cls: str = "player"
    team_id: Optional[int] = None
    foot_px: Optional[Tuple[float, float]] = None
    foot_field_m: Optional[Tuple[float, float]] = None

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @property
    def feet(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (0.5 * (x1 + x2), y2)

@dataclass
class Ball:
    x: float
    y: float
    conf: float
    source: str  # "yolo" or "motion"
    field_m: Optional[Tuple[float, float]] = None

@dataclass
class FrameResult:
    frame_idx: int
    t_sec: float
    tracks: List[Track]
    ball: Optional[Ball]
    possession_team: int  # -1 unknown
    possession_player: int  # -1 unknown

def xyxy_to_int(xyxy: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def clamp_xyxy(xyxy: Tuple[float, float, float, float], w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    x1 = float(max(0, min(w - 1, x1)))
    x2 = float(max(0, min(w - 1, x2)))
    y1 = float(max(0, min(h - 1, y1)))
    y2 = float(max(0, min(h - 1, y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)
