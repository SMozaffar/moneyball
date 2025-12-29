from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2

from soccer_cv.types import Track, Ball, xyxy_to_int

TEAM_COLORS = {
    0: (255, 200, 0),
    1: (0, 200, 255),
    -1: (200, 200, 200),
}

@dataclass
class TrailState:
    max_len: int = 20
    points: Dict[int, List[Tuple[int, int]]] = None

    def __post_init__(self):
        if self.points is None:
            self.points = {}

    def push(self, track_id: int, pt: Tuple[int, int]) -> None:
        if track_id not in self.points:
            self.points[track_id] = []
        self.points[track_id].append(pt)
        if len(self.points[track_id]) > self.max_len:
            self.points[track_id] = self.points[track_id][-self.max_len:]

def draw_tracks(
    frame: np.ndarray,
    tracks: List[Track],
    ball: Optional[Ball],
    possession_team: int,
    possession_player: int,
    show_text: bool = True,
    trail: Optional[TrailState] = None,
    update_trail: bool = True,
    draw_boxes: bool = True,
    draw_feet_points: bool = True,
    draw_ball: bool = True,
) -> np.ndarray:
    out = frame.copy()

    # trails (draw always if present; only update when update_trail=True)
    if trail is not None:
        if update_trail:
            for tr in tracks:
                cx, cy = tr.center
                trail.push(tr.track_id, (int(cx), int(cy)))

        for tid, pts in trail.points.items():
            if len(pts) < 2:
                continue
            for i in range(1, len(pts)):
                cv2.line(out, pts[i - 1], pts[i], (160, 160, 160), 2)

    for tr in tracks:
        x1, y1, x2, y2 = xyxy_to_int(tr.xyxy)
        team = -1 if tr.team_id is None else int(tr.team_id)
        color = TEAM_COLORS.get(team, (200, 200, 200))
        thickness = 3 if tr.track_id == possession_player else 2

        if draw_boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        if show_text:
            label = f"#{tr.track_id} T{team} {tr.conf:.2f}"
            cv2.putText(
                out,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        if draw_feet_points:
            fx, fy = tr.feet
            cv2.circle(out, (int(fx), int(fy)), 4, color, -1)

    if draw_ball and ball is not None:
        bx, by = int(ball.x), int(ball.y)
        cv2.circle(out, (bx, by), 6, (0, 0, 255), -1)
        cv2.circle(out, (bx, by), 12, (255, 255, 255), 2)
        if show_text:
            cv2.putText(
                out,
                f"BALL {ball.source} {ball.conf:.2f}",
                (bx + 10, by - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # HUD
    if show_text:
        team_color = TEAM_COLORS.get(possession_team, (255, 255, 255))
        hud = f"Possession: Team {possession_team}  Player {possession_player}"
        cv2.putText(out, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, team_color, 3, cv2.LINE_AA)

    return out
