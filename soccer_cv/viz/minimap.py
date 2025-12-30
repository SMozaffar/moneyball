from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np
from tqdm import tqdm

from soccer_cv.logging_utils import get_logger
from soccer_cv.render import _tracks_from_frame_dict, _ball_from_frame_dict
from soccer_cv.types import Track, Ball
from soccer_cv.utils.video_io import get_video_info, make_writer
from soccer_cv.viz.annotate import TEAM_COLORS, TrailState

logger = get_logger()


def _pitch_canvas(width: int, height: int) -> np.ndarray:
    """Creates a green pitch canvas with simple markings."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (20, 80, 20)
    cv2.rectangle(canvas, (6, 6), (width - 7, height - 7), (230, 230, 230), 2)
    mid_y = height // 2
    cv2.line(canvas, (6, mid_y), (width - 7, mid_y), (230, 230, 230), 1)
    cv2.circle(canvas, (width // 2, mid_y), max(8, int(min(width, height) * 0.05)), (230, 230, 230), 1)
    cv2.circle(canvas, (width // 2, mid_y), 3, (230, 230, 230), -1)
    return canvas


def _field_to_px(pt_m: tuple[float, float], pitch_w_m: float, pitch_l_m: float, width_px: int, height_px: int) -> tuple[int, int]:
    x_m, y_m = pt_m
    x = int(np.clip(round(x_m / pitch_w_m * (width_px - 1)), 0, width_px - 1))
    y = int(np.clip(round(y_m / pitch_l_m * (height_px - 1)), 0, height_px - 1))
    return x, y


def _draw_frame(
    base: np.ndarray,
    tracks: list[Track],
    ball: Optional[Ball],
    pitch_w_m: float,
    pitch_l_m: float,
    trail: Optional[TrailState],
    update_trail: bool,
    show_possession: bool,
    possession_team: int,
    possession_player: int,
) -> np.ndarray:
    canvas = base.copy()
    h, w = canvas.shape[:2]

    def map_pt(pt_m: tuple[float, float]) -> tuple[int, int]:
        return _field_to_px(pt_m, pitch_w_m, pitch_l_m, w, h)

    # Trails
    if trail is not None:
        if update_trail:
            for tr in tracks:
                if tr.foot_field_m is None:
                    continue
                trail.push(tr.track_id, map_pt(tr.foot_field_m))

        for _, pts in trail.points.items():
            if len(pts) < 2:
                continue
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], (90, 90, 90), 2)

    # Players
    for tr in tracks:
        if tr.foot_field_m is None:
            continue
        px, py = map_pt(tr.foot_field_m)
        color = TEAM_COLORS.get(-1 if tr.team_id is None else int(tr.team_id), (200, 200, 200))
        radius = 9 if tr.track_id == possession_player else 7
        cv2.circle(canvas, (px, py), radius + 2, (30, 30, 30), 1)
        cv2.circle(canvas, (px, py), radius, color, -1)

    # Ball
    if ball is not None and ball.field_m is not None:
        bx, by = map_pt(ball.field_m)
        cv2.circle(canvas, (bx, by), 6, (0, 0, 0), -1)
        cv2.circle(canvas, (bx, by), 10, (255, 255, 255), 2)

    if show_possession:
        hud_color = TEAM_COLORS.get(possession_team, (255, 255, 255))
        text = f"Possession: Team {possession_team}"
        cv2.putText(canvas, text, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, hud_color, 2, cv2.LINE_AA)

    return canvas


def render_minimap_video(
    video_path: str,
    analysis: Dict[str, Any],
    out_video: str,
    width_px: int = 640,
    height_px: Optional[int] = None,
    fps: Optional[float] = None,
    full_speed: bool = True,
    draw_trails: bool = True,
    trail_seconds: float = 3.0,
    show_possession: bool = True,
) -> None:
    meta = analysis.get("meta", {})
    if not meta.get("field_homography_enabled", False):
        logger.warning("Mini-map render skipped because field homography was disabled for this analysis.")
        return

    frames = analysis.get("frames", [])
    by_idx = {int(f["frame_idx"]): f for f in frames}
    if not by_idx:
        raise ValueError("No frames in analysis for mini-map render.")
    sorted_indices = sorted(by_idx.keys())

    field_cfg = (meta.get("config") or {}).get("field", {}) if isinstance(meta.get("config"), dict) else {}
    pitch_length_m = float(field_cfg.get("pitch_length_m", 105.0))
    pitch_width_m = float(field_cfg.get("pitch_width_m", 68.0))

    info = get_video_info(video_path)
    out_fps = float(info.fps if fps is None else fps)
    total_frames = int(info.frame_count)
    if full_speed and total_frames <= 0:
        logger.warning("Video frame count is zero; rendering mini-map using analyzed frames only.")
        full_speed = False
        total_frames = len(sorted_indices)
    target_w = int(width_px)
    target_h = int(height_px) if height_px is not None else int(round(target_w * pitch_length_m / pitch_width_m))

    writer = make_writer(out_video, fps=out_fps, size=(target_w, target_h))
    base = _pitch_canvas(target_w, target_h)

    trail = None
    if draw_trails and trail_seconds > 0:
        trail_len = max(2, int(round(out_fps * float(trail_seconds))))
        trail = TrailState(max_len=trail_len)

    # Rendering loop mirrors annotated video: keep duration at original FPS using latest analysis frame.
    pbar_total = total_frames if full_speed else len(sorted_indices)
    pbar = tqdm(total=pbar_total, desc="Rendering(minimap)", unit="frame")

    if full_speed:
        j = 0
        current_fd = by_idx[sorted_indices[0]]
        for idx in range(total_frames):
            while j + 1 < len(sorted_indices) and sorted_indices[j + 1] <= idx:
                j += 1
                current_fd = by_idx[sorted_indices[j]]

            update_trail = (idx == sorted_indices[j])
            tracks = _tracks_from_frame_dict(current_fd)
            ball = _ball_from_frame_dict(current_fd)
            p_team = int(current_fd.get("possession_team", -1))
            p_player = int(current_fd.get("possession_player", -1))

            frame_img = _draw_frame(
                base,
                tracks,
                ball,
                pitch_width_m,
                pitch_length_m,
                trail,
                update_trail,
                show_possession,
                p_team,
                p_player,
            )
            writer.write(frame_img)
            pbar.update(1)
    else:
        for idx in sorted_indices:
            fd = by_idx[idx]
            tracks = _tracks_from_frame_dict(fd)
            ball = _ball_from_frame_dict(fd)
            p_team = int(fd.get("possession_team", -1))
            p_player = int(fd.get("possession_player", -1))
            frame_img = _draw_frame(
                base,
                tracks,
                ball,
                pitch_width_m,
                pitch_length_m,
                trail,
                True,
                show_possession,
                p_team,
                p_player,
            )
            writer.write(frame_img)
            pbar.update(1)
            if idx == sorted_indices[-1]:
                break

    pbar.close()
    writer.release()
    logger.info("Wrote mini-map video: %s", out_video)
