from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

from soccer_cv.logging_utils import get_logger
from soccer_cv.utils.video_io import get_video_info, iter_frames, make_writer
from soccer_cv.viz.annotate import draw_tracks, TrailState
from soccer_cv.types import Track, Ball

logger = get_logger()

def _tracks_from_frame_dict(fd: Dict[str, Any]) -> List[Track]:
    out: List[Track] = []
    for tr in fd.get("tracks", []):
        out.append(Track(
            track_id=int(tr["track_id"]),
            xyxy=tuple(map(float, tr["xyxy"])),
            conf=float(tr.get("conf", 1.0)),
            cls=str(tr.get("cls", "player")),
            team_id=None if tr.get("team_id") is None else int(tr["team_id"]),
            foot_px=None if tr.get("foot_px") is None else tuple(map(float, tr["foot_px"])),
            foot_field_m=None if tr.get("foot_field_m") is None else tuple(map(float, tr["foot_field_m"])),
        ))
    return out

def _ball_from_frame_dict(fd: Dict[str, Any]) -> Optional[Ball]:
    b = fd.get("ball")
    if b is None:
        return None
    return Ball(
        x=float(b["x"]),
        y=float(b["y"]),
        conf=float(b.get("conf", 1.0)),
        source=str(b.get("source", "yolo")),
        field_m=None if b.get("field_m") is None else tuple(map(float, b["field_m"])),
    )

def render_annotated_video(
    video_path: str,
    analysis: Dict[str, Any],
    out_video: str,
    fps: Optional[float] = None,
    full_speed: bool = True,
    show_text: bool = True,
    draw_trails: bool = True,
    trail_seconds: float = 1.5,
    draw_boxes: bool = True,
    draw_feet_points: bool = True,
    draw_ball: bool = True,
) -> None:
    info = get_video_info(video_path)
    out_fps = float(info.fps) if fps is None else float(fps)

    frames = analysis.get("frames", [])
    by_idx = {int(f["frame_idx"]): f for f in frames}
    sorted_indices = sorted(by_idx.keys())
    if not sorted_indices:
        raise ValueError("No frames in analysis")

    writer = make_writer(out_video, fps=out_fps, size=(info.width, info.height))

    # Trail length in frames (seconds -> frames)
    trail = None
    if draw_trails and trail_seconds > 0:
        max_len = max(2, int(round(out_fps * float(trail_seconds))))
        trail = TrailState(max_len=max_len)

    if full_speed:
        # Render every frame, overlay most recent analysis result (keeps normal playback duration).
        j = 0
        current_fd = by_idx[sorted_indices[j]]

        pbar = tqdm(total=int(info.frame_count), desc="Rendering(full_speed)", unit="frame")
        for idx, frame_bgr in iter_frames(video_path, step=1):
            # advance current analysis frame pointer
            while j + 1 < len(sorted_indices) and sorted_indices[j + 1] <= idx:
                j += 1
                current_fd = by_idx[sorted_indices[j]]

            update_trail = (idx == sorted_indices[j])

            tracks = _tracks_from_frame_dict(current_fd)
            ball = _ball_from_frame_dict(current_fd)
            p_team = int(current_fd.get("possession_team", -1))
            p_player = int(current_fd.get("possession_player", -1))

            out = draw_tracks(
                frame_bgr,
                tracks,
                ball if draw_ball else None,
                p_team,
                p_player,
                show_text=show_text,
                trail=trail,
                update_trail=update_trail,
                draw_boxes=draw_boxes,
                draw_feet_points=draw_feet_points,
                draw_ball=draw_ball,
            )

            writer.write(out)
            pbar.update(1)

        pbar.close()
    else:
        # Old behavior: only render analyzed frames
        pbar = tqdm(total=len(sorted_indices), desc="Rendering(sampled)", unit="frame")
        for idx, frame_bgr in iter_frames(video_path, step=1):
            if idx not in by_idx:
                continue
            fd = by_idx[idx]
            tracks = _tracks_from_frame_dict(fd)
            ball = _ball_from_frame_dict(fd)
            p_team = int(fd.get("possession_team", -1))
            p_player = int(fd.get("possession_player", -1))
            out = draw_tracks(
                frame_bgr,
                tracks,
                ball if draw_ball else None,
                p_team,
                p_player,
                show_text=show_text,
                trail=trail,
                update_trail=True,
                draw_boxes=draw_boxes,
                draw_feet_points=draw_feet_points,
                draw_ball=draw_ball,
            )
            writer.write(out)
            pbar.update(1)
            if idx == sorted_indices[-1]:
                break
        pbar.close()

    writer.release()
    logger.info(f"Wrote annotated video: {out_video}")
