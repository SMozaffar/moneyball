from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import csv

from soccer_cv.types import FrameResult, Track, Ball

def frame_result_to_dict(fr: FrameResult) -> Dict[str, Any]:
    return {
        "frame_idx": fr.frame_idx,
        "t_sec": fr.t_sec,
        "tracks": [
            {
                "track_id": tr.track_id,
                "xyxy": list(tr.xyxy),
                "conf": tr.conf,
                "cls": tr.cls,
                "team_id": tr.team_id,
            }
            for tr in fr.tracks
        ],
        "ball": None if fr.ball is None else {"x": fr.ball.x, "y": fr.ball.y, "conf": fr.ball.conf, "source": fr.ball.source},
        "possession_team": fr.possession_team,
        "possession_player": fr.possession_player,
    }

def save_analysis(path: str | Path, meta: Dict[str, Any], frames: List[FrameResult]) -> None:
    path = Path(path)
    payload = {
        "meta": meta,
        "frames": [frame_result_to_dict(fr) for fr in frames],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def load_analysis(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))

def save_possession_timeline_csv(path: str | Path, frames: List[FrameResult]) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "frame_idx", "team_id", "player_track_id", "ball_x", "ball_y", "ball_conf", "ball_source"])
        for fr in frames:
            if fr.ball is None:
                w.writerow([f"{fr.t_sec:.3f}", fr.frame_idx, fr.possession_team, fr.possession_player, "", "", "", ""])
            else:
                w.writerow([f"{fr.t_sec:.3f}", fr.frame_idx, fr.possession_team, fr.possession_player,
                            f"{fr.ball.x:.1f}", f"{fr.ball.y:.1f}", f"{fr.ball.conf:.3f}", fr.ball.source])

def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
