from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

from soccer_cv.types import Track, Ball

@dataclass
class PossessionEstimator:
    max_ball_to_player_px: float = 95.0
    min_frames_to_confirm: int = 3
    hold_last_seconds_if_ball_missing: float = 1.2
    analysis_fps: float = 8.0

    # internal
    _candidate_player: int = -1
    _candidate_team: int = -1
    _candidate_count: int = 0
    _current_player: int = -1
    _current_team: int = -1
    _ball_missing_frames: int = 0

    def reset(self) -> None:
        self._candidate_player = -1
        self._candidate_team = -1
        self._candidate_count = 0
        self._current_player = -1
        self._current_team = -1
        self._ball_missing_frames = 0

    def _nearest_player(self, ball_xy: Tuple[float, float], tracks: List[Track]) -> Tuple[int, int, float]:
        bx, by = ball_xy
        best_id, best_team, best_d = -1, -1, float("inf")
        for tr in tracks:
            px, py = tr.feet
            d = float(np.hypot(bx - px, by - py))
            if d < best_d:
                best_d = d
                best_id = tr.track_id
                best_team = -1 if tr.team_id is None else int(tr.team_id)
        return best_id, best_team, best_d

    def update(self, ball: Optional[Ball], tracks: List[Track]) -> Tuple[int, int]:
        """Returns (team_id, player_track_id). -1 indicates unknown."""
        # If no ball, hold last possession for some time.
        if ball is None:
            self._ball_missing_frames += 1
            hold_frames = int(round(self.hold_last_seconds_if_ball_missing * self.analysis_fps))
            if self._ball_missing_frames <= hold_frames:
                return self._current_team, self._current_player
            return -1, -1

        self._ball_missing_frames = 0

        if not tracks:
            return -1, -1

        pid, team, d = self._nearest_player((ball.x, ball.y), tracks)
        if d > self.max_ball_to_player_px:
            # ball too far from any player
            return -1, -1

        # hysteresis: require stable candidate for min_frames_to_confirm
        if pid == self._candidate_player and team == self._candidate_team:
            self._candidate_count += 1
        else:
            self._candidate_player = pid
            self._candidate_team = team
            self._candidate_count = 1

        if self._candidate_count >= self.min_frames_to_confirm:
            self._current_player = pid
            self._current_team = team

        return self._current_team, self._current_player
