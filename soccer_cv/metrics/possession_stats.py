from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, DefaultDict
from collections import defaultdict

from soccer_cv.types import FrameResult

@dataclass
class PossessionStats:
    dt_sec: float
    team_seconds: DefaultDict[int, float] = field(default_factory=lambda: defaultdict(float))
    player_seconds: DefaultDict[int, float] = field(default_factory=lambda: defaultdict(float))

    def on_frame(self, fr: FrameResult) -> None:
        if fr.possession_team is not None and fr.possession_team >= 0:
            self.team_seconds[int(fr.possession_team)] += self.dt_sec
        if fr.possession_player is not None and fr.possession_player >= 0:
            self.player_seconds[int(fr.possession_player)] += self.dt_sec

    def finalize(self) -> Dict[str, Any]:
        return {
            "team_seconds": dict(self.team_seconds),
            "player_seconds": dict(self.player_seconds),
        }
