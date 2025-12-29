from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

import supervision as sv

from soccer_cv.types import Detection, Track

@dataclass
class ByteTrackTracker:
    track_thresh: float = 0.25
    track_buffer: int = 45
    match_thresh: float = 0.8
    frame_rate: int = 30

    def __post_init__(self) -> None:
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.track_thresh,
            lost_track_buffer=self.track_buffer,
            minimum_matching_threshold=self.match_thresh,
            frame_rate=self.frame_rate,
        )

    def update(self, detections: List[Detection]) -> List[Track]:
        if not detections:
            # still need to update tracker with empty detections
            sv_det = sv.Detections.empty()
            tracked = self.tracker.update_with_detections(sv_det)
            return []

        xyxy = np.array([d.xyxy for d in detections], dtype=float)
        conf = np.array([d.conf for d in detections], dtype=float)
        # ByteTrack doesn't need class ids for our use (single class: player)
        class_id = np.zeros((len(detections),), dtype=int)
        sv_det = sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)

        tracked = self.tracker.update_with_detections(sv_det)
        out: List[Track] = []
        if tracked.tracker_id is None:
            return out

        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = tracked.xyxy[i].tolist()
            c = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
            out.append(Track(track_id=tid, xyxy=(x1, y1, x2, y2), conf=c, cls="player"))
        return out
