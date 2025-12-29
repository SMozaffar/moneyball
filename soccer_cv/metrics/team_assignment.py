from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import numpy as np
import cv2
from sklearn.cluster import KMeans

from soccer_cv.types import Track, xyxy_to_int, clamp_xyxy

@dataclass
class OnlineTeamAssigner:
    enabled: bool = True
    sample_top_fraction: float = 0.20
    sample_mid_fraction: float = 0.55
    exclude_green: bool = True
    green_exclude_h_min: int = 28
    green_exclude_h_max: int = 95
    green_exclude_s_min: int = 40
    green_exclude_v_min: int = 40

    min_samples_to_fit: int = 80
    refit_every_n_frames: int = 120
    random_state: int = 13

    # internal
    _samples: List[np.ndarray] = field(default_factory=list)
    _track_feat_sum: Dict[int, np.ndarray] = field(default_factory=dict)
    _track_feat_n: Dict[int, int] = field(default_factory=dict)
    _kmeans: Optional[KMeans] = None
    _frame_counter: int = 0

    def _extract_jersey_feature(self, frame_bgr: np.ndarray, track: Track) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = clamp_xyxy(track.xyxy, w, h)
        ix1, iy1, ix2, iy2 = map(int, [x1, y1, x2, y2])
        bw = max(1, ix2 - ix1)
        bh = max(1, iy2 - iy1)

        # Crop a “jersey-ish” region: upper-to-mid torso.
        y_top = int(round(iy1 + self.sample_top_fraction * bh))
        y_mid = int(round(iy1 + self.sample_mid_fraction * bh))
        x_l = int(round(ix1 + 0.20 * bw))
        x_r = int(round(ix1 + 0.80 * bw))

        y_top = max(0, min(h - 1, y_top))
        y_mid = max(0, min(h, y_mid))
        x_l = max(0, min(w - 1, x_l))
        x_r = max(0, min(w, x_r))
        if (y_mid - y_top) < 4 or (x_r - x_l) < 4:
            return None

        crop = frame_bgr[y_top:y_mid, x_l:x_r]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        if self.exclude_green:
            hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            green = ((hh >= self.green_exclude_h_min) & (hh <= self.green_exclude_h_max) &
                     (ss >= self.green_exclude_s_min) & (vv >= self.green_exclude_v_min))
            mask = (~green).astype(np.uint8)
            if mask.sum() < 25:
                return None
            # Use masked median for robustness.
            h_med = float(np.median(hh[mask.astype(bool)]))
            s_med = float(np.median(ss[mask.astype(bool)]))
            v_med = float(np.median(vv[mask.astype(bool)]))
        else:
            h_med = float(np.median(hsv[:, :, 0]))
            s_med = float(np.median(hsv[:, :, 1]))
            v_med = float(np.median(hsv[:, :, 2]))

        # Normalize to [0,1] range-ish
        feat = np.array([h_med / 179.0, s_med / 255.0, v_med / 255.0], dtype=np.float32)
        return feat

    def update(self, frame_bgr: np.ndarray, tracks: List[Track]) -> None:
        if not self.enabled or not tracks:
            self._frame_counter += 1
            return

        self._frame_counter += 1

        for tr in tracks:
            feat = self._extract_jersey_feature(frame_bgr, tr)
            if feat is None:
                continue
            self._samples.append(feat)
            if tr.track_id not in self._track_feat_sum:
                self._track_feat_sum[tr.track_id] = feat.copy()
                self._track_feat_n[tr.track_id] = 1
            else:
                self._track_feat_sum[tr.track_id] += feat
                self._track_feat_n[tr.track_id] += 1

        # Fit / refit KMeans periodically (after we have enough samples).
        if len(self._samples) >= self.min_samples_to_fit:
            if (self._kmeans is None) or (self._frame_counter % self.refit_every_n_frames == 0):
                X = np.stack(self._samples, axis=0)
                self._kmeans = KMeans(n_clusters=2, random_state=self.random_state, n_init="auto")
                self._kmeans.fit(X)

        # Assign teams to tracks based on current KMeans.
        if self._kmeans is not None:
            for tr in tracks:
                n = self._track_feat_n.get(tr.track_id, 0)
                if n <= 0:
                    continue
                mean_feat = self._track_feat_sum[tr.track_id] / float(n)
                team = int(self._kmeans.predict(mean_feat.reshape(1, -1))[0])
                tr.team_id = team
