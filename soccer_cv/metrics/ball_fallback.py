from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import cv2

from soccer_cv.types import Ball


@dataclass
class MotionBallFallback:
    """A lightweight motion-based ball estimator.

    This is *not* a full ball tracker — it's a pragmatic fallback that works well enough to stabilize possession
    metrics when the ball detector is noisy or missing.

    It looks for small, roughly circular moving blobs between consecutive frames.
    """

    enabled: bool = True

    # Process at smaller scale for speed (0 < scale <= 1). Thresholds below are in SCALED pixel units.
    scale: float = 0.50

    diff_thresh: int = 12
    min_area: int = 2
    max_area: int = 80
    circularity_min: float = 0.28
    max_candidates: int = 25
    near_last_ball_px: int = 80

    _prev_gray: Optional[np.ndarray] = None
    _last_ball_xy_scaled: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self._prev_gray = None
        self._last_ball_xy_scaled = None

    @staticmethod
    def _circularity(cnt: np.ndarray) -> float:
        area = float(cv2.contourArea(cnt))
        if area <= 1e-6:
            return 0.0
        peri = float(cv2.arcLength(cnt, True))
        if peri <= 1e-6:
            return 0.0
        return 4.0 * np.pi * area / (peri * peri)

    def _prep(self, frame_bgr: np.ndarray, field_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        s = float(self.scale)
        if s <= 0.0 or s > 1.0:
            s = 1.0

        if s == 1.0:
            return frame_bgr, field_mask, 1.0

        h, w = frame_bgr.shape[:2]
        nw, nh = int(round(w * s)), int(round(h * s))
        frame_s = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

        mask_s = None
        if field_mask is not None:
            mask_s = cv2.resize(field_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        return frame_s, mask_s, s

    def find_candidates(self, frame_bgr: np.ndarray, field_mask: Optional[np.ndarray]) -> List[Tuple[float, float, float, float]]:
        """Return motion candidates as (score, x, y, area) in ORIGINAL (unscaled) pixel coords.

        This mirrors the internal logic used by update(), but returns multiple candidates for debugging/export.
        The method updates the internal previous-frame buffer, so it should be called in a streaming fashion.
        """
        if not self.enabled:
            return []

        frame_s, mask_s, s = self._prep(frame_bgr, field_mask)

        gray = cv2.cvtColor(frame_s, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return []

        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray

        if mask_s is not None:
            diff = cv2.bitwise_and(diff, diff, mask=mask_s)

        _, th = cv2.threshold(diff, int(self.diff_thresh), 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cands_s: List[Tuple[float, float, float, float]] = []  # (score, cx, cy, area) in scaled coords
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < float(self.min_area) or area > float(self.max_area):
                continue
            circ = self._circularity(cnt)
            if circ < float(self.circularity_min):
                continue
            M = cv2.moments(cnt)
            if abs(M.get("m00", 0.0)) < 1e-6:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

            score = float(circ)
            if self._last_ball_xy_scaled is not None:
                lx, ly = self._last_ball_xy_scaled
                d = float(np.hypot(cx - lx, cy - ly))
                if d < float(self.near_last_ball_px):
                    score += 0.75
                else:
                    score += max(0.0, 0.25 - (d / (4.0 * float(self.near_last_ball_px))))
            cands_s.append((score, cx, cy, area))

        if not cands_s:
            return []

        cands_s.sort(key=lambda t: t[0], reverse=True)
        cands_s = cands_s[: int(self.max_candidates)]

        out: List[Tuple[float, float, float, float]] = []
        for score, cx_s, cy_s, area_s in cands_s:
            out.append((float(score), float(cx_s / s), float(cy_s / s), float(area_s)))
        return out

    def update(self, frame_bgr: np.ndarray, field_mask: Optional[np.ndarray]) -> Optional[Ball]:
        if not self.enabled:
            return None

        cands = self.find_candidates(frame_bgr, field_mask)
        if not cands:
            return None

        score, cx, cy, _ = cands[0]

        # Update state in scaled coords (for the "near last ball" bonus)
        s = float(self.scale)
        if s <= 0.0 or s > 1.0:
            s = 1.0
        self._last_ball_xy_scaled = (float(cx * s), float(cy * s))

        conf = float(min(0.95, max(0.40, score / 2.0)))
        return Ball(x=float(cx), y=float(cy), conf=conf, source="motion")
