# soccer_cv/field/homography.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:
    cv2 = None  # pragma: no cover


Point2 = Tuple[float, float]


@dataclass
class PitchSpec:
    """
    Pitch coordinate system (meters):
      - x: left -> right across width  (0 .. pitch_width_m)
      - y: top  -> bottom along length (0 .. pitch_length_m)

    You are free to define "top" as the end of the pitch that appears "up" in your frame.
    Just be consistent with your chosen field_points.
    """
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0


class PitchHomography:
    """
    Compute and apply a homography H that maps image pixels -> pitch coordinates (meters).

    Requires >= 4 non-collinear point correspondences:
      image_points[i] = (x_px, y_px)
      field_points[i] = (x_m,  y_m)
    """

    def __init__(
        self,
        pitch: PitchSpec,
        image_points: Sequence[Point2],
        field_points: Sequence[Point2],
        ransac_reproj_threshold: float = 4.0,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for homography but could not be imported.")
        if len(image_points) != len(field_points):
            raise ValueError("image_points and field_points must have the same length.")
        if len(image_points) < 4:
            raise ValueError("Need at least 4 point correspondences to compute a homography.")

        self.pitch = pitch
        self.image_points = np.array(image_points, dtype=np.float64)
        self.field_points = np.array(field_points, dtype=np.float64)
        self.ransac_reproj_threshold = float(ransac_reproj_threshold)

        self.H_img_to_field: Optional[np.ndarray] = None
        self.H_field_to_img: Optional[np.ndarray] = None

        self._compute()

    def _compute(self) -> None:
        assert cv2 is not None
        src = self.image_points.reshape(-1, 1, 2)  # px
        dst = self.field_points.reshape(-1, 1, 2)  # meters

        H, inliers = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_reproj_threshold)
        if H is None:
            raise RuntimeError("cv2.findHomography failed. Check that your points are correct and non-collinear.")

        self.H_img_to_field = H.astype(np.float64)
        self.H_field_to_img = np.linalg.inv(self.H_img_to_field)

    def pixel_to_field(self, xy_px: Point2) -> Point2:
        if self.H_img_to_field is None:
            raise RuntimeError("Homography not initialized.")
        x, y = float(xy_px[0]), float(xy_px[1])
        p = np.array([[x, y, 1.0]], dtype=np.float64).T
        q = self.H_img_to_field @ p
        q = q / (q[2, 0] + 1e-12)
        return (float(q[0, 0]), float(q[1, 0]))

    def field_to_pixel(self, xy_m: Point2) -> Point2:
        if self.H_field_to_img is None:
            raise RuntimeError("Homography not initialized.")
        x, y = float(xy_m[0]), float(xy_m[1])
        p = np.array([[x, y, 1.0]], dtype=np.float64).T
        q = self.H_field_to_img @ p
        q = q / (q[2, 0] + 1e-12)
        return (float(q[0, 0]), float(q[1, 0]))

    def pixels_to_field(self, pts_px: Iterable[Point2]) -> List[Point2]:
        return [self.pixel_to_field(p) for p in pts_px]

    def clip_to_pitch(self, xy_m: Point2) -> Point2:
        """
        Clamp coords to pitch extent (useful for numeric stability).
        """
        x = min(max(xy_m[0], 0.0), self.pitch.pitch_width_m)
        y = min(max(xy_m[1], 0.0), self.pitch.pitch_length_m)
        return (x, y)
