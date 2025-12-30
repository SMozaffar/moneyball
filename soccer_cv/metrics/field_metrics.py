from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from soccer_cv.types import FrameResult


Point2 = Tuple[float, float]


@dataclass
class FieldMetrics:
    """
    Aggregates field-relative occupancy heatmaps and compact team shape metrics.
    Requires field homography so tracks/ball carry field_m coordinates.
    """
    pitch_length_m: float
    pitch_width_m: float
    bin_size_m: float = 2.0
    min_players_for_shape: int = 3
    save_heatmap_images: bool = True
    heatmap_upsample: int = 8
    gaussian_kernel: int = 11
    out_dir: Optional[Path] = None

    heat_all: np.ndarray = field(init=False)
    heat_ball: np.ndarray = field(init=False)
    heat_team: Dict[int, np.ndarray] = field(init=False)
    shape_series: Dict[int, List[Dict[str, Any]]] = field(init=False)
    grid_h: int = field(init=False)
    grid_w: int = field(init=False)

    def __post_init__(self) -> None:
        self.grid_h = max(1, int(np.ceil(float(self.pitch_length_m) / float(self.bin_size_m))))
        self.grid_w = max(1, int(np.ceil(float(self.pitch_width_m) / float(self.bin_size_m))))

        self.heat_all = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.heat_ball = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.heat_team = {
            0: np.zeros((self.grid_h, self.grid_w), dtype=np.float32),
            1: np.zeros((self.grid_h, self.grid_w), dtype=np.float32),
        }
        self.shape_series = {0: [], 1: []}

        if self.gaussian_kernel % 2 == 0:
            # GaussianBlur requires an odd kernel; bump to the next odd number.
            self.gaussian_kernel += 1

    def _bin_xy(self, pt_m: Point2) -> Point2:
        x_m, y_m = pt_m
        gx = int(np.clip(np.floor(x_m / float(self.bin_size_m)), 0, self.grid_w - 1))
        gy = int(np.clip(np.floor(y_m / float(self.bin_size_m)), 0, self.grid_h - 1))
        return gx, gy

    def _add_heat(self, heat: np.ndarray, pt_m: Point2) -> None:
        gx, gy = self._bin_xy(pt_m)
        heat[gy, gx] += 1.0

    def _record_shape(self, team: int, pts: np.ndarray, frame_idx: int, t_sec: float) -> None:
        if pts.shape[0] < int(self.min_players_for_shape):
            return

        centroid = pts.mean(axis=0)
        width = float(pts[:, 0].max() - pts[:, 0].min())
        length = float(pts[:, 1].max() - pts[:, 1].min())
        bbox_area = width * length
        spread = float(np.mean(np.linalg.norm(pts - centroid, axis=1)))

        hull_area = None
        hull_perimeter = None
        if pts.shape[0] >= 3:
            hull = cv2.convexHull(pts.astype(np.float32).reshape(-1, 1, 2))
            hull_area = float(cv2.contourArea(hull))
            hull_perimeter = float(cv2.arcLength(hull, True))

        self.shape_series[team].append(
            {
                "frame_idx": int(frame_idx),
                "t_sec": float(t_sec),
                "n_players": int(pts.shape[0]),
                "centroid": [float(centroid[0]), float(centroid[1])],
                "width_m": width,
                "length_m": length,
                "bbox_area_m2": bbox_area,
                "hull_area_m2": hull_area,
                "hull_perimeter_m": hull_perimeter,
                "avg_spread_m": spread,
            }
        )

    def on_frame(self, fr: FrameResult) -> None:
        # Accumulate player footprints.
        for tr in fr.tracks:
            if tr.foot_field_m is None:
                continue
            self._add_heat(self.heat_all, tr.foot_field_m)
            if tr.team_id in self.heat_team:
                self._add_heat(self.heat_team[int(tr.team_id)], tr.foot_field_m)

        # Accumulate ball positions if available.
        if fr.ball is not None and fr.ball.field_m is not None:
            self._add_heat(self.heat_ball, fr.ball.field_m)

        # Team shape metrics (convex hull + spread).
        for team_id in self.heat_team.keys():
            pts: List[Point2] = []
            for tr in fr.tracks:
                if tr.team_id != team_id or tr.foot_field_m is None:
                    continue
                pts.append(tr.foot_field_m)
            if not pts:
                continue
            pts_arr = np.array(pts, dtype=np.float32)
            self._record_shape(team_id, pts_arr, frame_idx=fr.frame_idx, t_sec=fr.t_sec)

    def _summaries(self, team: int) -> Dict[str, Any]:
        series = self.shape_series.get(team, [])
        if not series:
            return {"samples": 0}

        def _mean(key: str) -> Optional[float]:
            vals = [s[key] for s in series if s.get(key) is not None]
            if not vals:
                return None
            return float(np.mean(vals))

        return {
            "samples": len(series),
            "mean_width_m": _mean("width_m"),
            "mean_length_m": _mean("length_m"),
            "mean_bbox_area_m2": _mean("bbox_area_m2"),
            "mean_hull_area_m2": _mean("hull_area_m2"),
            "mean_hull_perimeter_m": _mean("hull_perimeter_m"),
            "mean_spread_m": _mean("avg_spread_m"),
        }

    def _heatmap_to_image(self, heat: np.ndarray) -> np.ndarray:
        if heat.max() > 0:
            norm = heat / float(heat.max())
        else:
            norm = heat

        smoothed = cv2.GaussianBlur(norm.astype(np.float32), (self.gaussian_kernel, self.gaussian_kernel), 0)
        img = np.clip(smoothed * 255.0, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)

        scale = max(1, int(self.heatmap_upsample))
        if scale != 1:
            color = cv2.resize(
                color, (self.grid_w * scale, self.grid_h * scale), interpolation=cv2.INTER_CUBIC
            )

        # Simple pitch markings to keep orientation clear.
        h, w = color.shape[:2]
        out = color.copy()
        cv2.rectangle(out, (2, 2), (w - 3, h - 3), (255, 255, 255), 2)
        cy = h // 2
        cv2.line(out, (2, cy), (w - 3, cy), (255, 255, 255), 1)
        cv2.circle(out, (w // 2, cy), max(6, int(min(w, h) * 0.06)), (255, 255, 255), 1)
        return out

    def _write_heatmap_images(self) -> Dict[str, str]:
        if self.out_dir is None:
            return {}
        hm_dir = Path(self.out_dir) / "heatmaps"
        hm_dir.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}
        pairs = [
            ("players_all", self.heat_all),
            ("team0", self.heat_team[0]),
            ("team1", self.heat_team[1]),
            ("ball", self.heat_ball),
        ]
        for name, heat in pairs:
            img = self._heatmap_to_image(heat)
            fpath = hm_dir / f"{name}.png"
            cv2.imwrite(str(fpath), img)
            paths[name] = str(fpath)
        return paths

    def finalize(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "bin_size_m": float(self.bin_size_m),
            "grid_shape": [int(self.grid_h), int(self.grid_w)],
            "all_players_heatmap": self.heat_all.tolist(),
            "ball_heatmap": self.heat_ball.tolist(),
            "team_heatmaps": {str(k): v.tolist() for k, v in self.heat_team.items()},
            "shape_summary": {str(k): self._summaries(k) for k in self.shape_series.keys()},
        }
        if self.save_heatmap_images:
            out["heatmap_images"] = self._write_heatmap_images()
        return out
