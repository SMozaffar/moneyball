from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from soccer_cv.config import AppConfig, dump_config
from soccer_cv.logging_utils import get_logger
from soccer_cv.types import Detection, Track, Ball, FrameResult
from soccer_cv.utils.video_io import get_video_info, iter_frames, resize_keep_aspect
from soccer_cv.utils.masking import (
    black_border_mask,
    green_field_mask,
    combine_masks,
    apply_mask,
    keep_component_with_seed,
)
from soccer_cv.detectors.yolo import YoloDetector
from soccer_cv.trackers.bytetrack import ByteTrackTracker
from soccer_cv.metrics.team_assignment import OnlineTeamAssigner
from soccer_cv.metrics.ball_fallback import MotionBallFallback
from soccer_cv.metrics.possession import PossessionEstimator
from soccer_cv.metrics.possession_stats import PossessionStats
from soccer_cv.metrics.base import MetricRegistry

# NEW: homography
from soccer_cv.field.homography import PitchHomography, PitchSpec

logger = get_logger()


@dataclass
class PipelineArtifacts:
    field_mask: Optional[np.ndarray] = None
    sample_frame: Optional[np.ndarray] = None


class SoccerCVPipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        self.detector = YoloDetector(
            model_path=cfg.detector.model_path,
            device=cfg.detector.device,
            half=cfg.detector.half,
            verbose=cfg.detector.verbose,
        )

        self.ball_detector = YoloDetector(
            model_path=cfg.detector.ball_model_path or cfg.detector.model_path,
            device=cfg.detector.device,
            half=cfg.detector.half,
            verbose=cfg.detector.verbose,
        )

        self.tracker = ByteTrackTracker(
            track_thresh=cfg.tracking.track_thresh,
            track_buffer=cfg.tracking.track_buffer,
            match_thresh=cfg.tracking.match_thresh,
            frame_rate=cfg.tracking.frame_rate,
        )

        self.team_assigner = OnlineTeamAssigner(
            enabled=cfg.team_assignment.enabled,
            sample_top_fraction=cfg.team_assignment.sample_top_fraction,
            sample_mid_fraction=cfg.team_assignment.sample_mid_fraction,
            exclude_green=cfg.team_assignment.exclude_green,
            green_exclude_h_min=cfg.team_assignment.green_exclude_h_min,
            green_exclude_h_max=cfg.team_assignment.green_exclude_h_max,
            green_exclude_s_min=cfg.team_assignment.green_exclude_s_min,
            green_exclude_v_min=cfg.team_assignment.green_exclude_v_min,
            min_samples_to_fit=cfg.team_assignment.min_samples_to_fit,
            refit_every_n_frames=cfg.team_assignment.refit_every_n_frames,
            random_state=cfg.team_assignment.random_state,
        )

        self.ball_fallback = MotionBallFallback(
            enabled=cfg.ball_fallback.enabled,
            scale=cfg.ball_fallback.scale,
            diff_thresh=cfg.ball_fallback.diff_thresh,
            min_area=cfg.ball_fallback.min_area,
            max_area=cfg.ball_fallback.max_area,
            circularity_min=cfg.ball_fallback.circularity_min,
            max_candidates=cfg.ball_fallback.max_candidates,
            near_last_ball_px=cfg.ball_fallback.near_last_ball_px,
        )

        self.possession = PossessionEstimator(
            max_ball_to_player_px=cfg.possession.max_ball_to_player_px,
            min_frames_to_confirm=cfg.possession.min_frames_to_confirm,
            hold_last_seconds_if_ball_missing=cfg.possession.hold_last_seconds_if_ball_missing,
            analysis_fps=float(cfg.video.analysis_fps),
        )

        # NEW: Optional pitch homography mapper
        self.pitch_mapper: Optional[PitchHomography] = None
        try:
            field_cfg = getattr(cfg, "field", None)
            if field_cfg is not None and getattr(field_cfg, "enabled", False):
                is_valid = True
                if hasattr(field_cfg, "is_valid"):
                    is_valid = bool(field_cfg.is_valid())
                else:
                    # fallback validity check if you didn’t add is_valid()
                    ip = getattr(field_cfg, "image_points", [])
                    fp = getattr(field_cfg, "field_points", [])
                    is_valid = (len(ip) >= 4) and (len(ip) == len(fp))

                if not is_valid:
                    logger.warning(
                        "Field homography enabled but config is invalid. "
                        "Need >=4 points and equal-length image_points/field_points."
                    )
                else:
                    self.pitch_mapper = PitchHomography(
                        pitch=PitchSpec(
                            pitch_length_m=float(getattr(field_cfg, "pitch_length_m", 105.0)),
                            pitch_width_m=float(getattr(field_cfg, "pitch_width_m", 68.0)),
                        ),
                        image_points=list(getattr(field_cfg, "image_points")),
                        field_points=list(getattr(field_cfg, "field_points")),
                        ransac_reproj_threshold=float(getattr(field_cfg, "ransac_reproj_threshold", 4.0)),
                    )
                    logger.info(
                        "Field homography ENABLED (pixel->meters). "
                        "Pitch: %.1fm x %.1fm, points=%d",
                        float(getattr(field_cfg, "pitch_length_m", 105.0)),
                        float(getattr(field_cfg, "pitch_width_m", 68.0)),
                        len(getattr(field_cfg, "image_points", [])),
                    )
            else:
                logger.info("Field homography disabled (cfg.field.enabled=false).")
        except Exception as e:
            logger.exception("Failed to initialize pitch homography; continuing without it. Error: %s", e)
            self.pitch_mapper = None

    def _build_field_mask(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        masks = []
        if self.cfg.masking.use_black_border_mask:
            masks.append(
                black_border_mask(
                    frame_bgr,
                    v_thresh=self.cfg.masking.black_v_thresh,
                    morph_kernel=self.cfg.masking.black_morph_kernel,
                )
            )
        if self.cfg.masking.use_green_field_mask:
            masks.append(
                green_field_mask(
                    frame_bgr,
                    h_min=self.cfg.masking.green_h_min,
                    h_max=self.cfg.masking.green_h_max,
                    s_min=self.cfg.masking.green_s_min,
                    v_min=self.cfg.masking.green_v_min,
                    morph_kernel=self.cfg.masking.green_morph_kernel,
                )
            )
        if not masks:
            return None

        mask = combine_masks(*masks)

        if self.cfg.masking.keep_seed_component:
            h, w = mask.shape[:2]
            sx = int(round(self.cfg.masking.seed_x_frac * w))
            sy = int(round(self.cfg.masking.seed_y_frac * h))
            mask = keep_component_with_seed(mask, seed_x=sx, seed_y=sy)

        return mask

    def _foot_point(self, xyxy: Tuple[float, float, float, float], y_offset: int = 0) -> Tuple[int, int]:
        x1, y1, x2, y2 = xyxy
        x = int(round(0.5 * (x1 + x2)))
        y = int(round(y2)) - int(y_offset)
        return x, y

    def _attach_field_coords_to_track(self, t: Track) -> None:
        """
        Adds:
          - t.foot_px = (x_px, y_px)
          - t.foot_field_m = (x_m, y_m)  (if homography enabled)
        Does not require modifying Track dataclass; attaches attributes dynamically.
        """
        fx, fy = self._foot_point(t.xyxy, y_offset=int(self.cfg.heuristics.player_foot_point_offset_px))
        foot_px = (float(fx), float(fy))

        # Always attach pixel foot point (useful for debugging even without homography)
        try:
            setattr(t, "foot_px", foot_px)
        except Exception:
            pass

        if self.pitch_mapper is None:
            return

        try:
            foot_m = self.pitch_mapper.pixel_to_field(foot_px)
            foot_m = self.pitch_mapper.clip_to_pitch(foot_m)
            setattr(t, "foot_field_m", foot_m)
        except Exception:
            # Don’t break analysis if a point goes out of bounds or homography is imperfect
            pass

    def _attach_field_coords_to_ball(self, ball: Ball) -> None:
        """
        Adds:
          - ball.field_m = (x_m, y_m) (if homography enabled)
        """
        if self.pitch_mapper is None:
            return
        try:
            m = self.pitch_mapper.pixel_to_field((float(ball.x), float(ball.y)))
            m = self.pitch_mapper.clip_to_pitch(m)
            setattr(ball, "field_m", m)
        except Exception:
            pass

    def _filter_player_detections(
        self,
        dets: List[Detection],
        field_mask: Optional[np.ndarray],
        frame_shape: Tuple[int, int, int],
    ) -> List[Detection]:
        """Simple, high-ROI filtering to reduce false tracks."""
        if not dets:
            return dets

        h, w = frame_shape[:2]
        frame_area = float(h * w)

        cfg = self.cfg.heuristics
        out: List[Detection] = []
        for d in dets:
            x1, y1, x2, y2 = d.xyxy
            bw = max(1.0, float(x2 - x1))
            bh = max(1.0, float(y2 - y1))
            area = bw * bh
            area_frac = area / frame_area

            ar = bw / bh

            if area_frac < float(cfg.player_min_area_frac) or area_frac > float(cfg.player_max_area_frac):
                continue
            if ar < float(cfg.player_min_aspect) or ar > float(cfg.player_max_aspect):
                continue

            if field_mask is not None and bool(cfg.filter_players_by_field_mask):
                fx, fy = self._foot_point(d.xyxy, y_offset=int(cfg.player_foot_point_offset_px))
                fx = int(np.clip(fx, 0, w - 1))
                fy = int(np.clip(fy, 0, h - 1))
                if field_mask[fy, fx] == 0:
                    continue

            out.append(d)

        return out

    def _min_ball_distance_to_players_px(self, ball_xy: Tuple[float, float], tracks: List[Track]) -> Optional[float]:
        if not tracks:
            return None
        bx, by = float(ball_xy[0]), float(ball_xy[1])
        best: Optional[float] = None
        for t in tracks:
            fx, fy = self._foot_point(t.xyxy, y_offset=int(self.cfg.heuristics.player_foot_point_offset_px))
            d = float(np.hypot(bx - fx, by - fy))
            if best is None or d < best:
                best = d
        return best

    def analyze(self, video_path: str, out_dir: str) -> Tuple[List[FrameResult], Dict[str, Any], PipelineArtifacts]:
        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)
        (out_dir_p / "debug").mkdir(parents=True, exist_ok=True)

        info = get_video_info(video_path)
        analysis_fps = float(self.cfg.video.analysis_fps)
        step = max(1, int(round(info.fps / analysis_fps)))
        dt_sec = step / info.fps

        logger.info(f"Video: {video_path}")
        logger.info(f"Original FPS: {info.fps:.3f}  frames: {info.frame_count}  size: {info.width}x{info.height}")
        logger.info(f"Analysis FPS: ~{info.fps/step:.3f} (step={step})  dt={dt_sec:.3f}s")

        frames: List[FrameResult] = []
        artifacts = PipelineArtifacts()

        self.ball_fallback.reset()
        self.possession.reset()

        metrics = MetricRegistry(metrics=[PossessionStats(dt_sec=dt_sec)])

        # Peek first frame for masks/debug + ensure video readable
        first_frame_bgr = None
        for idx, frame_bgr in iter_frames(video_path, step=step):
            first_frame_bgr = frame_bgr
            break
        if first_frame_bgr is None:
            raise RuntimeError("Could not read first frame")

        # Optional analysis scaling
        frame0 = resize_keep_aspect(first_frame_bgr, self.cfg.video.analysis_scale)
        field_mask = self._build_field_mask(frame0)
        artifacts.field_mask = field_mask
        artifacts.sample_frame = frame0

        if field_mask is not None:
            cv2.imwrite(str(out_dir_p / "debug" / "field_mask.png"), field_mask)
        cv2.imwrite(str(out_dir_p / "debug" / "sample_frame.jpg"), frame0)

        # Reset iterator after peeking
        pbar_total = max(1, int(info.frame_count / step))
        pbar = tqdm(total=pbar_total, desc="Analyzing", unit="frame")

        for frame_idx, frame_bgr in iter_frames(video_path, step=step):
            frame_bgr = resize_keep_aspect(frame_bgr, self.cfg.video.analysis_scale)

            # mask for motion fallback and (optionally) to help detection focus
            if field_mask is not None:
                masked = apply_mask(frame_bgr, field_mask)
            else:
                masked = frame_bgr

            # detect players
            player_dets = self.detector.detect_players(
                masked,
                conf=self.cfg.detector.conf_players,
                iou=self.cfg.detector.iou_players,
                imgsz=self.cfg.detector.imgsz_players,
            )

            # reduce false positives before tracking (simple heuristics)
            player_dets = self._filter_player_detections(player_dets, field_mask, frame_bgr.shape)

            # track players
            tracks = self.tracker.update(player_dets)

            # team assignment (updates track.team_id in-place)
            self.team_assigner.update(frame_bgr, tracks)

            # NEW: attach per-track foot_px and foot_field_m (if homography enabled)
            for t in tracks:
                self._attach_field_coords_to_track(t)

            # detect ball via YOLO
            ball_dets = self.ball_detector.detect(
                masked,
                conf=self.cfg.detector.conf_ball,
                iou=self.cfg.detector.iou_ball,
                imgsz=self.cfg.detector.imgsz_ball,
                classes=[self.cfg.detector.ball_class_name],
            )

            ball: Optional[Ball] = None

            # 1) Try YOLO sports-ball (or custom "ball" class if you trained)
            yolo_ball: Optional[Ball] = None
            if ball_dets:
                best = max(ball_dets, key=lambda d: d.conf)
                cx, cy = best.center
                yolo_ball = Ball(x=cx, y=cy, conf=best.conf, source="yolo")

            # 2) Optionally gate YOLO ball by distance to tracked players (removes many false positives)
            if yolo_ball is not None and self.cfg.heuristics.gate_ball_by_near_player and tracks:
                dmin = self._min_ball_distance_to_players_px((yolo_ball.x, yolo_ball.y), tracks)
                if dmin is not None and dmin > float(self.cfg.heuristics.ball_near_player_px):
                    yolo_ball = None

            ball = yolo_ball

            # 3) If no good YOLO ball, fall back to motion
            if ball is None:
                ball = self.ball_fallback.update(frame_bgr, field_mask)

                # Optional gate for motion candidate too
                if ball is not None and self.cfg.heuristics.gate_ball_by_near_player and tracks:
                    dmin = self._min_ball_distance_to_players_px((ball.x, ball.y), tracks)
                    if dmin is not None and dmin > float(self.cfg.heuristics.ball_near_player_px):
                        ball = None

            # NEW: attach ball field coords if we have a ball + homography
            if ball is not None:
                self._attach_field_coords_to_ball(ball)

            # possession
            team_id, player_id = self.possession.update(ball, tracks)

            t_sec = frame_idx / info.fps
            fr = FrameResult(
                frame_idx=frame_idx,
                t_sec=t_sec,
                tracks=tracks,
                ball=ball,
                possession_team=int(team_id),
                possession_player=int(player_id),
            )

            metrics.on_frame(fr)
            frames.append(fr)
            pbar.update(1)

        pbar.close()

        meta: Dict[str, Any] = {
            "video_path": video_path,
            "video_fps": info.fps,
            "video_size": [info.width, info.height],
            "analysis_step": step,
            "analysis_fps": info.fps / step,
            "dt_sec": dt_sec,
            "config": dump_config(self.cfg),
        }
        meta["metrics"] = metrics.finalize()

        # Helpful flag for downstream code / debugging
        meta["field_homography_enabled"] = bool(self.pitch_mapper is not None)

        return frames, meta, artifacts
