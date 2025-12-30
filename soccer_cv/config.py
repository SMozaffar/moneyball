from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any, Dict, Tuple, List
import yaml
from pathlib import Path

Point2 = Tuple[float, float]

class _Cfg(BaseModel):
    # Avoid warnings for fields like "model_path" in DetectorConfig
    model_config = ConfigDict(protected_namespaces=())

class ProjectConfig(_Cfg):
    name: str = "soccer_match_cv"
    version: str = "0.1.5"

class FieldConfig(BaseModel):
    enabled: bool = False

    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0

    # Must be same length; >= 4 points.
    image_points: List[Point2] = Field(default_factory=list)
    field_points: List[Point2] = Field(default_factory=list)

    ransac_reproj_threshold: float = 4.0

    def is_valid(self) -> bool:
        return (
            self.enabled
            and len(self.image_points) >= 4
            and len(self.image_points) == len(self.field_points)
        )

class VideoConfig(_Cfg):
    analysis_fps: int = 8
    analysis_scale: float = 1.0
    render_fps: Optional[float] = None

class MaskingConfig(_Cfg):
    use_black_border_mask: bool = True
    black_v_thresh: int = 35
    black_morph_kernel: int = 19

    use_green_field_mask: bool = True
    green_h_min: int = 25
    green_h_max: int = 70
    green_s_min: int = 80
    green_v_min: int = 40
    green_morph_kernel: int = 11

    # NEW
    keep_seed_component: bool = True
    seed_x_frac: float = 0.50
    seed_y_frac: float = 0.82

class DetectorConfig(_Cfg):
    model_path: str = "yolov8m.pt"

    # Optional: separate fine-tuned ball model (keeps person model unchanged)
    ball_model_path: Optional[str] = None
    ball_class_name: str = "sports ball"

    device: str = "auto"  # "auto", "cpu", "cuda:0", "mps"
    half: bool = True
    verbose: bool = False

    conf_players: float = 0.25
    iou_players: float = 0.6
    imgsz_players: int = 960

    conf_ball: float = 0.06
    iou_ball: float = 0.30
    imgsz_ball: int = 1536

class TrackingConfig(_Cfg):
    track_thresh: float = 0.25
    track_buffer: int = 45
    match_thresh: float = 0.8
    frame_rate: int = 30

class TeamAssignmentConfig(_Cfg):
    enabled: bool = True
    sample_top_fraction: float = 0.20
    sample_mid_fraction: float = 0.55
    exclude_green: bool = True
    green_exclude_h_min: int = 25
    green_exclude_h_max: int = 70
    green_exclude_s_min: int = 80
    green_exclude_v_min: int = 40
    min_samples_to_fit: int = 80
    refit_every_n_frames: int = 120
    random_state: int = 13

class BallFallbackConfig(_Cfg):
    enabled: bool = True

    # NEW: process motion at smaller scale for speed (0 < scale <= 1).
    # Thresholds below are in scaled pixel units.
    scale: float = 0.50

    diff_thresh: int = 12
    min_area: int = 2
    max_area: int = 80
    circularity_min: float = 0.28
    max_candidates: int = 25
    near_last_ball_px: int = 80


class HeuristicsConfig(_Cfg):
    """Small, high-ROI heuristics to reduce noise without adding complexity."""

    # Player detection filtering using the field mask
    filter_players_by_field_mask: bool = True
    player_foot_point_offset_px: int = 0

    # Additional lightweight filters (fractions are relative to full-frame area)
    player_min_area_frac: float = 0.00012
    player_max_area_frac: float = 0.04000
    player_min_aspect: float = 0.18
    player_max_aspect: float = 1.25

    # Ball gating: ignore ball candidates far from all tracked players
    gate_ball_by_near_player: bool = True
    ball_near_player_px: float = 180.0

class PossessionConfig(_Cfg):
    max_ball_to_player_px: float = 115.0
    min_frames_to_confirm: int = 3
    hold_last_seconds_if_ball_missing: float = 1.2

class RenderConfig(_Cfg):
    enabled: bool = True
    draw_field_mask_outline: bool = False
    show_debug_text: bool = True
    draw_player_boxes: bool = True
    draw_ball: bool = True
    draw_tracks: bool = True
    trail_seconds: float = 1.5

class FieldMetricsConfig(_Cfg):
    enabled: bool = True
    bin_size_m: float = 2.0
    min_players_for_shape: int = 3
    save_heatmap_images: bool = True
    heatmap_upsample: int = 8  # scale factor for saved PNGs
    gaussian_kernel: int = 11  # odd kernel size for smoothing saved heatmaps

class MiniMapConfig(_Cfg):
    enabled: bool = True
    width_px: int = 640
    height_px: Optional[int] = None  # auto-compute from pitch ratio when None
    fps: Optional[float] = None  # default: original video FPS
    full_speed: bool = True
    draw_trails: bool = True
    trail_seconds: float = 3.0
    show_possession: bool = True

class AppConfig(_Cfg):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    team_assignment: TeamAssignmentConfig = Field(default_factory=TeamAssignmentConfig)
    ball_fallback: BallFallbackConfig = Field(default_factory=BallFallbackConfig)
    heuristics: HeuristicsConfig = Field(default_factory=HeuristicsConfig)
    possession: PossessionConfig = Field(default_factory=PossessionConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    field: FieldConfig = Field(default_factory=FieldConfig)
    field_metrics: FieldMetricsConfig = Field(default_factory=FieldMetricsConfig)
    mini_map: MiniMapConfig = Field(default_factory=MiniMapConfig)


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(data)

def dump_config(cfg: AppConfig) -> Dict[str, Any]:
    return cfg.model_dump()
