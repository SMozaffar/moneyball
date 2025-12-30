from __future__ import annotations

import typer
from pathlib import Path
from typing import Optional
import json

from soccer_cv.config import load_config, AppConfig
from soccer_cv.logging_utils import get_logger
from soccer_cv.pipeline import SoccerCVPipeline
from soccer_cv.utils.outputs import save_analysis, save_possession_timeline_csv, save_json, load_analysis
from soccer_cv.render import render_annotated_video
from soccer_cv.viz.minimap import render_minimap_video

logger = get_logger()
app = typer.Typer(add_completion=False, help="Soccer Match CV: detection, tracking, team clustering, and possession metrics.")

@app.command()
def analyze(
    video: str = typer.Option(..., help="Path to stitched match MP4."), 
    out_dir: str = typer.Option("out/run1", help="Output directory."), 
    config: str = typer.Option("configs/default.yaml", help="Path to YAML config."),
    no_render: bool = typer.Option(False, help="Disable rendering annotated MP4."),
) -> None:
    """Run end-to-end analysis and write artifacts to out_dir."""
    cfg: AppConfig = load_config(config)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    pipeline = SoccerCVPipeline(cfg)
    frames, meta, artifacts = pipeline.analyze(video_path=video, out_dir=out_dir)

    analysis_path = out_dir_p / "analysis.json"
    save_analysis(analysis_path, meta=meta, frames=frames)
    logger.info(f"Wrote analysis: {analysis_path}")

    # summary + timeline
    metrics = meta.get("metrics", {})
    summary = {
        "team_seconds": metrics.get("PossessionStats", {}).get("team_seconds", {}),
        "player_seconds": metrics.get("PossessionStats", {}).get("player_seconds", {}),
        "config": meta.get("config", {}),
        "video": {
            "path": meta.get("video_path"),
            "fps": meta.get("video_fps"),
            "size": meta.get("video_size"),
        },
    }
    save_json(out_dir_p / "possession_summary.json", summary)
    save_possession_timeline_csv(out_dir_p / "possession_timeline.csv", frames=frames)

    # optional render
    if not no_render:
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        if cfg.render.enabled:
            out_video = out_dir_p / "annotated.mp4"
            # Render at original FPS (or override with cfg.video.render_fps)
            fps = cfg.video.render_fps
            render_annotated_video(
                video_path=video,
                analysis=analysis,
                out_video=str(out_video),
                fps=fps,
                full_speed=True,                      # normal playback duration
                show_text=cfg.render.show_debug_text, # no words/numbers when False
                draw_trails=cfg.render.draw_tracks,   # treat draw_tracks as "trails"
                trail_seconds=cfg.render.trail_seconds,
                draw_boxes=cfg.render.draw_player_boxes,
                draw_feet_points=False,              # boxes only
                draw_ball=cfg.render.draw_ball,                     # boxes only (set True if you still want the ball dot)
            )

        if cfg.mini_map.enabled:
            try:
                render_minimap_video(
                    video_path=video,
                    analysis=analysis,
                    out_video=str(out_dir_p / "minimap.mp4"),
                    width_px=cfg.mini_map.width_px,
                    height_px=cfg.mini_map.height_px,
                    fps=cfg.mini_map.fps,
                    full_speed=cfg.mini_map.full_speed,
                    draw_trails=cfg.mini_map.draw_trails,
                    trail_seconds=cfg.mini_map.trail_seconds,
                    show_possession=cfg.mini_map.show_possession,
                )
            except Exception as e:
                logger.exception("Mini-map render failed: %s", e)
    logger.info("Done.")

@app.command()
def render(
    video: str = typer.Option(..., help="Path to stitched match MP4."), 
    analysis: str = typer.Option(..., help="Path to analysis.json produced by analyze."), 
    out_video: str = typer.Option(..., help="Output annotated video path."), 
    fps: Optional[float] = typer.Option(None, help="FPS override (default: original video fps)."), 
) -> None:
    """Render an annotated MP4 from a previously generated analysis.json."""
    analysis_obj = load_analysis(analysis)
    render_annotated_video(video_path=video, analysis=analysis_obj, out_video=out_video, fps=fps)

@app.command()
def export_ball_crops(
    video: str = typer.Option(..., help="Path to stitched match MP4."),
    out_dir: str = typer.Option("out/ball_crops", help="Output directory for crops."),
    config: str = typer.Option("configs/default.yaml", help="Config YAML."),
    fps: int = typer.Option(8, help="Frame sampling rate for crop export (frames per second)."),
    crop_size: int = typer.Option(320, help="Square crop size in pixels (recommended 256-384)."),
    max_crops: int = typer.Option(2500, help="Maximum crops to export."),
    max_motion_crops_per_frame: int = typer.Option(2, help="How many motion candidates to export per frame."),
    include_yolo_ball: bool = typer.Option(True, help="Also export crops around YOLO 'sports ball' detections."),
    yolo_conf: float = typer.Option(0.15, help="YOLO confidence for ball crop export (lower = more candidates)."),
    yolo_iou: float = typer.Option(0.30, help="YOLO IoU for ball crop export."),
    include_random_player_crops: bool = typer.Option(False, help="Optionally include random player-centered crops as hard negatives."),
    random_player_crops_per_frame: int = typer.Option(1, help="If include_random_player_crops, how many per frame."),
    seed: int = typer.Option(13, help="Random seed for sampling."),
) -> None:
    """Export candidate ball crops for labeling + future fine-tuning.

    Output:
      - images/*.jpg  (square crops)
      - manifest.csv  (crop metadata)

    You can import the images directory into an open-source labeling tool (CVAT or Label Studio),
    draw a bounding box around the ball, then export in YOLO format for training.
    """
    import csv
    from pathlib import Path

    import cv2
    import numpy as np

    from soccer_cv.config import load_config
    from soccer_cv.detectors.yolo import YoloDetector
    from soccer_cv.metrics.ball_fallback import MotionBallFallback
    from soccer_cv.pipeline import SoccerCVPipeline
    from soccer_cv.utils.masking import apply_mask
    from soccer_cv.utils.video_io import get_video_info, iter_frames, resize_keep_aspect

    cfg = load_config(config)

    out_p = Path(out_dir)
    img_p = out_p / "images"
    out_p.mkdir(parents=True, exist_ok=True)
    img_p.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed))

    info = get_video_info(video)
    step = max(1, int(round(float(info.fps) / float(fps))))
    logger.info(f"Exporting crops at ~{info.fps/step:.2f} FPS (step={step}) from: {video}")

    # Peek first frame for mask + scaling
    first = None
    for _, fr in iter_frames(video, step=step):
        first = fr
        break
    if first is None:
        raise RuntimeError("Could not read first frame")

    first_s = resize_keep_aspect(first, cfg.video.analysis_scale)

    pipe = SoccerCVPipeline(cfg)
    field_mask = pipe._build_field_mask(first_s)

    player_detector = YoloDetector(
        model_path=cfg.detector.model_path,
        device=cfg.detector.device,
        half=False,  # safer for CPU/MPS
        verbose=cfg.detector.verbose,
    )

    ball_detector = YoloDetector(
    model_path=cfg.detector.ball_model_path or cfg.detector.model_path,
    device=cfg.detector.device,
    half=False,
    verbose=cfg.detector.verbose,
    )


    fallback = MotionBallFallback(
        enabled=True,
        scale=cfg.ball_fallback.scale,
        diff_thresh=cfg.ball_fallback.diff_thresh,
        min_area=cfg.ball_fallback.min_area,
        max_area=cfg.ball_fallback.max_area,
        circularity_min=cfg.ball_fallback.circularity_min,
        max_candidates=cfg.ball_fallback.max_candidates,
        near_last_ball_px=cfg.ball_fallback.near_last_ball_px,
    )

    saved = 0
    rows: list[dict] = []

    def _save_crop(frame_bgr: np.ndarray, center_xy: tuple[float, float], frame_idx: int, time_s: float, score: float, source: str) -> None:
        nonlocal saved
        if saved >= int(max_crops):
            return

        h, w = frame_bgr.shape[:2]
        cx, cy = float(center_xy[0]), float(center_xy[1])

        half = int(crop_size // 2)
        x1 = int(round(cx)) - half
        y1 = int(round(cy)) - half
        x2 = x1 + int(crop_size)
        y2 = y1 + int(crop_size)

        # clamp to image bounds
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(1, min(w, x2))
        y2 = max(1, min(h, y2))

        if x2 <= x1 + 4 or y2 <= y1 + 4:
            return

        crop = frame_bgr[y1:y2, x1:x2].copy()
        fname = f"crop_f{frame_idx:07d}_t{time_s:08.3f}_{source}_{saved:06d}.jpg"
        fpath = img_p / fname
        cv2.imwrite(str(fpath), crop)

        rows.append({
            "file": str(Path("images") / fname),
            "frame_idx": int(frame_idx),
            "time_s": float(time_s),
            "source": str(source),
            "score": float(score),
            "center_x": float(cx),
            "center_y": float(cy),
            "crop_x1": int(x1),
            "crop_y1": int(y1),
            "crop_x2": int(x2),
            "crop_y2": int(y2),
            "crop_size": int(crop_size),
            "video_fps": float(info.fps),
            "analysis_scale": float(cfg.video.analysis_scale),
        })
        saved += 1

    for frame_idx, frame_bgr in iter_frames(video, step=step):
        if saved >= int(max_crops):
            break

        frame_s = resize_keep_aspect(frame_bgr, cfg.video.analysis_scale)
        t = float(frame_idx) / float(info.fps)

        masked = apply_mask(frame_s, field_mask) if field_mask is not None else frame_s

        # 1) motion candidates (best for bootstrapping labels)
        motion_cands = fallback.find_candidates(frame_s, field_mask)
        for score, cx, cy, area in motion_cands[: int(max_motion_crops_per_frame)]:
            _save_crop(frame_s, (cx, cy), frame_idx=frame_idx, time_s=t, score=float(score), source="motion")

        # 2) YOLO ball candidates (helpful but noisy on wide shots)
        if include_yolo_ball and saved < int(max_crops):
            balls = ball_detector.detect(masked, conf=float(yolo_conf), iou=float(yolo_iou), imgsz=int(cfg.detector.imgsz_ball), classes=[cfg.detector.ball_class_name])
            if balls:
                best = max(balls, key=lambda d: d.conf)
                cx, cy = best.center
                _save_crop(frame_s, (cx, cy), frame_idx=frame_idx, time_s=t, score=float(best.conf), source="yolo")

        # 3) optional player-centered hard negatives
        if include_random_player_crops and saved < int(max_crops):
            players = player_detector.detect_players(masked, conf=float(cfg.detector.conf_players), iou=float(cfg.detector.iou_players), imgsz=int(cfg.detector.imgsz_players))
            players = pipe._filter_player_detections(players, field_mask, frame_s.shape)
            if players:
                k = min(len(players), int(random_player_crops_per_frame))
                idxs = rng.choice(len(players), size=k, replace=False)
                for ii in idxs:
                    px, py = players[int(ii)].center
                    _save_crop(frame_s, (float(px), float(py)), frame_idx=frame_idx, time_s=t, score=0.0, source="player_neg")

    manifest_path = out_p / "manifest.csv"
    if rows:
        keys = list(rows[0].keys())
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    logger.info(f"Exported {saved} crops to: {out_p}")
    logger.info(f"Images: {img_p}")
    logger.info(f"Manifest: {manifest_path}")
