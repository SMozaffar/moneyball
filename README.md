# Detection, Tracking, Possession, and Sports Ball Fine-Tuning

This project analyzes stitched multi-GoPro panoramas and produces:

- **Player detections** (YOLO)
- **Player tracks** (ByteTrack-style association)
- **Ball localization** (YOLO + optional motion fallback)
- **Team assignment** (unsupervised jersey-color clustering)
- **Possession timeline** (heuristic “closest player to ball” with hysteresis)
- **Rendered output video** with overlays (boxes, ball marker, optional text/trails)
- **Exported ball crops** for fast labeling + fine-tuning on your own footage

The project is designed to start simple (no labeled data required) while staying **adaptable**: you can add new metrics and custom detectors (especially for the ball, and optionally for players) as you iterate.

---

## Table of Contents

- [Project Goals](#project-goals)
- [High-Level Pipeline](#high-level-pipeline)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
  - [1) Run Analysis + Render Output](#1-run-analysis--render-output)
  - [2) Export Ball Crops for Labeling](#2-export-ball-crops-for-labeling)
  - [3) Label in CVAT (Open Source)](#3-label-in-cvat-open-source)
  - [4) Prepare Dataset for Training](#4-prepare-dataset-for-training)
  - [5) Fine-Tune YOLO for Ball](#5-fine-tune-yolo-for-ball)
  - [6) Use Your Fine-Tuned Ball Model in Analysis](#6-use-your-fine-tuned-ball-model-in-analysis)
- [Configuration Guide](#configuration-guide)
- [Improving Player Detection (Far-Side Players)](#improving-player-detection-far-side-players)
- [Rendering Options](#rendering-options)
- [How Possession is Computed](#how-possession-is-computed)
- [Extending the System](#extending-the-system)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

---

## Project Goals

This project focuses on “good enough” soccer analytics from a single wide sideline camera:

1. **Track players** reliably enough to compute simple team-level metrics.
2. **Track the ball** well enough to estimate possession and transitions.
3. Provide a clean workflow to improve ball tracking via **fast crop labeling + fine-tuning**.

Key design constraints:
- You may have **no labeled data** initially.
- The video may be a stitched panorama with black borders and unusual aspect ratio (e.g., 1920×818).

---

## High-Level Pipeline

Each analyzed frame goes through:

1. **Frame Sampling**
   - Read the video, sample at `analysis_fps` (e.g., 8–12 FPS).
2. **Field Masking**
   - Build a “field mask” from green pixels + optional black border exclusion.
   - Used to filter out detections off the field.
3. **Player Detection (YOLO)**
   - Detect `person` boxes using a YOLO model (default COCO weights).
4. **Heuristic Filtering**
   - Remove detections likely off-field or too small/oddly shaped.
5. **Tracking**
   - Associate detections across frames (ByteTrack-style).
6. **Team Assignment**
   - Extract jersey-region colors and cluster into two teams (unsupervised).
7. **Ball Detection**
   - Either:
     - COCO “sports ball” (baseline; often weak on wide soccer), or
     - **Your fine-tuned ball model** (recommended).
   - Optional **motion fallback** to propose candidates when the detector misses.
8. **Possession**
   - If ball is present: assign possession to the closest player (foot-point distance).
   - Apply hysteresis (require N consecutive frames to confirm a switch).
9. **Artifacts + Render**
   - Save `analysis.json`, debug images, and an annotated output video.

---

## Repository Structure

Typical structure:

```
soccer_match_cv/
  soccer_cv/
    cli.py                # CLI entry points (analyze, export-ball-crops, etc.)
    pipeline.py            # main SoccerCVPipeline
    config.py              # Pydantic config models
    detectors/
      yolo.py              # YOLO inference wrapper
    tracking/
      bytetrack.py         # tracker wrapper / association logic
    masking/
      field_mask.py        # green-field + border masking
    team/
      assign.py            # jersey sampling + clustering
    ball/
      fallback.py          # motion fallback candidate generator
    viz/
      annotate.py          # draw overlays
    render.py              # render annotated output video
    types.py               # Detection/Track/Ball types
  configs/
    default.yaml           # main config
  scripts/
    prepare_ball_dataset.py   # CVAT YOLO export -> Ultralytics dataset builder
    preview_ball_dataset.py   # render label sanity-check images
  output/
    ... runs created here ...
```

(Your repo may include additional helper files and versions as you iterate.)

---

## Installation

### Recommended Environment (macOS / Apple Silicon)
Use CPU inference for reliability.

1) Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

2) Install the repo editable:
```bash
pip install -e .
```

3) Install dependencies (if you have `requirements.txt` + optional `constraints.txt`):
```bash
pip install -r requirements.txt
# Optional: pin known-good versions
# pip install -r requirements.txt -c constraints.txt
```

### Verify Ultralytics / Torch
```bash
python -c "import torch; print('torch', torch.__version__)"
python -c "from ultralytics import YOLO; print('ultralytics ok')"
```

---

## Quickstart

### 1) Run Analysis + Render Output

**Analyze** a video and generate outputs:

```bash
soccer-cv analyze \
  --video /path/to/match.mp4 \
  --out-dir output/run1 \
  --config configs/default.yaml
```

Expected outputs:
- `output/run1/analysis.json` — all per-frame detections/tracks/possession
- `output/run1/annotated.mp4` — rendered overlay video
- `output/run1/debug/` — optional debug frames/masks

If the directory is empty, see [Troubleshooting](#troubleshooting).

---

### 2) Export Ball Crops for Labeling

Ball labeling on full panoramic frames is painful because the ball is tiny. This project supports exporting **square crops** (e.g., 320×320) around likely ball locations (motion candidates, optionally YOLO ball detections).

```bash
soccer-cv export-ball-crops \
  --video /path/to/match.mp4 \
  --out-dir out/ball_crops \
  --config configs/default.yaml \
  --fps 8 \
  --crop-size 320 \
  --max-crops 2500 \
  --include-yolo-ball true
```

This produces:
- `out/ball_crops/images/*.jpg` — crop images
- `out/ball_crops/manifest.csv` — metadata (frame/time/crop source)

These crops are what you upload to CVAT for labeling.

---

### 3) Label in CVAT (Open Source)

CVAT is open source and works well for bounding boxes.

**Guidelines:**
- Create a task with the crop images from `out/ball_crops/images/`.
- Create a single label/class: `ball`
- Draw a tight bounding box around the ball when present.

**Important labeling rule (negatives):**
- If a crop has **no ball**, **do not delete it**.
- Keep the image and **do not annotate anything**.
- Those become negative examples and reduce false positives.

**Export:**
- Export dataset as **YOLO 1.1**
- You’ll get `.txt` label files like:

```
0 0.749406 0.473469 0.023438 0.022187
```

Format:
`class_id x_center y_center width height` (all normalized 0–1)

---

### 4) Prepare Dataset for Training

Ultralytics expects a standard dataset layout (`images/train`, `labels/train`, etc.). Use the helper script to combine:

- your crop images (`out/ball_crops/images/`)
- your CVAT YOLO export zip/folder

#### Option A: CVAT export is a `.zip`
```bash
python scripts/prepare_ball_dataset.py \
  --cvat-export /path/to/cvat_yolo_export.zip \
  --crops-dir out/ball_crops/images \
  --out datasets/ball \
  --class-name ball \
  --val-split 0.15 \
  --seed 13 \
  --mode copy
```

#### Option B: CVAT export is already extracted to a folder
```bash
python scripts/prepare_ball_dataset.py \
  --cvat-export /path/to/cvat_export_folder \
  --crops-dir out/ball_crops/images \
  --out datasets/ball \
  --class-name ball
```

What this script does:
- Creates:
  - `datasets/ball/images/train|val`
  - `datasets/ball/labels/train|val`
  - `datasets/ball/data.yaml`
- Ensures that if a crop has no exported `.txt`, it still gets an **empty label file** (valid YOLO behavior).

#### Sanity-check visually (recommended)
```bash
python scripts/preview_ball_dataset.py \
  --dataset datasets/ball \
  --split train \
  --n 200 \
  --out out/ball_preview_train
```

Open `out/ball_preview_train/` and ensure boxes are on the ball.

---

### 5) Fine-Tune YOLO for Ball

Train a small YOLO model on your crop dataset. Start with `yolov8n.pt` (fast iterations) and use `imgsz` matching your crop size.

```bash
yolo detect train \
  model=yolov8n.pt \
  data=datasets/ball/data.yaml \
  imgsz=320 \
  epochs=200 \
  batch=32 \
  device=cpu
```

After training, Ultralytics will create something like:
- `runs/detect/train/weights/best.pt`

If you do multiple runs, it becomes `train2`, `train3`, etc.

**Iteration strategy:**
- Your first run improves noticeably.
- To keep improving:
  - export crops from longer/more varied footage
  - label more examples (especially hard cases: motion blur, shadow, ball near lines, aerial ball)
  - retrain

---

### 6) Use Your Fine-Tuned Ball Model in Analysis

Edit `configs/default.yaml`:

```yaml
detector:
  model_path: "yolov8m.pt"              # players (COCO)
  ball_model_path: "/abs/path/to/best.pt"
  ball_class_name: "ball"              # must match datasets/ball/data.yaml
  device: "cpu"
  half: false
```

Run analysis again:

```bash
soccer-cv analyze \
  --video /path/to/match.mp4 \
  --out-dir output/run_with_ball_model \
  --config configs/default.yaml
```

---

## Configuration Guide

All runtime behavior is controlled by `configs/default.yaml`. Key sections:

### `video`
- `analysis_fps`: frames per second to analyze (8–12 common)
- `analysis_scale`: scale input frames (1.0 keeps original)

### `masking`
- `use_black_border_mask`: removes black panoramic borders
- `use_green_field_mask`: segments the playable field
- seed options (`seed_x_frac`, `seed_y_frac`) help keep the correct connected component

### `detector`
- `model_path`: player detector YOLO weights (COCO usually)
- `ball_model_path`: optional fine-tuned ball weights
- `ball_class_name`: `"sports ball"` (COCO) or `"ball"` (your custom dataset)
- `imgsz_players`: raises far-side player accuracy (but slower)
- `conf_players`: lower for small far-side players
- `imgsz_ball`: if using full-frame ball detection (less effective than crop training)
- `device`: `"cpu"` recommended on macOS for stability
- `half`: set `false` on CPU/MPS

### `heuristics`
- `filter_players_by_field_mask`: keep players whose foot-point is on the field
- `player_min_area_frac`: lower this if far-side players are being filtered out
- `player_min_aspect` / `player_max_aspect`: filter weird shapes
- `gate_ball_by_near_player`: reduce false positives by requiring ball near some player

### `tracking`
- `track_thresh`: detection confidence threshold for tracks
- `track_buffer`: how long to keep a track alive without detection
- `match_thresh`: association strictness

### `possession`
- `max_ball_to_player_px`: distance threshold for possession
- `min_frames_to_confirm`: hysteresis (prevents flicker)

### `render`
- `show_debug_text`: if false, no words/IDs
- `draw_tracks`: if false, no trails
- `draw_player_boxes`: if true, boxes drawn
- `draw_ball`: draw ball marker

> Note: If you edited your CLI/render call previously, ensure ball drawing is controlled by config:
> `draw_ball=cfg.render.draw_ball` (not hard-coded `False`), otherwise the ball may be present in analysis but not drawn.

---

## Improving Player Detection (Far-Side Players)

Sideline cameras create a strong scale problem:
- near-side players are large → easy
- far-side players are tiny → missed/unstable detections

### The simplest high-ROI fixes

1) Increase `imgsz_players`:
```yaml
detector:
  imgsz_players: 1280   # try 1280, then 1536 if CPU can handle it
```

2) Lower player confidence:
```yaml
detector:
  conf_players: 0.18    # from 0.25
```

3) Lower min-area filtering:
```yaml
heuristics:
  player_min_area_frac: 0.00005   # from 0.00012
```

4) Increase analysis FPS slightly (helps track continuity):
```yaml
video:
  analysis_fps: 10
```

5) Make tracks more “sticky”:
```yaml
tracking:
  track_buffer: 90
  track_thresh: 0.18
```

### Best next upgrade (still simple): tiled/zoomed far-side detection
If you want a big jump without training:
- Crop the far-side region of the frame
- Upscale it
- Run detection on the crop
- Merge detections back

This is one of the most effective ways to boost far-side detection without labeling.

### Best long-term: fine-tune a player detector
Same workflow you used for ball:
- export challenging far-side frames/tiles
- label `person` boxes
- train YOLO
- set `detector.model_path` to your custom person model

---

## Rendering Options

Common “clean output” settings:

### Normal-speed playback + minimal clutter
Set:

```yaml
render:
  show_debug_text: false
  draw_player_boxes: true
  draw_ball: true
  draw_tracks: false
  trail_seconds: 0.0
```

If you previously edited code to hardcode `draw_ball=False`, make sure render uses the config value.

### Only boxes, no ball marker
```yaml
render:
  draw_player_boxes: true
  draw_ball: false
  show_debug_text: false
  draw_tracks: false
```

---

## How Possession is Computed

Possession is estimated using:
- ball position (from detector or fallback)
- player foot-point positions (bottom center of track bbox)

Algorithm (simplified):
1. For each analyzed frame with a ball:
   - compute distance from ball to each player “foot point”
   - if the nearest player is within `max_ball_to_player_px`, possession candidate = that player’s team
2. Apply hysteresis:
   - require `min_frames_to_confirm` consecutive frames before switching possession
3. If ball is missing:
   - hold last possession for `hold_last_seconds_if_ball_missing`

This is intentionally simple, but works well once ball + player tracking are reasonable.

---

## Extending the System

Easy extension ideas:
- Pass maps, shot detection, touches (requires better ball + interaction detection)
- Team heatmaps (requires stable homography to field coordinates)
- Off-ball runs and sprints (requires consistent player tracks)
- Event detection: turnovers, interceptions, out-of-bounds (field mask + ball track)

Where to add features:
- `pipeline.py` is the orchestrator
- add per-frame computations after tracking/team assignment
- store results into the `analysis.json` schema
- optionally render additional overlays

---

## Troubleshooting

### 1) “Output directory is empty”
Common causes:
- repo not installed editable (`pip install -e .`)
- wrong command entry point
- missing write permissions to output folder

Try:
```bash
pip install -e .
soccer-cv analyze --video /path/to/video.mp4 --out-dir output/test --config configs/default.yaml
```

### 2) NumPy / torch / ultralytics version conflicts
If something installs NumPy 2.x and torch complains, pin:
- `numpy<2`

### 3) Apple Silicon MPS issues
MPS can fail on certain ops (e.g., NMS). CPU is the most reliable on macOS.
Set:
```yaml
detector:
  device: "cpu"
  half: false
```

### 4) Half precision errors (FP16)
If you see errors involving `Half` kernels, set:
```yaml
detector:
  half: false
```

### 5) “ball_model_path attribute not found”
Your `DetectorConfig` must include:
- `ball_model_path`
- `ball_class_name`

(Add them to `soccer_cv/config.py`.)

### 6) Ball detections exist in analysis but not in rendered video
This is almost always rendering config or a hard-coded render call:
- ensure `render.draw_ball: true`
- ensure the render call uses `cfg.render.draw_ball` (not hard-coded `False`)

---

## FAQ

### Should I delete crops where the ball isn’t visible?
No. Keep them. In CVAT, leave them unlabeled. Negative examples reduce false positives.

### How many labeled crops do I need?
Rough rule of thumb:
- First “useful” model: **500–1500** positives
- Better: **3000–8000** positives across varied lighting/angles/ball states

### My ball model improves, but still misses aerial balls.
You need more aerial examples in training. Export crops from sequences with lofted passes and label them.

### Why do far-side players still flicker?
They are tiny in pixels. Boost `imgsz_players`, reduce filters, or implement a far-side tile pass. Long-term: fine-tune a player model.

---

## Suggested “Good Defaults” to Start With (CPU)

```yaml
video:
  analysis_fps: 10

detector:
  device: "cpu"
  half: false
  model_path: "yolov8m.pt"
  conf_players: 0.18
  imgsz_players: 1280

heuristics:
  filter_players_by_field_mask: true
  player_min_area_frac: 0.00005

tracking:
  track_thresh: 0.18
  track_buffer: 90
```
