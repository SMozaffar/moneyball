# Soccer Match CV (stitched dual-GoPro panorama) — possession + extensible metrics

## Installation (recommended)

Use the provided `constraints.txt` to prevent pip from upgrading to NumPy 2.x (which can break some PyTorch/Ultralytics wheels on macOS).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt -c constraints.txt
pip install -e .
```
You have a **single stitched 1920×818 MP4** (two GoPros stitched into a panorama with black curved borders).  
This repo implements a **modular CV pipeline** to start small (possession) and make it easy to add more features later.

## What you get (v0.1)

### Core pipeline
1. **Field mask** (exclude black borders + non-field)
2. **Detection** (YOLOv8 pretrained)
   - players: COCO `person`
   - ball: COCO `sports ball` (with a motion-based fallback when YOLO misses)
3. **Tracking** (ByteTrack for stable player IDs)
4. **Team assignment** (unsupervised jersey-color clustering into 2 teams)
5. **Possession** (ball → nearest player’s “feet point” with hysteresis + smoothing)
6. **Outputs**
   - `analysis.json` (per-frame events)
   - `possession_summary.json` (team + player possession totals)
   - `possession_timeline.csv` (timestamped possession state)
   - Optional: **annotated MP4** overlay (players, ball, possession HUD)

### Why this works without labels
- We rely on **strong pretrained detectors** for `person` and `sports ball`
- We add a **motion-based ball fallback** because the ball is tiny in wide-angle footage
- We cluster jersey colors with **KMeans** to separate the two teams automatically

> Note: ball detection quality varies by lighting, distance, and resolution.  
> If ball detection is weak, the repo includes a “label-assist” script to quickly export candidate ball crops for annotation and eventual fine-tuning.

---

## Quickstart

### 1) Create env + install
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Run analysis
```bash
soccer-cv analyze \
  --video /path/to/match.mp4 \
  --out-dir out/run1 \
  --config configs/default.yaml
```

### 3) Render annotated video (optional)
```bash
soccer-cv render \
  --video /path/to/match.mp4 \
  --analysis out/run1/analysis.json \
  --out-video out/run1/annotated.mp4
```

---

## Outputs

After `analyze`, you’ll see:

```
out/run1/
  analysis.json
  possession_summary.json
  possession_timeline.csv
  debug/
    field_mask.png
    sample_frame.jpg
```

### `possession_summary.json`
- `team_seconds`: time in possession for each team
- `player_seconds`: per tracked player (track_id) possession
- `settings`: config snapshot for reproducibility

### `possession_timeline.csv`
A time series with columns:
- `t_sec`
- `team_id` (0/1 or -1 unknown)
- `player_track_id` (or -1 unknown)
- `ball_x`, `ball_y` (or empty)

---

## Config

All knobs live in `configs/default.yaml`:
- analysis fps
- YOLO model path, conf/iou thresholds, inference size, device
- tracking params
- field mask params (green thresholds, black border removal)
- ball fallback params
- possession distance threshold + smoothing/hysteresis

---

## Extending the system (examples)

The codebase is organized so you can add new metrics as “plugins”:

- **Pass count**: possession change within short time + ball displacement
- **Shots**: ball speed spike + trajectory toward goal region
- **Heatmaps**: accumulate player positions onto a downsampled grid
- **Offside line approximation**: last defender x-position in field coords (requires homography calibration for accuracy)

See: `soccer_cv/metrics/base.py` and `soccer_cv/pipeline.py`.

---

## Notes / Practical tips
- For speed, run analysis at **5–10 FPS**, then render at original FPS using interpolated overlays if desired.
- If your stitch introduces a visible seam, you can ignore it—detections/tracks will still work; the field mask helps.
- If jersey clustering fails (two teams wearing similar colors), set manual team colors in config or disable clustering.

---

## Troubleshooting

### Ball not detected often
- Increase `detector.ball_conf` (lower) and `detector.imgsz_ball` (higher)
- Enable motion fallback (default on)
- Use `soccer-cv export-ball-crops` to quickly create labeling data

### People missed
- Use a larger model (`yolov8s.pt` → `yolov8m.pt`)
- Increase `detector.imgsz_players` (e.g., 960 or 1280)

---

## License
MIT


## Running the CLI

Recommended (installs the `soccer-cv` command into your venv):

```bash
pip install -e .
soccer-cv --help
soccer-cv analyze --video /path/to/match.mp4 --out-dir out/run1 --config configs/default.yaml
```

Alternatively (no editable install), you can run the module directly:

```bash
python -m soccer_cv.cli --help
python -m soccer_cv.cli analyze --video /path/to/match.mp4 --out-dir out/run1 --config configs/default.yaml
```


## Improving tracking (simple heuristics)

Added in v0.1.5 (keeps things simple, but noticeably reduces noise):
- Player detection filtering using the field mask:
  - Keep a player detection only if the bottom-center of the bounding box (a proxy for the feet) is on the field.
  - Also filters extreme box sizes/aspect ratios to reduce background false positives.
- Optional ball gating:
  - Ignore ball candidates that are very far from all tracked players (removes many false positives).
- `export-ball-crops` command:
  - Exports square crops around motion candidates (and optionally YOLO ball detections) to accelerate labeling.

Tune these in `configs/default.yaml` under `heuristics:`.

## Export ball crops for labeling

```bash
soccer-cv export-ball-crops \
  --video /path/to/match.mp4 \
  --out-dir out/ball_crops \
  --config configs/default.yaml \
  --fps 8 \
  --crop-size 320 \
  --max-crops 2500
```

Output:
- `out/ball_crops/images/*.jpg`
- `out/ball_crops/manifest.csv`

## Open-source labeling

Two good open-source options:

- **CVAT** (recommended): Great for CV, supports YOLO export.
- **Label Studio**: Also open-source, easier to get started.

Label one class: `ball` (tight bounding box), then export YOLO detection labels.

## Fine-tune YOLO for ball

After exporting YOLO labels, place them into a dataset structure:

```
datasets/ball/
  images/train
  images/val
  labels/train
  labels/val
  data.yaml
```

Example `datasets/ball/data.yaml`:

```yaml
path: datasets/ball
train: images/train
val: images/val
names:
  0: ball
```

Train:

```bash
yolo detect train model=yolov8n.pt data=datasets/ball/data.yaml imgsz=640 epochs=100 device=cpu
```

Then update `detector.model_path` in `configs/default.yaml` to your new `best.pt`.


### Using a separate fine-tuned ball model (recommended)

If you fine-tune a YOLO model that only has the `ball` class (instead of COCO's `sports ball`),
keep the person model as-is and configure:

```yaml
detector:
  model_path: weights/yolov8n.pt          # keeps person detection
  ball_model_path: /path/to/ball/best.pt  # your fine-tuned ball detector
  ball_class_name: ball                   # class name inside your ball model
```
