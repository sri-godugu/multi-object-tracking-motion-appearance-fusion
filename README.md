# Multi-Object Tracking using Motion–Appearance Fusion

A DeepSORT-style multiple object tracking pipeline combining object detection, Kalman filter–based motion modeling, and Hungarian algorithm–based data association with deep appearance features for robust identity preservation under occlusion and object crossings.

## Pipeline

```
Detection → Feature Extraction → Prediction → Association → Track Update
```

| Stage | Component | Description |
|-------|-----------|-------------|
| **Detection** | YOLOv8 | Frame-level bounding box detection |
| **Feature Extraction** | ResNet-50 | 128-d L2-normalised appearance embedding per crop |
| **Prediction** | Kalman Filter | Constant-velocity motion model forecasts next state |
| **Association** | Hungarian Algorithm | Assigns detections to tracks via motion + appearance cost |
| **Track Update** | Track Manager | Updates matched tracks, confirms tentative ones, deletes lost ones |

---

## Architecture

### Kalman Filter — Motion Modeling
**State**: `[cx, cy, w, h, vcx, vcy, vw, vh]` — bounding-box centre, size, and their velocities.

Models each track as a constant-velocity system. Position noise is scaled proportionally to bounding-box size so the filter adapts naturally to different object scales. The covariance matrix grows during missed frames and is corrected on each matched detection.

### Appearance Feature Extraction
A pre-trained ResNet-50 extracts appearance embeddings from detection crops (resized to 128×64). Features are projected to 128 dimensions and L2-normalised. Each track maintains a rolling gallery of up to 100 embeddings; matching uses nearest-neighbour cosine distance over the gallery.

### Data Association — Hungarian Algorithm
Two-stage cascade matching:

1. **Cascade appearance matching** — Confirmed tracks are matched using cosine distance on appearance galleries. Tracks seen most recently are prioritised (level 1 of the cascade gets first pick over older tracks).
2. **IoU fallback matching** — Unconfirmed / tentative tracks and recently-lost confirmed tracks are matched using 1 − IoU distance. This handles fast-moving objects where appearance may drift.

The linear sum assignment at each stage is solved with `scipy.optimize.linear_sum_assignment` (Kuhn–Munkres / Hungarian algorithm, O(n³)).

---

## Installation

```bash
git clone https://github.com/sri-godugu/multi-object-tracking-motion-appearance-fusion.git
cd multi-object-tracking-motion-appearance-fusion

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **GPU note**: A CUDA-capable GPU is strongly recommended. On first run YOLOv8 weights (~6 MB for `yolov8n.pt`) are downloaded automatically by Ultralytics.

---

## Usage

### Track a video file
```bash
python scripts/track_video.py --source video.mp4 --output tracked.mp4
```

### Track persons only and show live preview
```bash
python scripts/track_video.py --source video.mp4 --classes 0 --display
```

### Use webcam
```bash
python scripts/track_video.py --source 0 --display
```

### Full argument reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | *required* | Video path or camera index |
| `--output` | `None` | Output `.mp4` path |
| `--detector` | `yolov8n.pt` | YOLOv8 weights (`yolov8n/s/m/l/x.pt`) |
| `--conf` | `0.5` | Detection confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--classes` | `[0]` | COCO class IDs to track (`0` = person) |
| `--max-cosine-dist` | `0.3` | Appearance matching threshold |
| `--max-iou-dist` | `0.7` | IoU matching threshold |
| `--max-age` | `30` | Frames without match before deleting a track |
| `--n-init` | `3` | Consecutive hits before confirming a track |
| `--device` | auto | `cuda` or `cpu` |
| `--display` | `False` | Show OpenCV preview window |

---

### Evaluate on MOT benchmark
```bash
# Produces per-sequence .txt files in MOT challenge format
python scripts/evaluate.py \
    --data-dir data/MOT17/train \
    --output-dir results/MOT17

# Feed results to TrackEval
python -m trackeval.scripts.run_mot_challenge \
    --GT_FOLDER data/MOT17/train \
    --TRACKERS_FOLDER results \
    --BENCHMARK MOT17
```

---

## Project Structure

```
├── configs/
│   └── mot.yaml              # Hyper-parameters
├── src/
│   ├── detection/
│   │   ├── detection.py      # Detection data class (tlwh / tlbr / xywh)
│   │   └── detector.py       # YOLOv8 wrapper
│   ├── features/
│   │   └── extractor.py      # ResNet-50 appearance extractor
│   ├── tracking/
│   │   ├── kalman_filter.py  # Kalman predict / update / gating
│   │   ├── track.py          # Track state machine (Tentative → Confirmed → Deleted)
│   │   └── tracker.py        # DeepSORT-style tracker orchestration
│   ├── association/
│   │   ├── iou_matching.py   # IoU cost matrix
│   │   ├── nn_matching.py    # Nearest-neighbour cosine cost matrix
│   │   └── linear_assignment.py  # Hungarian + cascade matching
│   └── utils/
│       ├── visualization.py  # Draw tracks / detections on frames
│       └── io_utils.py       # VideoReader / VideoWriter
├── scripts/
│   ├── track_video.py        # End-to-end tracking script
│   └── evaluate.py           # MOT benchmark evaluation
├── tests/
│   ├── test_kalman_filter.py
│   ├── test_hungarian.py
│   └── test_tracker.py
├── models/                   # Place custom re-ID weights here
├── data/                     # Place video / MOT sequences here
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Results

*(To be added after GPU evaluation — push results videos / metrics here)*

---

## References

- Wojke, N., Bewley, A., & Paulus, D. (2017). [Simple Online and Realtime Tracking with a Deep Association Metric (DeepSORT)](https://arxiv.org/abs/1703.07402). *ICIP 2017*.
- Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). [Simple Online and Realtime Tracking (SORT)](https://arxiv.org/abs/1602.00763). *ICIP 2016*.
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [MOT Challenge Benchmark](https://motchallenge.net/)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
