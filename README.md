# Human Safety Monitoring System

A FastAPI computer-vision service for workplace human-safety monitoring. It
combines a fine-tuned **YOLO PPE detector** (hard hat / mask / safety vest and
their "missing" counterparts) with a pretrained **YOLOv11 person detector +
ByteTrack tracker**, and exposes them over HTTP for single-frame detection and
full-video person tracking.

Both models are loaded once at startup and served from memory. For tracking,
each detected person is annotated with their PPE compliance state and assigned a
stable track ID, with tracking-identity consistency reported via an ID Switch
Rate and throughput as FPS.

## Project structure

```
app/
├── main.py                  # App factory + lifespan (loads both models on boot)
├── core/
│   ├── config.py            # Settings (model paths, thresholds, output dir)
│   └── logging.py           # Logging setup
├── api/
│   ├── deps.py              # Shared singletons (detector, monitor)
│   └── v1/
│       ├── router.py        # Aggregates v1 routers
│       └── endpoints/
│           ├── health.py        # /health, /ready
│           ├── detections.py    # /detections
│           ├── monitoring.py    # /monitoring/evaluate, /monitoring/alerts
│           ├── detect.py        # /detect  (detection_type: ppe | person)
│           └── track.py         # /track-person (video -> overlay video + JSON)
├── schemas/                 # Pydantic models (detection, monitoring, tracking)
├── services/
│   ├── detector.py          # YOLO wrapper: load / detect / annotate / track_persons
│   └── safety_monitor.py    # Rules: detections -> events -> alerts
└── utils/
    └── storage.py           # Persistence helpers (image / JSON / video artifacts)
models/                      # YOLO weights (yolo_50.pt, yolo_person_tracker.pt)
tests/                       # Pytest smoke tests
```

## Models

| Role                     | Weights                          | Notes                              |
|--------------------------|----------------------------------|------------------------------------|
| PPE detection            | `models/yolo_50.pt`              | 6 classes (see below)              |
| Person detection/tracker | `models/yolo_person_tracker.pt`  | Pretrained YOLOv11, COCO class `0` |

PPE classes: `hard_hat`, `no_hard_hat`, `mask`, `no_mask`, `safety_vest`,
`no_safety_vest`. Paths and the confidence threshold are configurable in
`app/core/config.py` (`PPE_MODEL_PATH`, `PERSON_MODEL_PATH`,
`PERSON_DETECTION_THRESHOLD`, `DETECTION_OUTPUT_DIR`).

## Quickstart

```bash
# 1. Install deps (using the existing .venv)
.venv/bin/pip install -e ".[dev]"

# 2. Configure
cp .env.example .env

# 3. Run
.venv/bin/uvicorn app.main:app --reload

# 4. Open API docs
open http://localhost:8000/docs
```

## Endpoints

| Method | Path                          | Description                                        |
|--------|-------------------------------|----------------------------------------------------|
| GET    | `/api/v1/health`              | Liveness probe                                     |
| GET    | `/api/v1/ready`               | Readiness (both models loaded?)                    |
| POST   | `/api/v1/detect`              | Detect PPE or persons in an image                  |
| POST   | `/api/v1/track-person`        | Track persons + PPE through a video                |
| POST   | `/api/v1/detections`          | Run detection on an uploaded frame                 |
| POST   | `/api/v1/monitoring/evaluate` | Detection → safety events                          |
| POST   | `/api/v1/monitoring/alerts`   | Detection → escalated alerts                       |

### Examples

```bash
# Single-frame detection (detection_type: ppe | person)
curl -X POST http://localhost:8000/api/v1/detect \
  -F "file=@image.jpg" \
  -F "detection_type=ppe" \
  -F "save_detection=true"

# Person tracking over a video
curl -X POST http://localhost:8000/api/v1/track-person \
  -F "file=@datas/hardhat.mp4"
```

## Output artifacts

When `save_detection=true` (detection) or for every tracking run, artifacts are
written to `DETECTION_OUTPUT_DIR` (default `datas/detections/`):

- **Detection** — annotated image `{type}_detection_{id}.{ext}` + JSON response `.json`.
- **Tracking** — annotated video `track_person_{id}.mp4` + per-frame JSON
  `track_person_{id}.json` (tracked persons, PPE booleans, ID Switch Rate, FPS).

## Tests

```bash
.venv/bin/pytest
```

## Docker

```bash
docker build -t hsms .
docker run -p 8000:8000 hsms
```
