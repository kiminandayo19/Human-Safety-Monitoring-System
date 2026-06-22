# Human Safety Monitoring System

A FastAPI boilerplate for a computer-vision **Human Safety Monitoring System** —
detecting safety conditions (missing PPE, falls, restricted-zone intrusion, …)
from camera frames and escalating them into events and alerts.

> This is a **boilerplate**. The CV model is a pluggable stub — wire in your own
> model (YOLO/Ultralytics, ONNX, Detectron2, …) inside `app/services/detector.py`.

## Project structure

```
app/
├── main.py                  # App factory + lifespan (loads model on boot)
├── core/
│   ├── config.py            # Settings (env / .env)
│   └── logging.py           # Logging setup
├── api/
│   ├── deps.py              # Shared dependencies
│   └── v1/
│       ├── router.py        # Aggregates v1 routers
│       └── endpoints/
│           ├── health.py        # /health, /ready
│           ├── detections.py    # /detections
│           └── monitoring.py    # /monitoring/evaluate, /monitoring/alerts
├── schemas/                 # Pydantic models (detection, monitoring)
└── services/
    ├── detector.py          # CV model wrapper (stub — replace me)
    └── safety_monitor.py    # Rules: detections -> events -> alerts
tests/                       # Pytest smoke tests
```

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

| Method | Path                        | Description                              |
|--------|-----------------------------|------------------------------------------|
| GET    | `/api/v1/health`            | Liveness probe                           |
| GET    | `/api/v1/ready`             | Readiness (model loaded?)                |
| POST   | `/api/v1/detections`        | Run detection on an uploaded frame       |
| POST   | `/api/v1/monitoring/evaluate` | Detection → safety events              |
| POST   | `/api/v1/monitoring/alerts` | Detection → escalated alerts             |

## Wiring in a real model

Edit `app/services/detector.py`:

- `load()` — load model weights.
- `predict()` — preprocess, run inference, return a `DetectionResult`.

Extend safety rules in `app/services/safety_monitor.py` (`SafetyMonitor.evaluate`).

## Tests

```bash
.venv/bin/pytest
```

## Docker

```bash
docker build -t inaai .
docker run -p 8000:8000 inaai
```
