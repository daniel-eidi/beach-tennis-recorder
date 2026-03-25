# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Beach Tennis Recorder — mobile app (iOS + Android via Flutter) that records beach tennis matches, auto-detects rallies using YOLOv8 computer vision, and generates video clips of each point. MVP is offline-first with optional cloud sync.

Full project spec: `docs/project_spec.md`

## Stack

| Layer | Tech |
|---|---|
| Mobile | Flutter 3.x (Dart) |
| Computer Vision | YOLOv8n (Python), deployed as TFLite (INT8) / ONNX on device |
| Video Processing | FFmpeg via ffmpeg_kit (mobile) and Python wrapper (backend) |
| Backend | FastAPI (Python 3.11+) |
| Storage | Google Cloud Storage (GCS), Firestore |
| CI/CD iOS | Codemagic |

## Build & Run Commands

```bash
# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# Vision / ML training
cd vision
pip install ultralytics roboflow opencv-python
python train.py --config dataset/data.yaml --model yolov8n.pt

# Mobile
cd mobile
flutter pub get
flutter run

# Run backend tests
cd backend && pytest

# Run video processing tests
cd video && pytest tests/

# Run Flutter tests
cd mobile && flutter test
```

## Architecture

The codebase is split into 4 independent modules that communicate only through defined contracts:

```
mobile/    → Flutter app (camera, UI, on-device inference, clip library)
vision/    → YOLOv8 training, export, and inference logic (Python)
video/     → FFmpeg-based clip cutting, naming, thumbnails, compression (Python)
backend/   → FastAPI REST API, GCS storage, auth (Python)
tests/     → Cross-module integration tests and field test data
```

### Key Data Flow

1. **Camera → Buffer**: `CameraService` captures 1080p/30fps; `BufferService` maintains a 60s circular buffer on disk (not memory).
2. **Buffer → Rally Detection**: `RallyController` runs a state machine (IDLE → EM_JOGO → FIM_RALLY) using TFLite inference in a separate Flutter `Isolate`.
3. **Rally → Clip**: On FIM_RALLY, `ClipService` extracts buffer segment (pre-rally 3s + rally + post-rally 2s) via FFmpeg.
4. **Clip → Cloud** (optional): `clip_processor.py` uploads via `POST /api/v1/clips/upload` to backend, which stores in GCS.

### TFLite Model Contract

- File: `mobile/assets/models/ball_detector.tflite`
- Input: `[1, 640, 640, 3]` float32 (RGB normalized 0–1)
- Output: `[1, 25200, 6]` float32 (x, y, w, h, confidence, class)
- Classes: `{0: "ball", 1: "net", 2: "court_line"}`

### Rally State Machine Constants

```python
BUFFER_PRE_RALLY_SECONDS = 3
BUFFER_POST_RALLY_SECONDS = 2
VELOCITY_THRESHOLD = 15.0        # px/frame
CONFIDENCE_THRESHOLD = 0.45
RALLY_TIMEOUT_SECONDS = 8
NET_CROSS_REQUIRED = True         # feature flag
```

### API Endpoints

- `POST /api/v1/clips/upload` — multipart upload (file, match_id, rally_number, duration_seconds, timestamp_start)
- `GET /api/v1/clips?match_id={id}` — list clips with signed URLs
- `POST /api/v1/matches` — create match, returns match_id
- `GET /health` — health check

### Clip Naming Convention

`rally_{match_id}_{number}_{date}_{time}.mp4`

## Development Rules

- **Commit prefix**: `[AGENT-XX] type: description` (e.g. `[AGENT-02] feat: export tflite model`)
- **API contracts between modules are immutable** — changes require explicit coordination across affected modules
- **Structured JSON logs**: `{"agent": "02", "task": "02-11", "status": "ok", "ms": 42}`
- **Use feature flags** for experimental behavior (e.g. `NET_CROSS_REQUIRED`)
- **No direct cross-module imports** — modules communicate only via the defined contracts above
- **Frame processing must run in a separate Isolate** in Flutter (never on UI thread)
- **Buffer is disk-based**, not in-memory — must support 60s of 1080p on low-RAM devices
- **Court calibration** is done once per location, saved locally

## MVP Acceptance Criteria

| Metric | Target |
|---|---|
| Rally recall | >= 85% |
| False positive rate | <= 10% |
| Detection latency | <= 50ms/frame |
| Clip generation time | <= 5s after rally end |
| Battery (1h recording) | <= 40% drain |
| Avg clip size | <= 30MB |
| Test coverage | >= 60% per module |

## Environment Variables

Required in `.env`:
```
GCS_BUCKET_NAME=beach-tennis-clips
GCS_PROJECT_ID=<gcp-project>
API_KEY=<api-key>
CODEMAGIC_API_TOKEN=<token>
```
