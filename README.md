# AIoT Flight Face Lookup

End-to-end airport flight lookup by face recognition. A passenger registers a face vector in the browser, the server stores booking and vector data, and a MaixCAM edge device recognizes the passenger to show flight, gate, boarding, and seat information.

## Current Status

- Web booking/check-in prototype: `web_stage3/`
- FastAPI backend: `server/`
- Qdrant vector storage via Docker Compose
- MaixCAM edge runtime with local encrypted cache: `MaixCAM_App/`
- Face pipeline: YOLO face detection, V9 landmarks, ArcFace P3 embeddings
- Security path: AES-GCM for Web to Server, XTEA-CTR for Server to Edge cache sync

## Quick Start

### 1. Create environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r server\requirements.txt
```

For training, export, and benchmark scripts, install the additional ML packages used by the scripts:

```powershell
pip install torch torchvision opencv-python pandas numpy scikit-learn matplotlib paramiko python-dotenv tqdm
```

### 2. Start Qdrant

```powershell
docker compose up -d
```

Qdrant runs on `http://localhost:6333`.

### 3. Start the server and web app

```powershell
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- Web app: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Vector health: `POST http://localhost:8000/health/vector`

## Project Structure

```text
AIoT_DoAn/
+-- MaixCAM_App/          # Edge app running on MaixCAM
|   +-- main.py           # YOLO -> V9 -> ArcFace P3 -> cache match
|   +-- config.py         # Runtime config loader
|   +-- config.json       # Device/server settings
|   +-- display.py        # HUD/LCD drawing helpers
|   +-- mjpeg_server.py   # HTTP MJPEG stream
|   +-- sync_cache.py     # Encrypted cache sync from server
|   +-- stress_test_maixcam.py
+-- server/               # FastAPI backend
|   +-- main.py           # App entry, routes, static web mount
|   +-- routes.py         # Core flight, booking, payment APIs
|   +-- face_routes.py    # Face register, match, sync APIs
|   +-- admin_routes.py   # Admin dashboard and device sync APIs
|   +-- database.py       # SQLite/MySQL schema and connection helpers
|   +-- vector_service.py # Qdrant collection/search helpers
|   +-- crypto_service.py # AES-GCM and XTEA-CTR helpers
+-- web_stage3/           # Browser UI served by FastAPI
|   +-- index.html
|   +-- pages/            # Login, search, booking, check-in, admin, kiosk
|   +-- js/               # Router, API client, page modules, components
|   +-- css/              # Base, layout, component, utility styles
+-- scripts/              # Training, export, sync, and test helpers
|   +-- train/
|   +-- export/
|   +-- sync/
|   +-- tests/
|   +-- utils/
+-- data/                 # Calibration data and training metadata
+-- models/               # Checkpoints and exported ONNX/cvimodel artifacts
+-- results/              # Evaluation outputs
+-- docs/                 # Design notes, reports, and fix logs
+-- docker-compose.yml    # Qdrant service
```

## Main API

Core passenger and booking APIs live under `/api`:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/airports` | List airports |
| `GET` | `/api/flights/search` | Search available flights |
| `GET` | `/api/flights/{flight_id}` | Get flight details |
| `GET` | `/api/flights/{flight_id}/seats` | Get seat map and availability |
| `POST` | `/api/seat-holds` | Temporarily hold a seat |
| `POST` | `/api/bookings` | Create a booking |
| `GET` | `/api/bookings/code/{code}` | Lookup booking by code |
| `PATCH` | `/api/bookings/{booking_id}/checkin` | Mark booking checked in |
| `POST` | `/api/payments/init` | Start a payment flow |

Face recognition APIs:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/face/register` | Store a 128D embedding for a booking |
| `POST` | `/api/face/match` | Match an embedding against Qdrant |
| `GET` | `/api/sync/{flight_id}` | Sync encrypted vectors to edge cache |

Admin APIs live under `/api/admin`, including flight status updates, booking status updates, sync triggers, Qdrant status, and maintenance endpoints.

## Web Routes

FastAPI serves `web_stage3/` directly:

- `/login`
- `/register`
- `/search`
- `/booking`
- `/seat-map`
- `/payment`
- `/confirmation`
- `/my-tickets`
- `/checkin`
- `/lookup`
- `/register-face`
- `/kiosk`
- `/admin`

## MaixCAM Workflow

1. Configure server/device settings in `MaixCAM_App/config.json`.
2. Deploy the app to the device:

```powershell
python deploy_to_device.py
```

3. Sync data or trigger a remote run with the helper scripts:

```powershell
python sync_maixcam.py
python restart_maixcam.py
```

4. Open the MJPEG monitoring stream when the app is running:

```text
http://<maixcam-ip>:8080
```

## Tests And Benchmarks

Useful checks:

```powershell
python scripts\tests\test_imports.py
python scripts\tests\test_crypto_e2e.py
python scripts\tests\test_edge_integration.py
python scripts\tests\benchmark_edge_pc.py
python scripts\run_maixcam_stress_test.py
```

Recent benchmark notes are in:

- `docs/maixcam_runtime_optimization_report.md`
- `docs/face_similarity_fix.md`
- `docs/benchmark_report.md`
- `docs/development_plan.md`

## Configuration

The project uses `.env` for local secrets and device/server settings. Keep `.env` local only.

Common values used by scripts and services:

| Variable | Purpose |
| --- | --- |
| `SERVER_HOST` / `SERVER_PORT` | FastAPI binding or helper script target |
| `QDRANT_URL` | Qdrant endpoint |
| `DATABASE_URL` | Optional MySQL/PostgreSQL-style DB URL when not using SQLite |
| `AES_MASTER_KEY` | AES-GCM key material for face vector transport |
| `XTEA_DEVICE_KEY` | Edge cache encryption key |
| `MAIXCAM_HOST` | Device IP or host |
| `MAIXCAM_USER` / `MAIXCAM_PASSWORD` | SSH/SFTP deployment credentials |

## Cleanup Policy

Generated files should stay out of git:

- Python caches: `__pycache__/`, `*.pyc`
- Runtime logs: `*.log`
- Local environments: `.venv/`, `.venv-tpu310/`
- Local secrets: `.env`
- CodeGraph runtime DB/logs: `.codegraph/`
- Large datasets and model exports unless intentionally versioned
- Local database files and backups: `*.db`, `*.db.bak`, `*.sqlite*`

The repository keeps source code, docs, calibration samples, and selected project artifacts needed for reproducible demos.

## Documentation

- System plan: `docs/development_plan.md`
- Edge cache architecture: `docs/edge_cache_architecture.md`
- Face similarity fix: `docs/face_similarity_fix.md`
- Seat uniqueness notes: `docs/booking-seat-uniqueness.md`
- Sync notes: `docs/sync_fix_notes.md`
