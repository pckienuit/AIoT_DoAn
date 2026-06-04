# AIoT Flight Face Lookup

Airport booking and check-in prototype with face lookup on an edge device.

Passengers search flights, book seats, pay through a mock payment flow, and
register a face embedding in the browser. The FastAPI server stores booking data
and Qdrant vectors, then a MaixCAM device syncs encrypted vectors and recognizes
passengers locally to show flight, gate, boarding, and seat information.

## What Is Included

- FastAPI backend in `server/`
- Static browser app in `web_stage3/`, served by FastAPI
- Qdrant vector storage through Docker Compose
- SQLite prototype database by default, with optional MySQL mode
- MaixCAM edge runtime in `MaixCAM_App/`
- Face pipeline: YOLO face detection, V9 landmarks, ArcFace P3 embeddings
- Crypto path:
  - AES-GCM for browser-to-server face vector registration
  - XTEA-CTR for server-to-edge cache sync and edge fallback matching
- Training, export, benchmark, device sync, and hardware test scripts in `scripts/`

## Quick Start

### 1. Create a Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r server\requirements.txt
```

Optional packages for device deployment, model work, and benchmarks:

```powershell
pip install paramiko torch torchvision opencv-python pandas numpy scikit-learn matplotlib python-dotenv tqdm
```

If you switch to `DB_TYPE=mysql`, install the MySQL driver used by
`server/database.py`:

```powershell
pip install mysql-connector-python
```

### 2. Start Qdrant

```powershell
docker compose up -d
```

Qdrant runs on:

- HTTP: `http://localhost:6333`
- gRPC: `localhost:6334`

### 3. Start the API and web app

Use the helper script:

```powershell
.\start_server.ps1 -Reload
```

Or run Uvicorn directly:

```powershell
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- Web app: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Vector health: `POST http://localhost:8000/health/vector`

On startup the server creates the local SQLite database at
`server/data/prototype.db` and seeds prototype flights when SQLite is used.

## System Flow

1. Browser app calls `/api/flights/search`, `/api/seat-holds`, `/api/bookings`,
   and payment APIs for the booking flow.
2. Browser captures a 128D face embedding and registers it through
   `/api/face/register`.
3. Server validates the vector, stores booking data in SQLite/MySQL, and upserts
   the embedding into Qdrant collection `face_embeddings`.
4. MaixCAM syncs flight vectors from `/api/sync/{flight_id}` into
   `/root/cache/flight_<id>.json`.
5. MaixCAM runs camera inference locally and matches against the encrypted local
   cache. It can call `/api/face/match` as a fallback when enabled.
6. Admin tools can trigger edge sync, inspect Qdrant points, update flight
   states, and clear/reseed prototype data.

## Project Structure

```text
AIoT_DoAn/
|-- server/                 # FastAPI backend
|   |-- main.py             # App entry, startup, health, static web mount
|   |-- routes.py           # Auth, flight, seat, booking, payment APIs
|   |-- face_routes.py      # Face registration, matching, edge sync payloads
|   |-- admin_routes.py     # Admin dashboard APIs and MaixCAM sync controls
|   |-- database.py         # SQLite/MySQL schema and connection helpers
|   |-- vector_service.py   # Qdrant collection/search helpers
|   |-- crypto_service.py   # AES-GCM and XTEA-CTR helpers
|   `-- seed.py             # Prototype data seeding
|-- web_stage3/             # Static browser UI served by FastAPI
|   |-- index.html
|   |-- pages/              # Login, search, booking, check-in, admin, kiosk
|   |-- js/                 # Router, API client, page modules, components
|   `-- css/                # Base, layout, component, utility styles
|-- MaixCAM_App/            # Edge app copied to /root on MaixCAM
|   |-- main.py             # Camera -> YOLO -> V9 -> ArcFace P3 -> cache match
|   |-- config.py           # Loads /root/config.json with local defaults
|   |-- config.json         # Local config deployed to /root/config.json
|   |-- sync_cache.py       # Encrypted server sync and local cache matching
|   |-- display.py          # HUD/LCD drawing helpers
|   `-- mjpeg_server.py     # MJPEG monitor stream on port 8080
|-- scripts/
|   |-- train/              # Training entry points
|   |-- export/             # ONNX/cvimodel export and upload helpers
|   |-- sync/               # VPS/model sync helpers
|   |-- tests/              # Import, crypto, edge, webcam, hardware checks
|   `-- utils/              # Diagnostics and VPS helpers
|-- models/                 # Selected model artifacts and exports
|-- data/                   # Labels, calibration metadata, reference files
|-- docs/                   # Design notes, reports, and fix logs
|-- report/                 # Thesis/report LaTeX source
|-- docker-compose.yml      # Qdrant service
|-- start_server.ps1        # Windows helper for FastAPI
|-- deploy_to_device.py     # Upload MaixCAM_App files to device
|-- sync_maixcam.py         # Trigger or upload edge cache sync
`-- restart_maixcam.py      # Graceful MaixCAM app restart helper
```

## Main API

Authentication:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/auth/register` | Create user account and return a bearer token |
| `POST` | `/api/auth/login` | Login and return a bearer token |
| `GET` | `/api/auth/me` | Get current user profile |
| `PATCH` | `/api/auth/me` | Update current user profile |

Flights, seats, bookings, and payments:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/airports` | List airports |
| `GET` | `/api/stats` | Public summary stats |
| `GET` | `/api/flights/search` | Search available flights |
| `GET` | `/api/flights/{flight_id}` | Get flight details |
| `GET` | `/api/flights/{flight_id}/seats` | Get seat map and availability |
| `POST` | `/api/seat-holds` | Hold a seat for 10 minutes |
| `DELETE` | `/api/seat-holds/{hold_token}` | Release a seat hold |
| `POST` | `/api/bookings` | Create a booking |
| `GET` | `/api/bookings` | List current user's bookings |
| `GET` | `/api/bookings/code/{code}` | Lookup booking by code |
| `PATCH` | `/api/bookings/{booking_id}/cancel` | Cancel a booking |
| `POST` | `/api/bookings/{booking_id}/change-seat` | Change a booking seat |
| `PATCH` | `/api/bookings/{booking_id}/checkin` | Mark booking checked in |
| `POST` | `/api/payments/init` | Start mock VNPay/MoMo/cash payment |
| `POST` | `/api/payments/callback` | Mock payment callback |
| `GET` | `/api/payments/{booking_id}` | Get latest payment status |

Face recognition:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/face/register` | Store an encrypted or plaintext 128D embedding for a booking |
| `POST` | `/api/face/match` | Match an encrypted or plaintext embedding against Qdrant |
| `GET` | `/api/sync/{flight_id}` | Return XTEA-encrypted embeddings for edge cache sync |

Admin APIs live under `/api/admin`. The admin router includes login, stats,
flight and booking management, face deletion, sync trigger/status/cache cleanup,
database clear/seed actions, health checks, flight status update jobs, and Qdrant
status/point maintenance.

## Web Routes

FastAPI serves the browser app directly:

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

1. Edit `MaixCAM_App/config.json`.

   Important fields:

   | Key | Purpose |
   | --- | --- |
   | `server_url` | FastAPI server URL reachable by the device |
   | `flight_ids` | Flights to sync when running direct multi-sync |
   | `sync_interval_sec` | Periodic sync interval |
   | `cache_dir` | Device cache directory, usually `/root/cache` |
   | `match_threshold` | Cosine distance threshold for a successful match |
   | `fallback_enabled` | Whether edge can call `/api/face/match` on cache miss |
   | `device_secret_key` | Shared server/edge key for XTEA cache encryption |
   | `enable_lcd` | Enable LCD display drawing |
   | `stream_fps_limit` | MJPEG stream FPS cap |
   | `stream_jpeg_quality` | MJPEG JPEG quality |
   | `ai_frame_interval` | Run detector once every N camera frames |
   | `recognition_frame_interval` | Run recognizer once every N detector frames |

2. Deploy the edge app:

```powershell
python deploy_to_device.py --host 10.154.36.1 --user root --password root
```

The deploy script copies `MaixCAM_App/*.py` and `MaixCAM_App/config.json` to
`/root` on the device.

3. Run or restart the app:

```powershell
python restart_maixcam.py
```

4. Trigger a sync:

```powershell
python sync_maixcam.py --flight 12
python sync_maixcam.py --flight 12 --direct
python sync_maixcam.py --all
```

5. Monitor the stream:

```text
http://<maixcam-ip>:8080
```

The running edge app also reacts to these files:

- `/root/active_flight.txt`: active flight ID
- `/root/sync_now.flag`: manual sync trigger
- `/root/cache/flight_<id>.json`: local encrypted cache

## Configuration

Use `.env` for local secrets and deployment-specific settings. Keep it out of
git.

Common environment variables:

| Variable | Purpose | Default/Notes |
| --- | --- | --- |
| `QDRANT_URL` | Qdrant HTTP endpoint | `http://127.0.0.1:6333` |
| `DB_TYPE` | Database mode | `sqlite` or `mysql`; default is `sqlite` |
| `MYSQL_HOST` / `MYSQL_PORT` | MySQL host and port | Used when `DB_TYPE=mysql` |
| `MYSQL_USER` / `MYSQL_PASSWORD` | MySQL credentials | Used when `DB_TYPE=mysql` |
| `MYSQL_DATABASE` | MySQL database name | Default `aiot_flight` |
| `AES_SECRET_KEY` | 64-hex-character AES-GCM key for browser registration | Override prototype default |
| `DEVICE_SECRET_KEY` | Shared edge key used by XTEA-CTR | Must match MaixCAM `device_secret_key` |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | Admin login override | Change before sharing/deploying |
| `SERVER_URL` | Server URL used by sync helpers/admin flows | Example `http://<server-ip>:8000` |
| `MAIXCAM_HOST` / `MAIXCAM_PORT` | MaixCAM SSH target for supported scripts | Common port `22` |
| `MAIXCAM_USER` / `MAIXCAM_PASS` | MaixCAM SSH credentials for supported scripts | Common user `root` |
| `VPS_HOST` / `VPS_PORT` | VPS SSH target for model sync/export helpers | Optional |
| `VPS_USER` / `VPS_PASS` | VPS SSH credentials | Optional |

Production notes:

- Replace prototype JWT/admin/crypto defaults before deployment.
- Use HTTPS or a private network for browser-to-server and server-to-device traffic.
- Keep `.env`, local databases, runtime logs, and large generated artifacts out of git.

## Scripts

Useful server/device scripts:

```powershell
.\start_server.ps1 -Reload
.\stop_server.ps1
python deploy_to_device.py
python sync_maixcam.py --flight 12
python restart_maixcam.py
python check_maixcam.py
```

Training and export entry points:

```powershell
python scripts\train\train_v9.py
python scripts\train\train_recognize.py
python scripts\export\export_v9.py
python scripts\export\export_recognize_onnx.py
python scripts\export\compile_recognize_p3_local.py
python scripts\export\compile_recognize_p3_vps.py
```

## Tests And Benchmarks

Script-style checks:

```powershell
python scripts\tests\test_imports.py
python scripts\tests\test_crypto_e2e.py
python scripts\tests\test_face_vector_guards.py
python scripts\tests\test_edge_integration.py
python scripts\tests\benchmark_edge_pc.py
python scripts\run_maixcam_stress_test.py
```

Hardware and live-camera helpers:

```powershell
python scripts\tests\run_hardware_test.py
python scripts\tests\run_live_test.py
python scripts\tests\run_live_mjpeg.py
python scripts\tests\test_recognize_webcam.py
```

## Documentation

- System plan: `docs/development_plan.md`
- Edge cache architecture: `docs/edge_cache_architecture.md`
- Security and performance report: `docs/aiot_security_and_performance_report.md`
- Runtime optimization report: `docs/maixcam_runtime_optimization_report.md`
- V9/P3 deployment notes: `docs/deploy_v9_p3_pipeline.md`
- Face similarity fix: `docs/face_similarity_fix.md`
- Benchmark report: `docs/benchmark_report.md`
- Seat uniqueness notes: `docs/booking-seat-uniqueness.md`
- Sync notes: `docs/sync_fix_notes.md`
- Thesis/report source: `report/`

## Cleanup Policy

Generated and local-only files should stay out of git:

- Python caches: `__pycache__/`, `*.pyc`
- Runtime logs: `*.log`
- Local environments: `.venv/`, `.venv-tpu310/`
- Local secrets: `.env`
- CodeGraph runtime files: `.codegraph/`
- Local databases and backups: `*.db`, `*.db-*`, `*.db.bak`, `*.sqlite*`
- Large generated datasets and model checkpoints unless intentionally versioned

The repository keeps source code, docs, calibration/reference metadata, selected
model exports, and project artifacts needed for reproducible demos.
