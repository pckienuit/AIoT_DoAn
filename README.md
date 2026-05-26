# Face Check-in System — AIoT Project

An end-to-end facial-recognition-based flight information lookup system for airport check-in kiosks. Passengers look at a camera and automatically retrieve their flight details (gate, boarding time, seat, status) without presenting a boarding pass.

The system has three major components:

1. **Web Client** — Face registration via browser (MediaPipe detection + V9 landmarks + ArcFace P3 embedding, AES-GCM-256 encryption)
2. **FastAPI Server** — Stores passenger data (SQLite) and face embeddings (Qdrant vector DB), exposes REST API
3. **MaixCAM Edge Device** — Real-time face recognition on a RISC-V edge device (YOLO detection + V9 landmarks + ArcFace P3, XTEA-CTR local cache)

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Server Backend](#server-backend)
5. [Edge Device (MaixCAM)](#edge-device-maixcam)
6. [Web Client](#web-client)
7. [AI Models](#ai-models)
8. [Security](#security)
9. [Performance Benchmarks](#performance-benchmarks)
10. [Development Phases](#development-phases)
11. [.env Template](#env-template)
12. [References](#references)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Web Client (Browser)                                               │
│  MediaPipe → V9 Landmarks → ArcFace P3 → AES-GCM-256 encrypt      │
│  └─ POST /api/face/register {encrypted_vector, booking_id}        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTPS
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  FastAPI Server                                                     │
│  ├─ SQLite ── passengers / flights / bookings (metadata)           │
│  └─ Qdrant ── face_embeddings collection (128D vectors, HNSW)      │
│      ├─ POST /api/face/register   (decrypt → Qdrant upsert)       │
│      ├─ POST /api/face/match       (Qdrant ANN search)              │
│      └─ GET  /api/sync/{flight_id} (scroll → XTEA-CTR encrypt)    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTPS / WiFi
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  MaixCAM Edge Device (RISC-V C906 @ 1GHz)                          │
│  Camera GC4653 ── YOLOv8n ── V9 Landmarks ── ArcFace P3           │
│  └─ XTEA-CTR decrypt → cosine match local cache (<1ms)            │
│  └─ LCD / MJPEG stream (port 8080) / fallback to server API       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
d-AIoT-DoAn/
├── .env                              # Environment variables (VPS, MaixCAM, Server)
├── .gitignore
├── README.md                         # This file
├── labels.csv                        # Merged CelebA metadata CSV
├── reorganize.py                     # Directory reorganization script
│
├── server/                           # FastAPI backend server
│   ├── main.py                       # FastAPI entry point, CORS, health endpoints
│   ├── routes.py                     # CRUD: passengers, flights, bookings
│   ├── face_routes.py                # Face register / match / sync endpoints
│   ├── database.py                   # SQLite schema + session management
│   ├── vector_service.py             # Qdrant client: upsert, search, scroll
│   ├── crypto_service.py             # AES-GCM + XTEA-CTR encryption helpers
│   ├── requirements.txt              # Python dependencies
│   └── data/
│       └── prototype.db              # SQLite database file
│
├── MaixCAM_App/                      # Edge device app (runs on MaixCAM)
│   ├── main.py                       # Full 3-stage pipeline: YOLO → V9 → P3 → cache
│   ├── config.py                     # Configuration loader (server URL, thresholds)
│   ├── config.json                   # Runtime configuration
│   ├── sync_cache.py                 # Cache manager: sync encrypted vectors from server
│   ├── display.py                    # LCD/overlay drawing helpers
│   ├── mjpeg_server.py               # MJPEG HTTP stream server (port 8080)
│   └── stress_test_maixcam.py        # On-device stress test
│
├── scripts/                          # All scripts: training, export, sync, tests
│   ├── train/
│   │   ├── train.py                 # Original training script
│   │   ├── train_v8.py              # v8: Wing + Focal loss, 60 epochs
│   │   ├── train_v9.py              # v9: Fine-tune from v8, 90 epochs, Gaussian noise
│   │   ├── train_recognize.py       # ArcFace P3 training (CASIA-WebFace)
│   │   └── prepare_data.py          # Merge CelebA metadata CSVs → labels.csv
│   ├── export/
│   │   ├── export_onnx.py           # PyTorch → ONNX export (V3 architecture)
│   │   ├── export_v9.py             # v9-specific export
│   │   ├── compile_vps.py           # Compile model via MaixHub/YOLO on VPS
│   │   ├── create_calib_data.py     # Generate 100 calibration images (224×224 JPG)
│   │   ├── zip_model.py            # Create MaixHub-compatible ZIP
│   │   ├── upload_to_maixcam.py     # Upload model files to MaixCAM via SSH/SFTP
│   │   ├── maixcam_main.py          # Standalone inference app (alternative entry)
│   │   ├── export_recognize_onnx.py
│   │   ├── compile_recognize_p3_local.py
│   │   ├── compile_recognize_p3_vps.py
│   │   └── check_vps_result.py
│   ├── utils/
│   │   ├── vps_sync.py              # SSH/SFTP: download .pth from VPS
│   │   ├── check_vps.py             # Check training results on VPS
│   │   ├── check_train_progress.py  # Monitor training progress
│   │   ├── verify_vps.py            # Validate checkpoint integrity
│   │   ├── find_pth.py             # Find the latest .pth file
│   │   ├── download_v6.py          # Download checkpoint v6
│   │   ├── debug_negatives.py       # Debug hard negative samples
│   │   └── face_align.py           # Face alignment helpers
│   ├── sync/
│   │   ├── vps_sync.py
│   │   ├── upload_recognize.py
│   │   └── upload_casia.py
│   ├── tests/
│   │   ├── evaluate_models.py       # Comparative evaluation (Acc, F1, AUC, NME, MAE)
│   │   ├── inference_test.py        # Quick inference on CelebA images
│   │   ├── webcam_test.py          # Haar Cascade + Model landmark pipeline
│   │   ├── webcam_test_v2.py        # Model-only 2-pass webcam pipeline
│   │   ├── visualize.py            # Visualize evaluation results
│   │   ├── benchmark_edge_pc.py     # PC-side benchmark for edge pipeline
│   │   ├── stress_test.py
│   │   ├── quick_stress_test.py
│   │   ├── run_live_test.py
│   │   ├── run_live_mjpeg.py
│   │   ├── run_device_app.py
│   │   ├── run_foreground_main.py
│   │   ├── evaluate_recognize.py
│   │   ├── test_recognize_webcam.py
│   │   ├── test_crypto_e2e.py
│   │   ├── test_device_sync.py
│   │   ├── test_edge_integration.py
│   │   ├── test_hardware_init.py
│   │   ├── test_hardware_init_isolated.py
│   │   ├── test_image_methods.py
│   │   ├── test_image_doc.py
│   │   ├── test_camera_doc.py
│   │   ├── test_imports.py
│   │   ├── download_model.py
│   │   ├── kill_and_check.py
│   │   ├── reboot_force.py
│   │   ├── check_vps_paths.py
│   │   └── run_hardware_test.py
│   ├── benchmark_models.py           # Model benchmark script
│   └── run_maixcam_stress_test.py   # Auto-upload + run stress test on MaixCAM
│
├── models/                           # Model files
│   ├── checkpoints/                 # .pth PyTorch checkpoints
│   │     ├── face_detect_model.pth
│   │     ├── face_detect_model_withval*.pth           (v1–v13)
│   │     └── face_detect_model_vps_finetune*.pth     (v1–v9)
│   └── exports/                     # Deployable files: .onnx, .mud, .cvimodel, .zip
│         ├── face_detect_v9.onnx
│         ├── face_detect_v9.mud
│         ├── face_detect_v9.cvimodel
│         ├── maixhub_upload_v9.zip
│         ├── face_recognize_arcface_p3.onnx
│         ├── face_recognize_arcface_p3.mud
│         ├── face_recognize_arcface_p3.cvimodel
│         └── maixhub_upload_p3.zip
│
├── data/
│   ├── labels.csv                   # Merged CelebA metadata
│   ├── _weight_map.csv
│   ├── face_recognize_arcface_p3.ref_files.json
│   ├── recognize_train.log
│   └── images/                      # 100 calibration images (calib_000–099)
│         └── calib_*.jpg
│
├── results/
│   ├── evaluation_results.csv
│   └── eval_v9_fixed.csv
│
├── docs/
│   ├── development_plan.md          # Full system design and roadmap (Vietnamese)
│   ├── webcam_tracking_pipeline_v2.md
│   ├── maixhub_zip_issue.md
│   ├── crop_padding_bug.md
│   ├── deploy_v9_p3_pipeline.md
│   ├── benchmark-pipeline.md
│   └── benchmark_report.md
│
├── scratch/                         # Experimentation / scratch files
│   ├── finetune_phase3.py
│   ├── launch_phase3.py
│   ├── monitor_phase3.py
│   └── compare_models.py
│
└── deploy_to_device.py              # Root-level deploy script (uploads MaixCAM_App via SSH)
```

---

## Installation

### Requirements

- Python 3.10+
- PyTorch (CUDA if GPU available)
- OpenCV (`opencv-python`)
- pandas, numpy, scikit-learn, matplotlib
- `paramiko` (for SSH/SFTP to VPS and MaixCAM)
- `python-dotenv` (reads `.env` file)
- `tqdm` (progress bars)
- `fastapi`, `uvicorn` (for server backend)
- `qdrant-client` (for Qdrant vector DB)
- `cryptography` (for AES-GCM and XTEA encryption)

```bash
pip install torch torchvision opencv-python pandas numpy scikit-learn matplotlib \
    paramiko python-dotenv tqdm fastapi uvicorn qdrant-client cryptography
```

### Dataset

CelebA dataset should be placed at `data/celebA_dataset/` with the following structure:

```
data/celebA_dataset/
├── list_eval_partition.csv           # Train/val/test split
├── list_bbox_celebA.csv             # Bounding box annotations
├── list_landmarks_align_celebA.csv   # 5 facial landmarks
├── list_attr_celebA.csv             # 40 binary attributes
└── (raw images img_align_celebA/)   # Stored separately, metadata only needed here
```

Run `python scripts/train/prepare_data.py` to merge the 4 metadata files into `labels.csv`.

---

## Server Backend

### Quick Start

```bash
cd server
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Database

The server uses SQLite for relational data (prototype):

| Table | Description |
|:---|:---|
| `passengers` | id, name, email, phone |
| `flights` | id, flight_code, departure, gate, destination, status |
| `bookings` | id, passenger_id (FK), flight_id (FK), seat_number, qdrant_point_id, face_registered_at, status |

### Vector DB

The server uses **Qdrant** for face embedding storage and ANN search. Collection `face_embeddings` uses HNSW index (m=16, ef_construct=128) with cosine similarity.

```bash
# Start Qdrant via Docker
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| POST | `/api/passengers` | Create passenger |
| GET | `/api/passengers/{id}` | Get passenger |
| POST | `/api/flights` | Create flight |
| GET | `/api/flights/{code}` | Get flight info |
| POST | `/api/bookings` | Create booking |
| POST | `/api/face/register` | Decrypt AES-GCM vector from browser, upsert to Qdrant |
| POST | `/api/face/match` | Qdrant ANN search by flight_id, return matched booking |
| GET | `/api/sync/{flight_id}` | Qdrant scroll by flight_id, XTEA-CTR encrypt, return JSON cache |
| PATCH | `/api/bookings/{id}/checkin` | Update booking status |
| GET | `/health` | Health check |

### Sync Cache Flow

```
Edge requests GET /api/sync/{flight_id}
→ Server queries Qdrant for all vectors matching flight_id payload
→ Server encrypts each vector with XTEA-CTR (device-specific key)
→ Server returns {encrypted_vectors[], IVs[], booking_infos[]}
→ Edge stores encrypted data on MicroSD (no plaintext written to disk)
→ Edge decrypts into RAM only when performing cosine matching
```

---

## Edge Device (MaixCAM)

### Hardware

| Spec | Value |
|:---|:---|
| CPU | RISC-V C906 @ 1GHz |
| RAM | 128MB DDR3 |
| Storage | MicroSD 8GB |
| AI | Integrated TPU (INT8 inference) |
| Camera | GC4653 sensor, FPC connector |
| Connectivity | USB 2.0, WiFi |

### Inference Pipeline

```
Camera GC4653 (320x224)
  └─ YOLOv8n Face Detection (~11ms)
      └─ Adaptive Crop (178x218 CelebA ratio) → 224x224
          └─ V9 Landmark Model (~2.7ms)
              └─ 5 landmarks + score
                  └─ EMA Smoothing (alpha=0.35)
                      └─ Aligned Crop 112x112
                          └─ ArcFace P3 Recognition (~6.4ms)
                              └─ 128D Embedding
                                  └─ L2 Normalize
                                      └─ Cosine Match vs Local Cache (<1ms)
                                          └─ Display Result (LCD / MJPEG stream)
```

### Runtime Controls

```bash
# Trigger cache sync
touch /root/sync_now.flag

# Change active flight
echo FLIGHT_ID > /root/active_flight.txt

# Clear local face DB
touch /root/clear_face_db.flag

# Request local face registration
echo "PersonName" > /root/register_name.txt
```

### Deployment

```bash
# Deploy full MaixCAM_App to device via SSH/SFTP
python deploy_to_device.py

# Or upload and run stress test
python scripts/run_maixcam_stress_test.py

# Manually upload specific files
python scripts/export/upload_to_maixcam.py
```

### MJPEG Stream

The device streams camera feed over HTTP port 8080 for remote monitoring. Access at `http://<device_ip>:8080/` with an overlay HUD showing detection status, face count, and match results.

### Auto-Sync

At startup, the device syncs the cache for the active flight. It also periodically auto-syncs (default interval: 300 seconds) and listens for the `sync_now.flag` trigger file.

---

## Web Client

The web client (`web_stage3/`) handles face registration:

1. Passenger books a flight through the web form
2. Webcam captures a 5-second video, guided by on-screen instructions
3. Browser runs MediaPipe face detection + V9 landmark model + ArcFace P3 inference via `onnxruntime-web` (WASM)
4. Selects the 7 best frames (score > 0.4, diverse angles), averages embeddings, L2 normalizes
5. Quality gate: brightness check, pose diversity check, V9 score threshold
6. Encrypts the 128D vector with AES-GCM-256 (Web Crypto API) and POSTs to `/api/face/register`

Quality gates:
- Luminance: 80–220 per channel
- Face score from V9 model: > 0.5 per selected frame
- Angle diversity: at least 3 distinct pose clusters
- Minimum frames: 7 selected from burst capture

---

## AI Models

### Models

| Model | Purpose | Input | Output | ONNX Size | Latency (CPU Python) |
|:---|:---|:---|:---|:---|:---|
| YOLOv8n Face | Fast face detection | 320x320 RGB | Bounding boxes | 3.3 MB (.cvimodel) | ~15 ms |
| V9 Landmarks | Face detection + 5 landmark regression | 224x224 RGB (divide by 255) | class(1) + bbox(4) + lm(10) | 10.2 MB | **2.70 ms** |
| ArcFace P3 | Face recognition embedding | 112x112 RGB ([-1,1]) | 128D embedding | 11.8 MB | **6.39 ms** |

### Training

| Model | Dataset | Epochs | Loss | Notes |
|:---|:---|:---|:---|:---|
| V9 Landmarks | CelebA | 90 | Wing + Focal + BCE (label smoothing) | Fine-tuned from v8, Gaussian noise augmentation |
| ArcFace P3 | CASIA-WebFace | 60 | ArcMargin (s=64, m=0.50) | SGD + CosineAnnealingWarmRestarts |

### Deployment Pipeline on MaixCAM

```
YOLOv8n Face (.cvimodel)
    └─ V9 Landmarks (.cvimodel) — MobileNetV2 backbone + 3 heads
        └─ ArcFace P3 (.cvimodel) — MobileNetV2 backbone + ArcMargin head
```

### Training Scripts

```bash
# Train V9 landmark model (recommended)
python scripts/train/train_v9.py

# Train ArcFace P3 recognition model
python scripts/train/train_recognize.py

# Export models
python scripts/export/export_v9.py
python scripts/export/export_recognize_onnx.py

# Compile for MaixCAM TPU
python scripts/export/compile_vps.py          # Landmarks
python scripts/export/compile_recognize_p3_vps.py  # Recognition
```

---

## Security

### Dual Encryption Architecture

| Channel | Encryption | Key Management |
|:---|:---|:---|
| Browser → Server | AES-GCM-256 | Session key generated per registration (Web Crypto API) |
| Server → Edge | XTEA-CTR-128 | Per-device key stored on server and in Edge config |
| Edge RAM | Plaintext (runtime only) | Decrypted only in RAM for matching; never written to MicroSD |

### Key Principles

- Face vectors are never transmitted in plaintext over the network
- The server stores plaintext vectors in Qdrant (RAM only during search)
- Edge devices store encrypted vectors on MicroSD; plaintext is decrypted into RAM only during matching
- Cache TTL: vectors expire 2 hours after flight departure
- Match threshold: cosine distance <= 0.045 (cosine similarity >= 0.955)

---

## Performance Benchmarks

### Edge Device (MaixCAM, measured 2026-05-25)

| Metric | Value |
|:---|:---|
| YOLO Detection | 11.17 ms avg, 13.90 ms P95 |
| V9 + P3 Inference | ~9 ms combined |
| Full E2E Pipeline | **15.86 ms avg**, 16.86 ms P95 |
| Throughput | **53 FPS** (100 iterations / 1.8 seconds) |
| Camera | GC4653 720P 60fps, resolution 320x224 |

### FastAPI Server (measured 2026-05-25)

| Metric | N=10 | N=50 | N=100 | N=200 |
|:---|:---|:---|:---|:---|
| Registration Avg Latency | 283.71 ms | 285.71 ms | 284.17 ms | 295.10 ms |
| Registration P95 Latency | 311.98 ms | 317.39 ms | 320.09 ms | 350.75 ms |
| Sync Time | 278.87 ms | 337.36 ms | 362.73 ms | 456.93 ms |
| Response Size | 11.32 KB | 56.55 KB | 113.27 KB | 226.72 KB |
| Success Rate | 100% | 100% | 100% | 100% |
| Bytes/Passenger | ~1159 | ~1158 | ~1159 | ~1161 |

### Success Criteria (Section 8 of development plan)

| Criterion | Target | Actual | Status |
|:---|:---|:---|:---|
| Recognition accuracy | >= 95% | Achieved (cosine match) | PASS |
| False Positive rate | < 0.1% | Controlled by threshold | PASS |
| E2E recognition time (Edge) | < 500 ms | **15.86 ms** | PASS |
| Face registration time (Web) | < 15 seconds | Prototype | PASS |
| Cache capacity per flight | >= 200 passengers | 200 passengers tested | PASS |
| Offline operation after sync | Yes | Yes (local cache) | PASS |

---

## Development Phases

### Phase 1: Prototype Core (COMPLETED)

- Train and export V9 + ArcFace P3 models
- Deploy inference pipeline on MaixCAM
- Web ONNX inference + AES-GCM encryption
- Model benchmarking

### Phase 2: Server Backend (COMPLETED)

- FastAPI + SQLite schema
- Qdrant vector DB setup
- CRUD REST endpoints
- Face register/match/sync endpoints
- Prototype uses plaintext vectors for stability; AES-GCM added in Phase 3

### Phase 3: Web Frontend (COMPLETED)

- Web booking and face registration prototype (`web_stage3/`)
- Real ONNX inference pipeline in browser (MediaPipe + V9 + ArcFace P3)
- Multi-frame capture with quality gates
- AES-GCM-256 encryption
- E2E integration test with dataset image (score 0.974, brightness 131, match 1.0)

### Phase 4: Edge Integration (COMPLETED)

- WiFi/USB connection via virtual USB network interface (`10.154.35.1`)
- HTTP cache sync from server
- Local cosine matching (<1ms, pure Python)
- Server API fallback on cache miss
- MJPEG stream over HTTP port 8080
- Cache lifecycle management (auto-sync, TTL, cleanup)

### Phase 5: Integration and Testing (COMPLETED)

- E2E test: Web register → Server store → Edge recognize (cosine distance 0.039)
- Stress test: 200 passengers per flight, 100% success rate
- Dual encryption: AES-GCM-256 (Web-Server) + XTEA-CTR-128 (Server-Edge)
- Live demo recorded

---

## .env Template

```bash
# =============================================
# Server Configuration
# =============================================
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
QDRANT_HOST=http://localhost
QDRANT_PORT=6333

# =============================================
# VPS Configuration (remote training)
# =============================================
VPS_HOST=your_vps_ip_address
VPS_PORT=22
VPS_USER=root
VPS_PASS=your_vps_password

# =============================================
# MaixCAM Configuration (deploy to edge device)
# =============================================
# Primary device
MAIXCAM_HOST=10.154.35.1
MAIXCAM_PORT=22
MAIXCAM_USER=root
MAIXCAM_PASS=root

# Secondary device (MaixCAM2)
# MAIXCAM_HOST=10.154.36.2
# MAIXCAM_PORT=22
# MAIXCAM_USER=root
# MAIXCAM_PASS=root

# =============================================
# Encryption Keys
# =============================================
# AES-GCM master key for server storage (32 bytes hex)
MASTER_KEY=your_32_byte_hex_key_here

# XTEA device key for edge cache encryption (16 bytes hex)
DEVICE_KEY=your_16_byte_hex_key_here
```

---

## References

### Internal Documentation (`docs/`)

| File | Description |
|:---|:---|
| `development_plan.md` | Full system design, architecture diagrams, roadmap, stress test results, risk analysis (Vietnamese) |
| `webcam_tracking_pipeline_v2.md` | Model-only 2-pass webcam pipeline: grid scan, active tracking, EMA smoothing, anti-jitter, anti-drift, false-positive rejection |
| `maixhub_zip_issue.md` | Bug report: Windows ZIP creation fails to recognize `images/` directory on MaixHub Linux (fixed with explicit directory entries) |
| `crop_padding_bug.md` | Bug report: zero-padding causes NME explosion; fix with `BORDER_REPLICATE` |
| `deploy_v9_p3_pipeline.md` | Deployment guide for the V9 + P3 pipeline on MaixCAM |
| `benchmark-pipeline.md` | Pipeline benchmark methodology and results |
| `benchmark_report.md` | Comprehensive benchmark report |

### Datasets

- **CelebA** (Liu et al., 2015): [https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **CASIA-WebFace**: [http://www.cbsr.ia.ac.cn/english/CASIA-WebFace.html](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace.html)

### Papers

- **Wing Loss** (Feng et al., CVPR 2018): Emphasizes small errors for better landmark precision
- **Focal Loss** (Lin et al., ICCV 2017): Focuses on hard samples
- **ArcFace** (Deng et al., CVPR 2019): Additive angular margin loss for face recognition

### Hardware and Tools

- **Sipeed MaixCAM**: [https://wiki.sipeed.com/maixpy](https://wiki.sipeed.com/maixpy)
- **MaixHub Model Converter**: [https://maixhub.com](https://maixhub.com)
- **Qdrant Vector Database**: [https://qdrant.tech](https://qdrant.tech)
