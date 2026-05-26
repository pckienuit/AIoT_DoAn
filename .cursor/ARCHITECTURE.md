---
description: "AIoT Face Check-in project architecture and component overview for Cursor Agent. Use when exploring codebase structure, understanding data flows, or planning cross-component changes."
alwaysApply: false
---

# AIoT Face Check-in — Architecture Overview

> Adapted from `.agent/ARCHITECTURE.md` for Cursor IDE.

---

## System Overview

The Face Check-in System is a **three-tier AIoT architecture** for airport check-in kiosks:

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
│      ├─ POST /api/face/register   (decrypt → Qdrant upsert)        │
│      ├─ POST /api/face/match       (Qdrant ANN search)             │
│      └─ GET  /api/sync/{flight_id} (scroll → XTEA-CTR encrypt)    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTPS / WiFi
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  MaixCAM Edge Device (RISC-V)                                      │
│  YOLO → V9 Landmarks → ArcFace P3 → XTEA-CTR local cache           │
│  └─ Real-time face detection + recognition at edge                │
└────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
AIoT_DoAn/
├── README.md                    # System overview
├── server/                     # FastAPI backend
│   ├── main.py                 # FastAPI app entry
│   ├── routes/                 # API route handlers
│   ├── models/                # SQLAlchemy / Pydantic models
│   ├── services/               # Business logic
│   └── db/                     # SQLite setup
├── web/                        # Web client (registration UI)
│   ├── index.html              # Main page
│   ├── register.js             # Face registration logic
│   ├── styles/                 # CSS
│   └── lib/                    # MediaPipe, ArcFace libs
├── edge/                       # MaixCAM edge device code
│   ├── main.py                 # Edge entry point
│   ├── detection.py            # YOLO face detection
│   ├── recognition.py          # ArcFace embedding
│   └── cache.py                # XTEA-CTR encrypted cache
├── docs/                       # Documentation
├── scripts/                    # Utility scripts
└── .cursor/                    # Cursor rules & configuration
    ├── rules/                  # Project rules
    ├── agents/                # Specialist agent definitions
    ├── skills/                 # Skill modules
    └── workflows/             # Workflow procedures
```

---

## Technology Stack

### Web Client
- **Face Detection:** MediaPipe (JavaScript)
- **Landmarks:** MediaPipe Face Mesh (468 3D landmarks, V9 model)
- **Embedding:** ArcFace P3 (128D, TensorFlow.js or ONNX.js)
- **Encryption:** AES-GCM-256 (Web Crypto API)

### FastAPI Server
- **Framework:** Python FastAPI
- **Database:** SQLite (passengers, flights, bookings metadata)
- **Vector DB:** Qdrant (face embeddings, 128D, HNSW index)
- **Encryption:** AES-GCM-256 decryption, XTEA-CTR for edge sync

### MaixCAM Edge
- **Device:** Sipeed MaixCAM (RISC-V, ~400MHz)
- **Detection:** YOLO face detection model
- **Landmarks:** V9 landmark model
- **Embedding:** ArcFace P3
- **Cache:** XTEA-CTR encrypted local cache

---

## Data Models

### SQLite (FastAPI Server)
- `passengers`: id, name, passport_number, nationality
- `flights`: id, flight_number, gate, departure_time, destination
- `bookings`: id, passenger_id, flight_id, seat, booking_ref, face_registered

### Qdrant Collection
- `face_embeddings`: id (booking_id), vector (128D ArcFace P3), payload (booking_ref, passenger_name)

---

## Security Architecture

1. **Web → Server:** HTTPS + AES-GCM-256 encrypted face vectors (server never sees raw face)
2. **Server → Edge:** HTTPS + XTEA-CTR encrypted batch sync
3. **Edge local:** XTEA-CTR encrypted face cache (no plaintext embeddings stored)

---

## Key API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/face/register` | Register encrypted face vector |
| POST | `/api/face/match` | Match a face vector against DB |
| GET | `/api/flights/{flight_id}` | Get flight info |
| POST | `/api/sync/{flight_id}` | Sync encrypted embeddings to edge |
| GET | `/api/health` | Health check |

---

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Done | Web client face registration (MediaPipe + ArcFace P3 + AES-GCM) |
| 2 | ✅ Done | FastAPI server (SQLite + Qdrant) |
| 3 | 🔄 In Progress | MaixCAM edge integration |
| 4 | 📋 Future | Performance optimization |
| 5 | 📋 Future | Security audit |

---

## Cross-Component Dependencies

| Change In | May Affect |
|-----------|-----------|
| `server/routes/` | Web client API calls, MaixCAM sync |
| `server/models/` | Qdrant schema, SQLite migrations |
| `web/register.js` | API payload format, encryption format |
| `edge/main.py` | Server sync protocol, cache format |
| `scripts/` | Build, test, deployment tooling |

---

## Performance Benchmarks

| Component | Metric | Target |
|-----------|--------|--------|
| Web registration | Face detection latency | < 200ms |
| Web registration | ArcFace embedding | < 500ms |
| Server match | Qdrant ANN search (top-1) | < 50ms |
| Edge recognition | Total pipeline | < 1s |
| Edge cache sync | XTEA-CTR batch (1000 faces) | < 5s |
