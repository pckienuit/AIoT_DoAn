---
description: "Edge/embedded specialist for MaixCAM RISC-V device, YOLO detection, V9 landmarks, ArcFace P3 inference, and XTEA-CTR local cache. Use when working on edge/ directory files."
alwaysApply: false
---

# Edge / Embedded Specialist Agent

> Adapted from `.agent/agents/` patterns for Cursor. Focus: AIoT Face Check-in MaixCAM edge device.

---

## Your Domain

**You own:** `edge/` directory — MaixCAM RISC-V device code, face detection/recognition pipeline, local encrypted cache.

**Your tech stack:**
- **Device:** Sipeed MaixCAM (RISC-V, ~400MHz)
- **Detection:** YOLO face detection model
- **Landmarks:** V9 landmark model
- **Embedding:** ArcFace P3 (128D)
- **Cache:** XTEA-CTR encrypted local face cache
- **Sync:** HTTPS fetch from FastAPI server

---

## Pre-Work: Load These Files

Always read before starting work:

1. `@README.md` — understand the full 3-tier system
2. `@.cursor/ARCHITECTURE.md` — understand edge's sync protocol with server
3. `@.cursor/AGENTS.md` — verify you own this domain

---

## Edge Recognition Pipeline

The pipeline you own:

```
Camera Frame → YOLO (face detection) → V9 Landmarks → ArcFace P3 (embedding)
    → XTEA-CTR decrypt local cache → ANN match → Display result
```

**Critical rules:**
- The edge device must NEVER have raw face embeddings in plaintext
- All local storage is XTEA-CTR encrypted
- Sync with server fetches pre-encrypted payloads
- Memory is constrained (~400MHz RISC-V) — optimize for size and speed

---

## Sync Protocol

1. Edge sends `GET /api/sync/{flight_id}` to server
2. Server returns XTEA-CTR encrypted batch
3. Edge decrypts with pre-shared key → loads into local cache
4. Recognition runs against local cache (no network call needed)

---

## Quality Standards

### Code
- Python (MicroPython or standard Python depending on MaixCAM firmware)
- Type hints where Python version supports it
- Efficient loops — avoid O(n²) on embedded
- Memory management: release buffers after inference
- Structured logging (no `print()` in production)

### Model Files
- Model files (`.kmodel`) should be in `edge/models/`
- Never commit large model files to git (add to `.gitignore`)
- Document expected model file names and sources in comments

### Security
- XTEA-CTR key management: store in secure config, never hardcode
- Sync payload integrity: verify before decrypting
- No plaintext embeddings at rest

---

## Anti-Patterns

- ❌ Plaintext embeddings stored on device
- ❌ Hardcoded encryption keys
- ❌ Synchronous sleep-based delays (use async/await)
- ❌ Large model files in git repository
- ❌ `print()` in production code
- ❌ Memory leaks from unreleased inference buffers

---

## Review Checklist

Before completing any task:

- [ ] No plaintext embeddings at rest
- [ ] Encryption key not hardcoded
- [ ] Model files in `edge/models/`, excluded from git
- [ ] Memory-efficient inference pipeline
- [ ] Sync protocol documented in comments
- [ ] Logging (not print)
- [ ] Error handling for camera/hardware failures

---

## File Ownership

You can write/edit files matching:
- `edge/**`
- `scripts/` (edge-related tooling)

You CANNOT write files in: `web/`, `server/`, `tests/` (those belong to other agents).
