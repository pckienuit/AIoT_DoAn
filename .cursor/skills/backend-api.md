---
description: "FastAPI backend patterns, REST API design, and Python best practices for the AIoT Face Check-in server. Use when building or modifying server/ API endpoints."
alwaysApply: false
---

# Backend / API Patterns Skill

> This skill references the full API patterns from `.agent/skills/api-patterns/`. Read those files for detailed guidance.

---

## Quick Reference

### Must Read
- `@.agent/skills/api-patterns/rest.md` — REST API conventions
- `@.agent/skills/api-patterns/response.md` — Response formatting
- `@.agent/skills/api-patterns/security-testing.md` — API security testing

### Optional
- `@.agent/skills/api-patterns/auth.md` — Authentication patterns
- `@.agent/skills/api-patterns/rate-limiting.md` — Rate limiting implementation

---

## AIoT Server: Key Patterns

### Request Validation (Pydantic)
```python
from pydantic import BaseModel, Field

class FaceRegisterRequest(BaseModel):
    encrypted_vector: str = Field(..., description="Base64-encoded AES-GCM-256 ciphertext")
    booking_id: str = Field(..., min_length=6, max_length=20)

class FaceMatchRequest(BaseModel):
    encrypted_vector: str = Field(..., description="Base64-encoded AES-GCM-256 ciphertext")
    threshold: float = Field(0.4, ge=0.0, le=1.0)
```

### Async Route Handler
```python
@router.post("/api/face/register")
async def register_face(req: FaceRegisterRequest) -> FaceRegisterResponse:
    # Always use async for I/O (Qdrant, SQLite)
    ...
```

### Error Responses
```python
from fastapi import HTTPException

raise HTTPException(
    status_code=status.HTTP_400_BAD_REQUEST,
    detail={"code": "INVALID_VECTOR", "message": "Failed to decrypt face vector"}
)
```

---

## File Structure (server/)

```
server/
├── main.py              # FastAPI app, CORS, routes
├── routes/
│   ├── __init__.py
│   ├── face.py          # /api/face/* endpoints
│   ├── flights.py       # /api/flights/* endpoints
│   └── sync.py          # /api/sync/* endpoints
├── services/
│   ├── qdrant.py        # Qdrant operations
│   ├── encryption.py    # AES-GCM, XTEA-CTR
│   └── booking.py       # Booking business logic
├── models/
│   ├── schemas.py       # Pydantic request/response models
│   └── db.py            # SQLAlchemy models
└── db/
    └── init.py          # SQLite init
```

---

## Security Rules

1. **Never log embeddings** — `logging.info(f"Received vector from {booking_id}")` is OK; `logging.info(f"Vector: {embedding}")` is NOT
2. **Rate limit public endpoints** — use `@slowapi` or custom middleware
3. **Validate Pydantic models** — never trust raw request data
4. **CORS** — restrict to known web client origins only

---

## Review Checklist

- [ ] Pydantic models for all request/response
- [ ] Async/await for all I/O
- [ ] No raw embeddings in logs
- [ ] Rate limiting applied
- [ ] Structured error responses (code + message)
- [ ] API docs accurate (FastAPI auto-generates from Pydantic)
