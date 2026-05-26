from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from server.database import get_connection
from server.routes import get_booking
from server.vector_service import (
    search_face_embedding,
    scroll_flight_embeddings,
    upsert_face_embedding,
    validate_embedding,
)
from server.crypto_service import (
    decrypt_aes_gcm_vector,
    encrypt_xtea_vector,
    decrypt_xtea_vector,
)

router = APIRouter(prefix="/api", tags=["face"])
MATCH_DISTANCE_THRESHOLD = 0.045


class FaceRegisterRequest(BaseModel):
    booking_id: int
    ciphertext: str | None = None
    iv: str | None = None
    embedding: list[float] | None = None


class FaceMatchRequest(BaseModel):
    flight_id: int
    ciphertext: str | None = None
    iv: str | None = None
    embedding: list[float] | None = None
    threshold: float = MATCH_DISTANCE_THRESHOLD


def build_face_payload(booking: dict[str, Any]) -> dict[str, Any]:
    return {
        "booking_id": booking["id"],
        "booking_code": booking["booking_code"],
        "passenger_name": booking["passenger_name"],
        "passenger_email": booking.get("passenger_email"),
        "flight_id": booking["flight_id"],
        "flight_number": booking["flight_number"],
        "flight_date": booking["flight_date"],
        "origin_city": booking["origin_city"],
        "dest_city": booking["dest_city"],
        "seat_number": booking.get("seat_number"),
        "departure_time": booking.get("departure_time"),
        "arrival_time": booking.get("arrival_time"),
        "flight_status": booking["flight_status"],
    }


@router.post("/face/register", status_code=201)
def register_face(payload: FaceRegisterRequest) -> dict[str, Any]:
    # Determine the vector
    if payload.ciphertext is not None and payload.iv is not None:
        try:
            vector = decrypt_aes_gcm_vector(payload.ciphertext, payload.iv)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"AES-GCM decryption failed: {str(exc)}") from exc
    elif payload.embedding is not None:
        try:
            vector = validate_embedding(payload.embedding)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        raise HTTPException(
            status_code=422,
            detail="Either encrypted vector (ciphertext + iv) or plaintext embedding must be provided",
        )

    booking = get_booking(payload.booking_id)
    point_id = str(uuid4())
    face_payload = build_face_payload(booking)
    upsert_face_embedding(point_id, vector, face_payload)

    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE bookings
            SET qdrant_point_id = ?, face_registered_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (point_id, now, now, payload.booking_id),
        )

    updated_booking = get_booking(payload.booking_id)
    return {"status": "registered", "point_id": point_id, "booking": updated_booking}


@router.post("/face/match")
def match_face(payload: FaceMatchRequest) -> dict[str, Any]:
    # Determine the vector
    if payload.ciphertext is not None and payload.iv is not None:
        try:
            vector = decrypt_xtea_vector(payload.ciphertext, payload.iv)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"XTEA-CTR decryption failed: {str(exc)}") from exc
    elif payload.embedding is not None:
        vector = payload.embedding
    else:
        raise HTTPException(
            status_code=422,
            detail="Either encrypted vector (ciphertext + iv) or plaintext embedding must be provided",
        )

    try:
        results = search_face_embedding(vector, payload.flight_id, limit=1)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not results:
        return {"matched": False, "distance": None, "score": None, "booking": None}

    hit = results[0]
    score = float(hit.score)
    distance = 1.0 - score
    matched = distance <= payload.threshold
    booking = get_booking(int(hit.payload["booking_id"])) if matched else None
    return {
        "matched": matched,
        "distance": distance,
        "score": score,
        "threshold": payload.threshold,
        "point_id": str(hit.id),
        "booking": booking,
    }


@router.get("/sync/{flight_id}")
def sync_flight_cache(flight_id: int) -> dict[str, Any]:
    records = scroll_flight_embeddings(flight_id)
    items = []
    for record in records:
        try:
            ciphertext, iv = encrypt_xtea_vector(record.vector)
            items.append(
                {
                    "point_id": str(record.id),
                    "ciphertext": ciphertext,
                    "iv": iv,
                    "payload": record.payload,
                }
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to encrypt sync data: {str(exc)}") from exc
    return {"flight_id": flight_id, "count": len(items), "items": items}
