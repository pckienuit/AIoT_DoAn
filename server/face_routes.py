import math
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.database import get_connection
from server.routes import _fetch_one, get_booking
from server.vector_service import (
    search_face_embedding,
    scroll_flight_embeddings,
    upsert_face_embedding,
    validate_embedding,
    delete_face_embedding,
)
from server.crypto_service import (
    decrypt_aes_gcm_vector,
    encrypt_xtea_vector,
    decrypt_xtea_vector,
)

router = APIRouter(prefix="/api", tags=["face"])
MATCH_DISTANCE_THRESHOLD = 0.020


def clamp_match_threshold(value: float) -> float:
    threshold = float(value)
    if not math.isfinite(threshold):
        raise ValueError("Match threshold must be finite")
    return max(0.0, min(threshold, MATCH_DISTANCE_THRESHOLD))


class FaceRegisterRequest(BaseModel):
    booking_id: int
    ciphertext: str | None = None
    iv: str | None = None
    embedding: list[float] | None = None


class FaceMatchRequest(BaseModel):
    flight_id: int | None = None
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
        vector = payload.embedding
    else:
        raise HTTPException(
            status_code=422,
            detail="Either encrypted vector (ciphertext + iv) or plaintext embedding must be provided",
        )

    try:
        vector = validate_embedding(vector)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    booking = get_booking(payload.booking_id)
    
    # Delete old face embedding from Qdrant if exists to prevent orphans / duplicates
    old_point_id = booking.get("qdrant_point_id")
    if old_point_id:
        try:
            delete_face_embedding(old_point_id)
        except Exception as exc:
            # Log error but don't block registration
            print(f"[face_register] Warning: failed to delete old point {old_point_id}: {exc}")

    point_id = str(uuid4())
    face_payload = build_face_payload(booking)
    upsert_face_embedding(point_id, vector, face_payload)

    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE bookings
            SET qdrant_point_id = ?, face_registered = 1, face_registered_at = ?, updated_at = ?
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
        threshold = clamp_match_threshold(payload.threshold)
        results = search_face_embedding(vector, payload.flight_id, limit=1)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not results:
        return {"matched": False, "distance": None, "score": None, "booking": None}

    hit = results[0]
    score = float(hit.score)
    distance = 1.0 - score
    matched = distance <= threshold
    booking = get_booking(int(hit.payload["booking_id"])) if matched else None
    return {
        "matched": matched,
        "distance": distance,
        "score": score,
        "threshold": threshold,
        "requested_threshold": payload.threshold,
        "point_id": str(hit.id),
        "booking": booking,
    }


@router.get("/sync/{flight_id}")
def sync_flight_cache(flight_id: int) -> dict[str, Any]:
    # Get flight info for status check on edge
    flight_row = _fetch_one(
        """
        SELECT fl.status, s.departure_time, s.arrival_time
        FROM flights fl
        JOIN schedules s ON s.id = fl.schedule_id
        WHERE fl.id = ? AND fl.deleted_at IS NULL
        """,
        (flight_id,),
    )
    if not flight_row:
        raise HTTPException(status_code=404, detail="Flight not found")

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

    return {
        "flight_id": flight_id,
        "status": flight_row.get("status"),
        "departure_time": flight_row.get("departure_time"),
        "arrival_time": flight_row.get("arrival_time"),
        "count": len(items),
        "items": items,
    }
