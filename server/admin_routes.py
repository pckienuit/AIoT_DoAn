"""Admin API routes — flight management, booking management, and edge sync control."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import paramiko
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from server.auth import create_access_token
from server.database import get_connection, row_to_dict
from server.vector_service import delete_flight_embeddings, scroll_flight_embeddings


router = APIRouter(prefix="/api/admin", tags=["admin"])

# ---------------------------------------------------------------------------
# Config (mirrors sync_maixcam.py defaults)
# ---------------------------------------------------------------------------

DEVICE_HOST = os.getenv("MAIXCAM_HOST", "10.154.36.1")
DEVICE_USER = os.getenv("MAIXCAM_USER", "root")
DEVICE_PASS = os.getenv("MAIXCAM_PASS", "root")
DEVICE_PORT = int(os.getenv("MAIXCAM_PORT", "22"))

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@aiot.flight")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin1234")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_one(query: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(query, params).fetchone()
    if row:
        return row_to_dict(row)
    return None


def _fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [row_to_dict(r) for r in rows]


def _execute(query: str, params: tuple[Any, ...]) -> int:
    with get_connection() as conn:
        cur = conn.execute(query, params)
        conn.commit()
        return cur.rowcount


def _utc_now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _cleanup_flight_cache(client, flight_id: int) -> dict[str, Any]:
    """Delete /root/cache/flight_{id}.json from the edge device. Returns a result dict."""
    result = {"flight_id": flight_id, "device_reachable": False, "cache_deleted": False, "error": None}
    if client is None:
        try:
            client = _make_ssh_client()
        except Exception as exc:
            result["error"] = str(exc)
            return result
    try:
        sftp = client.open_sftp()
        try:
            sftp.remove(f"/root/cache/flight_{flight_id}.json")
            result["cache_deleted"] = True
        except FileNotFoundError:
            result["cache_deleted"] = False  # already gone, that's fine
        finally:
            sftp.close()

        # Check if this flight was the active flight. If so, clear it.
        try:
            rc, out = _ssh_run(client, "cat /root/active_flight.txt 2>/dev/null || echo ''")
            if rc == 0 and out.strip() == str(flight_id):
                _ssh_run(client, "rm -f /root/active_flight.txt")
        except Exception:
            pass

        result["device_reachable"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


# ---------------------------------------------------------------------------
# Admin Auth
# ---------------------------------------------------------------------------

class AdminLoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    email: str


@router.post("/login", response_model=TokenResponse)
def admin_login(payload: AdminLoginRequest) -> TokenResponse:
    if payload.email != ADMIN_EMAIL or payload.password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials",
        )
    token = create_access_token({"sub": "admin", "email": ADMIN_EMAIL, "role": "admin"})
    return TokenResponse(access_token=token, email=ADMIN_EMAIL)


# ---------------------------------------------------------------------------
# Dashboard Stats
# ---------------------------------------------------------------------------

@router.get("/stats")
def admin_stats() -> dict[str, Any]:
    today = datetime.now().date().isoformat()

    flights = _fetch_one("SELECT COUNT(*) as c FROM flights WHERE deleted_at IS NULL", ())
    total_flights = flights["c"] if flights else 0

    flights_today = _fetch_one(
        "SELECT COUNT(*) as c FROM flights WHERE flight_date = ? AND deleted_at IS NULL",
        (today,),
    )
    flights_today_count = flights_today["c"] if flights_today else 0

    bookings = _fetch_one("SELECT COUNT(*) as c FROM bookings", ())
    total_bookings = bookings["c"] if bookings else 0

    checked_in = _fetch_one(
        "SELECT COUNT(*) as c FROM bookings WHERE status = 'checked_in'", ()
    )
    checked_in_count = checked_in["c"] if checked_in else 0

    revenue = _fetch_one(
        "SELECT COALESCE(SUM(total_price), 0) as s FROM bookings WHERE payment_status = 'paid'",
        (),
    )
    total_revenue = float(revenue["s"]) if revenue else 0.0

    users = _fetch_one("SELECT COUNT(*) as c FROM users", ())
    total_users = users["c"] if users else 0

    face_registered = _fetch_one(
        "SELECT COUNT(*) as c FROM bookings WHERE face_registered = 1", ()
    )
    face_count = face_registered["c"] if face_registered else 0

    return {
        "total_flights": total_flights,
        "flights_today": flights_today_count,
        "total_bookings": total_bookings,
        "checked_in": checked_in_count,
        "total_revenue": total_revenue,
        "total_users": total_users,
        "face_registered": face_count,
        "checkin_rate": round(checked_in_count / total_bookings * 100, 1) if total_bookings > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Flights
# ---------------------------------------------------------------------------

@router.get("/flights")
def list_flights(
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    date: Annotated[str | None, Query(description="YYYY-MM-DD")] = None,
    search: Annotated[str | None, Query(description="flight number or route")] = None,
) -> list[dict[str, Any]]:
    q = """
        SELECT
            fl.id,
            fl.flight_number,
            fl.flight_date,
            fl.status,
            fl.available_seats,
            fl.price_multiplier,
            fl.deleted_at,
            s.departure_time,
            s.arrival_time,
            s.base_price,
            r.distance_km,
            a_o.code AS origin_code,
            a_o.city AS origin_city,
            a_d.code AS dest_code,
            a_d.city AS dest_city,
            p.plane_type,
            p.total_seats
        FROM flights fl
        JOIN schedules s ON s.id = fl.schedule_id
        JOIN routes r ON r.id = s.route_id
        JOIN airports a_o ON a_o.id = r.origin_id
        JOIN airports a_d ON a_d.id = r.destination_id
        JOIN planes p ON p.id = s.plane_id
        WHERE 1=1
    """
    params: list[Any] = []

    if status_filter:
        q += " AND fl.status = ?"
        params.append(status_filter)
    if date:
        q += " AND fl.flight_date = ?"
        params.append(date)
    if search:
        q += " AND (fl.flight_number LIKE ? OR a_o.code LIKE ? OR a_d.code LIKE ?)"
        like = f"%{search}%"
        params.extend([like, like, like])

    q += " ORDER BY fl.flight_date DESC, s.departure_time DESC"

    rows = _fetch_all(q, tuple(params))
    for row in rows:
        # Booking count
        count_row = _fetch_one(
            "SELECT COUNT(*) as c FROM bookings WHERE flight_id = ? AND status NOT IN ('cancelled','refunded')",
            (row["id"],),
        )
        row["booking_count"] = count_row["c"] if count_row else 0
        # Calculate price
        base = float(row.get("base_price") or 0)
        mult = float(row.get("price_multiplier") or 1.0)
        row["price_per_person"] = round(base * mult, 0)
        # Check if has face registrations
        face_row = _fetch_one(
            "SELECT COUNT(*) as c FROM bookings WHERE flight_id = ? AND face_registered = 1",
            (row["id"],),
        )
        row["face_registered_count"] = face_row["c"] if face_row else 0

    return rows


@router.get("/flights/{flight_id}")
def get_flight_detail(flight_id: int) -> dict[str, Any]:
    flight = _fetch_one(
        """
        SELECT
            fl.*,
            s.departure_time, s.arrival_time, s.base_price,
            r.distance_km,
            a_o.code AS origin_code, a_o.name AS origin_name, a_o.city AS origin_city,
            a_d.code AS dest_code, a_d.name AS dest_name, a_d.city AS dest_city,
            p.plane_type, p.total_seats, p.seat_layout
        FROM flights fl
        JOIN schedules s ON s.id = fl.schedule_id
        JOIN routes r ON r.id = s.route_id
        JOIN airports a_o ON a_o.id = r.origin_id
        JOIN airports a_d ON a_d.id = r.destination_id
        JOIN planes p ON p.id = s.plane_id
        WHERE fl.id = ?
        """,
        (flight_id,),
    )
    if not flight:
        raise HTTPException(status_code=404, detail="Flight not found")

    bookings = _fetch_all(
        """
        SELECT b.*, u.full_name AS user_name, u.email AS user_email
        FROM bookings b
        LEFT JOIN users u ON u.id = b.user_id
        WHERE b.flight_id = ?
        ORDER BY b.created_at DESC
        """,
        (flight_id,),
    )
    flight["bookings"] = bookings

    base = float(flight.get("base_price") or 0)
    mult = float(flight.get("price_multiplier") or 1.0)
    flight["price_per_person"] = round(base * mult, 0)

    return flight


class FlightStatusUpdate(BaseModel):
    status: str = Field(pattern="^(scheduled|boarding|departed|arrived|cancelled)$")


@router.patch("/flights/{flight_id}/status")
def update_flight_status(flight_id: int, payload: FlightStatusUpdate) -> dict[str, Any]:
    row = _fetch_one("SELECT * FROM flights WHERE id = ?", (flight_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Flight not found")

    new_status = payload.status
    now = _utc_now_str()

    if new_status == "cancelled":
        _execute(
            "UPDATE flights SET status = ?, deleted_at = ? WHERE id = ?",
            (new_status, now, flight_id),
        )
        _delete_flight_embeddings(flight_id)
    else:
        _execute("UPDATE flights SET status = ? WHERE id = ?", (new_status, flight_id))

    return _fetch_one("SELECT * FROM flights WHERE id = ?", (flight_id,)) or {}


@router.delete("/flights/{flight_id}")
def soft_delete_flight(flight_id: int) -> dict[str, Any]:
    row = _fetch_one("SELECT * FROM flights WHERE id = ? AND deleted_at IS NULL", (flight_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Active flight not found")

    now = _utc_now_str()
    _execute("UPDATE flights SET deleted_at = ? WHERE id = ?", (now, flight_id))
    try:
        _delete_flight_embeddings(flight_id)
    except Exception:
        pass

    return {"status": "deleted", "flight_id": flight_id}


def _delete_flight_embeddings(flight_id: int) -> None:
    """Delete all face embeddings for a flight (local wrapper)."""
    try:
        delete_flight_embeddings(flight_id)
    except Exception:
        pass


@router.patch("/flights/{flight_id}/restore")
def restore_flight(flight_id: int) -> dict[str, Any]:
    row = _fetch_one("SELECT * FROM flights WHERE id = ? AND deleted_at IS NOT NULL", (flight_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Deleted flight not found")

    _execute("UPDATE flights SET deleted_at = NULL WHERE id = ?", (flight_id,))
    return {"status": "restored", "flight_id": flight_id}


# ---------------------------------------------------------------------------
# Bookings
# ---------------------------------------------------------------------------

@router.get("/bookings")
def list_bookings(
    flight_id: Annotated[int | None, Query()] = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    payment_status: Annotated[str | None, Query()] = None,
    face_registered: Annotated[bool | None, Query()] = None,
) -> list[dict[str, Any]]:
    q = """
        SELECT
            b.*,
            fl.flight_number, fl.flight_date, fl.status AS flight_status,
            s.departure_time, s.arrival_time,
            a_o.code AS origin_code, a_o.city AS origin_city,
            a_d.code AS dest_code, a_d.city AS dest_city,
            u.full_name AS user_name, u.email AS user_email
        FROM bookings b
        JOIN flights fl ON fl.id = b.flight_id
        JOIN schedules s ON s.id = fl.schedule_id
        JOIN routes r ON r.id = s.route_id
        JOIN airports a_o ON a_o.id = r.origin_id
        JOIN airports a_d ON a_d.id = r.destination_id
        LEFT JOIN users u ON u.id = b.user_id
        WHERE 1=1
    """
    params: list[Any] = []

    if flight_id is not None:
        q += " AND b.flight_id = ?"
        params.append(flight_id)
    if status_filter:
        q += " AND b.status = ?"
        params.append(status_filter)
    if payment_status:
        q += " AND b.payment_status = ?"
        params.append(payment_status)
    if face_registered is not None:
        q += " AND b.face_registered = ?"
        params.append(1 if face_registered else 0)

    q += " ORDER BY b.created_at DESC"
    return _fetch_all(q, tuple(params))


@router.get("/bookings/{booking_id}")
def get_booking_detail(booking_id: int) -> dict[str, Any]:
    row = _fetch_one(
        """
        SELECT
            b.*,
            fl.flight_number, fl.flight_date, fl.status AS flight_status,
            s.departure_time, s.arrival_time,
            a_o.code AS origin_code, a_o.name AS origin_name, a_o.city AS origin_city,
            a_d.code AS dest_code, a_d.name AS dest_name, a_d.city AS dest_city,
            u.full_name AS user_name, u.email AS user_email
        FROM bookings b
        JOIN flights fl ON fl.id = b.flight_id
        JOIN schedules s ON s.id = fl.schedule_id
        JOIN routes r ON r.id = s.route_id
        JOIN airports a_o ON a_o.id = r.origin_id
        JOIN airports a_d ON a_d.id = r.destination_id
        LEFT JOIN users u ON u.id = b.user_id
        WHERE b.id = ?
        """,
        (booking_id,),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Booking not found")
    return row


class BookingStatusUpdate(BaseModel):
    status: str = Field(pattern="^(pending|confirmed|checked_in|cancelled|refunded)$")


@router.patch("/bookings/{booking_id}/status")
def update_booking_status(booking_id: int, payload: BookingStatusUpdate) -> dict[str, Any]:
    row = _fetch_one("SELECT * FROM bookings WHERE id = ?", (booking_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Booking not found")

    now = _utc_now_str()
    _execute(
        "UPDATE bookings SET status = ?, updated_at = ? WHERE id = ?",
        (payload.status, now, booking_id),
    )

    # If cancelling, restore seat
    if payload.status == "cancelled":
        _execute(
            "UPDATE flights SET available_seats = available_seats + 1 WHERE id = ?",
            (row["flight_id"],),
        )

    return _fetch_one(
        """
        SELECT b.*, fl.flight_number, fl.flight_date
        FROM bookings b
        JOIN flights fl ON fl.id = b.flight_id
        WHERE b.id = ?
        """,
        (booking_id,),
    ) or {}


@router.delete("/bookings/{booking_id}/face")
def delete_face_registration(booking_id: int) -> dict[str, Any]:
    row = _fetch_one("SELECT * FROM bookings WHERE id = ?", (booking_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Booking not found")

    flight_id = row["flight_id"]
    point_id = row.get("qdrant_point_id")

    if point_id:
        try:
            from server.vector_service import get_qdrant_client, COLLECTION_NAME
            client = get_qdrant_client()
            client.delete(
                collection_name=COLLECTION_NAME,
                points=[point_id],
            )
        except Exception:
            pass

    now = _utc_now_str()
    _execute(
        "UPDATE bookings SET qdrant_point_id = NULL, face_registered = 0, updated_at = ? WHERE id = ?",
        (now, booking_id),
    )

    # Check if this was the last face for this flight — if so, delete the edge cache file
    remaining = _fetch_one(
        "SELECT COUNT(*) as c FROM bookings WHERE flight_id = ? AND face_registered = 1",
        (flight_id,),
    )
    cleanup_result = None
    if not remaining or remaining["c"] == 0:
        cleanup_result = _cleanup_flight_cache(client=None, flight_id=flight_id)

    return {"status": "face_deleted", "booking_id": booking_id, "cache_cleanup": cleanup_result}


# ---------------------------------------------------------------------------
# Edge Sync
# ---------------------------------------------------------------------------

@router.get("/sync/registered-flights")
def get_registered_flights() -> list[dict[str, Any]]:
    """List flights that have at least one face-registered booking."""
    rows = _fetch_all(
        """
        SELECT DISTINCT
            fl.id,
            fl.flight_number,
            fl.flight_date,
            fl.status,
            a_o.code AS origin_code,
            a_d.code AS dest_code,
            s.departure_time,
            s.arrival_time,
            (SELECT COUNT(*) FROM bookings b WHERE b.flight_id = fl.id AND b.face_registered = 1) AS face_count
        FROM flights fl
        JOIN schedules s ON s.id = fl.schedule_id
        JOIN routes r ON r.id = s.route_id
        JOIN airports a_o ON a_o.id = r.origin_id
        JOIN airports a_d ON a_d.id = r.destination_id
        JOIN bookings b ON b.flight_id = fl.id
        WHERE b.face_registered = 1 AND fl.deleted_at IS NULL
        ORDER BY fl.flight_date DESC, s.departure_time DESC
        """
    )
    return rows


class SyncTriggerRequest(BaseModel):
    flight_id: int | None = None
    mode: str = Field(default="flag", pattern="^(flag|direct)$")
    sync_all: bool = False


def _make_ssh_client() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        DEVICE_HOST,
        port=DEVICE_PORT,
        username=DEVICE_USER,
        password=DEVICE_PASS,
        timeout=15,
    )
    return client


def _ssh_run(client: paramiko.SSHClient, command: str) -> tuple[int, str]:
    _, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    return rc, (out + err).strip()


@router.post("/sync/trigger")
def trigger_sync(payload: SyncTriggerRequest) -> dict[str, Any]:
    try:
        client = _make_ssh_client()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to MaixCAM device at {DEVICE_HOST}: {str(exc)}",
        ) from exc

    try:
        outputs = []

        if payload.sync_all:
            server_url = os.getenv("SERVER_URL", "http://10.154.36.100:8000")
            # Fetch all embeddings server-side and upload directly
            registered_ids = _fetch_all(
                """
                SELECT DISTINCT flight_id FROM bookings
                WHERE face_registered = 1 AND flight_id IS NOT NULL
                """
            )
            all_items = []
            combined_status = "scheduled"
            for row in registered_ids:
                fid = row["flight_id"]
                try:
                    records = scroll_flight_embeddings(fid)
                    from server.face_routes import build_face_payload
                    from server.crypto_service import encrypt_xtea_vector
                    for rec in records:
                        pt_id = str(rec.id)
                        # Skip if booking was cancelled/refunded or face deleted
                        bk = _fetch_one(
                            """
                            SELECT
                                b.*,
                                fl.flight_number, fl.flight_date, fl.status AS flight_status,
                                s.departure_time, s.arrival_time,
                                a_o.city AS origin_city,
                                a_d.city AS dest_city
                            FROM bookings b
                            JOIN flights fl ON fl.id = b.flight_id
                            JOIN schedules s ON s.id = fl.schedule_id
                            JOIN routes r ON r.id = s.route_id
                            JOIN airports a_o ON a_o.id = r.origin_id
                            JOIN airports a_d ON a_d.id = r.destination_id
                            WHERE b.qdrant_point_id = ?
                              AND b.status NOT IN ('cancelled', 'refunded')
                              AND b.face_registered = 1
                            """,
                            (pt_id,),
                        )
                        if not bk:
                            continue
                        ct, iv = encrypt_xtea_vector(rec.vector)
                        face_payload = build_face_payload(bk)
                        all_items.append({
                            "point_id": pt_id,
                            "ciphertext": ct,
                            "iv": iv,
                            "payload": face_payload,
                        })
                    # Check flight status
                    fr = _fetch_one("SELECT status FROM flights WHERE id = ?", (fid,))
                    if fr and fr["status"] in ("departed", "arrived", "cancelled"):
                        combined_status = fr["status"]
                except Exception:
                    pass

            cache_entry = {
                "flight_id": 999999,
                "synced_at": datetime.utcnow().timestamp(),
                "count": len(all_items),
                "items": all_items,
                "test_all_faces": True,
                "status": combined_status,
                "recognition_allowed": combined_status not in ("departed", "arrived", "cancelled"),
            }

            # Upload via SFTP
            sftp = client.open_sftp()
            try:
                try:
                    sftp.mkdir("/root/cache")
                except OSError:
                    pass
                with sftp.file("/root/cache/flight_999999.json", "w") as remote_file:
                    remote_file.write(json.dumps(cache_entry))
            finally:
                sftp.close()
            outputs.append(f"Uploaded {len(all_items)} embeddings to /root/cache/flight_999999.json")

            # Set active flight
            rc2, out2 = _ssh_run(client, "echo 999999 > /root/active_flight.txt")
            if rc2 != 0:
                outputs.append(f"Warning: could not set active flight: {out2}")
            else:
                outputs.append("Active flight set to 999999")

            return {
                "success": True,
                "flight_id": 999999,
                "mode": "all",
                "sync_all": True,
                "output": outputs,
                "embedding_count": len(all_items),
            }

        elif payload.flight_id is not None:
            fid = int(payload.flight_id)
            flight_row = _fetch_one(
                """
                SELECT fl.status, s.departure_time, s.arrival_time
                FROM flights fl JOIN schedules s ON s.id = fl.schedule_id
                WHERE fl.id = ? AND fl.deleted_at IS NULL
                """,
                (fid,),
            )
            if not flight_row:
                raise HTTPException(status_code=404, detail="Flight not found")

            records = scroll_flight_embeddings(fid)
            from server.face_routes import build_face_payload
            from server.crypto_service import encrypt_xtea_vector
            items = []
            for rec in records:
                pt_id = str(rec.id)
                # Skip cancelled/refunded bookings or ones with deleted face registrations
                bk = _fetch_one(
                    """
                    SELECT
                        b.*,
                        fl.flight_number, fl.flight_date, fl.status AS flight_status,
                        s.departure_time, s.arrival_time,
                        a_o.city AS origin_city,
                        a_d.city AS dest_city
                    FROM bookings b
                    JOIN flights fl ON fl.id = b.flight_id
                    JOIN schedules s ON s.id = fl.schedule_id
                    JOIN routes r ON r.id = s.route_id
                    JOIN airports a_o ON a_o.id = r.origin_id
                    JOIN airports a_d ON a_d.id = r.destination_id
                    WHERE b.qdrant_point_id = ?
                      AND b.status NOT IN ('cancelled', 'refunded')
                      AND b.face_registered = 1
                    """,
                    (pt_id,),
                )
                if not bk:
                    continue
                ct, iv = encrypt_xtea_vector(rec.vector)
                face_payload = build_face_payload(bk)
                items.append({
                    "point_id": pt_id,
                    "ciphertext": ct,
                    "iv": iv,
                    "payload": face_payload,
                })

            # If no embeddings left, delete the cache file from edge instead of uploading empty data
            if not items:
                _cleanup_flight_cache(client, fid)
                return {
                    "success": True,
                    "flight_id": fid,
                    "mode": payload.mode,
                    "output": ["No embeddings found — edge cache file removed"],
                    "embedding_count": 0,
                    "cache_cleared": True,
                }

            cache_entry = {
                "flight_id": fid,
                "synced_at": datetime.utcnow().timestamp(),
                "count": len(items),
                "items": items,
                "status": flight_row.get("status", "scheduled"),
                "recognition_allowed": flight_row.get("status", "scheduled") not in ("departed", "arrived", "cancelled"),
            }

            sftp = client.open_sftp()
            try:
                try:
                    sftp.mkdir("/root/cache")
                except OSError:
                    pass
                with sftp.file(f"/root/cache/flight_{fid}.json", "w") as remote_file:
                    remote_file.write(json.dumps(cache_entry))
            finally:
                sftp.close()

            if payload.mode == "direct":
                _ssh_run(client, f"echo {fid} > /root/active_flight.txt")
                rc, out = _ssh_run(
                    client,
                    f"cd /root && python -c \"from sync_cache import CacheManager; from config import load_config; cfg = load_config(); CacheManager(cfg).sync({fid})\" 2>&1 || true"
                )
                outputs.append(f"Direct sync output: {out}")
            else:
                # Set active flight and create sync flag
                _ssh_run(client, f"echo {fid} > /root/active_flight.txt")
                rc, out = _ssh_run(client, "touch /root/sync_now.flag && ls /root/sync_now.flag")
                outputs.append(f"Sync flag created. Active flight: {fid}")

            return {
                "success": True,
                "flight_id": fid,
                "mode": payload.mode,
                "output": outputs,
                "embedding_count": len(items),
            }
        else:
            raise HTTPException(status_code=400, detail="Either flight_id or sync_all must be provided")

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        client.close()


@router.get("/sync/status")
def get_sync_status() -> dict[str, Any]:
    try:
        client = _make_ssh_client()
    except Exception as exc:
        return {
            "connected": False,
            "host": DEVICE_HOST,
            "error": str(exc),
            "process": None,
            "cache_files": [],
            "active_flight": None,
        }

    try:
        _, proc_out = _ssh_run(client, "ps aux | grep main.py | grep -v grep")
        _, cache_out = _ssh_run(client, "ls -lh /root/cache/flight_*.json 2>/dev/null || true")
        _, active_out = _ssh_run(
            client, "cat /root/active_flight.txt 2>/dev/null || echo ''"
        )
        _, sync_out = _ssh_run(
            client,
            "tail -30 /root/main.log 2>/dev/null | grep -E 'sync|Sync|edge|Edge' | tail -10 || true",
        )

        cache_files = [f.strip() for f in cache_out.strip().split("\n") if f.strip()]

        return {
            "connected": True,
            "host": DEVICE_HOST,
            "process": proc_out.strip() or None,
            "cache_files": cache_files,
            "active_flight": active_out.strip() or None,
            "recent_sync_log": sync_out.strip() or None,
        }
    except Exception as exc:
        return {"connected": False, "host": DEVICE_HOST, "error": str(exc)}
    finally:
        client.close()


@router.delete("/sync/cache/{flight_id}")
def delete_flight_cache(flight_id: str) -> dict[str, Any]:
    """Delete cache file(s) from the edge device.

    Path /sync/cache/all  → delete all flight cache files.
    Path /sync/cache/{N}  → delete cache for flight ID N only.
    """
    if flight_id == "all":
        return _delete_all_caches_impl()

    try:
        fid = int(flight_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid flight ID: {flight_id}")

    result = _cleanup_flight_cache(client=None, flight_id=fid)
    return result


def _delete_all_caches_impl() -> dict[str, Any]:
    """Shared implementation for deleting all caches (used by both the route and face-deletion)."""
    try:
        client = _make_ssh_client()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to MaixCAM device: {str(exc)}",
        ) from exc

    try:
        _, ls_out = _ssh_run(client, "ls -1 /root/cache/flight_*.json 2>/dev/null || true")
        existing_files = [f.strip() for f in ls_out.split("\n") if f.strip()]

        if not existing_files:
            return {"success": True, "deleted": [], "deleted_count": 0,
                    "device_reachable": True, "message": "No cache files found"}

        deleted, errors = [], []
        for path in existing_files:
            fname = path.rsplit("/", 1)[-1]
            rc, out = _ssh_run(client, f"rm -f /root/cache/{fname}")
            (deleted if rc == 0 else errors).append(
                fname if rc == 0 else {"file": fname, "error": out}
            )

        _ssh_run(client, "rm -f /root/active_flight.txt")

        return {
            "success": len(errors) == 0,
            "deleted": deleted,
            "deleted_count": len(deleted),
            "errors": errors,
            "device_reachable": True,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        client.close()


# NOTE: The /sync/cache/all static route is removed — the "all" case is now
# handled inside the parameterized /sync/cache/{flight_id} route above.
# This avoids FastAPI's route-matching ambiguity between static and param routes.


# ---------------------------------------------------------------------------
# Database Management
# ---------------------------------------------------------------------------

@router.post("/db/clear")
def clear_database_endpoint() -> dict[str, Any]:
    """Clear all data from the SQLite prototype database."""
    from server.database import USE_MYSQL
    if USE_MYSQL:
        raise HTTPException(
            status_code=400,
            detail="This operation is only supported for SQLite (prototype) DB",
        )
    try:
        from server.seed import clear_database
        counts = clear_database()
        return {"success": True, "cleared": counts}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/db/seed")
def seed_database_endpoint() -> dict[str, Any]:
    """Clear and re-seed the SQLite prototype database with fresh demo data."""
    from server.database import USE_MYSQL
    if USE_MYSQL:
        raise HTTPException(
            status_code=400,
            detail="This operation is only supported for SQLite (prototype) DB",
        )
    try:
        from server.seed import seed_database
        result = seed_database()
        return {"success": True, **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

@router.get("/health")
def admin_health() -> dict[str, Any]:
    try:
        from server.database import iter_tables
        tables = list(iter_tables())
        db_ok = True
    except Exception as exc:
        tables = []
        db_ok = False

    try:
        from server.vector_service import get_vector_status
        vector = get_vector_status()
        vector_ok = True
    except Exception as exc:
        vector = {"error": str(exc)}
        vector_ok = False

    return {
        "db": {"ok": db_ok, "tables": tables},
        "vector": {**vector, "ok": vector_ok},
    }


class RunFlightStatusUpdate(BaseModel):
    pass


@router.post("/flight-status/update")
def run_flight_status_update(_payload: RunFlightStatusUpdate = None) -> dict[str, Any]:
    try:
        from server.update_flight_status import (
            update_arrived_flights,
            update_departed_flights,
        )

        departed = update_departed_flights()
        arrived = update_arrived_flights()

        return {
            "success": True,
            "departed": departed,
            "arrived": arrived,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
