"""Core API routes: auth, airports, flights, bookings, payments."""
from __future__ import annotations

import secrets
import uuid
from datetime import date, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from server.auth import create_access_token, get_current_user, get_optional_user, hash_password, verify_password
from server.database import USE_MYSQL, get_connection, row_to_dict


router = APIRouter(prefix="/api", tags=["core"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class UserRegister(BaseModel):
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=6, max_length=128)
    full_name: str = Field(min_length=1, max_length=255)
    phone: str | None = None
    date_of_birth: str | None = None  # YYYY-MM-DD
    id_card: str | None = None


class UserLogin(BaseModel):
    email: str
    password: str


class UserUpdate(BaseModel):
    full_name: str | None = None
    phone: str | None = None
    date_of_birth: str | None = None
    id_card: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict[str, Any]


class BookingCreate(BaseModel):
    flight_id: int
    passenger_id: int | None = None
    passenger_name: str = Field(min_length=1, max_length=255)
    passenger_email: str | None = None
    passenger_phone: str | None = None
    seat_number: str | None = None
    total_price: float | None = None
    hold_token: str | None = None


class SeatChangeRequest(BaseModel):
    new_seat_number: str = Field(min_length=1, max_length=64)


class SeatHoldRequest(BaseModel):
    flight_id: int
    seat_number: str = Field(min_length=1, max_length=64)
    hold_token: str | None = None


class PaymentInit(BaseModel):
    booking_id: int
    method: str = Field(pattern="^(vnpay|momo|cash)$")


class FaceRegisterPayload(BaseModel):
    booking_id: int
    ciphertext: str | None = None
    iv: str | None = None
    embedding: list[float] | None = None


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
    """Execute a write query and return rows affected."""
    with get_connection() as conn:
        cur = conn.execute(query, params)
        conn.commit()
        return cur.rowcount


def _generate_booking_code() -> str:
    return secrets.token_hex(4).upper()


HOLD_SECONDS = 10 * 60


def _utc_now() -> datetime:
    return datetime.utcnow().replace(microsecond=0)


def _format_hold_time(value: datetime) -> str:
    return value.isoformat() + "Z"


def _cleanup_expired_holds(conn: Any) -> None:
    conn.execute("DELETE FROM seat_holds WHERE expires_at <= ?", (_format_hold_time(_utc_now()),))


def _begin_booking_write(conn: Any) -> None:
    if not USE_MYSQL:
        conn.execute("BEGIN IMMEDIATE")


def _parse_seat_numbers(seat_number: str | None) -> list[str]:
    return [
        seat.strip().upper()
        for seat in str(seat_number or "").split(",")
        if seat.strip()
    ]


def _normalize_seat_number(seat_number: str | None) -> str | None:
    seats = _parse_seat_numbers(seat_number)
    return ", ".join(seats) if seats else None


def _find_taken_seats(
    conn: Any,
    flight_id: int,
    requested_seats: list[str],
    exclude_booking_id: int | None = None,
    allowed_hold_token: str | None = None,
) -> list[str]:
    if not requested_seats:
        return []

    params: list[Any] = [flight_id]
    exclude_clause = ""
    if exclude_booking_id is not None:
        exclude_clause = "AND id != ?"
        params.append(exclude_booking_id)

    existing = conn.execute(
        f"""
        SELECT seat_number FROM bookings
        WHERE flight_id = ?
          AND seat_number IS NOT NULL
          AND status NOT IN ('cancelled', 'refunded')
          {exclude_clause}
        """,
        tuple(params),
    ).fetchall()
    taken = {
        seat
        for row in existing
        for seat in _parse_seat_numbers(row["seat_number"])
    }

    hold_params: list[Any] = [flight_id, _format_hold_time(_utc_now())]
    hold_clause = ""
    if allowed_hold_token:
        hold_clause = "AND hold_token != ?"
        hold_params.append(allowed_hold_token)
    active_holds = conn.execute(
        f"""
        SELECT seat_number FROM seat_holds
        WHERE flight_id = ?
          AND expires_at > ?
          {hold_clause}
        """,
        tuple(hold_params),
    ).fetchall()
    taken.update(
        seat
        for row in active_holds
        for seat in _parse_seat_numbers(row["seat_number"])
    )
    return sorted(taken.intersection(requested_seats))


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

auth_router = APIRouter(prefix="/api/auth", tags=["auth"])


@auth_router.post("/register", status_code=201)
def register(payload: UserRegister) -> dict[str, Any]:
    hp = hash_password(payload.password)
    with get_connection() as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO users (email, password_hash, full_name, phone, date_of_birth, id_card)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (payload.email, hp, payload.full_name, payload.phone,
                 payload.date_of_birth, payload.id_card),
            )
            user_id = cur.lastrowid
        except Exception as exc:
            raise HTTPException(status_code=409, detail="Email already registered") from exc

    user = _fetch_one("SELECT id, email, full_name, phone, date_of_birth, id_card, created_at FROM users WHERE id = ?", (user_id,))
    token = create_access_token({"sub": str(user["id"]), "email": user["email"]})
    return {"access_token": token, "token_type": "bearer", "user": user}


@auth_router.post("/login")
def login(payload: UserLogin) -> TokenResponse:
    user = _fetch_one("SELECT * FROM users WHERE email = ?", (payload.email,))
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": str(user["id"]), "email": user["email"]})
    safe_user = {k: v for k, v in user.items() if k != "password_hash"}
    return TokenResponse(access_token=token, user=safe_user)


@auth_router.get("/me")
def get_me(user: Annotated[dict, Depends(get_current_user)]) -> dict[str, Any]:
    uid = user["sub"]
    u = _fetch_one(
        "SELECT id, email, full_name, phone, date_of_birth, id_card, created_at FROM users WHERE id = ?",
        (int(uid),),
    )
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    return u


@auth_router.patch("/me")
def update_me(payload: UserUpdate, user: Annotated[dict, Depends(get_current_user)]) -> dict[str, Any]:
    fields, values = [], []
    for field, value in payload.model_dump(exclude_unset=True).items():
        if value is not None:
            fields.append(f"{field} = ?")
            values.append(value)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    values.append(int(user["sub"]))
    _execute(f"UPDATE users SET {', '.join(fields)} WHERE id = ?", tuple(values))
    return get_me(user)


# ---------------------------------------------------------------------------
# Airports
# ---------------------------------------------------------------------------

@router.get("/airports")
def list_airports() -> list[dict[str, Any]]:
    return _fetch_all("SELECT * FROM airports ORDER BY city ASC")


# ---------------------------------------------------------------------------
# Flights
# ---------------------------------------------------------------------------

@router.get("/flights/search")
def search_flights(
    origin: Annotated[str | None, Query(description="Airport code e.g. HAN")] = None,
    destination: Annotated[str | None, Query(description="Airport code e.g. SGN")] = None,
    flight_date: Annotated[str | None, Query(description="YYYY-MM-DD")] = None,
    passengers: Annotated[int, Query(ge=1, le=9)] = 1,
) -> list[dict[str, Any]]:
    q = """
        SELECT
            fl.id,
            fl.flight_number,
            fl.flight_date,
            fl.status,
            fl.available_seats,
            fl.price_multiplier,
            s.departure_time,
            s.arrival_time,
            s.base_price,
            r.distance_km,
            a_o.code    AS origin_code,
            a_o.name    AS origin_name,
            a_o.city    AS origin_city,
            a_d.code    AS dest_code,
            a_d.name    AS dest_name,
            a_d.city    AS dest_city,
            p.plane_type,
            p.total_seats,
            p.seat_layout
        FROM flights fl
        JOIN schedules s   ON s.id = fl.schedule_id
        JOIN routes   r   ON r.id = s.route_id
        JOIN airports a_o ON a_o.id = r.origin_id
        JOIN airports a_d ON a_d.id = r.destination_id
        JOIN planes   p   ON p.id = s.plane_id
        WHERE fl.status IN ('scheduled', 'boarding')
          AND fl.available_seats >= ?
    """
    params: list[Any] = [passengers]

    if origin:
        q += " AND a_o.code = ?"
        params.append(origin.upper())
    if destination:
        q += " AND a_d.code = ?"
        params.append(destination.upper())
    if flight_date:
        q += " AND fl.flight_date = ?"
        params.append(flight_date)

    q += " ORDER BY s.departure_time ASC"

    rows = _fetch_all(q, tuple(params))
    for row in rows:
        # Calculate flight duration
        if row.get("departure_time") and row.get("arrival_time"):
            try:
                dep = row["departure_time"]
                arr = row["arrival_time"]
                if isinstance(dep, str):
                    dep = datetime.strptime(str(dep), "%H:%M:%S")
                    arr = datetime.strptime(str(arr), "%H:%M:%S")
                diff = arr - dep
                if diff.total_seconds() < 0:
                    diff += timedelta(hours=24)
                hours, remainder = divmod(int(diff.total_seconds()), 3600)
                minutes = remainder // 60
                row["duration_minutes"] = hours * 60 + minutes
            except Exception:
                row["duration_minutes"] = 0
        # Calculate price
        base = float(row.get("base_price") or 0)
        mult = float(row.get("price_multiplier") or 1.0)
        row["price_per_person"] = round(base * mult, 0)
        row["total_price"] = round(base * mult * passengers, 0)
    return rows


@router.get("/flights/{flight_id}")
def get_flight(flight_id: int) -> dict[str, Any]:
    row = _fetch_one(
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
    if not row:
        raise HTTPException(status_code=404, detail="Flight not found")
    return row


@router.get("/flights/{flight_id}/seats")
def get_flight_seats(flight_id: int) -> dict[str, Any]:
    flight = get_flight(flight_id)
    layout_raw = flight.get("seat_layout")
    if isinstance(layout_raw, str):
        import json
        layout = json.loads(layout_raw)
    elif isinstance(layout_raw, dict):
        layout = layout_raw
    else:
        layout = {"rows": 30, "cols": 6, "aisle": 2, "extra_legroom_cols": [1, 6]}

    with get_connection() as conn:
        _cleanup_expired_holds(conn)
        booked = conn.execute(
            """
            SELECT seat_number FROM bookings
            WHERE flight_id = ?
              AND seat_number IS NOT NULL
              AND status NOT IN ('cancelled', 'refunded')
            """,
            (flight_id,),
        ).fetchall()
        held = conn.execute(
            """
            SELECT seat_number FROM seat_holds
            WHERE flight_id = ?
              AND expires_at > ?
            """,
            (flight_id, _format_hold_time(_utc_now())),
        ).fetchall()

    booked_seats = {
        seat
        for row in booked
        for seat in _parse_seat_numbers(row["seat_number"])
    }
    held_seats = {
        seat
        for row in held
        for seat in _parse_seat_numbers(row["seat_number"])
    }
    unavailable_seats = booked_seats | held_seats

    rows_count = layout.get("rows", 30)
    cols_count = layout.get("cols", 6)
    aisle_at = layout.get("aisle", 2)
    extra_cols = set(layout.get("extra_legroom_cols", [1, 6]))

    seats = []
    for r in range(1, rows_count + 1):
        row_label = str(r)
        for c in range(1, cols_count + 1):
            seat_label = f"{row_label}{chr(64 + c)}"
            status = "booked" if seat_label in unavailable_seats else "available"
            extra = c in extra_cols
            surcharge = 150000 if extra else 0
            seats.append({
                "seat": seat_label,
                "row": r,
                "col": c,
                "status": status,
                "extra_legroom": extra,
                "surcharge": surcharge,
            })

    return {
        "flight_id": flight_id,
        "flight_number": flight["flight_number"],
        "layout": layout,
        "seats": seats,
        "booked_count": len(booked_seats),
        "held_count": len(held_seats - booked_seats),
        "available_count": len(seats) - len(unavailable_seats),
    }


# ---------------------------------------------------------------------------
# Seat holds
# ---------------------------------------------------------------------------

@router.post("/seat-holds", status_code=201)
def hold_seats(payload: SeatHoldRequest) -> dict[str, Any]:
    get_flight(payload.flight_id)
    requested_seats = _parse_seat_numbers(payload.seat_number)
    normalized_seat_number = _normalize_seat_number(payload.seat_number)
    if not requested_seats or not normalized_seat_number:
        raise HTTPException(status_code=422, detail="Invalid seat number")

    hold_token = payload.hold_token or uuid.uuid4().hex
    expires_at = _format_hold_time(_utc_now() + timedelta(seconds=HOLD_SECONDS))

    with get_connection() as conn:
        _begin_booking_write(conn)
        _cleanup_expired_holds(conn)
        _cleanup_expired_holds(conn)
        conn.execute("DELETE FROM seat_holds WHERE hold_token = ?", (hold_token,))

        conflict = _find_taken_seats(conn, payload.flight_id, requested_seats)
        if conflict:
            raise HTTPException(status_code=409, detail=f"Seat already taken: {', '.join(conflict)}")

        conn.execute(
            """
            INSERT INTO seat_holds (hold_token, flight_id, seat_number, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (hold_token, payload.flight_id, normalized_seat_number, expires_at),
        )
        conn.commit()

    return {
        "hold_token": hold_token,
        "flight_id": payload.flight_id,
        "seat_number": normalized_seat_number,
        "expires_at": expires_at,
        "ttl_seconds": HOLD_SECONDS,
    }


@router.delete("/seat-holds/{hold_token}")
def release_seat_hold(hold_token: str) -> dict[str, Any]:
    with get_connection() as conn:
        _begin_booking_write(conn)
        cur = conn.execute("DELETE FROM seat_holds WHERE hold_token = ?", (hold_token,))
        conn.commit()
    return {"released": cur.rowcount > 0}


# ---------------------------------------------------------------------------
# Bookings
# ---------------------------------------------------------------------------

@router.post("/bookings", status_code=201)
def create_booking(
    payload: BookingCreate,
    user: Annotated[dict | None, Depends(get_optional_user)] = None,
) -> dict[str, Any]:
    flight = get_flight(payload.flight_id)
    requested_seats = _parse_seat_numbers(payload.seat_number)
    normalized_seat_number = _normalize_seat_number(payload.seat_number)
    seats_count = max(1, len(requested_seats))

    if flight["available_seats"] < seats_count:
        raise HTTPException(status_code=409, detail="No seats available on this flight")

    code = _generate_booking_code()
    uid = int(user["sub"]) if user else None
    total = payload.total_price or flight.get("price_per_person", 0)

    with get_connection() as conn:
        _begin_booking_write(conn)

        seat_capacity = conn.execute(
            "SELECT available_seats FROM flights WHERE id = ?",
            (payload.flight_id,),
        ).fetchone()
        if not seat_capacity:
            raise HTTPException(status_code=404, detail="Flight not found")
        if seat_capacity["available_seats"] < seats_count:
            raise HTTPException(status_code=409, detail="No seats available on this flight")

        # If guest booking, map to user account if email and phone matches
        if not uid and payload.passenger_email and payload.passenger_phone:
            existing_user = conn.execute(
                "SELECT id FROM users WHERE email = ? AND phone = ?",
                (payload.passenger_email.strip().lower(), payload.passenger_phone.strip()),
            ).fetchone()
            if existing_user:
                uid = existing_user["id"]

        if requested_seats:
            conflict = _find_taken_seats(
                conn,
                payload.flight_id,
                requested_seats,
                allowed_hold_token=payload.hold_token,
            )
            if conflict:
                raise HTTPException(status_code=409, detail=f"Seat already taken: {', '.join(conflict)}")

        # Resolve passenger_id: use provided or create from name/email/phone
        passenger_id = payload.passenger_id
        if not passenger_id:
            # Check if matching passenger exists
            existing = conn.execute(
                "SELECT id FROM passengers WHERE email = ?",
                (payload.passenger_email or "",),
            ).fetchone()
            if existing:
                passenger_id = existing["id"]
            else:
                # Create new passenger record
                cur = conn.execute(
                    "INSERT INTO passengers (name, email, phone) VALUES (?, ?, ?)",
                    (payload.passenger_name, payload.passenger_email or "", payload.passenger_phone or ""),
                )
                passenger_id = cur.lastrowid

        try:
            cur = conn.execute(
                """
                INSERT INTO bookings
                (booking_code, user_id, flight_id, passenger_id, seat_number,
                 passenger_name, passenger_email, passenger_phone,
                 total_price, status, payment_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'confirmed', 'unpaid')
                """,
                (code, uid, payload.flight_id, passenger_id,
                 normalized_seat_number,
                 payload.passenger_name, payload.passenger_email,
                 payload.passenger_phone, total),
            )
            booking_id = cur.lastrowid
        except Exception as exc:
            raise HTTPException(status_code=409, detail="Booking failed") from exc

        # Decrease available seats
        updated = conn.execute(
            "UPDATE flights SET available_seats = available_seats - ? WHERE id = ? AND available_seats >= ?",
            (seats_count, payload.flight_id, seats_count),
        )
        if updated.rowcount != 1:
            raise HTTPException(status_code=409, detail="No seats available on this flight")
        if payload.hold_token:
            conn.execute("DELETE FROM seat_holds WHERE hold_token = ?", (payload.hold_token,))
        conn.commit()

    return _get_booking_detail(booking_id)


def _get_booking_detail(booking_id: int) -> dict[str, Any]:
    return _fetch_one(
        """
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
        WHERE b.id = ?
        """,
        (booking_id,),
    )


def get_booking(booking_id: int) -> dict[str, Any]:
    """Public alias for face_routes.py — raises 404 if not found."""
    row = _get_booking_detail(booking_id)
    if not row:
        raise HTTPException(status_code=404, detail="Booking not found")
    return row


@router.get("/bookings")
def list_bookings(user: Annotated[dict, Depends(get_current_user)]) -> list[dict[str, Any]]:
    rows = _fetch_all(
        """
        SELECT
            b.*,
            fl.flight_number, fl.flight_date, fl.status AS flight_status,
            s.departure_time,
            a_o.code AS origin_code, a_o.city AS origin_city,
            a_d.code AS dest_code, a_d.city AS dest_city
        FROM bookings b
        JOIN flights fl ON fl.id = b.flight_id
        JOIN schedules s ON s.id = fl.schedule_id
        JOIN routes r ON r.id = s.route_id
        JOIN airports a_o ON a_o.id = r.origin_id
        JOIN airports a_d ON a_d.id = r.destination_id
        WHERE b.user_id = ?
        ORDER BY b.created_at DESC
        """,
        (int(user["sub"]),),
    )
    return rows


@router.get("/bookings/code/{code}")
def get_booking_by_code(code: str) -> dict[str, Any]:
    row = _fetch_one(
        """
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
        WHERE b.booking_code = ?
        """,
        (code.upper(),),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Booking not found")
    return row


@router.patch("/bookings/{booking_id}/cancel")
def cancel_booking(
    booking_id: int,
    user: Annotated[dict, Depends(get_current_user)],
) -> dict[str, Any]:
    row = _fetch_one("SELECT * FROM bookings WHERE id = ?", (booking_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Booking not found")
    if row.get("user_id") != int(user["sub"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    if row["status"] in ("cancelled", "refunded"):
        raise HTTPException(status_code=409, detail="Booking already cancelled or refunded")
    if row["status"] == "checked_in":
        raise HTTPException(status_code=409, detail="Cannot cancel a checked-in booking")

    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _execute(
        "UPDATE bookings SET status = 'cancelled', updated_at = ? WHERE id = ?",
        (now, booking_id),
    )
    _execute(
        "UPDATE flights SET available_seats = available_seats + 1 WHERE id = ?",
        (row["flight_id"],),
    )
    return _get_booking_detail(booking_id)


@router.post("/bookings/{booking_id}/change-seat")
def change_seat(
    booking_id: int,
    payload: SeatChangeRequest,
) -> dict[str, Any]:
    row = _fetch_one("SELECT * FROM bookings WHERE id = ?", (booking_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Booking not found")
    if row["status"] in ("cancelled", "checked_in"):
        raise HTTPException(status_code=409, detail="Cannot change seat in current booking state")

    # Check seat is free
    requested_seats = _parse_seat_numbers(payload.new_seat_number)
    if not requested_seats:
        raise HTTPException(status_code=422, detail="Invalid seat number")

    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    normalized_seat_number = ", ".join(requested_seats)

    with get_connection() as conn:
        _begin_booking_write(conn)
        _cleanup_expired_holds(conn)
        conflict = _find_taken_seats(conn, row["flight_id"], requested_seats, booking_id)
        if conflict:
            raise HTTPException(status_code=409, detail=f"Seat already taken: {', '.join(conflict)}")
        conn.execute(
            "UPDATE bookings SET seat_number = ?, updated_at = ? WHERE id = ?",
            (normalized_seat_number, now, booking_id),
        )
        conn.commit()
    return _get_booking_detail(booking_id)


@router.patch("/bookings/{booking_id}/checkin")
def checkin_booking(
    booking_id: int,
) -> dict[str, Any]:
    row = _fetch_one("SELECT * FROM bookings WHERE id = ?", (booking_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Booking not found")
    if row["status"] == "checked_in":
        raise HTTPException(status_code=409, detail="Already checked in")
    if row["status"] == "cancelled":
        raise HTTPException(status_code=409, detail="Cannot check in a cancelled booking")
    if row["payment_status"] != "paid":
        raise HTTPException(status_code=409, detail="Payment not completed")

    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _execute(
        "UPDATE bookings SET status = 'checked_in', updated_at = ? WHERE id = ?",
        (now, booking_id),
    )
    return _get_booking_detail(booking_id)


# ---------------------------------------------------------------------------
# Payments (mock VNPay/MoMo)
# ---------------------------------------------------------------------------

@router.post("/payments/init")
def init_payment(
    payload: PaymentInit,
) -> dict[str, Any]:
    booking = _fetch_one("SELECT * FROM bookings WHERE id = ?", (payload.booking_id,))
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    if booking["payment_status"] == "paid":
        raise HTTPException(status_code=409, detail="Already paid")

    tx_id = f"{payload.method.upper()}-{secrets.token_hex(6)}"
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO payments (booking_id, amount, method, transaction_id) VALUES (?, ?, ?, ?)",
            (booking["id"], booking["total_price"], payload.method, tx_id),
        )
        payment_id = cur.lastrowid
        conn.commit()

    if payload.method == "vnpay":
        pay_url = f"https://vnpay.example.com/pay?tx={tx_id}&amount={booking['total_price']}"
    elif payload.method == "momo":
        pay_url = f"https://momo.example.com/qr?tx={tx_id}&amount={booking['total_price']}"
    else:
        pay_url = None

    return {
        "payment_id": payment_id,
        "transaction_id": tx_id,
        "amount": booking["total_price"],
        "method": payload.method,
        "payment_url": pay_url,
        "status": "pending",
    }


@router.post("/payments/callback")
def payment_callback(
    tx_id: str = Query(...),
    status: str = Query(...),
) -> dict[str, Any]:
    """
    Mock callback from VNPay/MoMo.
    status: success | failed
    """
    with get_connection() as conn:
        payment = conn.execute(
            "SELECT * FROM payments WHERE transaction_id = ?", (tx_id,)
        ).fetchone()
    if not payment:
        raise HTTPException(status_code=404, detail="Payment not found")

    payment_id = payment["id"]
    booking_id = payment["booking_id"]

    with get_connection() as conn:
        if status == "success":
            conn.execute(
                "UPDATE payments SET status = 'success' WHERE id = ?", (payment_id,)
            )
            conn.execute(
                "UPDATE bookings SET payment_status = 'paid', updated_at = ? WHERE id = ?",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), booking_id),
            )
        else:
            conn.execute(
                "UPDATE payments SET status = 'failed' WHERE id = ?", (payment_id,)
            )
        conn.commit()

    return {"status": "ok", "booking_code": _fetch_one(
        "SELECT booking_code FROM bookings WHERE id = ?", (booking_id,)
    )["booking_code"]}


@router.get("/payments/{booking_id}")
def get_payment_status(
    booking_id: int,
) -> dict[str, Any]:
    booking = _fetch_one("SELECT * FROM bookings WHERE id = ?", (booking_id,))
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")

    row = _fetch_one(
        "SELECT * FROM payments WHERE booking_id = ? ORDER BY created_at DESC LIMIT 1",
        (booking_id,),
    )
    if not row:
        return {"booking_id": booking_id, "payment_status": booking["payment_status"], "payment": None}
    return {"booking_id": booking_id, "payment_status": booking["payment_status"], "payment": row}


# ---------------------------------------------------------------------------
# Register router with FastAPI app
# ---------------------------------------------------------------------------

def register_routes(app):
    app.include_router(auth_router)
    app.include_router(router)
